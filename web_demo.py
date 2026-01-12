# spider2_web_demo_streamlit.py
# Streamlit demo: Spider2-lite multi-hop (steps) -> SQL compile (one-shot / sequential),
# with DAG visualization, scrollable output panes, and optional SQLite execution.
#
# Run:
#   pip install -U streamlit datasets openai
#   streamlit run spider2_web_demo_streamlit.py
#
# Requires:
#   export OPENAI_API_KEY="sk-..."

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from datasets import load_dataset
from openai import OpenAI


# -----------------------------
# Utilities
# -----------------------------

def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    fence = re.compile(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", re.MULTILINE)
    m = fence.match(text)
    if m:
        return m.group(1).strip()
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def read_text(path: str, max_chars: int) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


def resolve_doc_path(spider2_root: str, doc_name: str) -> Path:
    # .../xlang-spider2/spider2-lite/resource/documents/<doc_name>
    p = Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name
    if not p.exists():
        raise FileNotFoundError(f"Doc not found: {p}")
    return p


def find_api_key() -> Optional[str]:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPEN_AI_KEY")
    )


@dataclass
class Throttle:
    min_interval_s: float
    last_call_t: float = 0.0

    def wait(self):
        if self.min_interval_s <= 0:
            return
        now = time.time()
        dt = now - self.last_call_t
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self.last_call_t = time.time()


def call_openai_text(
    client: OpenAI,
    model: str,
    prompt: str,
    throttle: Throttle,
    max_retries: int = 4,
) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            throttle.wait()
            resp = client.responses.create(model=model, input=prompt)
            out = getattr(resp, "output_text", None)
            if not out:
                out = str(resp)
            return out.strip()
        except Exception as e:
            last_err = e
            sleep_s = min(2**attempt, 20)
            st.warning(f"[WARN] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")


def _find_matching_paren(s: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_cte_body_if_wrapped(sql_text: str, cte_name: str) -> str:
    """
    If the model accidentally returns:
      WITH step1 AS ( SELECT ... ) SELECT ...
    or:
      step1 AS ( SELECT ... )
    extract the body inside step1 AS ( ... )
    """
    t = strip_code_fences(sql_text).strip()
    t = t.rstrip(";").strip()

    pat = re.compile(rf"(?:WITH\s+)?{re.escape(cte_name)}\s+AS\s*\(", re.IGNORECASE)
    m = pat.search(t)
    if not m:
        # Also strip a leading WITH if it appears without the CTE name
        if re.match(r"^\s*WITH\s+", t, re.IGNORECASE):
            # Not safe to fully parse; return as-is
            return t
        return t

    open_paren = t.find("(", m.end() - 1)
    if open_paren < 0:
        return t

    close_paren = _find_matching_paren(t, open_paren)
    if close_paren < 0:
        return t

    body = t[open_paren + 1 : close_paren].strip().rstrip(";").strip()

    # unwrap again if nested
    if re.search(rf"^{re.escape(cte_name)}\s+AS\s*\(", body, re.IGNORECASE):
        return extract_cte_body_if_wrapped(body, cte_name)

    return body


def enforce_table_suffix(sql_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Best-effort: If wildcard events_* used but filters event_date, rewrite to _TABLE_SUFFIX.
    """
    info = {"applied": False, "changed": []}
    s = sql_text

    if ("events_*" not in s) and ("events_*" not in s.lower()):
        return s.strip(), info
    if "_TABLE_SUFFIX" in s:
        return s.strip(), info

    between_pat = re.compile(r"event_date\s+BETWEEN\s+'(\d{8})'\s+AND\s+'(\d{8})'", re.IGNORECASE)
    if between_pat.search(s):
        s = between_pat.sub(r"_TABLE_SUFFIX BETWEEN '\1' AND '\2'", s)
        info["applied"] = True
        info["changed"].append("event_date BETWEEN -> _TABLE_SUFFIX BETWEEN")

    eq_pat = re.compile(r"event_date\s*=\s*'(\d{8})'", re.IGNORECASE)
    if eq_pat.search(s):
        s = eq_pat.sub(r"_TABLE_SUFFIX = '\1'", s)
        info["applied"] = True
        info["changed"].append("event_date = -> _TABLE_SUFFIX =")

    return s.strip(), info


# -----------------------------
# Prompts
# -----------------------------

def build_steps_prompt(doc_text: str, question: str) -> str:
    return f"""
You are an expert data analyst and SQL planner.

You are given:
1) Database / analytics documentation snippet (possibly involving multiple tables and joins).
2) A complex analytics question that likely requires multi-hop reasoning across multiple steps.

Task:
Decompose the question into a SEQUENCE of step-by-step sub-queries / reasoning steps.

Output format (STRICT):
Return ONLY valid JSON (no markdown, no extra text).
Use this schema:

{{
  "question": <string>,
  "steps": [
    {{
      "step_id": <int starting from 1>,
      "goal": <string>,
      "tables_or_entities": <array of strings>,
      "filters_or_conditions": <array of strings>,
      "intermediate_output": <string>,
      "depends_on_step_ids": <array of ints>
    }}
  ],
  "final_answer_derivation": <string>
}}

Constraints:
- Do NOT write executable SQL. Describe operations in natural language.
- Be concrete but implementation-ready.
- Keep steps minimal but complete (typically 3-7 steps).

[Database documentation snippet]
{doc_text}

[Question]
{question}

Return ONLY JSON:
""".strip()


def build_one_shot_sql_prompt(doc_text: str, steps_json: Dict[str, Any], dialect: str, prefer_table_suffix: bool) -> str:
    suffix_rules = ""
    if prefer_table_suffix:
        suffix_rules = """
BigQuery / GA4 conventions (IMPORTANT):
- Prefer querying wildcard tables (e.g. `analytics_<property_id>.events_*`).
- Prefer filtering by `_TABLE_SUFFIX` for date ranges.
""".strip()

    return f"""
You are an expert analytics engineer and SQL writer.

Goal:
Given (1) a documentation snippet and (2) a multi-hop decomposition plan (JSON steps),
produce ONE final SQL query that follows the steps using CTEs.

Requirements:
- Output ONLY SQL (no markdown, no explanations).
- Use CTE names step1, step2, step3... corresponding to step_id.
- The final SELECT should return the final answer.
- Dialect: {dialect}
{suffix_rules}

Documentation snippet:
--------------------
{doc_text}
--------------------

Steps plan JSON:
----------------
{json.dumps(steps_json, ensure_ascii=False, indent=2)}
----------------

Now produce the SQL:
""".strip()


def build_step_sql_prompt(
    doc_text: str,
    question: str,
    steps_json: Dict[str, Any],
    step: Dict[str, Any],
    dialect: str,
    prefer_table_suffix: bool,
) -> str:
    sid = int(step["step_id"])
    cte_name = f"step{sid}"

    ga4_rules = ""
    if prefer_table_suffix:
        ga4_rules = """
BigQuery / GA4 conventions (IMPORTANT):
- Prefer querying wildcard tables (e.g. `analytics_<property_id>.events_*`).
- Prefer filtering by `_TABLE_SUFFIX` for date ranges (e.g. `_TABLE_SUFFIX BETWEEN '20210101' AND '20210107'`).
""".strip()

    return f"""
You are an expert analytics engineer and SQL writer.

You will write SQL for ONE step in a multi-hop plan.

Context:
- Overall question: {question}
- Dialect: {dialect}

{ga4_rules}

STRICT OUTPUT RULES:
- Output ONLY a SINGLE SELECT query that can be used as the BODY of a CTE.
- Do NOT output `WITH`.
- Do NOT output `{cte_name} AS (...)`.
- Do NOT output explanations.
- Do NOT output markdown fences.
- Avoid trailing semicolons.

You may reference previous steps by CTE names: step1, step2, ... (only if needed by depends_on_step_ids).

Documentation snippet:
--------------------
{doc_text}
--------------------

Full steps plan JSON (for reference):
--------------------
{json.dumps(steps_json, ensure_ascii=False, indent=2)}
--------------------

Now write the SQL body for:
- {cte_name}
- goal: {step.get("goal","")}
- tables/entities: {step.get("tables_or_entities", [])}
- filters/conditions: {step.get("filters_or_conditions", [])}
- depends_on: {step.get("depends_on_step_ids", [])}
""".strip()


def build_final_select_prompt(question: str, steps_json: Dict[str, Any], dialect: str) -> str:
    steps = steps_json.get("steps", [])
    last_id = steps[-1]["step_id"] if steps else 1
    return f"""
You are an expert SQL writer.

Given a multi-step plan, output ONLY the FINAL SELECT statement that returns the final answer,
using the last CTE step{last_id}.

Rules:
- Output ONLY one SELECT statement.
- Do NOT output WITH.
- Do NOT output markdown.
- Dialect: {dialect}

Question:
{question}

Steps JSON:
{json.dumps(steps_json, ensure_ascii=False, indent=2)}

Now output the FINAL SELECT only:
""".strip()


# -----------------------------
# DAG (Graphviz DOT)
# -----------------------------

def build_dot_from_steps(steps: List[Dict[str, Any]]) -> str:
    lines = ["digraph G {", 'rankdir="LR";', 'node [shape=box, style="rounded"];']
    for s in steps:
        sid = int(s["step_id"])
        goal = (s.get("goal", "") or "").replace('"', "'")
        label = f"step{sid}\\n{goal[:60]}"
        lines.append(f'step{sid} [label="{label}"];')

    for s in steps:
        sid = int(s["step_id"])
        for dep in s.get("depends_on_step_ids", []) or []:
            lines.append(f"step{dep} -> step{sid};")

    lines.append("}")
    return "\n".join(lines)


# -----------------------------
# Report builder
# -----------------------------

def make_report_md(
    *,
    instance_id: str,
    db: str,
    split: str,
    idx: int,
    doc_path: str,
    question: str,
    model_steps: str,
    model_sql: str,
    mode: str,
    dialect: str,
    prefer_table_suffix: bool,
    max_doc_chars: int,
    steps_json: Optional[Dict[str, Any]],
    step_sql_map: Optional[Dict[str, str]],
    final_sql: Optional[str],
    notes: Dict[str, Any],
    doc_head: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Spider2 Web Demo Report")
    lines.append("")
    lines.append(f"- Time: {ts}")
    lines.append(f"- split / idx: {split} / {idx}")
    lines.append(f"- instance_id: {instance_id}")
    lines.append(f"- db: {db}")
    lines.append(f"- doc_path: `{doc_path}`")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- dialect: `{dialect}`")
    lines.append(f"- model_steps: `{model_steps}`")
    lines.append(f"- model_sql: `{model_sql}`")
    lines.append(f"- prefer_table_suffix: `{prefer_table_suffix}`")
    lines.append(f"- max_doc_chars: `{max_doc_chars}`")
    lines.append("")

    lines.append("## Question")
    lines.append("")
    lines.append(question.strip())
    lines.append("")

    lines.append("## Doc head")
    lines.append("")
    lines.append("```text")
    lines.append((doc_head or "[NO DOC]").strip())
    lines.append("```")
    lines.append("")

    if steps_json:
        lines.append("## Steps JSON")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(steps_json, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if step_sql_map:
        lines.append("## Per-step SQL (CTE bodies)")
        lines.append("")
        for k, v in step_sql_map.items():
            lines.append(f"### {k}")
            lines.append("")
            lines.append("```sql")
            lines.append(v.strip())
            lines.append("```")
            lines.append("")

    if final_sql:
        lines.append("## Final SQL")
        lines.append("")
        lines.append("```sql")
        lines.append(final_sql.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Notes / Diagnostics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(notes, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Cached loading
# -----------------------------

@st.cache_data(show_spinner=False)
def load_spider2_split(split: str):
    return load_dataset("xlangai/spider2-lite", split=split)


@st.cache_data(show_spinner=False)
def cached_read_doc(path: str, max_chars: int):
    return read_text(path, max_chars=max_chars)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Spider2 Multi-hop → SQL Demo", layout="wide")
st.title("Spider2-lite: Multi-hop steps → SQL Demo (Streamlit)")
st.caption("Pick an example → edit question → Generate steps → Compile SQL → (Optional) Run SQL (SQLite) → Download report")

api_key = find_api_key()
if not api_key:
    st.error("Missing API key. Please set OPENAI_API_KEY in your environment.")
    st.stop()

client = OpenAI(api_key=api_key)

# init session state
for k, default in {
    "steps_json": None,
    "step_sql_map": None,
    "final_sql": None,
    "notes": {},
    "run_result": None,
    "run_error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    spider2_root = st.text_input(
        "spider2_root",
        value="/Users/yangsongzhou/Year3/xlang-spider2",
    )

    split = st.selectbox("Split", ["train", "validation", "test"], index=0)
    ds = load_spider2_split(split)
    total = len(ds)

    idx = st.number_input("Example idx", min_value=0, max_value=max(total - 1, 0), value=0, step=1)

    max_doc_chars = st.slider("max_doc_chars", 500, 12000, 2500, 250)
    doc_head_chars = st.slider("doc_head_chars", 200, 4000, 1200, 100)

    st.divider()
    mode = st.radio("Compile mode", ["Sequential (per-step)", "One-shot (final SQL)"], index=0)
    prefer_table_suffix = st.checkbox("Prefer _TABLE_SUFFIX (BigQuery wildcard)", value=True)
    dialect = st.selectbox("Dialect label", ["BigQuery", "SQLite", "PostgreSQL", "MySQL"], index=0)

    st.divider()
    model_steps = st.text_input("model_steps", value="gpt-4o-mini")
    model_sql = st.text_input("model_sql", value="gpt-4o-mini")
    min_call_interval_s = st.number_input("min_call_interval_s", 0.0, 120.0, 22.0, 1.0)
    sleep_between_steps_s = st.number_input("sleep_between_steps_s", 0.0, 20.0, 2.0, 0.5)

    st.divider()
    st.subheader("Run SQL (SQLite only)")
    sqlite_db_path = st.text_input("sqlite_db_path", value="", help="Provide a .db path only if dialect=SQLite")

throttle = Throttle(min_interval_s=float(min_call_interval_s))

# Load selected example
ex = ds[int(idx)]
instance_id = ex.get("instance_id", "")
db = ex.get("db", "")
question_default = ex.get("question", "")
doc_name = ex.get("external_knowledge", None)
temporal = ex.get("temporal", None)

doc_missing = (not isinstance(doc_name, str)) or (not doc_name)

# Layout
colL, colR = st.columns([1.15, 1.0], gap="large")

with colL:
    st.subheader("Input")
    st.write(f"**instance_id**: `{instance_id}`  |  **db**: `{db}`  |  **doc**: `{doc_name}`  |  **temporal**: `{temporal}`")

    question = st.text_area("Question (editable)", value=question_default, height=90)

    if doc_missing:
        st.warning("This example has no external_knowledge doc. We'll generate steps from question only (empty doc).")
        doc_path_str = "(NO DOC)"
        doc_text = ""
        doc_head = ""
    else:
        doc_path = resolve_doc_path(spider2_root, doc_name)
        doc_path_str = str(doc_path)
        doc_text = cached_read_doc(doc_path_str, max_doc_chars)
        doc_head = cached_read_doc(doc_path_str, doc_head_chars)

    with st.expander("Doc snippet (head)", expanded=False):
        st.text_area("doc_head", value=(doc_head or "[NO DOC]"), height=260, label_visibility="collapsed")

    st.divider()

    # Put buttons into a form for more reliable click behavior.
    with st.form("actions", clear_on_submit=False):
        b1, b2, b3, b4 = st.columns(4)
        gen_steps = b1.form_submit_button("Generate steps", type="primary", use_container_width=True)
        compile_sql = b2.form_submit_button("Compile SQL", use_container_width=True)
        run_sql = b3.form_submit_button("Run SQL", use_container_width=True)
        clear = b4.form_submit_button("Clear output", use_container_width=True)

    if clear:
        st.session_state["steps_json"] = None
        st.session_state["step_sql_map"] = None
        st.session_state["final_sql"] = None
        st.session_state["notes"] = {}
        st.session_state["run_result"] = None
        st.session_state["run_error"] = None
        st.toast("Cleared output.", icon="🧹")

    if gen_steps:
        try:
            with st.spinner("Generating steps JSON..."):
                t0 = time.time()
                prompt = build_steps_prompt(doc_text=doc_text, question=question)
                raw = call_openai_text(client, model_steps, prompt, throttle=throttle)
                steps_json = safe_json_loads(raw)
                st.session_state["steps_json"] = steps_json
                st.session_state["notes"] = {**st.session_state["notes"], "steps_latency_s": round(time.time() - t0, 2)}
            st.toast("Steps generated.", icon="✅")
        except Exception as e:
            st.error(f"Generate steps failed: {e}")

    if compile_sql:
        steps_json = st.session_state.get("steps_json")
        if not steps_json:
            st.error("No steps JSON yet. Click 'Generate steps' first.")
        else:
            try:
                with st.spinner("Compiling SQL..."):
                    notes: Dict[str, Any] = dict(st.session_state.get("notes", {}))
                    notes["mode"] = mode
                    notes["dialect"] = dialect
                    notes["postprocess"] = []

                    if mode.startswith("One-shot"):
                        t0 = time.time()
                        prompt = build_one_shot_sql_prompt(doc_text, steps_json, dialect, prefer_table_suffix)
                        raw_sql = call_openai_text(client, model_sql, prompt, throttle=throttle)
                        sql = strip_code_fences(raw_sql).strip().rstrip(";").strip()

                        if prefer_table_suffix:
                            sql, info = enforce_table_suffix(sql)
                            notes["postprocess"].append({"one_shot_table_suffix": info})

                        st.session_state["final_sql"] = sql + ";"
                        st.session_state["step_sql_map"] = None
                        notes["compile_latency_s"] = round(time.time() - t0, 2)

                    else:
                        steps = steps_json.get("steps", [])
                        if not steps:
                            raise RuntimeError("steps_json has no steps.")
                        step_sql_map: Dict[str, str] = {}
                        t0 = time.time()

                        for step in steps:
                            sid = int(step["step_id"])
                            cte_name = f"step{sid}"

                            step_prompt = build_step_sql_prompt(
                                doc_text=doc_text,
                                question=question,
                                steps_json=steps_json,
                                step=step,
                                dialect=dialect,
                                prefer_table_suffix=prefer_table_suffix,
                            )
                            raw_sql = call_openai_text(client, model_sql, step_prompt, throttle=throttle)
                            cleaned = strip_code_fences(raw_sql).strip().rstrip(";").strip()

                            body = extract_cte_body_if_wrapped(cleaned, cte_name)

                            post = {"cte": cte_name}
                            if prefer_table_suffix:
                                body, info = enforce_table_suffix(body)
                                post["table_suffix"] = info
                            notes["postprocess"].append(post)

                            step_sql_map[cte_name] = body

                            if float(sleep_between_steps_s) > 0:
                                time.sleep(float(sleep_between_steps_s))

                        final_select_prompt = build_final_select_prompt(question, steps_json, dialect)
                        raw_sel = call_openai_text(client, model_sql, final_select_prompt, throttle=throttle)
                        final_select = strip_code_fences(raw_sel).strip().rstrip(";").strip()

                        if re.search(r"^\s*WITH\s", final_select, re.IGNORECASE):
                            last_id = int(steps[-1]["step_id"])
                            final_select = f"SELECT * FROM step{last_id}"

                        # assemble final
                        lines: List[str] = ["WITH"]
                        for j, step in enumerate(steps):
                            sid = int(step["step_id"])
                            cte_name = f"step{sid}"
                            comma = "," if j < len(steps) - 1 else ""
                            lines.append(f"{cte_name} AS (")
                            lines.append(step_sql_map[cte_name].strip())
                            lines.append(f"){comma}")
                        final_sql = "\n".join(lines) + "\n" + final_select + ";"

                        st.session_state["step_sql_map"] = step_sql_map
                        st.session_state["final_sql"] = final_sql
                        notes["compile_latency_s"] = round(time.time() - t0, 2)

                    st.session_state["notes"] = notes
                st.toast("SQL compiled.", icon="✅")
            except Exception as e:
                st.error(f"Compile SQL failed: {e}")

    if run_sql:
        st.session_state["run_result"] = None
        st.session_state["run_error"] = None

        final_sql = st.session_state.get("final_sql")
        if not final_sql:
            st.session_state["run_error"] = "No final SQL. Compile first."
            st.toast("Run failed: no final SQL", icon="⚠️")
        elif dialect != "SQLite":
            st.session_state["run_error"] = "Run SQL is supported only for SQLite in this demo. (BigQuery needs credentials/project.)"
            st.toast("Run skipped (non-SQLite dialect).", icon="ℹ️")
        elif not sqlite_db_path:
            st.session_state["run_error"] = "Please set sqlite_db_path in the sidebar."
            st.toast("Run failed: missing sqlite_db_path.", icon="⚠️")
        else:
            try:
                import sqlite3
                with st.spinner("Running SQL on SQLite..."):
                    with sqlite3.connect(sqlite_db_path) as conn:
                        cur = conn.cursor()
                        cur.execute(final_sql)
                        rows = cur.fetchall()
                        cols = [d[0] for d in cur.description] if cur.description else []
                st.session_state["run_result"] = {"cols": cols, "rows": rows[:200]}  # cap
                st.toast("SQL executed.", icon="✅")
            except Exception as e:
                st.session_state["run_error"] = str(e)
                st.toast("SQL execution failed.", icon="❌")


with colR:
    st.subheader("Output")

    tabs = st.tabs(["Steps JSON", "DAG", "SQL", "Run result", "Downloads", "Diagnostics"])

    steps_json = st.session_state.get("steps_json")
    step_sql_map = st.session_state.get("step_sql_map")
    final_sql = st.session_state.get("final_sql")
    notes = st.session_state.get("notes", {})

    # 1) Steps JSON
    with tabs[0]:
        st.caption("Scrollable view (won't explode page height).")
        box = st.container(height=680)
        with box:
            if steps_json:
                box.text_area(
                    "steps_json",
                    value=json.dumps(steps_json, ensure_ascii=False, indent=2),
                    height=650,
                    label_visibility="collapsed",
                )
            else:
                st.info("No steps yet. Click Generate steps.")

    # 2) DAG
    with tabs[1]:
        box = st.container(height=680)
        with box:
            if steps_json and steps_json.get("steps"):
                dot = build_dot_from_steps(steps_json["steps"])
                st.graphviz_chart(dot, use_container_width=True)
            else:
                st.info("No steps to visualize.")

    # 3) SQL
    with tabs[2]:
        st.caption("SQL is shown in scrollable boxes. Per-step bodies appear in Sequential mode.")
        box = st.container(height=680)
        with box:
            if final_sql:
                st.text_area("final_sql", value=final_sql, height=260, label_visibility="collapsed")
            else:
                st.info("No final SQL yet. Click Compile SQL.")

            if step_sql_map:
                st.divider()
                st.write("Per-step SQL (CTE bodies):")
                for cte, body in step_sql_map.items():
                    with st.expander(cte, expanded=False):
                        st.text_area(f"{cte}_body", value=body, height=200, label_visibility="collapsed")

    # 4) Run result
    with tabs[3]:
        box = st.container(height=680)
        with box:
            if st.session_state.get("run_error"):
                st.error(st.session_state["run_error"])
            elif st.session_state.get("run_result"):
                rr = st.session_state["run_result"]
                cols = rr.get("cols", [])
                rows = rr.get("rows", [])
                st.success(f"Returned {len(rows)} rows (capped).")
                if cols:
                    st.dataframe([dict(zip(cols, r)) for r in rows])
                else:
                    st.write(rows)
            else:
                st.info("No run result. Click Run SQL (SQLite only).")

    # 5) Downloads
    with tabs[4]:
        box = st.container(height=680)
        with box:
            if not (steps_json or final_sql):
                st.info("Nothing to download yet.")
            else:
                doc_head_for_report = doc_head or "[NO DOC]"
                report_md = make_report_md(
                    instance_id=instance_id,
                    db=db,
                    split=split,
                    idx=int(idx),
                    doc_path=doc_path_str,
                    question=question,
                    model_steps=model_steps,
                    model_sql=model_sql,
                    mode=mode,
                    dialect=dialect,
                    prefer_table_suffix=prefer_table_suffix,
                    max_doc_chars=max_doc_chars,
                    steps_json=steps_json,
                    step_sql_map=step_sql_map,
                    final_sql=final_sql,
                    notes=notes,
                    doc_head=doc_head_for_report,
                )

                st.download_button(
                    "Download report.md",
                    data=report_md.encode("utf-8"),
                    file_name=f"spider2_demo_{split}_{int(idx):04d}_{instance_id}.report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                if steps_json:
                    st.download_button(
                        "Download steps.json",
                        data=(json.dumps(steps_json, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                        file_name=f"spider2_demo_{split}_{int(idx):04d}_{instance_id}.steps.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                if final_sql:
                    st.download_button(
                        "Download final.sql",
                        data=(final_sql.strip() + "\n").encode("utf-8"),
                        file_name=f"spider2_demo_{split}_{int(idx):04d}_{instance_id}.final.sql",
                        mime="text/plain",
                        use_container_width=True,
                    )

    # 6) Diagnostics
    with tabs[5]:
        box = st.container(height=680)
        with box:
            st.code(json.dumps(notes, ensure_ascii=False, indent=2), language="json")
