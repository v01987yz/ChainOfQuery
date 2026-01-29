# Streamlit demo: Spider2-lite -> Steps -> Compile SQL -> Execute (BigQuery)
#
# Run:
#   export OPENAI_API_KEY=...
#   gcloud auth application-default login
#   streamlit run web_demo.py

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import textwrap
import streamlit as st
from datasets import load_dataset
from openai import OpenAI

try:
    from coq_agent import SequentialCoQAgent
except ImportError:
    st.error("❌ Could not import 'SequentialCoQAgent'. Put coq_agent.py in the same folder.")
    st.stop()


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


def read_text(path: str, max_chars: int) -> str:
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n\n[TRUNCATED]"
        return txt
    except Exception:
        return ""


def resolve_doc_path(spider2_root: str, doc_name: str) -> Path:
    return Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name


def find_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("OPEN_AI_KEY")


@st.cache_data(show_spinner=False)
def load_spider2_train():
    # spider2-lite on HF typically only has train
    return load_dataset("xlangai/spider2-lite", split="train")


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


def build_dot_from_steps(steps: List[Dict[str, Any]]) -> str:
    lines = [
        "digraph G {",
        'rankdir="LR";',
        'node [shape=note, style="filled", fillcolor="#f0f2f6", fontname="Arial"];',
        'edge [color="#555555"];',
    ]

    for s in steps:
        sid = int(s.get("step_id", 0))
        desc = s.get("description", s.get("goal", "")) or ""
        wrapped_desc = "\\n".join(textwrap.wrap(desc, width=35))
        label = f"Step {sid}\\n{wrapped_desc}"
        lines.append(f'step{sid} [label="{label}"];')

    for s in steps:
        sid = int(s.get("step_id", 0))
        dep = s.get("dependency", "None")
        if dep and str(dep).lower() != "none":
            try:
                dep_i = int(dep)
                lines.append(f"step{dep_i} -> step{sid};")
            except Exception:
                pass

    lines.append("}")
    return "\n".join(lines)


# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="Spider2 CoQ Agent", layout="wide")
st.title("Spider2-lite: Steps → Compile SQL → Execute (BigQuery)")
st.caption("推荐流程：Generate Steps → Compile SQL → Run/Execute（更稳定、更适合做评测）")

api_key = find_api_key()
if not api_key:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()

openai_client = OpenAI(api_key=api_key)
throttle = Throttle(min_interval_s=0.8)

# session state
for k, default in {
    "steps_json": None,
    "compiled_sql": None,
    "run_result": None,
    "final_answer": None,
    "run_error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    spider2_root = st.text_input(
        "spider2_root",
        value="/Users/yangsongzhou/Year3/xlang-spider2",
        help="Used to load external_knowledge docs (markdown).",
    )

    ds = load_spider2_train()
    total = len(ds)
    idx = st.number_input("Example idx", min_value=0, max_value=max(total - 1, 0), value=0, step=1)
    ex = ds[int(idx)]

    max_doc_chars = st.slider("max_doc_chars", 500, 12000, 2500, 250)

    st.divider()
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")

    st.divider()
    st.subheader("BigQuery")
    bq_project = st.text_input("Project ID", value="coq-finalproject")
    bq_location = st.text_input("Location", value="US")
    bq_dataset_override = st.text_input(
        "Dataset override (ga4 default ok)",
        value="bigquery-public-data.ga4_obfuscated_sample_ecommerce",
        help="If SQL contains analytics_<property_id>, it will be replaced with this dataset.",
    )

    st.divider()
    st.caption("Tip: 先在 terminal 确认 BQ 能跑：python -c 'from google.cloud import bigquery; print(list(bigquery.Client().query(\"select 1\").result())[0][0])'")

instance_id = ex.get("instance_id", "")
db_id = ex.get("db", "")
doc_name = ex.get("external_knowledge", None)
question_default = ex.get("question", "")

doc_text = ""
if doc_name and os.path.exists(spider2_root):
    doc_path = resolve_doc_path(spider2_root, doc_name)
    doc_text = read_text(str(doc_path), max_doc_chars)

extra_context = ""
if doc_text:
    extra_context = "External Knowledge / Docs:\n" + doc_text

# -----------------------------
# Main Layout
# -----------------------------
colL, colR = st.columns([1.15, 1.0], gap="large")

with colL:
    st.subheader("Input")
    st.write(f"**instance_id**: `{instance_id}` | **db**: `{db_id}` | **doc**: `{doc_name}`")
    question = st.text_area("Question", value=question_default, height=90)

    with st.expander("Doc snippet"):
        st.text((doc_text[:1200] + "...") if doc_text else "[No doc loaded]")

    st.divider()

    # 3 buttons flow
    with st.form("actions", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        btn_steps = c1.form_submit_button("1) Generate Steps", type="primary", use_container_width=True)
        btn_compile = c2.form_submit_button("2) Compile SQL", use_container_width=True)
        btn_run = c3.form_submit_button("3) Run / Execute", use_container_width=True)
        btn_clear = c4.form_submit_button("Clear", use_container_width=True)

    if btn_clear:
        st.session_state["steps_json"] = None
        st.session_state["compiled_sql"] = None
        st.session_state["run_result"] = None
        st.session_state["final_answer"] = None
        st.session_state["run_error"] = None
        st.rerun()

    # 1) Generate Steps
    if btn_steps:
        with st.spinner("Generating steps..."):
            agent = SequentialCoQAgent(
                backend="bigquery",
                bq_project=bq_project,
                bq_location=bq_location,
                bq_db=db_id,
                bq_dataset_override=bq_dataset_override.strip() or None,
                api_key=api_key,
                model=model_name,
            )
            schema_text = agent._get_schema_info()
            if extra_context:
                schema_text = schema_text + "\n\n" + extra_context

            plan = agent.plan_decomposition(question, schema_text)
            st.session_state["steps_json"] = {"steps": plan}
            st.session_state["compiled_sql"] = None
            st.session_state["run_result"] = None
            st.session_state["final_answer"] = None
            st.session_state["run_error"] = None
            st.toast("Steps generated.", icon="✅")

    # 2) Compile SQL
    if btn_compile:
        if not st.session_state.get("steps_json"):
            st.error("Please Generate Steps first.")
        else:
            with st.spinner("Compiling ONE final SQL..."):
                agent = SequentialCoQAgent(
                    backend="bigquery",
                    bq_project=bq_project,
                    bq_location=bq_location,
                    bq_db=db_id,
                    bq_dataset_override=bq_dataset_override.strip() or None,
                    api_key=api_key,
                    model=model_name,
                )
                steps = st.session_state["steps_json"]["steps"]
                sql = agent.compile_final_sql(question=question, steps=steps, extra_context=extra_context)

                # store compiled sql
                st.session_state["compiled_sql"] = sql
                st.session_state["run_result"] = None
                st.session_state["final_answer"] = None
                st.session_state["run_error"] = None
                st.toast("SQL compiled.", icon="🧩")

    # 3) Run / Execute
    if btn_run:
        if not st.session_state.get("compiled_sql"):
            st.error("Please Compile SQL first.")
        else:
            with st.spinner("Executing SQL on BigQuery..."):
                agent = SequentialCoQAgent(
                    backend="bigquery",
                    bq_project=bq_project,
                    bq_location=bq_location,
                    bq_db=db_id,
                    bq_dataset_override=bq_dataset_override.strip() or None,
                    api_key=api_key,
                    model=model_name,
                )
                sql = st.session_state["compiled_sql"]
                res = agent._execute_sql(sql)

                if isinstance(res, str) and "SQL Error" in res:
                    st.session_state["run_error"] = res
                    st.error(res)
                else:
                    st.session_state["run_result"] = res

                    # Natural language answer
                    with st.spinner("Generating final answer..."):
                        answer_prompt = f"""
User Question: "{question}"
SQL Result (JSON): {json.dumps(res, ensure_ascii=False)}

Task:
- Answer the question in a clear sentence.
- If result is a single number, format with commas.
- If result is a table, summarize key fields/rows briefly.
Output ONLY the answer.
""".strip()

                        resp = openai_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": answer_prompt}],
                            temperature=0,
                        )
                        st.session_state["final_answer"] = (resp.choices[0].message.content or "").strip()

                    st.success("Execution complete ✅")


# -----------------------------
# Output Panel
# -----------------------------
with colR:
    st.subheader("Output")

    tabs = st.tabs(["Steps", "DAG", "Compiled SQL", "Result"])

    with tabs[0]:
        if st.session_state["steps_json"]:
            st.json(st.session_state["steps_json"])
        else:
            st.info("No steps yet.")

    with tabs[1]:
        if st.session_state["steps_json"]:
            try:
                dot = build_dot_from_steps(st.session_state["steps_json"]["steps"])
                st.graphviz_chart(dot)
            except Exception as e:
                st.warning(f"Cannot draw DAG: {e}")
        else:
            st.info("No DAG yet.")

    with tabs[2]:
        if st.session_state["compiled_sql"]:
            st.code(st.session_state["compiled_sql"], language="sql")
        else:
            st.info("No compiled SQL yet. Click 2) Compile SQL.")

    with tabs[3]:
        if st.session_state.get("run_error"):
            st.error(st.session_state["run_error"])
        elif st.session_state.get("run_result") is not None:
            if st.session_state.get("final_answer"):
                st.subheader("🎉 Final Answer")
                st.info(st.session_state["final_answer"])
            st.subheader("Raw Result")
            st.write(st.session_state["run_result"])
        else:
            st.info("No result yet. Click 3) Run / Execute.")
