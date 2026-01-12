# web_demo.py
# Streamlit demo: Spider2-lite multi-hop (steps) -> SQL compile -> Execution
# Integrated with: SequentialCoQAgent + Natural Language Answer Generation
#
# Run:
#   streamlit run web_demo.py

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import textwrap 
import streamlit as st
from datasets import load_dataset
from openai import OpenAI

# ==========================================
# [Core] 引入你的 Agent 引擎
# ==========================================
try:
    from coq_agent import SequentialCoQAgent
except ImportError:
    st.error("❌ Critical Error: Could not import 'SequentialCoQAgent'. Please ensure 'coq_agent.py' is in the same directory.")
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
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n\n[TRUNCATED]"
        return txt
    except Exception:
        return ""


def resolve_doc_path(spider2_root: str, doc_name: str) -> Path:
    p = Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name
    return p

def resolve_db_path(spider2_root: str, db_name: str) -> str:
    p = Path(spider2_root) / "spider2-lite" / "resource" / "databases" / "sqlite" / db_name / f"{db_name}.sqlite"
    return str(p)


def find_api_key() -> Optional[str]:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPEN_AI_KEY")
    )

@st.cache_data(show_spinner=False)
def load_spider2_split(split: str):
    try:
        return load_dataset("xlangai/spider2-lite", split=split)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}. Please check your internet connection.")
        return []


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
            resp = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}]
            )
            out = resp.choices[0].message.content
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
    t = strip_code_fences(sql_text).strip()
    t = t.rstrip(";").strip()

    pat = re.compile(rf"(?:WITH\s+)?{re.escape(cte_name)}\s+AS\s*\(", re.IGNORECASE)
    m = pat.search(t)
    if not m:
        if re.match(r"^\s*WITH\s+", t, re.IGNORECASE):
            return t
        return t

    open_paren = t.find("(", m.end() - 1)
    if open_paren < 0:
        return t

    close_paren = _find_matching_paren(t, open_paren)
    if close_paren < 0:
        return t

    body = t[open_paren + 1 : close_paren].strip().rstrip(";").strip()
    if re.search(rf"^{re.escape(cte_name)}\s+AS\s*\(", body, re.IGNORECASE):
        return extract_cte_body_if_wrapped(body, cte_name)

    return body


def enforce_table_suffix(sql_text: str) -> Tuple[str, Dict[str, Any]]:
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
Task: Decompose the question into a SEQUENCE of step-by-step sub-queries.

[Database documentation snippet]
{doc_text}

[Question]
{question}

Output format (STRICT JSON):
{{
  "steps": [
    {{
      "step_id": <int>,
      "goal": <string>,
      "tables_or_entities": [<string>],
      "depends_on_step_ids": [<int>]
    }}
  ]
}}
""".strip()

def build_one_shot_sql_prompt(doc_text: str, steps_json: Dict[str, Any], dialect: str, prefer_table_suffix: bool) -> str:
    return f"""
You are an expert SQL writer. Dialect: {dialect}.
Produce ONE final SQL query using CTEs based on the plan.

Documentation:
{doc_text}

Steps:
{json.dumps(steps_json, indent=2)}

SQL:
""".strip()

def build_step_sql_prompt(doc_text: str, question: str, steps_json: Dict[str, Any], step: Dict[str, Any], dialect: str, prefer_table_suffix: bool) -> str:
    cte_name = f"step{step['step_id']}"
    return f"""
Dialect: {dialect}. Write SQL for ONE step: {cte_name}.
Goal: {step.get("goal")}

Documentation:
{doc_text}

Output ONLY the SELECT statement (no WITH, no explanations).
""".strip()

def build_final_select_prompt(question: str, steps_json: Dict[str, Any], dialect: str) -> str:
    return f"Output ONLY the FINAL SELECT statement for {dialect} to answer: {question}"

# -----------------------------
# DAG (Graphviz DOT) - 
# -----------------------------

def build_dot_from_steps(steps: List[Dict[str, Any]]) -> str:
    lines = [
        "digraph G {",
        'rankdir="LR";',  # LR是从左到右，如果你想从上到下变成流程图，改成 "TB"
        'node [shape=note, style="filled", fillcolor="#f0f2f6", fontname="Arial"];', # 优化节点样式
        'edge [color="#555555"];'
    ]
    
    for s in steps:
        sid = int(s["step_id"])
        # 获取描述
        desc = s.get("description", s.get("goal", "")) or ""
        
        # [关键修改] 使用 textwrap 自动换行，每30个字符换一行，不再截断！
        wrapped_desc = "\\n".join(textwrap.wrap(desc, width=35))
        
        # 组装标签
        label = f"Step {sid}\\n{wrapped_desc}"
        lines.append(f'step{sid} [label="{label}"];')

    for s in steps:
        sid = int(s["step_id"])
        
        # 获取依赖关系
        deps = []
        raw_dep = s.get("dependency")
        if raw_dep and raw_dep != "None":
            try:
                deps.append(int(raw_dep))
            except:
                pass
        
        if "depends_on_step_ids" in s and s["depends_on_step_ids"]:
            deps.extend(s["depends_on_step_ids"])
            
        for dep in set(deps):
            lines.append(f"step{dep} -> step{sid};")

    lines.append("}")
    return "\n".join(lines)

# -----------------------------
# UI Setup
# -----------------------------

st.set_page_config(page_title="Spider2 CoQ Agent", layout="wide")
st.title("Spider2-lite: Agentic Text-to-SQL Demo")
st.caption("Supports both Standard CTE Generation and Agentic Sequential Execution.")

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
    "coq_context": None,
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
        help="Path to the xlang-spider2 folder"
    )

    split = st.selectbox("Split", ["train", "validation", "test"], index=0)
    
    # Load dataset safely
    ds = load_spider2_split(split)
    if len(ds) > 0:
        total = len(ds)
        idx = st.number_input("Example idx", min_value=0, max_value=max(total - 1, 0), value=0, step=1)
        ex = ds[int(idx)]
    else:
        ex = {"instance_id": "dummy", "db": "", "question": "", "external_knowledge": None}
        st.warning("Dataset not loaded. Using dummy data.")

    max_doc_chars = st.slider("max_doc_chars", 500, 12000, 2500, 250)

    st.divider()
    
    mode = st.radio(
        "Compile/Execution Mode", 
        ["Agentic CoQ (New Engine)", "Sequential (Standard)", "One-shot (Standard)"], 
        index=0
    )
    
    dialect = st.selectbox("Dialect", ["SQLite", "BigQuery", "PostgreSQL"], index=0)

    st.divider()
    model_name = st.text_input("Model", value="gpt-4-turbo")
    
    st.subheader("Database Path")
    db_id = ex.get("db", "")
    
    resolved_db_path = ""
    if os.path.exists(spider2_root) and db_id:
        resolved_db_path = resolve_db_path(spider2_root, db_id)
        
    sqlite_db_path = st.text_input("sqlite_db_path", value=resolved_db_path)
    if mode.startswith("Agentic") and not os.path.exists(sqlite_db_path):
         if db_id:
            st.warning(f"⚠️ Warning: Database file not found at {sqlite_db_path}")


throttle = Throttle(min_interval_s=1.0)

instance_id = ex.get("instance_id", "")
question_default = ex.get("question", "")
doc_name = ex.get("external_knowledge", None)

if doc_name and os.path.exists(spider2_root):
    doc_path = resolve_doc_path(spider2_root, doc_name)
    doc_text = read_text(str(doc_path), max_doc_chars)
else:
    doc_text = ""

# -----------------------------
# Main Layout
# -----------------------------
colL, colR = st.columns([1.15, 1.0], gap="large")

with colL:
    st.subheader("Input")
    st.write(f"**DB**: `{db_id}` | **Doc**: `{doc_name}`")
    question = st.text_area("Question", value=question_default, height=90)

    with st.expander("Doc snippet"):
        st.text(doc_text[:1000] + "...")

    st.divider()

    with st.form("actions", clear_on_submit=False):
        b1, b2, b3 = st.columns([1, 1, 1])
        gen_steps = b1.form_submit_button("1. Generate Steps", type="primary", use_container_width=True)
        
        run_label = "2. Run Agent (Execute)" if "Agentic" in mode else "2. Compile SQL"
        action_btn = b2.form_submit_button(run_label, use_container_width=True)
        
        clear = b3.form_submit_button("Clear", use_container_width=True)

    if clear:
        st.session_state["steps_json"] = None
        st.session_state["step_sql_map"] = None
        st.session_state["final_sql"] = None
        st.session_state["run_result"] = None
        st.session_state["coq_context"] = None
        st.rerun()

    if gen_steps:
        with st.spinner("Analyzing & Decomposing..."):
            if "Agentic" in mode:
                if not os.path.exists(sqlite_db_path):
                    st.error(f"Cannot initialize Agent: DB not found at {sqlite_db_path}. \n(For testing, ensure 'test_fy_project.db' exists if using mock data)")
                else:
                    agent = SequentialCoQAgent(db_path=sqlite_db_path, api_key=api_key, model=model_name)
                    schema = agent._get_schema_info()
                    plan = agent.plan_decomposition(question, schema)
                    st.session_state["steps_json"] = {"steps": plan}
                    st.toast("Agent Plan Generated!", icon="🤖")
            else:
                prompt = build_steps_prompt(doc_text=doc_text, question=question)
                raw = call_openai_text(client, model_name, prompt, throttle=throttle)
                st.session_state["steps_json"] = safe_json_loads(raw)
                st.toast("Standard Steps Generated.", icon="📄")

    if action_btn:
        steps_json = st.session_state.get("steps_json")
        
        # --- A. Agentic CoQ Execution (Updated) ---
        if "Agentic" in mode:
            if not steps_json:
                st.error("Please Generate Steps first.")
            elif not os.path.exists(sqlite_db_path):
                st.error("Database path invalid.")
            else:
                agent = SequentialCoQAgent(db_path=sqlite_db_path, api_key=api_key, model=model_name)
                schema = agent._get_schema_info()
                steps = steps_json.get("steps", [])
                
                context = {} 
                step_sql_map = {}
                
                progress_bar = st.progress(0)
                
                try:
                    for i, step in enumerate(steps):
                        st.text(f"🚀 Running Step {step['step_id']}: {step.get('description', step.get('goal'))}")
                        
                        if 'dependency' not in step:
                            deps = step.get('depends_on_step_ids', [])
                            step['dependency'] = deps[0] if deps else "None"
                            
                        sql = agent.generate_step_sql(step, context, schema)
                        step_sql_map[f"Step {step['step_id']}"] = sql
                        
                        res = agent._execute_sql(sql)
                        context[step['step_id']] = res
                        
                        if not res or (isinstance(res, str) and "Error" in res):
                             st.error(f"Step {step['step_id']} failed or returned empty: {res}")
                             break
                        
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5) 
                    
                    st.session_state["coq_context"] = context
                    st.session_state["step_sql_map"] = step_sql_map
                    
                    if steps:
                        final_step_id = steps[-1]['step_id']
                        final_res = context.get(final_step_id)
                        st.session_state["run_result"] = {"rows": final_res, "cols": ["Result"]}
                        
                        # === [NEW] Final Answer Generation ===
                        with st.spinner("🤖 Generating Final Natural Language Answer..."):
                            answer_prompt = f"""
                            User Question: "{question}"
                            Database Result: {final_res}
                            
                            Task: Answer the user's question in a natural, complete sentence based on the result.
                            If the result is a number, format it nicely (e.g., currency, commas).
                            """
                            final_resp = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": answer_prompt}]
                            )
                            natural_answer = final_resp.choices[0].message.content
                            
                        st.divider()
                        st.subheader("🎉 Final Answer")
                        st.info(natural_answer)
                        # =====================================
                    
                    st.success("Agent Execution Complete!")
                    
                except Exception as e:
                    st.error(f"Agent Execution Error: {e}")

        # --- B. Standard Compilation ---
        else:
            if not steps_json:
                st.error("No steps found.")
            else:
                prefer_table_suffix = True
                with st.spinner("Compiling CTEs..."):
                    if "One-shot" in mode:
                        prompt = build_one_shot_sql_prompt(doc_text, steps_json, dialect, prefer_table_suffix)
                        raw = call_openai_text(client, model_name, prompt, throttle=throttle)
                        sql = strip_code_fences(raw)
                        st.session_state["final_sql"] = sql
                    else:
                        steps = steps_json.get("steps", [])
                        step_map = {}
                        for step in steps:
                            prompt = build_step_sql_prompt(doc_text, question, steps_json, step, dialect, prefer_table_suffix)
                            raw = call_openai_text(client, model_name, prompt, throttle=throttle)
                            step_map[f"step{step['step_id']}"] = strip_code_fences(raw)
                        
                        full_sql = "WITH \n" + ",\n".join([f"{k} AS ({v})" for k,v in step_map.items()])
                        full_sql += f"\nSELECT * FROM step{steps[-1]['step_id']}"
                        st.session_state["final_sql"] = full_sql
                        st.session_state["step_sql_map"] = step_map
                st.toast("SQL Compiled.")


# -----------------------------
# Output Panel
# -----------------------------
with colR:
    st.subheader("Output / Execution")
    
    tabs = st.tabs(["Steps (JSON)", "DAG", "SQL / Process", "Final Result"])

    with tabs[0]:
        if st.session_state["steps_json"]:
            st.json(st.session_state["steps_json"])
        else:
            st.info("No steps generated.")

    with tabs[1]:
        if st.session_state["steps_json"]:
            try:
                dot = build_dot_from_steps(st.session_state["steps_json"]["steps"])
                st.graphviz_chart(dot)
            except Exception as e:
                st.warning(f"Could not draw DAG: {e}")
        else:
            st.info("No plan to visualize.")

    with tabs[2]:
        if "Agentic" in mode:
            if st.session_state["step_sql_map"]:
                for k, v in st.session_state["step_sql_map"].items():
                    with st.expander(f"📝 {k} SQL", expanded=True):
                        st.code(v, language="sql")
            
            if st.session_state["coq_context"]:
                with st.expander("📊 Intermediate Context (Values)", expanded=True):
                    st.write(st.session_state["coq_context"])
        else:
            if st.session_state["final_sql"]:
                st.code(st.session_state["final_sql"], language="sql")

    with tabs[3]:
        res = st.session_state.get("run_result")
        if res:
            st.success("Execution Successful")
            st.write(res.get("rows"))
        else:
            st.info("No results yet. Run the Agent/SQL.")