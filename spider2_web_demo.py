import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from datasets import load_dataset
from openai import OpenAI


# ---------------- helpers ----------------

def find_api_key() -> Optional[str]:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPEN_AI_KEY")
    )

def get_client() -> OpenAI:
    api_key = find_api_key()
    if not api_key:
        raise RuntimeError("API key not found in env (OPENAI_API_KEY / OPENAI_KEY / OPEN_AI_KEY).")
    return OpenAI(api_key=api_key)

def read_text(path: str, max_chars: int = 12000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text

def resolve_doc_path(spider2_root: str, doc_name: str) -> Optional[Path]:
    if not doc_name:
        return None
    p = Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name
    return p if p.exists() else None

def call_openai_text(model: str, prompt: str, max_retries: int = 4) -> str:
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(model=model, input=prompt)
            out = getattr(resp, "output_text", None)
            if not out:
                out = str(resp)
            return out.strip()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise

def build_steps_prompt(doc_text: str, question: str) -> str:
    return f"""
You are an expert data analyst and SQL planner.

Task:
Decompose the question into a SEQUENCE of step-by-step sub-queries / reasoning steps.

Output format (STRICT):
Return ONLY valid JSON (no markdown, no extra text).
Schema:

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
- Do NOT write executable SQL.

[Doc snippet]
{doc_text}

[Question]
{question}

Return ONLY JSON:
""".strip()

def build_sql_prompt(doc_text: str, steps_json: Dict[str, Any], dialect: str) -> str:
    return f"""
You are an expert analytics engineer and SQL writer.

Goal:
Given doc snippet + steps JSON, produce ONE final SQL query using CTEs.

Requirements:
- Output ONLY SQL.
- Use CTE names step1, step2, ...
- Dialect: {dialect}

BigQuery / GA4 conventions (IMPORTANT):
- Prefer wildcard tables `analytics_<property_id>.events_*`
- Prefer `_TABLE_SUFFIX` for date ranges.

Doc snippet:
{doc_text}

Steps JSON:
{json.dumps(steps_json, ensure_ascii=False, indent=2)}

Now output SQL only:
""".strip()


# ---------------- UI ----------------

st.set_page_config(page_title="Spider2 Multi-hop Demo", layout="wide")
st.title("Spider2 Multi-hop (Steps → SQL) Demo")

with st.sidebar:
    st.header("Settings")
    spider2_root = st.text_input("spider2_root", value="")
    split = st.selectbox("split", ["train", "validation", "test"], index=0)
    idx = st.number_input("example idx", min_value=0, value=0, step=1)
    model_steps = st.text_input("model_steps", value="gpt-4o-mini")
    model_sql = st.text_input("model_sql", value="gpt-4o-mini")
    dialect = st.selectbox("dialect", ["BigQuery", "SQLite", "PostgreSQL"], index=0)
    max_doc_chars = st.slider("max_doc_chars", 2000, 20000, 12000, step=1000)
    out_dir = st.text_input("out_dir", value="tmp/spider2_web_demo")

@st.cache_data(show_spinner=False)
def load_spider2(split: str):
    return load_dataset("xlangai/spider2-lite", split=split)

ds = load_spider2(split)
total = len(ds)

if idx >= total:
    st.error(f"idx={idx} out of range. total={total}")
    st.stop()

ex = ds[int(idx)]
instance_id = ex.get("instance_id", "")
db = ex.get("db", "")
question = ex.get("question", "")
doc_name = ex.get("external_knowledge", None)
doc_name = doc_name if isinstance(doc_name, str) else ""

doc_path = resolve_doc_path(spider2_root, doc_name) if spider2_root else None
doc_missing = (not doc_path)

colA, colB = st.columns(2)

with colA:
    st.subheader("Example")
    st.write({"idx": int(idx), "instance_id": instance_id, "db": db, "doc_name": doc_name, "doc_missing": doc_missing})
    st.markdown("### Question")
    st.write(question)

with colB:
    st.subheader("Doc head")
    if doc_path:
        doc_text = read_text(str(doc_path), max_chars=max_doc_chars)
        st.code(doc_text[:1200], language="text")
    else:
        doc_text = ""
        st.info("No external doc found / provided for this example. (doc_text will be empty)")

# state
if "steps_text" not in st.session_state:
    st.session_state.steps_text = ""
if "sql_text" not in st.session_state:
    st.session_state.sql_text = ""
if "diag" not in st.session_state:
    st.session_state.diag = {}

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Steps JSON")
    if st.button("Generate Steps", use_container_width=True):
        prompt = build_steps_prompt(doc_text=doc_text, question=question)
        out = call_openai_text(model=model_steps, prompt=prompt)
        steps_json = safe_json_loads(out)
        st.session_state.steps_text = json.dumps(steps_json, ensure_ascii=False, indent=2)

    st.session_state.steps_text = st.text_area(
        "Editable Steps JSON",
        value=st.session_state.steps_text,
        height=420,
    )

with c2:
    st.subheader("SQL")
    if st.button("Generate SQL from Steps", use_container_width=True):
        try:
            steps_json = safe_json_loads(st.session_state.steps_text)
        except Exception as e:
            st.error(f"Steps JSON parse error: {e}")
            st.stop()

        prompt = build_sql_prompt(doc_text=doc_text, steps_json=steps_json, dialect=dialect)
        sql = call_openai_text(model=model_sql, prompt=prompt)
        st.session_state.sql_text = sql

        st.session_state.diag = {
            "contains_property_id_placeholder": "<property_id>" in sql,
            "has_events_wildcard": ("events_*" in sql) or ("events_*" in sql.lower()),
            "has_table_suffix": "_TABLE_SUFFIX" in sql,
            "doc_missing": doc_missing,
        }

    st.session_state.sql_text = st.text_area(
        "Editable SQL",
        value=st.session_state.sql_text,
        height=420,
    )

st.markdown("### Diagnostics")
st.json(st.session_state.diag or {})

if st.button("Save artifacts", use_container_width=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    (out / "sql").mkdir(parents=True, exist_ok=True)

    steps_path = out / "steps" / f"{int(idx):04d}_{instance_id}.json"
    sql_path = out / "sql" / f"{int(idx):04d}_{instance_id}.sql"

    steps_path.write_text(st.session_state.steps_text + "\n", encoding="utf-8")
    sql_path.write_text(st.session_state.sql_text + "\n", encoding="utf-8")

    st.success(f"Saved:\n- {steps_path}\n- {sql_path}")
