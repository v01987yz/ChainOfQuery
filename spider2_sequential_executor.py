import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI


# -----------------------------
# Utils
# -----------------------------

def read_text(path: str, max_chars: int) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.compile(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", re.MULTILINE)
    m = fence.match(text)
    if m:
        return m.group(1).strip()
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # attempt recovery: grab the largest {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def find_api_key() -> Optional[str]:
    # You said this is already solved; we just read it.
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


def call_openai_text(model: str, prompt: str, throttle: Throttle, max_retries: int = 4) -> str:
    api_key = find_api_key()
    if not api_key:
        raise RuntimeError("No API key found in env var OPENAI_API_KEY (or OPENAI_KEY / OPEN_AI_KEY).")

    client = OpenAI(api_key=api_key)

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
            sleep_s = min(2 ** attempt, 20)
            print(f"[WARN] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[WARN] Sleeping {sleep_s}s then retry ...")
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")


# -----------------------------
# Spider2 doc resolving
# -----------------------------

def resolve_doc_path(spider2_root: str, doc_name: str) -> Path:
    # /.../xlang-spider2/spider2-lite/resource/documents/<doc_name>
    p = Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name
    if not p.exists():
        raise FileNotFoundError(f"Doc not found: {p}")
    return p


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
- Do NOT write executable SQL. Describe the operations in natural language.
- Be concrete but implementation-ready.
- Keep steps minimal but complete (typically 3-7 steps).

[Database documentation snippet]
{doc_text}

[Question]
{question}

Return ONLY JSON:
""".strip()


def build_step_sql_prompt(
    *,
    doc_text: str,
    question: str,
    steps_json: Dict[str, Any],
    step: Dict[str, Any],
    dialect: str,
    prefer_table_suffix: bool,
) -> str:
    step_id = step["step_id"]
    cte_name = f"step{step_id}"

    ga4_rules = ""
    if prefer_table_suffix:
        ga4_rules = """
BigQuery / GA4 conventions (IMPORTANT):
- Prefer querying wildcard tables (e.g. `analytics_<property_id>.events_*`).
- Prefer filtering by `_TABLE_SUFFIX` for date ranges (e.g. `_TABLE_SUFFIX BETWEEN '20210101' AND '20210107'`)
  instead of hardcoding daily tables.
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

CTE naming:
- You may reference previous steps by CTE names: step1, step2, ... step{step_id-1}
  (only if this step depends on them).

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


def build_final_select_prompt(
    *,
    question: str,
    steps_json: Dict[str, Any],
    dialect: str,
) -> str:
    """
    Ask model for final SELECT that reads from last step only.
    We still keep it simple: SELECT * FROM stepN; is often enough,
    but for safety let the model decide if needs projection.
    """
    steps = steps_json.get("steps", [])
    last_id = steps[-1]["step_id"] if steps else 1
    return f"""
You are an expert SQL writer.

Given a multi-step plan, produce ONLY the FINAL SELECT statement
that returns the final answer using the last CTE step{last_id}.

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
# SQL cleanup / normalization
# -----------------------------

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
    If model returns:
      WITH step1 AS ( SELECT ... ) SELECT ...
    or
      step1 AS ( SELECT ... )
    extract the inside of step1 AS ( ... )
    """
    t = strip_code_fences(sql_text).strip()
    t = t.rstrip(";").strip()

    # pattern: optional WITH, then stepX AS (
    pat = re.compile(rf"(?:WITH\s+)?{re.escape(cte_name)}\s+AS\s*\(", re.IGNORECASE)
    m = pat.search(t)
    if not m:
        return t

    open_paren = t.find("(", m.end() - 1)
    if open_paren < 0:
        return t

    close_paren = _find_matching_paren(t, open_paren)
    if close_paren < 0:
        return t

    body = t[open_paren + 1 : close_paren].strip()
    body = body.rstrip(";").strip()

    # sometimes it returns nested "stepX AS (" again; unwrap once more
    if re.search(rf"^{re.escape(cte_name)}\s+AS\s*\(", body, re.IGNORECASE):
        return extract_cte_body_if_wrapped(body, cte_name)
    if re.search(r"^WITH\s", body, re.IGNORECASE):
        # if body still starts with WITH, keep as-is (rare); better return original t
        return body

    return body


def enforce_table_suffix(sql_body: str) -> Tuple[str, Dict[str, Any]]:
    """
    Best-effort: if step body uses events_* and filters event_date, rewrite to _TABLE_SUFFIX.
    """
    info = {"applied": False, "changed": []}
    s = sql_body

    # only for wildcard style
    if "events_*" not in s and "events_*" not in s.lower():
        return s, info
    if "_TABLE_SUFFIX" in s:
        return s, info

    # event_date BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'
    between_pat = re.compile(r"event_date\s+BETWEEN\s+'(\d{8})'\s+AND\s+'(\d{8})'", re.IGNORECASE)
    if between_pat.search(s):
        s = between_pat.sub(r"_TABLE_SUFFIX BETWEEN '\1' AND '\2'", s)
        info["applied"] = True
        info["changed"].append("event_date BETWEEN -> _TABLE_SUFFIX BETWEEN")

    # event_date = 'YYYYMMDD'
    eq_pat = re.compile(r"event_date\s*=\s*'(\d{8})'", re.IGNORECASE)
    if eq_pat.search(s):
        s = eq_pat.sub(r"_TABLE_SUFFIX = '\1'", s)
        info["applied"] = True
        info["changed"].append("event_date = -> _TABLE_SUFFIX =")

    return s.strip(), info


# -----------------------------
# Reporting
# -----------------------------

def write_report(
    report_path: Path,
    *,
    question: str,
    instance_id: str,
    db: str,
    doc_path: str,
    doc_head: str,
    model_steps: str,
    model_step_sql: str,
    dialect: str,
    steps_json: Dict[str, Any],
    step_sql_map: Dict[str, str],
    final_sql: str,
    final_select: str,
    timings: Dict[str, Any],
    postprocess_notes: List[Dict[str, Any]],
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Spider2 Sequential Executor Report")
    lines.append("")
    lines.append(f"- Time: {ts}")
    lines.append(f"- instance_id: {instance_id}")
    lines.append(f"- db: {db}")
    lines.append(f"- dialect: `{dialect}`")
    lines.append(f"- model_steps: `{model_steps}`")
    lines.append(f"- model_step_sql: `{model_step_sql}`")
    lines.append(f"- doc_path: `{doc_path}`")
    lines.append("")

    lines.append("## Question")
    lines.append("")
    lines.append(question.strip())
    lines.append("")

    lines.append("## Doc head")
    lines.append("")
    lines.append("```text")
    lines.append(doc_head.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Steps JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(steps_json, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Per-step SQL (CTE bodies)")
    lines.append("")
    for k, v in step_sql_map.items():
        lines.append(f"### {k}")
        lines.append("")
        lines.append("```sql")
        lines.append(v.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Final SELECT")
    lines.append("")
    lines.append("```sql")
    lines.append(final_select.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Final SQL")
    lines.append("")
    lines.append("```sql")
    lines.append(final_sql.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Timings / Notes")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"timings": timings, "postprocess": postprocess_notes}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--spider2_root", required=True)

    parser.add_argument("--out_dir", default="tmp/seq_run")
    parser.add_argument("--max_doc_chars", type=int, default=2500)
    parser.add_argument("--doc_head_chars", type=int, default=1200)

    parser.add_argument("--model_steps", default="gpt-4o-mini")
    parser.add_argument("--model_step_sql", default="gpt-4o-mini")
    parser.add_argument("--model_final_select", default="gpt-4o-mini")

    parser.add_argument("--dialect", default="BigQuery")
    parser.add_argument("--prefer_table_suffix", action="store_true")

    parser.add_argument("--min_call_interval_s", type=float, default=22.0)
    parser.add_argument("--sleep_between_steps_s", type=float, default=2.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_sql_dir = out_dir / "step_sql"
    step_sql_dir.mkdir(parents=True, exist_ok=True)

    throttle = Throttle(min_interval_s=args.min_call_interval_s)

    print("[INFO] Loading dataset xlangai/spider2-lite ...")
    ds = load_dataset("xlangai/spider2-lite", split=args.split)
    ex = ds[args.idx]

    instance_id = ex.get("instance_id", "")
    db = ex.get("db", "")
    question = ex.get("question", "")
    doc_name = ex.get("external_knowledge", None)

    print(f"[INFO] Using example idx = {args.idx}")
    print(f"Instance ID : {instance_id}")
    print(f"DB          : {db}")
    print(f"Question    : {question}")
    print(f"Doc name    : {doc_name}")

    if not isinstance(doc_name, str) or not doc_name:
        raise RuntimeError("external_knowledge is missing for this example (doc_name is None/empty).")

    print(f"\n[INFO] Searching for doc '{doc_name}' under {args.spider2_root} ...")
    doc_path = resolve_doc_path(args.spider2_root, doc_name)
    print(f"[INFO] Found doc file: {doc_path}")

    doc_text = read_text(str(doc_path), max_chars=args.max_doc_chars)
    doc_head = read_text(str(doc_path), max_chars=args.doc_head_chars)

    timings: Dict[str, Any] = {}
    postprocess_notes: List[Dict[str, Any]] = []

    # 1) Generate steps JSON
    print("[INFO] Generating steps JSON ...")
    t0 = time.time()
    steps_prompt = build_steps_prompt(doc_text=doc_text, question=question)
    steps_raw = call_openai_text(model=args.model_steps, prompt=steps_prompt, throttle=throttle)
    steps_json = safe_json_loads(steps_raw)
    timings["steps_s"] = round(time.time() - t0, 2)

    steps_path = out_dir / "steps.json"
    steps_path.write_text(json.dumps(steps_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] steps saved: {steps_path}")

    # 2) Sequential per-step SQL (CTE bodies)
    print("[INFO] Generating per-step SQL sequentially ...")
    step_sql_map: Dict[str, str] = {}
    steps: List[Dict[str, Any]] = steps_json.get("steps", [])
    if not steps:
        raise RuntimeError("steps_json has no 'steps'.")

    t1 = time.time()
    for step in steps:
        sid = int(step["step_id"])
        cte_name = f"step{sid}"
        print(f"[INFO] Step {sid}: {step.get('goal','').strip()}")

        step_prompt = build_step_sql_prompt(
            doc_text=doc_text,
            question=question,
            steps_json=steps_json,
            step=step,
            dialect=args.dialect,
            prefer_table_suffix=args.prefer_table_suffix,
        )

        raw_sql = call_openai_text(model=args.model_step_sql, prompt=step_prompt, throttle=throttle)
        cleaned = strip_code_fences(raw_sql).strip().rstrip(";").strip()

        # If model still returned WITH stepX AS (...) or stepX AS (...), extract body
        cleaned2 = extract_cte_body_if_wrapped(cleaned, cte_name)

        # Optional: enforce _TABLE_SUFFIX rewrite
        post = {"cte": cte_name, "table_suffix": {"applied": False, "changed": []}}
        if args.prefer_table_suffix:
            cleaned2, info = enforce_table_suffix(cleaned2)
            post["table_suffix"] = info
        postprocess_notes.append(post)

        step_sql_map[cte_name] = cleaned2

        step_file = step_sql_dir / f"{sid:02d}_{cte_name}.sql"
        step_file.write_text(cleaned2.strip() + "\n", encoding="utf-8")
        print(f"[OK] Saved {cte_name} SQL: {step_file}")

        if args.sleep_between_steps_s > 0:
            time.sleep(args.sleep_between_steps_s)

    timings["steps_sql_s"] = round(time.time() - t1, 2)

    # 3) Final SELECT (from last step)
    t2 = time.time()
    final_select_prompt = build_final_select_prompt(
        question=question,
        steps_json=steps_json,
        dialect=args.dialect,
    )
    final_select_raw = call_openai_text(model=args.model_final_select, prompt=final_select_prompt, throttle=throttle)
    final_select = strip_code_fences(final_select_raw).strip().rstrip(";").strip()
    # Guard: if it accidentally returns WITH, just fallback to SELECT * FROM last step
    if re.search(r"^\s*WITH\s", final_select, re.IGNORECASE):
        last_id = steps[-1]["step_id"]
        final_select = f"SELECT * FROM step{last_id}"
    timings["final_select_s"] = round(time.time() - t2, 2)

    # 4) Assemble final SQL
    cte_lines: List[str] = ["WITH"]
    for i, step in enumerate(steps):
        sid = int(step["step_id"])
        cte_name = f"step{sid}"
        body = step_sql_map[cte_name].strip()
        comma = "," if i < len(steps) - 1 else ""
        cte_lines.append(f"{cte_name} AS (")
        cte_lines.append(body)
        cte_lines.append(f"){comma}")

    final_sql = "\n".join(cte_lines) + "\n" + final_select + ";\n"

    final_sql_path = out_dir / "final.sql"
    final_sql_path.write_text(final_sql, encoding="utf-8")
    print(f"[OK] Final SQL saved: {final_sql_path}")

    # 5) Report
    report_path = out_dir / "report.md"
    write_report(
        report_path,
        question=question,
        instance_id=instance_id,
        db=db,
        doc_path=str(doc_path),
        doc_head=doc_head,
        model_steps=args.model_steps,
        model_step_sql=args.model_step_sql,
        dialect=args.dialect,
        steps_json=steps_json,
        step_sql_map=step_sql_map,
        final_sql=final_sql,
        final_select=final_select,
        timings=timings,
        postprocess_notes=postprocess_notes,
    )
    print(f"[OK] Report saved: {report_path}")

    # Preview
    print("\n===== FINAL SQL PREVIEW (first 60 lines) =====")
    for i, line in enumerate(final_sql.splitlines()[:60], start=1):
        print(f"{i:02d} | {line}")


if __name__ == "__main__":
    main()
