import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

from datasets import load_dataset
from openai import OpenAI


# ---- shared helpers (kept local so this script is standalone) ----

def read_text(path: str, max_chars: int = 12000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def find_api_key() -> Optional[str]:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPEN_AI_KEY")
    )


def call_openai_text(model: str, prompt: str, max_retries: int = 4) -> str:
    api_key = find_api_key()
    if not api_key:
        raise RuntimeError("No API key found in env. (OPENAI_API_KEY / OPENAI_KEY / OPEN_AI_KEY)")

    client = OpenAI(api_key=api_key)

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
            sleep_s = min(2 ** attempt, 20)
            print(f"[WARN] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[WARN] Sleeping {sleep_s}s and retrying...")
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")


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
      "goal": <string, what this step aims to compute>,
      "tables_or_entities": <array of strings, which tables/entities are involved>,
      "filters_or_conditions": <array of strings, key filters/conditions>,
      "intermediate_output": <string, what intermediate result is produced>,
      "depends_on_step_ids": <array of ints, which previous steps this step depends on>
    }}
  ],
  "final_answer_derivation": <string, how to obtain the final answer from the last step output>
}}

Constraints:
- Do NOT write executable SQL. Describe the operations in natural language.
- Be concrete (date ranges, sets, joins, grouping) but keep it implementation-ready.
- If the doc snippet does not specify exact column names, state assumptions briefly in filters/conditions.
- Keep steps minimal but complete (typically 3-7 steps).

[Database documentation snippet]
{doc_text}

[Question]
{question}

Return ONLY JSON:
""".strip()


def build_sql_prompt(doc_text: str, steps_json: Dict[str, Any], dialect: str) -> str:
    return f"""
You are an expert analytics engineer and SQL writer.

Goal:
Given (1) a documentation snippet and (2) a multi-hop decomposition plan (JSON steps),
produce ONE final SQL query that follows the steps using CTEs.

Requirements:
- Output ONLY SQL (no markdown, no explanations).
- Use CTE names that match the steps, e.g. step1, step2, step3...
- The final SELECT should return the final answer.
- Dialect: {dialect}

BigQuery / GA4 conventions (IMPORTANT):
- Prefer querying wildcard tables (e.g. `analytics_<property_id>.events_*`).
- Prefer filtering by `_TABLE_SUFFIX` for date ranges, e.g.
  `_TABLE_SUFFIX BETWEEN '20210101' AND '20210107'`
  rather than hardcoding multiple daily tables or using event_date only.
- If you also use `event_date`, keep it consistent as YYYYMMDD string.

Columns:
- If exact column names are not explicitly specified in the doc snippet, make reasonable GA4 assumptions:
  `user_pseudo_id`, `event_name`, `_TABLE_SUFFIX` and/or `event_date`.
- Keep assumptions consistent across all steps.

Style:
- Avoid unnecessary SELECT *.
- Keep it minimal and implementation-ready.

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


def postprocess_sql_for_table_suffix(sql: str) -> Tuple[str, Dict[str, Any]]:
    info = {"applied": False, "changed_patterns": []}

    # If it's not wildcard style, don't touch
    if "events_*" not in sql and "events_*" not in sql.lower():
        return sql.strip(), info
    if "_TABLE_SUFFIX" in sql:
        return sql.strip(), info

    import re
    new_sql = sql

    # event_date BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'
    pattern_between = re.compile(
        r"(event_date\s+BETWEEN\s+'(\d{8})'\s+AND\s+'(\d{8})')",
        flags=re.IGNORECASE,
    )
    if pattern_between.search(new_sql):
        new_sql = pattern_between.sub(r"_TABLE_SUFFIX BETWEEN '\2' AND '\3'", new_sql)
        info["applied"] = True
        info["changed_patterns"].append("event_date BETWEEN -> _TABLE_SUFFIX BETWEEN")

    # event_date = 'YYYYMMDD'
    pattern_eq = re.compile(r"(event_date\s*=\s*'(\d{8})')", flags=re.IGNORECASE)
    if pattern_eq.search(new_sql):
        new_sql = pattern_eq.sub(r"_TABLE_SUFFIX = '\2'", new_sql)
        info["applied"] = True
        info["changed_patterns"].append("event_date = -> _TABLE_SUFFIX =")

    return new_sql.strip(), info


def write_report_md(
    report_path: Path,
    *,
    idx: int,
    instance_id: str,
    db: str,
    question: str,
    doc_path_str: str,
    doc_missing: bool,
    model_steps: str,
    model_sql: str,
    dialect: str,
    steps_json: Dict[str, Any],
    sql: str,
    doc_head: str,
    postprocess_info: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    diag = {
        "doc_missing": doc_missing,
        "contains_property_id_placeholder": "<property_id>" in sql,
        "has_events_wildcard": ("events_*" in sql) or ("events_*" in sql.lower()),
        "has_table_suffix": "_TABLE_SUFFIX" in sql,
    }

    lines = []
    lines.append("# Spider2 Batch Report")
    lines.append("")
    lines.append(f"- Time: {ts}")
    lines.append(f"- idx: {idx}")
    lines.append(f"- instance_id: {instance_id}")
    lines.append(f"- db: {db}")
    lines.append(f"- dialect: `{dialect}`")
    lines.append(f"- model_steps: `{model_steps}`")
    lines.append(f"- model_sql: `{model_sql}`")
    lines.append(f"- doc_path: `{doc_path_str}`")
    lines.append("")

    lines.append("## Question")
    lines.append("")
    lines.append(question.strip())
    lines.append("")

    lines.append("## Doc head")
    lines.append("")
    lines.append("```text")
    lines.append((doc_head or "[NO DOC PROVIDED]").strip())
    lines.append("```")
    lines.append("")

    lines.append("## Steps JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(steps_json, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## SQL")
    lines.append("")
    lines.append("```sql")
    lines.append(sql.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Diagnostics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"postprocess": postprocess_info, "diag": diag}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_doc_path(spider2_root: str, doc_name: str) -> Path:
    p = Path(spider2_root) / "spider2-lite" / "resource" / "documents" / doc_name
    if not p.exists():
        raise FileNotFoundError(f"Doc not found: {p}")
    return p


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            sub = text[start:end + 1]
            return json.loads(sub)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider2_root", required=True, help="Path to xlang-spider2 root")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out_dir", default="tmp/spider2_batch10")
    parser.add_argument("--max_doc_chars", type=int, default=3000)
    parser.add_argument("--doc_head_chars", type=int, default=1200)

    parser.add_argument("--model_steps", default="gpt-4o-mini")
    parser.add_argument("--model_sql", default="gpt-4o-mini")
    parser.add_argument("--dialect", default="BigQuery")

    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--skip_missing_doc", action="store_true",
                        help="Skip examples whose external_knowledge is None/empty.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    steps_dir = out_dir / "steps"
    sql_dir = out_dir / "sql"
    report_dir = out_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    sql_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"

    print(f"[INFO] Loading dataset xlangai/spider2-lite ...")
    ds = load_dataset("xlangai/spider2-lite", split=args.split)
    total = len(ds)
    print(f"[INFO] Split={args.split}, total={total}, start_idx={args.start_idx}, n={args.n}")

    rows = []
    for idx in range(args.start_idx, min(args.start_idx + args.n, total)):
        ex = ds[idx]
        instance_id = ex.get("instance_id", "")
        db = ex.get("db", "")
        question = ex.get("question", "")

        doc_name_raw = ex.get("external_knowledge", None)
        doc_name = doc_name_raw if isinstance(doc_name_raw, str) else ""
        doc_missing = (not doc_name)

        row = {
            "idx": idx,
            "instance_id": instance_id,
            "db": db,
            "doc_name": doc_name_raw,
            "doc_missing": doc_missing,
            "question": question,
            "steps_path": "",
            "sql_path": "",
            "report_path": "",
            "ok_steps": False,
            "ok_sql": False,
            "err_steps": "",
            "err_sql": "",
            "latency_steps_s": "",
            "latency_sql_s": "",
        }

        print("\n" + "=" * 80)
        print(f"[INFO] Example idx={idx} | instance_id={instance_id} | db={db}")
        print(f"[INFO] doc_name={doc_name_raw}")
        print(f"[INFO] question={question}")

        if doc_missing and args.skip_missing_doc:
            row["err_steps"] = "doc_missing (skipped)"
            rows.append(row)
            print("[WARN] external_knowledge is missing; skipped due to --skip_missing_doc")
            continue

        # Resolve doc (if exists) else use empty
        doc_path_str = ""
        doc_text = ""
        doc_head = ""
        if not doc_missing:
            try:
                doc_path = resolve_doc_path(args.spider2_root, doc_name)
                doc_path_str = str(doc_path)
                doc_text = read_text(str(doc_path), max_chars=args.max_doc_chars)
                doc_head = read_text(str(doc_path), max_chars=args.doc_head_chars)
            except Exception as e:
                row["err_steps"] = f"doc_error: {e}"
                rows.append(row)
                print(f"[ERROR] Doc resolve failed: {e}")
                continue
        else:
            doc_path_str = "(NO DOC PROVIDED)"
            doc_text = ""
            doc_head = ""

        # 1) Steps JSON
        steps_json = None
        t0 = time.time()
        try:
            steps_prompt = build_steps_prompt(doc_text=doc_text, question=question)
            steps_out = call_openai_text(model=args.model_steps, prompt=steps_prompt)
            steps_json = safe_json_loads(steps_out)
            row["ok_steps"] = True
            row["latency_steps_s"] = f"{time.time() - t0:.2f}"
        except Exception as e:
            row["err_steps"] = str(e)
            row["latency_steps_s"] = f"{time.time() - t0:.2f}"
            rows.append(row)
            print(f"[ERROR] steps generation failed: {e}")
            continue

        steps_path = steps_dir / f"{idx:04d}_{instance_id}.json"
        steps_path.write_text(json.dumps(steps_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row["steps_path"] = str(steps_path)

        # 2) Steps -> SQL
        t1 = time.time()
        sql = ""
        post_info = {"applied": False, "changed_patterns": []}
        try:
            sql_prompt = build_sql_prompt(doc_text=doc_text, steps_json=steps_json, dialect=args.dialect)
            sql_out = call_openai_text(model=args.model_sql, prompt=sql_prompt)
            sql = sql_out.strip()
            if not args.no_postprocess:
                sql, post_info = postprocess_sql_for_table_suffix(sql)
            row["ok_sql"] = True
            row["latency_sql_s"] = f"{time.time() - t1:.2f}"
        except Exception as e:
            row["err_sql"] = str(e)
            row["latency_sql_s"] = f"{time.time() - t1:.2f}"
            rows.append(row)
            print(f"[ERROR] SQL generation failed: {e}")
            continue

        sql_path = sql_dir / f"{idx:04d}_{instance_id}.sql"
        sql_path.write_text(sql.strip() + "\n", encoding="utf-8")
        row["sql_path"] = str(sql_path)

        # 3) Report
        report_path = report_dir / f"{idx:04d}_{instance_id}.report.md"
        write_report_md(
            report_path,
            idx=idx,
            instance_id=instance_id,
            db=db,
            question=question,
            doc_path_str=doc_path_str,
            doc_missing=doc_missing,
            model_steps=args.model_steps,
            model_sql=args.model_sql,
            dialect=args.dialect,
            steps_json=steps_json,
            sql=sql,
            doc_head=doc_head,
            postprocess_info=post_info,
        )
        row["report_path"] = str(report_path)

        print(f"[OK] steps:  {steps_path}")
        print(f"[OK] sql:    {sql_path}")
        print(f"[OK] report: {report_path}")

        rows.append(row)

    # Write summary.csv
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    print("\n" + "=" * 80)
    print(f"[DONE] Wrote summary: {summary_path}")
    print(f"[DONE] Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
