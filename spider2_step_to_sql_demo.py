import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from openai import OpenAI


def read_text(path: str, max_chars: int = 12000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.compile(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", re.MULTILINE)
    m = fence.match(text)
    if m:
        return m.group(1).strip()
    return text


def find_api_key() -> Optional[str]:
    # Support a few common env var names (no extra nagging in terminal output)
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPEN_AI_KEY")
    )


def build_prompt(doc_text: str, steps_json: Dict[str, Any], dialect: str) -> str:
    """
    Ask the model to produce ONE final SQL query with CTEs.

    Key improvement:
    - Strongly prefer wildcard tables + _TABLE_SUFFIX in BigQuery style
      (especially for GA4 `events_*` tables).
    """
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


def call_openai_text(model: str, prompt: str, max_retries: int = 4) -> str:
    api_key = find_api_key()
    if not api_key:
        raise RuntimeError(
            "No API key found in env. (Expected OPENAI_API_KEY / OPENAI_KEY / OPEN_AI_KEY)"
        )

    client = OpenAI(api_key=api_key)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            out = getattr(resp, "output_text", None)
            if not out:
                out = str(resp)
            return strip_code_fences(out)

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20)
            print(f"[WARN] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[WARN] Sleeping {sleep_s}s and retrying...")
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")


def _has_events_wildcard(sql: str) -> bool:
    # rough detection for GA4-style wildcard
    return bool(re.search(r"events_\*", sql, flags=re.IGNORECASE))


def _has_table_suffix(sql: str) -> bool:
    return "_TABLE_SUFFIX" in sql


def postprocess_sql_for_table_suffix(sql: str) -> Tuple[str, Dict[str, Any]]:
    """
    If SQL uses wildcard events_* but filters only event_date, convert common patterns to _TABLE_SUFFIX.
    This is a best-effort heuristic (demo-friendly, not perfect).
    """
    info = {
        "applied": False,
        "changed_patterns": [],
    }

    if not _has_events_wildcard(sql):
        return sql, info

    if _has_table_suffix(sql):
        # already good
        return sql, info

    new_sql = sql

    # event_date BETWEEN 'YYYYMMDD' AND 'YYYYMMDD' -> _TABLE_SUFFIX BETWEEN ...
    pattern_between = re.compile(
        r"(event_date\s+BETWEEN\s+'(\d{8})'\s+AND\s+'(\d{8})')",
        flags=re.IGNORECASE,
    )
    if pattern_between.search(new_sql):
        new_sql = pattern_between.sub(r"_TABLE_SUFFIX BETWEEN '\2' AND '\3'", new_sql)
        info["applied"] = True
        info["changed_patterns"].append("event_date BETWEEN -> _TABLE_SUFFIX BETWEEN")

    # event_date = 'YYYYMMDD' -> _TABLE_SUFFIX = 'YYYYMMDD'
    pattern_eq = re.compile(
        r"(event_date\s*=\s*'(\d{8})')",
        flags=re.IGNORECASE,
    )
    if pattern_eq.search(new_sql):
        new_sql = pattern_eq.sub(r"_TABLE_SUFFIX = '\2'", new_sql)
        info["applied"] = True
        info["changed_patterns"].append("event_date = -> _TABLE_SUFFIX =")

    return new_sql, info


def diagnose_sql(sql: str) -> Dict[str, Any]:
    diag = {
        "has_events_wildcard": _has_events_wildcard(sql),
        "has_table_suffix": _has_table_suffix(sql),
        "contains_property_id_placeholder": "<property_id>" in sql,
        "looks_like_sql": bool(re.search(r"\bSELECT\b", sql, re.IGNORECASE)),
    }
    return diag


def write_report_md(
    report_path: Path,
    *,
    question: str,
    doc_path: str,
    model: str,
    dialect: str,
    steps_json: Dict[str, Any],
    sql: str,
    doc_head: str,
    postprocess_info: Dict[str, Any],
    diag: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Spider2 Step→SQL Demo Report")
    lines.append("")
    lines.append(f"- Time: {ts}")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Dialect: `{dialect}`")
    lines.append(f"- Doc path: `{doc_path}`")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(question.strip())
    lines.append("")
    lines.append("## Doc head (snippet)")
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
    lines.append("## Generated SQL")
    lines.append("")
    lines.append("```sql")
    lines.append(sql.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"postprocess": postprocess_info, "diagnostics": diag}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_json", required=True, help="Path to steps JSON file (from spider2_multihop_demo.py)")
    parser.add_argument("--doc_path", required=True, help="Path to the Spider2 document markdown file")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--dialect", default="BigQuery", help="SQL dialect label for prompting")
    parser.add_argument("--max_doc_chars", type=int, default=3000, help="Max chars of doc to include in prompt")
    parser.add_argument("--out_sql", default="tmp/step_to_sql.sql", help="Output SQL file path")

    # new:
    parser.add_argument("--out_report", default=None, help="Output report markdown path (default: <out_sql>.report.md)")
    parser.add_argument("--doc_head_chars", type=int, default=1200, help="How many doc chars to include in report head")
    parser.add_argument("--no_postprocess", action="store_true", help="Disable _TABLE_SUFFIX post-processing")
    parser.add_argument("--print_prompt", action="store_true", help="Print prompt for debugging (truncated)")

    args = parser.parse_args()

    steps_path = Path(args.steps_json)
    doc_path = Path(args.doc_path)
    out_sql_path = Path(args.out_sql)
    out_report_path = Path(args.out_report) if args.out_report else Path(str(out_sql_path) + ".report.md")

    steps = json.loads(steps_path.read_text(encoding="utf-8"))
    doc_text = read_text(str(doc_path), max_chars=args.max_doc_chars)
    doc_head = read_text(str(doc_path), max_chars=args.doc_head_chars)

    question = steps.get("question", "")

    prompt = build_prompt(doc_text=doc_text, steps_json=steps, dialect=args.dialect)

    if args.print_prompt:
        print("===== PROMPT (truncated to 2000 chars) =====")
        print(prompt[:2000])
        print("===== END PROMPT =====")

    print("[INFO] Generating SQL from steps JSON ...")
    sql = call_openai_text(model=args.model, prompt=prompt)

    postprocess_info = {"applied": False, "changed_patterns": []}
    if not args.no_postprocess:
        sql2, postprocess_info = postprocess_sql_for_table_suffix(sql)
        sql = sql2

    diag = diagnose_sql(sql)

    out_sql_path.parent.mkdir(parents=True, exist_ok=True)
    out_sql_path.write_text(sql.strip() + "\n", encoding="utf-8")

    write_report_md(
        out_report_path,
        question=question,
        doc_path=str(doc_path),
        model=args.model,
        dialect=args.dialect,
        steps_json=steps,
        sql=sql,
        doc_head=doc_head,
        postprocess_info=postprocess_info,
        diag=diag,
    )

    print(f"[OK] SQL saved to: {out_sql_path}")
    print(f"[OK] Report saved to: {out_report_path}")
    print("\n===== SQL PREVIEW (first 80 lines) =====")
    for i, line in enumerate(sql.splitlines()[:80], start=1):
        print(f"{i:02d} | {line}")


if __name__ == "__main__":
    main()
