import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset

# OpenAI Python SDK (you have openai==2.8.0)
from openai import OpenAI


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_spider2_lite(split: str) -> Any:
    print("[INFO] Loading dataset xlangai/spider2-lite ...")
    ds = load_dataset("xlangai/spider2-lite")
    if split not in ds:
        # pick a reasonable default
        available = list(ds.keys())
        raise ValueError(
            f"Split '{split}' not found in dataset. Available splits: {available}"
        )
    return ds[split]


def find_doc_file(spider2_root: Path, doc_name: str) -> Path:
    """
    Spider2-lite repo structure usually:
      <root>/spider2-lite/resource/documents/<doc_name>
    We'll check common locations first, then fallback to rglob.
    """
    candidates = [
        spider2_root / "spider2-lite" / "resource" / "documents" / doc_name,
        spider2_root / "spider2-lite" / "resources" / "documents" / doc_name,
        spider2_root / "resource" / "documents" / doc_name,
        spider2_root / "resources" / "documents" / doc_name,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # Fallback: search (can be slow on huge repos, but fine for your local case)
    matches = list(spider2_root.rglob(doc_name))
    for m in matches:
        if m.is_file():
            return m

    raise FileNotFoundError(
        f"Could not find doc '{doc_name}' under spider2_root={spider2_root}"
    )


def read_doc_head(doc_path: Path, max_chars: int) -> str:
    # Docs are usually markdown; read as utf-8 with errors ignored
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def build_prompt(question: str, doc_snippet: str) -> str:
    """
    Ask for STRICT JSON output so we can parse it reliably.
    """
    return f"""You are an expert data analyst and SQL planner.

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
{doc_snippet}

[Question]
{question}
"""


def safe_json_load(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def call_openai_json(client: OpenAI, model: str, prompt: str, max_retries: int = 4) -> str:
    """
    Basic retry for timeouts / rate limits.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful assistant that outputs strictly valid JSON when asked."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            # Simple backoff
            wait = 2 ** attempt
            eprint(f"[WARN] OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
            eprint(f"[WARN] Sleeping {wait}s then retry ...")
            time.sleep(wait)

    raise RuntimeError(f"OpenAI API failed after {max_retries} attempts. Last error: {last_err}")


def render_steps_as_text(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", [])
    lines = []
    for st in steps:
        sid = st.get("step_id")
        goal = st.get("goal", "")
        tables = st.get("tables_or_entities", [])
        filters = st.get("filters_or_conditions", [])
        out = st.get("intermediate_output", "")
        deps = st.get("depends_on_step_ids", [])

        lines.append(f"Step {sid}: {goal}")
        if tables:
            lines.append(f"  - Tables/Entities: {', '.join(map(str, tables))}")
        if filters:
            lines.append("  - Filters/Conditions:")
            for f in filters:
                lines.append(f"    * {f}")
        if deps:
            lines.append(f"  - Depends on: {deps}")
        if out:
            lines.append(f"  - Output: {out}")
        lines.append("")
    final_deriv = parsed.get("final_answer_derivation", "")
    if final_deriv:
        lines.append(f"Final answer derivation: {final_deriv}")
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="Example index in the split")
    ap.add_argument("--split", type=str, default="train", help="Dataset split: train/dev/test")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    ap.add_argument(
        "--spider2_root",
        type=str,
        default=os.environ.get("SPIDER2_ROOT", ""),
        help="Path to local Spider2 repository root (e.g., /Users/.../xlang-spider2). You can also set env SPIDER2_ROOT.",
    )
    ap.add_argument("--max_doc_chars", type=int, default=2000, help="How many chars of external doc to include")
    ap.add_argument("--print_prompt_head", type=int, default=2000, help="Print first N chars of prompt for debug")
    ap.add_argument("--save_json", type=str, default="", help="Optional path to save parsed JSON output")
    args = ap.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("No API key found. Please set OPENAI_API_KEY in your environment.")

    # Check spider2 root
    if not args.spider2_root:
        raise ValueError(
            "Missing --spider2_root and env SPIDER2_ROOT is not set.\n"
            "Example: --spider2_root /Users/yangsongzhou/Year3/xlang-spider2"
        )
    spider2_root = Path(args.spider2_root).expanduser().resolve()
    if not spider2_root.exists():
        raise FileNotFoundError(f"spider2_root does not exist: {spider2_root}")

    # Load dataset and example
    split_ds = load_spider2_lite(args.split)
    total = len(split_ds)
    print(f"[INFO] Using split = {args.split}")
    print(f"[INFO] Total examples in {args.split} split: {total}")

    if args.idx < 0 or args.idx >= total:
        raise IndexError(f"--idx out of range: {args.idx} (0..{total-1})")

    ex = split_ds[args.idx]

    instance_id = ex.get("instance_id", "")
    db = ex.get("db", "")
    question = ex.get("question", "")
    doc_name = ex.get("external_knowledge", "")
    temporal = ex.get("temporal", None)

    print(f"[INFO] Using example idx = {args.idx}")
    print(f"Instance ID : {instance_id}")
    print(f"DB          : {db}")
    print(f"Question    : {question}")
    print(f"Doc name    : {doc_name}")
    print(f"temporal    : {temporal}")
    print("")

    # Find and read doc
    print(f"[INFO] Searching for doc '{doc_name}' under {spider2_root} ...")
    doc_path = find_doc_file(spider2_root, doc_name)
    print(f"[INFO] Found doc file: {doc_path}")
    doc_head = read_doc_head(doc_path, args.max_doc_chars)

    # Prompt
    prompt = build_prompt(question=question, doc_snippet=doc_head)
    print("\n===== PROMPT (for debug, truncated) =====")
    print(prompt[: args.print_prompt_head])
    print("===== END OF PROMPT =====\n")

    # Call OpenAI
    client = OpenAI()
    print("[INFO] Calling OpenAI API for multi-hop decomposition (JSON) ...")
    raw = call_openai_json(client=client, model=args.model, prompt=prompt)
    print("\n===== RAW MODEL OUTPUT =====")
    print(raw)
    print("===== END RAW OUTPUT =====\n")

    parsed, err = safe_json_load(raw)
    if parsed is None:
        eprint(f"[ERROR] Failed to parse JSON: {err}")
        eprint("[HINT] The model did not return strict JSON. Try re-run or lower temperature.")
        sys.exit(1)

    # Pretty print JSON
    print("===== PARSED JSON (pretty) =====")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("===== END PARSED JSON =====\n")

    # Render as steps
    print("===== MULTI-HOP STEPS (rendered) =====")
    print(render_steps_as_text(parsed))
    print("===== END OF STEPS =====\n")

    # Save JSON if requested
    if args.save_json:
        out_path = Path(args.save_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()
