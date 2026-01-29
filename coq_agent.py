import os
import json
import re
import sqlite3
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    t = re.sub(r"```sql\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```[\s]*", "", t).strip()
    return t


def _lstrip_leading_backslash(text: str) -> str:
    if text is None:
        return ""
    return text.lstrip("\\").strip()


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Try to parse a JSON array from model output even if it added extra text.
    """
    t = (text or "").strip()
    t = _strip_code_fences(t)

    try:
        return json.loads(t)
    except Exception:
        start = t.find("[")
        end = t.rfind("]")
        if start >= 0 and end > start:
            return json.loads(t[start : end + 1])
        raise


class SequentialCoQAgent:
    """
    Sequential CoQ Agent supporting:
      - SQLite backend (local db_path)
      - BigQuery backend (bq_project/bq_location/bq_db + optional dataset override)

    Pipeline:
      1) plan_decomposition(question, schema_text) -> list[steps]
      2) compile_final_sql(question, steps, extra_context) -> ONE SQL
      3) _execute_sql(sql) -> result
    """

    def __init__(
        self,
        backend: str = "sqlite",
        db_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        # BigQuery params:
        bq_project: Optional[str] = None,
        bq_location: str = "US",
        bq_db: Optional[str] = None,  # e.g., "ga4"
        bq_dataset_override: Optional[str] = None,  # e.g., "bigquery-public-data.ga4_obfuscated_sample_ecommerce"
    ):
        self.backend = backend.lower().strip()
        self.db_path = db_path
        self.model = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key=...")

        self.client = OpenAI(api_key=api_key)

        # BigQuery fields
        self.bq_project = bq_project
        self.bq_location = bq_location
        self.bq_db = bq_db
        self.bq_dataset_override = bq_dataset_override or self._default_dataset_override(bq_db)

        if self.backend not in {"sqlite", "bigquery"}:
            raise ValueError(f"Unsupported backend={backend}. Use 'sqlite' or 'bigquery'.")

        if self.backend == "sqlite":
            if not self.db_path:
                raise ValueError("SQLite backend requires db_path.")
        else:
            if not self.bq_project:
                raise ValueError("BigQuery backend requires bq_project.")

    def _default_dataset_override(self, bq_db: Optional[str]) -> Optional[str]:
        """
        Safe default for ga4 only (since you already validated it works).
        """
        if not bq_db:
            return None
        key = str(bq_db).lower().strip()
        if key == "ga4":
            return "bigquery-public-data.ga4_obfuscated_sample_ecommerce"
        return None

    # ---------------------------
    # Schema
    # ---------------------------
    def _get_schema_info(self) -> str:
        if self.backend == "sqlite":
            return self._get_schema_info_sqlite()

        # BigQuery: rely on docs snippet + guidance
        msg = [
            "Backend: BigQuery",
            f"Project: {self.bq_project}",
            f"Location: {self.bq_location}",
            f"DB (benchmark id): {self.bq_db}",
            "",
            "IMPORTANT RULES:",
            "- Use BigQuery Standard SQL.",
            "- Use backticks for table names.",
            "- For wildcard tables like `...events_*`, scope by `_TABLE_SUFFIX`.",
            "- Do NOT invent years/dates; follow the question dates exactly.",
            "- Avoid placeholders like <property_id> unless you also provide a dataset override.",
            f"- dataset override (if needed): {self.bq_dataset_override}",
        ]
        return "\n".join(msg)

    def _get_schema_info_sqlite(self) -> str:
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema_str = ""
            for table_name in tables:
                table = table_name[0]
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                col_strs = [f"{col[1]} ({col[2]})" for col in columns]
                schema_str += f"Table: {table}\nColumns: {', '.join(col_strs)}\n\n"

            conn.close()
            return schema_str
        except Exception as e:
            return f"Error reading schema: {str(e)}"

    # ---------------------------
    # Planning
    # ---------------------------
    def plan_decomposition(self, question: str, schema: str) -> List[Dict[str, Any]]:
        """
        Output:
        [
          {"step_id": 1, "description": "...", "dependency": "None"},
          {"step_id": 2, "description": "...", "dependency": 1},
          ...
        ]
        """
        prompt = f"""
You are a Data Analyst Agent. Decompose the question into sequential steps.

Context / Schema / Docs:
{schema}

User Question: "{question}"

Instructions:
1) Break down into logical steps (2-6 steps).
2) Steps should describe queryable sub-goals, but DO NOT write SQL here.
3) Each step can depend on exactly ONE previous step via "dependency" (int) or "None".
4) Do NOT create a "date range" step unless it is essential; prefer embedding date filters directly in SQL later.
5) Output JSON ONLY (no markdown, no explanations).

Format:
[
  {{"step_id": 1, "description": "...", "dependency": "None"}},
  {{"step_id": 2, "description": "...", "dependency": 1}}
]
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            return _extract_json_array(content)
        except Exception as e:
            print(f"Planning Error: {e}")
            return []

    # ---------------------------
    # Compile ONE final SQL (new)
    # ---------------------------
    def compile_final_sql(self, question: str, steps: List[Dict[str, Any]], extra_context: str = "") -> str:
        """
        Compile the plan into ONE executable SQL query.
        This is the "Compile SQL" step you want in the UI.
        """
        schema = self._get_schema_info()
        if extra_context:
            schema = schema + "\n\n" + extra_context

        if self.backend == "sqlite":
            dialect = "SQLite"
            dataset_hint = ""
        else:
            dialect = "BigQuery Standard SQL"
            dataset_hint = ""
            if self.bq_dataset_override:
                dataset_hint = (
                    f"\nDATASET OVERRIDE (use this if the plan uses analytics_<property_id>): "
                    f"`{self.bq_dataset_override}.events_*`\n"
                )

        prompt = f"""
You are an expert text-to-SQL compiler.

Dialect: {dialect}

Context / Schema / Docs:
{schema}

Question:
{question}

Plan (steps):
{json.dumps(steps, ensure_ascii=False, indent=2)}

{dataset_hint}

Requirements (STRICT):
1) Output ONLY ONE final SQL query (no markdown, no explanations).
2) Use CTEs if helpful, but must be runnable directly.
3) If using wildcard tables like `...events_*`, ALWAYS scope time using `_TABLE_SUFFIX`.
4) Follow the question dates exactly. Do NOT invent years (e.g., do not jump to 2023 if question is 2021).
5) Avoid placeholders like <property_id>. If you must, use analytics_<property_id> and we will replace it with dataset override at execution time.
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        sql = (resp.choices[0].message.content or "").strip()
        sql = _strip_code_fences(sql)
        sql = _lstrip_leading_backslash(sql)
        return sql.strip()

    # ---------------------------
    # Execute SQL
    # ---------------------------
    def _execute_sql(self, sql: str) -> Any:
        if self.backend == "sqlite":
            return self._execute_sql_sqlite(sql)
        return self._execute_sql_bigquery(sql)

    def _execute_sql_sqlite(self, sql: str) -> Any:
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def _sanitize_bigquery_sql(self, sql: str) -> str:
        s = _strip_code_fences(sql)
        s = _lstrip_leading_backslash(s)
        s = s.strip().rstrip(";").strip()

        if "analytics_<property_id>" in s:
            if not self.bq_dataset_override:
                raise ValueError(
                    "SQL contains analytics_<property_id> but no bq_dataset_override is set."
                )
            s = s.replace("analytics_<property_id>", self.bq_dataset_override)

        return s

    def _execute_sql_bigquery(self, sql: str) -> Any:
        try:
            from google.cloud import bigquery  # type: ignore
        except Exception as e:
            return f"SQL Error: BigQuery client not available. Install google-cloud-bigquery. Details: {e}"

        try:
            s = self._sanitize_bigquery_sql(sql)
            client = bigquery.Client(project=self.bq_project)
            job = client.query(s, location=self.bq_location)
            rows = list(job.result())
            return [dict(r) for r in rows]
        except Exception as e:
            return f"SQL Error: {str(e)}"
