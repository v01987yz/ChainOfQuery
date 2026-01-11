# utils/agents/withas.py

"""
Simplified WITH-AS agent stub.

The pipeline expects a function WITHAS_clause(ctx) that returns an
object with:
  - flag_valid: bool
  - updates: dict (must contain "sql" and "flag")
  - next_agent: str | None

Here we implement a no-op version that simply forwards the
previous SQL query.
"""

class AgentResult:
    def __init__(self, flag_valid: bool, updates: dict, next_agent=None):
        self.flag_valid = flag_valid
        self.updates = updates
        self.next_agent = next_agent


def WITHAS_clause(ctx):
    # 如果之前有 SQL，就用之前的；否则随便造一个最简单的
    table_name = getattr(ctx, "title", "main_table")
    default_sql = f"SELECT * FROM {table_name} LIMIT 10;"
    sql = getattr(ctx, "previous_sql_query", default_sql)

    updates = {
        "sql": sql,
        # 设成 False：pipeline 会把 sql 记进 ctx.previous_sql_query，然后继续
        "flag": False,
    }

    # 这里不再引出其他 Agent
    next_agent = None

    return AgentResult(
        flag_valid=True,
        updates=updates,
        next_agent=next_agent,
    )
