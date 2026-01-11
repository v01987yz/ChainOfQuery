# utils/agents/column_selector.py

"""
Simplified column selector agents for Chain-of-Query.

The official repo expects BASIC_clause and SELECT_clause
to return an object with:
  - flag_valid: bool
  - updates: dict
  - next_agent: str | None

Here we implement a minimal stub so that the pipeline
can run end-to-end, even though the SQL may be very simple.
"""

class AgentResult:
    def __init__(self, flag_valid: bool, updates: dict, next_agent=None):
        self.flag_valid = flag_valid
        self.updates = updates
        self.next_agent = next_agent


def BASIC_clause(ctx):
    """
    'Basic' agent: produce a very simple initial SQL query (sql1),
    and optionally a second candidate (sql2).

    For now we just do: SELECT * FROM {table} LIMIT 10;
    """
    # 有的 ctx 里 table 名叫 title，有的叫 prompt_schema，这里按 utils/pipeline 的用法，
    # ctx.title 就是当前要查的表名
    table_name = getattr(ctx, "title", "main_table")

    sql1 = f"SELECT * FROM {table_name} LIMIT 10;"
    sql2 = ""  # 先不提供第二个候选

    updates = {
        "sql1": sql1,
        "sql2": sql2,
    }

    # 简化：不再进入后续的 "Select" agent，直接让 pipeline
    # 走 sufficiency -> answer 这一条路
    next_agent = None

    return AgentResult(
        flag_valid=True,
        updates=updates,
        next_agent=next_agent,
    )


def SELECT_clause(ctx):
    """
    'Select' agent: refine the SELECT clause.

    在我们的简化版里，不做任何修改，直接复用 ctx.previous_sql_query。
    但为了和 pipeline 对上接口，还是要提供:
      - updates['sql']
      - updates['flag']
      - next_agent
    """
    # 如果之前的 SQL 存在就用之前的，否则重新造一个最简单的
    table_name = getattr(ctx, "title", "main_table")
    default_sql = f"SELECT * FROM {table_name} LIMIT 10;"
    sql = getattr(ctx, "previous_sql_query", default_sql)

    updates = {
        "sql": sql,
        # flag 为 False 的时候，pipeline 里会：
        #   ctx.previous_sql_query = sql_query
        #   continue
        # 相当于“先不强制触发 sufficiency 检查”，这里我们就设成 False 即可。
        "flag": False,
    }

    next_agent = None  # 不再进入后面的 Agg / Order agents

    return AgentResult(
        flag_valid=True,
        updates=updates,
        next_agent=next_agent,
    )
