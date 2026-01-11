# utils/agents/order2.py

class AgentResult:
    def __init__(self, flag_valid: bool, updates: dict, next_agent=None):
        self.flag_valid = flag_valid
        self.updates = updates
        self.next_agent = next_agent


def ORDERBY_clause2(ctx):
    table_name = getattr(ctx, "title", "main_table")
    default_sql = f"SELECT * FROM {table_name} LIMIT 10;"
    sql = getattr(ctx, "previous_sql_query", default_sql)

    updates = {
        "sql": sql,
        "flag": False,
    }
    next_agent = None

    return AgentResult(True, updates, next_agent)
