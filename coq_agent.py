import os
import json
import sqlite3
from typing import List, Dict, Any
from openai import OpenAI

class SequentialCoQAgent:
    def __init__(self, db_path: str, api_key: str, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.db_path = db_path
        self.model = model

    def _get_schema_info(self) -> str:
        """获取数据库 Schema 信息"""
        try:
            # 使用 URI 模式以只读方式打开，防止误修改
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_str = ""
            for table_name in tables:
                table = table_name[0]
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                # column[1] is name, column[2] is type
                col_strs = [f"{col[1]} ({col[2]})" for col in columns]
                schema_str += f"Table: {table}\nColumns: {', '.join(col_strs)}\n\n"
            
            conn.close()
            return schema_str
        except Exception as e:
            return f"Error reading schema: {str(e)}"

    def _execute_sql(self, sql: str) -> Any:
        """执行 SQL 并返回结果"""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def plan_decomposition(self, question: str, schema: str) -> List[Dict]:
        """Step 1: 规划拆解"""
        prompt = f"""
        You are a Data Analyst Agent. Decompose the question into sequential steps.
        
        Database Schema:
        {schema}

        User Question: "{question}"

        Instructions:
        1. Break down into logical steps (2-4 steps).
        2. Steps must be sequential. Step 2 can use results from Step 1.
        3. Output JSON ONLY.
        
        Format:
        [
            {{"step_id": 1, "description": "Find the ID...", "dependency": "None"}},
            {{"step_id": 2, "description": "Use ID from Step 1 to find...", "dependency": 1}}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            # 清洗 Markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            print(f"Planning Error: {e}")
            return []

    def generate_step_sql(self, step: Dict, context: Dict, schema: str) -> str:
        """Step 2: 生成单步 SQL"""
        context_str = ""
        dep_id = step.get('dependency')
        # 兼容 int 或 "None"
        if dep_id and str(dep_id).lower() != "none":
            prev_res = context.get(int(dep_id))
            context_str = f"NOTE: The result from Step {dep_id} is: {prev_res}. Use this exact value in WHERE clause."

        prompt = f"""
        Write a single SQLite query.
        
        Schema:
        {schema}
        
        Task: {step.get('description')}
        {context_str}

        Requirements:
        1. Output ONLY SQL. No markdown.
        2. Do not use CTEs or complex JOINs if you have the specific value in NOTE.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        sql = response.choices[0].message.content.strip()
        return sql.replace("```sql", "").replace("```", "").strip()

    def run(self, question: str):
        """主执行逻辑"""
        schema = self._get_schema_info()
        plan = self.plan_decomposition(question, schema)
        
        if not plan:
            return {"error": "Failed to plan", "context": {}}

        context = {}
        history = [] # 用于记录每一步详情，方便评测分析

        for step in plan:
            sql = self.generate_step_sql(step, context, schema)
            result = self._execute_sql(sql)
            
            # 记录历史
            history.append({
                "step_id": step['step_id'],
                "description": step['description'],
                "sql": sql,
                "result": str(result)
            })

            # 错误阻断
            if isinstance(result, str) and "Error" in result:
                context["error"] = result
                break
            if not result:
                # 结果为空，可能也是一种正常状态，但在多跳中通常意味着断链
                context[step['step_id']] = [] 
            else:
                context[step['step_id']] = result
                
        return {"plan": plan, "history": history, "final_context": context}