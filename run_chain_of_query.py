import sys
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm

# 这三行是新加的：为了保存 CSV 和调试
import pandas as pd
import time
import traceback

from utils.load_data import *
from utils.myllm import MyChatGPT
from utils.database import MYSQLDB
from utils.helper import PipelineContext, AgentResult
from utils.pipeline import agent_pipeline
from utils.general_prompt import *
from utils.reasoner import chainofthought_answer_agent


def evaluator(generated_answer: str, answer: str) -> bool:
    if generated_answer.lower() == answer.lower():
        return True
    else:
        return False


global_dataset = None
global_data_process_func = None


def init_dataset():
    global global_dataset, global_data_process_func
    global_dataset, global_data_process_func = load_hg_dataset("wikitq")


def process_one_example_with_cfg(args):
    i, model_name, api_key = args
    return process_one_example(i, model_name, api_key)


def process_one_example(i, model_name, api_key):
    global global_dataset, global_data_process_func

    try:
        data = global_dataset["test"][i]
        data_dict = global_data_process_func(data)
        question = data_dict["question"]
        standard_answer = ", ".join(data_dict["answer"])
        tables = data_dict["tables"]
        sqldb = MYSQLDB(tables=tables)

        new_tables = sqldb.get_table_df()
        new_title = sqldb.get_table_title()
        table_dict = sqldb.get_table()

        # -------- 这里把 log 设计得更详细 --------
        log = {
            "idx": i,
            "question": question,
            "table_title": new_title,
            "sqls": [],          # agent_pipeline 里的 SQL 轨迹
            "final_sql": None,   # 最后一次 SQL（如果有）
            "s_answer": "",      # standard answer
            "p_answer": "",      # predicted answer
            "valid": None,       # 是否生成了可执行 SQL
            "correct": None,     # 最终是否答对
            "valid_flag": False,
            "answer_flag": False,
            "use_cot_fallback": False,  # 是否启用了 CoT fallback
        }

        num_example_rows = 3
        prompt_table = create_table_prompt(df=new_tables, title=new_title)
        total_rows, prompt_rows = select_x_rows_prompt(
            full_table=False,
            df=new_tables,
            title=new_title,
            num_rows=num_example_rows,
        )
        prompt_schema = prompt_table + prompt_rows

        my_llm = MyChatGPT(
            model_name=model_name,
            key=api_key
        )

        ctx = PipelineContext(
            llm=my_llm,
            sqldb=sqldb,
            question=question,
            prompt_schema=prompt_schema,
            title=new_title,
            previous_sql_query=None,
            total_rows=total_rows,
            log=log,
            flag=None,
            num_rows=num_example_rows,
            llm_options=None,
            debug=False,
            strategy="top",
            extras={}
        )

        # ---------------- CoQ 主流程 ----------------
        valid_flag, answer_flag, sql_query, generated_answer, log = agent_pipeline(
            ctx, standard_answer
        )

        log["valid_flag"] = bool(valid_flag)
        log["answer_flag"] = bool(answer_flag)
        log["final_sql"] = sql_query

        if valid_flag:
            log["valid"] = True
            predicted_answer = generated_answer
        else:
            log["valid"] = False
            if not answer_flag:
                # SQL 不 valid 且 CoQ 自己没答出，启用 CoT baseline
                log["use_cot_fallback"] = True
                analysis, predicted_answer = chainofthought_answer_agent(
                    llm=my_llm, question=question, table_dict=table_dict, debug=False
                )
            else:
                predicted_answer = generated_answer

        log["p_answer"] = predicted_answer
        log["s_answer"] = standard_answer

        # 评估是否答对（优先用 CoQ 的 answer_flag，其次用字符串比对）
        if answer_flag:
            log["correct"] = True
        else:
            log["correct"] = evaluator(predicted_answer, standard_answer)

        sqldb.close()
        del sqldb

        # 直接把 log 返回，后面 main() 会汇总并写 CSV
        return log

    except Exception as e:
        # 如果某个 sample 崩了，在这里记录一个简单的 log，方便不影响整体
        print(f"[Error] Sample {i} failed: {e}", flush=True)
        traceback.print_exc()
        time.sleep(1)
        return {
            "idx": i,
            "question": "",
            "table_title": "",
            "sqls": [],
            "final_sql": None,
            "s_answer": "",
            "p_answer": "",
            "valid": False,
            "correct": False,
            "valid_flag": False,
            "answer_flag": False,
            "use_cot_fallback": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Chain-of-Query experiments.")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--num_samples", type=int, help="Number of test samples")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Please set OPENAI_API_KEY in your environment.")

    indices = list(range(args.num_samples))
    cfg_iter = ((i, args.model, api_key) for i in indices)
    results = []

    with Pool(processes=8, initializer=init_dataset) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_example_with_cfg, cfg_iter),
            total=len(indices),
        ):
            results.append(res)

    # -------- 统计整体指标，逻辑跟原来类似 --------
    correct_count = sum(1 for r in results if r.get("correct"))
    invalid_sql = sum(1 for r in results if not r.get("valid"))

    print("Chain-of-Query:", correct_count, flush=True)
    print("Invalid:", invalid_sql, flush=True)

    # -------- 新增：把详细 log 写到 CSV --------
    os.makedirs("tmp", exist_ok=True)
    out_path = os.path.join("tmp", f"coq_{args.model}_{args.num_samples}.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"[LOG] Detailed results saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
