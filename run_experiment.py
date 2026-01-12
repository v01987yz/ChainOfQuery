import json
import os
from coq_agent import SequentialCoQAgent

# ================= 配置 =================
REAL_DB_PATH = "database/spider.sqlite"  # 指向你的真实数据库
DATASET_PATH = "data/dev.json"           # 指向你的数据集
OUTPUT_FILE = "experiment_results.json"
MAX_QUESTIONS = 5  # 先测 5 个，省钱！跑通了再全量跑
# =======================================

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # 1. 初始化 Agent，连接真实数据库
    # 确保 API_KEY 已经在 coq_agent.py 里设置好，或者在这里通过环境变量传入
    agent = SequentialCoQAgent(db_path=REAL_DB_PATH, api_key=os.getenv("OPENAI_API_KEY"))
    
    # 2. 加载真实数据
    data = load_dataset(DATASET_PATH)
    print(f"📚 Loaded {len(data)} questions from dataset.")
    
    results = []
    
    # 3. 批量运行
    for i, item in enumerate(data[:MAX_QUESTIONS]):
        question = item['question']
        gold_query = item['query'] # 标准答案 SQL，用于后续对比
        db_id = item.get('db_id')  # Spider 数据集通常有 db_id，如果是单库可忽略
        
        # 如果是多库数据集(Spider)，需要动态切换 Agent 的数据库路径
        if db_id:
             agent.db_path = f"database/{db_id}/{db_id}.sqlite"
        
        print(f"\n[{i+1}/{MAX_QUESTIONS}] Processing: {question}")
        
        try:
            # 运行你的多跳逻辑
            final_context = agent.run(question)
            
            # 记录结果
            results.append({
                "question_id": i,
                "question": question,
                "gold_sql": gold_query,
                "agent_context": str(final_context), # 记录最终结果
                "status": "success"
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "question_id": i,
                "error": str(e),
                "status": "failed"
            })

    # 4. 保存实验结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n🎉 Experiment finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()