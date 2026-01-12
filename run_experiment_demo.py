import os
import json
import time
from coq_agent import SequentialCoQAgent

# 指向我们之前生成的那个测试库 (Company/Acquisition)
DB_PATH = "test_fy_project.db" 
OUTPUT_FILE = f"experiment_results_demo_{int(time.time())}.json"
API_KEY = os.getenv("OPENAI_API_KEY")

# 手动构造几个符合 test_fy_project.db 逻辑的测试题
# 这些题目能完美展示多跳拆解能力
TEST_DATA = [
    {
        "instance_id": "demo_001",
        "question": "What is the revenue of the company that acquired Youtube?",
        "gold_sql": "SELECT T1.revenue..."
    },
    {
        "instance_id": "demo_002",
        "question": "Which company acquired GitHub and what is its revenue?",
        "gold_sql": "SELECT T1.name, T1.revenue..."
    },
    {
        "instance_id": "demo_003",
        "question": "Find the name of the company that acquired a target in the year 2006.",
        "gold_sql": "SELECT T1.name..."
    }
]

def main():
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY not set.")
        return

    # 检查数据库是否存在
    if not os.path.exists(DB_PATH):
        # 如果没有，自动创建一个简单的以便演示
        print(f"⚠️ {DB_PATH} not found. Running generation script...")
        os.system("python coq_agent.py") 
        time.sleep(1)

    print(f"🚀 Starting DEMO experiment on {len(TEST_DATA)} samples...\n")
    results = []

    # 初始化 Agent
    agent = SequentialCoQAgent(db_path=DB_PATH, api_key=API_KEY)

    for i, item in enumerate(TEST_DATA):
        print(f"[{i+1}/{len(TEST_DATA)}] Processing {item['instance_id']}...")
        question = item['question']
        
        try:
            start_time = time.time()
            # --- 核心运行逻辑 ---
            output = agent.run(question)
            duration = time.time() - start_time
            
            # 记录结果
            result_entry = {
                "instance_id": item['instance_id'],
                "question": question,
                "status": "success",
                "duration_sec": round(duration, 2),
                "steps_plan": output.get('plan'),
                "execution_trace": output.get('history'), # 这就是 CoT 的痕迹
                "final_answer_context": str(output.get('final_context'))
            }
            results.append(result_entry)
            print(f"   ✅ Success! (Time: {duration:.2f}s)")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n🎉 Demo finished. Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()