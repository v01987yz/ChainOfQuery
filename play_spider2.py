# 简单浏览 Spider 2.0-Lite 数据集，并尝试在本地 Spider2 仓库中
# 找到对应的 external_knowledge Markdown 文档并打印前几行。

from datasets import load_dataset
from pathlib import Path

SPIDER2_ROOT = Path("/Users/yangsongzhou/Year3/xlang-spider2")

def find_doc_file(filename: str) -> Path | None:
    """
    在 SPIDER2_ROOT 目录下递归搜索给定文件名的路径。
    找到第一个匹配的就返回，没有找到返回 None。
    """
    if not SPIDER2_ROOT.exists():
        print(f"[WARN] Spider2 root not found: {SPIDER2_ROOT}")
        return None

    for p in SPIDER2_ROOT.rglob(filename):
        if p.is_file():
            return p

    return None


def main():
    # 1. 加载 HuggingFace 上的 spider2-lite 数据集
    print("[INFO] Loading dataset xlangai/spider2-lite ...")
    ds = load_dataset("xlangai/spider2-lite")
    train = ds["train"]

    print(f"[INFO] Total examples in train split: {len(train)}")

    # 2. 先看第 0 条样本
    ex = train[0]

    print("\n=== Example 0 (raw fields) ===")
    for k, v in ex.items():
        print(f"{k}: {v}")

    instance_id = ex["instance_id"]
    db_name = ex["db"]
    question = ex["question"]
    doc_name = ex["external_knowledge"]

    print("\n=== Parsed fields ===")
    print(f"Instance ID : {instance_id}")
    print(f"DB          : {db_name}")
    print(f"Question    : {question}")
    print(f"Doc name    : {doc_name}")

    # 3. 在本地 Spider2 仓库中查找 external_knowledge 对应的文档
    print(f"\n[INFO] Searching for doc '{doc_name}' under {SPIDER2_ROOT} ...")
    doc_path = find_doc_file(doc_name)

    if doc_path is None:
        print(f"[WARN] Could not find doc file '{doc_name}' under {SPIDER2_ROOT}")
        print("       请确认已经 clone 了 Spider2 仓库，并检查 SPIDER2_ROOT 路径是否正确。")
        return

    print(f"[INFO] Found doc file: {doc_path}")

    # 4. 读取并展示前 1000 字符
    try:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ERROR] Failed to read doc file: {e}")
        return

    print("\n===== DOC HEAD (first 1000 chars) =====")
    print(text[:1000])
    print("===== END OF DOC HEAD =====")


if __name__ == "__main__":
    main()
