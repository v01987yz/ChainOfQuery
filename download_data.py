from huggingface_hub import snapshot_download

# 这会把整个 database 文件夹下下来，可能会有点大
# 如果你只想下 ga4，可以手动去网页下
print("Downloading Spider 2 databases...")
snapshot_download(
    repo_id="xlangai/spider2-lite",
    repo_type="dataset",
    local_dir="/Users/yangsongzhou/Year3/xlang-spider2/spider2-lite", # 直接下载到你的目录
    allow_patterns=["resource/databases/*"] 
)
print("Download finished!")