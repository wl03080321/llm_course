from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path as path
import os

base_dir = path(__file__).parent.parent


def download_model(repo_id, filename=None, cache_dir=None):
    """
    下載 Hugging Face Hub 上的模型或檔案，並回傳本地路徑與預設儲存位置。

    :param repo_id: 模型的 repo id，例如 'bert-base-uncased'
    :param filename: 指定下載的檔案名稱（如 config.json），若為 None 則下載整個模型快照
    :param cache_dir: 指定快取資料夾，預設為 Hugging Face 預設路徑
    :return: (模型檔案路徑)
    """
    if filename:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        )
    else:
        local_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    return local_path


# 範例用法
if __name__ == "__main__":
    path = download_model(
        "sentence-transformers/all-MiniLM-L6-v2", cache_dir=base_dir / "cache"
    )
    print("模型下載路徑:", path)
