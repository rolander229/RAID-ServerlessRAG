#!/usr/bin/env python3
"""Downloads Qwen/Qwen3-1.7B weights into shared model directory."""

import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SHARED_DIR = "/storage/xuedh_work2/models/qwen3-1.7b"
LOCAL_CACHE = "/storage/xuedh_workspace/.cache/huggingface"

# 优先下载/保存到公共共享目录，若无权限则回退到本地缓存
target_local_dir = SHARED_DIR if os.access("/storage/xuedh_work2/models", os.W_OK) else None
if not target_local_dir:
    os.environ["HF_HOME"] = LOCAL_CACHE

from huggingface_hub import snapshot_download

print("Starting download of Qwen/Qwen3-1.7B (total size: ~3.8 GB)...")
print(f"Target location: {target_local_dir if target_local_dir else LOCAL_CACHE}")
start_time = time.time()

try:
    if target_local_dir:
        os.makedirs(target_local_dir, exist_ok=True)
        path = snapshot_download(
            repo_id="Qwen/Qwen3-1.7B",
            local_dir=target_local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        os.system(f"chmod -R a+rX {target_local_dir}")
    else:
        path = snapshot_download(
            repo_id="Qwen/Qwen3-1.7B",
            resume_download=True,
            max_workers=4
        )
    duration = time.time() - start_time
    print(f"\nDownload completed successfully in {duration:.1f}s!")
    print(f"Model path: {path}")
except Exception as e:
    print(f"\nDownload failed with error: {e}", file=sys.stderr)
    sys.exit(1)
