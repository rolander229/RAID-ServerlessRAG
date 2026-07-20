import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

# snapshot_download(repo_id="openai-community/gpt2", ignore_patterns=["*.onnx"])
# snapshot_download(repo_id="Qwen/Qwen2.5-0.5B", ignore_patterns=["*.onnx", "pytorch_model.bin"])
snapshot_download(repo_id="BAAI/bge-reranker-base", ignore_patterns=["*.onnx", "pytorch_model.bin"])
snapshot_download(repo_id="BAAI/bge-small-en-v1.5", ignore_patterns=["*.onnx", "pytorch_model.bin"])
