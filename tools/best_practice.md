# Best Practice for Deployment
- Environment
  - Python 3.11 + CUDA 12.1
  - setuptool == 76.0.0
- Steps
```bash
# 1. Install vllm 0.2.2
cd vllm
pip install -r requirements.txt
pip install -e .

# 2. Install pg database
# pls follow pg_install.sh in ./tools

# 3. Install Ayo (move to base dir)
pip install -r requirements.txt
pip install -e .

# 4. Install Models
cd tools
python download_hf_model.py

# 5. Before Running Inference
# Modify the model configs in optimized_embedding_ingestion_searching_reranking_llm.py
# vector_dim in vector_db should be 384

# 6. Run Inference
python optimized_embedding_ingestion_searching_reranking_llm.py
```


