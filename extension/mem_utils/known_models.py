KNOWN_MODELS_CONFIG = {
    "BAAI/bge-small-en-v1.5": {
        "param_count": 0.033,  # 33M params
        "embedding_dim": 384,
        "model_size": 0.5,
    },
    "BAAI/bge-base-en-v1.5": {
        "param_count": 0.109,  # 109M params
        "embedding_dim": 768,
        "model_size": 1.0,
    },
    "BAAI/bge-large-en-v1.5": {
        "param_count": 0.335,  # 335M params
        "embedding_dim": 1024,
        "model_size": 2.0,
    },
    "BAAI/bge-reranker-base": {
        "param_count": 0.278,  # 278M params
        "model_size": 1.5,
    },
    "BAAI/bge-reranker-large": {
        "param_count": 0.560,  # 560M params
        "model_size": 3.0,
    },
    "openai-community/gpt2": {
        "param_count": 0.124,  # 124M params
        "context_length": 1024,
        "model_size": 0.5,
    },
    "gpt2-medium": {
        "param_count": 0.355,  # 355M params
        "context_length": 1024,
        "model_size": 1.5,
    },
    "gpt2-large": {
        "param_count": 0.774,  # 774M params
        "context_length": 1024,
        "model_size": 3.0,
    },
    "gpt2-xl": {
        "param_count": 1.5,
        "context_length": 1024,
        "model_size": 6.0,
    },
    "meta-llama/Llama-2-7b": {
        "param_count": 7.0,
        "context_length": 4096,
        "model_size": 14.0,
    },
    "meta-llama/Llama-2-13b": {
        "param_count": 13.0,
        "context_length": 4096,
        "model_size": 26.0,
    },
}