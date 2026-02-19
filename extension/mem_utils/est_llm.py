from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from extension.config_planner import ModelProfile

MEMORY_PER_BILLION_PARAMS = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

def estimate_hf_model_size(
    model_profile: ModelProfile,
    precision: str = "fp16"
) -> int:
    config = AutoConfig.from_pretrained(model_profile.hf_model_path)

    # 2. Initialize the model on the 'meta' device
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # 3. Calculate total bytes
    # p.numel() is the number of parameters
    # do not use p.element_size() because it may not reflect the actual memory usage due to quantization and other optimizations
    # use precision-based estimation instead
    if precision is not None:
        try: 
            precision_bytes = MEMORY_PER_BILLION_PARAMS.get(precision, 4.0)
            total_size_bytes = sum(p.numel() for p in model.parameters()) * precision_bytes
        except KeyError:
            total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    else:
        total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return int(total_size_bytes)

def get_num_parameters(config):
    """Estimate total parameters from config"""
    hidden = config.hidden_size
    vocab = config.vocab_size
    layers = config.num_hidden_layers
    
    # Embeddings
    params = vocab * hidden
    
    # Each transformer layer
    if hasattr(config, 'intermediate_size'):
        intermediate = config.intermediate_size
    else:
        intermediate = hidden * 4
    
    per_layer = (
        hidden * hidden * 4 +  # Q, K, V, O projections
        hidden * intermediate * 2 +  # FFN up and down
        hidden * 4  # Layer norms (approximate)
    )
    
    params += layers * per_layer
    params += hidden  # Final layer norm
    
    return params

def estimate_llm_inference_memory(model_profile: ModelProfile, 
                              batch_size=1, 
                              seq_length=512,
                              dtype="fp16"):
    try:
        weights_memory = estimate_hf_model_size(model_profile, precision=dtype)
    except Exception as e:
        config = AutoConfig.from_pretrained(model_profile.hf_model_path)

        # Bytes per parameter based on dtype
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        param_bytes = bytes_per_param[dtype]
    
        # 1. Model weights
        num_params = get_num_parameters(config)
        weights_memory = num_params * param_bytes
    
    # 2. KV cache estimation
    kv_cache = (2 * config.num_hidden_layers * batch_size * seq_length * 
                config.hidden_size * param_bytes)
    
    # 3. Activations (rough estimate)
    # Main activation: batch × seq_len × hidden_size
    activations = batch_size * seq_length * config.hidden_size * param_bytes * 4
    
    # 4. CUDA context overhead
    cuda_overhead = 1 * (1024**3)  # ~1GB
    
    total_bytes = weights_memory + kv_cache + activations + cuda_overhead
    total_gb = total_bytes / (1024**3)
    
    return total_gb


def estimate_llm_infer_memory(
    model_profile: ModelProfile,
    batch_size: int = 1,
    seq_length: int = 512,
    dtype: str = "fp16",
) -> float:
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    param_bytes = bytes_per_param.get(dtype, 2)

    # --- Load config (needed for kv_cache & activations regardless of path) ---
    try:
        config = AutoConfig.from_pretrained(model_profile.hf_model_path)
    except Exception:
        config = None

    # --- 1. Model weights ---
    try:
        weights_bytes = estimate_hf_model_size(model_profile, precision=dtype)
    except Exception:
        # Fallback: estimate from config if available, else from param_count
        if config is not None:
            num_params = get_num_parameters(config)
        elif model_profile.param_count is not None:
            num_params = int(model_profile.param_count)
        else:
            raise ValueError(
                f"Cannot estimate weights: both hf_model_path and param_count are unavailable "
                f"for model '{model_profile.model_name}'."
            )
        weights_bytes = int(num_params * param_bytes)

    # --- 2. KV cache ---
    # kv_cache requires architecture info; fallback to a fixed proportion of weights
    if config is not None and hasattr(config, "num_hidden_layers") and hasattr(config, "hidden_size"):
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        # Support GQA (grouped-query attention): use num_key_value_heads if present
        num_kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
        num_heads = getattr(config, "num_attention_heads", 1)
        head_dim = hidden_size // num_heads
        # 2 = K + V, kv_heads may be < num_heads for GQA models (e.g. Llama-3)
        kv_cache_bytes = int(2 * num_layers * batch_size * seq_length * num_kv_heads * head_dim * param_bytes)
    else:
        # Fallback: KV cache ≈ 10% of weights, scaled by batch and seq
        kv_cache_bytes = int(weights_bytes * 0.10 * batch_size * (seq_length / 512))

    # --- 3. Activations ---
    if config is not None and hasattr(config, "hidden_size"):
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        num_layers = getattr(config, "num_hidden_layers", 12)
        # Per layer: attention output + FFN intermediate + residuals
        per_layer = batch_size * seq_length * (hidden_size + intermediate_size + hidden_size * 2)
        activation_bytes = int(per_layer * num_layers * param_bytes)
    else:
        # Fallback: activations ≈ 5% of weights
        activation_bytes = int(weights_bytes * 0.05 * batch_size * (seq_length / 512))

    # # --- 4. CUDA context overhead ---
    # cuda_overhead_bytes = 1 * (1024 ** 3)  # ~1 GB

    total_bytes = weights_bytes + kv_cache_bytes + activation_bytes
    return total_bytes / (1024 ** 3)