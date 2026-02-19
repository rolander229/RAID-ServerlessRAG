from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from extension.model_profile import ModelProfile
import math

MEMORY_PER_BILLION_PARAMS = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

def estimate_sentence_transformer_size(
    model_profile: ModelProfile,
    model_kwargs: Optional[Dict[str, Any]] = None,
    precision: str = "fp16"
) -> float:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for get_sentence_transformer_size"
        ) from exc

    default_model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "meta",
    }
    combined_model_kwargs = {**default_model_kwargs, **(model_kwargs or {})}

    # Instantiate SentenceTransformer with meta device to avoid allocating weights
    model = SentenceTransformer(
        model_profile.hf_model_path,
        device="meta",
        model_kwargs=combined_model_kwargs,
    )
    
    if precision is not None:
        try: 
            precision_bytes = MEMORY_PER_BILLION_PARAMS.get(precision, 4.0)
            total_size_bytes = sum(p.numel() for p in model.parameters()) * precision_bytes
        except KeyError:
            total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    else:
        total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    return int(total_size_bytes)


def _get_model_config_params(hf_model_path: str) -> Optional[Dict[str, int]]:
    """
    Try to load architecture params from HuggingFace config.json.
    Returns dict with hidden_size, num_hidden_layers, num_attention_heads, intermediate_size,
    or None if not available.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(hf_model_path)
        return {
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
        }
    except Exception:
        return None


def _estimate_bert_activation_bytes(
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    batch_size: int,
    seq_length: int,
    dtype: str,
) -> int:
    """
    Estimate activation memory for a BERT-style transformer during inference.
    
    Per layer, major activations are:
      - Attention QKV projections:  3 * (batch * seq * hidden)
      - Attention scores matrix:    batch * heads * seq * seq
      - Attention output:           batch * seq * hidden
      - FFN intermediate:           batch * seq * intermediate_size
      - FFN output:                 batch * seq * hidden
      - LayerNorm inputs (x2):      2 * batch * seq * hidden

    We keep all layer activations in memory simultaneously for inference
    (no gradient checkpointing needed, but full forward pass buffers exist).
    """
    bytes_per_element = MEMORY_PER_BILLION_PARAMS.get(dtype, 2.0)
    
    B, S, H, L, I = batch_size, seq_length, hidden_size, num_hidden_layers, intermediate_size
    A = num_attention_heads
    head_dim = H // A

    # Per-layer activations (in elements)
    attn_qkv        = 3 * B * S * H                  # Q, K, V projections
    attn_scores     = B * A * S * S                  # attention score matrix
    attn_context    = B * S * H                      # weighted sum output
    attn_out_proj   = B * S * H                      # output projection
    ffn_intermediate = B * S * I                     # FFN up-projection
    ffn_output      = B * S * H                      # FFN down-projection
    layernorm_bufs  = 2 * B * S * H                  # 2x LayerNorm inputs

    per_layer_elements = (
        attn_qkv + attn_scores + attn_context +
        attn_out_proj + ffn_intermediate + ffn_output + layernorm_bufs
    )

    # Embedding layer output + final layer norm output
    embedding_output = B * S * H
    final_layernorm  = B * S * H

    total_elements = per_layer_elements * L + embedding_output + final_layernorm
    return int(total_elements * bytes_per_element)


def _estimate_activation_bytes_fallback(
    model_weights_bytes: int,
    batch_size: int,
    seq_length: int,
) -> int:
    """
    Fallback: activation ≈ 25% of model weights, scaled by batch_size and
    a mild seq_length factor (relative to typical 512).
    """
    seq_scale = seq_length / 512
    batch_scale = batch_size
    return int(model_weights_bytes * 0.25 * seq_scale * batch_scale)


def _estimate_activation_memory(
    hf_model_path: str,
    model_weights_bytes: int,
    batch_size: int,
    seq_length: int,
    dtype: str,
) -> int:
    """
    Estimate activation memory: try architecture-aware formula first,
    fallback to empirical coefficient.
    """
    config_params = _get_model_config_params(hf_model_path)
    
    if config_params and all(v is not None for v in config_params.values()):
        return _estimate_bert_activation_bytes(
            hidden_size=config_params["hidden_size"],
            num_hidden_layers=config_params["num_hidden_layers"],
            num_attention_heads=config_params["num_attention_heads"],
            intermediate_size=config_params["intermediate_size"],
            batch_size=batch_size,
            seq_length=seq_length,
            dtype=dtype,
        )
    else:
        return _estimate_activation_bytes_fallback(
            model_weights_bytes=model_weights_bytes,
            batch_size=batch_size,
            seq_length=seq_length,
        )

def estimate_reranker_infer_memory(
    model_profile: ModelProfile,
    batch_size: int = 1,
    seq_length: int = 512,
    dtype: str = "fp16",
) -> float:
    """
    Estimate total GPU/CPU memory (in GB) for reranker model inference.
    
    Total = model weights + activations
    
    For rerankers (Cross-Encoder), query and document are concatenated,
    so the effective sequence length is typically up to 2x that of an
    embedding model. seq_length here should already reflect the combined
    query+doc length (default 512 = max for most BERT-based rerankers).
    """
    weights_bytes = estimate_sentence_transformer_size(model_profile, precision=dtype)
    activation_bytes = _estimate_activation_memory(
        hf_model_path=model_profile.hf_model_path,
        model_weights_bytes=weights_bytes,
        batch_size=batch_size,
        seq_length=seq_length,
        dtype=dtype,
    )

    total_bytes = weights_bytes + activation_bytes
    return total_bytes / (1024 ** 3)


