#!/usr/bin/env python3
"""
Config Planner Demo Script

Run this script to test and demonstrate the config planner functionality.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extension.config_planner import (
    ConfigPlanner,
    HardwareSpec,
    WorkloadProfile,
    ModelProfile,
    quick_estimate,
)
from Ayo.engines.engine_types import EngineType


def print_separator(title: str = ""):
    """Print a visual separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print('=' * 80)
    else:
        print('-' * 80)


def search_existing_model_configs(model_name: str, engine_type: EngineType) -> Optional[ModelProfile]:
    """Search for existing model configurations in the known models list"""
    if model_name in ConfigPlanner.KNOWN_MODELS:
        model_info = ConfigPlanner.KNOWN_MODELS[model_name]
        return ModelProfile(
            model_name=model_name,
            hf_model_path=model_name,
            engine_type=engine_type,
            param_count=model_info.get("param_count"),
            embedding_dim=model_info.get("embedding_dim"),
            context_length=model_info.get("context_length"),
            model_size=model_info.get("model_size"),
        )
    return None


def demo_single_engine():
    """Demonstrate single engine configuration"""
    print_separator("Single Engine Configuration Demo")
    model_name = "BAAI/bge-small-en-v1.5"
    engine_type = EngineType.EMBEDDER
    model_profile = search_existing_model_configs(model_name, engine_type)
    if not model_profile:
        print(f"Warning: No existing model config found for '{model_name}'")
        model_profile = ModelProfile(
            model_name=model_name,
            engine_type=engine_type
        )
    
    # Define hardware (example: workstation with 2 GPUs)
    hardware = HardwareSpec(
        total_gpus=1,
        total_cpus=16,
        gpu_memory_gb=32.0,  # 24GB per GPU (e.g., RTX 3090)
        total_memory_gb=128.0,
        used_memory_gb=0.0,
        gpu_type="v100"
    )
    
    # Define workload
    workload = WorkloadProfile(
        expected_qps=5.0,
        max_batch_size=64,
        avg_input_length=512,
        latency_requirement_ms=1000.0
    )
    
    # Create planner
    planner = ConfigPlanner()

    estimation = planner.estimate_resources(model_profile, hardware, workload)
    print(f"Model: {model_profile.model_name}")
    print(f"GPUs: {estimation.num_gpus}")
    print(f"CPUs: {estimation.num_cpus}")
    print(f"Instances: {estimation.instances}")
    print(f"Memory per instance: {estimation.memory_per_instance_gb:.2f} GB")
    print(f"Recommended batch size: {estimation.recommended_batch_size}")
    print(f"Estimated throughput: {estimation.estimated_throughput:.2f} req/s")
    print(f"Estimated latency: {estimation.estimated_latency_ms:.1f} ms")
    print(f"Reasoning: {estimation.reasoning}")
    
    # Test LLM model
    print("\n2. LLM Model Configuration:")
    print_separator()
    model_profile = ModelProfile(
        model_name="openai-community/gpt2",
        engine_type=EngineType.LLM
    )
    
    estimation = planner.estimate_resources(model_profile, hardware, workload)
    print(f"Model: {model_profile.model_name}")
    print(f"GPUs: {estimation.num_gpus}")
    print(f"CPUs: {estimation.num_cpus}")
    print(f"Instances: {estimation.instances}")
    print(f"Memory per instance: {estimation.memory_per_instance_gb:.2f} GB")
    print(f"Recommended batch size: {estimation.recommended_batch_size}")
    print(f"Estimated throughput: {estimation.estimated_throughput:.2f} req/s")
    print(f"Estimated latency: {estimation.estimated_latency_ms:.1f} ms")
    print(f"Reasoning: {estimation.reasoning}")


def demo_quick_estimate():
    """Demonstrate quick estimation feature"""
    print_separator("Quick Estimation Demo")
    
    hardware = HardwareSpec(
        total_gpus=1,
        total_cpus=8,
        gpu_memory_gb=16.0,
        total_memory_gb=32.0
    )
    
    models_to_test = [
        ("BAAI/bge-small-en-v1.5", EngineType.EMBEDDER),
        ("BAAI/bge-reranker-base", EngineType.RERANKER),
        ("openai-community/gpt2", EngineType.LLM),
    ]
    
    print("\nQuick resource estimates for common models:")
    print_separator()
    
    for model_name, engine_type in models_to_test:
        estimation = quick_estimate(model_name, engine_type, hardware)
        print(f"\n{model_name} ({engine_type}):")
        print(f"  GPU: {estimation.num_gpus}, CPU: {estimation.num_cpus}, Instances: {estimation.instances}")
        print(f"  Memory: {estimation.memory_per_instance_gb:.2f} GB")
        print(f"  Throughput: {estimation.estimated_throughput:.2f} req/s")


def test_est_model_mem():
    model_name = "BAAI/bge-small-en-v1.5"
    engine_type = EngineType.EMBEDDER
    model_profile = search_existing_model_configs(model_name, engine_type)
    if not model_profile:
        print(f"Warning: No existing model config found for '{model_name}'")
        model_profile = ModelProfile(
            model_name=model_name,
            engine_type=engine_type
        )
    
    from mem_utils.est_embed import estimate_embed_infer_memory, estimate_sentence_transformer_size
    model_weights_size = estimate_sentence_transformer_size(model_profile=model_profile) 
    print(f"Estimated model weights size for {model_profile.model_name}: {model_weights_size} bytes, which is {model_weights_size / (1024**3):.2f} GB")

    total_required_memory_gb = estimate_embed_infer_memory(model_profile=model_profile)
    print(f"Estimated total GPU memory required for inference: {total_required_memory_gb:.2f} GB")


def test_est_reranker_mem():
    model_name = "BAAI/bge-reranker-base"
    engine_type = EngineType.RERANKER
    model_profile = search_existing_model_configs(model_name, engine_type)
    if not model_profile:
        print(f"Warning: No existing model config found for '{model_name}'")
        model_profile = ModelProfile(
            model_name=model_name,
            engine_type=engine_type
        )
    from mem_utils.est_reranker import estimate_reranker_infer_memory, estimate_sentence_transformer_size
    model_weights_size = estimate_sentence_transformer_size(model_profile=model_profile) 
    print(f"Estimated model weights size for {model_profile.model_name}: {model_weights_size} bytes, which is {model_weights_size / (1024**3):.2f} GB")

    total_required_memory_gb = estimate_reranker_infer_memory(model_profile=model_profile)
    print(f"Estimated total GPU memory required for inference: {total_required_memory_gb:.2f} GB")


def main():
    """Main entry point"""
    print_separator("AYO Config Planner Demo")
    demo_single_engine()

def test_entry():
    test_est_model_mem()
    # test_est_reranker_mem()
if __name__ == "__main__":
    # main()
    test_entry()
