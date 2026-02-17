#!/usr/bin/env python3
"""
Config Planner Demo Script

Run this script to test and demonstrate the config planner functionality.
"""

import sys
from pathlib import Path

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


def demo_single_engine():
    """Demonstrate single engine configuration"""
    print_separator("Single Engine Configuration Demo")
    
    # Define hardware (example: workstation with 2 GPUs)
    hardware = HardwareSpec(
        total_gpus=2,
        total_cpus=16,
        gpu_memory_gb=24.0,  # 24GB per GPU (e.g., RTX 3090)
        total_memory_gb=64.0,
        gpu_type="RTX3090"
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
    
    # Test embedding model
    print("\n1. Embedding Model Configuration:")
    print_separator()
    model_profile = ModelProfile(
        model_name="BAAI/bge-small-en-v1.5",
        engine_type=EngineType.EMBEDDER
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


def demo_multi_engine_rag_pipeline():
    """Demonstrate multi-engine RAG pipeline configuration"""
    print_separator("Multi-Engine RAG Pipeline Configuration")
    
    # Define hardware (matching typical deployment scenario)
    hardware = HardwareSpec(
        total_gpus=2,
        total_cpus=16,
        gpu_memory_gb=24.0,
        total_memory_gb=64.0,
        gpu_type="RTX3090"
    )
    
    # Define workload
    workload = WorkloadProfile(
        expected_qps=5.0,
        max_batch_size=64,
        avg_input_length=512,
        avg_output_length=128,
        latency_requirement_ms=2000.0
    )
    
    # Create planner
    planner = ConfigPlanner()
    
    # Define all engines for RAG pipeline
    engine_specs = [
        # Embedding service
        ("embedding_service",
         ModelProfile(
             model_name="BAAI/bge-small-en-v1.5",
             engine_type=EngineType.EMBEDDER,
             embedding_dim=384
         ),
         {"max_batch_size": 1024, "device": "cuda"}),
        
        # Vector database
        ("vector_db_service",
         ModelProfile(
             model_name="pgvector",
             engine_type=EngineType.VECTOR_DB,
             embedding_dim=384
         ),
         {
             "host": "localhost",
             "port": 5432,
             "user": "asplos25",
             "password": "123456",
             "database": "database_asplos",
             "vector_dim": 384
         }),
        
        # Reranker
        ("reranker_service",
         ModelProfile(
             model_name="BAAI/bge-reranker-base",
             engine_type=EngineType.RERANKER
         ),
         {"device": "cuda"}),
        
        # LLM
        ("llm_service",
         ModelProfile(
             model_name="openai-community/gpt2",
             engine_type=EngineType.LLM
         ),
         {
             "tensor_parallel_size": 1,
             "trust_remote_code": False,
             "dtype": "auto"
         }),
    ]
    
    # Plan all engines
    configs, estimations, summary = planner.plan_multi_engine_config(
        engine_specs=engine_specs,
        hardware=hardware,
        workload=workload,
        dataset_size=500000  # 500K vectors in database
    )
    
    # Print summary
    print("\nResource Allocation Summary:")
    print_separator()
    print(f"Total GPUs allocated: {summary['total_gpus_allocated']:.2f} / {hardware.total_gpus}")
    print(f"Total CPUs allocated: {summary['total_cpus_allocated']} / {hardware.total_cpus}")
    print(f"Total Memory allocated: {summary['total_memory_allocated_gb']:.2f} GB / {hardware.total_memory_gb:.2f} GB")
    print(f"GPU utilization: {summary['gpu_utilization']:.1f}%")
    print(f"CPU utilization: {summary['cpu_utilization']:.1f}%")
    print(f"Deployment feasible: {'✓ YES' if summary['feasible'] else '✗ NO'}")
    
    # Print per-engine details
    print("\nPer-Engine Configuration:")
    print_separator()
    for config, est in zip(configs, estimations):
        print(f"\n{config.name}:")
        print(f"  Model: {config.model_config.get('model_name', 'N/A')}")
        print(f"  Engine Type: {config.engine_type}")
        print(f"  Resources: {est.num_gpus} GPU(s), {est.num_cpus} CPU(s)")
        print(f"  Instances: {est.instances}")
        print(f"  Memory: {est.memory_per_instance_gb:.2f} GB per instance")
        print(f"  Batch Size: {est.recommended_batch_size}")
        print(f"  Performance: {est.estimated_throughput:.2f} req/s, {est.estimated_latency_ms:.1f} ms")
        print(f"  Reasoning: {est.reasoning}")
    
    # Print generated configs (for copy-paste into code)
    print("\nGenerated EngineConfig Objects (Python code):")
    print_separator()
    for config in configs:
        print(f"\n{config.name} = EngineConfig(")
        print(f"    name=\"{config.name}\",")
        print(f"    engine_type=EngineType.{config.engine_type.upper()},")
        print(f"    num_gpus={config.num_gpus},")
        print(f"    num_cpus={config.num_cpus},")
        print(f"    instances={config.instances},")
        print(f"    resources={{}},")
        print(f"    model_config={config.model_config},")
        print(f"    latency_profile={config.latency_profile},")
        print(f")")


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


def demo_hardware_scenarios():
    """Demonstrate different hardware scenarios"""
    print_separator("Hardware Scenario Comparison")
    
    scenarios = [
        ("High-end Workstation", HardwareSpec(
            total_gpus=2, total_cpus=16, gpu_memory_gb=24.0, 
            total_memory_gb=64.0, gpu_type="RTX3090"
        )),
        ("Single GPU Server", HardwareSpec(
            total_gpus=1, total_cpus=8, gpu_memory_gb=16.0,
            total_memory_gb=32.0, gpu_type="RTX3080"
        )),
        ("CPU-Only Server", HardwareSpec(
            total_gpus=0, total_cpus=32, gpu_memory_gb=0.0,
            total_memory_gb=128.0
        )),
        ("Cloud GPU Instance", HardwareSpec(
            total_gpus=4, total_cpus=32, gpu_memory_gb=40.0,
            total_memory_gb=256.0, gpu_type="A100"
        )),
    ]
    
    workload = WorkloadProfile(expected_qps=10.0, max_batch_size=64)
    planner = ConfigPlanner()
    model = ModelProfile(model_name="BAAI/bge-small-en-v1.5", engine_type=EngineType.EMBEDDER)
    
    print("\nEmbedding Model Estimates Across Hardware:")
    print_separator()
    
    for scenario_name, hardware in scenarios:
        estimation = planner.estimate_resources(model, hardware, workload)
        print(f"\n{scenario_name}:")
        print(f"  Hardware: {hardware.total_gpus} GPU, {hardware.total_cpus} CPU, {hardware.gpu_memory_gb}GB VRAM")
        print(f"  Allocation: {estimation.num_gpus} GPU, {estimation.num_cpus} CPU, {estimation.instances} instance(s)")
        print(f"  Performance: {estimation.estimated_throughput:.2f} req/s, {estimation.estimated_latency_ms:.1f} ms")
        print(f"  {estimation.reasoning}")


def interactive_mode():
    """Interactive mode for custom estimations"""
    print_separator("Interactive Config Planner")
    
    print("\nThis mode allows you to input custom parameters for estimation.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        # Get hardware specs
        print("Hardware Configuration:")
        total_gpus = int(input("  Total GPUs [2]: ") or "2")
        total_cpus = int(input("  Total CPUs [16]: ") or "16")
        gpu_memory_gb = float(input("  GPU Memory per GPU (GB) [24.0]: ") or "24.0")
        total_memory_gb = float(input("  Total System Memory (GB) [64.0]: ") or "64.0")
        
        hardware = HardwareSpec(
            total_gpus=total_gpus,
            total_cpus=total_cpus,
            gpu_memory_gb=gpu_memory_gb,
            total_memory_gb=total_memory_gb
        )
        
        # Get workload profile
        print("\nWorkload Profile:")
        expected_qps = float(input("  Expected QPS [5.0]: ") or "5.0")
        max_batch_size = int(input("  Max Batch Size [64]: ") or "64")
        avg_input_length = int(input("  Avg Input Length [512]: ") or "512")
        
        workload = WorkloadProfile(
            expected_qps=expected_qps,
            max_batch_size=max_batch_size,
            avg_input_length=avg_input_length
        )
        
        # Get model info
        print("\nModel Selection:")
        print("  1. BAAI/bge-small-en-v1.5 (Embedder)")
        print("  2. BAAI/bge-reranker-base (Reranker)")
        print("  3. openai-community/gpt2 (LLM)")
        print("  4. Custom model")
        
        choice = input("\nSelect model [1]: ") or "1"
        
        if choice == "1":
            model = ModelProfile(model_name="BAAI/bge-small-en-v1.5", engine_type=EngineType.EMBEDDER)
        elif choice == "2":
            model = ModelProfile(model_name="BAAI/bge-reranker-base", engine_type=EngineType.RERANKER)
        elif choice == "3":
            model = ModelProfile(model_name="openai-community/gpt2", engine_type=EngineType.LLM)
        else:
            model_name = input("  Model name: ")
            engine_type_str = input("  Engine type (embedder/reranker/llm) [embedder]: ") or "embedder"
            engine_type = EngineType.EMBEDDER if engine_type_str == "embedder" else (
                EngineType.RERANKER if engine_type_str == "reranker" else EngineType.LLM
            )
            model = ModelProfile(model_name=model_name, engine_type=engine_type)
        
        # Perform estimation
        planner = ConfigPlanner()
        estimation = planner.estimate_resources(model, hardware, workload)
        
        # Print results
        print_separator("Estimation Results")
        print(f"\nModel: {model.model_name}")
        print(f"Engine Type: {model.engine_type}")
        print(f"\nRecommended Configuration:")
        print(f"  GPUs: {estimation.num_gpus}")
        print(f"  CPUs: {estimation.num_cpus}")
        print(f"  Instances: {estimation.instances}")
        print(f"  Memory per instance: {estimation.memory_per_instance_gb:.2f} GB")
        print(f"  Batch size: {estimation.recommended_batch_size}")
        print(f"\nExpected Performance:")
        print(f"  Throughput: {estimation.estimated_throughput:.2f} requests/second")
        print(f"  Latency: {estimation.estimated_latency_ms:.1f} ms")
        print(f"\nReasoning: {estimation.reasoning}")
        
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode.")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Main entry point"""
    print_separator("AYO Config Planner Demo")
    print("\nThis script demonstrates the config planner functionality.")
    print("\nAvailable demos:")
    print("  1. Single engine configuration")
    print("  2. Multi-engine RAG pipeline")
    print("  3. Quick estimation")
    print("  4. Hardware scenario comparison")
    print("  5. Interactive mode")
    print("  6. Run all demos")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo [1]: ").strip() or "1"
            
            if choice == "0":
                print("Exiting...")
                break
            elif choice == "1":
                demo_single_engine()
            elif choice == "2":
                demo_multi_engine_rag_pipeline()
            elif choice == "3":
                demo_quick_estimate()
            elif choice == "4":
                demo_hardware_scenarios()
            elif choice == "5":
                interactive_mode()
            elif choice == "6":
                demo_single_engine()
                demo_multi_engine_rag_pipeline()
                demo_quick_estimate()
                demo_hardware_scenarios()
                print_separator("All Demos Completed")
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Ask if user wants to continue
            cont = input("\nRun another demo? (y/n) [y]: ").strip().lower() or "y"
            if cont != "y":
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nThank you for using the Config Planner!")


if __name__ == "__main__":
    main()
