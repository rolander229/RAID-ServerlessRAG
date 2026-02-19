# Config Planner Usage Guide

The Config Planner automatically estimates optimal resource allocation for Ayo engines based on model characteristics, available hardware, and workload requirements.

## Overview

The planner helps you:
- **Estimate resource requirements** (GPUs, CPUs, memory, instances)
- **Optimize configurations** for your hardware constraints
- **Plan multi-engine deployments** with resource sharing
- **Predict performance** (throughput, latency)

## Quick Start

```python
from extension.config_planner import (
    ConfigPlanner,
    HardwareSpec,
    WorkloadProfile,
    ModelProfile,
)
from Ayo.engines.engine_types import EngineType

# 1. Define your hardware
hardware = HardwareSpec(
    total_gpus=2,
    total_cpus=16,
    gpu_memory_gb=24.0,  # 24GB per GPU
    total_memory_gb=64.0,
    gpu_type="RTX3090"
)

# 2. Define your workload
workload = WorkloadProfile(
    expected_qps=10.0,  # 10 queries per second
    max_batch_size=32,
    avg_input_length=512,
    latency_requirement_ms=500.0
)

# 3. Create planner and estimate
planner = ConfigPlanner()

model_profile = ModelProfile(
    model_name="BAAI/bge-small-en-v1.5",
    engine_type=EngineType.EMBEDDER
)

estimation = planner.estimate_resources(model_profile, hardware, workload)

print(f"Recommended GPUs: {estimation.num_gpus}")
print(f"Recommended CPUs: {estimation.num_cpus}")
print(f"Instances: {estimation.instances}")
print(f"Batch size: {estimation.recommended_batch_size}")
print(f"Expected throughput: {estimation.estimated_throughput} req/s")
print(f"Expected latency: {estimation.estimated_latency_ms} ms")
print(f"Reasoning: {estimation.reasoning}")
```

## Input Classes

### HardwareSpec

Defines available hardware resources:

```python
hardware = HardwareSpec(
    total_gpus=2,           # Number of GPUs available
    total_cpus=16,          # Number of CPU cores
    gpu_memory_gb=24.0,     # Memory per GPU in GB
    total_memory_gb=64.0,   # Total system RAM in GB
    gpu_type="A100"         # Optional: GPU model name
)
```

### WorkloadProfile

Describes expected workload characteristics:

```python
workload = WorkloadProfile(
    expected_qps=5.0,              # Queries per second
    max_batch_size=64,             # Maximum batch size
    avg_input_length=512,          # Average input tokens/chars
    avg_output_length=128,         # Average output length (for LLMs)
    latency_requirement_ms=1000.0, # Latency SLA in milliseconds
    concurrent_requests=20         # Concurrent requests
)
```

### ModelProfile

Specifies model characteristics:

```python
model_profile = ModelProfile(
    model_name="meta-llama/Llama-2-7b",
    engine_type=EngineType.LLM,
    model_size=ModelSize.MEDIUM,      # Optional: TINY, SMALL, MEDIUM, LARGE, XLARGE
    param_count=7.0,                  # Optional: Billions of parameters
    embedding_dim=768,                # Optional: For embedding models
    context_length=4096,              # Optional: For LLMs
    estimated_memory_gb=14.0          # Optional: Manual memory estimate
)
```

**Note**: The planner has a built-in database of common models, so you often only need `model_name` and `engine_type`.

## Output Classes

### ResourceEstimation

Contains recommended resource allocation and performance predictions:

```python
estimation = planner.estimate_resources(model_profile, hardware, workload)

# Access results
estimation.num_gpus                  # GPUs per instance (can be fractional)
estimation.num_cpus                  # CPUs per instance
estimation.instances                 # Number of instances to deploy
estimation.memory_per_instance_gb    # Memory needed per instance
estimation.recommended_batch_size    # Optimal batch size
estimation.estimated_throughput      # Expected requests per second
estimation.estimated_latency_ms      # Expected latency in milliseconds
estimation.reasoning                 # Explanation of the recommendation
```

## Common Usage Patterns

### 1. Single Engine Configuration

Create an optimized `EngineConfig` directly:

```python
from Ayo.configs.config import EngineConfig

planner = ConfigPlanner()

# Create engine config with optimization
config, estimation = planner.create_engine_config(
    name="embedding_service",
    model_profile=ModelProfile(
        model_name="BAAI/bge-small-en-v1.5",
        engine_type=EngineType.EMBEDDER
    ),
    hardware=hardware,
    workload=workload,
    base_model_config={"device": "cuda"}  # Optional base config
)

# Use the config
print(f"Config: {config.dict()}")
print(f"Expected throughput: {estimation.estimated_throughput} req/s")
```

### 2. Multi-Engine Planning

Plan multiple engines with resource sharing:

```python
# Define all engines
engine_specs = [
    ("embedding_service", 
     ModelProfile(model_name="BAAI/bge-small-en-v1.5", engine_type=EngineType.EMBEDDER),
     {"device": "cuda"}),
    
    ("reranker_service",
     ModelProfile(model_name="BAAI/bge-reranker-base", engine_type=EngineType.RERANKER),
     {"device": "cuda"}),
    
    ("llm_service",
     ModelProfile(model_name="openai-community/gpt2", engine_type=EngineType.LLM),
     {"tensor_parallel_size": 1}),
    
    ("vector_db_service",
     ModelProfile(model_name="pgvector", engine_type=EngineType.VECTOR_DB, embedding_dim=384),
     {"host": "localhost", "port": 5432}),
]

# Plan all engines together
configs, estimations, summary = planner.plan_multi_engine_config(
    engine_specs=engine_specs,
    hardware=hardware,
    workload=workload,
    dataset_size=100000  # For vector DB
)

# Check resource allocation
print(f"Total GPUs used: {summary['total_gpus_allocated']:.2f}")
print(f"Total CPUs used: {summary['total_cpus_allocated']}")
print(f"GPU utilization: {summary['gpu_utilization']:.1f}%")
print(f"Feasible: {summary['feasible']}")

# Use configs
for config, est in zip(configs, estimations):
    print(f"\n{config.name}:")
    print(f"  GPUs: {est.num_gpus}, CPUs: {est.num_cpus}, Instances: {est.instances}")
    print(f"  Throughput: {est.estimated_throughput:.2f} req/s")
    print(f"  {est.reasoning}")
```

### 3. Engine-Specific Estimation

#### Embedding Model

```python
model_profile = ModelProfile(
    model_name="BAAI/bge-large-en-v1.5",
    engine_type=EngineType.EMBEDDER
)

estimation = planner.estimate_embedder_resources(
    model_profile, hardware, workload
)
```

#### LLM Model

```python
model_profile = ModelProfile(
    model_name="meta-llama/Llama-2-7b",
    engine_type=EngineType.LLM,
    param_count=7.0
)

estimation = planner.estimate_llm_resources(
    model_profile, hardware, workload
)
```

#### Reranker Model

```python
model_profile = ModelProfile(
    model_name="BAAI/bge-reranker-large",
    engine_type=EngineType.RERANKER
)

estimation = planner.estimate_reranker_resources(
    model_profile, hardware, workload
)
```

#### Vector Database

```python
model_profile = ModelProfile(
    model_name="pgvector",
    engine_type=EngineType.VECTOR_DB,
    embedding_dim=768
)

estimation = planner.estimate_vectordb_resources(
    model_profile, hardware, workload,
    dataset_size=1000000  # 1M vectors
)
```

### 4. Quick Estimation

For rapid prototyping:

```python
from extension.config_planner import quick_estimate

estimation = quick_estimate(
    model_name="BAAI/bge-small-en-v1.5",
    engine_type=EngineType.EMBEDDER,
    hardware=hardware
)
```

## Complete Example: RAG Pipeline

```python
from extension.config_planner import ConfigPlanner, HardwareSpec, WorkloadProfile, ModelProfile
from Ayo.engines.engine_types import EngineType
from Ayo.configs.config import EngineConfig

# Define hardware
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

# Plan all engines for RAG pipeline
planner = ConfigPlanner()

engine_specs = [
    # Embedding service for documents
    ("embedding_service",
     ModelProfile(
         model_name="BAAI/bge-small-en-v1.5",
         engine_type=EngineType.EMBEDDER,
         embedding_dim=384
     ),
     {"max_batch_size": 1024}),
    
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
         "database": "database_asplos"
     }),
    
    # Reranker
    ("reranker_service",
     ModelProfile(
         model_name="BAAI/bge-reranker-base",
         engine_type=EngineType.RERANKER
     ),
     {}),
    
    # LLM for generation
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

# Generate configs
configs, estimations, summary = planner.plan_multi_engine_config(
    engine_specs=engine_specs,
    hardware=hardware,
    workload=workload,
    dataset_size=500000  # 500K documents
)

# Print summary
print("=" * 60)
print("RAG Pipeline Resource Plan")
print("=" * 60)
print(f"GPU Utilization: {summary['gpu_utilization']:.1f}%")
print(f"CPU Utilization: {summary['cpu_utilization']:.1f}%")
print(f"Total Memory: {summary['total_memory_allocated_gb']:.2f} GB")
print(f"Deployment Feasible: {summary['feasible']}")
print()

# Print per-engine details
for config, est in zip(configs, estimations):
    print(f"{config.name}:")
    print(f"  Model: {config.model_config.get('model_name', 'N/A')}")
    print(f"  Resources: {est.num_gpus} GPU, {est.num_cpus} CPU, {est.instances} instance(s)")
    print(f"  Memory: {est.memory_per_instance_gb:.2f} GB per instance")
    print(f"  Performance: {est.estimated_throughput:.2f} req/s, {est.estimated_latency_ms:.1f} ms")
    print(f"  Reasoning: {est.reasoning}")
    print()

# Register configs with Ayo APP
from Ayo.app import APP

app = APP.init()
for config in configs:
    app.register_engine(config)
```

## Advanced Features

### Memory Estimation

The planner estimates model memory based on:
- Known model database (pre-computed values)
- Parameter count and precision (fp32, fp16, int8, int4)
- Activation and KV cache overhead

```python
memory_gb = planner.estimate_model_memory(
    model_profile=ModelProfile(
        model_name="meta-llama/Llama-2-13b",
        engine_type=EngineType.LLM,
        param_count=13.0
    ),
    precision="fp16"  # Options: fp32, fp16, int8, int4
)
print(f"Estimated memory: {memory_gb} GB")
```

### Custom Model Specifications

For models not in the database:

```python
custom_model = ModelProfile(
    model_name="my-custom-llm",
    engine_type=EngineType.LLM,
    param_count=3.5,              # 3.5B parameters
    context_length=2048,
    estimated_memory_gb=8.0       # Manual estimate
)

estimation = planner.estimate_resources(custom_model, hardware, workload)
```

### Resource Constraint Scenarios

#### Limited GPU Memory

```python
hardware_limited = HardwareSpec(
    total_gpus=1,
    total_cpus=8,
    gpu_memory_gb=8.0,  # Only 8GB GPU
    total_memory_gb=32.0
)

# Planner will adjust recommendations
estimation = planner.estimate_resources(model_profile, hardware_limited, workload)
print(estimation.reasoning)  # Shows why certain choices were made
```

#### CPU-Only Setup

```python
hardware_cpu_only = HardwareSpec(
    total_gpus=0,
    total_cpus=32,
    gpu_memory_gb=0.0,
    total_memory_gb=128.0
)

# Planner falls back to CPU configurations
estimation = planner.estimate_resources(model_profile, hardware_cpu_only, workload)
```

#### High Throughput Requirements

```python
high_throughput_workload = WorkloadProfile(
    expected_qps=100.0,  # 100 queries/second
    max_batch_size=256,
    latency_requirement_ms=5000.0  # Relaxed latency for higher throughput
)

# Planner will recommend more instances
estimation = planner.estimate_resources(model_profile, hardware, high_throughput_workload)
print(f"Recommended instances: {estimation.instances}")
```

## Supported Models

The planner has built-in support for:

### Embedding Models
- `BAAI/bge-small-en-v1.5` (33M params, 384d)
- `BAAI/bge-base-en-v1.5` (109M params, 768d)
- `BAAI/bge-large-en-v1.5` (335M params, 1024d)

### Reranker Models
- `BAAI/bge-reranker-base` (278M params)
- `BAAI/bge-reranker-large` (560M params)

### Language Models
- `openai-community/gpt2` (124M params)
- `gpt2-medium` (355M params)
- `gpt2-large` (774M params)
- `gpt2-xl` (1.5B params)
- `meta-llama/Llama-2-7b` (7B params)
- `meta-llama/Llama-2-13b` (13B params)

For other models, the planner will estimate based on parameter count or use conservative defaults.

## Best Practices

1. **Start with realistic workload estimates**: Use production metrics or load testing data
2. **Leave headroom**: Don't allocate 100% of resources; leave 10-20% buffer
3. **Validate estimations**: Run benchmarks to verify throughput/latency predictions
4. **Monitor in production**: Adjust configs based on actual usage patterns
5. **Use multi-engine planning**: For better resource sharing and conflict detection
6. **Consider cost-performance tradeoffs**: Higher instances = better throughput but more cost

## Limitations

- Estimations are based on heuristics and may vary by 20-40%
- Actual performance depends on model optimization, batching strategy, and hardware specifics
- Network latency and I/O bottlenecks are not modeled
- Multi-GPU configurations assume ideal scaling (may not hold for all models)

## Integration with Ayo

The planner generates `EngineConfig` objects compatible with Ayo's APP:

```python
# Generate optimized config
config, estimation = planner.create_engine_config(
    name="my_service",
    model_profile=model_profile,
    hardware=hardware,
    workload=workload
)

# Direct integration with Ayo
from Ayo.app import APP

app = APP.init()
app.register_engine(config)  # Use the optimized config
await app.start()
```

## Troubleshooting

**Issue**: "WARNING: LLM requires GPU but none available"
- **Solution**: Ensure `hardware.total_gpus > 0` or use smaller models for CPU inference

**Issue**: High memory estimates
- **Solution**: Consider model quantization (int8, int4) or use smaller model variants

**Issue**: Low estimated throughput
- **Solution**: Increase `instances`, use larger batch sizes, or add more hardware

**Issue**: Infeasible resource allocation
- **Solution**: Check `summary['feasible']` in multi-engine planning; reduce model sizes or instances

## API Reference

See docstrings in [config_planner.py](config_planner.py) for detailed API documentation.
