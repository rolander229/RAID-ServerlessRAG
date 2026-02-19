"""
Config Planner for Ayo Framework

This module provides resource planning and estimation for engine configurations
based on model characteristics and available hardware resources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from Ayo.configs.config import EngineConfig
from Ayo.engines.engine_types import EngineType
from extension.model_profile import ModelProfile
from extension.mem_utils.est_embed import estimate_embed_infer_memory
from extension.mem_utils.est_reranker import estimate_reranker_infer_memory
from extension.mem_utils.est_llm import estimate_llm_infer_memory
from extension.mem_utils.known_models import KNOWN_MODELS_CONFIG


class ModelScale(str, Enum):
    """Model size categories"""
    TINY = "tiny"  # < 100M params
    SMALL = "small"  # 100M - 1B params
    MEDIUM = "medium"  # 1B - 7B params
    LARGE = "large"  # 7B - 30B params
    XLARGE = "xlarge"  # > 30B params


@dataclass
class HardwareSpec:
    """Available hardware specifications"""
    
    total_gpus: int = 0
    total_cpus: int = 1
    gpu_memory_gb: float = 0.0  # Memory per GPU in GB
    total_memory_gb: float = 0.0  # Total system memory in GB
    gpu_type: Optional[str] = None  # e.g., "A100", "V100", "RTX3090"
    
    def dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_gpus": self.total_gpus,
            "total_cpus": self.total_cpus,
            "gpu_memory_gb": self.gpu_memory_gb,
            "total_memory_gb": self.total_memory_gb,
            "gpu_type": self.gpu_type,
        }


@dataclass
class WorkloadProfile:
    """Expected workload characteristics"""
    
    expected_qps: float = 1.0  # Queries per second
    max_batch_size: int = 32
    avg_input_length: int = 512  # Average input token/text length
    avg_output_length: int = 128  # Average output length (for LLM)
    latency_requirement_ms: float = 1000.0  # Latency requirement in ms
    concurrent_requests: int = 10
    
    def dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "expected_qps": self.expected_qps,
            "max_batch_size": self.max_batch_size,
            "avg_input_length": self.avg_input_length,
            "avg_output_length": self.avg_output_length,
            "latency_requirement_ms": self.latency_requirement_ms,
            "concurrent_requests": self.concurrent_requests,
        }


@dataclass
class ResourceEstimation:
    """Estimated resource requirements"""
    
    num_gpus: float
    num_cpus: int
    instances: int
    memory_per_instance_gb: float
    recommended_batch_size: int
    estimated_throughput: float  # requests per second
    estimated_latency_ms: float
    reasoning: str  # Explanation of the estimation
    
    def dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "instances": self.instances,
            "memory_per_instance_gb": self.memory_per_instance_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "estimated_throughput": self.estimated_throughput,
            "estimated_latency_ms": self.estimated_latency_ms,
            "reasoning": self.reasoning,
        }


class ConfigPlanner:
    """
    Resource planner for engine configurations.
    
    Estimates optimal resource allocation based on:
    - Model characteristics (size, type, memory requirements)
    - Available hardware (GPUs, CPUs, memory)
    - Workload profile (throughput, latency requirements)
    """
    
    # Model memory estimation heuristics (in GB per billion parameters)
    MEMORY_PER_BILLION_PARAMS = {
        "fp32": 4.0,
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    
    # Known model specs
    KNOWN_MODELS = KNOWN_MODELS_CONFIG
    
    def __init__(self):
        """Initialize the config planner"""
        pass
        
    def estimate_model_memory(
        self,
        model_profile: ModelProfile,
        precision: str = "fp16"
    ) -> float:
        """
        Estimate model memory requirements in GB.
        
        Args:
            model_profile: Model characteristics
            precision: Model precision (fp32, fp16, int8, int4)
        
        Returns:
            Estimated memory in GB
        """
        # Check if we have precomputed values
        if model_profile.model_name in self.KNOWN_MODELS:
            return self.KNOWN_MODELS[model_profile.model_name]["model_size"]
        
        if model_profile.estimated_memory_gb:
            return model_profile.estimated_memory_gb
        
        # Estimate based on parameter count
        if model_profile.param_count:
            base_memory = (
                model_profile.param_count * 
                self.MEMORY_PER_BILLION_PARAMS.get(precision, 2.0)
            )
            # Add overhead for activations and KV cache (20-30%)
            return base_memory * 1.3
        
        # Fallback defaults by engine type
        defaults = {
            EngineType.EMBEDDER: 1.0,
            EngineType.RERANKER: 1.5,
            EngineType.LLM: 5.0,
            EngineType.VECTOR_DB: 0.5,
        }
        return defaults.get(model_profile.engine_type, 1.0)
    
    def estimate_embedder_resources(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile
    ) -> ResourceEstimation:
        """Estimate resources for embedding models"""
        memory_required = self.estimate_model_memory(model_profile)
        # try:
        #     memory_required = estimate_embed_infer_memory(model_profile)
        # except Exception as e:
        #     print(f"Error estimating embedder memory: {e}\nFalling back to default model memory estimation.")
        #     memory_required = self.estimate_model_memory(model_profile)
        
        # Embedders are usually fast, prioritize throughput
        if hardware.total_gpus > 0 and hardware.gpu_memory_gb >= memory_required:
            # Use GPU for better throughput
            num_gpus = min(0.2, hardware.total_gpus)  # Fractional GPU is fine
            num_cpus = 1
            batch_size = min(workload.max_batch_size, 1024)
            throughput = batch_size * 50  # ~50 batches/sec on GPU
            latency = 20  # ~20ms per batch
            reasoning = "GPU-accelerated embedding with fractional GPU allocation"
        else:
            # CPU fallback
            num_gpus = 0
            num_cpus = min(4, hardware.total_cpus)
            batch_size = min(workload.max_batch_size, 128)
            throughput = batch_size * 10  # ~10 batches/sec on CPU
            latency = 100
            reasoning = "CPU-based embedding processing"
        
        # Calculate instances based on throughput requirement
        required_throughput = workload.expected_qps * workload.avg_input_length / batch_size
        instances = max(1, int(required_throughput / (throughput / batch_size)))
        
        return ResourceEstimation(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            instances=instances,
            memory_per_instance_gb=memory_required + 0.5,  # Model + overhead
            recommended_batch_size=batch_size,
            estimated_throughput=throughput,
            estimated_latency_ms=latency,
            reasoning=reasoning
        )
    
    def estimate_llm_resources(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile
    ) -> ResourceEstimation:
        """Estimate resources for LLM models"""
        memory_required = self.estimate_model_memory(model_profile)
        # try:
        #     memory_required = estimate_llm_infer_memory(model_profile)
        # except Exception as e:
        #     print(f"Error estimating LLM memory: {e}\nFalling back to default model memory estimation.")
        #     memory_required = self.estimate_model_memory(model_profile)
        
        # LLMs require GPU and substantial memory
        if hardware.total_gpus == 0:
            reasoning = "WARNING: LLM requires GPU but none available. Using minimal CPU config."
            return ResourceEstimation(
                num_gpus=0,
                num_cpus=hardware.total_cpus,
                instances=1,
                memory_per_instance_gb=memory_required,
                recommended_batch_size=1,
                estimated_throughput=0.1,
                estimated_latency_ms=10000,
                reasoning=reasoning
            )
        
        # Calculate required GPUs based on memory
        gpus_needed = max(1, int((memory_required * 1.5) / hardware.gpu_memory_gb) + 1)
        gpus_needed = min(gpus_needed, hardware.total_gpus)
        
        # Estimate throughput (very rough heuristics)
        tokens_per_second = 50 * gpus_needed  # ~50 tokens/sec per GPU
        avg_total_tokens = workload.avg_input_length + workload.avg_output_length
        requests_per_second = tokens_per_second / avg_total_tokens
        
        # Calculate instances
        instances = max(1, int(workload.expected_qps / requests_per_second) + 1)
        instances = min(instances, max(1, hardware.total_gpus // gpus_needed))
        
        # Batch size for throughput optimization
        batch_size = min(workload.max_batch_size, 256)
        latency = (avg_total_tokens / tokens_per_second) * 1000  # Convert to ms
        
        reasoning = (
            f"LLM requires {gpus_needed} GPU(s) per instance, "
            f"supporting ~{requests_per_second:.2f} req/s per instance"
        )
        
        return ResourceEstimation(
            num_gpus=gpus_needed,
            num_cpus=2,
            instances=instances,
            memory_per_instance_gb=memory_required * 1.5,
            recommended_batch_size=batch_size,
            estimated_throughput=requests_per_second * instances,
            estimated_latency_ms=latency,
            reasoning=reasoning
        )
    
    def estimate_reranker_resources(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile
    ) -> ResourceEstimation:
        """Estimate resources for reranker models"""
        memory_required = self.estimate_model_memory(model_profile)
        # try:
        #     memory_required = estimate_reranker_infer_memory(model_profile)
        # except Exception as e:
        #     print(f"Error estimating reranker memory: {e}\nFalling back to default model memory estimation.")
        #     memory_required = self.estimate_model_memory(model_profile)
        
        # Rerankers process query-document pairs
        if hardware.total_gpus > 0 and hardware.gpu_memory_gb >= memory_required:
            num_gpus = min(0.2, hardware.total_gpus)
            num_cpus = 1
            batch_size = min(workload.max_batch_size, 256)
            pairs_per_second = batch_size * 30  # ~30 batches/sec
            latency = 35
            reasoning = "GPU-accelerated reranking with fractional GPU"
        else:
            num_gpus = 0
            num_cpus = min(4, hardware.total_cpus)
            batch_size = min(workload.max_batch_size, 64)
            pairs_per_second = batch_size * 5
            latency = 200
            reasoning = "CPU-based reranking"
        
        instances = max(1, int(workload.expected_qps / (pairs_per_second / batch_size)))
        
        return ResourceEstimation(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            instances=instances,
            memory_per_instance_gb=memory_required + 0.5,
            recommended_batch_size=batch_size,
            estimated_throughput=pairs_per_second,
            estimated_latency_ms=latency,
            reasoning=reasoning
        )
    
    def estimate_vectordb_resources(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile,
        dataset_size: int = 100000
    ) -> ResourceEstimation:
        """Estimate resources for vector database"""
        # Vector DB is CPU and memory intensive
        # Estimate memory based on dataset size and vector dimensions
        vector_dim = model_profile.embedding_dim or 768
        bytes_per_vector = vector_dim * 4  # float32
        estimated_data_memory = (dataset_size * bytes_per_vector) / (1024**3)  # GB
        
        num_cpus = min(4, hardware.total_cpus)
        memory_per_instance = max(2.0, estimated_data_memory * 1.5)  # 50% overhead
        
        # Vector DB throughput depends on index type and dataset size
        searches_per_second = 1000 if dataset_size < 1000000 else 500
        
        instances = max(1, int(workload.expected_qps / searches_per_second))
        
        reasoning = (
            f"Vector DB for {dataset_size} vectors ({vector_dim}D), "
            f"~{estimated_data_memory:.2f}GB data"
        )
        
        return ResourceEstimation(
            num_gpus=0,
            num_cpus=num_cpus,
            instances=instances,
            memory_per_instance_gb=memory_per_instance,
            recommended_batch_size=workload.max_batch_size,
            estimated_throughput=searches_per_second,
            estimated_latency_ms=10,
            reasoning=reasoning
        )
    
    def estimate_resources(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile,
        **kwargs
    ) -> ResourceEstimation:
        """
        Estimate resources for any engine type.
        
        Args:
            model_profile: Model characteristics
            hardware: Available hardware specs
            workload: Expected workload profile
            **kwargs: Additional engine-specific parameters
        
        Returns:
            ResourceEstimation with recommended configuration
        """
        if model_profile.engine_type == EngineType.EMBEDDER:
            return self.estimate_embedder_resources(model_profile, hardware, workload)
        elif model_profile.engine_type == EngineType.LLM:
            return self.estimate_llm_resources(model_profile, hardware, workload)
        elif model_profile.engine_type == EngineType.RERANKER:
            return self.estimate_reranker_resources(model_profile, hardware, workload)
        elif model_profile.engine_type == EngineType.VECTOR_DB:
            dataset_size = kwargs.get("dataset_size", 100000)
            return self.estimate_vectordb_resources(
                model_profile, hardware, workload, dataset_size
            )
        else:
            # Default conservative estimation
            return ResourceEstimation(
                num_gpus=0,
                num_cpus=2,
                instances=1,
                memory_per_instance_gb=2.0,
                recommended_batch_size=32,
                estimated_throughput=10.0,
                estimated_latency_ms=100,
                reasoning="Default estimation for unknown engine type"
            )
    
    def create_engine_config(
        self,
        name: str,
        model_profile: ModelProfile,
        hardware: HardwareSpec,
        workload: WorkloadProfile,
        base_model_config: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[EngineConfig, ResourceEstimation]:
        """
        Create an optimized EngineConfig based on resource estimation.
        
        Args:
            name: Engine name
            model_profile: Model characteristics
            hardware: Available hardware specs
            workload: Expected workload profile
            base_model_config: Base model configuration to extend
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (EngineConfig, ResourceEstimation)
        """
        estimation = self.estimate_resources(
            model_profile, hardware, workload, **kwargs
        )
        
        # Prepare model config
        model_config = base_model_config.copy() if base_model_config else {}
        model_config.update({
            "max_batch_size": estimation.recommended_batch_size,
        })
        
        # Add model name if not present
        if "model_name" not in model_config:
            model_config["model_name"] = model_profile.model_name
        
        # Set device based on GPU availability
        if estimation.num_gpus > 0 and "device" not in model_config:
            model_config["device"] = "cuda"
        
        # Create latency profile
        latency_profile = {
            "timeout": max(60, int(estimation.estimated_latency_ms / 1000) * 3),
            "batch_wait": 0.1 if estimation.num_gpus > 0 else 0.05,
        }
        
        engine_config = EngineConfig(
            name=name,
            engine_type=model_profile.engine_type,
            num_gpus=estimation.num_gpus,
            num_cpus=estimation.num_cpus,
            instances=estimation.instances,
            resources={},
            model_config=model_config,
            latency_profile=latency_profile,
        )
        
        return engine_config, estimation
    
    def plan_multi_engine_config(
        self,
        engine_specs: List[Tuple[str, ModelProfile, Optional[Dict]]],
        hardware: HardwareSpec,
        workload: WorkloadProfile,
        **kwargs
    ) -> Tuple[List[EngineConfig], List[ResourceEstimation], Dict[str, Any]]:
        """
        Plan configurations for multiple engines with resource sharing.
        
        Args:
            engine_specs: List of (name, model_profile, base_model_config) tuples
            hardware: Available hardware specs
            workload: Expected workload profile
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (configs, estimations, resource_summary)
        """
        configs = []
        estimations = []
        total_gpus_used = 0.0
        total_cpus_used = 0
        total_memory_used = 0.0
        
        for name, model_profile, base_model_config in engine_specs:
            # Adjust hardware availability based on already allocated resources
            remaining_hardware = HardwareSpec(
                total_gpus=max(0, hardware.total_gpus - total_gpus_used),
                total_cpus=max(1, hardware.total_cpus - total_cpus_used),
                gpu_memory_gb=hardware.gpu_memory_gb,
                total_memory_gb=max(0, hardware.total_memory_gb - total_memory_used),
                gpu_type=hardware.gpu_type,
            )
            
            config, estimation = self.create_engine_config(
                name=name,
                model_profile=model_profile,
                hardware=remaining_hardware,
                workload=workload,
                base_model_config=base_model_config,
                **kwargs
            )
            
            configs.append(config)
            estimations.append(estimation)
            
            # Update used resources
            total_gpus_used += estimation.num_gpus * estimation.instances
            total_cpus_used += estimation.num_cpus * estimation.instances
            total_memory_used += estimation.memory_per_instance_gb * estimation.instances
        
        resource_summary = {
            "total_gpus_allocated": total_gpus_used,
            "total_cpus_allocated": total_cpus_used,
            "total_memory_allocated_gb": total_memory_used,
            "gpu_utilization": (total_gpus_used / hardware.total_gpus * 100) 
                if hardware.total_gpus > 0 else 0,
            "cpu_utilization": (total_cpus_used / hardware.total_cpus * 100)
                if hardware.total_cpus > 0 else 0,
            "feasible": (
                total_gpus_used <= hardware.total_gpus and
                total_cpus_used <= hardware.total_cpus and
                total_memory_used <= hardware.total_memory_gb
            ),
        }
        
        return configs, estimations, resource_summary


# Convenience functions

def create_config_planner() -> ConfigPlanner:
    """Create a new ConfigPlanner instance"""
    return ConfigPlanner()


def quick_estimate(
    model_name: str,
    engine_type: EngineType,
    hardware: HardwareSpec,
    workload: Optional[WorkloadProfile] = None
) -> ResourceEstimation:
    """
    Quick resource estimation with minimal parameters.
    
    Args:
        model_name: Name of the model
        engine_type: Type of engine
        hardware: Available hardware
        workload: Workload profile (uses defaults if None)
    
    Returns:
        ResourceEstimation
    """
    planner = ConfigPlanner()
    model_profile = ModelProfile(
        model_name=model_name,
        engine_type=engine_type,
    )
    workload = workload or WorkloadProfile()
    return planner.estimate_resources(model_profile, hardware, workload)
