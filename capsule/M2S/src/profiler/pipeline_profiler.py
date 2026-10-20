"""
Non-Intrusive Runtime Pipeline Profiler (Profiling Engine)
Captures execution latency, peak memory/VRAM footprint, and inter-operator tensor payload sizes.
"""
import ast
import gc
import json
import logging
import math
import os
import pickle
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineProfiler")


@dataclass
class OperatorProfile:
    name: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    peak_cpu_memory_mb: float
    peak_gpu_memory_mb: float
    compute_device: str  # GPU_Inference, CPU_Heavy, CPU_Light
    call_count: int
    is_async_candidate: bool


@dataclass
class EdgeDataPassing:
    source_operator: str
    target_operator: str
    payload_size_bytes: int
    payload_size_mb: float
    tensor_shape: Optional[List[int]] = None
    tensor_dtype: Optional[str] = None
    is_in_memory_direct: bool = True


@dataclass
class PipelineProfilingReport:
    pipeline_name: str
    benchmark_id: str
    total_operators: int
    operators: Dict[str, Dict[str, Any]]
    data_passing_matrix: List[Dict[str, Any]]
    total_pipeline_latency_ms: float
    total_inter_operator_data_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorPayloadInspector:
    """Calculates true byte sizes of tensors and arbitrary Python objects."""

    @staticmethod
    def inspect_size(obj: Any) -> Tuple[int, Optional[List[int]], Optional[str]]:
        if HAS_TORCH and isinstance(obj, torch.Tensor):
            num_bytes = obj.element_size() * obj.nelement()
            return num_bytes, list(obj.shape), str(obj.dtype)
        elif isinstance(obj, np.ndarray):
            return obj.nbytes, list(obj.shape), str(obj.dtype)
        elif isinstance(obj, (bytes, bytearray)):
            return len(obj), [len(obj)], "bytes"
        elif isinstance(obj, str):
            encoded = obj.encode("utf-8")
            return len(encoded), [len(obj)], "string"
        elif isinstance(obj, dict):
            try:
                raw_bytes = json.dumps(obj).encode("utf-8")
                return len(raw_bytes), [len(obj)], "json_dict"
            except Exception:
                return len(pickle.dumps(obj)), [len(obj)], "pickle_dict"
        elif isinstance(obj, (list, tuple)):
            # Check if list of tensors
            if obj and HAS_TORCH and isinstance(obj[0], torch.Tensor):
                total_bytes = sum(t.element_size() * t.nelement() for t in obj if isinstance(t, torch.Tensor))
                return total_bytes, [len(obj)] + list(obj[0].shape), f"list[{obj[0].dtype}]"
            try:
                raw_bytes = pickle.dumps(obj)
                return len(raw_bytes), [len(obj)], "list"
            except Exception:
                return sys.getsizeof(obj), [len(obj)], "object_ref"
        else:
            try:
                return len(pickle.dumps(obj)), None, type(obj).__name__
            except Exception:
                return sys.getsizeof(obj), None, type(obj).__name__


class PipelineProfiler:
    """Non-intrusive probe harness to profile pipeline operators and tensor data passing."""

    def __init__(self, pipeline_name: str, benchmark_id: str):
        self.pipeline_name = pipeline_name
        self.benchmark_id = benchmark_id
        self.operator_timings: Dict[str, List[float]] = {}
        self.operator_memory: Dict[str, List[float]] = {}
        self.operator_gpu: Dict[str, List[float]] = {}
        self.edge_transfers: List[EdgeDataPassing] = []
        self.call_chains: List[Tuple[str, str, Any]] = []

    def profile_operator(self, op_name: str, func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
        """Wrap and execute a single operator function, capturing latency and memory."""
        gc.collect()
        tracemalloc.start()
        
        gpu_before = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated() / (1024 * 1024)

        t_start = time.perf_counter_ns()
        try:
            result = func(*args, **kwargs)
        finally:
            t_end = time.perf_counter_ns()

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms = (t_end - t_start) / 1_000_000.0
        cpu_mem_mb = peak_mem / (1024 * 1024)

        gpu_mem_mb = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_mem_mb = max(0.0, gpu_peak - gpu_before)

        self.operator_timings.setdefault(op_name, []).append(latency_ms)
        self.operator_memory.setdefault(op_name, []).append(cpu_mem_mb)
        self.operator_gpu.setdefault(op_name, []).append(gpu_mem_mb)

        return result, latency_ms, cpu_mem_mb

    def record_inter_operator_transfer(
        self,
        source_op: str,
        target_op: str,
        data_payload: Any
    ) -> EdgeDataPassing:
        """Inspect and record the size of tensors passed between operators."""
        size_bytes, shape, dtype = TensorPayloadInspector.inspect_size(data_payload)
        size_mb = size_bytes / (1024.0 * 1024.0)

        edge = EdgeDataPassing(
            source_operator=source_op,
            target_operator=target_op,
            payload_size_bytes=size_bytes,
            payload_size_mb=round(size_mb, 4),
            tensor_shape=shape,
            tensor_dtype=dtype,
            is_in_memory_direct=True
        )
        self.edge_transfers.append(edge)
        return edge

    def generate_report(self) -> PipelineProfilingReport:
        """Aggregate statistical profile metrics into a standardized report."""
        operators_dict: Dict[str, Dict[str, Any]] = {}
        total_lat = 0.0

        for op_name, timings in self.operator_timings.items():
            if not timings:
                continue
            # Filter outliers if multiple runs
            sorted_t = sorted(timings)
            avg_t = float(np.mean(sorted_t))
            min_t = float(np.min(sorted_t))
            max_t = float(np.max(sorted_t))
            p95_t = float(np.percentile(sorted_t, 95))
            total_lat += avg_t

            mem_list = self.operator_memory.get(op_name, [0.0])
            peak_cpu = float(np.max(mem_list)) if mem_list else 0.0

            gpu_list = self.operator_gpu.get(op_name, [0.0])
            peak_gpu = float(np.max(gpu_list)) if gpu_list else 0.0

            # Determine compute device
            if peak_gpu > 0.0:
                dev = "GPU_Inference"
            elif avg_t > 500.0 or peak_cpu > 512.0:
                dev = "CPU_Heavy"
            else:
                dev = "CPU_Light"

            is_async = avg_t >= 3000.0 or (dev == "CPU_Heavy" and avg_t >= 1500.0)

            op_prof = OperatorProfile(
                name=op_name,
                avg_latency_ms=round(avg_t, 2),
                min_latency_ms=round(min_t, 2),
                max_latency_ms=round(max_t, 2),
                p95_latency_ms=round(p95_t, 2),
                peak_cpu_memory_mb=round(peak_cpu, 2),
                peak_gpu_memory_mb=round(peak_gpu, 2),
                compute_device=dev,
                call_count=len(timings),
                is_async_candidate=is_async
            )
            operators_dict[op_name] = asdict(op_prof)

        data_matrix = []
        total_data_mb = 0.0
        for edge in self.edge_transfers:
            data_matrix.append(asdict(edge))
            total_data_mb += edge.payload_size_mb

        return PipelineProfilingReport(
            pipeline_name=self.pipeline_name,
            benchmark_id=self.benchmark_id,
            total_operators=len(operators_dict),
            operators=operators_dict,
            data_passing_matrix=data_matrix,
            total_pipeline_latency_ms=round(total_lat, 2),
            total_inter_operator_data_mb=round(total_data_mb, 4),
            metadata={
                "profiler_version": "2.0-adaptive",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "torch_available": HAS_TORCH,
                "cuda_available": (HAS_TORCH and torch.cuda.is_available())
            }
        )

    def save_report(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_report()
        output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved pipeline profiling report to: %s", output_path)
        return output_path
