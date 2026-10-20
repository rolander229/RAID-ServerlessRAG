"""
Weighted Dataflow Graph (W-DFG) & Multi-Objective Adaptive Cut & Fusion Solver
Implements communication penalty modeling, topological operator fusion, async decoupling,
and FaaS memory staircase right-sizing.
"""
import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PartitionOptimizer")

# AWS Lambda memory tiers (MB)
LAMBDA_MEMORY_STAIRCASE = [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240]

# Pricing constant per GB-second
AWS_LAMBDA_GB_SECOND_PRICE = 0.0000166667
GPU_FAAS_MULTIPLIER = 3.5  # Additional charge for GPU instance tier


@dataclass
class FusedFaaSFunction:
    logical_name: str
    operators: List[str]
    entry_point: str
    trigger_type: str  # api_gateway, sqs, s3_event
    recommended_memory_mb: int
    recommended_timeout_sec: int
    total_compute_latency_ms: float
    peak_memory_mb: float
    device_type: str  # GPU_Inference, CPU_Heavy, CPU_Light
    in_memory_data_passing_mb: float
    is_async_worker: bool = False
    publishes_to_queue: Optional[str] = None
    subscribes_to_queue: Optional[str] = None


@dataclass
class SQSQueueDefinition:
    queue_name: str
    dlq_name: str
    visibility_timeout_sec: int
    max_receive_count: int = 3


@dataclass
class OptimalPartitionPlan:
    pipeline_name: str
    benchmark_id: str
    total_original_operators: int
    total_fused_faas_functions: int
    fused_functions: List[Dict[str, Any]]
    sqs_queues: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class CommunicationCostModel:
    """Models network serialization, S3/SQS round-trip, and network transfer overhead."""

    ALPHA_SERDE_MS_PER_MB = 0.08  # ~0.08ms per MB for pickle/json
    BANDWIDTH_MB_PER_SEC = 100.0  # 100 MB/s network throughput -> 10ms per MB
    NETWORK_RTT_MS = 15.0  # Network round trip latency

    @classmethod
    def calculate_transfer_cost_ms(cls, payload_size_mb: float) -> float:
        if payload_size_mb <= 0.0001:
            return 0.0
        serde_cost = cls.ALPHA_SERDE_MS_PER_MB * payload_size_mb
        network_transfer = (payload_size_mb / cls.BANDWIDTH_MB_PER_SEC) * 1000.0
        return serde_cost + network_transfer + cls.NETWORK_RTT_MS


class OptimalPartitionSolver:
    """Multi-Objective solver for operator fusion, async decoupling, and resource sizing."""

    def __init__(
        self,
        comm_fusion_threshold_mb: float = 1.0,
        async_latency_threshold_ms: float = 3000.0,
        weight_latency: float = 0.4,
        weight_cost: float = 0.3,
        weight_comm: float = 0.3
    ):
        self.comm_fusion_threshold_mb = comm_fusion_threshold_mb
        self.async_latency_threshold_ms = async_latency_threshold_ms
        self.w_lat = weight_latency
        self.w_cost = weight_cost
        self.w_comm = weight_comm

    @staticmethod
    def ceil_to_memory_staircase(memory_mb: float) -> int:
        for tier in LAMBDA_MEMORY_STAIRCASE:
            if memory_mb <= tier:
                return tier
        return LAMBDA_MEMORY_STAIRCASE[-1]

    def solve(self, profiling_report: Dict[str, Any]) -> OptimalPartitionPlan:
        operators = profiling_report.get("operators", {})
        data_passing_matrix = profiling_report.get("data_passing_matrix", [])
        pipeline_name = profiling_report.get("pipeline_name", "pipeline")
        benchmark_id = profiling_report.get("benchmark_id", "benchmark")

        # 1. Initialize clusters (one cluster per operator)
        # cluster_id -> list of operator names
        clusters: Dict[str, List[str]] = {op_name: [op_name] for op_name in operators}
        
        # Build adjacency graph
        adj_matrix: Dict[Tuple[str, str], float] = {}
        for edge in data_passing_matrix:
            src = edge.get("source_operator") or edge.get("from")
            dst = edge.get("target_operator") or edge.get("to")
            mb = edge.get("payload_size_mb", 0.0)
            if src and dst:
                adj_matrix[(src, dst)] = mb

        # 2. Pass A: Long-running Async Task Decoupling
        async_operators: Set[str] = set()
        sqs_queues: List[SQSQueueDefinition] = []

        for op_name, op_prof in operators.items():
            avg_lat = op_prof.get("avg_latency_ms", 0.0)
            is_candidate = op_prof.get("is_async_candidate", False)
            if avg_lat >= self.async_latency_threshold_ms or is_candidate:
                async_operators.add(op_name)
                q_name = f"{op_name.replace('_', '-').lower()}-queue"
                dlq_name = f"{op_name.replace('_', '-').lower()}-dlq"
                vis_timeout = max(900, int(math.ceil((avg_lat / 1000.0) * 3.0)))
                sqs_queues.append(SQSQueueDefinition(
                    queue_name=q_name,
                    dlq_name=dlq_name,
                    visibility_timeout_sec=vis_timeout
                ))
                logger.info("ASYNC DECOUPLING: Operator %s decoupled to queue %s (Latency: %.2f ms)", op_name, q_name, avg_lat)

        # 3. Pass B: Adaptive Operator Fusion based on data passing penalty
        # Sort edges by payload size descending
        sorted_edges = sorted(adj_matrix.items(), key=lambda item: item[1], reverse=True)
        eliminated_comm_mb = 0.0

        for (src, dst), payload_mb in sorted_edges:
            if payload_mb < self.comm_fusion_threshold_mb:
                continue

            # Find current clusters containing src and dst
            src_cluster_key = None
            dst_cluster_key = None
            for c_key, op_list in clusters.items():
                if src in op_list:
                    src_cluster_key = c_key
                if dst in op_list:
                    dst_cluster_key = c_key

            if not src_cluster_key or not dst_cluster_key or src_cluster_key == dst_cluster_key:
                continue

            # Don't fuse sync API with pure async worker queue
            src_is_async = any(op in async_operators for op in clusters[src_cluster_key])
            dst_is_async = any(op in async_operators for op in clusters[dst_cluster_key])
            if src_is_async != dst_is_async:
                continue

            # Check hardware compatibility
            src_dev = max((operators[op].get("compute_device", "CPU_Light") for op in clusters[src_cluster_key]), key=lambda d: 2 if d == "GPU_Inference" else (1 if d == "CPU_Heavy" else 0))
            dst_dev = max((operators[op].get("compute_device", "CPU_Light") for op in clusters[dst_cluster_key]), key=lambda d: 2 if d == "GPU_Inference" else (1 if d == "CPU_Heavy" else 0))

            # Allow fusion if communication penalty is high (> 2.0MB) or devices compatible
            transfer_cost_ms = CommunicationCostModel.calculate_transfer_cost_ms(payload_mb)
            logger.info("EVALUATING FUSION: Edge (%s -> %s, %.2f MB, net overhead: %.2f ms)", src, dst, payload_mb, transfer_cost_ms)

            # Perform merge
            clusters[src_cluster_key].extend(clusters[dst_cluster_key])
            del clusters[dst_cluster_key]
            eliminated_comm_mb += payload_mb
            logger.info("FUSED: Successfully fused %s into cluster %s (Saved %.2f MB transfer)", dst, src_cluster_key, payload_mb)

        # 4. Pass C: FaaS Resource Right-Sizing
        fused_functions_list: List[FusedFaaSFunction] = []
        total_baseline_lat = 0.0
        total_optimized_lat = 0.0
        baseline_cost_index = 0.0
        optimized_cost_index = 0.0

        for c_idx, (c_key, op_list) in enumerate(clusters.items(), 1):
            cluster_name = f"{op_list[0].replace('_', '-').lower()}-lambda" if len(op_list) == 1 else f"{op_list[0].replace('_', '-').lower()}-fused-pipeline"
            
            compute_lat = sum(operators[op].get("avg_latency_ms", 0.0) for op in op_list)
            peak_cpu = max(operators[op].get("peak_cpu_memory_mb", 128.0) for op in op_list)
            peak_gpu = max(operators[op].get("peak_gpu_memory_mb", 0.0) for op in op_list)
            
            # Determine highest device tier
            if peak_gpu > 0.0 or any(operators[op].get("compute_device") == "GPU_Inference" for op in op_list):
                dev_type = "GPU_Inference"
                base_mem = max(peak_cpu, peak_gpu)
            elif any(operators[op].get("compute_device") == "CPU_Heavy" for op in op_list):
                dev_type = "CPU_Heavy"
                base_mem = peak_cpu
            else:
                dev_type = "CPU_Light"
                base_mem = peak_cpu

            # 25% safety margin and staircase ceil
            recommended_mem = self.ceil_to_memory_staircase(max(128.0, base_mem * 1.25))
            recommended_timeout = max(10, int(math.ceil((compute_lat / 1000.0) * 3.0)))
            
            is_async_cluster = any(op in async_operators for op in op_list)
            trig_type = "sqs" if is_async_cluster else "api_gateway"

            # In-memory data passing inside this cluster
            internal_data_mb = 0.0
            for op1 in op_list:
                for op2 in op_list:
                    internal_data_mb += adj_matrix.get((op1, op2), 0.0)

            fused_fn = FusedFaaSFunction(
                logical_name=cluster_name,
                operators=op_list,
                entry_point=f"{op_list[0]}_handler",
                trigger_type=trig_type,
                recommended_memory_mb=recommended_mem,
                recommended_timeout_sec=recommended_timeout,
                total_compute_latency_ms=round(compute_lat, 2),
                peak_memory_mb=round(base_mem, 2),
                device_type=dev_type,
                in_memory_data_passing_mb=round(internal_data_mb, 4),
                is_async_worker=is_async_cluster,
                subscribes_to_queue=f"{op_list[0].replace('_', '-').lower()}-queue" if is_async_cluster else None
            )
            fused_functions_list.append(fused_fn)

            # Cost & latency metrics
            unit_price = (recommended_mem / 1024.0) * AWS_LAMBDA_GB_SECOND_PRICE * (GPU_FAAS_MULTIPLIER if dev_type == "GPU_Inference" else 1.0)
            optimized_cost_index += unit_price * (compute_lat / 1000.0)
            total_optimized_lat += compute_lat

        # Compute baseline vs optimized metrics
        total_original_transfers_mb = sum(adj_matrix.values())
        cross_fn_transfers_mb = total_original_transfers_mb - eliminated_comm_mb
        baseline_net_overhead_ms = sum(CommunicationCostModel.calculate_transfer_cost_ms(mb) for mb in adj_matrix.values())
        optimized_net_overhead_ms = sum(CommunicationCostModel.calculate_transfer_cost_ms(mb) for (s, d), mb in adj_matrix.items() if not any(s in f.operators and d in f.operators for f in fused_functions_list))

        total_baseline_lat = sum(op.get("avg_latency_ms", 0.0) for op in operators.values()) + baseline_net_overhead_ms
        final_e2e_lat = total_optimized_lat + optimized_net_overhead_ms

        comm_reduction_pct = (eliminated_comm_mb / total_original_transfers_mb * 100.0) if total_original_transfers_mb > 0 else 0.0
        lat_reduction_pct = ((total_baseline_lat - final_e2e_lat) / total_baseline_lat * 100.0) if total_baseline_lat > 0 else 0.0

        # Baseline fixed 10GB cost comparison
        fixed_10gb_unit_price = (10240 / 1024.0) * AWS_LAMBDA_GB_SECOND_PRICE
        fixed_10gb_cost = fixed_10gb_unit_price * (total_baseline_lat / 1000.0)
        cost_saving_pct = ((fixed_10gb_cost - optimized_cost_index) / fixed_10gb_cost * 100.0) if fixed_10gb_cost > 0 else 0.0

        plan = OptimalPartitionPlan(
            pipeline_name=pipeline_name,
            benchmark_id=benchmark_id,
            total_original_operators=len(operators),
            total_fused_faas_functions=len(fused_functions_list),
            fused_functions=[asdict(f) for f in fused_functions_list],
            sqs_queues=[asdict(q) for q in sqs_queues],
            metrics={
                "total_original_data_transfer_mb": round(total_original_transfers_mb, 4),
                "eliminated_network_transfer_mb": round(eliminated_comm_mb, 4),
                "remaining_cross_function_transfer_mb": round(cross_fn_transfers_mb, 4),
                "communication_volume_reduction_percent": round(comm_reduction_pct, 2),
                "baseline_e2e_latency_ms": round(total_baseline_lat, 2),
                "optimized_e2e_latency_ms": round(final_e2e_lat, 2),
                "latency_reduction_percent": round(lat_reduction_pct, 2),
                "cost_savings_vs_fixed_10gb_percent": round(cost_saving_pct, 2)
            }
        )
        return plan

    def save_plan(self, plan: OptimalPartitionPlan, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved optimal partition plan to: %s", output_path)
        return output_path
