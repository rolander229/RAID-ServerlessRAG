"""
Core Evaluation Engine for Cloud-Native Function Encapsulation Accuracy
(Theoretical Grounding: ICSE, FSE, ASE, TOSEM, TSE, EuroSys, ASPLOS, OSDI)

Performs AST-level inspection, specification comparison, SAM template validation,
and dynamic differential testing across 7 AI/DL algorithm benchmarks.
"""
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .metrics import AccuracyMetricCalculator, BenchmarkEvaluationResult, DimensionScore
    from .runtime_executor import RuntimeConsistencyExecutor
except ImportError:
    from evaluation.metrics import AccuracyMetricCalculator, BenchmarkEvaluationResult, DimensionScore
    from evaluation.runtime_executor import RuntimeConsistencyExecutor


class FunctionAstInspector:
    """Inspects Python FaaS handlers using Abstract Syntax Tree (AST)."""

    @staticmethod
    def inspect_handler_file(file_path: Path) -> Dict[str, Any]:
        result = {
            "has_lambda_handler": False,
            "has_event_parsing": False,
            "has_structured_response": False,
            "has_error_handling": False,
            "has_sqs_publish": False,
            "has_shared_layer_import": False,
            "compliance_score": 0.0,
        }

        if not file_path.exists():
            return result

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception:
            return result

        # Check handler function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("lambda_handler", "handler"):
                result["has_lambda_handler"] = True
                args = [arg.arg for arg in node.args.args]
                if "event" in args and "context" in args:
                    result["has_lambda_handler"] = True

            if isinstance(node, ast.ImportFrom):
                if node.module in ("shared_utils", "rag_service", "recsys_service", "dnn_service", "traffic_service", "driving_service", "video_service", "image_service"):
                    result["has_shared_layer_import"] = True

        # Text/AST heuristic checks
        content_lower = content.lower()
        if any(k in content_lower for k in ["pathparameters", "querystringparameters", "parse_json_body", "json.loads", "event.get('body')"]):
            result["has_event_parsing"] = True

        if any(k in content_lower for k in ["statuscode", "format_response", "format_api_response", "batchitemfailures"]):
            result["has_structured_response"] = True

        if any(k in content_lower for k in ["try:", "handle_api_errors", "faas_handler_wrapper", "except exception"]):
            result["has_error_handling"] = True

        if any(k in content_lower for k in ["send_message", "sqs.client", "document_ingestion_queue", "queue_url"]):
            result["has_sqs_publish"] = True

        # Calculate template compliance score for this function
        points = 0
        total = 4
        if result["has_lambda_handler"]:
            points += 1.0
        if result["has_event_parsing"]:
            points += 1.0
        if result["has_structured_response"]:
            points += 1.0
        if result["has_error_handling"]:
            points += 1.0

        result["compliance_score"] = points / total
        return result


class EncapsulationEvaluator:
    """Evaluates a single benchmark directory or all benchmarks."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.calculator = AccuracyMetricCalculator()

    def load_benchmark_data(self, benchmark_dir: Path) -> Dict[str, Any]:
        """Load expert ground truth and tool migration data for a benchmark."""
        gt_dir = benchmark_dir / "ground_truth_expert"
        tool_dir = benchmark_dir / "tool_migrated"

        # Load specifications
        gt_spec_file = gt_dir / "specification.json"
        tool_spec_file = tool_dir / "specification.json"

        gt_spec = {}
        if gt_spec_file.exists():
            gt_spec = json.loads(gt_spec_file.read_text(encoding="utf-8"))

        tool_spec = {}
        if tool_spec_file.exists():
            tool_spec = json.loads(tool_spec_file.read_text(encoding="utf-8"))
        elif (self.workspace_root / "pipeline-artifacts" / "blueprint.json").exists() and benchmark_dir.name == "rag_anything":
            # Extract from root blueprint.json for rag_anything
            bp = json.loads((self.workspace_root / "pipeline-artifacts" / "blueprint.json").read_text(encoding="utf-8"))
            funcs = []
            for lf in bp.get("lambda_functions", []):
                ep = lf.get("entry_points", [])
                method, path = (None, None)
                if ep:
                    parts = ep[0].split(" ", 1)
                    method = parts[0]
                    path = parts[1] if len(parts) > 1 else ""
                funcs.append({
                    "name": lf["name"],
                    "method": method,
                    "path": path,
                    "trigger": "api_gateway" if lf.get("trigger_type") == "api" else lf.get("trigger_type"),
                    "type": "orchestrator" if lf.get("publishes_to") else "worker",
                    "async_publishes": [p.get("target") for p in lf.get("publishes_to", [])]
                })
            tool_spec = {
                "application_name": bp.get("metadata", {}).get("application_name", "rag-anything-service"),
                "domain_type": "Large Language Model & Multimodal RAG Pipeline",
                "functions": funcs,
                "infrastructure": {
                    "api_gateway": True,
                    "cognito": bp.get("cognito", {}).get("enabled", True),
                    "sqs_queues": [q.get("logical_name") for q in bp.get("sqs_queues", [])],
                    "dynamodb_tables": [t.get("logical_name") for t in bp.get("dynamodb_tables", [])],
                    "s3_buckets": [b.get("logical_name") for b in bp.get("s3_buckets", [])]
                }
            }

        return {"gt_dir": gt_dir, "tool_dir": tool_dir, "gt_spec": gt_spec, "tool_spec": tool_spec}

    def evaluate_benchmark(self, benchmark_path: Path) -> BenchmarkEvaluationResult:
        """Run complete 6-dimension evaluation on a given benchmark."""
        data = self.load_benchmark_data(benchmark_path)
        gt_spec = data["gt_spec"]
        tool_spec = data["tool_spec"]

        app_name = gt_spec.get("application_name", benchmark_path.name)
        domain_type = gt_spec.get("domain_type", "AI/DL Algorithm Pipeline")
        tool_funcs = {f["name"] for f in tool_spec.get("functions", [])}

        # 1. API Interface Coverage (S_api) - Weight 0.20
        gt_apis = set()
        for f in gt_spec.get("functions", []):
            if f.get("method") and f.get("path"):
                gt_apis.add((f["method"], f["path"]))

        tool_apis = set()
        for f in tool_spec.get("functions", []):
            if f.get("method") and f.get("path"):
                tool_apis.add((f["method"], f["path"]))

        s_api = self.calculator.calculate_api_coverage(tool_apis, gt_apis)

        # 2. Function Granularity & Boundary (S_granularity) - Weight 0.20
        # Prefer genuine partition-boundary comparison against the independent expert
        # baseline; fall back to name-set Jaccard for legacy benchmarks without partitions.
        tool_partition = tool_spec.get("tool_partition")
        expert_partition = gt_spec.get("expert_partition")
        if tool_partition and expert_partition:
            s_granularity = self.calculator.calculate_granularity_boundaries(tool_partition, expert_partition)
        else:
            gt_funcs = {f["name"] for f in gt_spec.get("functions", [])}
            s_granularity = self.calculator.calculate_granularity(tool_funcs, gt_funcs)

        # 3. Template Compliance via AST (S_template) - Weight 0.15
        compliance_scores = []
        compliance_details = []
        lambdas_root = data["tool_dir"] / "lambdas"
        if not lambdas_root.exists() and (self.workspace_root / "after" / "lambdas").exists() and benchmark_path.name == "rag_anything":
            lambdas_root = self.workspace_root / "after" / "lambdas"

        for f_name in tool_funcs:
            handler_candidates = list(lambdas_root.rglob(f"{f_name}/handler.py")) + list(lambdas_root.rglob("handler.py"))
            matched_candidate = None
            for c in handler_candidates:
                if c.parent.name == f_name or f_name in str(c):
                    matched_candidate = c
                    break

            if matched_candidate:
                ast_res = FunctionAstInspector.inspect_handler_file(matched_candidate)
            else:
                # Missing handler code is a hard failure, never a default pass.
                ast_res = {"compliance_score": 0.0, "has_lambda_handler": False, "has_event_parsing": False, "has_structured_response": False, "has_error_handling": False, "handler_found": False}

            compliance_scores.append(ast_res["compliance_score"])
            compliance_details.append({"function": f_name, "score": ast_res["compliance_score"]})

        s_template = self.calculator.calculate_template_compliance(compliance_scores, compliance_details)

        # 4. Trigger & Async Decoupling (S_trigger) - Weight 0.15
        gt_triggers = {f["name"]: f.get("trigger") for f in gt_spec.get("functions", [])}
        tool_triggers = {f["name"]: f.get("trigger") for f in tool_spec.get("functions", [])}
        s_trigger = self.calculator.calculate_trigger_decoupling_categories(tool_triggers, gt_triggers)

        # 5. Infrastructure Decoupling (S_infra) - Weight 0.10
        gt_infra = gt_spec.get("infrastructure", {})
        tool_infra = tool_spec.get("infrastructure", {})
        s_infra = self.calculator.calculate_infra_alignment(tool_infra, gt_infra)

        # 6. Differential Runtime Execution Consistency (S_runtime) - Weight 0.20
        diff_res = RuntimeConsistencyExecutor.run_benchmark_differential_test(benchmark_path)
        s_runtime = self.calculator.calculate_runtime_consistency(diff_res)

        # Total 6-dimension accuracy (raw measured scores only — no calibration)
        dimensions = [s_api, s_granularity, s_template, s_trigger, s_infra, s_runtime]
        evaluated_weight = sum(d.weight for d in dimensions if d.details.get("evaluated", True))
        raw_weighted_sum = sum(d.weighted_score for d in dimensions if d.details.get("evaluated", True))
        # Full-scale score: unevaluated dimensions count as 0
        overall_accuracy = sum(d.weighted_score for d in dimensions) * 100.0
        # Evaluated-scale score: renormalized over dimensions that were actually measured
        overall_accuracy_evaluated = (raw_weighted_sum / evaluated_weight * 100.0) if evaluated_weight > 0 else 0.0
        not_evaluated = [d.dimension_name for d in dimensions if not d.details.get("evaluated", True)]
        passed = overall_accuracy >= 70.0

        return BenchmarkEvaluationResult(
            benchmark_id=benchmark_path.name,
            benchmark_name=app_name,
            domain_type=domain_type,
            dimensions=dimensions,
            overall_accuracy=round(overall_accuracy, 2),
            passed=passed,
            target_threshold=70.0,
            summary_notes=(
                f"Evaluated against independent expert baseline and differential runtime tests. "
                f"Overall Accuracy: {overall_accuracy:.2f}% (evaluated-scale: {overall_accuracy_evaluated:.2f}%; "
                f"not evaluated: {', '.join(not_evaluated) if not_evaluated else 'none'}). "
                f"Status: {'PASSED' if passed else 'FAILED'}"
            )
        )

    def evaluate_all(self) -> List[BenchmarkEvaluationResult]:
        """Evaluate all 7 benchmarks under benchmarks/ directory."""
        benchmarks_dir = self.workspace_root / "benchmarks"
        results = []
        if not benchmarks_dir.exists():
            return results

        for child in sorted(benchmarks_dir.iterdir()):
            if child.is_dir() and (child / "ground_truth_expert").exists():
                results.append(self.evaluate_benchmark(child))
        return results
