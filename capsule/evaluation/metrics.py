"""
Formal Evaluation Metrics for AI/DL Cloud-Native Function Encapsulation Accuracy
(Scholarly Grounding: ICSE '21, ESEC/FSE '21, IEEE TSE '07/'16, ACM TOSEM '22, EuroSys '26, ASPLOS '21, OSDI '20)

Implements a 6-dimensional quantitative evaluation model comparing automated tool migration
results against expert manual ground-truth baselines and executing differential semantic tests.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DimensionScore:
    dimension_name: str
    weight: float
    raw_score: float  # 0.0 ~ 1.0
    weighted_score: float
    description: str
    literature_reference: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkEvaluationResult:
    benchmark_id: str
    benchmark_name: str
    domain_type: str
    dimensions: List[DimensionScore]
    overall_accuracy: float  # Percentage: 0.0 ~ 100.0%
    passed: bool
    target_threshold: float = 70.0  # Required threshold >= 70%
    summary_notes: str = ""


class AccuracyMetricCalculator:
    """
    Calculates the 6-dimensional Function Encapsulation Accuracy (A_encap):
      1. S_api: API Interface Coverage & HTTP Contract Match (weight: 0.20, ref: CARGO ASE'21, Mono2Micro FSE'21)
      2. S_granularity: Function Decomposition & Boundary Alignment (weight: 0.20, ref: MoJoFM IEEE TSE'07)
      3. S_template: FaaS Template Schema & Signature Compliance (weight: 0.15, ref: ToLambda IWoR'19)
      4. S_trigger: Trigger Type & Async Decoupling Alignment (weight: 0.15, ref: GRouter EuroSys'26, Nightcore ASPLOS'21)
      5. S_infra: Shared Layer & Infrastructure Decoupling Alignment (weight: 0.10, ref: Crucial EuroSys'19, TOSEM'22)
      6. S_runtime: Runtime Execution & Semantic Output Equivalence (weight: 0.20, ref: Differential Testing IEEE TSE'16)
    """

    WEIGHTS = {
        "api_coverage": 0.20,
        "granularity_boundary": 0.20,
        "template_compliance": 0.15,
        "trigger_decoupling": 0.15,
        "infra_alignment": 0.10,
        "runtime_consistency": 0.20,
    }

    @staticmethod
    def calculate_api_coverage(
        tool_apis: Set[Tuple[str, str]], expert_apis: Set[Tuple[str, str]]
    ) -> DimensionScore:
        """Calculate Precision, Recall, and F1-score for API endpoint coverage (CARGO / Mono2Micro)."""
        if not expert_apis:
            score = 1.0 if not tool_apis else 0.0
            return DimensionScore(
                dimension_name="API Interface Coverage & Mapping (S_api)",
                weight=AccuracyMetricCalculator.WEIGHTS["api_coverage"],
                raw_score=score,
                weighted_score=score * AccuracyMetricCalculator.WEIGHTS["api_coverage"],
                description="Coverage and contract alignment of monolith HTTP endpoints",
                literature_reference="CARGO (IEEE/ACM ASE '21) & Mono2Micro (ESEC/FSE '21)",
                details={"precision": score, "recall": score, "f1": score}
            )

        intersection = tool_apis.intersection(expert_apis)
        precision = len(intersection) / len(tool_apis) if tool_apis else 0.0
        recall = len(intersection) / len(expert_apis)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return DimensionScore(
            dimension_name="API Interface Coverage & Mapping (S_api)",
            weight=AccuracyMetricCalculator.WEIGHTS["api_coverage"],
            raw_score=f1,
            weighted_score=f1 * AccuracyMetricCalculator.WEIGHTS["api_coverage"],
            description="Coverage and contract alignment of monolith HTTP endpoints",
            literature_reference="CARGO (IEEE/ACM ASE '21) & Mono2Micro (ESEC/FSE '21)",
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "matched_apis": sorted([f"{m} {p}" for m, p in intersection]),
                "missing_apis": sorted([f"{m} {p}" for m, p in expert_apis - tool_apis]),
                "extra_apis": sorted([f"{m} {p}" for m, p in tool_apis - expert_apis])
            }
        )

    @staticmethod
    def calculate_granularity(
        tool_functions: Set[str], expert_functions: Set[str]
    ) -> DimensionScore:
        """Calculate Jaccard similarity and boundary alignment of decomposed functions (MoJoFM / Cluster Accuracy)."""
        if not expert_functions and not tool_functions:
            jaccard = 1.0
        elif not expert_functions or not tool_functions:
            jaccard = 0.0
        else:
            intersection = tool_functions.intersection(expert_functions)
            union = tool_functions.union(expert_functions)
            jaccard = len(intersection) / len(union)

        return DimensionScore(
            dimension_name="Function Decomposition Granularity (S_granularity)",
            weight=AccuracyMetricCalculator.WEIGHTS["granularity_boundary"],
            raw_score=jaccard,
            weighted_score=jaccard * AccuracyMetricCalculator.WEIGHTS["granularity_boundary"],
            description="Alignment of function decomposition boundaries and single responsibility",
            literature_reference="MoJoFM Architecture Metric (IEEE TSE '07, Maqbool & Babar)",
            details={
                "jaccard_similarity": round(jaccard, 4),
                "matched_functions": sorted(list(tool_functions.intersection(expert_functions))),
                "missing_functions": sorted(list(expert_functions - tool_functions)),
                "extra_functions": sorted(list(tool_functions - expert_functions)),
                "total_expert_functions": len(expert_functions),
                "total_tool_functions": len(tool_functions)
            }
        )

    @staticmethod
    def calculate_granularity_boundaries(
        tool_partition: List[List[str]], expert_partition: List[List[str]]
    ) -> DimensionScore:
        """Partition-boundary alignment via pairwise co-membership Jaccard (MoJoFM-style, IEEE TSE '07)."""
        try:
            from .expert_baseline import partition_similarity
        except ImportError:
            from evaluation.expert_baseline import partition_similarity
        score, details = partition_similarity(tool_partition, expert_partition)
        return DimensionScore(
            dimension_name="Function Decomposition Granularity (S_granularity)",
            weight=AccuracyMetricCalculator.WEIGHTS["granularity_boundary"],
            raw_score=score,
            weighted_score=score * AccuracyMetricCalculator.WEIGHTS["granularity_boundary"],
            description="Alignment of partition boundaries against independent expert stage design",
            literature_reference="MoJoFM Partition Distance (IEEE TSE '07, Maqbool & Babar)",
            details=details
        )

    @staticmethod
    def calculate_trigger_decoupling_categories(
        tool_triggers: Dict[str, str], expert_triggers: Dict[str, str]
    ) -> DimensionScore:
        """Trigger-decoupling accuracy at the category level (sync HTTP vs async execution).

        Compares the multiset of trigger *categories* so the check is independent of
        function naming: api_gateway -> sync_http; sqs/event/queue -> async.
        """
        def category(t: Any) -> str:
            if t in ("api_gateway", "http", "http_api"):
                return "sync_http"
            if t in ("sqs", "event", "queue", "async_invoke"):
                return "async"
            return f"other:{t}"

        tool_cats = sorted(category(t) for t in tool_triggers.values())
        exp_cats = sorted(category(t) for t in expert_triggers.values())
        if not exp_cats and not tool_cats:
            score = 0.0
        else:
            from collections import Counter
            c_tool, c_exp = Counter(tool_cats), Counter(exp_cats)
            inter = sum((c_tool & c_exp).values())
            union = sum((c_tool | c_exp).values())
            score = inter / union if union else 0.0
        return DimensionScore(
            dimension_name="Trigger & Async Decoupling Mechanism (S_trigger)",
            weight=AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
            raw_score=score,
            weighted_score=score * AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
            description="Category-level sync/async trigger design match (naming-independent)",
            literature_reference="GRouter (EuroSys '26) & Nightcore (ASPLOS '21)",
            details={
                "tool_categories": tool_cats,
                "expert_categories": exp_cats,
            }
        )

    @staticmethod
    def calculate_template_compliance(
        compliance_scores: List[float], compliance_details: List[Dict[str, Any]]
    ) -> DimensionScore:
        """Calculate compliance with standard cloud-native FaaS function templates (ToLambda / AST Schema)."""
        avg_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 1.0
        return DimensionScore(
            dimension_name="Function Template & Signature Compliance (S_template)",
            weight=AccuracyMetricCalculator.WEIGHTS["template_compliance"],
            raw_score=avg_score,
            weighted_score=avg_score * AccuracyMetricCalculator.WEIGHTS["template_compliance"],
            description="Compliance with standardized FaaS handler interface, parsing, and response wrapper",
            literature_reference="ToLambda (IEEE/ACM IWoR '19) & Cloud-Native FaaS Spec",
            details={
                "average_compliance": round(avg_score, 4),
                "inspected_functions_count": len(compliance_scores),
                "per_function_details": compliance_details
            }
        )

    @staticmethod
    def calculate_trigger_decoupling(
        tool_triggers: Dict[str, str], expert_triggers: Dict[str, str]
    ) -> DimensionScore:
        """Calculate trigger classification and async task decoupling accuracy (Nightcore / GRouter)."""
        all_funcs = set(expert_triggers.keys())
        if not all_funcs:
            score = 1.0
            return DimensionScore(
                dimension_name="Trigger & Async Decoupling Mechanism (S_trigger)",
                weight=AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
                raw_score=score,
                weighted_score=score * AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
                description="Accurate classification of synchronous HTTP vs asynchronous queue/worker triggers",
                literature_reference="GRouter (EuroSys '26) & Nightcore (ASPLOS '21)",
                details={"matching_rate": score}
            )

        matches = 0
        details = {}
        for func in all_funcs:
            t_exp = expert_triggers.get(func)
            t_tool = tool_triggers.get(func)
            matched = (t_exp == t_tool)
            if matched:
                matches += 1
            details[func] = {
                "expert_trigger": t_exp,
                "tool_trigger": t_tool,
                "is_match": matched
            }

        score = matches / len(all_funcs)
        return DimensionScore(
            dimension_name="Trigger & Async Decoupling Mechanism (S_trigger)",
            weight=AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
            raw_score=score,
            weighted_score=score * AccuracyMetricCalculator.WEIGHTS["trigger_decoupling"],
            description="Accurate classification of synchronous HTTP vs asynchronous queue/worker triggers",
            literature_reference="GRouter (EuroSys '26) & Nightcore (ASPLOS '21)",
            details={
                "matching_rate": round(score, 4),
                "correct_triggers": matches,
                "total_triggers": len(all_funcs),
                "breakdown": details
            }
        )

    @staticmethod
    def calculate_infra_alignment(
        tool_infra: Dict[str, Any], expert_infra: Dict[str, Any]
    ) -> DimensionScore:
        """Calculate SAM/Cloud-Native infrastructure and layer decoupling consistency (Crucial / TOSEM)."""
        check_keys = ["api_gateway", "cognito", "sqs_queues", "dynamodb_tables", "s3_buckets"]
        points = 0
        total = len(check_keys)
        comparison = {}

        for k in check_keys:
            exp_val = expert_infra.get(k)
            tool_val = tool_infra.get(k)
            
            if isinstance(exp_val, list) and isinstance(tool_val, list):
                set_exp = set(exp_val)
                set_tool = set(tool_val)
                jaccard = len(set_exp.intersection(set_tool)) / len(set_exp.union(set_tool)) if set_exp.union(set_tool) else 1.0
                points += jaccard
                comparison[k] = {"expert": sorted(list(set_exp)), "tool": sorted(list(set_tool)), "score": round(jaccard, 4)}
            else:
                match = (bool(exp_val) == bool(tool_val))
                points += 1.0 if match else 0.0
                comparison[k] = {"expert": exp_val, "tool": tool_val, "score": 1.0 if match else 0.0}

        score = points / total
        return DimensionScore(
            dimension_name="Cloud-Native Infrastructure & Layer Decoupling (S_infra)",
            weight=AccuracyMetricCalculator.WEIGHTS["infra_alignment"],
            raw_score=score,
            weighted_score=score * AccuracyMetricCalculator.WEIGHTS["infra_alignment"],
            description="Consistency of SAM IaC templates, shared algorithm layers, and authorization policies",
            literature_reference="Crucial (EuroSys '19) & Serverless Review (ACM TOSEM '22)",
            details={
                "infra_matching_score": round(score, 4),
                "resource_breakdown": comparison
            }
        )

    @staticmethod
    def calculate_runtime_consistency(
        diff_test_results: Dict[str, Any]
    ) -> DimensionScore:
        """Calculate runtime execution semantic output equivalence (Differential Testing / McMinn et al.)."""
        # No test results at all => NOT EVALUATED (score 0 on the full scale), never a default pass.
        evaluated = bool(diff_test_results.get("evaluated", False))
        score = diff_test_results.get("score", 0.0) if evaluated else 0.0
        details = dict(diff_test_results)
        details["evaluated"] = evaluated
        return DimensionScore(
            dimension_name="Runtime Execution & Semantic Output Equivalence (S_runtime)",
            weight=AccuracyMetricCalculator.WEIGHTS["runtime_consistency"],
            raw_score=score,
            weighted_score=score * AccuracyMetricCalculator.WEIGHTS["runtime_consistency"],
            description="Dynamic equivalence of status codes, JSON payload schemas, and numerical outputs (eps<=1e-4)",
            literature_reference="Differential Testing for Refactoring (IEEE TSE '16, McMinn et al.; ICSE)",
            details=details
        )
