"""
Capsule: Cloud-Native Serverless Migration, FaaS Synthesizer & Accuracy Evaluation Framework
Imported directly into MOPAR for end-to-end micro-partitioned serverless delivery.
"""

from .evaluation.evaluator import EncapsulationEvaluator, FunctionAstInspector
from .evaluation.metrics import AccuracyMetricCalculator, BenchmarkEvaluationResult, DimensionScore
from .evaluation.runtime_executor import RuntimeConsistencyExecutor
from .bridge import SliceCapsuleBridge

__all__ = [
    "EncapsulationEvaluator",
    "FunctionAstInspector",
    "AccuracyMetricCalculator",
    "BenchmarkEvaluationResult",
    "DimensionScore",
    "RuntimeConsistencyExecutor",
    "SliceCapsuleBridge",
]
