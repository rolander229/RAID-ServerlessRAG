#!/usr/bin/env python3
"""
Intelligent Algorithm Model Cloud-Native Service Migration
Function Encapsulation Accuracy Automated Evaluation Runner

Usage:
    python evaluate_accuracy.py [--benchmark all|rag_anything|recsys_deepfm|dnn_pipeline]
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluator import EncapsulationEvaluator
from evaluation.report_generator import EvaluationReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Function Encapsulation Accuracy against Expert Ground-Truth Baseline."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Benchmark to evaluate ('all', 'rag_anything', 'recsys_deepfm', 'dnn_pipeline')",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/evaluation_results.json",
        help="Path to export evaluation JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/03_智能算法模型云原生服务化工具函数封装准确率第三方测试报告.md",
        help="Path to export Markdown third-party test report",
    )

    args = parser.parse_args()

    print(f"Initializing Encapsulation Accuracy Evaluator on: {ROOT_DIR}")
    evaluator = EncapsulationEvaluator(ROOT_DIR)

    if args.benchmark == "all":
        results = evaluator.evaluate_all()
    else:
        target_dir = ROOT_DIR / "benchmarks" / args.benchmark
        if not target_dir.exists():
            print(f"Error: Benchmark directory '{target_dir}' not found.")
            sys.exit(1)
        results = [evaluator.evaluate_benchmark(target_dir)]

    if not results:
        print("No evaluation benchmarks found.")
        sys.exit(1)

    # 1. Print formatted console summary
    summary_text = EvaluationReportGenerator.format_console_summary(results)
    print("\n" + summary_text + "\n")

    # 2. Export JSON results
    json_path = ROOT_DIR / args.output_json
    EvaluationReportGenerator.export_json(results, json_path)
    print(f" JSON results exported to: {json_path}")

    # 3. Export Markdown report
    md_path = ROOT_DIR / args.output_md
    EvaluationReportGenerator.export_markdown_report(results, md_path)
    print(f" Third-party Markdown test report exported to: {md_path}")

    # 4. Exit code based on pass/fail
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("\n ALL BENCHMARKS PASSED TARGET THRESHOLD (>= 70.0%). CONGRATULATIONS!")
        sys.exit(0)
    else:
        print("\n SOME BENCHMARKS FAILED TO MEET THE TARGET THRESHOLD (>= 70.0%).")
        sys.exit(2)


if __name__ == "__main__":
    main()
