"""指标 v2 的裸机计量探针（metrics-review-k8s.md §3.2 的 D1/D2 + 计费记账）。

- :mod:`diff_meter` — D1 连续化数值等价（切分链 vs 单体，N 固定种子输入）
- :mod:`mem_meter`  — D2 裸机版（逐片子进程 RSS 峰值 / alloc 配额比 + 聚合削减率）
- :mod:`cost_meter` — 计量 → 资源量表 + AWS Lambda 锚点影子价（计量与定价解耦）
- :mod:`backfill_baselines` — D3 的分母：从 runs/ 既有 JSON 回填裸机基线
"""
