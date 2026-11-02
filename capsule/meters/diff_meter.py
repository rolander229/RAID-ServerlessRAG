"""D1 连续化数值等价计量（指标 v2 的裸机版探针）。

N 个固定种子的随机输入分别跑「切分链」与「单体 forward」，逐请求取输出张量的
max-abs-diff，按连续公式计分：

    S = clamp(1 - log10(maxdiff / tol), 0, 1),  tol = 1e-4

即 maxdiff ≤ 1e-4 满分；超差每恶化一个数量级扣 1 分（半个数量级扣 0.5），
优于容差按满分截断。

与《metrics-review-k8s.md》§3.2 D1 字面公式的差异（重要）：原文
``S = max(0, 1 - log10(maxdiff)/log10(1e-4))`` 若按字面计算，分子分母同为负数，
**误差越大分反而越高**，且在 maxdiff 恰为 1e-4 时得 0 分——与同一句话的两条
锚点（“maxdiff≤1e-4 得满分”“差 1 个数量级扣一档”）矛盾，属笔误。本实现保留
两条锚点，把比值改成对容差的超额数量级（excess = log10(maxdiff/tol)）。

链的执行走“线性部署契约”：每个片只允许一个上游、一个下游且必须相邻
（slice-runner 的 --next-peer 形态，见 capsule/slice_server.py）。违反契约的
切分产物无法部署为链，D1 直接计 0 分——这是负对照 ①（一层划错片）能被指标
体系抓住的原因。

Run: python -m capsule.meters.diff_meter --model resnet50 --slices 3 --n 100
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

from slicer.models import zoo
from slicer.models.graphify import _iter_tensors
from slicer.faas.program import _fill, record
from slicer.models.weight_shard import expand_parts

REPO_ROOT = Path(__file__).resolve().parents[2]  # capsule/meters/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOLERANCE = 1e-4  # RQ3 的验收线，也是 D1 计分的满分锚点


class PlanTopologyError(Exception):
    """切分产物不满足线性链契约（一片多上游/多下游/跨片非相邻）。"""


def continuous_diff_score(maxdiff: float, tol: float = TOLERANCE) -> float:
    """按超额数量级连续扣分：≤tol 满分，每差 10× 扣 1 分，下限 0。"""
    if maxdiff <= 0.0:
        return 1.0
    excess = math.log10(maxdiff) - math.log10(tol)
    return max(0.0, 1.0 - excess)


def check_linear(cases) -> None:
    """线性部署契约：inbound 只能来自 i-1，outbound 只能发给 i+1。"""
    for case in cases:
        bad_in = [p for p, _ in case.inbound if p != case.index - 1]
        bad_out = [p for p, _ in case.outbound if p != case.index + 1]
        if bad_in or bad_out:
            raise PlanTopologyError(
                "slice %d: inbound from %s, outbound to %s -- not a linear chain"
                % (case.index, bad_in, bad_out))


def run_linear_chain(program, parts, inputs):
    """:func:`slicer.faas.program.run_chain` + 线性契约检查（先契约后执行）。"""
    cases = program.slices(parts)
    check_linear(cases)
    envs: list[dict] = [{} for _ in cases]
    request = dict(zip(program.inputs, _iter_tensors(inputs)))
    for case, env in zip(cases, envs):
        for key in case.request:
            env[key] = request[key]
        for peer, keys in case.inbound:
            for key in keys:
                env[key] = envs[peer][key]
        case.run(env)
    merged: dict = {}
    for env in envs:
        merged.update(env)
    return _fill(program.template, merged)


def plan_membership_f1(parts, reference) -> float:
    """层归属 micro-F1：逐层比较两份切分的所属片，粒度对齐的占位。"""
    home_got, home_ref = {}, {}
    for index, names in enumerate(parts):
        for name in names:
            home_got[name] = index
    for index, names in enumerate(reference):
        for name in names:
            home_ref[name] = index
    universe = set(home_got) | set(home_ref)
    correct = sum(1 for name in universe
                  if name in home_got and name in home_ref
                  and home_got[name] == home_ref[name])
    if not universe:
        return 1.0
    precision = correct / max(1, len(home_got))
    recall = correct / max(1, len(home_ref))
    return 2 * precision * recall / max(1e-12, precision + recall)


def _max_diff(a, b) -> float:
    diffs = [(x.float() - y.float()).abs().max().item()
             for x, y in zip(_iter_tensors(a), _iter_tensors(b))]
    return max(diffs) if diffs else 0.0


def evaluate(model_key: str, slices: int = 3, n: int = 100, seed: int = 0,
             device: str = "cpu", tol: float = TOLERANCE,
             parts_override=None, program=None) -> dict:
    """N 输入的链 vs 单体差分；返回分数、maxdiff 分布与计划元数据。

    ``program`` 可注入已 record 的 Program（负对照测试里两次 evaluate 共用
    一次 record：resnet50 的 record 要十几秒，没必要付两遍）。
    """
    if program is None:
        model, inputs = zoo.load(model_key, device=device)
        program = record(model, inputs, name=model_key, device=device)
    model = program.model
    parts = parts_override if parts_override is not None else expand_parts(program, slices)
    reference = expand_parts(program, slices)

    spec = zoo.get(model_key)
    maxdiffs: list[float] = []
    scores: list[float] = []
    chain_error: str | None = None
    try:
        for i in range(n):
            torch.manual_seed(seed * 100000 + i)  # 逐输入固定且可复现
            inp = spec.inputs(spec.batch)  # zoo 的输入本来就是 (tensor,) 元组
            with torch.no_grad():
                mono = model(*inp)
            chain = run_linear_chain(program, parts, inp)
            diff = _max_diff(mono, chain)
            maxdiffs.append(diff)
            scores.append(continuous_diff_score(diff, tol))
    except PlanTopologyError as exc:
        # 契约违约的产物无法部署为链：D1 计 0 分（负对照 ① 的路径）。
        chain_error = str(exc)
        scores = [0.0]
        maxdiffs = [float("nan")]

    result = {
        "model_key": model_key, "slices_requested": slices,
        "n_inputs": n if chain_error is None else 0, "seed": seed,
        "tolerance": tol, "device": device,
        "score_mean": statistics.fmean(scores),
        "pass_rate_1e4": (sum(1 for d in maxdiffs if d <= tol) / len(maxdiffs)
                          if chain_error is None else 0.0),
        "maxdiff_distribution": {
            "min": min(maxdiffs), "median": statistics.median(maxdiffs),
            "p95": sorted(maxdiffs)[min(len(maxdiffs) - 1, int(round(0.95 * len(maxdiffs))) - 1)]
            if maxdiffs and not math.isnan(maxdiffs[0]) else float("nan"),
            "max": max(maxdiffs)},
        "plan_membership_f1": plan_membership_f1(parts, reference),
        "expanded_parts": parts,
        "chain_error": chain_error,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D1 continuous numeric-equivalence meter.")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tol", type=float, default=TOLERANCE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = evaluate(args.model, args.slices, args.n, args.seed, args.device, args.tol)
    dist = report["maxdiff_distribution"]
    print("[diff_meter] model=%s slices=%d n=%d seed=%d" %
          (args.model, args.slices, report["n_inputs"], args.seed))
    if report["chain_error"] is None:
        print("[diff_meter] maxdiff: min=%.3e median=%.3e p95=%.3e max=%.3e" %
              (dist["min"], dist["median"], dist["p95"], dist["max"]))
        print("[diff_meter] pass_rate(<=1e-4)=%.2f  score(D1)=%.4f" %
              (report["pass_rate_1e4"], report["score_mean"]))
    else:
        print("[diff_meter] chain refused: %s" % report["chain_error"])
        print("[diff_meter] score(D1)=0.0 (plan cannot deploy as a linear chain)")
    out = args.out or ("runs/diff_meter_%s_k%d.json" % (args.model, args.slices))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[diff_meter] report -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
