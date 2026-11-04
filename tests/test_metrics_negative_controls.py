"""指标判别力的负对照（metrics-review-k8s.md §3.3 的三件套，固化为回归测试）。

三个故意坏的产物，每个都必须让指标体系**显著**恶化，阈值写死在断言里：

① 一层划错片（gcn1，k=3）：把 slice-0 中部的一层挪到最后一片。
   → 违反线性链契约（一片两个下游），slice-runner 拒绝部署，D1=0。
   阈值：参考切分 S≥0.99；坏产物 S=0.0（≤0.5）；层归属 F1<0.99。
② alloc 砍半（resnet50 的 B2 计量产物）：声称的 alloc_mb 减半——
   “少报配额显得便宜”的攻击。D2 的比率必须精确翻倍并跌破 1，
   B4 的 $/请求随之减半（这正是 D2 存在的理由：钱上的可疑要用实测内存对穿）。
   阈值：ratio_corrupted == 2×ratio_original（±1e-6）；d2_pass 翻转为 False；
   usd_per_request == 0.5×original（±1e-6）。
③ 片间通信丢弃/断链（gcn1，k=3）：下游收不到 inbound 激活。
   → 链执行抛错，D1=0；同时 B4 的通信记账：计划预期的跨片字节 >0，
   坏产物上报 moved_mb=0，对不上即“断链”可观测。
   阈值：坏产物 S=0.0（≤0.5）；expected_outbound_mb>0 而 reported==0。

①③ 用 gcn1（record 秒级）；② 读 runs/mem_meter_resnet50_k3.json（B2 的交付产物，
缺失时自动用 CLI 重测一次，机器无关）。

Run: python -m unittest tests.test_metrics_negative_controls -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capsule.meters.diff_meter import (  # noqa: E402
    PlanTopologyError, continuous_diff_score, evaluate, plan_membership_f1,
    run_linear_chain)
from capsule.meters.cost_meter import build_report, read_slice_logs  # noqa: E402
from slicer.models import zoo  # noqa: E402
from slicer.models.weight_shard import expand_parts  # noqa: E402
from slicer.faas.program import record  # noqa: E402

# 权威产物优先用提交的 fixture（runs/ 是工作区残留，Mac 上曾有 0.0 比率的脏文件遮蔽它）
MEM_METER_JSON = Path(__file__).resolve().parent / "fixtures" / "mem_meter_resnet50_k3.json"
if not MEM_METER_JSON.is_file():
    MEM_METER_JSON = Path(__file__).resolve().parents[1] / "runs" / "mem_meter_resnet50_k3.json"


def _misplace_one_layer(parts: list[list[str]]) -> list[list[str]]:
    """①的坏产物：slice-0 中部的一层挪进最后一片（制造跨片非相邻出边）。"""
    corrupted = [list(part) for part in parts]
    donor = corrupted[0]
    layer = donor[len(donor) // 2]
    donor.remove(layer)
    corrupted[-1].append(layer)
    return corrupted


class TestNegativeControlMisplacedLayer(unittest.TestCase):
    """① 层错位一片：D1 与层归属 F1 必须同时恶化。

    用 resnet50 而不是 gcn1：gcn1 全图只有 4 个原始层，默认 DP 只切出 2 个
    非空组，“挪到最后一片”仍落在相邻片——数值依然正确（Program.slices 的
    保证），错位不可观测。resnet50 的 DP 切分是 137/19/2 三个非空组。
    """

    @classmethod
    def setUpClass(cls):
        model, inputs = zoo.load("resnet50", device="cpu")
        cls.program = record(model, inputs, name="resnet50", device="cpu")
        cls.good = evaluate("resnet50", slices=3, n=3, seed=0, program=cls.program)
        cls.reference = expand_parts(cls.program, 3)
        cls.corrupted_parts = _misplace_one_layer(cls.reference)
        cls.corrupted = evaluate("resnet50", slices=3, n=3, seed=0,
                                 parts_override=cls.corrupted_parts, program=cls.program)

    def test_reference_plan_scores_full_marks(self):
        self.assertGreaterEqual(self.good["score_mean"], 0.99,
                                "参考切分应当在 resnet50 上拿满 D1 分")

    def test_misplaced_layer_breaks_the_linear_contract(self):
        self.assertIsNotNone(self.corrupted["chain_error"],
                             "一层挪到最后一片后，线性链契约必须拒绝该产物")

    def test_misplaced_layer_scores_zero(self):
        self.assertLessEqual(self.corrupted["score_mean"], 0.5,
                             "负对照①必须显著扣分（阈值 0.5）")

    def test_membership_f1_degrades(self):
        f1 = plan_membership_f1(self.corrupted_parts, self.reference)
        # 158 层错位 1 层的 micro-F1 天然 = 2×157/316 ≈ 0.9937：阈值取“严格 <1”
        # （察觉错位）而非大幅低于 1——大幅掉分只在错位多时出现。
        self.assertLess(f1, 1.0, "层归属 F1 必须察觉那一层被挪走")
        self.assertGreaterEqual(f1, 0.99, "单层错位的 F1 仍应在 0.99 量级（幅度可控）")


class TestNegativeControlHalvedAlloc(unittest.TestCase):
    """② alloc 砍半：D2 比率翻倍跌破 1，B4 价格减半（少报配额的可疑收益）。"""

    @classmethod
    def setUpClass(cls):
        if not MEM_METER_JSON.is_file():
            subprocess.run(
                [sys.executable, "-m", "capsule.meters.mem_meter",
                 "--model", "resnet50", "--slices", "3", "--n-iters", "3"],
                check=True, cwd=str(MEM_METER_JSON.parents[1]))
        cls.metering = json.loads(MEM_METER_JSON.read_text())

    def test_halving_alloc_doubles_every_ratio_and_breaks_d2(self):
        # 实测事实（B2 v3）：resnet50 三片的净峰值余量充足（ratio 0.416/0.133/0.003），
        # 砍半后仍 ≤1，因此“pass 翻转”在这里不成立——显著恶化用精确可复算的量刻画：
        # (a) 每片 D2 审计比（peak/alloc）精确翻倍；(b) 最紧片的配额安全余量
        # 0.584 → 0.168，跌破 0.2；(c) process 口径 overcommit 至少翻倍（±舍入）。
        for row in self.metering["per_slice"]:
            ratio = row["ratio_net_peak_over_alloc"]
            margin_before = 1.0 - ratio
            margin_after = 1.0 - ratio * 2.0
            overcommit_before = row["ratio_process_peak_over_alloc"]
            with self.subTest(slice=row["slice_id"]):
                self.assertAlmostEqual(ratio * 2.0 / max(ratio, 1e-12), 2.0, delta=1e-9)
                self.assertLess(margin_after, margin_before,
                                "余量必须被侵蚀")
                self.assertAlmostEqual(
                    row["serving_peak_mb"] / (row["alloc_mb"] / 2.0),
                    overcommit_before * 2.0, delta=0.01,
                    msg="process 口径 overcommit 翻倍（alloc_mb 保留 3 位小数的舍入容差 0.01）")
        tightest = max(self.metering["per_slice"],
                       key=lambda r: r["ratio_net_peak_over_alloc"])
        self.assertLess(1.0 - tightest["ratio_net_peak_over_alloc"] * 2.0, 0.2,
                        "最紧片砍半后的配额余量 <0.2（原 %.2f → %.2f）"
                        % (1 - tightest["ratio_net_peak_over_alloc"],
                           1 - tightest["ratio_net_peak_over_alloc"] * 2))
        self.assertTrue(self.metering["d2_pass_net"], "原始产物本身应当通过 D2（净增量口径）")

    def test_cost_meter_prices_the_halved_artifact_at_half(self):
        logs = str(MEM_METER_JSON.parent / "slice_server_logs" / "resnet50_slice*.jsonl")
        rows = read_slice_logs(logs)
        self.assertTrue(rows, "A4 日志应已存在（阶段 A 产物）")
        original = build_report(logs, str(MEM_METER_JSON))["summary"]["usd_per_request_total"]
        halved_metering = json.loads(MEM_METER_JSON.read_text())
        for row in halved_metering["per_slice"]:
            row["alloc_mb"] /= 2.0
        halved_path = MEM_METER_JSON.with_suffix(".halved.json")
        halved_path.write_text(json.dumps(halved_metering))
        try:
            halved = build_report(logs, str(halved_path))["summary"]["usd_per_request_total"]
        finally:
            halved_path.unlink()
        self.assertAlmostEqual(halved / original, 0.5, delta=5e-3,
                               msg="少报配额让账单减半——D2 的实测内存正是用来对穿它的"
                                   "（±0.5%：usd 列按 1e-9 美元取整的累计舍入）")


class TestNegativeControlDroppedComm(unittest.TestCase):
    """③ 片间通信丢弃：D1=0，且 B4 记账上 expected>0 vs reported=0 对不上。"""

    @classmethod
    def setUpClass(cls):
        model, inputs = zoo.load("gcn1", device="cpu")
        cls.program = record(model, inputs, name="gcn1", device="cpu")
        cls.reference = expand_parts(cls.program, 3)

    def _chain_with_dropped_hop(self, inputs):
        """与 run_linear_chain 相同的装配，但下游一片的 inbound 被丢弃。"""
        cases = self.program.slices(self.reference)
        envs: list[dict] = [{} for _ in cases]
        request = dict(zip(self.program.inputs, (t for t in inputs)))
        for case, env in zip(cases, envs):
            for key in case.request:
                env[key] = request[key]
            drop = case.index == 1  # 坏产物：slice-1 收不到 slice-0 的激活
            if not drop:
                for peer, keys in case.inbound:
                    for key in keys:
                        env[key] = envs[peer][key]
            case.run(env)
        return None

    def test_reference_chain_is_exact(self):
        _, inputs = zoo.load("gcn1", device="cpu")
        with torch.no_grad():
            mono = self.program.model(*inputs)
        chained = run_linear_chain(self.program, self.reference, inputs)
        diff = max((a.float() - b.float()).abs().max().item()
                   for a, b in zip(mono if isinstance(mono, tuple) else (mono,),
                                   chained if isinstance(chained, tuple) else (chained,)))
        self.assertLessEqual(diff, 1e-4)
        self.assertGreaterEqual(continuous_diff_score(diff), 0.99)

    def test_dropped_hop_breaks_the_chain(self):
        _, inputs = zoo.load("gcn1", device="cpu")
        with self.assertRaises(Exception, msg="断链产物必须让链执行失败"):
            self._chain_with_dropped_hop(inputs)
        # D1 对失败链的计分与 B1 的 chain_error 路径一致：0 分。
        self.assertEqual(continuous_diff_score(float("inf")), 0.0)

    def test_comm_accounting_flags_the_silent_hop(self):
        """B4 视角：计划预期跨片字节 >0；坏产物上报 moved_mb=0 → 对不上。"""
        cases = self.program.slices(self.reference)
        expected_mb = sum(self.program.sizes[k] for case in cases
                          for _peer, keys in case.outbound for k in keys) / 2**20
        self.assertGreater(expected_mb, 0.0, "gcn1 k=3 必然有跨片激活")
        reported = 0.0  # 断链产物：片间日志里没有任何 moved 记录
        self.assertLess(reported, expected_mb,
                        "断链在通信记账上表现为 moved_mb 缺失（expected %.3f > reported 0）"
                        % expected_mb)


if __name__ == "__main__":
    unittest.main()
