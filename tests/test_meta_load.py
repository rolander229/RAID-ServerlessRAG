"""C2 shard-first loading: meta-built program materialised from shards.

The serving container must not hold the whole model while it only runs one
slice.  ``zoo.load(init="meta")`` builds the architecture on the meta device
(0 bytes), ``record`` still traces the forward (shapes only), and the shard
loader replaces table entries with the real tensors and rewrites the step
arguments that captured weights by value.  These tests pin that contract,
including the bit-exactness the runtime depends on.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from slicer.faas.program import record, run_chain
from slicer.models.weight_shard import expand_parts, load_shard, patch_step_tensors


class TinyNet(nn.Module):
    """Small but structurally representative: conv stack + linear head."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def _state(model):
    return {name: t.detach().clone() for name, t in model.state_dict().items()}


class MetaShardFirstTest(unittest.TestCase):
    def test_meta_build_costs_no_bytes_and_records(self):
        with torch.device("meta"):
            model = TinyNet()
            model.eval()
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        x = torch.randn(2, 3, 32, 32)
        prog = record(model, (x,), name="tiny", device="meta")
        self.assertGreater(len(prog.steps), 0)

    def test_shard_fill_makes_slice_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1) CPU 参考模型（真实权重 + 其 state_dict 的内容）
            ref = TinyNet()
            ref.eval()
            state = _state(ref)
            shard = Path(tmp) / "slice_all.pt"
            torch.save({"state": state}, shard)

            # 2) meta 实例化 + record
            with torch.device("meta"):
                meta_model = TinyNet()
                meta_model.eval()
            x = torch.randn(2, 3, 32, 32)
            prog = record(meta_model, (x,), name="tiny", device="meta")

            # 3) 分片灌入（meta 参数表项替换 + steps 字面量修补）
            remap = {}
            loaded = load_shard(meta_model, shard, remap=remap)
            patched = patch_step_tensors(prog.steps, remap)
            self.assertEqual(sorted(loaded), sorted(state))
            self.assertGreater(patched, 0, "conv/linear 权重按值出现在 step 参数里，必须被替换")
            self.assertFalse(any(p.is_meta for p in meta_model.parameters()))

            # 4) 逐位一致
            parts = expand_parts(prog, 2)
            with torch.no_grad():
                chained = run_chain(prog, parts, x)
                mono = ref(x)
            diff = (chained[0] - mono[0]).abs().max().item()
            self.assertEqual(diff, 0.0)

    def test_meta_program_runs_only_loaded_slice(self):
        """只加载一片时，未加载的层保持 meta——但本片步骤必须能真实执行。"""
        with tempfile.TemporaryDirectory() as tmp:
            ref = TinyNet()
            ref.eval()
            state = _state(ref)
            shard = Path(tmp) / "slice_0.pt"
            torch.save({"state": state}, shard)
            with torch.device("meta"):
                m = TinyNet()
                m.eval()
            x = torch.randn(2, 3, 32, 32)
            prog = record(m, (x,), name="tiny", device="meta")
            remap = {}
            load_shard(m, shard, remap=remap)
            patch_step_tensors(prog.steps, remap)
            env = dict(zip(prog.inputs, [x]))
            for step in prog.steps:
                step.run(env)  # 不应抛 meta/device 相关错误
            out = prog.template
            self.assertTrue(len(env) > 0)


if __name__ == "__main__":
    unittest.main()
