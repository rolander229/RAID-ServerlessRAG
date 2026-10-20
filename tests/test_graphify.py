"""Tests for the layer-graph extractor: the topology, not the timings.

These need torch, so they are skipped on a machine without it; they run on v100
via ``python -m unittest discover -s tests``. They exist
because a mis-recovered edge is not a visible failure -- the partitioner happily
cuts a graph that the model does not implement, and the experiment then measures
a different network. Two topologies are checked by hand: an in-place activation,
which broke the first version, and a residual block, whose skip edge is the one
that has to survive.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - the local editing mirror has no torch
    torch = None

from slicer.models.graphify import build_structure_graph  # noqa: E402


@unittest.skipIf(torch is None, "torch not installed")
class TestStructure(unittest.TestCase):
    def setUp(self) -> None:
        class Block(nn.Module):
            """fc -> relu(inplace) -> fc, with the input added back."""

            def __init__(self, width: int) -> None:
                super().__init__()
                self.fc1 = nn.Linear(width, width)
                self.act = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(width, width)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.act(self.fc1(x))) + x

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Linear(8, 8)
                self.block_a = Block(8)
                self.block_b = Block(8)
                self.head = nn.Linear(8, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(self.block_b(self.block_a(self.stem(x))))

        torch.manual_seed(0)
        self.net = Net().eval()
        self.width = 8
        self.graph = build_structure_graph(self.net, (torch.randn(4, 8),), name="net")

    def names(self) -> set:
        return set(self.graph.layers)

    def test_every_module_that_ran_is_a_node(self) -> None:
        leaves = {name for name, m in self.net.named_modules() if not list(m.children())}
        self.assertEqual(self.names(), leaves)

    def test_in_place_activation_keeps_its_input_edge(self) -> None:
        # ReLU(inplace=True) writes its output over its own input tensor, so a
        # producer map read after the forward reports the relu as its own source.
        self.assertEqual(self.graph.preds("block_a.act"), ["block_a.fc1"])
        self.assertEqual(self.graph.preds("block_a.fc2"), ["block_a.act"])

    def test_residual_edge_survives_op_contraction(self) -> None:
        # The add at the end of a block is a plain function, so block_b.fc1's two
        # producers are block_a.fc2 (the branch) and block_a's input (the skip).
        # block_a's input is the stem, since the stem is the last module to emit
        # that value.
        self.assertEqual(self.graph.preds("block_b.fc1"),
                         ["block_a.fc2", "stem"])

    def test_edges_carry_the_consumed_value_size(self) -> None:
        expected = 4 * self.width * 4  # batch x width x float32
        for edge in self.graph.tensors:
            self.assertEqual(edge.bytes, expected, edge)

    def test_graph_is_a_dag_with_one_sink(self) -> None:
        self.graph.topological_order()
        self.assertEqual(self.graph.sinks(), ["head"])

    def test_model_input_is_the_only_source(self) -> None:
        # block_a's skip operand is the value stem emitted, so the add contributes
        # a second stem edge rather than making block_a.fc1 an input node.
        self.assertEqual(self.graph.sources(), ["stem"])


@unittest.skipIf(torch is None, "torch not installed")
class TestFanIn(unittest.TestCase):
    """A residual chain must not accumulate its history into one node's in-edges.

    The first version expanded an op node's operands recursively, so the last node
    of a deep stack was fed by every module above it: measured on the 722-node
    Transformer, 145 edges into one ``norm``, 6.16 MB each, which Eq. (3) then read
    as a 223 MB layer inside a 16 MB service.
    """

    def graph_of(self, blocks: int):
        class Block(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.fc = nn.Linear(width, width)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        class Stack(nn.Module):
            def __init__(self, depth: int, width: int) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(Block(width) for _ in range(depth))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = torch.add(block(x), x)  # the add is a function, not a module
                return torch.norm(x)

        torch.manual_seed(0)
        return build_structure_graph(Stack(blocks, 8).eval(), (torch.randn(4, 8),),
                                     name="stack")

    def in_degree(self, graph, name: str) -> int:
        return len(graph.in_edges(name))

    def test_a_deep_chain_keeps_its_fan_in_bounded(self) -> None:
        deep = self.graph_of(64)
        self.assertLessEqual(max(self.in_degree(deep, n) for n in deep.layers), 2)
        # Roughly one edge per module, not one per module above it: an 8-deep stack
        # would need 28 edges for its last node alone under recursive expansion.
        self.assertLess(len(deep.tensors), 2 * len(deep.layers))

    def test_the_edge_out_of_a_residual_add_is_the_accumulator(self) -> None:
        # The block after an add reads one buffer, so it has one producer: the newest
        # module the accumulator descends from, not both operands of the add.
        graph = self.graph_of(4)
        second = [name for name in graph.order if name.endswith("blocks.1.fc")]
        self.assertEqual(self.in_degree(graph, second[0]), 1)
        self.assertEqual(graph.preds(second[0]), ["blocks.0.fc"])

    def test_a_fusion_still_ships_every_branch(self) -> None:
        class Branches(nn.Module):
            """Three leaves whose outputs meet in a ``cat``: all three must cross."""

            def __init__(self, width: int) -> None:
                super().__init__()
                self.a = nn.Linear(width, width)
                self.b = nn.Linear(width, width)
                self.c = nn.Linear(width, width)
                self.head = nn.Linear(width * 3, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(torch.cat([self.a(x), self.b(x), self.c(x)], dim=1))

        torch.manual_seed(0)
        graph = build_structure_graph(Branches(8).eval(), (torch.randn(4, 8),),
                                      name="branches")
        self.assertEqual(sorted(graph.preds("head")), ["a", "b", "c"])
        self.assertEqual(sum(edge.bytes for edge in graph.in_edges("head")),
                         3 * 4 * 8 * 4)

    def test_nested_fusion_keeps_every_branch(self) -> None:
        class Mixed(nn.Module):
            """The ``Mixed_7b`` shape: a ``cat`` whose operands are themselves ``cat``s.

            Bound the other way from a residual chain -- all four branch buffers are
            live at the cut above the consumer, so all four need an edge. Numbering
            the branches by "newest module wins" dropped three of them and left
            ``branch3x3_2a.bn`` a sink with no out-edge at all.
            """

            def __init__(self, width: int) -> None:
                super().__init__()
                self.a = nn.Linear(width, width)
                self.b = nn.Linear(width, width)
                self.c = nn.Linear(width, width)
                self.d = nn.Linear(width, width)
                self.head = nn.Linear(width * 4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                left = torch.cat([self.a(x), self.b(x)], dim=1)
                right = torch.cat([self.c(x), self.d(x)], dim=1)
                return self.head(torch.relu(torch.cat([left, right], dim=1)))

        torch.manual_seed(0)
        graph = build_structure_graph(Mixed(8).eval(), (torch.randn(4, 8),),
                                      name="mixed")
        self.assertEqual(sorted(graph.preds("head")), ["a", "b", "c", "d"])
        self.assertEqual(graph.sinks(), ["head"])


@unittest.skipIf(torch is None, "torch not installed")
class TestParameterOwnership(unittest.TestCase):
    """Every parameter a layer has to hold has to be billed to some node.

    The extractor sees leaf modules only, and ``parameters()`` on a leaf is just its
    own tensors, so a weight held by a *container* belongs to no node. The case that
    makes this fatal is functional attention: ``nn.MultiheadAttention`` with
    ``need_weights=False`` takes a fast path that enters no submodule at all, so a
    BERT block's fused QKV matrix -- 48 MB of it -- has no leaf to attach to. The
    first version billed 3329 MB for a 4866 MB model, and HyPAD then sized every
    slice of a transformer against two-thirds of its weights.
    """

    def bytes_of(self, *modules) -> float:
        return sum(float(p.numel()) * p.element_size()
                   for m in modules for p in m.parameters())

    def charged(self, graph) -> float:
        return sum(layer.params for layer in graph.layers.values())

    def test_a_functional_modules_weights_reach_the_next_leaf(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Linear(8, 8)
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.after = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.stem(x)
                return self.after(self.attn(h, h, h, need_weights=False)[0])

        torch.manual_seed(0)
        net = Net().eval()
        graph = build_structure_graph(net, (torch.randn(4, 4, 8),), name="net")
        # Nothing is orphaned: the whole model is charged, not just what leaves hold.
        self.assertAlmostEqual(self.charged(graph), self.bytes_of(net), places=6)
        # And it is charged where it runs, not smeared over the chain.
        self.assertEqual(graph.layer("after").params, self.bytes_of(net.attn, net.after))
        self.assertAlmostEqual(graph.layer("stem").params, self.bytes_of(net.stem),
                               places=6)

    def test_a_containers_own_bytes_land_on_its_first_leaf(self) -> None:
        class Block(nn.Module):
            """A parameter on the block itself, read by its first leaf's output."""

            def __init__(self, width: int) -> None:
                super().__init__()
                self.gain = nn.Parameter(torch.ones(width))
                self.fc = nn.Linear(width, width)
                self.out = nn.Linear(width, width)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.out(self.gain * torch.relu(self.fc(x)))

        torch.manual_seed(0)
        block = Block(8).eval()
        graph = build_structure_graph(block, (torch.randn(4, 8),), name="block")
        gain = float(block.gain.numel()) * block.gain.element_size()
        self.assertAlmostEqual(graph.layer("fc").params,
                               gain + self.bytes_of(block.fc), places=6)
        self.assertAlmostEqual(graph.layer("out").params, self.bytes_of(block.out),
                               places=6)

    def test_a_branch_that_never_runs_still_costs_memory(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Linear(8, 8)
                self.unused = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.stem(x)

        torch.manual_seed(0)
        net = Net().eval()
        graph = build_structure_graph(net, (torch.randn(4, 8),), name="net")
        # The function loads the whole model, so weights this request skipped are
        # resident while the first layer runs. Dropping them instead would bill a
        # slice less memory than the platform has to hand it.
        self.assertAlmostEqual(graph.layer("stem").params, self.bytes_of(net),
                               places=6)

    def test_a_reentered_module_reports_its_footprint_each_time(self) -> None:
        """The number on a node is what running it needs resident, not a share of total.

        Summing ``params`` over a slice of an unrolled cell therefore over-counts the
        weights every step shares; no benchmark model does this (the fused ``nn.LSTM``
        is one leaf), and the partitioner's groups are meant to hold unique bytes.
        """
        class Cell(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.gain = nn.Parameter(torch.ones(width))
                self.fc = nn.Linear(width, width)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.gain * torch.relu(self.fc(x))

        class Net(nn.Module):
            def __init__(self, width: int, steps: int) -> None:
                super().__init__()
                self.cell = Cell(width)
                self.steps = steps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for _ in range(self.steps):
                    x = self.cell(x)
                return x

        torch.manual_seed(0)
        net = Net(8, 3).eval()
        graph = build_structure_graph(net, (torch.randn(4, 8),), name="net")
        steps = [name for name in graph.layers if name.startswith("cell")]
        self.assertEqual(steps, ["cell.fc", "cell.fc#1", "cell.fc#2"])
        for name in steps:
            self.assertAlmostEqual(graph.layer(name).params,
                                   self.bytes_of(net.cell), places=6)


if __name__ == "__main__":
    unittest.main()
