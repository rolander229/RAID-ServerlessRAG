"""Self-checks for Eq. (1)-(3). Pure standard library, no torch.

These are arithmetic on the paper's three consolidation rules, and the point of
each one is which reading it forbids: ``chain`` may never add memory (a chain's
operands are freed as it advances), ``branches`` must add the operands at one
position and take the maximum of their durations, and ``layer_profile`` has to
prefer the wider of the two, because a fused operand is live for the whole
fusion whether or not any single operator inside the layer materialised it.

Run: python3 tests/test_profile_topology.py
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.partition.graph import Layer, LayerGraph, Tensor
from slicer.profile.topology import (Profile, branches, chain, dominant,
                                    dominant_operators, hybrid, layer_profile)


@dataclass
class Op:
    """Stand-in for :class:`~slicer.profile.probe.OpRecord`: topology only cares
    about ``out`` and ``ns``, so the torch import stays out of this file."""

    out: float
    ns: float


@dataclass
class Record:
    """Stand-in for :class:`~slicer.profile.probe.LayerRecord`."""

    ops: list
    ns: float


class TestChain(unittest.TestCase):
    def test_memory_peaks_and_time_adds(self) -> None:
        profile = chain([Op(3e6, 100.0), Op(9e6, 200.0), Op(1e6, 50.0)])
        self.assertEqual(profile.memory, 9e6)
        self.assertEqual(profile.time, 350.0)

    def test_empty_layer_is_free_not_missing(self) -> None:
        self.assertEqual(chain([]), Profile(0.0, 0.0))

    def test_seconds_is_the_nanosecond_vector_in_seconds(self) -> None:
        self.assertAlmostEqual(chain([Op(0.0, 1.5e6)]).seconds, 1.5e-3, places=12)


class TestBranches(unittest.TestCase):
    def test_operands_at_one_position_are_live_together(self) -> None:
        # Two branches of two operators each: position 0 holds 2MB and 3MB at the
        # same time because the fusion downstream needs both, and it costs as
        # much as the slower of the two since eager PyTorch issues them back to
        # back and the second hides under the first only if a stream says so.
        profile = branches([[Op(2e6, 100.0), Op(5e6, 40.0)],
                            [Op(3e6, 300.0), Op(1e6, 10.0)]])
        self.assertEqual(profile.memory, 6e6)  # max(2+3, 5+1), the widest position
        self.assertEqual(profile.time, 340.0)  # max(100,300) + max(40,10)

    def test_ragged_branches_do_not_shift_positions(self) -> None:
        # The paper's m_ij is indexed by position within a branch, so a short
        # branch must stop contributing rather than let the long one's tail be
        # counted alone.
        profile = branches([[Op(4e6, 10.0), Op(4e6, 10.0), Op(4e6, 10.0)],
                            [Op(1e6, 90.0)]])
        self.assertEqual(profile.memory, 5e6)
        self.assertEqual(profile.time, 110.0)

    def test_no_branches_no_footprint(self) -> None:
        self.assertEqual(branches([]), Profile(0.0, 0.0))


class TestHybrid(unittest.TestCase):
    def test_memory_takes_the_worse_of_the_two(self) -> None:
        serial = Profile(10e6, 100.0)
        parallel = Profile(26e6, 30.0)
        profile = hybrid(serial, parallel)
        self.assertEqual(profile.memory, 26e6)
        self.assertEqual(profile.time, 130.0)  # both phases run

    def test_chaining_the_two_is_not_idempotent(self) -> None:
        # A regression guard on the asymmetry: Eq. (3) adds time but maxes memory,
        # so a hybrid of a chain with itself keeps the memory and doubles the time.
        profile = hybrid(chain([Op(1e6, 5.0)]), chain([Op(1e6, 5.0)]))
        self.assertEqual((profile.memory, profile.time), (1e6, 10.0))


def graph_with_input_edges(name: str, sizes: list[float]) -> LayerGraph:
    graph = LayerGraph([Layer(s, 0.0, 0.0) for s in ["in%d" % i for i in range(len(sizes))]
                        + [name]], [], name="fan")
    for index, size in enumerate(sizes):
        graph.add_tensor(Tensor("in%d" % index, name, size))
    return graph


class TestLayerProfile(unittest.TestCase):
    def test_operand_bytes_win_over_operator_bytes(self) -> None:
        # A cat of two 8MB inputs inside a layer whose largest single operator
        # returns 8MB: the layer is 16MB, because 16MB had to be resident for it
        # to run. Dropping this term is how a partition ends up over quota.
        graph = graph_with_input_edges("join", [8e6, 8e6])
        profile = layer_profile(graph, "join", Record([Op(8e6, 42.0)], 42.0))
        self.assertEqual(profile.memory, 16e6)
        self.assertEqual(profile.time, 42.0)

    def test_the_layer_is_timed_by_its_own_span_not_by_its_operators(self) -> None:
        # Eq. (1) would add the operator rows, and on a GPU they name less of the
        # layer than its own span measures: the gap is the host work between the
        # calls, which the slice still waits for.
        graph = graph_with_input_edges("body", [1e6])
        profile = layer_profile(graph, "body", Record([Op(1e6, 10.0),
                                                      Op(4e6, 10.0)], 90.0))
        self.assertEqual((profile.memory, profile.time), (4e6, 90.0))

    def test_work_inside_the_layer_wins_when_it_is_larger(self) -> None:
        graph = graph_with_input_edges("body", [1e6])
        record = Record([Op(1e6, 10.0), Op(4e6, 10.0)], 10.0)
        self.assertEqual(layer_profile(graph, "body", record).memory, 4e6)

    def test_a_source_node_is_priced_only_by_its_operators(self) -> None:
        graph = LayerGraph([Layer("stem", 0.0, 0.0)], [], name="single")
        record = Record([Op(2e6, 7.0)], 7.0)
        self.assertEqual(layer_profile(graph, "stem", record).memory, 2e6)


class TestDominant(unittest.TestCase):
    def test_covers_eighty_percent_from_the_top(self) -> None:
        ops = [Op(1e6, 1.0) for _ in range(8)] + [Op(10e6, 1.0), Op(10e6, 1.0)]
        picked = dominant(ops)
        # 20MB of 28MB is only 71%, so the tail of 1MB operators has to be taken
        # in as well before the set covers the paper's 80%.
        self.assertEqual(len(picked), 5)
        self.assertEqual(sum(r.out for r in picked), 23e6)

    def test_picks_the_whole_layer_at_the_tail(self) -> None:
        ops = [Op(1e6, 1.0), Op(1e6, 1.0), Op(1e6, 1.0), Op(1e6, 1.0), Op(1e6, 1.0)]
        self.assertEqual(len(dominant(ops)), 4)  # 4/5 = 80%, exactly the threshold

    def test_an_all_time_layer_has_no_dominant_operator(self) -> None:
        # An in-place activation materialises nothing, so a memory-ordered top-n
        # has nothing to rank. Its time must still survive into the service
        # vector, which is what profile_model's T does with it.
        self.assertEqual(dominant([Op(0.0, 50.0), Op(0.0, 60.0)]), [])

    def test_share_is_a_knob_not_a_constant(self) -> None:
        ops = [Op(6e6, 1.0), Op(3e6, 1.0), Op(1e6, 1.0)]
        self.assertEqual(len(dominant(ops, share=0.6)), 1)
        self.assertEqual(len(dominant(ops, share=0.99)), 3)


class TestDominantOperators(unittest.TestCase):
    def test_one_row_per_family(self) -> None:
        class L:
            def __init__(self, ops) -> None:
                self.ops = ops

        class O(Op):
            def __init__(self, op, out, ns) -> None:
                super().__init__(out, ns)
                self.op = op

            @property
            def family(self) -> str:
                return "conv" if "conv" in self.op else "relu"

        rows = dominant_operators([L([O("conv2d", 9e6, 1.0), O("relu", 0.0, 1.0)]),
                                   L([O("conv2d", 5e6, 1.0)])])
        self.assertEqual([r.family for r in rows], ["conv"])  # in-place relu: no bytes
        self.assertEqual(rows[0].out, 9e6)  # the largest instance represents it


if __name__ == "__main__":
    unittest.main()
