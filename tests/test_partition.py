"""Self-checks for the HyPAD partitioner (Algorithm 1). Pure standard library, no torch.

Run: python3 tests/test_partition.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.partition.cost import PlatformConfig, TimingModel
from slicer.partition.eliminate import (SIMILARITY_THRESHOLD, edge_elimination,
                                        node_elimination, simplify)
from slicer.partition.graph import Layer, LayerGraph, Tensor
from slicer.partition.hypad import (_cut, _simplified, partition, slice_curve,
                                   vertical_partition)


def chain(n: int, memory: float, time: float, params: float = 0.0, tensor: float = 1e6) -> LayerGraph:
    layers = [Layer(f"L{i}", memory, time, params) for i in range(n)]
    tensors = [Tensor(f"L{i}", f"L{i + 1}", tensor) for i in range(n - 1)]
    return LayerGraph(layers, tensors, name=f"chain{n}")


def ramp(n: int, memory: float = 3 * 2**30, time: float = 1e-3,
         params: float = 0.0, tensor: float = 1e3) -> LayerGraph:
    """A chain no two layers of which elimination can fold: every k stays admissible."""
    layers = [Layer(f"L{i}", memory * (1 + 0.1 * i), time, params / n) for i in range(n)]
    tensors = [Tensor(f"L{i}", f"L{i + 1}", tensor) for i in range(n - 1)]
    return LayerGraph(layers, tensors, name=f"ramp{n}")


class TestNodeElimination(unittest.TestCase):
    def test_similar_successors_fold_into_one_group(self):
        # 10 layers within 5% of each other. L0 survives: Alg. 1 needs a single *input* edge to
        # fold a node, so the source can never be eliminated -- the DP may still group it with
        # the rest, which is what test_single_slice_when_nothing_is_worth_cutting asserts.
        graph = chain(10, memory=100e6, time=1e-3)
        folded, count = node_elimination(graph)
        self.assertEqual(count, 8)
        self.assertEqual(len(folded), 2)
        survivor = folded.layer("L9")
        self.assertAlmostEqual(survivor.time, 9e-3, places=12)  # Eq. (1): times accumulate
        self.assertAlmostEqual(survivor.memory, 100e6, places=6)  # Eq. (1): memory peaks
        self.assertEqual(len(survivor.members), 9)
        self.assertEqual([(t.src, t.dst) for t in folded.tensors], [("L0", "L9")])

    def test_dissimilar_layers_are_kept_apart(self):
        graph = chain(3, memory=100e6, time=1e-3)
        graph.layers["L2"].memory = 400e6  # 75% gap from L1
        folded, count = node_elimination(graph)
        self.assertLess(count, 2)

    def test_zero_memory_layer_folds_into_neighbor(self):
        # A zero-memory layer (like inplace relu) between two 100MB convs should fold
        graph = chain(3, memory=100e6, time=1e-3)
        graph.layers["L1"].memory = 0.0  # e.g. ReLU
        folded, count = simplify(graph)
        self.assertEqual(count["nodes_folded"], 1)
        self.assertEqual(len(folded), 2)


    def test_branch_and_join_nodes_are_not_folded(self):
        # A join point with two producers must not absorb a branch: it would bill one branch's
        # memory against a node that concurrently holds both branches' activations.
        layers = [
            Layer("stem", 100e6, 1e-3),
            Layer("brA", 100e6, 1e-3),
            Layer("brB", 100e6, 1e-3),
            Layer("join", 100e6, 1e-3),
        ]
        tensors = [
            Tensor("stem", "brA", 1e6),
            Tensor("stem", "brB", 1e6),
            Tensor("brA", "join", 1e6),
            Tensor("brB", "join", 1e6),
        ]
        graph = LayerGraph(layers, tensors, "diamond")
        folded, count = node_elimination(graph)
        self.assertEqual(count, 0)
        self.assertEqual(len(folded), 4)

    def test_elimination_is_idempotent(self):
        graph = chain(12, memory=100e6, time=1e-3)
        once, _ = node_elimination(graph)
        twice, extra = node_elimination(once)
        self.assertEqual(extra, 0)
        self.assertEqual(len(twice), len(once))


class TestEdgeElimination(unittest.TestCase):
    def test_parallel_edges_merge_by_summing_bytes(self):
        layers = [Layer("a", 10e6, 1e-3), Layer("b", 10e6, 1e-3)]
        tensors = [Tensor("a", "b", 3e6), Tensor("a", "b", 4e6)]
        merged, removed = edge_elimination(LayerGraph(layers, tensors, "two-edges"))
        self.assertEqual(removed, 1)
        self.assertEqual(len(merged.tensors), 1)
        self.assertAlmostEqual(merged.tensors[0].bytes, 7e6, places=6)


class TestVerticalPartition(unittest.TestCase):
    def test_dp_places_one_cut_at_the_memory_step(self):
        # Two plateaus of similar layers separated by a 4x jump: the cut belongs inside the jump.
        layers = [Layer(f"A{i}", 100e6, 1e-3) for i in range(4)]
        layers += [Layer(f"B{i}", 400e6, 1e-3) for i in range(4)]
        tensors = []
        names = [l.name for l in layers]
        for src, dst in zip(names, names[1:]):
            tensors.append(Tensor(src, dst, 2e6))
        graph = LayerGraph(layers, tensors, "two-plateaus")

        timing = TimingModel(platform=PlatformConfig(mem_price_per_gb_s=1.667e-5))
        groups, simplified, _ = vertical_partition(graph, timing)
        self.assertEqual(len(groups), 2)
        first, second = groups
        self.assertTrue(all(n.startswith("A") for n in first))
        self.assertTrue(all(n.startswith("B") for n in second))

    def test_single_slice_when_nothing_is_worth_cutting(self):
        # Free communication and a uniform chain: elimination already removes every cut point.
        graph = chain(8, memory=100e6, time=1e-3)
        plan = partition(graph, platform=PlatformConfig())
        self.assertEqual(plan.n_slices, 1)
        self.assertEqual(len(plan.slices[0].members), 8)


class TestParallelism(unittest.TestCase):
    def test_eta_is_capped_by_the_memory_to_vcpu_ratio(self):
        # λ of Eq. (6) is a memory-to-vCPU ratio, so what caps eta is the whole quota a
        # function has to hold: the activation peak *and* the weights it loads.
        platform = PlatformConfig(mb_per_vcpu=1792.0)
        # 8192 MB of activations alone -> 8192 MB -> 4 sub-slices.
        activations = chain(1, memory=8192 * 2**20, time=1e-3)
        self.assertEqual(partition(activations, platform=platform).slices[0].eta, 4)
        # The same layer plus 1 GB of parameters -> 9152 MB -> 5.
        with_weights = chain(1, memory=8192 * 2**20, time=1e-3, params=1e9)
        self.assertEqual(partition(with_weights, platform=platform).slices[0].eta, 5)

    def test_the_vcpu_cap_belongs_to_the_slice_not_the_service(self):
        """Eq. (6) reads ``1 <= eta <= M_gi / lambda`` with ``M_gi`` *this* slice's quota.

        12 GB of weights in six layers: the service affords six vCPUs, a slice of it
        affords one. Reading the cap off the service instead lets the search shard a
        2 GB slice six ways -- and since sharding is what divides the parameters, it
        also lets the slice bill itself for the sixth of the weights it does not
        hold, so a plan appeared to buy parallelism by shrinking the very quantity
        that pays for it.
        """
        platform = PlatformConfig(mem_price_per_gb_s=1.667e-5)
        graph = ramp(6, memory=64 * 2**20, time=1e-3, params=12 * 2**30, tensor=1e3)

        whole = partition(graph, platform=platform, slices=1).slices[0]
        self.assertGreater(whole.eta, 1)
        split = partition(graph, platform=platform, slices=6)
        self.assertEqual({s.eta for s in split.slices}, {1})
        for plan in (whole, *split.slices):
            self.assertLessEqual(plan.eta * platform.mb_per_vcpu, plan.alloc_mb)

    def test_sharding_cannot_make_the_service_slower_than_unsplit(self):
        # Eq. (6)'s constraint, exercised with an aggregation term large enough to fight back.
        platform = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=0.0)
        graph = chain(6, memory=4096 * 2**20, time=5e-3, params=1e9, tensor=50e6)
        timing = TimingModel(platform=platform)
        plan = partition(graph, timing=timing, platform=platform)
        self.assertLessEqual(plan.latency_s, plan.unsplit_latency_s * 1.0001)


class TestCostAccounting(unittest.TestCase):
    def test_heterogeneous_layers_save_money_by_splitting(self):
        """The paper's actual claim: each slice is billed for *its own* quota, not the model peak.

        A 6 GB layer followed by a 3 GB layer costs 6 GB x 10 ms unsplit. Cut between them and the
        bill is 6 GB x 5 ms + 3 GB x 5 ms, i.e. 25% less, which is where MOPAR's savings come from.

        The slice count is pinned because with ``mb_per_vcpu`` out of reach no slice can
        be sharded, so every cut leaves the service slower than the unsplit one and
        Eq. (6)'s constraint rules it out -- see :class:`TestLatencyConstraint`.
        """
        platform = PlatformConfig(
            mem_price_per_gb_s=1.667e-5,
            net_price_per_s=0.0,
            alloc_granularity_mb=64.0,
            # mb_per_vcpu is set so high that eta is pinned to 1: this test isolates the quota
            # effect from the parallelism effect, which test_parallelism_lowers_billed_time owns.
            mb_per_vcpu=1e12,
        )
        timing = TimingModel(platform=platform)
        graph = LayerGraph(
            [Layer("big", 6 * 2**30, 5e-3), Layer("small", 3 * 2**30, 5e-3)],
            [Tensor("big", "small", 1e3)],
            "heterogeneous",
        )
        plan = partition(graph, timing=timing, platform=platform, slices=2)
        self.assertEqual(plan.n_slices, 2)
        self.assertAlmostEqual(plan.cost_usd / plan.unsplit_cost_usd, 0.75, places=3)

    def test_parallelism_lowers_billed_time(self):
        # Horizontal sharding shortens each slice's execution, and since billing is memory x time
        # it lowers the bill too -- a second, independent source of MOPAR's savings.
        tight = PlatformConfig(mem_price_per_gb_s=1.667e-5, mb_per_vcpu=1024.0)
        pinned = PlatformConfig(mem_price_per_gb_s=1.667e-5, mb_per_vcpu=1e12)
        graph = chain(4, memory=4096 * 2**20, time=5e-3, params=1e9, tensor=1e3)
        fast = partition(graph, timing=TimingModel(platform=tight), platform=tight)
        slow = partition(graph, timing=TimingModel(platform=pinned), platform=pinned)
        self.assertGreater(fast.slices[0].eta, 1)
        self.assertLess(fast.latency_s, slow.latency_s)

    def test_uniform_layers_do_not_save_money(self):
        # Two identical 3 GB layers: splitting cannot lower the bill, only add hops. The DP should
        # say so, and MOPAR must not claim a saving here.
        platform = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=0.0)
        graph = LayerGraph(
            [Layer("A", 3 * 2**30, 5e-3), Layer("B", 3 * 2**30, 5e-3)],
            [Tensor("A", "B", 1e3)],
            "uniform",
        )
        plan = partition(graph, timing=TimingModel(platform=platform), platform=platform)
        self.assertAlmostEqual(plan.cost_usd, plan.unsplit_cost_usd, places=9)

    def test_a_cut_is_paid_for_by_the_hop_it_adds(self):
        """Cuts are not free, even with cn = 0.

        Eq. (6) prices transmission through a per-second rate that neither of our
        platforms quotes, so the hop is billed as the thing it actually is: the
        receiving slice holds its allocation while it waits. A 2 GB activation
        across one boundary therefore has to be worth the memory it saves, and a
        1 KB one does not -- which is what puts an interior optimum in the slice
        count instead of "cut everywhere".

        Compared at pinned counts rather than through :func:`partition`, because it is
        the two plans' bills that are being weighed, and the free search would drop the
        wider one on Eq. (6)'s latency constraint before the money was ever looked at.
        """
        platform = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=0.0)
        timing = TimingModel(platform=platform)

        def gap(across: float) -> float:
            """What one cut costs this service: k = 2 minus k = 1."""
            graph = LayerGraph(
                [Layer("big", 3 * 2**30, 5e-3), Layer("small", 1 * 2**30, 5e-3)],
                [Tensor("big", "small", across)],
                "hop-priced",
            )
            return (partition(graph, timing, platform, slices=2).cost_usd
                    - partition(graph, timing, platform, slices=1).cost_usd)

        self.assertLess(gap(1e3), 0)     # the quota saved beats the hop
        self.assertGreater(gap(2e9), 0)  # and a 2 GB hop beats the quota

    def test_price_per_second_of_transmission_penalises_extra_slices(self):
        # mb_per_vcpu is out of reach in both, so the only difference between them is
        # cn: with eta in play a merged slice could buy its way back to cheaper by
        # sharding, and this test is about the price of the hop, not of the cores.
        free = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=0.0,
                              mb_per_vcpu=1e12)
        costly = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=1.0,
                                mb_per_vcpu=1e12)
        layers = [Layer("big", 6 * 2**30, 5e-3), Layer("small", 3 * 2**30, 5e-3)]
        # Small enough that the hop's own wait still leaves the cut worth making: this
        # test is about cn, and the memory-billed wait is tested next to it.
        graph = LayerGraph(layers, [Tensor("big", "small", 20e6)], "priced-comm")

        def gap(platform: PlatformConfig) -> float:
            timing = TimingModel(platform=platform)
            return (partition(graph, timing, platform, slices=2).cost_usd
                    - partition(graph, timing, platform, slices=1).cost_usd)

        self.assertLess(gap(free), 0)
        self.assertGreater(gap(costly), gap(free))
        self.assertGreater(partition(graph, TimingModel(platform=costly), costly).latency_s, 0)


class TestSliceCurve(unittest.TestCase):
    """Fig. 2b/2d: the plan *at* k slices, which is what a curve in k needs."""

    def setUp(self) -> None:
        # mb_per_vcpu is out of reach, so eta is pinned to 1 everywhere and the only
        # thing a larger k changes is the cuts. That keeps the two assertions below
        # about the DP rather than about the parallelism search.
        self.platform = PlatformConfig(mem_price_per_gb_s=1.667e-5, net_price_per_s=0.0,
                                       mb_per_vcpu=1e12)
        self.timing = TimingModel(platform=self.platform)
        self.graph = ramp(6, params=1e9, tensor=1e3)

    def test_pinning_k_gives_exactly_k_groups_that_cover_the_service(self) -> None:
        plans = slice_curve(self.graph, [1, 2, 3, 6], self.timing, self.platform)
        self.assertEqual([1, 2, 3, 6], [p.n_slices for p in plans])
        for plan in plans:
            self.assertEqual(6, sum(len(s.members) for s in plan.slices))
            self.assertEqual(sorted(self.graph.order),
                             sorted(m for s in plan.slices for m in s.members))

    def test_more_slices_cannot_shorten_the_path(self) -> None:
        plans = slice_curve(self.graph, list(range(1, 7)), self.timing, self.platform)
        latencies = [p.latency_s for p in plans]
        self.assertEqual(sorted(latencies), latencies)

    def test_pinned_optima_and_the_free_one_agree(self) -> None:
        """min over k of the exact-k optimum must be the cost-only DP's answer.

        The two DPs are separate code paths -- one sweep per k against one sweep
        over all k -- so this is the check that the 2-D table is not quietly
        returning something other than "the best plan with k slices". The 1-D DP no
        longer answers :func:`partition`, which reads Eq. (6)'s constraint first, but
        it is still what :func:`vertical_partition` returns.
        """
        agg, _ = _simplified(self.graph, self.timing, SIMILARITY_THRESHOLD)
        free = sum(agg.objective(agg.position[group[0]], agg.position[group[0]] + len(group))
                   for group in _cut(agg, None))
        pinned = slice_curve(self.graph, list(range(1, 7)), self.timing, self.platform)
        self.assertAlmostEqual(free, min(p.cost_usd for p in pinned), places=12)

    def test_a_request_beyond_the_node_count_is_answered_at_the_maximum(self) -> None:
        plan = partition(self.graph, self.timing, self.platform, slices=99)
        self.assertEqual(6, plan.n_slices)

    def test_the_free_optimum_is_the_cheapest_point_on_its_own_curve(self) -> None:
        """Alg. 1 must not return a plan that a pinned k beats on the metric it optimises.

        It used to, once eta entered the picture: the DP priced every group at eta = 1
        while the plan was billed at whatever eta the search afterwards chose, so on a
        service big enough for lambda of Eq. (6) to allow sharding, the "optimal"
        placement cost more than one slice with eta > 1 did. The two DPs now read the
        same per-group optimum, which is what this compares.
        """
        platform = PlatformConfig(mem_price_per_gb_s=1.667e-5)
        timing = TimingModel(platform=platform)
        graph = ramp(8, params=8e9, tensor=1e6)
        free = partition(graph, timing, platform)
        curve = slice_curve(graph, list(range(1, 9)), timing, platform)
        self.assertGreater(max(s.eta for s in free.slices), 1)
        self.assertAlmostEqual(min(p.stats["dp_cost_usd"] for p in curve),
                               free.stats["dp_cost_usd"], places=15)

    def test_an_empty_or_impossible_curve_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            slice_curve(self.graph, [], self.timing, self.platform)
        with self.assertRaises(ValueError):
            slice_curve(self.graph, [0], self.timing, self.platform)


class TestLatencyConstraint(unittest.TestCase):
    """Eq. (6)'s ``t_tau <= sum t_li``, which is the only thing that prices a cut.

    The objective alone has no interior optimum in the slice count on a platform that
    charges nothing for transmission: every extra cut moves weight into a function
    billed for less of it, so the free DP answered 51 slices for BERT-1.3B at a latency
    slower than the unsplit service, and called that the optimum. The constraint is
    part of what Alg. 1 solves, so these tests hold it to its own definition.
    """

    def setUp(self) -> None:
        self.platform = PlatformConfig(mem_price_per_gb_s=1.667e-5)
        self.timing = TimingModel(platform=self.platform)

    def test_a_service_that_cannot_be_sharded_is_not_cut(self) -> None:
        """With every eta pinned to 1, a cut adds communication and buys nothing.

        ``mb_per_vcpu`` is set beyond any slice's quota, which is the situation a
        500 MB CNN is in on a platform handing out one vCPU per 1792 MB: the sliced
        service then runs every layer once, sequentially, plus the hops -- strictly
        over Eq. (6)'s budget.
        """
        pinned = PlatformConfig(mem_price_per_gb_s=1.667e-5, mb_per_vcpu=1e12)
        graph = LayerGraph(
            [Layer("big", 6 * 2**30, 5e-3), Layer("small", 3 * 2**30, 5e-3)],
            [Tensor("big", "small", 1e3)],
            "unshardable",
        )
        plan = partition(graph, TimingModel(platform=pinned), pinned)
        self.assertEqual(1, plan.n_slices)
        self.assertEqual([1], plan.stats["admissible_slices"])
        self.assertFalse(plan.stats["latency_over_budget"])
        cut = partition(graph, TimingModel(platform=pinned), pinned, slices=2)
        self.assertTrue(cut.stats["latency_over_budget"])
        self.assertLess(cut.cost_usd, plan.cost_usd)  # cheaper, and still no solution

    def test_a_plan_that_shards_instead_of_cutting_is_admissible(self) -> None:
        """The other half of the rule: where parallelism is available, the slack is real.

        8 GB of weights over eight layers, so one function can shard itself seven ways
        and finish in 1.5 ms of a 8 ms budget. What the constraint buys is then spent
        on nothing -- but note which k wins: the sharded single slice, not a partition.
        On a service whose first slice can already afford the sub-slices, Eq. (6) has
        no reason to cut at all, and this asserts that the search says so rather than
        cutting to spend money it does not need.
        """
        graph = ramp(8, params=8e9, tensor=1e6)
        plan = partition(graph, self.timing, self.platform)
        self.assertGreater(plan.slices[0].eta, 1)
        self.assertEqual(1, plan.n_slices)
        self.assertLessEqual(plan.latency_s, plan.unsplit_latency_s * (1 + 1e-9))
        self.assertLess(plan.cost_usd, plan.unsplit_cost_usd)

    def test_the_answer_is_the_cheapest_admissible_point_on_the_curve(self) -> None:
        graph = ramp(8, params=8e9, tensor=1e6)
        plan = partition(graph, self.timing, self.platform)
        curve = slice_curve(graph, list(range(1, 9)), self.timing, self.platform)
        admissible = [p for p in curve if not p.stats["latency_over_budget"]]
        self.assertTrue(admissible)
        self.assertAlmostEqual(min(p.cost_usd for p in admissible), plan.cost_usd, places=12)
        self.assertIn(plan.n_slices, [p.n_slices for p in admissible])

    def test_the_search_stops_where_the_constraint_binds(self) -> None:
        """The scan is over the pinned-count DP, so it must not walk to the node count.

        Each row costs O(n^2) comparisons, and a service whose admissible range is a
        handful of slice counts has no business pricing sixty-four of them. 200 MB per
        hop is the case where every extra cut is slower than the budget, so only the
        constraint -- not an exhausted table -- ends the search.
        """
        graph = ramp(40, params=4e9, tensor=200e6)
        plan = partition(graph, self.timing, self.platform)
        self.assertLess(plan.stats["searched_slices"], plan.stats["nodes_after"])
        self.assertFalse(plan.stats["search_truncated"])
        self.assertLessEqual(plan.n_slices, plan.stats["searched_slices"])
        self.assertLessEqual(plan.latency_s, plan.unsplit_latency_s * (1 + 1e-9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
