"""HyPAD: the hybrid partitioning algorithm of Sec. II-C (Algorithm 1).

Three steps, matching the paper: graph simplification by node/edge elimination, vertical split
points by dynamic programming over the simplified graph, then a parallelism search per slice.
The DP minimises Eq. (6)'s objective, and the equation's constraints are what turn the result
back into a plan: :func:`_admissible` chooses the slice count so that ``t_tau <= sum t_li``
holds, and :func:`_price` chooses each slice's eta within ``1 <= eta <= M_gi / lambda``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost import PlatformConfig, TimingModel, slice_footprint
from .eliminate import SIMILARITY_THRESHOLD, simplify
from .graph import LayerGraph


@dataclass(frozen=True)
class SlicePlan:
    """One vertical slice: consecutive nodes of the simplified graph, one function each."""

    nodes: tuple[str, ...]
    members: tuple[str, ...]
    alloc_mb: float
    eta: int
    compute_s: float
    aggregate_s: float
    comm_out_s: float
    comm_out_bytes: float
    cost_usd: float

    @property
    def time_s(self) -> float:
        return self.compute_s + self.aggregate_s + self.comm_out_s

    @property
    def memory_consumption_gb_s(self) -> float:
        """``M_gi * (tp + ta + tc)``: the MC quantity of Sec. III-C2, before any price.

        The paper's Eq. (6) bills memory against ``tp + ta`` and the hop separately
        through ``cn``, a price per second of transmission. Neither of our two
        platforms quotes one -- Lambda charges nothing for transfer, and a
        bare-metal cluster charges the *time* a function holds its allocation while
        its successor collects the tensor. So the hop is billed here, inside the
        memory product, which is also what Sec. III-C2's own definition says: MC is
        allocated memory times execution time, and Eq. (5)'s execution time is
        ``tp + ta + tc``. Leaving it out makes a cut free, and a free cut is taken
        everywhere: the DP returned 225 slices for a 722-node service.
        """
        return self.alloc_mb / 1024.0 * self.time_s


@dataclass
class PartitionPlan:
    slices: list[SlicePlan]
    latency_s: float
    unsplit_latency_s: float
    unsplit_cost_usd: float
    cost_usd: float
    stats: dict = field(default_factory=dict)

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def max_alloc_mb(self) -> float:
        return max((s.alloc_mb for s in self.slices), default=0.0)

    @property
    def comm_bytes(self) -> float:
        return sum(s.comm_out_bytes for s in self.slices)

    def summary(self) -> str:
        rows = [
            f"  slice{i+1}: {len(s.members):>3} layers  alloc={s.alloc_mb:>7.0f}MB  "
            f"eta={s.eta}  compute={s.compute_s*1e3:8.2f}ms  agg={s.aggregate_s*1e3:7.3f}ms  "
            f"comm={s.comm_out_s*1e3:7.3f}ms  {s.members[0]}..{s.members[-1]}"
            for i, s in enumerate(self.slices)
        ]
        return "\n".join(
            [
                f"slices={self.n_slices} latency={self.latency_s*1e3:.2f}ms "
                f"(unsplit {self.unsplit_latency_s*1e3:.2f}ms) "
                f"cost={self.cost_usd*1e6:.2f}us (unsplit {self.unsplit_cost_usd*1e6:.2f}us)"
            ]
            + rows
        )


def _boundary_bytes(graph: LayerGraph, order: list[str], cut: int) -> float:
    """Bytes that must cross the boundary between ``order[:cut]`` and ``order[cut:]``."""
    head = set(order[:cut])
    tail = set(order[cut:])
    return sum(t.bytes for t in graph.tensors if t.src in head and t.dst in tail)


class _Aggregates:
    """Every quantity the DP needs about a range of the topological order, in O(1).

    The paper's Alg. 1 is written for a chain and assumes a group's footprint can be read off
    as cheaply as its length. On a graph with parallel branches neither is a prefix sum: the
    memory of a group is the *maximum* over its nodes, and its outgoing bytes are every edge
    that leaves the range, so both are precomputed here once. Without this the DP calls
    :func:`_boundary_bytes` inside its inner loop and a 722-node service with 11k edges costs
    eight billion set operations per partition.

    The per-node timings are tabulated against eta for the same reason. Eq. (6) sums over
    slices *and* their sharding, so a DP that placed cuts at eta = 1 and let a later pass
    choose eta was optimising a different function from the one the plan is billed on --
    with a 128 MB floor and λ = 1792 MB/vCPU the two rankings disagree outright, and the
    free optimum lost to a single slice on the cost it was supposed to minimise.
    """

    def __init__(self, graph: LayerGraph, timing: TimingModel, order: list[str]) -> None:
        self.graph = graph
        self.timing = timing
        self.platform = timing.platform
        self.order = order
        self.position = {name: index for index, name in enumerate(order)}
        n = len(order)
        self.params = [0.0] * (n + 1)
        for i, name in enumerate(order, start=1):
            self.params[i] = self.params[i - 1] + graph.layer(name).params
        # mem[i][j] = max layer memory over order[i:j]; a sparse table would be O(n log n)
        # instead of O(n^2) memory, but n is a few hundred and this reads as what it means.
        mem = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            row = mem[i]
            best = 0.0
            for j in range(i + 1, n + 1):
                value = graph.layer(order[j - 1]).memory
                if value > best:
                    best = value
                row[j] = best
        self.mem = mem
        self.cross = [_boundary_bytes(graph, order, j) for j in range(n + 1)]
        # The widest ceiling any group can reach is the whole graph's, so one grid covers
        # every range the DP will ask about.
        self.grid = range(1, self.platform.max_subslices(
            self.platform.allocate_mb(self.params[n] + self.mem[0][n])) + 1)
        # compute_time is per node and per eta. aggregate_time is too, but it is also
        # linear in the bytes gathered (both models move out*(eta-1)/eta), so it is
        self.compute = []
        self.gather = []
        for eta in self.grid:

            compute = [0.0] * (n + 1)
            gather = [0.0] * (n + 1)
            for i, name in enumerate(order, start=1):
                layer = graph.layer(name)
                compute[i] = compute[i - 1] + timing.compute_time(layer, eta)
                gather[i] = gather[i - 1] + timing.aggregate_time(layer, eta, 1.0)
            self.compute.append(compute)
            self.gather.append(gather)


        self._best: dict[tuple[int, int], tuple[int, float]] = {}

    def footprint(self, i: int, j: int) -> float:
        """Eq. (4)'s ``M_{gi}`` for one group: the weights it holds plus its widest activation.

        Independent of eta by construction: sharding splits the parameters between
        sub-slices, it does not make the slice hold fewer bytes.
        """
        return (self.params[j] - self.params[i]) + self.mem[i][j]

    def cap(self, i: int, j: int) -> int:
        """Eq. (6)'s ``1 <= eta <= M_gi / lambda``, for this group.

        The cap is per slice, not per service: a 128 MB slice buys one vCPU and so
        cannot be sharded at all, while the 11 GB one it came from buys six. A search
        that read the cap off the whole graph -- as the first version of this one did
        -- returned eta = 6 for a 1856 MB slice, i.e. a plan that asks a platform for
        six vCPUs while paying for six hundred and forty.
        """
        return self.platform.max_subslices(self.platform.allocate_mb(self.footprint(i, j)))

    def plan(self, i: int, j: int, eta: int | None = None) -> SlicePlan:
        """Alg. 1 lines 19-23 for the group ``order[i:j]``, at ``eta`` or at its best."""
        if eta is None:
            eta = self.search(i, j)[0]
        compute = self.compute[eta - 1][j] - self.compute[eta - 1][i]
        across = self.cross[j]
        gathered = (self.gather[eta - 1][j] - self.gather[eta - 1][i]) * across
        return SlicePlan(
            nodes=tuple(self.order[i:j]),
            members=tuple(m for name in self.order[i:j]
                          for m in self.graph.layer(name).members),
            alloc_mb=self.platform.allocate_mb(self.footprint(i, j)),
            eta=eta,
            compute_s=compute,
            aggregate_s=gathered,
            comm_out_s=self.timing.comm_time(across, same_host=True),
            comm_out_bytes=across,
            cost_usd=self.cost(i, j, eta),
        )

    def search(self, i: int, j: int) -> tuple[int, float]:
        """The cheapest ``(eta, cost)`` for the group ``order[i:j]``.

        Communication time does not depend on eta, so minimising each slice's own
        ``tp + ta`` independently is exact rather than a heuristic. What makes the
        choice non-trivial is that ``tp`` falls with eta while ``ta`` -- the parameter
        aggregation Eq. (5) also charges -- rises with it, so the cheapest sharding of
        a slice is interior whenever it has a wide boundary to gather across, and the
        quota it is billed against does not move at all.

        Only the scalars are memoised, and not the slice itself: the DP asks about
        every one of the O(n^2) ranges, and a materialised slice carries a tuple of
        the node names and another of the layers behind them, which is cubic memory
        on a 722-node service.
        """
        cached = self._best.get((i, j))
        if cached is None:
            cached = min(((eta, self.cost(i, j, eta))
                          for eta in range(1, self.cap(i, j) + 1)),
                         key=lambda pair: pair[1])
            self._best[(i, j)] = cached
        return cached

    def cost(self, i: int, j: int, eta: int) -> float:
        """Eq. (6) for the group ``order[i:j]`` at one eta, including the hop it sends."""
        across = self.cross[j]
        tc = self.timing.comm_time(across, same_host=True)
        slot = eta - 1  # grid == range(1, ...)
        compute = self.compute[slot][j] - self.compute[slot][i]
        gathered = (self.gather[slot][j] - self.gather[slot][i]) * across
        alloc_mb = self.platform.allocate_mb(self.footprint(i, j))

        memory = alloc_mb / 1024.0 * (compute + gathered + tc) * self.platform.mem_price_per_gb_s
        return memory + self.platform.net_price_per_s * tc

    def objective(self, i: int, j: int) -> float:
        """Eq. (6) for the group, at the eta that suits it best."""
        return self.search(i, j)[1]


def _cut(agg: _Aggregates, slices: int | None) -> list[list[str]]:
    """The split points: the free optimum, or the best one that uses exactly ``slices``."""
    order = agg.order
    n = len(order)
    if slices is None:
        inf = float("inf")
        dp = [inf] * (n + 1)
        dp[0] = 0.0
        origin = [0] * (n + 1)
        for j in range(1, n + 1):
            best, best_i = dp[j], origin[j]
            for i in range(1, j + 1):
                if dp[i - 1] == inf:
                    continue
                candidate = dp[i - 1] + agg.objective(i - 1, j)
                if candidate < best:
                    best, best_i = candidate, i - 1
            dp[j], origin[j] = best, best_i
        return _trace(origin, agg, n)
    return _CutTable(agg, slices).groups(slices)


#: How many inadmissible slice counts the constrained search looks past before it
#: concludes Eq. (6)'s latency constraint has bound. The constraint is not proved
#: monotone in k here -- a wider cut can hand a group back an eta it lost -- so the
#: scan does not stop at the first miss.
CONSTRAINT_PATIENCE = 8

#: Relative slack on ``t_tau <= sum t_li``. A one-slice plan at eta = 1 sums the very
#: layer times the budget is made of, through a different accumulator, so without
#: slack the knife-edge case fails on one ULP.
LATENCY_TOLERANCE = 1e-9

#: How far :func:`_admissible` is willing to scan slice counts. The table it walks
#: costs O(n^2) comparisons per row, and no plan the paper reports is anywhere near
#: this, so the cap is a stop-loss on a pathological profile rather than a knob that
#: shapes an answer; ``stats["search_truncated"]`` says if it was the cap that ended
#: the search rather than the constraint.
MAX_SLICE_SEARCH = 64


class _CutTable:
    """The DP over (prefix, slice count), so one build answers every ``k`` up to ``limit``.

    Fig. 2b/2d are curves in the slice count, and Fig. 11 is a histogram of it, so the
    reproduction needs the optimum *at* k rather than only the free one -- and needs them
    from one profiling pass, not one per k.

    Rows are built on demand, because a caller that only wants small k should not pay for
    the table up to the node count: row s costs O(n^2) group evaluations, and the
    constrained search of :func:`_admissible` stops as soon as the latency constraint
    binds. The groups of :class:`_Aggregates` are memoised, so the rows after the first
    add comparisons rather than eta searches.
    """

    def __init__(self, agg: _Aggregates, limit: int | None = None) -> None:
        self.agg = agg
        n = len(agg.order)
        self.n = n
        self.limit = max(1, min(n, n if limit is None else limit))
        first: list[float] = [float("inf")] * (n + 1)
        first[0] = 0.0
        self.cost: list[list[float]] = [first]
        self.origin: list[list[int]] = [[0] * (n + 1)]

    def _row(self, s: int) -> None:
        """Fill DP row ``s`` from row ``s - 1``: the cheapest prefix of j at exactly s groups."""
        inf = float("inf")
        previous = self.cost[s - 1]
        current: list[float] = [inf] * (self.n + 1)
        origins = [0] * (self.n + 1)
        objective = self.agg.objective
        for j in range(s, self.n + 1):
            best, best_i = inf, 0
            for i in range(s - 1, j):
                if previous[i] == inf:
                    continue
                candidate = previous[i] + objective(i, j)
                if candidate < best:
                    best, best_i = candidate, i
            current[j] = best
            origins[j] = best_i
        self.cost.append(current)
        self.origin.append(origins)

    def groups(self, k: int) -> list[list[str]]:
        """The split points of the best plan with exactly ``k`` groups.

        Tracing a 2-D DP walks one row per group: from the state ``(s, j)`` the
        predecessor lives in row ``s - 1`` at the column the last cut ended on. A
        request beyond the node count is answered at ``self.limit``, the largest
        admissible k, and the plan's ``stats["slices"]`` reports what was built.
        """
        if k < 1:
            raise ValueError("a plan has at least one slice")
        k = min(k, self.limit)
        while len(self.cost) <= k:
            self._row(len(self.cost))
        groups: list[list[str]] = []
        j = self.n
        for s in range(k, 0, -1):
            i = self.origin[s][j]
            groups.append(self.agg.order[i:j])
            j = i
        groups.reverse()
        return [g for g in groups if g]


def _trace(origin: list[int], agg: _Aggregates, j: int) -> list[list[str]]:
    groups: list[list[str]] = []
    while j > 0:
        i = origin[j]
        groups.append(agg.order[i:j])
        j = i
    groups.reverse()
    return [g for g in groups if g]


def vertical_partition(
    graph: LayerGraph,
    timing: TimingModel,
    similarity: float = SIMILARITY_THRESHOLD,
    slices: int | None = None,
) -> tuple[list[list[str]], LayerGraph, dict]:
    """Algorithm 1 lines 1-18: simplify, then DP for the split points.

    Groups are contiguous ranges of the topological order. The paper's DP is written for a chain;
    for graphs with parallel branches this keeps every branch inside one slice unless a cut is
    placed before the branch or after the join, which is the same restriction Alg. 1 implies.
    ``slices`` pins the number of groups, which is what a curve in the slice count needs; the
    free optimum is the default.
    """
    agg, stats = _simplified(graph, timing, similarity)
    return _cut(agg, slices), agg.graph, stats


def optimize_parallelism(agg: _Aggregates, groups: list[list[str]]) -> list[SlicePlan]:
    """Algorithm 1 lines 19-23: per slice, the eta that minimises Eq. (6)'s terms.

    Kept as its own step because that is how the paper writes it, but the eta search
    lives in :class:`_Aggregates`, where the cut-placement DP can see its result. Eq. (6)
    sums over slices *and* their sharding, so a placement pass blind to eta minimises a
    different function from the one the plan is billed on.
    """
    slices: list[SlicePlan] = []
    for group in groups:
        start = agg.position[group[0]]
        slices.append(agg.plan(start, start + len(group)))
    return slices


def evaluate(slices: list[SlicePlan]) -> tuple[float, float]:
    """Eq. (6): ``min sum(M_gi*(tp+ta)*cm) + cn*sum(tc)``, returned as (seconds, dollars)."""
    latency = sum(s.time_s for s in slices)
    cost = sum(s.cost_usd for s in slices)
    return latency, cost


def _price(agg: _Aggregates, groups: list[list[str]],
           budget: float) -> tuple[list[SlicePlan], float, float, int]:
    """Search eta per slice, then enforce Eq. (6)'s latency constraint on the result.

    Returns ``(slices, latency, cost, backoff_steps)``. The DP minimises cost and never
    looks at latency, so a plan that overshoots the budget is repaired here by taking
    back the parallelism that did not pay: an eta buys compute time at the price of the
    parameter aggregation of Eq. (5), and past the point where the two cross, lowering
    eta moves latency *down*. Cuts are not undone by this loop -- that decision belongs
    to :func:`_admissible`, which is where the slice count is chosen.
    """
    slices = optimize_parallelism(agg, groups)
    latency, cost = evaluate(slices)
    backed_off = 0
    if budget > 0:
        while latency > budget:
            widest = max(range(len(slices)), key=lambda i: slices[i].eta, default=None)
            if widest is None or slices[widest].eta <= 1:
                break
            held = slices[widest]
            start = agg.position[held.nodes[0]]
            slices[widest] = agg.plan(start, start + len(held.nodes), eta=held.eta - 1)
            backed_off += 1
            latency, cost = evaluate(slices)
    return slices, latency, cost, backed_off


def _admissible(graph: LayerGraph, agg: _Aggregates, stats: dict) -> PartitionPlan:
    """Alg. 1's answer as written: the cheapest plan that satisfies Eq. (6)'s constraint.

    The paper's second constraint is ``t_tau(G, pi_S, pi_P) <= sum t_li`` -- "ensuring
    that the total latency with MP does not exceed the original latency without MP" --
    and it is the only thing in the formulation that prices a cut, because ``cn`` (the
    price per second of transmission) is unpublished and on neither platform we
    reproduce does anyone charge for transfer. Minimising cost alone is therefore
    unbounded in the slice count: the DP returned 51 slices for BERT-1.3B at a latency
    3.5% over budget, which the paper would not call a solution.

    So this is a scan over k with the constraint checked at each k, keeping the
    cheapest admissible plan and, on a tie, the fewest slices. A scan rather than a
    Lagrangian or a latency-indexed DP because the constraint couples the slices only
    through their sum, the pinned-count table is already built for Fig. 2b/2d, and its
    rows are generated on demand -- so the search costs the rows up to where the
    constraint binds, and nothing more.

    What makes the constraint bite is lambda. A slice may be sharded into at most
    ``alloc/1792`` sub-slices, so once the cuts have made every slice smaller than one
    vCPU's worth of memory each eta is 1, the sliced service then runs at exactly
    ``sum t_li`` plus the hops it added, and is inadmissible. Eq. (6) hence admits a
    cut only where the pieces are big enough to buy parallelism: for the paper's zoo
    that is the 4-10 GB transformers, at 2-5 slices, which is Fig. 11's range. That
    the eight CPU-side models come out unsplit is the paper's own inconsistency, not
    this implementation's -- see ``issues/hypad-cut-price.md``.
    """
    budget = graph.total_time()
    if budget <= 0:
        # Nothing to compare against: a service profile with no timings at all. Fall
        # back to the cost-only optimum and let ``latency_over_budget`` say so.
        return _assemble(graph, agg, _cut(agg, None), stats)
    table = _CutTable(agg, MAX_SLICE_SEARCH)
    limit = budget * (1 + LATENCY_TOLERANCE)
    best: tuple[float, int] | None = None
    admissible: list[int] = []
    misses = 0
    bound = False
    for k in range(1, table.limit + 1):
        _, latency, cost, _ = _price(agg, table.groups(k), budget)
        if latency <= limit:
            admissible.append(k)
            misses = 0
            if best is None or cost < best[0]:
                best = (cost, k)
        else:
            misses += 1
            if best is not None and misses >= CONSTRAINT_PATIENCE:
                bound = True
                break
    if best is None:  # not even one slice fits; report the cheapest thing measured
        best = (float("inf"), 1)
    return _assemble(graph, agg, table.groups(best[1]),
                     stats | {"admissible_slices": admissible, "searched_slices": k,
                              # Did the constraint end the search, or did the scan run
                              # out of slice counts with more still admissible?
                              "search_truncated": (not bound and table.limit >= MAX_SLICE_SEARCH)})


def _assemble(
    graph: LayerGraph,
    agg: _Aggregates,
    groups: list[list[str]],
    stats: dict,
) -> PartitionPlan:
    """Turn split points into a plan: search eta per slice, then price the result."""
    platform = agg.platform
    budget = graph.total_time()
    slices, latency, cost, backed_off = _price(agg, groups, budget)

    unsplit_latency = budget
    unsplit_alloc = platform.allocate_mb(slice_footprint(graph, graph.order))
    unsplit_cost = unsplit_alloc / 1024.0 * unsplit_latency * platform.mem_price_per_gb_s
    stats = stats | {
        "nodes_after": len(agg.order),
        "slices": len(slices),
        "latency_backoff_steps": backed_off,
        "latency_over_budget": latency > budget * (1 + LATENCY_TOLERANCE),
        "latency_budget_s": budget,
        # What the DP was minimising, before the latency repair above spent any of
        # it back. Equal to ``cost_usd`` unless a slice's eta had to come down.
        "dp_cost_usd": sum(s.cost_usd for s in optimize_parallelism(agg, groups)),
    }
    return PartitionPlan(
        slices=slices,
        latency_s=latency,
        unsplit_latency_s=unsplit_latency,
        unsplit_cost_usd=unsplit_cost,
        cost_usd=cost,
        stats=stats,
    )


def _simplified(
    graph: LayerGraph, timing: TimingModel, similarity: float
) -> tuple[_Aggregates, dict]:
    """Alg. 1 lines 1-4: fold the graph, then build the table the DP reads from it."""
    simplified, stats = simplify(graph, similarity)
    order = simplified.topological_order()
    return _Aggregates(simplified, timing, order), stats


def partition(
    graph: LayerGraph,
    timing: TimingModel | None = None,
    platform: PlatformConfig | None = None,
    similarity: float = SIMILARITY_THRESHOLD,
    slices: int | None = None,
) -> PartitionPlan:
    """Run Alg. 1 end to end and return the plan plus the simplification bookkeeping.

    ``slices`` pins the number of vertical slices; by default the constrained search of
    :func:`_admissible` picks it. :func:`slice_curve` is the batched form of the pinned
    call, which is what Fig. 2b/2d needs.
    """
    platform, timing = _defaults(platform, timing)
    agg, stats = _simplified(graph, timing, similarity)
    stats = stats | {"nodes_before": len(graph), "requested_slices": slices}
    if slices is None:
        return _admissible(graph, agg, stats)
    return _assemble(graph, agg, _CutTable(agg, slices).groups(slices), stats)


def slice_curve(
    graph: LayerGraph,
    ks: list[int],
    timing: TimingModel | None = None,
    platform: PlatformConfig | None = None,
    similarity: float = SIMILARITY_THRESHOLD,
) -> list[PartitionPlan]:
    """The best plan at *exactly* k slices, for every k in ``ks`` -- Fig. 2b/2d.

    Fig. 11's "3--5 slices per DLIS" is a claim about where that curve turns, and a
    plan is only comparable to the paper if it is the optimum *at* a given k rather
    than whatever the free DP happened to return. The simplification, the per-range
    aggregates and the DP table are built once and reused, so the whole curve costs
    one extra ``max(ks) * n^2`` sweep instead of one partition per k.
    """
    if not ks:
        raise ValueError("no slice counts requested")
    if min(ks) < 1:
        raise ValueError("a plan has at least one slice")
    platform, timing = _defaults(platform, timing)
    agg, stats = _simplified(graph, timing, similarity)
    table = _CutTable(agg, max(ks))
    return [
        _assemble(graph, agg, table.groups(k),
                  stats | {"nodes_before": len(graph), "requested_slices": k})
        for k in ks
    ]


def _defaults(
    platform: PlatformConfig | None, timing: TimingModel | None
) -> tuple[PlatformConfig, TimingModel]:
    """A sweep that names neither knob still has to price something: Lambda's rates."""
    if platform is None:
        platform = PlatformConfig()
    if timing is None:
        timing = TimingModel(platform=platform)
    return platform, timing
