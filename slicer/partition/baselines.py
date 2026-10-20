"""The four partitioning baselines Fig. 10 compares MOPAR against.

Sec. III-A's comparison has six methods per model, and the paper describes each of the
four partitioning ones by what it optimises and where it places. MOPAR's own plan comes
from :func:`slicer.partition.hypad.partition`; these come from here, and all five return
the same :class:`~slicer.partition.hypad.PartitionPlan`, because a baseline is only a
baseline if it is *plugged into the same service* -- same runtime, same quota rule,
same placement budget -- and only the plan differs.

**AlpaServe** [16] is "a dynamic programming algorithm to partition DLISs into several
distinct slices, all with the ultimate objective of decreasing inference latency", and
is criticised because it "disregard[s] resource utilization within each slice". Both
are implemented literally: the DP's objective is Eq. (5)'s time rather than Eq. (6)'s
money, while the plan is still *billed* by Eq. (6) so that Fig. 10a and Table III can
compare it against MOPAR on the same terms. A latency-only objective cuts wherever the
time falls and lets the quota follow the widest layer in the group, which is the
inefficiency Fig. 10a measures.

**Clockwork++** [8] takes the same objective and adds "a placement algorithm that
prioritizes selecting the same server". Its co-location is the whole difference from
AlpaServe, and it is charged as the cheaper link.

**NonSplit** [35] "employs an integer programming approach to exploit the parallelism
of non-contiguous subgraphs within DLISs". Reproduced as a local search over the
latency DP: a node may join any slice its dependencies straddle, provided the slice
graph stays acyclic, and a move is kept only where it lowers the objective. That is a
neighbourhood search rather than the paper's ILP, and it is stated here rather than
implied at a call site -- on a chain the two agree, so the difference shows up only
where a model has branches or residuals, and ``stats["moved_nodes"]`` says how often
it did.

**Uniform** is the paper's definition: "an even partitioning strategy, which divides a
DLIS into the same number of slices as MOPAR, so each slice has the same number of
layers" -- so it borrows its slice count from MOPAR rather than finding its own.

**Unsplit** is "the default strategy, which deploys a DLIS as a function without
partitioning": one function holding the whole service.

None of the four claims COM, so the timing model they are given must have
``compression_ratio = 1.0``; MOPAR's has whatever :attr:`R` the experiment settled on.
That is a property of the model passed in, not of this module, and a comparison that
gave every method the same compressed timing would silently delete Fig. 13c's effect
from Fig. 10.
"""

from __future__ import annotations

from typing import Sequence

from .cost import PlatformConfig, TimingModel, slice_footprint
from .graph import LayerGraph
from .hypad import (SIMILARITY_THRESHOLD, PartitionPlan, SlicePlan, _Aggregates,
                    _simplified)

#: Where a method asks the platform to put its slices. See :func:`place`.
SAME_SERVER = "same-server"
SPREAD = "spread"


class _Latency:
    """Eq. (5)'s time for a range of the order, at the eta that minimises it.

    Memoised for the same reason :class:`_Aggregates` exists: the DP asks about every
    one of the O(n^2) ranges, and a fresh eta sweep per query is a cubic pass over a
    158-layer service. The cap stays Eq. (6)'s -- a baseline may ignore memory in its
    *objective*, but no method can be given vCPUs the platform would not hand out.
    """

    def __init__(self, agg: _Aggregates) -> None:
        self.agg = agg
        self._best: dict[tuple[int, int], tuple[int, float]] = {}

    def at(self, eta: int, i: int, j: int) -> tuple[float, float, float]:
        agg = self.agg
        compute = agg.compute[eta - 1][j] - agg.compute[eta - 1][i]
        gather = (agg.gather[eta - 1][j] - agg.gather[eta - 1][i]) * agg.cross[j]
        return compute, gather, agg.timing.comm_time(agg.cross[j], same_host=True)

    def time(self, i: int, j: int, eta: int) -> float:
        return sum(self.at(eta, i, j))

    def best(self, i: int, j: int) -> tuple[int, float]:
        chosen = self._best.get((i, j))
        if chosen is None:
            chosen = min(((eta, self.time(i, j, eta))
                          for eta in range(1, self.agg.cap(i, j) + 1)),
                         key=lambda pair: pair[1])
            self._best[(i, j)] = chosen
        return chosen


def _latency_groups(agg: _Aggregates, latency: _Latency) -> list[list[str]]:
    """AlpaServe's split points: shortest total time over ranges of the order.

    The free optimum, with no quota term to make a wide slice expensive. Nothing in
    that objective rewards more slices, so it stops where the time stops falling;
    pinning a count would be a different method.
    """
    n = len(agg.order)
    best = [float("inf")] * (n + 1)
    origin = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j):
            if best[i] == float("inf"):
                continue
            candidate = best[i] + latency.best(i, j)[1]
            if candidate < best[j]:
                best[j], origin[j] = candidate, i
    groups: list[list[str]] = []
    j = n
    while j > 0:
        i = origin[j]
        groups.append(list(agg.order[i:j]))
        j = i
    return list(reversed([g for g in groups if g]))


def _set_quantities(agg: _Aggregates, names: Sequence[str]) -> tuple[float, float, float]:
    """Weights, widest activation and outgoing bytes of an arbitrary node set."""
    graph = agg.graph
    present = set(names)
    params = sum(graph.layer(name).params for name in names)
    memory = max((graph.layer(name).memory for name in names), default=0.0)
    across = sum(t.bytes for t in graph.tensors
                 if t.src in present and t.dst not in present)
    return params, memory, across


def _set_slice(agg: _Aggregates, names: Sequence[str],
               eta: int | None = None) -> SlicePlan:
    """One slice for an arbitrary node set, priced as :meth:`_Aggregates.plan` prices a range.

    Used only where a group is not a range -- NonSplit's moves -- so the scan of the
    graph's tensors is paid once per such slice rather than inside a DP. With ``eta``
    unspecified the whole admissible range is searched, which is why the bytes are
    gathered once above: the scan re-prices the same set, not the same sums.
    """
    graph, timing, platform = agg.graph, agg.timing, agg.platform
    params, memory, across = _set_quantities(agg, names)
    alloc = platform.allocate_mb(params + memory)
    comm = timing.comm_time(across, same_host=True)

    def at(shards: int) -> SlicePlan:
        compute = sum(timing.compute_time(graph.layer(name), shards) for name in names)
        gather = (sum(timing.aggregate_time(graph.layer(name), shards, 1.0)
                      for name in names)) * across
        return SlicePlan(
            nodes=tuple(names),
            members=tuple(m for name in names for m in graph.layer(name).members),
            alloc_mb=alloc, eta=shards, compute_s=compute, aggregate_s=gather,
            comm_out_s=comm, comm_out_bytes=across,
            cost_usd=alloc / 1024.0 * (compute + gather + comm)
            * platform.mem_price_per_gb_s + platform.net_price_per_s * comm)

    if eta is not None:
        return at(eta)
    return min((at(shards) for shards in
                range(1, platform.max_subslices(alloc) + 1)), key=lambda s: s.time_s)


def _slices_for(agg: _Aggregates, latency: _Latency,
                groups: list[list[str]]) -> list[SlicePlan]:
    """Each group at the eta its own time prefers, billed at Eq. (6)'s money.

    A group that is a range of the order goes through :class:`_Aggregates`, which is
    where the prefix sums are; a straddling group takes the set path. The two agree on
    a range by construction, so a plan's total does not depend on which path made it.
    """
    slices: list[SlicePlan] = []
    for group in groups:
        positions = [agg.position[name] for name in group]
        contiguous = positions == list(range(positions[0], positions[0] + len(group)))
        if contiguous:
            eta, _ = latency.best(positions[0], positions[-1] + 1)
            slices.append(agg.plan(positions[0], positions[-1] + 1, eta=eta))
        else:
            slices.append(_set_slice(agg, group))
    return slices


def _straddle(agg: _Aggregates, latency: _Latency, groups: list[list[str]]
              ) -> tuple[list[list[str]], int]:
    """NonSplit's move, applied until nothing improves.

    A node may leave slice *a* for slice *b* only when every producer of it sits at or
    before *b* and every consumer at or after *b*. That is the entire safety argument
    and it is checked, not assumed: it keeps every edge pointing forward in the slice
    order, so the slice graph stays a DAG and :mod:`slicer.faas.runtime` can still run
    the plan -- and ResNet's residuals and the GCNs are exactly where a straddle is
    available and a blind move would have closed a ring.

    Only the two touched slices are re-priced per candidate, which is what makes a pass
    over a 158-layer service cost time rather than days.
    """
    home = {name: index for index, group in enumerate(groups) for name in group}
    producers: dict[str, list[str]] = {name: [] for name in home}
    consumers: dict[str, list[str]] = {name: [] for name in home}
    for tensor in agg.graph.tensors:
        if tensor.src in home and tensor.dst in home:
            producers[tensor.dst].append(tensor.src)
            consumers[tensor.src].append(tensor.dst)
    rank = {name: position for position, name in enumerate(agg.order)}
    moved = 0
    changed = True
    while changed:
        changed = False
        for name in sorted(home, key=rank.get):
            source = home[name]
            low = max((home[node] for node in producers[name]), default=0)
            high = min((home[node] for node in consumers[name]),
                       default=len(groups) - 1)
            best_gain, best_target = 0.0, source
            for target in range(low, high + 1):
                if target == source:
                    continue
                mine = [node for node in groups[source] if node != name]
                theirs = groups[target] + [name]
                if not mine:
                    continue
                trial = _set_time(agg, latency, mine) + _set_time(agg, latency, theirs)
                gain = _set_time(agg, latency, groups[source]) + _set_time(
                    agg, latency, groups[target]) - trial
                if gain > best_gain:
                    best_gain, best_target = gain, target
            if best_target != source:
                groups[best_target].append(name)
                groups[best_target].sort(key=rank.get)
                groups[source].remove(name)
                home[name] = best_target
                moved += 1
                changed = True
    return [group for group in groups if group], moved


def _set_time(agg: _Aggregates, latency: _Latency, names: Sequence[str]) -> float:
    """The best Eq. (5) time for a node set, over the etas its quota affords.

    A range that happens to still be a range takes the prefix sums, which is the common
    case: most candidate moves in a pass leave the slice contiguous.
    """
    start = agg.position[names[0]]
    if list(names) == list(agg.order[start:start + len(names)]):
        return latency.best(start, start + len(names))[1]
    return _set_slice(agg, names).time_s


def place(plan: PartitionPlan, policy: str, servers: int = 14,
          platform: PlatformConfig | None = None) -> list[int]:
    """Which emulated server each slice runs on.

    Table II's fourteen servers hold 187 GB each. The two policies the paper
    distinguishes are MOPAR's affinity field -- "the ``preferredDuringScheduling...``
    policy to prioritize scheduling the slices on the same server" -- and a plan with
    no such preference. Co-location is packed first-fit by quota, so a pair of slices
    sharing a cut lands on one machine whenever its memory fits, which is what makes
    COM's shared directory reachable at all; spreading gives every slice its own server
    and pays the modelled link for every hop.
    """
    capacity = (platform or PlatformConfig()).server_memory_mb
    if policy == SPREAD:
        return [index % max(1, servers) for index in range(len(plan.slices))]
    used: list[float] = []
    where: list[int] = []
    for slice_plan in plan.slices:
        for index, held in enumerate(used[:servers]):
            if held + slice_plan.alloc_mb <= capacity:
                used[index] = held + slice_plan.alloc_mb
                where.append(index)
                break
        else:
            # Past the named servers the plan has nowhere left to go, and a sweep that
            # lost its last machine should not silently become a co-located one.
            if len(used) < servers:
                used.append(slice_plan.alloc_mb)
                where.append(len(used) - 1)
            else:
                where.append(-1)
    return where


def _prepared(graph: LayerGraph, timing: TimingModel | None,
              platform: PlatformConfig | None, similarity: float
              ) -> tuple[_Aggregates, _Latency, dict]:
    """The simplification every method shares, so no two differ in preprocessing."""
    platform = platform or PlatformConfig()
    timing = timing or TimingModel(platform=platform)
    agg, stats = _simplified(graph, timing, similarity)
    return agg, _Latency(agg), stats | {"nodes_before": len(graph)}


def _assemble(agg: _Aggregates, latency: _Latency, groups: list[list[str]],
              stats: dict, method: str) -> PartitionPlan:
    """Groups into a plan, with the unsplit service attached as its own reference."""
    platform = agg.platform
    slices = _slices_for(agg, latency, groups)
    whole = agg.plan(0, len(agg.order), eta=latency.best(0, len(agg.order))[0])
    unsplit_alloc = platform.allocate_mb(slice_footprint(agg.graph, agg.graph.order))
    return PartitionPlan(
        slices=tuple(slices), latency_s=sum(s.time_s for s in slices),
        unsplit_latency_s=whole.time_s,
        unsplit_cost_usd=unsplit_alloc / 1024.0 * whole.time_s
        * platform.mem_price_per_gb_s,
        cost_usd=sum(s.cost_usd for s in slices),
        stats=stats | {"method": method, "nodes_after": len(agg.order),
                       "slices": len(slices),
                       "alloc_mb": sum(s.alloc_mb for s in slices)})


def alpaserve(graph: LayerGraph, timing: TimingModel | None = None,
              platform: PlatformConfig | None = None,
              similarity: float = SIMILARITY_THRESHOLD) -> PartitionPlan:
    """Latency-optimal DP, spread over servers, and no coding on the hops."""
    agg, latency, stats = _prepared(graph, timing, platform, similarity)
    return _assemble(agg, latency, _latency_groups(agg, latency),
                     stats | {"placement": SPREAD}, "AlpaServe")


def clockwork(graph: LayerGraph, timing: TimingModel | None = None,
              platform: PlatformConfig | None = None,
              similarity: float = SIMILARITY_THRESHOLD) -> PartitionPlan:
    """The same latency DP, placed on as few servers as its memory allows."""
    agg, latency, stats = _prepared(graph, timing, platform, similarity)
    return _assemble(agg, latency, _latency_groups(agg, latency),
                     stats | {"placement": SAME_SERVER}, "Clockwork++")


def nonsplit(graph: LayerGraph, timing: TimingModel | None = None,
             platform: PlatformConfig | None = None,
             similarity: float = SIMILARITY_THRESHOLD) -> PartitionPlan:
    """The latency DP improved by non-contiguous moves, spread over servers."""
    agg, latency, stats = _prepared(graph, timing, platform, similarity)
    groups, moved = _straddle(agg, latency, _latency_groups(agg, latency))
    straddling = sum(1 for group in groups
                     if [agg.position[name] for name in group]
                     != list(range(agg.position[group[0]],
                                   agg.position[group[0]] + len(group))))
    return _assemble(agg, latency, groups, stats | {"placement": SPREAD,
                                           "moved_nodes": moved,
                                           "non_contiguous_slices": straddling},
                     "NonSplit")


def uniform(graph: LayerGraph, slices: int, timing: TimingModel | None = None,
            platform: PlatformConfig | None = None,
            similarity: float = SIMILARITY_THRESHOLD) -> PartitionPlan:
    """Equal layers per slice, at MOPAR's slice count."""
    agg, latency, stats = _prepared(graph, timing, platform, similarity)
    n = len(agg.order)
    step = -(-n // max(1, slices))
    groups = [list(agg.order[start:start + step]) for start in range(0, n, step)]
    return _assemble(agg, latency, groups,
                     stats | {"placement": SAME_SERVER, "requested_slices": slices},
                     "Uniform")


def unsplit(graph: LayerGraph, timing: TimingModel | None = None,
            platform: PlatformConfig | None = None,
            similarity: float = SIMILARITY_THRESHOLD) -> PartitionPlan:
    """One function, the whole service, and the vCPUs its own memory affords."""
    agg, latency, stats = _prepared(graph, timing, platform, similarity)
    return _assemble(agg, latency, [list(agg.order)],
                     stats | {"placement": SAME_SERVER}, "Unsplit")


#: The methods of Fig. 10, in the paper's own order. ``MOPAR`` is planned elsewhere.
FIGURE_10 = ("MOPAR", "AlpaServe", "Clockwork++", "NonSplit", "Uniform", "Unsplit")


def plan_for(method: str, graph: LayerGraph, slicer: PartitionPlan,
             timing: TimingModel | None = None,
             platform: PlatformConfig | None = None,
             similarity: float = SIMILARITY_THRESHOLD, servers: int = 14
             ) -> tuple[PartitionPlan, list[int]]:
    """The plan one method runs, and where it would put each slice.

    ``slicer`` is passed in rather than recomputed because ``Uniform`` is defined
    against it -- "the same number of slices as MOPAR" -- and a comparison that
    re-derived MOPAR per method would report a difference in two plans at once.
    """
    if method == "MOPAR":
        return slicer, place(slicer, SAME_SERVER, servers, platform)
    if method not in FIGURE_10:
        raise ValueError("%r is not one of %s" % (method, ", ".join(FIGURE_10)))
    if method == "Uniform":
        plan = uniform(graph, len(slicer.slices), timing, platform, similarity)
        return plan, place(plan, SAME_SERVER, servers)
    plan = {"AlpaServe": alpaserve, "Clockwork++": clockwork,
            "NonSplit": nonsplit, "Unsplit": unsplit}[method](
        graph, timing, platform, similarity)
    return plan, place(plan, {"AlpaServe": SPREAD, "NonSplit": SPREAD}.get(method,
                                                                          SAME_SERVER),
                       servers)
