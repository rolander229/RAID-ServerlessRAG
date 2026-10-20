"""Layout optimizer for serverless deployment of AI pipelines.

Rewritten from scratch. The previous version (see results/FABRICATION_DIFF.md)
was a threshold rule dressed up as multi-objective optimisation: it fused every
edge with payload >= 1MB regardless of cost, never read its own weights, and had
no constraints at all -- which is why it collapsed real pipelines into a single
function.

This version inverts the logic. A partition is *infeasible* if any function
breaches a provider ceiling, so splitting is often **forced** rather than
incidental (the Gillis ICDCS'21 framing: serverless functions have bounded
memory and time, so a large model simply cannot be served by one function).
Given the feasible set, the solver reports a Pareto front over latency and cost.

Inputs are measured, not assumed:
  * per-operator compute / device / GPU memory
        results/real_profiling_results.json
  * per-state-key wire bytes and serialisation cost
        results/op_key_costs.json
  * per-function process overhead and RSS baseline
        results/split_runtime_demo.json

Output keeps the legacy ``fused_functions`` / ``sqs_queues`` / ``metrics``
schema so downstream stages keep working.
"""
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------- ceilings
# Provider limits. These are hard: breaching one makes a plan infeasible.
CEILINGS = {
    "memory_mb": 10240,     # AWS Lambda maximum
    "timeout_s": 900,       # AWS Lambda maximum
    "package_mb": 250,      # deployment package (zipped)
}
LAMBDA_STAIRCASE = [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
                    10240]
LAMBDA_GB_S = 0.0000166667
GPU_PRICE_MULT = 3.5

# Fallbacks used only when a measurement is missing; the plan records which
# memory model it used so a fallback can never masquerade as a measurement.
RUNTIME_BASE_MB = 680.0
CUDA_CONTEXT_MB = 220.0
HOP_MS = 0.15            # measured tmpfs write+read per boundary


def ceil_staircase(mb: float) -> int:
    for t in LAMBDA_STAIRCASE:
        if mb <= t:
            return t
    return LAMBDA_STAIRCASE[-1]


@dataclass
class ColdStart:
    """Getting one function from "not running" to "serving".

    This is the term the earlier cost model omitted, and omitting it flatters
    splitting in exactly the wrong direction: a split has *more* functions, so
    it multiplies the one-off init, and the init is billed at the function's own
    tier. A plan that looks 40% cheaper warm can be more expensive than the
    monolith once the init is amortised over too few requests -- which is why
    ``reuse`` is a parameter of the evaluation and not a constant.

    Two measured terms, both from ``workloads.vit_split_run`` on v100:

    ``import_ms``
        interpreter start plus the handler's imports. Dominated by torch
        (381 MB / ~2.5 s) and torchvision (+131 MB). Roughly *independent of the
        model*, so it is a fixed tax per function, and it is most of the total:
        measured handler floors are 540 MB and the spawn is ~2.7 s, while a
        144 MB shard adds only ~0.17 s to load.

    ``weight_ms_per_mb``
        shard load from local NVMe, 0.17 s per 144 MB shard.

    ``calibrate`` fits both from a measurement file rather than trusting the
    defaults, so a report can say the model was fitted, not assumed.
    """

    import_ms: float = 2700.0
    weight_ms_per_mb: float = 1.2
    source: str = "default"

    def init_ms(self, peak_mb: float, floor_mb: float) -> float:
        weights_mb = max(0.0, peak_mb - floor_mb)
        return self.import_ms + self.weight_ms_per_mb * weights_mb

    @classmethod
    def calibrate(cls, path: str):
        """Least-squares fit of (peak RSS, measured init) pairs from a run.

        Pairs come from the cold pass of ``vit_split_run``: every function
        reports both its peak RSS and ``spawn + first-call`` wall time, so the
        two coefficients are observable rather than guessed. Falls back to the
        defaults (and says so via ``source``) if the file is absent or has too
        few points.
        """
        import json
        try:
            with open(path) as fh:
                data = json.load(fh)
            pts = []
            for r in data.get("runs", {}).values():
                for f in r.get("functions", []):
                    if f.get("rss_mb", 0) > 0 and f.get("init_ms", 0) > 0:
                        pts.append((f["rss_mb"], f["init_ms"]))
            if len(pts) < 4:
                return cls(source="default (fewer than 4 usable points)")
            n = float(len(pts))
            sx = sum(p[0] for p in pts)
            sy = sum(p[1] for p in pts)
            sxx = sum(p[0] * p[0] for p in pts)
            sxy = sum(p[0] * p[1] for p in pts)
            den = n * sxx - sx * sx
            if abs(den) < 1e-9:
                return cls(source="default (degenerate fit)")
            slope = (n * sxy - sx * sy) / den
            inter = (sy - slope * sx) / n
            return cls(import_ms=round(inter, 1),
                       weight_ms_per_mb=round(slope, 4),
                       source="fitted on %d measured functions from %s"
                              % (len(pts), path))
        except Exception as e:
            return cls(source="default (calibration failed: %s)" % e)


@dataclass
class OpCost:
    name: str
    device: str
    compute_ms: float
    gpu_mb: float = 0.0
    # Measured peak RSS of this operator running alone in its own function.
    # 0.0 means "not measured"; the estimator then says so instead of silently
    # substituting a constant.
    rss_mb: float = 0.0


@dataclass
class KeyCost:
    key: str
    wire_mb: float
    serde_ms: float
    kind: str = "unknown"


@dataclass
class FunctionPlan:
    logical_name: str
    operators: List[str]
    device_type: str
    recommended_memory_mb: int
    recommended_timeout_sec: float
    peak_memory_mb: float
    total_compute_latency_ms: float
    inbound_keys: List[str] = field(default_factory=list)
    inbound_mb: float = 0.0
    inbound_serde_ms: float = 0.0
    trigger_type: str = "api_gateway"
    entry_point: str = ""
    is_async_worker: bool = False
    memory_model: str = "measured_superposition"
    est_gpu_mb: float = 0.0


@dataclass
class Plan:
    pipeline: str
    functions: List[FunctionPlan]
    feasible: bool
    violations: List[str]
    latency_ms: float
    cost_usd: float
    comm_mb: float
    forced_split: bool = False
    # Cold-start accounting. ``latency_cold_ms`` is what the first request after
    # a scale-from-zero actually sees; ``cost_usd_cold`` amortises the init over
    # ``reuse`` requests per warm instance. Both are 0.0 when no cold-start
    # model was supplied, which is a different statement from "the init is free"
    # -- the report is expected to say which it is.
    latency_cold_ms: float = 0.0
    init_sum_ms: float = 0.0
    cost_usd_cold: float = 0.0
    reuse: float = 0.0
    cold_model: str = "none"
    overhead_model: str = ""

    def n(self) -> int:
        return len(self.functions)

    def signature(self) -> Tuple:
        return tuple(sorted(tuple(f.operators) for f in self.functions))


class LayoutOptimizer:
    def __init__(self, pipeline, op_costs: Dict[str, OpCost],
                 key_costs: Dict[str, KeyCost],
                 ceilings: Optional[dict] = None,
                 base_mb: float = RUNTIME_BASE_MB,
                 base_cpu_mb: Optional[float] = None,
                 base_gpu_mb: Optional[float] = None,
                 cold: Optional["ColdStart"] = None,
                 reuse: Optional[float] = None):
        self.p = pipeline
        self.ops = op_costs
        self.keys = key_costs
        self.ceilings = dict(CEILINGS)
        if ceilings:
            self.ceilings.update(ceilings)
        self.base_mb = base_mb
        # Measured per-function floors: the RSS of an almost-empty function
        # process, with and without a CUDA context. These replace the two
        # invented constants the old solver used to price a function.
        self.base_cpu_mb = base_cpu_mb if base_cpu_mb is not None else base_mb
        self.base_gpu_mb = (base_gpu_mb if base_gpu_mb is not None
                            else base_mb + CUDA_CONTEXT_MB)
        # ``reuse`` = invocations served by one warm instance. None means "only
        # price the steady state", which is the right default for comparing
        # layouts but must never be presented as the whole cost.
        self.cold = cold
        self.reuse = reuse
        self._edge_keys = self._map_edge_keys()

    # ------------------------------------------------------------ helpers
    def _map_edge_keys(self) -> Dict[Tuple[str, str], List[str]]:
        """(producer, consumer) -> state keys that cross that edge."""
        out: Dict[Tuple[str, str], List[str]] = {}
        for a, b in self.p.edges():
            shared = sorted(set(self.p.by_name(a).writes)
                            & set(self.p.by_name(b).reads))
            if shared:
                out[(a, b)] = shared
        return out

    def _groups_of(self, cut_edges: Set[Tuple[str, str]]) -> List[List[str]]:
        """Remove edges; each connected component becomes one function."""
        parent = {n: n for n in self.p.names()}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.p.edges():
            if (a, b) in cut_edges:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        comp: Dict[str, List[str]] = {}
        order = self.p.names()
        for n in order:
            comp.setdefault(find(n), []).append(n)
        groups = [sorted(v, key=order.index) for v in comp.values()]
        groups.sort(key=lambda g: order.index(g[0]))
        return groups

    def all_partitions(self, limit: int = 20000) -> List[List[List[str]]]:
        """Every partition induced by a choice of cut edges, plus the monolith.

        The trivial one-function layout is seeded explicitly, and that is not a
        cosmetic detail. Cutting nothing yields the connected components of the
        *data-flow* graph, and a pipeline whose operators talk through
        function-local warm state instead of through a shared key has more than
        one component even with zero cuts. Relying on the cut enumeration alone
        therefore drops the monolith candidate entirely, `solve()` finds no
        1-function plan, and the report claims "splitting is forced" at every
        memory ceiling -- a fabricated positive, and the exact opposite of the
        question the ceiling sweep exists to answer.
        """
        edges = self.p.edges()
        if len(edges) > 14:
            raise ValueError("edge subset enumeration too large: %d edges"
                             % len(edges))
        candidates = [[self.p.names()]]
        for r in range(len(edges) + 1):
            for combo in itertools.combinations(edges, r):
                candidates.append(self._groups_of(set(combo)))
        seen, out = set(), []
        for g in candidates:
            sig = tuple(sorted(tuple(x) for x in g))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(g)
            if len(out) >= limit:
                break
        return out

    # --------------------------------------------------------- evaluation
    def evaluate(self, groups: List[List[str]]) -> Plan:
        funcs: List[FunctionPlan] = []
        violations: List[str] = []
        latency = 0.0
        cost = 0.0
        comm_mb = 0.0
        init_sum = 0.0
        cost_cold = 0.0

        owner = {}
        for gi, g in enumerate(groups):
            for n in g:
                owner[n] = gi

        for gi, g in enumerate(groups):
            devs = {self.ops[n].device for n in g}
            has_gpu = "GPU" in devs
            compute = sum(self.ops[n].compute_ms for n in g)
            gpu_mb = sum(self.ops[n].gpu_mb for n in g) if has_gpu else 0.0
            base = self.base_gpu_mb if has_gpu else self.base_cpu_mb
            # Superposition of *measured* per-operator increments over the
            # measured floor of an almost-empty function of the same class.
            # This is a first-order estimate: it ignores tensor lifetimes, so
            # for a group of operators that never hold live data at the same
            # time it over-estimates. It never invents a number, though, which
            # is the property the previous solver lacked.
            measured = all(self.ops[n].rss_mb > 0 for n in g)
            if measured:
                mem = base + sum(max(0.0, self.ops[n].rss_mb - base) for n in g)
                mem_model = "measured_superposition"
            else:
                mem = (RUNTIME_BASE_MB + (CUDA_CONTEXT_MB if has_gpu else 0.0)
                       + gpu_mb)
                mem_model = "constant_fallback"

            in_keys, in_mb, in_serde = [], 0.0, 0.0
            for (a, b), ks in self._edge_keys.items():
                if b in g and owner.get(a) != gi:
                    for k in ks:
                        if k in in_keys:
                            continue
                        in_keys.append(k)
                        kc = self.keys.get(k)
                        if kc:
                            in_mb += kc.wire_mb
                            in_serde += kc.serde_ms

            n_boundaries = len(in_keys)
            hop = HOP_MS * n_boundaries
            timeout = max(10.0, compute / 1000.0 * 3.0 + 10.0)
            tier = ceil_staircase(mem * 1.25)

            if mem > self.ceilings["memory_mb"]:
                violations.append(
                    "function %s needs %.0f MB > ceiling %d MB"
                    % (",".join(g), mem, self.ceilings["memory_mb"]))
            if timeout > self.ceilings["timeout_s"]:
                violations.append(
                    "function %s needs %.0fs > ceiling %ds"
                    % (",".join(g), timeout, self.ceilings["timeout_s"]))

            billed_s = (compute + in_serde + hop) / 1000.0
            price = (tier / 1024.0) * LAMBDA_GB_S * (
                GPU_PRICE_MULT if has_gpu else 1.0)
            cost += price * billed_s

            # The same tier prices the init, because the platform reserves the
            # configured memory for the whole init -- that is why a split's
            # extra functions are not free even when their compute is nil.
            if self.cold is not None:
                init_ms = self.cold.init_ms(mem, base)
                init_sum += init_ms
                if self.reuse:
                    cost_cold += price * (init_ms / 1000.0) / float(self.reuse)
            else:
                init_ms = 0.0

            latency += compute + in_serde + hop
            comm_mb += in_mb
            funcs.append(FunctionPlan(
                logical_name=("%s-%s" % (g[0], "fused" if len(g) > 1 else "fn")),
                operators=list(g),
                device_type="GPU_Inference" if has_gpu else "CPU_Heavy",
                recommended_memory_mb=tier,
                recommended_timeout_sec=round(timeout, 2),
                peak_memory_mb=round(mem, 2),
                total_compute_latency_ms=round(compute, 3),
                inbound_keys=sorted(in_keys),
                inbound_mb=round(in_mb, 4),
                inbound_serde_ms=round(in_serde, 3),
                entry_point="%s_handler" % g[0],
                is_async_worker=any(self.p.by_name(n).is_worker for n in g),
                memory_model=mem_model,
                est_gpu_mb=round(gpu_mb, 2),
            ))

        # per-function invocation overhead: measured, not assumed
        latency += self.overhead_ms() * len(funcs)
        if self.cold is not None and self.reuse:
            cost += cost_cold
        return Plan(pipeline=self.p.key, functions=funcs,
                    feasible=not violations, violations=violations,
                    latency_ms=round(latency, 3), cost_usd=cost,
                    comm_mb=round(comm_mb, 4),
                    latency_cold_ms=round(latency + init_sum, 3),
                    init_sum_ms=round(init_sum, 3),
                    cost_usd_cold=cost_cold,
                    reuse=float(self.reuse or 0.0),
                    cold_model=(self.cold.source if self.cold else "none"),
                    overhead_model=self.overhead_source())

    def overhead_ms(self) -> float:
        """Per-invocation dispatch overhead, excluding cold start.

        Measured on the real runtime: a boundary costs encode + socket + decode
        + tmpfs, and what is left over once the codec and the model work are
        subtracted is ~2.1 ms per function (vit_split_run, V8_fission:
        wall 90.7 ms - compute 78.1 - decode 2.1 - encode 6.1 - hop 2.3). It is
        still a single constant applied to every function, so the model records
        where it came from instead of presenting it as a per-plan measurement.
        """
        return float(os.environ.get("M2S_FN_OVERHEAD_MS", "2.1"))

    def overhead_source(self) -> str:
        if os.environ.get("M2S_FN_OVERHEAD_MS"):
            return "env M2S_FN_OVERHEAD_MS"
        return "default 2.1 ms (measured on vit_split_run V8_fission)"

    def band(self, lo_mb: int = 256, hi_mb: int = 10240) -> dict:
        """Locate the forced-split band exactly, by bisection on the ceiling.

        A coarse sweep only shows the band quantised to its grid: at a 1024/1536
        stride the interesting transition is a single cell wide, so the report can
        say "somewhere between 1024 and 1536" and nothing sharper. Feasibility is
        monotone in the ceiling (a larger budget never makes a plan worse), so
        both edges can be found by bisection instead.

        Returns the two edges plus what lives between them:
          * `min_ceiling_deployable_mb` -- below this, *no* partition fits at all,
            because one function's floor alone already exceeds the budget. This
            is the "split cannot save you" side of the band.
          * `min_ceiling_monolith_mb`   -- at or above this the monolith fits and
            splitting has no memory argument left.
        """
        def _solve(ceil):
            o = LayoutOptimizer(self.p, self.ops, self.keys,
                               ceilings={**self.ceilings, "memory_mb": ceil},
                               base_mb=self.base_mb,
                               base_cpu_mb=self.base_cpu_mb,
                               base_gpu_mb=self.base_gpu_mb)
            return o.solve()

        def _edge(pred, lo, hi):
            """smallest ceiling in [lo, hi] with pred true; None if never true."""
            if not pred(hi):
                return None
            while lo < hi:
                mid = (lo + hi) // 2
                if pred(mid):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        cache = {}

        def depl(c):
            if c not in cache:
                cache[c] = _solve(c)
            return cache[c]["deployable"]

        def mono(c):
            if c not in cache:
                cache[c] = _solve(c)
            return cache[c]["monolith_feasible"]

        c_depl = _edge(depl, lo_mb, hi_mb)
        c_mono = _edge(mono, lo_mb, hi_mb)
        out = {
            "min_ceiling_deployable_mb": c_depl,
            "min_ceiling_monolith_mb": c_mono,
            "band_lo_mb": c_depl,
            "band_hi_mb": c_mono,
            "band_width_mb": (c_mono - c_depl) if (c_depl and c_mono) else None,
            "min_functions_in_band": None,
            "cheapest_layout_in_band": None,
        }
        if c_depl and c_mono and c_mono > c_depl:
            # the tightest mandatory-split budget: fewest functions that fit.
            o = LayoutOptimizer(self.p, self.ops, self.keys,
                                ceilings={**self.ceilings, "memory_mb": c_depl},
                                base_mb=self.base_mb,
                                base_cpu_mb=self.base_cpu_mb,
                                base_gpu_mb=self.base_gpu_mb)
            plans = [o.evaluate(g) for g in o.all_partitions()]
            feas = [pl for pl in plans if pl.feasible]
            if feas:
                best = min(feas, key=lambda pl: (pl.n(), pl.latency_ms))
                out["min_functions_in_band"] = best.n()
                out["cheapest_layout_in_band"] = {
                    "n_functions": best.n(),
                    "latency_ms": best.latency_ms,
                    "cost_usd": best.cost_usd,
                    "comm_mb": best.comm_mb,
                    "functions": [
                        {"name": f.logical_name, "device": f.device_type,
                         "ops": list(f.operators),
                         "memory_mb": f.recommended_memory_mb}
                        for f in best.functions
                    ],
                }
        return out

    # ------------------------------------------------------------- solving
    def solve(self, alphas: Optional[List[float]] = None) -> dict:
        alphas = alphas or [i / 20.0 for i in range(21)]
        allp = [self.evaluate(g) for g in self.all_partitions()]
        feas = [p for p in allp if p.feasible]
        infeas = [p for p in allp if not p.feasible]

        mono = next((p for p in allp if p.n() == 1), None)
        if mono is None:
            # all_partitions() seeds the one-function layout, so this cannot
            # happen; if it ever does, the ceiling sweep below would silently
            # report "monolith INFEASIBLE" and invert the conclusion.
            raise AssertionError(
                "no 1-function candidate for %s: all_partitions() is broken"
                % self.p.key)
        forced = not mono.feasible

        pool = feas or allp
        lat = [p.latency_ms for p in pool]
        cst = [p.cost_usd for p in pool]
        lo_l, hi_l = min(lat), max(lat)
        lo_c, hi_c = min(cst), max(cst)

        def norm(v, lo, hi):
            return 0.0 if hi <= lo else (v - lo) / (hi - lo)

        front: Dict[Tuple, Plan] = {}
        for a in alphas:
            best = min(pool, key=lambda p: a * norm(p.latency_ms, lo_l, hi_l)
                       + (1 - a) * norm(p.cost_usd, lo_c, hi_c))
            front[best.signature()] = best

        lat_opt = min(pool, key=lambda p: p.latency_ms)
        cost_opt = min(pool, key=lambda p: (p.cost_usd, p.latency_ms))
        # When nothing is feasible the answer is "no deployable layout", NOT
        # "1 function". Falling back to `allp` here used to report min=1
        # alongside monolith INFEASIBLE at the same ceiling -- a self-contradiction
        # that reads as "the monolith is both impossible and the smallest answer".
        min_funcs_feasible = min(feas, key=lambda p: p.n()) if feas else None

        return {
            "pipeline": self.p.key,
            "n_operators": len(self.p.names()),
            "n_partitions_evaluated": len(allp),
            "monolith_feasible": bool(mono.feasible),
            "monolith_violations": mono.violations,
            "forced_split": forced,
            "deployable": bool(feas),
            "min_functions_feasible": (min_funcs_feasible.n()
                                       if min_funcs_feasible else None),
            "n_infeasible_partitions": len(infeas),
            "latency_optimal": lat_opt,
            "cost_optimal": cost_opt,
            "pareto": sorted(front.values(), key=lambda p: p.latency_ms),
        }

    # ------------------------------------------------------------ emission
    def to_legacy_plan(self, plan: Plan) -> dict:
        return {
            "pipeline_name": self.p.title,
            "benchmark_id": plan.pipeline,
            "total_original_operators": len(self.p.names()),
            "total_fused_faas_functions": plan.n(),
            "fused_functions": [asdict(f) for f in plan.functions],
            "sqs_queues": [
                {"queue_name": "%s-queue" % f.logical_name,
                 "dlq_name": "%s-dlq" % f.logical_name,
                 "visibility_timeout_sec": max(
                     900, int(math.ceil(f.recommended_timeout_sec * 3))),
                 "max_receive_count": 3}
                for f in plan.functions if f.is_async_worker
            ],
            "metrics": {
                "predicted_e2e_latency_ms": plan.latency_ms,
                "predicted_cost_usd_per_request": plan.cost_usd,
                "cross_function_payload_mb": plan.comm_mb,
                "feasible": plan.feasible,
                "violations": plan.violations,
                "forced_split": plan.forced_split,
            },
        }


# ------------------------------------------------------------ loader
def load(root: str = ".", extra_prof=()):
    """Load measured inputs. Returns ``(op_costs, key_costs, floors)``.

    ``floors`` carries the measured per-function RSS floors (CPU / GPU) from
    ``results/op_memory.json`` when the probe has run; otherwise it is empty and
    the estimator falls back to constants -- which it then *says* it did.

    ``extra_prof`` names additional profiling files to merge, in the same shape
    as ``real_profiling_results.json``. Operators missing from the primary file
    would otherwise get ``compute_ms = 0.0`` from the memory record's fallback,
    and a solver fed a zero compute time prices an entire pipeline at a few
    milliseconds -- which is worse than refusing to answer. Merging is explicit
    rather than a glob so that a stale file cannot quietly change a result.
    """
    with open(os.path.join(root, "results",
                           "real_profiling_results.json")) as fh:
        prof = json.load(fh)
    with open(os.path.join(root, "results", "op_key_costs.json")) as fh:
        kc = json.load(fh)

    opcost = {}
    for b in prof["benchmarks"]:
        for o in b["operators"]:
            if "avg_ms" not in o:
                continue
            opcost[o["operator"]] = OpCost(
                name=o["operator"], device=o["device"],
                compute_ms=o["avg_ms"],
                gpu_mb=o.get("gpu_peak_extra_mb", 0.0))

    merged_from = []
    for rel in extra_prof:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            extra = json.load(fh)
        n = 0
        for b in extra.get("benchmarks", []):
            for o in b.get("operators", []):
                if o["operator"] in opcost:
                    continue        # never let a secondary file win
                opcost[o["operator"]] = OpCost(
                    name=o["operator"], device=o["device"],
                    compute_ms=o.get("avg_ms", 0.0),
                    gpu_mb=o.get("gpu_peak_extra_mb", 0.0))
                n += 1
        merged_from.append("%s (+%d operators)" % (rel, n))

    keycost = {}
    for rows in kc.values():
        for r in rows:
            if "wire_mb" not in r:
                continue
            keycost[r["key"]] = KeyCost(key=r["key"], wire_mb=r["wire_mb"],
                                        serde_ms=r["serde_ms"], kind=r["kind"])

    floors = {}
    mp = os.path.join(root, "results", "op_memory.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            mem = json.load(fh)
        floors = {"base_cpu_mb": mem.get("base_cpu_mb"),
                  "base_gpu_mb": mem.get("base_gpu_mb"),
                  "source": os.path.basename(mp)}
        for name, r in mem.get("ops", {}).items():
            cur = opcost.get(name)
            if cur is None:
                opcost[name] = OpCost(name=name,
                                      device=r.get("device", "CPU"),
                                      compute_ms=0.0,
                                      gpu_mb=r.get("gpu_mb", 0.0),
                                      rss_mb=r.get("rss_mb", 0.0))
            else:
                cur.rss_mb = r.get("rss_mb", 0.0)
                if r.get("gpu_mb"):
                    cur.gpu_mb = r["gpu_mb"]

    # An operator that reached the solver with no compute time is a missing
    # measurement, not a fast operator. Say so loudly: silently pricing it at
    # zero is how the ViT pipeline came to look 300x cheaper than it is.
    unprofiled = sorted(n for n, c in opcost.items()
                        if c.compute_ms <= 0.0 and not n.endswith("sink_worker"))
    if unprofiled:
        print("[optimizer.load] WARNING: %d operators have no compute "
              "measurement and would be priced at 0 ms: %s"
              % (len(unprofiled), unprofiled[:8]))
        print("[optimizer.load] run `python -m workloads.probe_op_time "
              "--keys <pipeline>` and pass the file via extra_prof=...")
    if merged_from:
        print("[optimizer.load] merged extra profiling: %s"
              % "; ".join(merged_from))
    return opcost, keycost, floors
