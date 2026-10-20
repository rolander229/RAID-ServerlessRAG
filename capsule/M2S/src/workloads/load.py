"""Load, elasticity and replica-capacity model -- measured, not simulated.

Why this file exists
--------------------
The old project had no notion of load at all: no arrivals, no queueing, no
concurrency, no cold starts. A layout was judged by a save/delete diff against
a monolith and by whether a declared memory ceiling was exceeded. That is not
the question a serverless deployment asks. The real question is:

    under a given arrival rate and a given memory budget, how much goodput does
    this partition sustain, and how much of it meets the SLO?

Splitting moves that answer in two opposite directions:

  * it *adds* cost -- every cut is a serialise/store/deserialise hop, and every
    extra function is another cold start;
  * it *removes* a constraint -- the largest single function is smaller, so more
    replicas fit inside the same memory budget, and a fan-out stage can scale
    independently of a heavy stage.

That trade is only visible when replicas are measured under load, which is what
this module does.

What is real here
  * concurrency: N warm worker *processes* per function, dispatched from a
    shared pool. A request that finds every replica busy waits -- queueing delay
    is measured, not modelled.
  * service times: the same real operators as the rest of the runtime.
  * cold start: every replica is really spawned; spawn+handshake is recorded.
  * memory: peak RSS per function (VmHWM) is measured, and replica capacity is
    derived from it.

What is NOT real (and must be stated in any write-up)
  * arrivals are synthetic. The default is a closed loop (``clients``), which
    measures the maximum sustainable rate; an open-loop Poisson mode is
    available when an absolute request rate matters.
  * the hop is tmpfs on one host: no WAN RTT, no TLS, no object-store latency.
    Queueing and cold start are real; network transport is not.
  * GPU memory is a separate pool. For GPU stages the RSS ceiling does not
    observe the weights, so RSS-based capacity applies to CPU-resident functions
    -- which is exactly the Gillis regime.
"""
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.runtime import Plan, Worker, compile_plan  # noqa: E402

DEFAULT_SLO_MS = 1000.0
SPOOL = os.environ.get("M2S_SPOOL", "/dev/shm/m2s-spool")


class ReplicaSet:
    """``replicas`` warm processes of one function, shared by all clients.

    Blocking on ``free.get()`` when the pool is empty *is* the queue: the wait
    is timed and reported as ``queue_wait_ms``. So merging two operators into
    one function is visibly rewarded (no hop, no second queue) while a heavily
    loaded stage is visibly penalised.
    """

    def __init__(self, pipeline: str, f, replicas: int = 1,
                 rlimit_mb: int = 0, logdir: Optional[str] = None):
        self.spec = f
        self.free: "queue.Queue" = queue.Queue()
        self.workers: List[Worker] = []
        self.lock = threading.Lock()
        for i in range(max(1, replicas)):
            w = Worker(pipeline, label="%s#%d" % (f.name, i),
                       rlimit_mb=rlimit_mb, logdir=logdir)
            self.workers.append(w)
            self.free.put(w)
        self.spawn_ms = [round(w.spawn_ms, 2) for w in self.workers]
        self.cold_ms = max(self.spawn_ms) if self.spawn_ms else 0.0
        self.busy_ms = 0.0
        self.wait_ms = 0.0
        self.calls = 0
        self.rss_mb = 0
        self.gpu_mb = 0.0
        self.gpu_reserved_mb = 0.0
        self.errors = 0

    def call(self, ops, inbound, outbound, timeout_s) -> dict:
        t0 = time.perf_counter()
        w = self.free.get()
        waited = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        try:
            r = w.call(ops, inbound, outbound, timeout_s)
        except Exception:
            with self.lock:
                self.errors += 1
            raise
        finally:
            elapsed = (time.perf_counter() - t1) * 1000.0
            with self.lock:
                self.wait_ms += waited
                self.busy_ms += elapsed
            self.free.put(w)
        with self.lock:
            self.calls += 1
            self.rss_mb = max(self.rss_mb, r.get("rss_mb", 0))
            self.gpu_mb = max(self.gpu_mb, r.get("gpu_mb", 0.0))
            self.gpu_reserved_mb = max(self.gpu_reserved_mb,
                                       r.get("gpu_reserved_mb", 0.0))
        return r

    def close(self):
        for w in self.workers:
            w.close()

    def util(self, wall_ms: float) -> float:
        cap = max(1e-9, wall_ms * len(self.workers))
        return min(1.0, self.busy_ms / cap)


@dataclass
class LoadResult:
    label: str
    pipeline: str
    n_functions: int
    replicas: int
    clients: int
    duration_s: float
    slo_ms: float = DEFAULT_SLO_MS
    mode: str = "closed"
    target_rps: Optional[float] = None
    n_ok: int = 0
    n_err: int = 0
    latencies: List[float] = field(default_factory=list)
    per_function: Dict[str, dict] = field(default_factory=dict)

    def summary(self) -> dict:
        lat = sorted(self.latencies)

        def pct(p):
            if not lat:
                return None
            i = min(len(lat) - 1, int(round(p / 100.0 * (len(lat) - 1))))
            return round(lat[i], 3)

        ok = self.n_ok
        good = sum(1 for x in lat if x <= self.slo_ms)
        return {
            "label": self.label,
            "pipeline": self.pipeline,
            "n_functions": self.n_functions,
            "replicas_per_function": self.replicas,
            "clients": self.clients,
            "mode": self.mode,
            "target_rps": self.target_rps,
            "duration_s": round(self.duration_s, 3),
            "requests_ok": ok,
            "requests_err": self.n_err,
            "throughput_rps": round(ok / self.duration_s, 2)
                              if self.duration_s else None,
            "latency_ms": {
                "mean": round(statistics.fmean(lat), 3) if lat else None,
                "p50": pct(50), "p95": pct(95), "p99": pct(99),
                "max": round(max(lat), 3) if lat else None,
            },
            "slo_ms": self.slo_ms,
            "slo_attainment": round(good / ok, 4) if ok else None,
            "per_function": self.per_function,
        }


class LoadTest:
    """Drive a compiled plan with real concurrent clients."""

    def __init__(self, plan: Plan, replicas: int = 1, spool: Optional[str] = None,
                 rlimit_mb: int = 0, timeout_s: float = 300.0,
                 logdir: Optional[str] = None):
        self.plan = plan
        self.replicas = replicas
        self.spool = spool or SPOOL
        os.makedirs(self.spool, exist_ok=True)
        self.rlimit_mb = rlimit_mb
        self.timeout_s = timeout_s
        self.sets: Dict[str, ReplicaSet] = {
            f.name: ReplicaSet(plan.pipeline, f, replicas, rlimit_mb, logdir)
            for f in plan.functions
        }

    def _hop(self, wire: Dict[str, bytes]) -> Dict[str, bytes]:
        """A real spool write/read per crossing value, as in a FaaS hop."""
        if not wire:
            return {}
        p = os.path.join(self.spool, "hop-%d.bin" % threading.get_ident())
        out = {}
        for k, buf in wire.items():
            with open(p, "wb") as fh:
                fh.write(buf)
            with open(p, "rb") as fh:
                out[k] = fh.read()
        return out

    def serve(self) -> float:
        t_req = time.perf_counter()
        wire: Dict[str, bytes] = {}
        for f in self.plan.functions:
            r = self.sets[f.name].call(f.ops, self._hop(
                {k: wire.get(k, b"") for k in f.inbound}), f.outbound,
                self.timeout_s)
            for k, buf in r["out"].items():
                wire[k] = buf
        return (time.perf_counter() - t_req) * 1000.0

    def run(self, duration_s: float = 20.0, clients: Optional[int] = None,
            rps: Optional[float] = None, label: str = "",
            slo_ms: float = DEFAULT_SLO_MS, warmup: bool = True) -> LoadResult:
        if warmup:
            self.serve()
        n_clients = clients or max(1, self.replicas)
        res = LoadResult(label=label or self.plan.pipeline,
                         pipeline=self.plan.pipeline,
                         n_functions=len(self.plan.functions),
                         replicas=self.replicas, clients=n_clients,
                         duration_s=duration_s, slo_ms=slo_ms,
                         mode="open" if rps else "closed", target_rps=rps)
        stop = threading.Event()
        lock = threading.Lock()

        def record(ms):
            with lock:
                res.n_ok += 1
                res.latencies.append(ms)

        def bump_err():
            with lock:
                res.n_err += 1

        def closed():
            while not stop.is_set():
                try:
                    ms = self.serve()
                except Exception:
                    bump_err()
                    continue
                record(ms)

        def opened(t0, seed):
            rng = random.Random(seed)
            k = 0
            while not stop.is_set():
                k += 1
                due = t0 + rng.expovariate(rps) * k
                now = time.perf_counter()
                if due > now:
                    time.sleep(min(due - now, 0.02))
                    continue
                if due > t0 + duration_s:
                    return
                try:
                    ms = self.serve()
                except Exception:
                    bump_err()
                    continue
                record(ms)

        t0 = time.perf_counter()
        threads = []
        for i in range(n_clients):
            t = threading.Thread(target=opened, args=(t0, i)) if rps \
                else threading.Thread(target=closed)
            t.start()
            threads.append(t)
        time.sleep(duration_s)
        stop.set()
        for t in threads:
            t.join(timeout=300)
        res.duration_s = time.perf_counter() - t0

        for f in self.plan.functions:
            rs = self.sets[f.name]
            res.per_function[f.name] = {
                "device": f.device,
                "n_ops": f.n_ops,
                "replicas": len(rs.workers),
                "calls": rs.calls,
                "peak_rss_mb": rs.rss_mb,
                "gpu_mb": round(rs.gpu_mb, 2),
                "gpu_reserved_mb": round(rs.gpu_reserved_mb, 2),
                "mem_ceiling_mb": f.memory_mb,
                "mem_ok": rs.rss_mb <= f.memory_mb,
                "gpu_ceiling_mb": f.gpu_memory_mb,
                "mem_ok_gpu": (f.gpu_memory_mb <= 0
                               or rs.gpu_mb <= f.gpu_memory_mb),
                "cold_start_ms": rs.cold_ms,
                "cold_starts_ms": rs.spawn_ms,
                "busy_ms": round(rs.busy_ms, 1),
                "queue_wait_ms": round(rs.wait_ms, 1),
                "queue_wait_per_call_ms": (round(rs.wait_ms / rs.calls, 3)
                                           if rs.calls else None),
                "utilization": round(rs.util(res.duration_s * 1000.0), 4),
                "errors": rs.errors,
            }
        return res

    def close(self):
        for rs in self.sets.values():
            rs.close()


def replica_capacity(peak_rss_mb: float, budget_mb: float,
                     reserve_mb: float = 256.0) -> int:
    """How many replicas of a function this size fit in the budget.

    This is the quantity splitting actually improves: a smaller per-replica
    footprint buys concurrency, and concurrency is what absorbs load.
    """
    usable = max(0.0, budget_mb - reserve_mb)
    if peak_rss_mb <= 0:
        return 1
    return max(1, int(usable // peak_rss_mb))


def elasticity_report(per_function: Dict[str, dict], budget_mb: float,
                      reserve_mb: float = 256.0) -> dict:
    """Per-function replica capacity under a fixed deployment memory budget."""
    rows = {}
    for name, r in per_function.items():
        rss = r.get("peak_rss_mb") or 0
        rows[name] = {
            "peak_rss_mb": rss,
            "replicas_in_budget": replica_capacity(rss, budget_mb, reserve_mb),
            "mem_ceiling_mb": r.get("mem_ceiling_mb"),
            "mem_ok": r.get("mem_ok"),
        }
    vals = list(rows.values())
    return {
        "budget_mb": budget_mb,
        "reserve_mb": reserve_mb,
        "per_function": rows,
        "bottleneck": (min(rows, key=lambda k: rows[k]["replicas_in_budget"])
                       if rows else None),
        "min_replicas": min((v["replicas_in_budget"] for v in vals), default=0),
        "max_replicas": max((v["replicas_in_budget"] for v in vals), default=0),
    }


def _partition_by_device(p):
    """Group operators so that declared device is constant inside a function."""
    groups, cur, curdev = [], [], None
    for o in p.ops:
        if o.is_worker:
            if cur:
                groups.append(cur)
                cur = []
            groups.append([o.name])
            curdev = None
            continue
        if curdev is not None and o.device != curdev:
            groups.append(cur)
            cur = []
        cur.append(o.name)
        curdev = o.device
    if cur:
        groups.append(cur)
    return groups


def main():
    """Self-check: one pipeline, several layouts, one load level."""
    from workloads import pipes
    key = os.environ.get("M2S_LOAD_KEY", "pipeline_driving")
    dur = float(os.environ.get("M2S_LOAD_DURATION", "8"))
    reps = int(os.environ.get("M2S_LOAD_REPLICAS", "2"))
    budget = float(os.environ.get("M2S_MEM_BUDGET_MB", "10240"))

    p = pipes.build(key)
    names = p.names()
    layouts = {
        "monolith": [names],
        "by_device": _partition_by_device(p),
        "fission": [[n] for n in names],
    }

    out = {}
    for label, groups in layouts.items():
        if sorted(n for g in groups for n in g) != sorted(names):
            print("  [skip] %s: not a valid partition" % label)
            continue
        plan = compile_plan(p, groups, memory_mb=int(budget))
        rt = LoadTest(plan, replicas=reps)
        try:
            res = rt.run(duration_s=dur, clients=reps, label=label)
            s = res.summary()
            s["elasticity"] = elasticity_report(res.per_function, budget)
            out[label] = s
            print("%-10s fns=%d  %7.2f rps  p50=%-7.1f p95=%-7.1f p99=%-7.1f "
                  "slo=%.2f  min_replicas=%d"
                  % (label, s["n_functions"], s["throughput_rps"] or 0,
                     s["latency_ms"]["p50"] or 0, s["latency_ms"]["p95"] or 0,
                     s["latency_ms"]["p99"] or 0, s["slo_attainment"] or 0,
                     s["elasticity"]["min_replicas"]))
        finally:
            rt.close()

    os.makedirs("results", exist_ok=True)
    path = os.environ.get("M2S_LOAD_OUT", "results/load_selfcheck.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nsaved %s" % path)


if __name__ == "__main__":
    main()
