"""Run named layouts of any pipeline through the real runtime and price them.

``vit_split_run`` is ViT-shaped: it knows about encoder-block shards, loads a
per-stage checkpoint and compares against a torchvision reference. That is the
right shape for the trade-off study, but it makes a pipeline's *runnability*
depend on having a ViT-style shard layout, which is why the two newer pipelines
(``llm_serve``, ``dlrm_ranking``) had no measured numbers at all.

This driver takes the pipeline as an argument and gets its layouts from the
module's own ``shard_layouts()`` -- the same function that documents which cuts
the pipeline considers interesting. It reports, per layout:

  * per-function peak RSS (VmHWM in that function's own process) and the
    Lambda memory tier that RSS rounds up to
  * per-function wall time and the end-to-end latency
  * per-function spawn and init, and the cold end-to-end
  * GB-seconds and u$/request at the same price and amnestisation model
    ``vit_split_run`` uses, so the numbers are comparable across pipelines

What it does *not* do is check accuracy. None of these pipelines has an
independent reference the way ViT does (torchvision's own forward), and a
comparison against our own monolith would only prove the monolith agrees with
itself. Correctness for pipe_f/pipe_g is established in their own smoke tests
(operator graph vs a flat reference); this driver is for cost and memory.

Run:
    PYTHONPATH=M2S/src OMP_NUM_THREADS=8 \\
      ./.venv/bin/python -u -m workloads.layout_run llm_serve
    ... -m workloads.layout_run dlrm_ranking --reps 3
    ... -m workloads.layout_run llm_serve --layouts L2_phase,L6_both
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads import pipes  # noqa: E402
from workloads.runtime import (LocalFaaSRuntime, compile_plan,  # noqa: E402
                               pin_supervisor)

PRICE_GB_S = 0.0000166667          # $ per GB-s, x86, us-east-1
TIERS = [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240]
CEILING_MAX = 10240
REUSE_FACTORS = [1, 10, 100]

# Which module owns each pipeline's layout catalogue.
LAYOUT_MODULES = {
    "llm_serve": "workloads.pipe_f",
    "dlrm_ranking": "workloads.pipe_g",
}


def tier_of(mb):
    """Smallest tier that holds ``mb``, or the largest if even that is short."""
    for t in TIERS:
        if mb <= t:
            return t
    return TIERS[-1]


def median(xs, default=0.0):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else default


def gb_seconds(fns):
    return sum((f["tier_mb"] / 1024.0) * (f["wall_ms"] / 1000.0) for f in fns)


def cost_per_request(fns, cold_e2e_ms, reuse=None):
    """Warm cost, plus the init amortised over ``reuse`` requests if given."""
    total = gb_seconds(fns) * PRICE_GB_S
    if reuse:
        total += (sum(f["tier_mb"] / 1024.0 for f in fns)
                  * (cold_e2e_ms / 1000.0) * PRICE_GB_S) / float(reuse)
    return total


def build_pipeline(key):
    """A Pipeline for ``key``, registered or not.

    Goes through the registry when it can, and otherwise assembles the pipeline
    from its module's own ``*_ops``/``*_spec`` pair. That fallback is not a
    convenience: registering a pipeline means editing pipes.py, and pipes.py is
    imported by every running worker, so a registration cannot be made while
    another sweep is in flight. Being able to measure before registering keeps
    the two activities independent.
    """
    if key in pipes.keys():
        return pipes.build(key)
    mod = __import__(LAYOUT_MODULES[key], fromlist=["x"])
    st = pipes.TracingState()
    ops = list(mod.llm_spec() if key == "llm_serve" else mod.dlrm_spec())
    fns = mod.llm_ops(st) if key == "llm_serve" else mod.dlrm_ops(st)
    return pipes.Pipeline(key=key, title=key, ops=ops, fns=dict(fns), state=st)


def measure(key, label, groups, reps, timeout_s):
    p = build_pipeline(key)
    plan = compile_plan(p, groups, memory_mb=CEILING_MAX, timeout_s=timeout_s)
    rt = LocalFaaSRuntime()
    try:
        t0 = time.perf_counter()
        cold = rt.invoke(plan, timeout_s=timeout_s)
        cold_e2e = (time.perf_counter() - t0) * 1000.0
        if cold.get("timeout"):
            raise TimeoutError(cold["timeout"])

        t0 = time.perf_counter()
        warm = [rt.invoke(plan, timeout_s=timeout_s) for _ in range(reps)]
        warm_wall = (time.perf_counter() - t0) * 1000.0
        for w in warm:
            if w.get("timeout"):
                raise TimeoutError(w["timeout"])
    finally:
        rt.shutdown()

    fns = []
    for i, cr in enumerate(cold["rows"]):
        wr = [w["rows"][i] for w in warm]
        spawn = cr["cold_start_ms"]
        fns.append({
            "index": i,
            "name": cr["function"],
            "n_ops": cr["n_ops"],
            "ops": cr["ops"],
            "spawn_ms": spawn,
            "cold_compute_ms": cr["compute_ms"],
            "init_ms": round(spawn + cr["wall_ms"], 1),
            "rss_mb": cr["rss_mb"],
            "tier_mb": tier_of(cr["rss_mb"]),
            "wall_ms": round(median([x["wall_ms"] for x in wr]), 3),
            "compute_ms": round(median([x["compute_ms"] for x in wr]), 3),
            "decode_ms": round(median([x["decode_ms"] for x in wr]), 3),
            "encode_ms": round(median([x["encode_ms"] for x in wr]), 3),
            "in_mb": cr["in_mb"],
            "out_mb": cr["out_mb"],
            "hop_ms": cr["hop_ms"],
        })

    e2e = [w["e2e_ms"] for w in warm]
    return {
        "layout": label,
        "n_functions": len(fns),
        "functions": fns,
        "latency_ms": round(median(e2e), 3),
        "latency_cold_ms": round(cold_e2e, 3),
        "warm_wall_ms": round(warm_wall, 1),
        "spawn_sum_ms": round(sum(f["spawn_ms"] for f in fns), 1),
        "init_sum_ms": round(sum(f["init_ms"] for f in fns), 1),
        "max_rss_mb": round(max(f["rss_mb"] for f in fns), 2),
        "rss_sum_mb": round(sum(f["rss_mb"] for f in fns), 2),
        "max_tier_mb": max(f["tier_mb"] for f in fns),
        "est_tier_sum_mb": sum(f["tier_mb"] for f in fns),
        "wire_total_mb": warm[0]["wire_total_mb"],
        "gb_seconds": round(gb_seconds(fns), 4),
        "spawn_mode": cold.get("spawn_mode"),
        "gpu_op_problem": cold.get("gpu_op_problem"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("--layouts", default="",
                    help="comma-separated layout names; default is all")
    ap.add_argument("--reps", type=int,
                    default=int(os.environ.get("M2S_REPS", "3")))
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("M2S_TIMEOUT", "900")))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Same rule as the ViT driver: the supervisor forks workers, and forking
    # out of a process that has run a parallel region wedges the child.
    pin_supervisor()

    key = args.pipeline
    if key not in LAYOUT_MODULES:
        raise SystemExit("no layout catalogue for %r (have %s)"
                         % (key, sorted(LAYOUT_MODULES)))
    mod = __import__(LAYOUT_MODULES[key], fromlist=["shard_layouts"])
    catalogue = mod.shard_layouts()

    only = [x for x in args.layouts.split(",") if x]
    layouts = {k: v for k, v in catalogue.items() if not only or k in only}
    if not layouts:
        raise SystemExit("no such layout; have %s" % sorted(catalogue))

    spawn = os.environ.get("M2S_SPAWN", "subprocess").strip().lower()
    print("%s: %d layouts, reps=%d, spawn=%s, threads=%s"
          % (key, len(layouts), args.reps, spawn,
             os.environ.get("OMP_NUM_THREADS", "auto")))
    print()

    runs = {}
    audited = 0
    for label, groups in sorted(layouts.items(), key=lambda kv: len(kv[1])):
        # Cheap and static, so it runs before anything is spawned. Without it a
        # missing cross-function key shows up as a worker EOF several minutes
        # later and names nothing.
        problems = pipes.audit_layout(pipes.build(key).ops, groups)
        if problems:
            print("  %-16s UNSOUND LAYOUT" % label)
            for x in problems:
                print("      %s" % x)
            continue
        audited += 1
        t0 = time.time()
        try:
            r = measure(key, label, groups, args.reps, args.timeout)
        except Exception as e:
            print("  %-16s FAILED %s: %s" % (label, type(e).__name__,
                                             str(e)[:160]))
            continue
        r["cost_usd"] = cost_per_request(r["functions"], r["latency_cold_ms"])
        r["cost_usd_reuse"] = {str(k): cost_per_request(
            r["functions"], r["latency_cold_ms"], reuse=k)
            for k in REUSE_FACTORS}
        runs[label] = r
        print("  %-16s fns=%d warm=%8.1fms cold=%9.1fms maxrss=%7.1fMB "
              "tier=%5dMB wire=%7.3fMB  (%.0fs)"
              % (label, r["n_functions"], r["latency_ms"], r["latency_cold_ms"],
                 r["max_rss_mb"], r["max_tier_mb"], r["wire_total_mb"],
                 time.time() - t0))

    if not runs:
        raise SystemExit("no layout completed")
    if audited != len(layouts):
        print("  (%d of %d layouts were sound)"
              % (audited, len(layouts)))

    print()
    print("  %-16s %3s %9s %10s %10s %10s %9s"
          % ("layout", "fn", "gbs", "u$/req", "reuse=1", "reuse=10", "cold_ms"))
    for label, r in runs.items():
        c = r["cost_usd_reuse"]
        print("  %-16s %3d %9.4f %10.3f %10.3f %10.3f %9.1f"
              % (label, r["n_functions"], r["gb_seconds"], r["cost_usd"] * 1e6,
                 c["1"] * 1e6, c["10"] * 1e6, r["latency_cold_ms"]))

    print()
    print("  per-function detail")
    print("  %-16s %-24s %3s %9s %7s %9s %9s"
          % ("layout", "function", "ops", "rss_mb", "tier", "wall", "init"))
    for label, r in runs.items():
        for f in r["functions"]:
            print("  %-16s %-24s %3d %9.1f %7d %9.1f %9.1f"
                  % (label, f["name"][:24], f["n_ops"], f["rss_mb"],
                     f["tier_mb"], f["wall_ms"], f["init_ms"]))

    base = runs.get("L1_monolith") or runs.get("G1_monolith")
    if base:
        print()
        print("  vs the monolith baseline")
        print("  %-16s %8s %10s %10s %9s"
              % ("layout", "dCost", "dLatency", "dTier", "dMaxRSS"))
        for label, r in runs.items():
            print("  %-16s %+7.1f%% %+9.1f%% %9s %+9.1fMB"
                  % (label,
                     (r["cost_usd"] / base["cost_usd"] - 1.0) * 100.0,
                     (r["latency_ms"] / base["latency_ms"] - 1.0) * 100.0,
                     "%d->%d" % (base["max_tier_mb"], r["max_tier_mb"]),
                     r["max_rss_mb"] - base["max_rss_mb"]))

    path = args.out or os.environ.get("M2S_LAYOUT_OUT",
                                      "results/layout_%s.json" % key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({
            "pipeline": key,
            "spawn": spawn,
            "worker_threads": os.environ.get("OMP_NUM_THREADS", "auto"),
            "reps": args.reps,
            "price_gb_s": PRICE_GB_S,
            "tiers_mb": TIERS,
            "reuse_factors": REUSE_FACTORS,
            "layouts": {k: list(v) for k, v in catalogue.items()},
            "runs": runs,
        }, fh, indent=2, ensure_ascii=False)
    print("\nsaved %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
