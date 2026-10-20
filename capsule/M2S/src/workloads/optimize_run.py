"""Run the layout optimizer on real measurements and report.

Adds a ceiling sweep: for each provider memory ceiling, what is the smallest
number of functions that is *feasible*? At a generous ceiling the monolith
fits; below a threshold it does not and splitting becomes mandatory. That
threshold is the quantitative answer to "is this pipeline splittable at all".

Run:  cd <root> && PYTHONPATH=M2S/src .venv/bin/python -m workloads.optimize_run
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads import optimizer as OPT  # noqa: E402
from workloads.pipes import build, keys  # noqa: E402

SWEEP = [512, 768, 1024, 1152, 1280, 1408, 1536, 2048, 3008, 4096, 6144, 10240]

# Invocations served by one warm instance before it is recycled. This is the
# knob that decides how much the one-off init matters, and it is the reason a
# split that wins on warm cost can still lose on the bill.
REUSE = float(os.environ.get("M2S_REUSE", "10"))
COLD_SOURCE = os.environ.get("M2S_COLD_SOURCE", "results/vit_split.json")
# Secondary profiling files, merged for operators the legacy profiler never saw
# (everything added after the workloads.pipes refactor).
EXTRA_PROF = [p for p in os.environ.get(
    "M2S_EXTRA_PROF", "results/vit_profiling_results.json").split(",") if p]


def floors_kw(floors):
    """Only pass floors that were actually measured."""
    out = {}
    if floors.get("base_cpu_mb"):
        out["base_cpu_mb"] = floors["base_cpu_mb"]
    if floors.get("base_gpu_mb"):
        out["base_gpu_mb"] = floors["base_gpu_mb"]
    return out


def cold_kw():
    cold = OPT.ColdStart.calibrate(COLD_SOURCE)
    return {"cold": cold, "reuse": REUSE}, cold


def dump_plan(p):
    return {
        "n_functions": p.n(),
        "latency_ms": p.latency_ms,
        "latency_cold_ms": p.latency_cold_ms,
        "init_sum_ms": p.init_sum_ms,
        "cost_usd": p.cost_usd,
        "cost_usd_cold": p.cost_usd_cold,
        "comm_mb": p.comm_mb,
        "feasible": p.feasible,
        "violations": p.violations,
        "cold_model": p.cold_model,
        "overhead_model": p.overhead_model,
        "functions": [
            {"name": f.logical_name, "ops": f.operators,
             "device": f.device_type, "memory_mb": f.recommended_memory_mb,
             "peak_mb": f.peak_memory_mb, "timeout_s": f.recommended_timeout_sec,
             "compute_ms": f.total_compute_latency_ms,
             "inbound": f.inbound_keys, "inbound_mb": f.inbound_mb,
             "inbound_serde_ms": f.inbound_serde_ms,
             "memory_model": f.memory_model, "est_gpu_mb": f.est_gpu_mb}
            for f in p.functions
        ],
    }


def main():
    opcost, keycost, floors = OPT.load(".", extra_prof=EXTRA_PROF)
    cold_kwargs, cold = cold_kw()
    out = {"sweep_ceilings_mb": SWEEP, "measured_floors": floors,
           "extra_prof": list(EXTRA_PROF),
           "cold_start_model": {"import_ms": cold.import_ms,
                                "weight_ms_per_mb": cold.weight_ms_per_mb,
                                "source": cold.source},
           "reuse": REUSE,
           "pipelines": {}, "skipped": {}}
    print("measured floors: %s" % (floors or
                                  "none (falling back to constants)"))
    print("cold-start model: import %.0f ms + %.2f ms/MB  [%s]"
          % (cold.import_ms, cold.weight_ms_per_mb, cold.source))
    print("reuse factor   : %g invocations per warm instance" % REUSE)

    for k in keys():
        p = build(k)
        missing = [n for n in p.names() if n not in opcost]
        if missing:
            out["skipped"][k] = {
                "reason": "operators without measured profiling",
                "operators": missing[:6]}
            print("=" * 78)
            print("%s  SKIPPED: %d operators have no profiling record: %s"
                  % (k, len(missing), missing[:4]))
            continue
        print("=" * 78)
        print("%s  (%d operators)" % (k, len(p.ops)))
        print("=" * 78)

        opt = OPT.LayoutOptimizer(p, opcost, keycost, **floors_kw(floors),
                                  **cold_kwargs)
        res = opt.solve()
        rec = {
            "n_operators": res["n_operators"],
            "n_partitions_evaluated": res["n_partitions_evaluated"],
            "monolith_feasible_at_default_ceiling": res["monolith_feasible"],
            "monolith_violations": res["monolith_violations"],
            "forced_split": res["forced_split"],
            "pareto": [dump_plan(x) for x in res["pareto"]],
            "latency_optimal": dump_plan(res["latency_optimal"]),
            "cost_optimal": dump_plan(res["cost_optimal"]),
            "sweep": {},
        }
        print("  partitions evaluated : %d" % res["n_partitions_evaluated"])
        print("  monolith feasible    : %s   %s"
              % (res["monolith_feasible"], res["monolith_violations"]))
        print("  latency-optimal      : %d fn, %.2f ms, $%.3e"
              % (res["latency_optimal"].n(), res["latency_optimal"].latency_ms,
                 res["latency_optimal"].cost_usd))
        print("  cost-optimal         : %d fn, %.2f ms, $%.3e"
              % (res["cost_optimal"].n(), res["cost_optimal"].latency_ms,
                 res["cost_optimal"].cost_usd))
        print("  pareto size          : %d" % len(res["pareto"]))
        for x in res["pareto"]:
            print("     %d fn  lat %8.2f ms  cost %.3e  comm %.4f MB"
                  % (x.n(), x.latency_ms, x.cost_usd, x.comm_mb))

        # Cold start: the first request after scale-from-zero, and the amortised
        # cost of the init. Reported for both ends of the Pareto front, because
        # "the split is cheaper" is only true once the init is paid for.
        print("  cold start (reuse=%g):" % REUSE)
        for tag, pl in (("latency-optimal", res["latency_optimal"]),
                        ("cost-optimal", res["cost_optimal"])):
            print("     %-16s init %8.0f ms across %d fn  ->  first request "
                  "%9.0f ms  (warm %8.0f ms)"
                  % (tag, pl.init_sum_ms, pl.n(), pl.latency_cold_ms,
                     pl.latency_ms))
            print("                      warm %.3e $/req  ->  incl. init "
                  "%.3e $/req" % (pl.cost_usd, pl.cost_usd + pl.cost_usd_cold))

        print("  ceiling sweep (minimum feasible functions):")
        for c in SWEEP:
            o2 = OPT.LayoutOptimizer(p, opcost, keycost,
                                     ceilings={"memory_mb": c},
                                     **floors_kw(floors), **cold_kwargs)
            r2 = o2.solve()
            mono_ok = r2["monolith_feasible"]
            lat = r2["latency_optimal"]
            mn = r2["min_functions_feasible"]
            rec["sweep"][str(c)] = {
                "monolith_feasible": mono_ok,
                "deployable": r2["deployable"],
                "min_functions_feasible": mn,
                "n_infeasible_partitions": r2["n_infeasible_partitions"],
                "latency_optimal_n_functions": lat.n(),
                "latency_optimal_ms": lat.latency_ms,
            }
            if mn is None:
                verdict = "NO FEASIBLE LAYOUT (below framework floor)"
            else:
                verdict = "min functions = %d" % mn
            print("     ceiling %5d MB -> monolith %-9s %s"
                  % (c, "feasible" if mono_ok else "INFEASIBLE", verdict))

        out["pipelines"][k] = rec

        # The band edges by bisection, so they are exact rather than snapped to
        # the sweep grid above.
        b = opt.band()
        rec["band"] = b
        if b["band_lo_mb"] and b["band_hi_mb"]:
            print("  forced-split band    : [%d, %d) MB, width %d MB, "
                  "min functions in band = %s"
                  % (b["band_lo_mb"], b["band_hi_mb"], b["band_width_mb"],
                     b["min_functions_in_band"]))
        else:
            print("  forced-split band    : none (no ceiling makes the monolith "
                  "infeasible while some split stays feasible)")
        print()

    os.makedirs("results", exist_ok=True)
    path = "results/layout_optimization.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("saved", path)


if __name__ == "__main__":
    main()
