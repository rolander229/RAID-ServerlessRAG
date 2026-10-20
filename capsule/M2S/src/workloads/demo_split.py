"""Turn a partition plan into N real OS processes and measure it.

Two plans per pipeline:
  monolith  -- every operator in one function (what the old solver degenerated to)
  by_device -- contiguous CPU/GPU runs become separate functions

This is the check the project never had: does the split actually exist at
runtime, how many processes are there, what crosses each boundary, and what
does the boundary cost.

Run:  cd <root> && PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
        .venv/bin/python -m workloads.demo_split [pipeline ...]
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipes import build, keys  # noqa: E402
from workloads.runtime import LocalFaaSRuntime, compile_plan  # noqa: E402


def monolith(p):
    return [p.names()]


def by_device(p):
    """Contiguous runs of the same device class."""
    groups, cur, curdev = [], [], None
    for o in p.ops:
        if curdev is None or o.device == curdev:
            cur.append(o.name)
            curdev = o.device
        else:
            groups.append(cur)
            cur = [o.name]
            curdev = o.device
    groups.append(cur)
    return groups


def fission(p):
    return [[n] for n in p.names()]


def measure(plan, warmup, repeat):
    rt = LocalFaaSRuntime()
    try:
        for _ in range(warmup):
            rt.invoke(plan)
        runs = [rt.invoke(plan) for _ in range(repeat)]
    finally:
        rt.shutdown()
    e2e = [r["e2e_ms"] for r in runs]
    last = runs[-1]
    return {
        "n_functions": last["n_functions"],
        "e2e_p50_ms": round(statistics.median(e2e), 3),
        "e2e_min_ms": round(min(e2e), 3),
        "e2e_all": [round(v, 3) for v in e2e],
        "cold_start_ms": [r["cold_start_ms"] for r in last["rows"]],
        "rows": last["rows"],
        "wire_total_mb": last["wire_total_mb"],
    }


PLANS = [("monolith", monolith), ("by_device", by_device), ("fission", fission)]


def main():
    which = sys.argv[1:] or keys()
    repeat = int(os.environ.get("M2S_REPEAT", "3"))
    warmup = int(os.environ.get("M2S_WARMUP", "1"))
    out = {"repeat": repeat, "warmup": warmup, "benchmarks": []}

    for k in which:
        p = build(k)
        print("=" * 74)
        print("%s  (%d operators)" % (k, len(p.ops)))
        print("=" * 74)
        rec = {"pipeline": k, "n_operators": len(p.ops), "plans": {}}
        for label, maker in PLANS:
            plan = compile_plan(p, maker(p))
            print("  [%s] %d function(s)" % (label, plan.n_functions()))
            if os.environ.get("M2S_SHOW_PLAN"):
                print(plan.render())
            m = measure(plan, warmup, repeat)
            rec["plans"][label] = m
            print("     e2e p50 = %8.2f ms   wire = %7.4f MB   cold = %s"
                  % (m["e2e_p50_ms"], m["wire_total_mb"],
                     [round(c) for c in m["cold_start_ms"]]))
            for r in m["rows"]:
                print("       %-30s %-4s ops=%d pid=%-7d compute=%8.2f "
                      "in=%7.4fMB hop=%6.2fms rss=%7.1fMB"
                      % (r["function"], r["device"], r["n_ops"], r["pid"],
                         r["compute_ms"], r["in_mb"], r["hop_ms"], r["rss_mb"]))
        out["benchmarks"].append(rec)
        print()

    os.makedirs("results", exist_ok=True)
    path = "results/split_runtime_demo.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("saved", path)


if __name__ == "__main__":
    main()
