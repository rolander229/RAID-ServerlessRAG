"""Registry-driven per-operator compute profile.

Why this exists
---------------
``results/real_profiling_results.json`` is produced by ``M2S/src/profiler/``,
which predates the ``workloads.pipes`` registry and is wired to the old
benchmark layout. So every operator added after the refactor -- the whole
ViT-L/16 stage family -- is absent from it, and ``optimizer.load`` then assigns
those operators ``compute_ms = 0.0``. The effect is not a small error: the
solver priced the entire ViT pipeline at 2.1 ms when the measured monolith is
656 ms, and consequently recommended a 4-way split that the real runtime shows
is worse than the monolith.

Silently absent measurements are the failure mode this whole rewrite is meant to
remove, so rather than patch a number in, this probe measures the operators and
emits records in the same shape, which ``optimizer.load`` merges on request.

What is measured
----------------
Per operator, in the pipeline's own order, on a warm state: repeated single-op
invocations, timed around ``util.sync()`` so any asynchronous device work is
included. GPU operators are timed with CUDA events; CPU operators with a wall
clock, which for a single-threaded call is the same thing.

Usage:
    python -m workloads.probe_op_time --keys vit_partition --reps 5 \
        --out results/vit_profiling_results.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipes import build, keys as all_keys  # noqa: E402
from workloads.util import sync  # noqa: E402


def _p95(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[i]


def profile_pipeline(key, reps=5):
    p = build(key)
    names = p.names()

    # One full pass first: it loads the weights and fills the state, so the
    # timings below are of the operator, not of its first-call initialisation.
    p.run_ops(names)
    sync()

    import torch
    has_cuda = torch.cuda.is_available()

    ops = []
    for name in names:
        o = p.by_name(name)
        samples = []
        if has_cuda and o.device == "GPU":
            for _ in range(reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                p.run_ops([name])
                e.record()
                torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
        else:
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                p.run_ops([name])
                sync()
                samples.append((time.perf_counter() - t0) * 1000.0)

        gpu_extra = 0.0
        if has_cuda and o.device == "GPU":
            d = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(d)
            base = torch.cuda.memory_allocated(d)
            p.run_ops([name])
            torch.cuda.synchronize()
            gpu_extra = (torch.cuda.max_memory_allocated(d) - base) / 1048576.0

        avg = sum(samples) / len(samples)
        ops.append({
            "operator": name,
            "device": o.device,
            "avg_ms": round(avg, 4),
            "min_ms": round(min(samples), 4),
            "max_ms": round(max(samples), 4),
            "p95_ms": round(_p95(samples), 4),
            "samples": len(samples),
            "gpu_peak_extra_mb": round(max(0.0, gpu_extra), 3),
            "source": "probe_op_time",
        })
        print("  %-18s %-4s avg=%9.4f ms  min=%9.4f  p95=%9.4f"
              % (name, o.device, avg, min(samples), _p95(samples)))
    return {"benchmark": key, "operators": ops}


def main():
    argv = sys.argv[1:]
    ks = None
    reps = 5
    out_path = None
    for i, a in enumerate(argv):
        if a == "--keys":
            ks = argv[i + 1].split(",")
        if a == "--reps":
            reps = int(argv[i + 1])
        if a == "--out":
            out_path = argv[i + 1]
    ks = ks or all_keys()
    out_path = out_path or "results/op_profiling_results.json"

    out = {"metadata": {"probe": "workloads.probe_op_time",
                        "reps": reps,
                        "note": "merge into the solver's input via "
                                "optimizer.load(root, extra_prof=[...])"},
           "benchmarks": []}
    for k in ks:
        print("%s" % k)
        out["benchmarks"].append(profile_pipeline(k, reps=reps))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("saved %s (%d benchmarks)"
          % (out_path, len(out["benchmarks"])))


if __name__ == "__main__":
    main()
