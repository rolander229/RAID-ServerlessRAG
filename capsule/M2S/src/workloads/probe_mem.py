"""Per-operator memory probe -- the memory model's inputs.

The old solver priced a function with three hard-coded constants
(base 680 MB, CUDA context 220 MB, 10 GB baseline). Those numbers were not
measured anywhere, which is the defect the fabrication audit called out.

This probe produces the replacement: every operator is run *alone in its own
function process* (the fission layout), so its peak RSS and peak VRAM are
observed rather than assumed, and the two per-function floors are taken as the
lowest RSS actually observed in each device class.

Caveat, stated up front: the floor measured this way is "the smallest function
we actually deployed", not an abstract empty interpreter. It therefore includes
whatever a real function of that class cannot avoid -- framework import, CUDA
context -- which is exactly the quantity a deployment budget has to pay.

Run:
    cd <root> && PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
        .venv/bin/python -u -m workloads.probe_mem
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipes import build, keys  # noqa: E402
from workloads.runtime import LocalFaaSRuntime, compile_plan  # noqa: E402

# NOTE: the layered import floor deliberately lives in `probe_floor.py` /
# `results/runtime_floor.json`, not here. It used to be embedded in this file's
# output as `import_floor_mb`, measured with `resource.getrusage(...).ru_maxrss`
# -- which is inherited across fork, so when this probe ran under a driver that
# had already imported torch, every layer read the driver's peak (540.8 MB) and
# the table was fiction. One fact, one file: `probe_floor.py` owns the floor.


def main():
    # NB: "".split(",") == [""] which is truthy -- filter first, then fall back,
    # otherwise an unset M2S_PROBE_KEYS silently yields an empty pipeline list and
    # the probe writes an empty op_memory.json.
    which = [k for k in os.environ.get("M2S_PROBE_KEYS", "").split(",") if k]
    if not which:
        which = keys()
    repeat = int(os.environ.get("M2S_PROBE_REPEAT", "2"))

    ops = {}
    rows_all = []
    for k in which:
        p = build(k)
        plan = compile_plan(p, [[n] for n in p.names()])
        rt = LocalFaaSRuntime()
        try:
            rt.invoke(plan)
            out = [rt.invoke(plan) for _ in range(repeat)]
        except Exception as e:
            print("  [fail] %s: %s: %s" % (k, type(e).__name__, str(e)[:90]))
            continue
        finally:
            rt.shutdown()
        last = out[-1]
        for r in last["rows"]:
            for opname in r["ops"]:
                ops[opname] = {
                    "pipeline": k,
                    "device": r["device"],
                    "rss_mb": max(x["rows"][i]["rss_mb"]
                                  for x in out
                                  for i in range(len(x["rows"]))
                                  if x["rows"][i]["function"] == r["function"]),
                    "gpu_mb": max(x["rows"][i].get("gpu_mb", 0.0)
                                  for x in out
                                  for i in range(len(x["rows"]))
                                  if x["rows"][i]["function"] == r["function"]),
                    "gpu_reserved_mb": max(
                        x["rows"][i].get("gpu_reserved_mb", 0.0)
                        for x in out
                        for i in range(len(x["rows"]))
                        if x["rows"][i]["function"] == r["function"]),
                }
            rows_all.append({"pipeline": k, **r})
        print("%-18s %d singleton functions measured" % (k, len(last["rows"])))

    cpu = [o["rss_mb"] for o in ops.values() if o["device"] == "CPU"]
    gpu = [o["rss_mb"] for o in ops.values() if o["device"] == "GPU"]
    base_cpu = min(cpu) if cpu else 0.0
    base_gpu = min(gpu) if gpu else base_cpu

    path = "results/op_memory.json"
    if not ops:
        # Never clobber a good result with an empty one. An empty probe means the
        # pipeline list was empty (or every pipeline failed), not that the
        # operators cost nothing -- writing 0 MB floors downstream is worse than
        # not writing at all.
        print("\n[ABORT] no operator measured (%d pipelines requested: %s); "
              "refusing to overwrite %s" % (len(which), ",".join(which) or "-",
                                            path))
        return 2

    doc = {
        "note": ("RSS/VRAM of each operator running alone in its own function; "
                 "bases are the observed floors, not invented constants. The "
                 "layered import floor lives in runtime_floor.json."),
        "base_cpu_mb": round(base_cpu, 2),
        "base_gpu_mb": round(base_gpu, 2),
        "n_operators": len(ops),
        "ops": ops,
        "raw_rows": rows_all,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)

    print("\nfloor cpu=%7.1f MB   floor gpu=%7.1f MB" % (base_cpu, base_gpu))
    print("saved %s (%d operators)" % (path, len(ops)))
    for name, o in sorted(ops.items(), key=lambda kv: -kv[1]["rss_mb"])[:12]:
        print("  %-32s %-4s rss=%8.1f (+%7.1f)  gpu=%8.1f"
              % (name, o["device"], o["rss_mb"],
                 o["rss_mb"] - (base_gpu if o["device"] == "GPU" else base_cpu),
                 o["gpu_mb"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
