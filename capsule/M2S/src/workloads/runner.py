"""Real profiling runner: replaces fabricated sleep-based operators at
M2S/src/profiler/benchmark_profiling_runner.py with real compute.

Run:  .venv/bin/python -m workloads.runner
"""
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads import util  # noqa: E402
from workloads.pipe_b import dnn_ops, recsys_ops, traffic_ops  # noqa: E402
from workloads.pipe_c import driving_ops, image_ops, video_ops  # noqa: E402

PIPELINES = [
    ("dnn_pipeline", "Deep Neural Network (DNN) batch inference", dnn_ops),
    ("recsys_deepfm", "Deep Recommendation (DeepFM ranking)", recsys_ops),
    ("pipeline_traffic", "Smart Traffic (SSDLite vehicle detection)",
     traffic_ops),
    ("pipeline_driving", "Autonomous Driving (det + monocular depth)",
     driving_ops),
    ("pipeline_video", "Video Analytics (decode + RAFT flow + retrieval)",
     video_ops),
    ("pipeline_image", "Image Processing (SR + LRASPP segmentation)",
     image_ops),
]


def has_cuda_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj.is_cuda
    if isinstance(obj, (list, tuple)):
        return any(has_cuda_tensor(x) for x in obj)
    if isinstance(obj, dict):
        return any(has_cuda_tensor(x) for x in obj.values())
    return False


def run_pipeline(name, desc, factory):
    print("=" * 72)
    print(name, "-", desc)
    print("=" * 72)
    ops = factory()
    rows = []
    for op_name, fn in ops:
        rec = {"operator": op_name}
        try:
            cuda = torch.cuda.is_available()
            before = torch.cuda.memory_allocated() if cuda else 0
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            out = fn()
            if cuda:
                torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() if cuda else 0
            peak = torch.cuda.max_memory_allocated() if cuda else 0
            gpu_used = cuda and peak > before + 512 * 1024
            avg, p95 = util.timeit(fn, warmup=2, repeat=8)
            rec.update({
                "avg_ms": round(avg, 3),
                "p95_ms": round(p95, 3),
                "payload_bytes": util.bytes_of(out),
                "payload_mb": round(util.bytes_of(out) / 1048576.0, 4),
                "device": "GPU" if (has_cuda_tensor(out) or gpu_used) else "CPU",
                "gpu_mem_delta_mb": round((after - before) / 1048576.0, 2),
                "gpu_peak_extra_mb": round(max(0, peak - before) / 1048576.0, 2),
            })
        except Exception as e:
            rec.update({"error": "%s: %s" % (type(e).__name__, str(e)[:200])})
            traceback.print_exc()
        rows.append(rec)
        print("  %-34s %s" % (op_name, json.dumps(
            {k: v for k, v in rec.items() if k != "operator"},
            ensure_ascii=False)))
    ok = [r for r in rows if "avg_ms" in r]
    total = sum(r["avg_ms"] for r in ok)
    print("  -> %d/%d operators measured, sequential sum = %.1f ms\n"
          % (len(ok), len(rows), total))
    return {"benchmark_id": name, "description": desc,
            "operators": rows,
            "measured_operators": len(ok),
            "total_operators": len(rows),
            "sequential_latency_ms": round(total, 3)}


OLD_FAKE = {
    "dnn_pipeline": 3541.1,
    "recsys_deepfm": 2149.6,
    "pipeline_traffic": 4196.9,
    "pipeline_driving": 5257.5,
    "pipeline_video": 6551.3,
    "pipeline_image": 3920.9,
}
OLD_FAKE_SLEEP = {
    "dnn_pipeline": 3120.0,
    "recsys_deepfm": 1815.0,
    "pipeline_traffic": 3208.0,
    "pipeline_driving": 3408.0,
    "pipeline_video": 3825.0,
    "pipeline_image": 3615.0,
}


def write_md(out):
    L = []
    L.append("# Real operator profiling (sleep/randn removed)\n")
    m = out["metadata"]
    L.append("> GPU: %s | torch %s (cuda %s) | %s\n"
             % (m["gpu"], m["torch"], m["cuda"], m["timestamp"]))
    L.append("- timing: %s" % m["timing"])
    L.append("- payload: real serialized bytes crossing each operator edge\n")

    L.append("\n## 1. Per-pipeline real measurements\n")
    for b in out["benchmarks"]:
        L.append("\n### %s — %s\n" % (b["benchmark_id"], b["description"]))
        L.append("| operator | device | avg ms | p95 ms | payload MB | "
                 "peak extra MB |")
        L.append("|---|---|---:|---:|---:|---:|")
        for o in b["operators"]:
            if "avg_ms" not in o:
                L.append("| %s | ERROR | %s | | | |"
                         % (o["operator"], o.get("error", "")[:60]))
                continue
            L.append("| %s | %s | %.3f | %.3f | %.4f | %.2f |"
                     % (o["operator"], o["device"], o["avg_ms"], o["p95_ms"],
                        o["payload_mb"], o["gpu_peak_extra_mb"]))
        L.append("\n- measured: %d/%d operators, sequential sum = **%.1f ms**"
                 % (b["measured_operators"], b["total_operators"],
                    b["sequential_latency_ms"]))

    L.append("\n## 2. Fabricated vs real (same benchmarks)\n")
    L.append("| benchmark | old 'measured' ms | of which hand-typed sleep | "
             "real ms now | faked share | reduction |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for b in out["benchmarks"]:
        k = b["benchmark_id"]
        old = OLD_FAKE.get(k, 0.0)
        slp = OLD_FAKE_SLEEP.get(k, 0.0)
        new = b["sequential_latency_ms"]
        share = (slp / old * 100.0) if old else 0.0
        red = ((old - new) / old * 100.0) if old else 0.0
        L.append("| %s | %.1f | %.1f | **%.1f** | %.1f%% | -%.1f%% |"
                 % (k, old, slp, new, share, red))
    L.append("\n> The previous numbers were produced by `time.sleep()` constants"
             " inside operator bodies (see `benchmark_profiling_runner.py`);"
             " they are not measurements.\n")

    p = "results/real_profiling_report.md"
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("saved", p)


def main():
    t0 = time.time()
    meta = {
        "runner_version": "real-1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fake_sleep_removed": True,
        "timing": "CUDA events + warmup(2) + repeat(8), p95 reported",
    }
    out = {"metadata": meta, "benchmarks": []}
    for name, desc, factory in PIPELINES:
        out["benchmarks"].append(run_pipeline(name, desc, factory))

    os.makedirs("results", exist_ok=True)
    with open("results/real_profiling_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved results/real_profiling_results.json")
    write_md(out)
    print("done (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
