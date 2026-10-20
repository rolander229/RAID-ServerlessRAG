"""Does the thread pool decide the billed memory tier?

A FaaS function is deployed with a fixed vCPU count, so ``torch.set_num_threads``
is part of the configuration, not an implementation detail. On this box the
default is the core count, and every OpenMP/MKL worker thread carries its own
scratch arena for the big GEMMs a ViT block issues (1024->4096). Those arenas
are RSS, and RSS picks the billing tier.

Each thread count is measured in its own fresh interpreter: VmHWM is a
high-water mark, so one process can only answer one question.

Run from the repository root:
    PYTHONPATH=M2S/src ./.venv/bin/python -u -m workloads.probe_threads
"""
import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
COUNTS = [1, 2, 4, 8, 16, 32]
REPS = int(os.environ.get("M2S_THREAD_REPS", "3"))

CHILD = textwrap.dedent("""
    import json, os, sys, time
    n = %(n)d
    # Both are load-bearing: OMP_NUM_THREADS governs the OpenMP pool that the
    # GEMM kernels create, torch.set_num_threads the intra-op pool torch sizes.
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    sys.path.insert(0, %(src)r)
    import torch
    torch.set_num_threads(n)
    from workloads import pipes, pipe_e

    pipe_e.ensure_shards(verbose=False)
    t0 = time.perf_counter()
    m = pipe_e.stage_module(1)
    load_ms = (time.perf_counter() - t0) * 1000.0

    x = torch.randn(4, 197, 1024)
    ts = []
    with torch.no_grad():
        for _ in range(%(reps)d):
            t = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()

    hwm = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    rss = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
    print(json.dumps({"threads": n, "hwm_mb": hwm, "rss_mb": rss,
                      "load_ms": load_ms, "fwd_ms": ts[len(ts) // 2],
                      "fwd_all": ts}))
""")


def main():
    src = os.path.join(ROOT, "M2S", "src")
    rows = []
    print("  %8s %10s %10s %10s %10s" % ("threads", "VmHWM", "VmRSS",
                                        "load_ms", "fwd_ms"))
    for n in COUNTS:
        code = CHILD % {"n": n, "src": src, "reps": REPS}
        env = dict(os.environ, M2S_DEVICE="cpu")
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                           env=env)
        if r.returncode != 0:
            print("  %8d FAILED %s" % (n, r.stderr.strip()[-200:]))
            continue
        d = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append(d)
        print("  %8d %9.1fM %9.1fM %10.1f %10.1f"
              % (d["threads"], d["hwm_mb"], d["rss_mb"], d["load_ms"],
                 d["fwd_ms"]))

    out = os.environ.get("M2S_THREAD_OUT", "results/thread_sweep.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": "vit_l_16 stage1", "batch": 4, "profiles": rows},
                  f, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
