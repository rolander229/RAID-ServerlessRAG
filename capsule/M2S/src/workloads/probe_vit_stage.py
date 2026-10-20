"""Per-stage peak RSS, measured in a fresh process per stage.

This is the premise the whole cost story rests on, so it gets measured directly
rather than inferred. A serverless function's *billed* memory tier is set by its
worst-case resident set -- which for a model-loading function is the load, not
the steady-state forward. So each stage is measured in its own process, from
interpreter start through first forward, and VmHWM is read at the end.

Two numbers per stage:
  * ``floor_mb``  -- peak after importing only the framework (torch/torchvision),
                     i.e. what an empty function of this type costs. Read from a
                     child that imports and exits.
  * ``stage_mb``  -- peak after loading this stage's shard and running one
                     forward. Difference = what the shard actually costs.

Usage:
    python -m workloads.probe_vit_stage [--batch N] [--json PATH]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = int(os.environ.get("M2S_VIT_BATCH", "4"))


def _vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


# ---------------------------------------------------------------- child modes
def child_floor():
    """Import exactly what a stage function imports, then stop."""
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    from torchvision.models import vit_l_16  # noqa: F401
    print(json.dumps({"floor_mb": _vmhwm_mb()}))


def child_stage(idx, batch):
    import torch
    from workloads import pipe_e

    t0 = time.perf_counter()
    m = pipe_e.stage_module(idx)
    load_s = time.perf_counter() - t0

    if idx == 0:
        x = torch.randn(batch, 3, pipe_e.RES, pipe_e.RES)
    else:
        x = torch.randn(batch, pipe_e.SEQ, pipe_e.HIDDEN)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = m(x)
    fwd_s = time.perf_counter() - t0

    print(json.dumps({
        "idx": idx,
        "param_mb": round(pipe_e.stage_param_mb(idx), 2),
        "load_s": round(load_s, 3),
        "fwd_ms": round(fwd_s * 1000.0, 1),
        "out_shape": list(y.shape),
        "stage_mb": round(_vmhwm_mb(), 2),
    }))


def run_child(args):
    py = sys.executable
    mod = "workloads.probe_vit_stage"
    p = subprocess.run([py, "-u", "-m", mod] + args, cwd=HERE + "/..",
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("child failed rc=%d\n%s\n%s"
                           % (p.returncode, p.stdout[-2000:], p.stderr[-2000:]))
    line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def main():
    batch = BATCH
    for i, a in enumerate(sys.argv):
        if a == "--batch":
            batch = int(sys.argv[i + 1])

    print("batch = %d, threads = %d" % (batch, __import__("torch").get_num_threads()))

    floor = run_child(["--floor"])
    print("framework floor (imports only) : %.1f MB" % floor["floor_mb"])
    print()

    from workloads import pipe_e
    rows = []
    for i in range(pipe_e.GROUPS):
        r = run_child(["--stage", str(i), "--batch", str(batch)])
        rows.append(r)
        print("  stage%-2d param=%7.2f MB  peak=%7.2f MB  load=%5.2fs  "
              "fwd=%7.1f ms  out=%s"
              % (i, r["param_mb"], r["stage_mb"], r["load_s"], r["fwd_ms"],
                 r["out_shape"]))

    print()
    print("  %-8s %10s %10s %10s" % ("", "param", "peak", "peak-floor"))
    for r in rows:
        print("  stage%-3d %9.2f %9.2f %9.2f"
              % (r["idx"], r["param_mb"], r["stage_mb"],
                 r["stage_mb"] - floor["floor_mb"]))

    out = {"batch": batch, "floor_mb": floor["floor_mb"], "stages": rows}
    path = os.environ.get("M2S_VIT_STAGE_OUT", "results/vit_stage_memory.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved %s" % path)


if __name__ == "__main__":
    if "--floor" in sys.argv:
        child_floor()
    elif "--stage" in sys.argv:
        i = sys.argv.index("--stage")
        b = sys.argv.index("--batch") if "--batch" in sys.argv else None
        child_stage(int(sys.argv[i + 1]), int(sys.argv[b + 1]) if b else BATCH)
    else:
        main()
