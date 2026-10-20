"""Where does a one-stage ViT function's RSS actually come from?

VmHWM is monotonic, so walking the load sequence in one process attributes the
peak correctly: each step's number is "the high-water mark reached by this
point". The question matters because RSS -- not bytes -- picks the billing tier.

The walk follows ``pipe_e.stage_module``, i.e. the path a deployed function
takes. An earlier revision also materialised a torchvision ViT here to carve
the architecture out of it; that step is reported separately and then skipped,
because the runtime no longer needs it (``pipe_e._skeleton`` replaced it).

Run from the repository root:
    PYTHONPATH=M2S/src OMP_NUM_THREADS=8 ./.venv/bin/python -u -m workloads.probe_stage_rss
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = []


def hwm():
    peak = rss = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024.0
            elif line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
    return peak, rss


def mark(label):
    peak, rss = hwm()
    STEPS.append({"step": label, "vmhwm_mb": round(peak, 1),
                  "vmrss_mb": round(rss, 1)})
    print("  %-42s hwm=%8.1f  rss=%8.1f" % (label, peak, rss))


def main():
    import torch
    n = int(os.environ.get("M2S_STAGE_IDX", "1"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    mark("bare interpreter (torch imported)")
    from workloads import payload  # noqa: F401
    mark("+payload (cv2)")
    from workloads import pipes, pipe_e
    mark("+pipes registry")
    pipes.build("vit_partition")
    mark("+pipes.build(vit_partition)")

    if os.environ.get("M2S_STAGE_WITH_TV") == "1":
        # What the previous implementation paid: materialising a ViT-L to read
        # the block count out of it. Kept as a control, not on the real path.
        pipe_e._full_meta()
        mark("[control] +_full_meta() (imports torchvision)")

    pipe_e.ensure_shards(verbose=False)
    mark("+ensure_shards")

    t0 = time.perf_counter()
    m = pipe_e.stage_module(n)
    mark("+stage_module(%d)  [%.0f ms]" % (n, (time.perf_counter() - t0) * 1000))
    tv_loaded = "torchvision" in sys.modules

    x = torch.randn(4, 197, 1024)
    with torch.no_grad():
        m(x)
    mark("+first forward")
    with torch.no_grad():
        m(x)
    mark("+second forward")

    out = {
        "stage_idx": n,
        "blocks_per_group": pipe_e.BLOCKS_PER_GROUP,
        "param_mb_per_group": round(pipe_e.stage_param_mb(n), 2),
        "torchvision_loaded": tv_loaded,
        "steps": STEPS,
    }
    path = os.environ.get("M2S_STAGE_OUT", "results/stage_rss.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s (%d blocks, %.1f MB params, torchvision loaded=%s)"
          % (path, pipe_e.BLOCKS_PER_GROUP, out["param_mb_per_group"],
             tv_loaded))


if __name__ == "__main__":
    main()
