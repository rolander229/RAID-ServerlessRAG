"""Import floor under different per-function dependency profiles.

Why this decides the achievable number of functions
--------------------------------------------------
The layout sweep showed the tier ladder forcing full fission: a one-block
function peaks at ~897 MB (tier 1024), and a two-block function at ~1069 MB
(tier 1536) -- 45 MB over the boundary. Adding one block costs ~172 MB, so no
layout other than one-block-per-function can reach the 1024 MB tier.

That 172 MB is mostly the 144 MB of weights. The rest, and the ~750 MB that is
*there before any weights are loaded*, is the framework the handler imports. A
middle encoder stage needs ``torch`` and nothing else: it consumes and produces
float tensors. But this repo's handler imports ``torchvision`` (to build the
stage architecture) and ``cv2`` (via the payload codec) unconditionally, so
every function in the graph pays for them.

This probe measures the floor for each import profile in a fresh process, which
is what makes the "how many functions could fit a 1024 MB tier" question
answerable without guessing.

Usage:
    python -m workloads.probe_vit_imports [--json PATH]
    python -m workloads.probe_vit_imports --child <profile>
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))

# Ordered cheapest-first. ``hold_mb`` allocates that many MB of float32
# parameters, standing in for the shard a stage of that size would hold: the
# probe is about the *floor* plus weights, and a stage's weights are just
# tensors with no runtime cost of their own.
PROFILES = {
    "torch":            {"imports": ["torch"], "hold_mb": 0},
    "torch+numpy":      {"imports": ["torch", "numpy"], "hold_mb": 0},
    "torch+tv":         {"imports": ["torch", "torchvision"], "hold_mb": 0},
    "torch+tv+cv2":     {"imports": ["torch", "torchvision", "cv2"], "hold_mb": 0},
    "handler_full":     {"imports": ["__handler__"], "hold_mb": 0},
    "torch+tv+cv2+288": {"imports": ["torch", "torchvision", "cv2"],
                         "hold_mb": 288},
    "torch+288":        {"imports": ["torch"], "hold_mb": 288},
    "torch+432":        {"imports": ["torch"], "hold_mb": 432},
    "torch+576":        {"imports": ["torch"], "hold_mb": 576},
}


def _vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def child(profile):
    spec = PROFILES[profile]
    for name in spec["imports"]:
        if name == "__handler__":
            # exactly what workloads.worker imports before it accepts a request
            from workloads import payload  # noqa: F401
            from workloads.pipes import build  # noqa: F401
            from workloads.util import sync  # noqa: F401
            from workloads import pipe_e  # noqa: F401
        else:
            __import__(name)
    floor = _vmhwm_mb()

    hold = spec["hold_mb"]
    if hold:
        import torch
        # one tensor per "block", kept alive, mirroring _LOADED
        keep = [torch.zeros(int(hold / 3) * 262144, dtype=torch.float32)
                for _ in range(3)]
        assert sum(t.numel() for t in keep) * 4 / 1048576.0 > hold * 0.95
        del keep
    print(json.dumps({"profile": profile, "hold_mb": hold,
                      "floor_mb": round(floor, 2),
                      "peak_mb": round(_vmhwm_mb(), 2)}))


def run(profile):
    p = subprocess.run([sys.executable, "-u", "-m", "workloads.probe_vit_imports",
                        "--child", profile],
                       cwd=os.path.join(HERE, ".."),
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("child rc=%d\n%s\n%s"
                           % (p.returncode, p.stdout[-1200:], p.stderr[-1200:]))
    line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def main():
    rows = []
    print("  %-18s %10s %10s" % ("profile", "floor_mb", "peak_mb"))
    for name in PROFILES:
        r = run(name)
        rows.append(r)
        print("  %-18s %10.1f %10.1f" % (name, r["floor_mb"], r["peak_mb"]))

    by = {r["profile"]: r for r in rows}
    full = by["handler_full"]["peak_mb"]
    slim = by["torch"]["peak_mb"]
    tv = by["torch+tv"]["peak_mb"]
    t288 = by["torch+288"]["peak_mb"]
    print()
    print("  current handler floor         : %7.1f MB" % full)
    print("  torch only                    : %7.1f MB" % slim)
    print("  torch + torchvision           : %7.1f MB" % tv)
    print("  floor removable by dropping the")
    print("  torchvision (+cv2) import set : %7.1f MB" % (full - slim))
    print()
    print("  marginal cost of a held shard : %7.1f MB per 288 MB of weights"
          % (t288 - by["torch"]["peak_mb"]))
    print("    (i.e. the allocator holds the weights themselves, nothing more,")
    print("     once the loader stops double-allocating them)")

    # Measured per-function peaks from the layout sweep, if it has been run.
    # Deliberately not extrapolated from the floors above: a naive
    # floor + weights model under-predicts the real handler by 100-200 MB
    # (live pipeline state, inbound buffers, the codec's working copy), which is
    # exactly the margin that decides the tier. Tiers have to come from
    # measurement, and this probe exists only to say how much of the floor is
    # removable.
    sweep_path = "results/vit_split.json"
    peak_table = None
    if os.path.exists(sweep_path):
        try:
            with open(sweep_path) as fh:
                sw = json.load(fh)
            peak_table = {}
            for label, r in sw.get("runs", {}).items():
                for f in r["functions"]:
                    n = len(f["stages"])
                    peak_table.setdefault(n, []).append(
                        (label, f["name"], f["rss_mb"], f["tier_mb"]))
        except Exception:
            peak_table = None

    if peak_table:
        print()
        print("  measured peak per model-function, by blocks held:")
        for n in sorted(peak_table):
            vals = [v[2] for v in peak_table[n]]
            tiers = sorted({v[3] for v in peak_table[n]})
            print("    %d block(s): %6.1f - %6.1f MB  -> tier %s"
                  % (n, min(vals), max(vals),
                     "/".join(str(t) for t in tiers)))

    path = os.environ.get("M2S_VIT_IMPORTS_OUT",
                          "results/vit_import_profiles.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"rows": rows, "handler_floor_mb": full,
                   "torch_only_floor_mb": slim,
                   "measured_peak_by_blocks": (
                       {str(k): [[v[0], v[1], v[2], v[3]]
                                 for v in peak_table[k]] for k in peak_table}
                       if peak_table else {})}, fh, indent=2)
    print("\nsaved %s" % path)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1])
    else:
        main()
