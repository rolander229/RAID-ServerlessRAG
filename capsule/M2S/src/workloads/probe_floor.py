"""Layered import-floor probe: what a function costs before it does any work.

Why this is its own module
--------------------------
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is **inherited across fork** on
this kernel. A child spawned from a process that has already imported torch
therefore reports the *parent's* peak. Measured on v100:

    fresh python, ru_maxrss                          ->  13.0 MB
    child of a torch-importing parent, ru_maxrss     -> 381.2 MB  (inherited!)
    same child, /proc/self/status:VmHWM              ->  11.1 MB  (correct)

So any ru_maxrss-based floor measurement taken from a fat driver process is
fiction -- which is exactly how the first version of this probe produced a table
where every layer read 540.8 MB.

This module reads ``/proc/self/status:VmHWM`` *inside each child*, which is not
inherited, and is therefore safe to call from anywhere. ``worker.py`` reports
its per-function ``rss_mb`` from VmHWM for the same reason.

Run:
    cd <root> && PYTHONPATH=M2S/src .venv/bin/python -m workloads.probe_floor
"""
import json
import os
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# layers, in import order. Each layer = "everything above, plus this".
# ---------------------------------------------------------------------------
LAYERS = [
    ("empty_interpreter", []),
    ("numpy", ["numpy"]),
    ("torch", ["numpy", "torch"]),
    ("torchvision", ["numpy", "torch", "torchvision"]),
    ("cv2", ["numpy", "torch", "torchvision", "cv2"]),
    ("transformers", ["numpy", "torch", "torchvision", "cv2",
                      "transformers"]),
]

_CHILD = r'''
import sys
for m in sys.argv[1:]:
    __import__(m)
hwm = None
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        hwm = int(line.split()[1]) / 1024.0
        break
print(hwm)
'''


def repo_root():
    """Walk up until we find the checkout that owns both M2S/ and results/."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "M2S")) and \
           os.path.isdir(os.path.join(d, "results")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.environ.get("M2S_ROOT", os.getcwd())


def _child_hwm(body):
    """Run *body* in a fresh interpreter and return its VmHWM peak in MB."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True)
        tail = (r.stdout or "").strip().splitlines()
        return float(tail[-1]) if tail else None
    except Exception:
        return None
    finally:
        os.unlink(path)


def measure(root=None):
    """Return the layered floor table. Safe to call from a fat process."""
    root = root or repo_root()
    out = {
        "method": ("/proc/self/status:VmHWM read inside each fresh child "
                   "process; ru_maxrss is inherited across fork and would "
                   "report the parent's peak"),
        "layers": {},
        "incremental_mb": {},
    }
    for name, mods in LAYERS:
        out["layers"][name] = _child_hwm(
            _CHILD.replace("sys.argv[1:]", repr(list(mods))))
    # plus the whole pipeline registry, which is what a real worker imports
    out["layers"]["workloads_registry"] = _child_hwm(
        "import sys\nsys.path.insert(0, %r)\nimport workloads.pipes\n"
        "for line in open('/proc/self/status'):\n"
        "    if line.startswith('VmHWM:'):\n"
        "        print(int(line.split()[1]) / 1024.0); break\n" % root)

    prev = 0.0
    for name, _ in LAYERS + [("workloads_registry", None)]:
        v = out["layers"].get(name)
        if v is not None:
            out["incremental_mb"][name] = round(v - prev, 1)
            prev = v
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def main():
    out = measure()
    results = os.path.join(repo_root(), "results")
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "runtime_floor.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("%-24s %10s %10s" % ("layer", "peak MB", "delta MB"))
    for name, _ in LAYERS + [("workloads_registry", None)]:
        print("%-24s %10s %10s"
              % (name, out["layers"].get(name),
                 out["incremental_mb"].get(name)))
    print("\nsaved %s" % path)


if __name__ == "__main__":
    main()
