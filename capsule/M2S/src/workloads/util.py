import os
import time

import numpy as np
import torch

ROOT = os.environ.get("M2S_ROOT", ".")

# Which device tensors are placed on.
#
# ``M2S_DEVICE`` overrides. The override is not cosmetic: the default path calls
# ``torch.cuda.is_available()``, which *initialises CUDA*. Two things follow. A
# CPU-only run would still pay for a CUDA context it never uses (worker.py makes
# the same point about RSS inflation), and a pre-forked worker
# (``M2S_SPAWN=fork``) would inherit a context PyTorch then refuses to let it
# use. Setting ``M2S_DEVICE=cpu`` keeps CUDA out of the process entirely.
_FORCE_DEV = os.environ.get("M2S_DEVICE", "").strip().lower()
if _FORCE_DEV == "cpu":
    DEV = torch.device("cpu")
elif _FORCE_DEV == "cuda":
    DEV = torch.device("cuda:0")
else:
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Asked once, and only about the device this run actually uses. ``sync`` used to
# call ``is_available()`` on every invocation, which is both a needless CUDA
# probe per call and, in a forked child, a CUDA call at a point where PyTorch
# has marked the process unfit to make one.
_ON_CUDA = DEV.type == "cuda"


def sync():
    if _ON_CUDA:
        torch.cuda.synchronize()


def timeit(fn, warmup=2, repeat=8):
    """Average + p95 latency in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    sync()
    use_cuda = _ON_CUDA
    vals = []
    for _ in range(repeat):
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            sync()
            vals.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            vals.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(vals)
    return float(arr.mean()), float(np.percentile(arr, 95))


def bytes_of(obj):
    """Real serialized payload size in bytes."""
    if isinstance(obj, torch.Tensor):
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    try:
        import pickle
        return len(pickle.dumps(obj))
    except Exception:
        return 0
