"""Calibration: how long does one ViT-L/16 forward take on this CPU?

Needed before the split experiment, because the latency side of the trade-off
is only meaningful if the compute time is large enough to be measured against
realistically. Also prints peak RSS so the monolith's memory tier is known
beforehand.
"""
import os
import time

import torch

T = int(os.environ.get("T", "8"))
torch.set_num_threads(T)
print("threads : %d" % T)
print("cpu cnt : %d" % (os.cpu_count() or 0), flush=True)

from torchvision.models import ViT_L_16_Weights, vit_l_16  # noqa: E402


def vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


t0 = time.perf_counter()
m = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).eval()
print("load    : %.2fs  VmHWM=%.1f MB" % (time.perf_counter() - t0, vmhwm_mb()),
      flush=True)

for b in (1, 4, 8):
    x = torch.randn(b, 3, 224, 224)
    with torch.no_grad():
        for _ in range(2):
            m(x)
        t0 = time.perf_counter()
        for _ in range(3):
            m(x)
        dt = (time.perf_counter() - t0) / 3
    print("batch %2d : %.3f s/forward   %.2f img/s   VmHWM=%.1f MB"
          % (b, dt, b / dt, vmhwm_mb()), flush=True)
