"""Measure raw vs wire-format payload on the real operator outputs.

The old optimizer charged tensor.nbytes for every edge. This probe shows what
the same edges cost once encoded the way a real FaaS boundary would carry them.

Run:  cd <root> && PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
        .venv/bin/python -m workloads.probe_payload
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from workloads import payload as P  # noqa: E402
from workloads.pipe_a import batch_tensor, frame  # noqa: E402


def safe(name, fn):
    try:
        return name, fn()
    except Exception as e:
        print("  [skip] %-28s %s: %s" % (name, type(e).__name__, str(e)[:80]))
        return name, None


def main():
    cases = []
    cases.append(safe("frame_720p_bgr", lambda: frame(0)))
    cases.append(safe("frame_1024_bgr", lambda: frame(1, (1024, 1024))))
    cases.append(safe("batch32_224_tensor", lambda: batch_tensor(n=32)))
    cases.append(safe("batch8_224_tensor", lambda: batch_tensor(n=8)))

    try:
        from workloads.pipe_c import sgbm
        f = frame(1)
        cases.append(safe("depth_sgbm", lambda: sgbm(f)))
    except Exception as e:
        print("  [skip] depth:", str(e)[:80])

    rows = []
    for name, obj in cases:
        if obj is None:
            continue
        m = P.metered(obj)
        rows.append(dict(name=name, **m))
        print("  %-24s raw %9.4f MB -> wire %9.4f MB  (%s, %.2f ms)  x%.1f"
              % (name, m["raw_mb"], m["wire_mb"], m["kind"], m["serde_ms"],
                 (m["raw_mb"] / m["wire_mb"]) if m["wire_mb"] else 0))

    os.makedirs("results", exist_ok=True)
    with open("results/payload_wire_format.json", "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("saved results/payload_wire_format.json")


if __name__ == "__main__":
    main()
