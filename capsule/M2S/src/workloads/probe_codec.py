"""Codec microbenchmark on the real boundary payload.

Motivation
----------
The layout sweep measured ~58 ms to encode one (4,197,1024) boundary tensor, and
isolating the codec showed zlib level 1 alone accounts for ~54 ms of it. That is
the single largest per-hop cost in the pipeline -- larger than the tmpfs write,
the socket transfer and the decode combined -- so before concluding anything
about the split's latency penalty, it is worth knowing whether that is a
property of the *wire format* or of the *codec choice*.

Candidates that need no new dependency:
    raw32   fp32 bytes, no codec          -- zero CPU, 2x the bytes
    raw16   fp16 cast, no codec           -- one cast, 1.57 MB
    z0/z1/z6  fp16 + deflate at level 0/1/6
    quant8  per-tensor linear quantisation to uint8

The interesting axis is encode CPU vs wire bytes, because a serverless hop pays
both: the encode happens inside a *billed* function, and the bytes cross a link
that in a real deployment is a network, not tmpfs.

Usage:
    python -m workloads.probe_codec [--real]
"""
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPS = int(os.environ.get("M2S_CODEC_REPS", "20"))


def payload(real=False):
    """A boundary tensor.

    ``real=False`` gives i.i.d. Gaussian noise, which is the *worst* case for a
    codec and the right one for a head-to-head comparison: it removes "this
    tensor happened to be compressible" from the result. ``real=True`` runs one
    real ViT-L encoder stage so the tensor has the statistics a deployed hop
    would actually see (weights, not noise -- ViT activations after a LayerNorm
    are close to unit-variance but strongly correlated across the sequence).
    """
    from workloads import pipe_e
    if not real:
        return torch.randn(pipe_e.BATCH, pipe_e.SEQ, pipe_e.HIDDEN)

    m = pipe_e.stage_module(1)
    x = torch.randn(pipe_e.BATCH, pipe_e.SEQ, pipe_e.HIDDEN)
    with torch.no_grad():
        return m(x)


def _timed(fn, reps=REPS):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t0) / reps * 1000.0, out


def cand_raw32(x):
    arr = x.detach().cpu().numpy()
    return arr.tobytes()


def cand_raw16(x):
    arr = x.detach().cpu().numpy().astype(np.float16)
    return arr.tobytes()


def _fp16_bytes(x):
    return x.detach().cpu().numpy().astype(np.float16).tobytes()


def cand_zlib(level):
    def f(x):
        return zlib.compress(_fp16_bytes(x), level)
    return f


def cand_quant8(x):
    arr = x.detach().cpu().numpy().ravel()
    lo, hi = float(arr.min()), float(arr.max())
    q = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return q.tobytes()


def main():
    real = "--real" in sys.argv
    x = payload(real)
    n = x.numel()
    print("payload %s  %s  (%.2f M elements)"
          % (tuple(x.shape), "REAL ViT stage-1 output" if real
             else "i.i.d. Gaussian", n / 1e6))
    print()
    print("  %-10s %10s %10s %10s %10s"
          % ("codec", "enc_ms", "dec_ms", "MB", "MB/s enc"))

    rows = []
    cands = [
        ("raw32", cand_raw32, lambda b: np.frombuffer(b, np.float32)),
        ("raw16", cand_raw16, lambda b: np.frombuffer(b, np.float16)),
        ("zlib0", cand_zlib(0), lambda b: zlib.decompress(b)),
        ("zlib1", cand_zlib(1), lambda b: zlib.decompress(b)),
        ("zlib6", cand_zlib(6), lambda b: zlib.decompress(b)),
        ("quant8", cand_quant8, lambda b: np.frombuffer(b, np.uint8)),
    ]
    for name, enc, dec in cands:
        enc_ms, buf = _timed(lambda: enc(x))
        try:
            dec_ms, _ = _timed(lambda: dec(buf), reps=max(5, REPS // 4))
        except Exception:
            dec_ms = float("nan")
        mb = len(buf) / 1048576.0
        rows.append({"codec": name, "enc_ms": round(enc_ms, 3),
                     "dec_ms": round(dec_ms, 3), "mb": round(mb, 4)})
        print("  %-10s %10.2f %10.2f %10.3f %10.1f"
              % (name, enc_ms, dec_ms, mb, mb / (enc_ms / 1000.0)))

    print()
    print("  quantisation error vs fp32 source (max |diff| / relative):")
    for name, enc, dec in cands[:3] + cands[5:]:
        buf = enc(x)
        got = dec(buf).astype(np.float32)
        if name == "quant8":
            arr = x.detach().cpu().numpy().ravel()
            lo, hi = float(arr.min()), float(arr.max())
            got = got / 255.0 * (hi - lo) + lo
        ref = x.detach().cpu().numpy().ravel()
        d = np.abs(got - ref)
        print("    %-8s max=%.5f  mean=%.6f  rel_max=%.2e"
              % (name, d.max(), d.mean(), d.max() / max(abs(ref).max(), 1e-9)))

    path = os.environ.get("M2S_CODEC_OUT")
    if path:
        import json
        with open(path, "w") as fh:
            json.dump({"real": real, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
