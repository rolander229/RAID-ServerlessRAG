"""One-way codec cost constants, per MB of *raw* input.

The choice function in ``payload.py`` has to compare two candidate encodings
before committing to one, which needs the encode and decode costs separately --
a round-trip number cannot tell you whether the CPU sits in the producer or the
consumer, and both are billed.

Everything is normalised per MB of raw input, because that is the only quantity
known before encoding. The output size is what the rule is trying to discover.

Run:  cd <root> && PYTHONPATH=M2S/src .venv/bin/python -u \
        -m workloads.probe_codec_calib
"""
import json
import os
import statistics
import sys
import time
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipe_a import batch_tensor, read_clip  # noqa: E402

REPS = int(os.environ.get("M2S_CALIB_REPS", "7"))
MB = 1048576.0


def timeit(fn, reps=REPS):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), out


def bench(name, arr, enc, dec, **kw):
    raw_mb = arr.nbytes / MB
    enc_ms, body = timeit(lambda: enc(arr, kw))
    dec_ms, back = timeit(lambda: dec(body, arr.shape))
    wire_mb = len(body) / MB
    print("%-22s raw=%7.3f MB  wire=%7.3f MB (%5.1f%%)  "
          "enc=%7.2f ms (%6.3f ms/raw-MB)  dec=%7.2f ms (%6.3f ms/raw-MB)"
          % (name, raw_mb, wire_mb, 100 * wire_mb / raw_mb,
             enc_ms, enc_ms / raw_mb, dec_ms, dec_ms / raw_mb))
    return {"raw_mb": round(raw_mb, 4), "wire_mb": round(wire_mb, 4),
            "enc_ms": round(enc_ms, 3), "dec_ms": round(dec_ms, 3),
            "enc_ms_per_raw_mb": round(enc_ms / raw_mb, 4),
            "dec_ms_per_raw_mb": round(dec_ms / raw_mb, 4)}


def jpeg_enc(a, kw):
    ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes()


def jpeg_dec(b, shape):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def png_enc(a, kw):
    ok, buf = cv2.imencode(".png", a)
    return buf.tobytes()


def png_dec(b, shape):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


def fp16_raw_enc(a, kw):
    return np.ascontiguousarray(a.astype(np.float16)).tobytes()


def fp16_raw_dec(b, shape):
    return np.frombuffer(b, np.float16).astype(np.float32).reshape(shape)


def fp16z_enc(a, kw):
    return zlib.compress(np.ascontiguousarray(a.astype(np.float16)).tobytes(), 1)


def fp16z_dec(b, shape):
    return np.frombuffer(zlib.decompress(b), np.float16).astype(np.float32) \
        .reshape(shape)


def u8_raw_enc(a, kw):
    return np.ascontiguousarray(a).tobytes()


def u8_raw_dec(b, shape):
    return np.frombuffer(b, np.uint8).reshape(shape)


def main():
    out = {}
    print("=" * 118)
    print("images (uint8), from the real clip the video pipeline decodes")
    print("=" * 118)
    frames = read_clip()
    if frames:
        f = frames[0]
        print("frame shape:", f.shape, f.dtype)
        out["frame"] = {}
        for nm, e, d in (("jpeg", jpeg_enc, jpeg_dec),
                         ("png", png_enc, png_dec),
                         ("u8_raw", u8_raw_enc, u8_raw_dec)):
            out["frame"][nm] = bench("frame/" + nm, f, e, d)
    else:
        print("(no frames available)")

    print()
    print("=" * 118)
    print("float tensors (the dnn / vit boundary payloads)")
    print("=" * 118)
    x = batch_tensor(n=32).detach().cpu().numpy()
    out["batch32"] = {}
    for nm, e, d in (("fp16_raw", fp16_raw_enc, fp16_raw_dec),
                     ("fp16z", fp16z_enc, fp16z_dec)):
        out["batch32"][nm] = bench("batch32/" + nm, x, e, d)

    print()
    print("=" * 118)
    print("exchange rate check: transfer is measured at ~1.4 ms/MB, so a codec")
    print("pays only if (raw-wire) x 1.4 exceeds its per-raw-MB CPU")
    print("=" * 118)
    RATE = float(os.environ.get("M2S_TRANSFER_MS_PER_MB", "1.4"))
    for grp, rows in out.items():
        if not rows:
            continue
        base = rows.get("u8_raw") or rows.get("fp16_raw")
        print("[%s]  baseline %s: %.3f ms/raw-MB" %
              (grp, "u8_raw" if "u8_raw" in rows else "fp16_raw",
               base["enc_ms_per_raw_mb"] + base["dec_ms_per_raw_mb"]))
        for nm, r in rows.items():
            if nm == "u8_raw" or nm == "fp16_raw":
                continue
            d_cpu = (r["enc_ms_per_raw_mb"] + r["dec_ms_per_raw_mb"]) \
                - (base["enc_ms_per_raw_mb"] + base["dec_ms_per_raw_mb"])
            saved_mb_per_raw_mb = (r["raw_mb"] - r["wire_mb"]) / r["raw_mb"]
            saved_ms = saved_mb_per_raw_mb * RATE
            print("    %-9s saves %.3f ms transfer per raw-MB for %+.3f ms CPU"
                  "  -> %s (net %+.3f ms/MB)"
                  % (nm, saved_ms, d_cpu,
                     "PAYS" if saved_ms > d_cpu else "loses",
                     saved_ms - d_cpu))

    with open("results/codec_calibration.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nsaved results/codec_calibration.json")


if __name__ == "__main__":
    main()
