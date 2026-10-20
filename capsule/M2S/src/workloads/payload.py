"""Wire-format payload codec for cross-function boundaries.

Two things the old optimizer got wrong, both fixed here:

1. It charged ``tensor.nbytes``. Real boundaries carry *encoded* bytes: JPEG
   for images, quantised/compressed buffers for feature maps.

2. Serialisation is not type-neutral. A naive pickle/numpy round trip turns a
   ``torch.Tensor`` into an ``ndarray`` and downstream torch operators then
   fail with e.g. "expected Tensor as element 1, but got numpy.float32". A
   partition can only be *executed* if the wire format restores types, so the
   codec here is recursive and type-preserving.

Framing:  len(header):4 | header(pickle) | body      header = {kind, meta}
"""
import os
import pickle
import time
import zlib

import cv2
import numpy as np
import torch

JPEG_Q = 85

# ---------------------------------------------------------------------------
# The codec is an optimiser, not a priority list.
#
# Serverless bills execution time, not bytes on the wire. A boundary payload is
# moved by the platform for free as far as the invoice is concerned, while every
# millisecond of encode/decode is billed -- once in the producer, once in the
# consumer. "Compress it, it got smaller" therefore converts free bytes into
# billed CPU.
#
# The rule implemented here is
#
#     pick the encoding that minimises
#         encode_ms + decode_ms + wire_mb x TRANSFER_MS_PER_MB
#
# using the one-way per-MB costs measured in results/codec_calibration.json
# (ms per MB of raw input; produced by workloads/probe_codec_calib.py):
#
#     u8_raw    0.041 enc / 0.002 dec     wire = 100%  of raw
#     jpeg      1.609 enc / 2.033 dec     wire =  10.8% of raw
#     png      15.603 enc / 7.150 dec     wire =  66.9% of raw
#     fp16      1.101 enc / 0.524 dec     wire =  50%   of raw (fp32 in)
#     fp16z     9.160 enc / 2.895 dec     wire =  26.4% of raw
#
# At the rate measured on this runtime -- 1.4 ms per boundary MB, from
# dnn_pipeline's 7.01 ms of hop for a 4.887 MB spool round trip -- every
# compressed form loses: jpeg by 2.4 ms per raw-MB, fp16z by 9.4 ms per raw-MB.
# fp16 itself is the one exception and is kept: it halves the bytes for about a
# tenth of the CPU deflating costs.
#
# Two knobs keep the policy honest, because the answer genuinely depends on the
# deployment:
#
#   M2S_TRANSFER_MS_PER_MB  cost of moving one MB across the boundary. tmpfs is
#                           1.4; a 20 MB/s link is ~50; by 50 the codec starts
#                           choosing jpeg again.
#   M2S_MAX_WIRE_MB         hard cap on a boundary payload, 0 = uncapped. Real
#                           FaaS has one -- AWS Lambda rejects synchronous
#                           invocations over 6 MB -- and a payload that cannot
#                           be sent at all must be compressed whatever the
#                           arithmetic says.
#
# ``M2S_WIRE`` selects the policy:
#     cost   (default) the rule above
#     auto   compress whenever the bytes shrink, ignoring what the CPU costs;
#            this is what produced results/split_runtime_demo.json
#     fast   never deflate; keep the fp16 framing
#     legacy always deflate, the behaviour before this file grew a model
#     raw32  no quantisation either; fp32, most accurate, 2x the bytes
# ---------------------------------------------------------------------------
_PROBE_BYTES = int(os.environ.get("M2S_WIRE_PROBE_BYTES", str(1 << 16)))
_MIN_GAIN = float(os.environ.get("M2S_WIRE_MIN_GAIN", "0.90"))
# Both knobs are read per call rather than frozen at import: a worker is a fresh
# process so import-time would work, but tests and in-process ablations need to
# be able to move them, and a constant that only responds to the environment
# once is a trap.
TRANSFER_DEFAULT_MS_PER_MB = 1.4
MAX_WIRE_DEFAULT_MB = 0.0

_MB = 1048576.0

# Measured, not guessed -- see workloads/probe_codec_calib.py.
_ENC_MS_PER_RAW_MB = {"u8": 0.35, "jpeg": 1.609, "png": 15.603, "fp16": 1.101}
_DEC_MS_PER_RAW_MB = {"u8": 0.002, "jpeg": 2.033, "png": 7.150, "fp16": 0.524}
# zlib is priced against the buffer it actually touches: deflate reads the fp16
# bytes, inflate reads the compressed bytes, and neither scales with raw size.
_ZLIB_ENC_MS_PER_IN_MB = 18.32
_ZLIB_DEC_MS_PER_WIRE_MB = 10.98
# Framing is not free and it scales with the *encoded* size, not the raw size:
# the body map is pickled on the producer and unpickled on the consumer, and a
# raw payload pays that over every byte it ships. Measured at 4.4 ms/MB to dump
# and 1.3 ms/MB to load on a 21 MB list of raw frames. Leaving this out makes
# the codec underprice raw by an order of magnitude and wrongly prefer it.
_FRAME_MS_PER_WIRE_MB = 5.7
# Below this the absolute stakes are smaller than the cost of deciding, so the
# cheaper-CPU option (raw) wins by default rather than by measurement.
_IMAGE_DECIDE_MIN_MB = 0.25

# One probe per distinct payload shape, not one per element. A boundary that
# carries a list of N frames is N encodings of the same kind; without this the
# codec pays N jpeg encodes to reach the same answer N times, and then throws
# all N away if the answer is "ship it raw".
_IMAGE_CACHE: dict = {}

# Set for the duration of one encode() when the payload as a whole cannot be
# sent uncompressed. The cap is a property of the payload, not of an element:
# 32 frames of 0.66 MB each are individually well under a 6 MB limit and
# collectively four times over it.
_FORCE_PACKED = False


def transfer_ms_per_mb():
    return float(os.environ.get("M2S_TRANSFER_MS_PER_MB",
                                str(TRANSFER_DEFAULT_MS_PER_MB)))


def max_wire_mb():
    return float(os.environ.get("M2S_MAX_WIRE_MB", str(MAX_WIRE_DEFAULT_MB)))


def wire_mode():
    return os.environ.get("M2S_WIRE", "cost").strip().lower()


def _total_ms(wire_mb, cpu_ms):
    """Billed CPU plus the transfer it buys. The thing being minimised.

    Framing is charged here rather than inside each codec's constant because it
    tracks the encoded size for every codec alike.
    """
    rate = transfer_ms_per_mb()
    return cpu_ms + _FRAME_MS_PER_WIRE_MB * wire_mb + wire_mb * rate


def _fits(wire_mb):
    cap = max_wire_mb()
    return cap <= 0.0 or wire_mb <= cap


def _compress_ratio(raw):
    """Compressed fraction of ``raw``, probed on windows spread across it.

    A single prefix is not representative. Feature batches and frame stacks are
    not stationary -- on this pipeline's 18.375 MB batch tensor the leading
    64 KB compresses to 0.61 while the whole buffer reaches 0.53, an error of
    18% in the direction that hides a real saving. The probe budget is spread
    over several windows instead, which costs the same and is far less biased.
    """
    n = len(raw)
    if n <= 0:
        return 1.0
    windows = max(int(os.environ.get("M2S_WIRE_PROBE_WINDOWS", "4")), 1)
    w = min(max(_PROBE_BYTES // windows, 4096), n)
    if windows == 1 or n <= w:
        return len(zlib.compress(raw[:w], 1)) / float(w)
    stride = (n - w) / float(windows - 1)
    packed = 0
    seen = 0
    for i in range(windows):
        off = min(int(i * stride), n - w)
        chunk = raw[off:off + w]
        packed += len(zlib.compress(chunk, 1))
        seen += len(chunk)
    return packed / float(seen)


def _deflate_pays(raw):
    """Is deflating ``raw`` worth it under the active policy?

    ``raw`` is the buffer deflate would read (the fp16 bytes). Only the parts
    that differ between the two options are counted: the fp16 conversion and its
    decode happen either way and cancel.
    """
    ratio = _compress_ratio(raw)
    if wire_mode() != "cost":
        return ratio <= _MIN_GAIN
    in_mb = len(raw) / _MB
    wire_mb = ratio * in_mb
    if _FORCE_PACKED or not _fits(wire_mb):
        return True
    saved = (in_mb - wire_mb) * transfer_ms_per_mb()
    extra_cpu = (_ZLIB_ENC_MS_PER_IN_MB * in_mb
                 + _ZLIB_DEC_MS_PER_WIRE_MB * wire_mb
                 + _FRAME_MS_PER_WIRE_MB * (wire_mb - in_mb))
    return saved > extra_cpu


def _choose_image(arr):
    """uint8 image -> (kind, body). Encodes once, then keeps or discards it.

    Both candidates are known values by the time the decision is made: the raw
    bytes are a memcpy, and the encoded body is exactly what the packed option
    would ship. Discarding a jpeg we already produced is wasted work, but it is
    bounded and it is the only way to learn the compressed size at all.
    """
    raw_mb = arr.nbytes / _MB
    raw = np.ascontiguousarray(arr).tobytes()
    mode = wire_mode()
    if arr.ndim == 3 and arr.shape[2] == 3:
        kind, ext, flags = "jpeg", ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q]
    else:
        kind, ext, flags = "png", ".png", []
    key = (arr.shape, arr.dtype.str, mode,
           transfer_ms_per_mb(), max_wire_mb(), _FORCE_PACKED)
    verdict = _IMAGE_CACHE.get(key)
    if verdict == "u8":
        return "u8", raw
    if mode == "cost" and not _FORCE_PACKED and verdict is None \
            and raw_mb < _IMAGE_DECIDE_MIN_MB:
        _IMAGE_CACHE[key] = "u8"
        return "u8", raw
    t0 = time.perf_counter()
    ok, buf = cv2.imencode(ext, arr, flags)
    enc_ms = (time.perf_counter() - t0) * 1000.0
    if not ok:
        _IMAGE_CACHE[key] = "u8"
        return "u8", raw
    body = buf.tobytes()
    if _FORCE_PACKED:
        _IMAGE_CACHE[key] = "packed"
        verdict = "packed"
    elif mode == "cost" and verdict is None:
        wire_mb = len(body) / _MB
        packed = _total_ms(wire_mb, enc_ms + _DEC_MS_PER_RAW_MB[kind] * raw_mb)
        # The probe is already paid for, so charge it to the option that
        # throws it away. Otherwise the two are compared as if deciding were
        # free, and marginal payloads flip on noise.
        plain = _total_ms(raw_mb, enc_ms
                          + (_ENC_MS_PER_RAW_MB["u8"]
                             + _DEC_MS_PER_RAW_MB["u8"]) * raw_mb)
        _IMAGE_CACHE[key] = "packed" if packed <= plain else "u8"
        verdict = _IMAGE_CACHE[key]
    if verdict == "u8":
        return "u8", raw
    return kind, body


def _dump_array(arr):
    """numpy array -> (kind, meta, body)"""
    shape = tuple(int(v) for v in arr.shape)
    mode = wire_mode()
    if arr.dtype == np.uint8:
        kind, body = _choose_image(arr)
        return kind, {"shape": shape}, body

    # Two-dimensional float maps (feature maps, spectra) have a spatial
    # structure that quantising to 8 bits exploits well; the probe below is for
    # the high-dimensional case, where that structure is absent. Under ``cost``
    # this path is skipped outright: it is a lossy format change, and PNG
    # encoding is exactly the kind of CPU-for-bytes trade ``cost`` exists to
    # refuse.
    if mode in ("auto", "legacy") and arr.ndim == 2:
        flat = arr.astype(np.float32).ravel()
        if flat.size:
            lo = float(np.nanmin(flat))
            hi = float(np.nanmax(flat))
            if hi > lo:
                q = ((flat - lo) / (hi - lo) * 255.0).astype(np.uint8)
                ok, buf = cv2.imencode(".png", q.reshape(arr.shape))
                if ok:
                    return ("quant8", {"shape": shape, "lo": lo, "hi": hi},
                            buf.tobytes())

    if mode == "raw32":
        return "fp32", {"shape": shape}, np.ascontiguousarray(
            arr, dtype=np.float32).tobytes()

    raw = arr.astype(np.float16)
    raw = np.ascontiguousarray(raw).tobytes()
    if mode == "fast":
        return "fp16", {"shape": shape}, raw
    if mode in ("auto", "cost") and not _deflate_pays(raw):
        return "fp16", {"shape": shape}, raw
    return "fp16z", {"shape": shape}, zlib.compress(raw, 1)


def _load_array(kind, meta, body):
    shape = meta.get("shape")
    if kind == "jpeg":
        return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if kind == "png":
        return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    if kind == "u8":
        return np.frombuffer(body, np.uint8).reshape(shape).copy()
    if kind == "quant8":
        q = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
        return q.astype(np.float32) / 255.0 * (meta["hi"] - meta["lo"]) \
            + meta["lo"]
    if kind in ("fp16", "fp16z"):
        raw = body if kind == "fp16" else zlib.decompress(body)
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(shape)
    if kind == "fp32":
        return np.frombuffer(body, np.float32).reshape(shape)
    raise ValueError("unknown array kind %r" % kind)


def _dump(obj):
    """-> (kind, meta, body); recursive, preserves tuple/list/dict shape."""
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        kind, meta, body = _dump_array(arr)
        meta = dict(meta)
        meta.update({"tensor": True, "dtype": str(obj.dtype),
                     "cuda": bool(obj.is_cuda)})
        return kind, meta, body
    if isinstance(obj, np.ndarray):
        return _dump_array(obj)
    if isinstance(obj, (list, tuple, dict)):
        payload = {k: _dump(v) for k, v in
                   (obj.items() if isinstance(obj, dict) else enumerate(obj))}
        tag = "list" if isinstance(obj, list) else (
            "dict" if isinstance(obj, dict) else "tuple")
        return tag, {}, pickle.dumps(payload, 2)
    return "pickle", {}, pickle.dumps(obj, 2)


ARRAY_KINDS = ("jpeg", "png", "u8", "quant8", "fp16", "fp16z", "fp32")


def _load(kind, meta, body):
    if kind in ARRAY_KINDS:
        arr = _load_array(kind, meta, body)
        if not meta.get("tensor"):
            return arr
        t = torch.from_numpy(np.ascontiguousarray(arr))
        t = t.to(getattr(torch, meta["dtype"].split(".")[-1]))
        if meta.get("cuda") and torch.cuda.is_available():
            t = t.cuda()
        return t
    if kind == "pickle":
        return pickle.loads(body)
    items = pickle.loads(body)
    vals = {k: _load(*v) for k, v in items.items()}
    if kind == "list":
        return [vals[i] for i in range(len(vals))]
    if kind == "tuple":
        return tuple(vals[i] for i in range(len(vals)))
    return vals


def _raw_mb(obj) -> float:
    """Total uncompressed size of a payload, recursively, in MB."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement() / _MB
    if isinstance(obj, np.ndarray):
        return obj.nbytes / _MB
    if isinstance(obj, (list, tuple)):
        return sum(_raw_mb(v) for v in obj)
    if isinstance(obj, dict):
        return sum(_raw_mb(v) for v in obj.values())
    return 0.0


def encode(obj):
    global _FORCE_PACKED
    prev = _FORCE_PACKED
    cap = max_wire_mb()
    try:
        if cap > 0.0:
            _FORCE_PACKED = _raw_mb(obj) > cap
        kind, meta, body = _dump(obj)
        head = pickle.dumps({"kind": kind, "meta": meta}, 2)
        return len(head).to_bytes(4, "little") + head + body
    finally:
        _FORCE_PACKED = prev


def decode(buf):
    n = int.from_bytes(buf[:4], "little")
    head = pickle.loads(buf[4:4 + n])
    return _load(head["kind"], head["meta"], buf[4 + n:])


def kind_of(buf):
    n = int.from_bytes(buf[:4], "little")
    return pickle.loads(buf[4:4 + n])["kind"]


def raw_bytes(obj):
    """What the OLD optimizer charged for. Kept for the diff report only."""
    if isinstance(obj, torch.Tensor):
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    try:
        return len(pickle.dumps(obj, 2))
    except Exception:
        return 0


def metered(obj):
    """-> dict(raw_mb, wire_mb, kind, serde_ms, roundtrip_ok)"""
    import time
    t0 = time.perf_counter()
    buf = encode(obj)
    back = decode(buf)
    serde_ms = (time.perf_counter() - t0) * 1000.0
    ok = type(back) is type(obj)
    if isinstance(obj, torch.Tensor):
        ok = ok and back.shape == obj.shape and back.dtype == obj.dtype
    return {
        "raw_mb": round(raw_bytes(obj) / 1048576.0, 4),
        "wire_mb": round(len(buf) / 1048576.0, 4),
        "kind": kind_of(buf),
        "serde_ms": round(serde_ms, 4),
        "roundtrip_ok": bool(ok),
    }
