"""Two ablations that decide the whole cost story.

Both are implementation properties, not laws of the model, so they have to be
measured separately from the layout sweep -- otherwise the sweep silently
reports "splitting does not pay" when the real statement is "this *loader* and
this *codec* do not pay".

Ablation 1: how a shard is loaded
---------------------------------
``stage_module`` builds the stage on the meta device and then calls
``to_empty`` before ``load_state_dict``. ``to_empty`` materialises 144 MB of
parameters, and ``load_state_dict`` (which copies by default) then needs the
shard's own 144 MB alive at the same time. Peak = 2x the shard. On a 2-stage
group that double allocation was what pushed the function 18 MB over the
1024 MB billing tier -- i.e. the tier, and therefore the entire cost verdict,
was set by the loader rather than by the model.

Three variants, same shards, same stage group:
    copy    : to_empty + load_state_dict(copy)          [current]
    assign  : meta + load_state_dict(assign=True)       [no double allocation]
    mmap    : meta + torch.load(mmap=True) + assign     [plus no read buffer]

Ablation 2: what the boundary encode actually costs
---------------------------------------------------
The sweep measured ~58 ms to encode one (4,197,1024) tensor, far more than
converting to fp16 and zlib-compressing 1.6 MB should take. The suspicion is
OpenMP spin-wait: torch's worker threads busy-wait between kernels and steal
CPU from the single-threaded numpy/zlib path. If so the "encode cost" is a
thread-configuration artefact, not a property of the wire format.

Usage:
    python -m workloads.probe_vit_load [--stages 3,4] [--json PATH]
    python -m workloads.probe_vit_load --child <variant> --stages 3,4
    python -m workloads.probe_vit_load --encode-bench
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))


def _vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _load_variant(m, idx, variant):
    """Load stage ``idx``'s shard into ``m`` using one of the three strategies."""
    import torch
    from workloads import pipe_e

    m = pipe_e._carve(pipe_e._full_meta(), idx)
    path = pipe_e.shard_paths()[idx]

    if variant == "copy":
        m = m.to_empty(device="cpu")
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        del sd
    elif variant == "assign":
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd, assign=True)
        del sd
    elif variant == "mmap":
        sd = torch.load(path, map_location="cpu", mmap=True)
        m.load_state_dict(sd, assign=True)
        del sd
    else:
        raise ValueError(variant)
    return m.eval()


def child_load(variant, stages):
    """Fresh process: import floor -> load the given stages -> one forward."""
    import torch
    from workloads import pipe_e

    floor = _vmhwm_mb()
    ms = []
    for i in stages:
        ms.append(_load_variant(None, i, variant))
    after_load = _vmhwm_mb()

    x = torch.randn(pipe_e.BATCH, pipe_e.SEQ, pipe_e.HIDDEN)
    t0 = time.perf_counter()
    with torch.no_grad():
        for m in ms:
            x = m(x)
    fwd_ms = (time.perf_counter() - t0) * 1000.0
    after_fwd = _vmhwm_mb()

    print(json.dumps({
        "variant": variant,
        "stages": stages,
        "n_stages": len(stages),
        "import_floor_mb": round(floor, 2),
        "after_load_mb": round(after_load, 2),
        "after_fwd_mb": round(after_fwd, 2),
        "fwd_ms": round(fwd_ms, 1),
    }))


def run_child(variant, stages):
    py = sys.executable
    mod = "workloads.probe_vit_load"
    p = subprocess.run([py, "-u", "-m", mod, "--child", variant,
                        "--stages", ",".join(str(s) for s in stages)],
                       cwd=os.path.join(HERE, ".."),
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("child rc=%d\n%s\n%s"
                           % (p.returncode, p.stdout[-1500:], p.stderr[-1500:]))
    line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def encode_bench():
    """Encode cost vs thread configuration, on the real payload shape."""
    import torch
    from workloads import payload as P
    from workloads import pipe_e

    x = torch.randn(pipe_e.BATCH, pipe_e.SEQ, pipe_e.HIDDEN)
    buf = P.encode(x)  # warm any lazily-created buffers

    print("encode of a %s tensor" % (tuple(x.shape),))
    print("  OMP_NUM_THREADS=%s OMP_WAIT_POLICY=%s torch threads=%d"
          % (os.environ.get("OMP_NUM_THREADS", "-"),
             os.environ.get("OMP_WAIT_POLICY", "-"), torch.get_num_threads()))

    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        buf = P.encode(x)
    enc = (time.perf_counter() - t0) / reps * 1000.0
    t0 = time.perf_counter()
    for _ in range(reps):
        P.decode(buf)
    dec = (time.perf_counter() - t0) / reps * 1000.0

    # the two halves of the codec, isolated
    t0 = time.perf_counter()
    for _ in range(reps):
        arr = x.detach().cpu().numpy()
        a16 = arr.astype("float16")
    cast = (time.perf_counter() - t0) / reps * 1000.0
    import zlib
    raw = a16.tobytes() if a16.flags["C_CONTIGUOUS"] else a16.tobytes()
    t0 = time.perf_counter()
    for _ in range(reps):
        zlib.compress(raw, 1)
    zc = (time.perf_counter() - t0) / reps * 1000.0

    print("  wire size        : %.3f MB (%s)"
          % (len(buf) / 1048576.0, P.kind_of(buf)))
    print("  full encode      : %7.2f ms" % enc)
    print("  full decode      : %7.2f ms" % dec)
    print("    of which cast  : %7.2f ms   (numpy asarray + fp16 cast)" % cast)
    print("    of which zlib  : %7.2f ms   (level-1 on 1.6 MB)" % zc)
    print("  unaccounted      : %7.2f ms" % (enc - cast - zc))
    return {"encode_ms": round(enc, 3), "decode_ms": round(dec, 3),
            "cast_ms": round(cast, 3), "zlib_ms": round(zc, 3),
            "wire_mb": round(len(buf) / 1048576.0, 4)}


def main():
    stages = [3, 4]
    for i, a in enumerate(sys.argv):
        if a == "--stages":
            stages = [int(s) for s in sys.argv[i + 1].split(",")]

    if "--encode-bench" in sys.argv:
        res = {"encode_bench": encode_bench()}
        path = os.environ.get("M2S_VIT_LOAD_OUT")
        if path:
            with open(path, "w") as fh:
                json.dump(res, fh, indent=2)
        return 0

    print("shard-load ablation, stage group %s (%d x %.2f MB params)"
          % (stages, len(stages), __import__("workloads.pipe_e",
                                             fromlist=["pipe_e"]).stage_param_mb(0) / 3))
    print()
    print("  %-8s %8s %10s %10s %10s %9s"
          % ("variant", "stages", "floor", "after_load", "after_fwd", "fwd_ms"))
    rows = []
    for v in ("copy", "assign", "mmap"):
        r = run_child(v, stages)
        rows.append(r)
        print("  %-8s %8d %10.1f %10.1f %10.1f %9.1f"
              % (v, r["n_stages"], r["import_floor_mb"], r["after_load_mb"],
                 r["after_fwd_mb"], r["fwd_ms"]))

    base = [r for r in rows if r["variant"] == "copy"][0]
    print()
    for r in rows:
        print("  %-8s peak - copy = %+8.1f MB" % (r["variant"],
                                                   r["after_fwd_mb"] - base["after_fwd_mb"]))

    path = os.environ.get("M2S_VIT_LOAD_OUT", "results/vit_load_ablation.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"stages": stages, "rows": rows}, fh, indent=2)
    print("\nsaved %s" % path)
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        j = sys.argv.index("--stages") if "--stages" in sys.argv else None
        child_load(sys.argv[i + 1],
                   [int(s) for s in sys.argv[j + 1].split(",")] if j else [3, 4])
    else:
        sys.exit(main())
