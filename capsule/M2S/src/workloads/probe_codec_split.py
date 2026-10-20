"""What does the boundary actually cost, and does compressing pay for itself?

``results/split_runtime_demo.json`` was measured when every boundary payload was
deflated unconditionally, and it reports the split latency overhead as 1.2-2.2x
the monolith. This probe asks where those milliseconds go and whether the codec
is optimising the right thing.

The trap it exposes: serverless bills *execution time*, not bytes on the wire.
Compressing a boundary payload converts free bytes into billed milliseconds.
The exchange rate measured on this runtime is ~1.4 ms per boundary MB, against
18-40 ms per MB for deflating and 6-11 ms per MB for inflating it. So the codec
has to compare the two, which is what ``M2S_WIRE=cost`` does.

Two design notes, both learned by getting them wrong first:

1. **Modes are interleaved, not run in blocks.** The monolith has no boundary at
   all, so the codec cannot affect it; any movement in the monolith baseline
   between modes is pure drift. A block-ordered version produced a "codec
   reclaimed 7%" figure smaller than the drift in the monolith itself.

2. **One runtime per (plan, mode), warmed once, then repeated.** Workers are
   cached per ``LocalFaaSRuntime`` and reused across invokes, so a fresh runtime
   per invocation folds a 2.6 s cold start into the e2e number and makes the
   fission/monolith ratio meaningless -- it measures function count, not the
   boundary. This mirrors ``demo_split.measure``, which is why its numbers are
   comparable with ``split_runtime_demo.json``.

Run:  cd <root> && PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
        HF_HUB_DISABLE_XET=1 .venv/bin/python -u -m workloads.probe_codec_split
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipes import build                      # noqa: E402
from workloads.runtime import LocalFaaSRuntime, compile_plan  # noqa: E402

# label -> env overrides. ``cost100`` is the sensitivity check: a boundary that
# costs 100 ms per MB is one where compressing pays again, and the point is to
# show the policy follows the exchange rate rather than a hardcoded taste. It is
# off by default because it roughly doubles the run.
MODES = (
    ("legacy", {"M2S_WIRE": "legacy"}),
    ("auto", {"M2S_WIRE": "auto"}),
    ("cost", {"M2S_WIRE": "cost"}),
)
EXTRA_MODES = {
    "cost100": {"M2S_WIRE": "cost", "M2S_TRANSFER_MS_PER_MB": "100"},
}

ROUNDS = int(os.environ.get("M2S_ROUNDS", "2"))
WARMUP = int(os.environ.get("M2S_WARMUP", "1"))
REPEAT = int(os.environ.get("M2S_REPEAT", "3"))
TIMEOUT_S = float(os.environ.get("M2S_TIMEOUT_S", "180"))


def monolith(p):
    return [p.names()]


def fission(p):
    return [[n] for n in p.names()]


PLANS = (("monolith", monolith), ("fission", fission))


def apply_env(env):
    for k in ("M2S_WIRE", "M2S_TRANSFER_MS_PER_MB", "M2S_MAX_WIRE_MB"):
        os.environ.pop(k, None)
    os.environ.update(env)


def measure(plan, warmup=WARMUP, repeat=REPEAT):
    """Warm-once-then-repeat, so the reported e2e is the warm path."""
    rt = LocalFaaSRuntime()
    try:
        for _ in range(warmup):
            rt.invoke(plan, timeout_s=TIMEOUT_S)
        runs = [rt.invoke(plan, timeout_s=TIMEOUT_S) for _ in range(repeat)]
    finally:
        rt.shutdown()
    e2e = [r["e2e_ms"] for r in runs]
    idx = sorted(range(len(e2e)), key=lambda i: e2e[i])[len(e2e) // 2]
    rows = runs[idx]["rows"]
    return {
        "e2e_p50_ms": round(statistics.median(e2e), 3),
        "e2e_all": [round(v, 3) for v in e2e],
        "e2e_spread_pct": round(100.0 * (max(e2e) - min(e2e)) / min(e2e), 1),
        "n_functions": runs[idx]["n_functions"],
        "compute_sum_ms": round(sum(r["compute_ms"] for r in rows), 3),
        "encode_sum_ms": round(sum(r["encode_ms"] for r in rows), 3),
        "decode_sum_ms": round(sum(r["decode_ms"] for r in rows), 3),
        "hop_sum_ms": round(sum(r["hop_ms"] for r in rows), 3),
        "wire_total_mb": runs[idx]["wire_total_mb"],
        "cold_start_ms": [r["cold_start_ms"] for r in rows],
        "rows": rows,
    }


def main():
    which = [a for a in sys.argv[1:] if a not in EXTRA_MODES]
    which = which or ["dnn_pipeline", "recsys_deepfm", "pipeline_traffic",
                      "pipeline_driving", "pipeline_video", "pipeline_image"]
    modes = list(MODES)
    for a in sys.argv[1:]:
        if a in EXTRA_MODES:
            modes.append((a, EXTRA_MODES[a]))

    out = {"rounds": ROUNDS, "warmup": WARMUP, "repeat": REPEAT,
           "modes": [m for m, _ in modes],
           "transfer_default_ms_per_mb": 1.4, "pipelines": {}}

    for k in which:
        p = build(k)
        rec = {"n_operators": len(p.ops), "plans": {}}
        print("=" * 104)
        print("%s  (%d operators)" % (k, len(p.ops)))
        print("=" * 104)
        for label, maker in PLANS:
            plan = compile_plan(p, maker(p))
            acc = {m: [] for m, _ in modes}
            for _ in range(ROUNDS):
                for mode, env in modes:
                    apply_env(env)
                    try:
                        acc[mode].append(measure(plan))
                    except Exception as e:
                        print("    [%s/%s] FAILED %s: %s"
                              % (label, mode, type(e).__name__, str(e)[:140]))
            merged = {}
            for mode, _ in modes:
                runs = acc[mode]
                if not runs:
                    continue
                best = sorted(runs, key=lambda r: r["e2e_p50_ms"])
                m = dict(best[len(best) // 2])
                m["e2e_p50_ms"] = round(
                    statistics.median([r["e2e_p50_ms"] for r in runs]), 3)
                merged[mode] = m
                print("  [%-8s %-9s] e2e=%9.2f ms (spread %4.1f%%)  fn=%d  "
                      "compute=%8.2f  enc=%7.2f dec=%7.2f hop=%5.2f  "
                      "wire=%7.4f MB"
                      % (mode, label, m["e2e_p50_ms"], m["e2e_spread_pct"],
                         m["n_functions"], m["compute_sum_ms"],
                         m["encode_sum_ms"], m["decode_sum_ms"],
                         m["hop_sum_ms"], m["wire_total_mb"]))
            rec["plans"][label] = merged
        out["pipelines"][k] = rec
        print()

    out["summary"] = {}
    print("%-20s %-8s %10s %10s %8s %10s %10s" %
          ("pipeline", "mode", "mono ms", "fission", "ratio", "overhead",
           "serde"))
    for k, rec in out["pipelines"].items():
        s = {}
        for mode, _ in modes:
            mo = rec["plans"].get("monolith", {}).get(mode)
            fi = rec["plans"].get("fission", {}).get(mode)
            if not (mo and fi):
                continue
            s[mode] = {
                "monolith_ms": mo["e2e_p50_ms"],
                "fission_ms": fi["e2e_p50_ms"],
                "ratio": round(fi["e2e_p50_ms"] / mo["e2e_p50_ms"], 3),
                "overhead_ms": round(fi["e2e_p50_ms"] - mo["e2e_p50_ms"], 3),
                "fission_serde_ms": round(fi["encode_sum_ms"]
                                          + fi["decode_sum_ms"], 3),
                "fission_wire_mb": fi["wire_total_mb"],
                "fission_hop_ms": fi["hop_sum_ms"],
                "fission_compute_ms": fi["compute_sum_ms"],
                "monolith_compute_ms": mo["compute_sum_ms"],
            }
            print("%-20s %-8s %10.1f %10.1f %7.2fx %10.1f %10.1f"
                  % (k, mode, s[mode]["monolith_ms"], s[mode]["fission_ms"],
                     s[mode]["ratio"], s[mode]["overhead_ms"],
                     s[mode]["fission_serde_ms"]))
        mono = [v["monolith_ms"] for v in s.values()]
        if mono:
            s["_monolith_drift_pct"] = round(
                100.0 * (max(mono) - min(mono)) / min(mono), 1)
            print("%-20s %-8s baseline drift across modes: %.1f%%  "
                  "(monolith has no boundary -- this bounds the noise)"
                  % (k, "", s["_monolith_drift_pct"]))
        out["summary"][k] = s
        print()

    os.makedirs("results", exist_ok=True)
    with open("results/split_codec_ablation.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("saved results/split_codec_ablation.json")


if __name__ == "__main__":
    main()
