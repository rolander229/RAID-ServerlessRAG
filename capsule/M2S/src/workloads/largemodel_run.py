"""Gillis-regime experiment: model size x layout, all measured.

The question this answers
-------------------------
"Your GPU has 32 GB -- why would anyone split the model across functions?"

The answer is that the binding constraint in function-as-a-service is the
*function's* budget, not the accelerator's capacity. This driver makes that
concrete by sweeping two axes against each other:

  model size   resnet18 (45 MB) / resnet50 (98 MB) / resnet152 (241 MB) /
               wide_resnet101_2 (484 MB)
  device       cpu  -- weights live in the function's own process, so an RSS
                       ceiling really does bind (this is ICDCS'21's regime)
               gpu  -- weights live in VRAM, which an RSS ceiling cannot see

and, for each cell, measuring every stage-aligned layout for real: per-function
peak RSS, peak VRAM, cold start, and end-to-end latency. The memory ceiling is
then swept to find the coarsest layout that is still feasible, which is exactly
Gillis's "minimum number of functions subject to the memory constraint".

Run:
    cd <root>
    PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
      .venv/bin/python -u -m workloads.largemodel_run
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads import pipes  # noqa: E402
from workloads.runtime import LocalFaaSRuntime, compile_plan  # noqa: E402

KEY = "largemodel_partition"
OPS = ["large_input_ingest", "large_preprocess", "large_stem_stage",
       "large_mid_stage", "large_late_stage", "large_topk_postprocess",
       "large_result_sink_worker"]

# Stage-aligned layouts: cuts are placed at real model boundaries, because
# cutting anywhere else would not let a function hold "only its own layers".
LAYOUTS = {
    "L1_monolith": [OPS],
    "L2": [OPS[0:3], OPS[3:7]],
    "L3": [OPS[0:3], OPS[3:4], OPS[4:7]],
    "L4": [OPS[0:2], OPS[2:3], OPS[3:4], OPS[4:7]],
    "L5": [OPS[0:2], OPS[2:3], OPS[3:4], OPS[4:5], OPS[5:7]],
    "L7_fission": [[o] for o in OPS],
}

CEILINGS_MB = [512, 768, 1024, 1536, 2048, 3008, 4096, 6144, 10240]


def _set_env(model, device):
    os.environ["M2S_LARGE_MODEL"] = model
    os.environ["M2S_LARGE_DEVICE"] = device


def run_layout(model, device, label, groups, timeout_s=900.0):
    _set_env(model, device)
    p = pipes.build(KEY)
    plan = compile_plan(p, groups, memory_mb=3008, timeout_s=timeout_s)
    rt = LocalFaaSRuntime()
    try:
        # Cold pass: workers do not exist yet, so this invoke carries every
        # function's spawn + first-load cost. Reported separately, because
        # folding a one-off cold start into "steady-state latency" would make
        # every split look artificially bad -- the same measurement error the
        # old project made in the opposite direction.
        t0 = time.perf_counter()
        cold = rt.invoke(plan, timeout_s=timeout_s)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        # Warm pass: all workers are alive, so this is steady-state.
        res = rt.invoke(plan, timeout_s=timeout_s)
        res["e2e_cold_pass_ms"] = round(cold_ms, 2)
        res["e2e_cold_pass_fn_ms"] = round(cold.get("e2e_ms", 0.0), 2)
        res["layout"] = label
        res["n_ops_per_fn"] = [len(g) for g in groups]
        res["max_rss_mb"] = max((r["rss_mb"] for r in res.get("rows", [])),
                                default=0)
        res["max_gpu_mb"] = max((r.get("gpu_mb", 0.0)
                                 for r in res.get("rows", [])), default=0.0)
        res["rss_sum_mb"] = round(sum(r["rss_mb"] for r in res.get("rows", [])),
                                  2)
        res["gpu_sum_mb"] = round(sum(r.get("gpu_mb", 0.0)
                                      for r in res.get("rows", [])), 2)
        res["cold_start_ms"] = [r["cold_start_ms"] for r in res.get("rows", [])]
        res["cold_start_max_ms"] = max(res["cold_start_ms"] or [0.0])
        return res
    finally:
        rt.shutdown()


def per_stage_weights(model):
    from workloads.pipe_d import STAGES, stage_param_mb
    return {s: round(stage_param_mb(s, model), 2) for s in STAGES}


def main():
    models = os.environ.get(
        "M2S_LARGE_MODELS", "resnet18,resnet50,resnet152").split(",")
    devices = os.environ.get("M2S_LARGE_DEVICES", "cpu").split(",")
    only = [x for x in os.environ.get("M2S_LARGE_LAYOUTS", "").split(",") if x]
    layouts = {k: v for k, v in LAYOUTS.items() if not only or k in only}

    from workloads.pipe_d import ensure_shards
    print("building stage shards (downloads real torchvision checkpoints)")
    for m in models:
        ensure_shards(m)

    out = {"models": models, "devices": devices, "ceilings_mb": CEILINGS_MB,
           "layouts": list(layouts), "runs": {}, "weights": {},
           "min_feasible_functions": {}}

    for model in models:
        out["weights"][model] = per_stage_weights(model)
        print("\n%s  stage weights: %s" % (model, out["weights"][model]))

    for device in devices:
        for model in models:
            for label, groups in layouts.items():
                t0 = time.time()
                try:
                    res = run_layout(model, device, label, groups)
                except Exception as e:
                    print("  %-12s %-9s %-12s FAILED %s: %s"
                          % (model, device, label, type(e).__name__,
                             str(e)[:70]))
                    continue
                out["runs"].setdefault(model, {}).setdefault(device, {})[label] = res
                rows = res.get("rows", [])
                print("  %-16s %-3s %-12s fns=%d  warm=%-9.1fms cold=%-9.1fms "
                      "max_rss=%-8.1f sum_rss=%-9.1f max_gpu=%-8.1f "
                      "cold_max=%-7.0fms  (%.0fs)"
                      % (model, device, label, res.get("n_functions", 0),
                         res.get("e2e_ms", 0) or 0,
                         res.get("e2e_cold_pass_ms", 0) or 0,
                         res["max_rss_mb"],
                         res["rss_sum_mb"], res["max_gpu_mb"],
                         res["cold_start_max_ms"], time.time() - t0))
                for r in rows:
                    print("        %-30s %-4s ops=%d rss=%8.1f gpu=%8.1f "
                          "cold=%7.0fms"
                          % (r["function"], r["device"], r["n_ops"],
                             r["rss_mb"], r.get("gpu_mb", 0.0),
                             r["cold_start_ms"]))

    # --- ceiling sweep: coarsest feasible layout per ceiling ----------------
    for model in models:
        for device in devices:
            runs = out["runs"].get(model, {}).get(device, {})
            if not runs:
                continue
            key = "%s/%s" % (model, device)
            table = {}
            for ceiling in CEILINGS_MB:
                best = None
                for label, res in runs.items():
                    rows = res.get("rows", [])
                    if not rows:
                        continue
                    # A layout is feasible only if EVERY function fits.
                    obs_rss = max(r["rss_mb"] for r in rows)
                    obs_gpu = max(r.get("gpu_mb", 0.0) for r in rows)
                    if device == "cpu":
                        ok = obs_rss <= ceiling
                    else:
                        # RSS ceiling still applies to the CPU side, and we
                        # additionally require VRAM to fit the same budget when
                        # a GPU ceiling is declared. Report both.
                        ok = obs_rss <= ceiling and obs_gpu <= ceiling
                    if ok:
                        n = len(rows)
                        if best is None or n < best[1]:
                            best = (label, n, obs_rss, obs_gpu)
                table[str(ceiling)] = (
                    {"layout": best[0], "n_functions": best[1],
                     "max_rss_mb": round(best[2], 1),
                     "max_gpu_mb": round(best[3], 1)} if best else None)
            out["min_feasible_functions"][key] = table
            print("\nceiling sweep %s" % key)
            for c in CEILINGS_MB:
                v = table[str(c)]
                print("   %6d MB -> %s" % (
                    c, ("%s (%d fn)" % (v["layout"], v["n_functions"]))
                    if v else "no feasible layout"))

    os.makedirs("results", exist_ok=True)
    path = os.environ.get("M2S_LARGE_OUT", "results/largemodel_split.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nsaved %s" % path)


if __name__ == "__main__":
    main()
