#!/usr/bin/env python3
"""Experiment 2: the three-level service profile, and how well it predicts itself.

This is Sec. II-B end to end: measure every layer and operator of every
benchmark service at several request sizes, consolidate the operators into
layers (Eq. 1-3) into the per-layer vectors of Eq. (4), then fit the paper's
``<X, s, p> -> <m, t>`` predictors on those rows and score them by RMSLE. What
lands in docs/results.md is two tables:

  Table I   layer memory RMSLE for LR / XGBoost / RF, against 0.156 / 0.105 / 0.139
  Fig. 5a   per-operator-family memory RMSLE, against Conv2D 0.0319, MatMul 0.0427
  Fig. 5b   the same for duration, where the paper reports >95% accuracy

Each memory table appears twice, because which of the two is a result depends on
what is being predicted: the allocator's own per-layer and per-operator high-water
reading is a measurement and can be learned, while the Eq. (1)-(3) byte model is a
formula in the request size and a linear fit reproduces it exactly. The paper
reports one number per method and does not say which of the two it measured, so
both are printed and the comparison is drawn against the measured column.

Three things this also checks, because a profiler that agrees with nothing is
just a slow tape measure:

  agreement   the consolidated layer footprint against the CUDA allocator's own
              high-water mark, per layer -- Eq. (1) is a model, peak is a measurement
  accounting  operator times inside a layer against the layer's own span, and the
              layer vector sum against an uninstrumented forward
  coverage    dominant operators per service against the "< 15 per DLIS" claim

Run on v100:  python experiments/e2_service_profiling.py \\
                  --models resnet50 --device cuda --batches 1 2 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph, move_tensors  # noqa: E402
from slicer.profile import regress  # noqa: E402
from slicer.profile.service import profile_model, to_dict  # noqa: E402

# A (model, batch) row is skipped when its parameters alone would exceed this
# much of the 251GB box; the request adds activations on top.
PARAM_BUDGET_GB = 60.0


def sweep(key: str, device: str, batches: list[int], warmup: int, repeats: int,
          budget: float) -> tuple[list[dict], list, list[str]]:
    """Profile one service at each request size, reusing the built module."""
    spec = zoo.get(key)
    model, _ = zoo.load(key, device=device)
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    rows, profiles, notes = [], [], []

    for batch in batches:
        if params * batch > budget * 10 ** 9:
            notes.append("%s b=%d skipped: %.1fGB of parameters"
                         % (key, batch, params * batch / 10 ** 9))
            continue
        started = time.perf_counter()
        inputs = move_tensors(spec.inputs(batch), device)
        graph = build_structure_graph(model, inputs, name=key, device=device)
        profile = profile_model(model, inputs, key=key, device=device, graph=graph,
                                warmup=warmup, repeats=repeats)
        rows.append(_row(profile, time.perf_counter() - started))
        profiles.append(profile)

    del model
    return rows, profiles, notes


def _row(profile, seconds: float) -> dict:
    """One line of the summary table: what was measured and whether it adds up.

    ``time_agreement`` compares the calibrated sum of layer spans with an
    uninstrumented forward, and ``raw_ms`` is the same sum before the measured
    instrumentation cost came off -- the gap on a GPU is several fold, since every
    operator boundary pays a dispatch and an event pair. Neither is Eq. (4)'s T:
    the vector is scaled onto the forward, and ``timing_factor`` is the scale.
    ``peak_ratio`` compares the CUDA allocator's high-water mark with the
    Eq. (1)-(3) footprint of the same layer, which is the only place the
    consolidation rules can be falsified.
    """
    memory, times = profile.vectors()
    return {
        **to_dict(profile),
        "sum_M_mb": round(sum(memory) / 2 ** 20, 2),
        "max_M_mb": round(max(memory, default=0.0) / 2 ** 20, 2),
        "sum_T_ms": round(sum(times) * 1e3, 3),
        "operator_ms": round(profile.operator_ns / 1e6, 3),
        # Eq. (4)'s T is the operator rows summed; the layer spans are the
        # independent reading of the same interval, and the raw column is what
        # they were before the measured instrumentation cost came off.
        "op_over_layer": round(profile.operator_ns / profile.layer_ns, 3)
        if profile.layer_ns else None,
        "layer_ms": round(profile.layer_ns / 1e6, 3),
        "raw_ms": round(profile.raw_layer_ns / 1e6, 3),
        "time_agreement": round(profile.layer_ns / profile.forward_ns, 3)
        if profile.forward_ns else None,
        "peak_ratio": round(profile.peak_ratio, 3),
        # The subtracted constants, so a reader can see how much of the raw sum
        # was the profiler rather than the service, and how far they drifted.
        "calib_layer_us": round(profile.calibration[0] / 1e3, 2),
        "calib_op_us": round(profile.calibration[1] / 1e3, 2),
        "seconds": round(seconds, 1),
    }


def write_samples(path: str, profiles: list) -> int:
    fieldnames = ["model", "device", "batch", "size", "layer", "op", "family",
                  "params", "in_bytes", "memory", "peak", "ns"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        count = 0
        for profile in profiles:
            for row in profile.operators:
                writer.writerow([row.model, row.device, row.batch, row.size, row.layer,
                                 row.op, row.family, row.params, row.in_bytes,
                                 row.memory, row.peak, row.ns])
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=None,
                        help="keys from slicer/models/zoo.py, default all")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int,
                        default=[1, 2, 3, 4, 6, 8],
                        help="at least four sizes, or per-operator groups have "
                             "too few points to hold any out")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=float, default=PARAM_BUDGET_GB)
    parser.add_argument("--out", default="runs/e2_service_profiling.json")
    args = parser.parse_args()

    keys = args.models or sorted(zoo.SPECS)
    rows, profiles, notes, failures = [], [], [], []
    for key in keys:
        try:
            swept, profiled, skipped = sweep(key, args.device, args.batches,
                                             args.warmup, args.repeats, args.budget)
        except Exception as error:
            failures.append("%s: %s: %s" % (key, type(error).__name__, str(error)[:200]))
            print("%-22s FAIL %s" % (key, failures[-1]), flush=True)
            continue
        notes += skipped
        for row in swept:
            rows.append(row)
            print("%-22s b=%-3d nodes=%4d max_M=%7.1fMB fwd_peak=%8.1fMB fwd=%7.2fms "
                  "spans=%7.2fms (raw %7.2f) op/spans=%5.2f /fwd=%5.2f scale=%5.2f "
                  "cal=%5.2f+%.2fus peak/eq1=%-6s dom=%-4d billed=%-4d %4.1fs" % (
                      row["paper"], row["batch"], row["nodes"], row["max_M_mb"],
                      row["peak_mb"] or 0.0, row["forward_ms"], row["layer_ms"],
                      row["raw_ms"], row["op_over_layer"], row["time_agreement"] or 0.0,
                      row["timing_factor"],
                      row["calib_layer_us"], row["calib_op_us"],
                      row["peak_ratio"], row["dominant_ops"], row["stray_ops"],
                      row["seconds"]), flush=True)
        profiles += profiled

    scored = regress.evaluate(profiles)
    # Named after --out so a second sweep does not overwrite the first one's
    # training rows: the operator CSV is the only artifact the regression reads
    # back, and a labelled run is how a fixed extractor gets an A/B.
    samples_path = os.path.join(os.path.dirname(args.out) or ".",
                                os.path.splitext(os.path.basename(args.out))[0]
                                + "_operator_samples.csv")
    samples = write_samples(samples_path, profiles)
    result = {
        "device": args.device,
        "batches": args.batches,
        "profiles": rows,
        "prediction": scored,
        "operator_samples": samples,
        "skipped": notes,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)

    _report(scored)
    print("\nwrote %s and %s (%d operator rows, %d profiles, %d failures)" % (
        args.out, samples_path, samples, len(rows), len(failures)))
    return 1 if failures else 0


def _report(scored: dict) -> None:
    """Print the two tables the run exists to fill."""
    print("\n== Table I: layer memory footprint, one predictor per (service, layer)")
    print("   paper: LR 0.156  XGBoost 0.105  RF 0.139")
    layer = scored["layer"]
    print("   target: the allocator's own reading (the paper's m)")
    _kinds(layer["memory_measured"], "rows split (paper's)")
    _kinds(layer["memory_measured_size"], "sizes held out")
    _kinds(layer["memory_measured_cross"], "layers held out")
    model = layer["byte_model"]["layer"]
    print("   %-28s %-8s RMSLE=%.4f  underestimated=%.0f%%  (n=%d)" % (
        "no fit: Eq. (1)-(3) as is", "-", model["rmsle"],
        100 * model.get("underestimated", 0.0), model["n"]))
    print("   target: the same bytes read as a formula (a check, not a result)")
    _kinds(layer["memory"], "rows split (paper's)")
    print("\n== Layer duration (paper Fig. 5b reports >95% accuracy)")
    _kinds(layer["time"], "rows split (paper's)")
    _kinds(layer["time_size"], "sizes held out")
    _kinds(layer["time_cross"], "one model, layers held out")
    print("\n== Fig. 5a: operator memory by family, target = measured peak")
    print("   paper: Conv2D 0.0319  MatMul 0.0427")
    print("%-10s %6s %7s %8s %8s %8s %10s   %s" % (
        "family", "rows", "groups", "LR", "XGBoost", "RF", "peak rows",
        "Eq.(1)-(3)"))
    for family, entry in scored["by_family"].items():
        if entry.get("insufficient"):
            print("%-10s %6d  fewer than 12 rows" % (family, entry["n"]))
            continue
        measured = entry["memory_measured"]
        model = entry["byte_model"]
        print("%-10s %6d %7d %8s %8s %8s %10d   %8.4f" % (
            family, entry["n"], measured["groups"],
            *["%.4f" % _rmsle(measured, kind) if measured["groups"] else "-"
              for kind in regress.KINDS],
            model["n"], model["rmsle"]))
    print("\n   zero-byte rows dropped from every fit: layer %d, operators per family"
          % layer["memory"]["zero_rows"])


def _rmsle(entry: dict, kind: str) -> float:
    """RMSLE or NaN, because a fit can legitimately have nothing to score."""
    row = entry.get(kind) or {}
    return float(row.get("rmsle", float("nan")))


def _kinds(entry: dict, label: str) -> None:
    for kind in regress.KINDS:
        row = entry[kind]
        if row.get("insufficient"):
            print("   %-28s %-8s too few groups" % (label, kind))
            continue
        print("   %-28s %-8s RMSLE=%.4f  accuracy=%.1f%%  underestimated=%.0f%%  (n=%d)" % (
            label, kind, row["rmsle"], 100 * row["accuracy"],
            100 * row["underestimated"], row["n"]))



if __name__ == "__main__":
    raise SystemExit(main())
