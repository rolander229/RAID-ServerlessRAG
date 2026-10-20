"""P3: what HyPAD makes of each service's profile -- paper Fig. 11, Fig. 2b/2d.

Each row profiles one service on one request size, turns the profile into the
layer graph Eq. (4) defines, and runs Algorithm 1 on it. Three things are read
out, because the paper makes a claim about each:

* **slice count** (Fig. 11: "3--5 slices per DLIS", and Fig. 2b: cost stops
  falling past that), which is what the DP decides;
* **latency against the unsplit service** (Fig. 10a-c, Table III), where Eq. (6)
  constrains the plan to be no slower, so a plan that only saves memory is not
  admissible;
* **``MC`` = allocated memory x execution time** (Sec. III-C2, the ablation
  target of Fig. 13a), the resource-efficiency quantity.

Two knobs are swept rather than fixed, because the reproduction has to say which
of them the numbers depend on:

``--quota eq3|measured|max`` (swept)
    Eq. (3)'s consolidated footprint, the allocator's measured high water, or the
    wider of the two. Table I says these disagree by RMSLE 0.89 on our services,
    so a slice plan is only as good as the reading behind its quota. On CPU the
    per-layer reading does not exist (see :mod:`slicer.profile.service`, whose
    service-level number is a resident-set rise that cannot be attributed to a
    layer), so only ``eq3`` runs there and the row says so.
``--timing analytic|measured`` (swept)
    The Amdahl term of :class:`~slicer.partition.cost.TimingModel` against the
    thread-scaling curves :mod:`slicer.profile.timing` measures on this box. The
    search over eta is the second half of Alg. 1, and analytic scaling is exactly
    the assumption that search is meant to test.

Run on v100: ``python experiments/e3_partition.py --device cpu``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph  # noqa: E402
from slicer.partition.cost import PlatformConfig, TimingModel  # noqa: E402
from slicer.partition.hypad import partition, slice_curve  # noqa: E402
from slicer.profile.service import profile_model  # noqa: E402
from slicer.profile.timing import (  # noqa: E402
    DEFAULT_ETAS,
    MeasuredTiming,
    measure_scaling,
)

MB = 2 ** 20


def _quota(graph, peaks: dict, mode: str):
    """A copy of the layer graph whose ``memory`` is the quota HyPAD allocates to."""
    if mode == "eq3":
        return graph
    out = graph.copy()
    for name, peak in peaks.items():
        layer = out.layer(name)
        if mode == "measured":
            layer.memory = peak
        else:  # "max": never under-provision, whichever reading is wider
            layer.memory = max(layer.memory, peak)
    return out


def _peaks(profile) -> dict:
    return {row.layer: row.peak for row in profile.layers if row.peak}


def run(key: str, device: str, batch: int | None, args) -> list[dict]:
    """One service, one request size, one row per (quota, timing) combination."""
    model, inputs = zoo.load(key, device=device, batch=batch)
    graph = build_structure_graph(model, inputs, name=key, device=device)
    profile = profile_model(model, inputs, key=key, device=device, graph=graph)
    platform = PlatformConfig()

    scaling = None
    if args.timing != "analytic":
        scaling = measure_scaling(model, inputs, device=device,
                                  etas=DEFAULT_ETAS[: args.max_eta_levels + 1],
                                  order=profile.graph.order,
                                  warmup=1, repeats=2)

    rows = []
    for quota in args.quotas:
        peaks = _peaks(profile)
        if quota != "eq3" and not peaks:
            rows.append({"key": key, "paper": profile.paper, "device": device,
                         "batch": profile.batch, "quota": quota,
                         "ok": False,
                         "error": "no per-layer measured peak on %s (service-level "
                                  "rise only: %.1f MB)"
                                  % (device, (profile.peak_bytes or 0.0) / MB)})
            continue
        sized = _quota(profile.graph, peaks, quota)
        for timing_kind in args.timings:
            timing = (MeasuredTiming(platform, scaling=scaling)
                      if timing_kind == "measured" and scaling is not None
                      else TimingModel(platform))
            started = time.time()
            # The plan Alg. 1 actually returns, and then -- if asked for -- the
            # optimum at each pinned slice count: Fig. 2b/2d is the paper's own
            # evidence for where the curve turns, so the free answer only means
            # something next to the curve around it.
            plans = [partition(sized, timing, platform)]
            if args.curve:
                plans += slice_curve(sized, args.curve, timing, platform)
            for plan in plans:
                rows.append(_row(key, device, profile, peaks, sized, platform,
                                 plan, quota, timing_kind, time.time() - started))
    return rows


def _row(key: str, device: str, profile, peaks: dict, sized, platform, plan,
         quota: str, timing_kind: str, seconds: float) -> dict:
    """One line of the sweep: what the DP decided, and what it cost in time and money."""
    return {
        "key": key,
        "paper": profile.paper,
        "device": device,
        "batch": profile.batch,
        "size": profile.size,
        "quota": quota,
        "timing": timing_kind,
        "nodes": len(sized),
        "forward_ms": round(profile.forward_ns / 1e6, 3),
        "eq3_max_mb": round(max((l.memory for l in profile.graph), default=0) / MB, 1),
        "measured_max_mb": round(max(peaks.values(), default=0.0) / MB, 1),
        "service_peak_mb": round((profile.peak_bytes or 0.0) / MB, 1),
        "nodes_before": plan.stats["nodes_before"],
        "nodes_after": plan.stats["nodes_after"],
        # What the caller asked the DP for, as opposed to what it could deliver: a
        # simplified graph of p nodes has no plan with k > p slices.
        "requested_slices": plan.stats.get("requested_slices"),
        "slices": plan.n_slices,
        "max_alloc_mb": round(plan.max_alloc_mb, 0),
        "unsplit_alloc_mb": round(platform.allocate_mb(sized.peak_memory()), 0),
        "latency_ms": round(plan.latency_s * 1e3, 3),
        "unsplit_latency_ms": round(plan.unsplit_latency_s * 1e3, 3),
        "latency_vs_unsplit": round(
            plan.latency_s / plan.unsplit_latency_s, 4)
        if plan.unsplit_latency_s else None,
        # Eq. (6)'s constraint: whether the plan still costs more wall time on one
        # cold request than the unsplit service does. It is the filter this sweep
        # solves under -- ``slices`` is chosen so that it holds -- and a pinned row
        # that breaks it is shown as broken rather than repaired.
        "latency_over_budget": plan.stats["latency_over_budget"],
        "eta_backoff_steps": plan.stats["latency_backoff_steps"],
        # Which pinned counts the free search found admissible, and where it stopped:
        # the answer to "why is this service not cut" is in these three fields.
        "admissible_slices": plan.stats.get("admissible_slices"),
        "searched_slices": plan.stats.get("searched_slices"),
        "search_truncated": plan.stats.get("search_truncated"),
        "cost_us": round(plan.cost_usd * 1e6, 3),
        "unsplit_cost_us": round(plan.unsplit_cost_usd * 1e6, 3),
        "comm_out_mb": round(plan.comm_bytes / MB, 3),
        "etas": [s.eta for s in plan.slices],
        "slice_layers": [len(s.members) for s in plan.slices],
        "slice_alloc_mb": [s.alloc_mb for s in plan.slices],
        "mc_gb_s": round(sum(s.memory_consumption_gb_s for s in plan.slices), 6),
        "seconds": round(seconds, 1),
        "ok": True,
    }


def _report(rows: list[dict]) -> None:
    print("%-18s %5s %-9s %-10s %6s %6s %8s %8s %8s %7s" % (
        "model", "b", "quota", "timing", "slices", "eta", "lat/ms", "unsplit",
        "alloc/MB", "cost/u$"))
    for row in rows:
        if not row.get("ok"):
            print("%-18s %5s %-9s %-10s  %s" % (row["paper"], row["batch"],
                                                row.get("quota", "-"), "-",
                                                row.get("error", "FAIL")))
            continue
        # A pinned k the simplified graph cannot supply is answered at the largest one
        # it can, so the column shows what was asked for as well as what was built.
        asked = row.get("requested_slices")
        slices = "%d->%d" % (asked, row["slices"]) if asked and asked != row["slices"] \
            else "%d" % row["slices"]
        print("%-18s %5d %-9s %-10s %6s %6s %8.2f %8.2f %8.0f %7.2f%s" % (
            row["paper"], row["batch"], row["quota"], row["timing"], slices,
            "/".join(str(e) for e in row["etas"]), row["latency_ms"],
            row["unsplit_latency_ms"], row["max_alloc_mb"], row["cost_us"],
            "!" if row["latency_over_budget"] else ""))


def _curve_report(rows: list[dict]) -> None:
    """Fig. 2b/2d: Eq. (6)'s bill and Eq. (5)'s latency against a *pinned* slice count.

    One pair of lines per (service, batch, quota, timing), with a ``!`` on the counts
    Eq. (6)'s latency constraint rules out and ``=`` on the one the free search
    returned. The pair together is the whole argument of the sweep: cost keeps falling
    with k, latency keeps rising, and the constraint is the only thing between them --
    which is why Fig. 11's "3--5 slices per DLIS" can only be read off a plan that is
    solved *at* a k rather than minimised over one.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("ok"):
            grouped.setdefault(
                (row["paper"], row["batch"], row["quota"], row["timing"]), []).append(row)
    if not grouped:
        return

    print("\n== slice-count curve (Fig. 2b/2d) ==")
    for key in sorted(grouped):
        group = grouped[key]
        free = next((r for r in group if r.get("requested_slices") is None), None)
        per_k: dict[int, dict] = {}
        for row in group:
            if row.get("requested_slices") is not None:
                per_k.setdefault(row["requested_slices"], row)
        if not per_k:
            continue
        cells = []
        for k in sorted(per_k):
            row = per_k[k]
            cells.append("%d:%.3f%s%s" % (k, row["cost_us"],
                                          "!" if row["latency_over_budget"] else "",
                                          "=" if free is not None and k == free["slices"]
                                          else ""))
        print("%-18s b=%-2d %-9s %-9s  cost/u$  %s" % (
            key[0], key[1], key[2], key[3], "  ".join(cells)))
        print("%-18s %-5s %-9s %-9s  lat/ms   %s" % (
            "", "", "", "",
            "  ".join("%d:%.2f%s%s" % (k, per_k[k]["latency_ms"],
                                       "!" if per_k[k]["latency_over_budget"] else "",
                                       "=" if free is not None and k == free["slices"]
                                       else "")
                     for k in sorted(per_k))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=None,
                        help="keys from slicer/models/zoo.py, default all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--quota", nargs="+", default=["max"],
                        choices=["eq3", "measured", "max"])
    parser.add_argument("--timing", nargs="+", default=["analytic", "measured"],
                        choices=["analytic", "measured"])
    parser.add_argument("--max-eta-levels", type=int, default=6,
                        help="how many points of the scaling curve to measure")
    parser.add_argument("--curve", nargs="+", type=int, default=None, metavar="K",
                        help="also solve the DP for exactly these slice counts")
    parser.add_argument("--out", default="runs/e3_partition.json")
    args = parser.parse_args()
    args.quotas = args.quota
    args.timings = args.timing
    if args.curve and min(args.curve) < 1:
        parser.error("--curve takes slice counts of at least one")

    rows: list[dict] = []
    for key in (args.models or sorted(zoo.SPECS)):
        for batch in args.batches:
            try:
                rows += run(key, args.device, batch, args)
            except Exception as error:  # one bad service must not hide the rest
                print("%-18s b=%-4d FAIL %s: %s" % (key, batch,
                                                    type(error).__name__, error),
                      flush=True)
                rows.append({"key": key, "paper": key, "batch": batch, "ok": False,
                             "error": "%s: %s" % (type(error).__name__, str(error)[:200])})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        _report([r for r in rows if r.get("key") == key])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)
    _curve_report(rows)
    print("\nwrote %s (%d/%d ok)" % (args.out, sum(r.get("ok") for r in rows), len(rows)))
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
