"""What makes Eq. (6) admit a cut at all: the price of a vCPU and the size of a request.

HyPAD's plan for the twelve services collapses to *one slice* on CPU (``runs/e3_analytic.json``:
9 of 12 services, all batches, both quota readings). That is not the search failing -- it is
the formulation. Two facts fix each other out:

1. The constraint under Eq. (6) reads :math:`t_\\tau(G,\\pi_S,\\pi_P)\\le\\sum_{l_i\\in G} t_{l_i}`,
   and the unsplit plan satisfies it *with equality*: at ``eta = 1`` a cut changes no layer's
   compute time and adds communication, so it is infeasible by definition.
2. ``eta`` is capped at :math:`M_{gi}/\\lambda`, so a slice needs ``2 * lambda`` of footprint
   before ``eta = 2`` exists. At Lambda's 1792 MB/vCPU that is 3.6 GB, and our measured CPU
   services hold 0.2--540 MB (``docs/results.md``: activations are MB-scale at batch 1-4,
   where the paper's Fig. 11 shows 0.5--3 GB *per slice*).

So the number of slices any method reports is a statement about lambda and about how big a
request was profiled, before it is a statement about the methods. This sweep varies both and
prints, for every combination, what each of the six Fig. 10 methods decides -- which is the
only way to say honestly whether the comparison in Fig. 10 has any content in this
reproduction, and where its threshold lies.

Run on v100:

    python experiments/e3b_cut_price.py --models resnet50,vgg16 \\
        --batches 1,4,16,64 --lambdas 1792
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph  # noqa: E402
from slicer.partition.baselines import FIGURE_10, plan_for  # noqa: E402
from slicer.partition.cost import PlatformConfig, TimingModel, slice_footprint  # noqa: E402
from slicer.partition.hypad import partition  # noqa: E402
from slicer.profile.service import profile_model  # noqa: E402

MB = 2 ** 20


def _plans(graph, slicer, platform) -> dict:
    """Each method's decision on this graph: how many slices, and the eta of each."""
    timing = TimingModel(platform)
    out = {}
    for method in FIGURE_10:
        started = time.time()
        plan, _ = plan_for(method, graph, slicer, timing, platform)
        out[method] = {"slices": len(plan.slices),
                       "etas": [s.eta for s in plan.slices],
                       "alloc_mb": [s.alloc_mb for s in plan.slices],
                       "latency_ms": round(plan.latency_s * 1e3, 3),
                       "cost_us": round(plan.cost_usd * 1e6, 3),
                       "seconds": round(time.time() - started, 2)}
    return out


def run(key: str, batch: int, lambdas: list[float]) -> list[dict]:
    model, inputs = zoo.load(key, device="cpu", batch=batch)
    graph = build_structure_graph(model, inputs, name=key, device="cpu")
    profile = profile_model(model, inputs, key=key, device="cpu", graph=graph)
    sized = profile.graph
    whole = slice_footprint(sized, sized.order)
    rows = []
    for lam in lambdas:
        platform = PlatformConfig(mb_per_vcpu=lam)
        started = time.time()
        slicer = partition(sized, TimingModel(platform), platform)
        rows.append({
            "key": key, "paper": profile.paper, "batch": batch,
            "request_mb": round(profile.size / MB, 3),
            "forward_ms": round(profile.forward_ns / 1e6, 3),
            "footprint_mb": round(whole / MB, 1),
            "alloc_mb": round(platform.allocate_mb(whole), 0),
            "mb_per_vcpu": lam,
            "eta2_needs_mb": 2 * lam,
            "seconds": round(time.time() - started, 1),
            "plans": _plans(sized, slicer, platform),
        })
        print("%-12s b=%-4d %-9.1f MB  lambda=%6.0f  slices=%s" % (
            key, batch, whole / MB, lam,
            "  ".join("%s:%d[%s]" % (m[:4], p["slices"],
                                     ",".join(str(e) for e in p["etas"]))
                      for m, p in rows[-1]["plans"].items())), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="resnet50,vgg16,bert_1_3b")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--lambdas", default="1792",
                        help="MB of memory per vCPU, i.e. lambda of Eq. (6)")
    parser.add_argument("--out", default="runs/e3b_cut_price.json")
    args = parser.parse_args()

    lambdas = [float(v) for v in args.lambdas.split(",")]
    batches = [int(v) for v in args.batches.split(",")]
    rows = []
    for key in args.models.split(","):
        for batch in batches:
            try:
                rows += run(key.strip(), batch, lambdas)
            except Exception as error:  # one bad shape must not lose the other eleven
                print("%-12s b=%-4d FAIL %s: %s"
                      % (key, batch, type(error).__name__, error), flush=True)
                rows.append({"key": key, "batch": batch, "ok": False,
                             "error": "%s: %s" % (type(error).__name__, error)})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)
    print("wrote %d rows to %s" % (len(rows), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
