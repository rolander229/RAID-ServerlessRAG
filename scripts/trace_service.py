"""Replay one service through MOPAR's decision path and print every intermediate.

This is the reading end of the reproduction: §II-B's Eq. (4) vectors, §II-C's
Fig. 6 elimination, then Eq. (6) / Algorithm 1's answer, with the numbers a
reviewer needs in order to disagree with any of the three.  It deploys nothing
and measures nothing -- the wall clock lives in ``experiments/e5_end_to_end.py``,
and the last section points at it.

Run on v100:

    python scripts/trace_service.py --model bert_1_3b
    python scripts/trace_service.py --model vgg16 --mb-per-vcpu 448
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph  # noqa: E402
from slicer.partition.cost import PlatformConfig, TimingModel, slice_footprint  # noqa: E402
from slicer.partition.eliminate import simplify  # noqa: E402
from slicer.partition.hypad import partition  # noqa: E402
from slicer.profile.service import profile_model  # noqa: E402

MB = 2 ** 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="bert_1_3b")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mb-per-vcpu", dest="mb_per_vcpu", type=float, default=1792.0,
                        help="lambda of Eq. (6); below ~448 the small models start splitting")
    parser.add_argument("--layers", type=int, default=14, help="how many layers to tabulate")
    args = parser.parse_args(argv)

    model, inputs = zoo.load(args.model, device="cpu", batch=args.batch)
    graph = build_structure_graph(model, inputs, name=args.model, device="cpu")
    profile = profile_model(model, inputs, key=args.model, device="cpu", graph=graph)
    sized = profile.graph
    mems = sorted(layer.memory / MB for layer in sized)
    on_nodes = sum(layer.params for layer in sized) / MB
    in_model = sum(p.numel() * p.element_size() for p in model.parameters()) / MB

    print("== 0. what SP hands over (Eq. 4)")
    print("   %d nodes, %d edges, whole-service footprint %.0f MB, one forward %.1f ms"
          % (len(sized), len(sized.tensors), slice_footprint(sized, sized.order) / MB,
             profile.forward_ns / 1e6))
    print("   parameters: %.1f MB reachable from the module tree, %.1f MB billed to a node"
          % (in_model, on_nodes))
    if in_model - on_nodes > 0.01 * max(in_model, 1.0):
        print("   !! %.1f MB of weights are not attributed to any layer, so Eq. (6) cannot "
              "bill\n      them and slice_footprint under-allocates by that much "
              "(a function holding\n      them still has to be quoted for them). "
              "Unbilled modules are those whose forward\n      consumes their own "
              "parameters without a node being charged for them." % ((in_model - on_nodes),))
    print("   per-layer activation M_i: min %.2f MB, median %.2f MB, max %.2f MB"
          % (mems[0], mems[len(mems) // 2], mems[-1]))
    print("   `M_i` is bytes a layer newly materialises, weights excluded; the weights "
          "sit in\n   `params` and enter the bill only at slice level "
          "(cost.slice_footprint = sum(params) + max M).")

    print("\n== 1. the first %d layers: M_i (MB), t_i (ms), params (MB), bytes on its out-edge"
          % args.layers)
    for name in sized.order[:args.layers]:
        layer = sized.layer(name)
        outs = sized.out_edges(name)
        print("   %-30s M=%8.2f  t=%7.3f  params=%8.1f  out %s MB"
              % (name, layer.memory / MB, layer.time * 1e3, layer.params / MB,
                 "%.2f" % (sum(t.bytes for t in outs) / MB)))

    simplified, stats = simplify(sized)
    print("\n== 2. Fig. 6 elimination: %d rounds folded %d nodes and merged %d edges "
          "-> %d remain"
          % (stats["rounds"], stats["nodes_folded"], stats["edges_merged"], len(simplified)))
    for name in simplified.order[:args.layers]:
        layer = simplified.layer(name)
        outs = [t for t in simplified.tensors if t.src == name and t.dst != name]
        print("   %-24s M=%9.2f MB  t=%8.3f ms  %3d layers  holds %s ..  out %s MB"
              % (name, layer.memory / MB, layer.time * 1e3, len(layer.members),
                 layer.members[0][:24], "%.2f" % (sum(t.bytes for t in outs) / MB)))
    print("   ... (%d nodes total; a folded node reports Eq. (1): memory maxes, time sums)"
          % len(simplified))

    platform = PlatformConfig(mb_per_vcpu=args.mb_per_vcpu)
    plan = partition(simplified, TimingModel(platform), platform)
    print("\n== 3. Eq. (6) / Algorithm 1's answer (lambda=%.0f MB per vCPU)" % args.mb_per_vcpu)
    print(plan.summary())
    for index, s in enumerate(plan.slices):
        shown = ", ".join(s.members[:5]) + (", ..." if len(s.members) > 5 else "")
        print("   slice%d: %d layers, %d nodes, %s" % (index, len(s.members),
                                                       len(s.nodes), shown))
    print("   Eq. (6)'s constraint compares %.1f ms against the unsplit, eta=1 reference "
          "%.1f ms\n   -- not against what Unsplit actually costs at its own eta."
          % (plan.latency_s * 1e3, plan.unsplit_latency_s * 1e3))

    print("\n== 4. where to read the consequences")
    print("   deployment + closed-loop wall clock:  experiments/e5_end_to_end.py "
          "--models %s" % args.model)
    print("   re-render every published table:      experiments/e5_end_to_end.py "
          "--report runs/e5_end_to_end.json")
    print("   per-slice weights/billed/alloc:        scripts/slice_budget.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
