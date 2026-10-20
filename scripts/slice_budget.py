"""What a slice is *billed* for, against what it has to hold.

Eq. (6) gives every slice an allocation ``M_g``, and :func:`slice_footprint` says where
that number comes from: the parameters of the layers in the slice, plus the widest
activation any of them produces. The runtime then drops everything the slice does not
own, so the two sides can be compared directly -- and if the second is bigger than the
first, the plan is not deployable at the price it optimised, which is a statement about
the profiling model rather than about this machine.

The comparison is made without releasing anything: the parameter table is read once and
each slice's kept ids are looked up in it, so one model load covers every slice.

Run on v100:

    python scripts/slice_budget.py --models bert_1_3b,inception_transformer
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from slicer.faas.program import record  # noqa: E402
from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph  # noqa: E402
from slicer.partition.cost import PlatformConfig, TimingModel, slice_footprint  # noqa: E402
from slicer.partition.hypad import partition  # noqa: E402
from slicer.profile.service import profile_model  # noqa: E402

MB = 2 ** 20


def _table(model: torch.nn.Module) -> dict[int, int]:
    """Every parameter and buffer the service owns, by object identity."""
    out: dict[int, int] = {}
    for module in model.modules():
        for value in list(module._parameters.values()) + list(module._buffers.values()):
            if value is not None:
                out[id(value)] = value.numel() * value.element_size()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="bert_1_3b,inception_transformer,bert_3_0b")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mb-per-vcpu", type=float, default=1792.0)
    args = parser.parse_args()

    platform = PlatformConfig(mb_per_vcpu=args.mb_per_vcpu)
    for key in [k.strip() for k in args.models.split(",")]:
        model, inputs = zoo.load(key, device="cpu", batch=args.batch)
        graph = build_structure_graph(model, inputs, name=key, device="cpu")
        profile = profile_model(model, inputs, key=key, device="cpu", graph=graph)
        sized = profile.graph
        plan = partition(sized, TimingModel(platform), platform)
        program = record(model, inputs, name=key, device="cpu")
        groups = [list(s.members) for s in plan.slices]
        cases = program.slices(groups)
        table = _table(model)
        total = sum(table.values())
        print("%s: %d slices, %.1f GB of parameters, lambda=%d MB/vCPU"
              % (key, len(groups), total / 1024 ** 3, int(args.mb_per_vcpu)), flush=True)
        print("  %-4s %8s %8s %8s %8s  %s"
              % ("slice", "alloc", "billed", "weights", "short", "layers"), flush=True)
        for sl, case, group in zip(plan.slices, cases, groups):
            # What the plan pays for, what the model says the slice needs, and what the
            # slice's own steps actually read. The last column is the one that decides
            # whether a function can be started at the price Eq. (6) optimised.
            weights = sum(table.get(key_id, 0) for key_id in case.keeps)
            billed = slice_footprint(sized, group)
            print("  %-4d %8.0f %8.0f %8.0f %8.0f  %s"
                  % (case.index, sl.alloc_mb, billed / MB, weights / MB,
                     max(0.0, weights / MB - sl.alloc_mb),
                     "%s..%s (%d)" % (group[0][:22], group[-1][:22], len(group))),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
