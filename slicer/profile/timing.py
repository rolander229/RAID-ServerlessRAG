"""``t_p``, ``t_a`` and ``t_c`` of Eq. (5) read off measurements.

:class:`~slicer.partition.cost.TimingModel` answers "how long does this layer take
on :math:`\\eta` workers" with an Amdahl law whose serial fraction is a constant
nobody measured, and aggregates a shard at a bandwidth the box was assumed to
have. Both numbers decide the partition: a serial fraction that is too small makes
every slice look sharding-friendly, and the DP then buys parallelism it cannot
deliver. This module replaces the two with curves taken on this machine.

What is measured is the **shape**, not the level: a layer's absolute time comes
from the service profile, and :class:`Scaling` supplies ``t(eta) / t(1)`` for the
layer that was actually run at ``eta`` threads. Running the whole service at each
thread count rather than each layer alone keeps the measurement honest about the
thing a layer cannot do on its own -- a small layer's wall time on this box is
mostly launch and Python, and those do not scale the way its flops do.

Three limits of this reading are reported rather than hidden:

* **Only the thread axis is measured.** Eq. (5)'s :math:`\\gamma_i` stands for any
  parallelization policy; here it is intra-op threading, because that is what a
  worker process of the bare-metal substitute can actually be given. Splitting a
  layer's output channels across :math:`\\eta` processes -- the sharding the paper
  describes -- is the runtime's job, and its aggregation cost is measured with the
  runtime rather than here.
* **A layer that never ran at :math:`\\eta` gets the whole-forward ratio**, which
  is the best available guess and is named as such in the ``source`` of
  :meth:`Scaling.ratio`.
* **The curve stops where the measurement stops.** Past the largest thread count
  measured the last ratio is reused rather than extrapolated, so a service cannot
  be claimed to run at linear speedup on threads it was never given.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from ..partition.cost import TimingModel
from ..partition.graph import Layer

# The thread counts a worker process can plausibly be given; 80 is this box's core
# count, so the top of the ladder is also the most one process can ask for.
DEFAULT_ETAS = (1, 2, 4, 8, 16, 32, 64, 80)


@dataclass
class Scaling:
    """Per-layer wall time at each thread count, plus the whole forward and a copy."""

    etas: tuple[int, ...] = DEFAULT_ETAS
    per_layer: dict[str, dict[int, float]] = field(default_factory=dict)
    whole: dict[int, float] = field(default_factory=dict)
    copy_bytes_per_s: float = 0.0

    def ratio(self, name: str, eta: int) -> tuple[float, str]:
        """``t(eta) / t(1)`` for one layer, and where the number came from.

        Piecewise linear between measured thread counts and flat past the largest,
        so a layer that got *slower* at 64 threads is never reported as faster at
        128.
        """
        if eta <= 1:
            return 1.0, "base"
        curve, source = self.per_layer.get(name), "layer"
        if curve is None:
            curve, source = self.whole, "forward"
        base = curve.get(1) if curve else None
        if not base:
            return 1.0, "none"
        steps = sorted(curve)
        if eta >= steps[-1]:
            return curve[steps[-1]] / base, source + "@max"
        if eta <= steps[0]:
            return curve[steps[0]] / base, source + "@min"
        if eta in curve:
            return curve[eta] / base, source
        low = max(step for step in steps if step < eta)
        high = min(step for step in steps if step > eta)
        weight = (eta - low) / (high - low)
        return (curve[low] + weight * (curve[high] - curve[low])) / base, \
            source + "@interp"


def measure_scaling(model: torch.nn.Module, inputs: Any, device: str = "cpu",
                    etas: Sequence[int] = DEFAULT_ETAS, order: Sequence[str] = (),
                    warmup: int = 1, repeats: int = 3) -> Scaling:
    """Run the service once per thread count and keep the per-layer medians.

    ``torch.set_num_threads`` is process-global, so this is the only caller
    allowed to change it, and it restores the previous value before returning.
    """
    from .probe import measure_forward, median_repeats

    scaling = Scaling(etas=tuple(sorted({int(e) for e in etas})))
    previous = torch.get_num_threads()
    try:
        for eta in scaling.etas:
            torch.set_num_threads(eta)
            runs = []
            for index in range(warmup + repeats):
                layers, _ = measure_forward(model, inputs, device=device,
                                            expect_order=tuple(order) or None,
                                            watch="layers")
                if index >= warmup:
                    runs.append(layers)
            medians = median_repeats(runs)
            for name, record in medians.items():
                scaling.per_layer.setdefault(name, {})[eta] = record.ns / 1e9
            scaling.whole[eta] = sum(r.ns for r in medians.values()) / 1e9
    finally:
        torch.set_num_threads(previous)
    scaling.copy_bytes_per_s = measure_copy_bandwidth(device)
    return scaling


def measure_copy_bandwidth(device: str = "cpu", mb: float = 64.0,
                           repeats: int = 5) -> float:
    """Bytes per second of an on-host copy of a representative tensor.

    ``t_a`` of Eq. (5) is a gather, and a gather inside one box is a copy. The
    paper leaves that bandwidth to the platform, so it is measured here rather than
    assumed at wire speed.
    """
    count = int(mb * 2 ** 20 / 4)
    source = torch.zeros(count, dtype=torch.float32, device=device)
    target = torch.empty_like(source)
    target.copy_(source)
    _sync(device)
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        target.copy_(source)
        _sync(device)
        best = min(best, time.perf_counter() - started)
    return source.numel() * source.element_size() / best


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@dataclass
class MeasuredTiming(TimingModel):
    """``TimingModel`` with ``t_p`` from a :class:`Scaling` and ``t_a`` from a copy."""

    scaling: Scaling = field(default_factory=Scaling)

    def compute_time(self, layer: Layer, eta: int) -> float:
        if eta > 1 and self.scaling.per_layer:
            return layer.time * max(group_ratio(self.scaling, layer, eta), 0.0)
        return super().compute_time(layer, eta)

    def aggregate_time(self, layer: Layer, eta: int, out_bytes: float) -> float:
        if eta <= 1 or out_bytes <= 0:
            return 0.0
        bandwidth = (self.scaling.copy_bytes_per_s
                     or self.platform.onhost_bandwidth_bytes_per_s)
        moved = out_bytes * (eta - 1) / eta / self.compression_ratio
        return moved / bandwidth


def group_ratio(scaling: Scaling, layer: Layer, eta: int) -> float:
    """The measured ``t(eta)/t(1)`` of a node that may fold several layers.

    An eliminated node stands for a group, the curve was taken per original layer,
    and each member keeps its own share of the group's time, so the group's ratio
    is the members' averaged over the layers it actually contains.
    """
    if len(layer.members) <= 1:
        return scaling.ratio(layer.name, eta)[0]
    return sum(scaling.ratio(name, eta)[0] for name in layer.members) / len(layer.members)
