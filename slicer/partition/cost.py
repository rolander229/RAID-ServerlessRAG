"""Cost and latency model of Eq. (5) and Eq. (6).

Every quantity is expressed in seconds, bytes and dollars. The analytic sharding model below is
the default; :class:`slicer.profile.timing.MeasuredTiming` replaces it with curves measured on the
box, which is what the paper's SP feeds into MPE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .graph import Layer, LayerGraph, Tensor

MB = 2**20
GB = 2**30


@dataclass(frozen=True)
class PlatformConfig:
    """Resource/pricing knobs of the serving platform.

    ``mb_per_vcpu`` is lambda (``λ``) in Eq. (6): the memory-to-vCPU ratio a platform fixes, which
    caps how many sub-slices a slice may be cut into.
    """

    name: str = "openfaas-bare"
    mb_per_vcpu: float = 1792.0  # AWS Lambda grants 1 vCPU per 1792 MB
    min_alloc_mb: float = 128.0  # Lambda's smallest function configuration
    alloc_granularity_mb: float = 64.0
    mem_price_per_gb_s: float = 1.667e-5  # $/GB/s, Sec. III-A2
    net_price_per_s: float = 0.0  # cn in Eq. (6): $ per second of transmission
    onhost_bandwidth_bytes_per_s: float = 10e9  # /dev/shm copy, same host
    # Table II's ConnectX-6 link is 50 *Gbit*/s, i.e. 6.25 GB/s. Everything else in
    # this module moves bytes, so the conversion happens here once rather than in
    # whichever formula divides by it.
    crosshost_bandwidth_bytes_per_s: float = 50e9 / 8
    serial_fraction: float = 0.05  # Amdahl term of eta-way channel sharding
    #: Table II: each of the fourteen servers holds 187 GB, which is what bounds how
    #: many slices a co-locating placement can put on one machine.
    server_memory_mb: float = 187 * 1024.0
    # What one function can actually be given: Table II's servers have 104
    # threads, and a slice whose weights allow more cannot use them. Without this
    # the eta scan runs to alloc/lambda, which for a 10 GB transformer is a
    # four-digit number of configurations that do not exist.
    max_eta: int = 104

    def allocate_mb(self, footprint_bytes: float) -> float:
        raw = max(footprint_bytes / MB, self.min_alloc_mb)
        steps = math.ceil(raw / self.alloc_granularity_mb)
        return max(self.min_alloc_mb, steps * self.alloc_granularity_mb)

    def max_subslices(self, alloc_mb: float) -> int:
        return max(1, min(self.max_eta, int(alloc_mb // self.mb_per_vcpu)))


@dataclass
class TimingModel:
    """tp / ta / tc of Eq. (5)."""

    platform: PlatformConfig = field(default_factory=PlatformConfig)
    compression_ratio: float = 1.0  # R = |To| / |Te| of the autoencoder in COM

    def compute_time(self, layer: Layer, eta: int) -> float:
        """``tp(l_i, gamma_i)``: layer time once sharded into ``eta`` equal sub-slices."""
        if eta <= 1:
            return layer.time
        serial = self.platform.serial_fraction
        return layer.time * (serial + (1 - serial) / eta)

    def aggregate_time(self, layer: Layer, eta: int, out_bytes: float) -> float:
        """``ta(l_i, gamma_i)``: gathering the ``eta`` parameter/result shards."""
        if eta <= 1 or out_bytes <= 0:
            return 0.0
        moved = out_bytes * (eta - 1) / eta / self.compression_ratio
        return moved / self.platform.onhost_bandwidth_bytes_per_s

    def comm_time(self, tensor_bytes: float, same_host: bool) -> float:
        """``tc(e, gamma_i, gamma_j)`` from data volume and link bandwidth."""
        if tensor_bytes <= 0:
            return 0.0
        bw = (
            self.platform.onhost_bandwidth_bytes_per_s
            if same_host
            else self.platform.crosshost_bandwidth_bytes_per_s
        )
        return tensor_bytes / self.compression_ratio / bw

    def comm_cost(self, tensor_bytes: float, same_host: bool) -> float:
        """The ``cn * tc`` term of Eq. (6): a price per second of transmission."""
        return self.platform.net_price_per_s * self.comm_time(tensor_bytes, same_host)


@dataclass(frozen=True)
class Slice:
    """A deployed slice: consecutive graph nodes served by one function."""

    nodes: tuple[str, ...]
    members: tuple[str, ...]  # original layer names behind those nodes
    alloc_mb: float
    eta: int

    @property
    def key(self) -> str:
        return "+".join(self.nodes)


def boundary_edges(graph: LayerGraph, groups: list[list[str]]) -> list[Tensor]:
    """Edges that cross between consecutive groups, i.e. the tensors that must be transferred."""
    group_of = {name: i for i, group in enumerate(groups) for name in group}
    return [t for t in graph.tensors if group_of.get(t.src) != group_of.get(t.dst)]


def slice_footprint(graph: LayerGraph, group: Sequence[str]) -> float:
    """Bytes a function running ``group`` has to hold.

    Two terms, and the paper only profiles one of them: Eq. (1)-(3) give the
    widest activation the slice ever has live, while the weights of every layer in
    the slice are resident for as long as the function is. Both matter for the
    quota -- a VGG slice with 528 MB of parameters cannot run in the 128 MB
    minimum configuration the pricing is written against -- and only the second
    one is additive, which is why :math:`\\lambda` of Eq. (6) caps sub-slices at
    ``alloc / 1792`` rather than at whatever the activation peak allows.

    Horizontal sharding is not a term here. It splits the parameters *between*
    sub-slices, so the bytes the slice holds -- the quantity Eq. (6) bills and the
    quantity that buys its vCPUs -- are the same whether one function reads them or
    ``eta`` functions do. Dividing by ``eta`` here makes ``eta`` free and breaks the
    constraint written under the objective, which reads :math:`1 \\le \\eta \\le
    M_{gi} / \\lambda`.
    """
    layers = [graph.layer(n) for n in group]
    params = sum(l.params for l in layers)
    return params + max((l.memory for l in layers), default=0.0)


def group_footprint(graph: LayerGraph, group: list[str]) -> Layer:
    """Aggregate a group of nodes as Eq. (1): memory peaks, time accumulates."""
    layers = [graph.layer(n) for n in group]
    members = tuple(m for l in layers for m in l.members)
    return Layer(
        name="+".join(group),
        memory=max(l.memory for l in layers),
        time=sum(l.time for l in layers),
        params=sum(l.params for l in layers),
        members=members,
    )
