"""Layer-graph representation shared by the profiler, the partitioner and the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Layer:
    """One node of the DAG. After elimination a node may stand for a group of layers."""

    name: str
    memory: float  # bytes, peak footprint of the layer
    time: float  # seconds, execution time before partitioning
    params: float = 0.0  # bytes of learnable parameters, drives horizontal sharding
    members: tuple[str, ...] = ()  # original layer names folded into this node

    def __post_init__(self) -> None:
        if not self.members:
            self.members = (self.name,)


@dataclass(frozen=True)
class Tensor:
    """One edge: the output of ``src`` is the input of ``dst``."""

    src: str
    dst: str
    bytes: float = 0.0


class LayerGraph:
    """Directed acyclic layer graph ``G = <V, E>`` of a deep learning inference service."""

    def __init__(self, layers: list[Layer], tensors: list[Tensor], name: str = "") -> None:
        self.name = name
        self.layers: dict[str, Layer] = {}
        self.order: list[str] = []
        for layer in layers:
            if layer.name in self.layers:
                raise ValueError(f"duplicate layer name {layer.name!r}")
            self.layers[layer.name] = layer
            self.order.append(layer.name)
        self.tensors: list[Tensor] = []
        for tensor in tensors:
            self.add_tensor(tensor)

    # -- topology ---------------------------------------------------------

    def add_tensor(self, tensor: Tensor) -> None:
        for endpoint in (tensor.src, tensor.dst):
            if endpoint not in self.layers:
                raise KeyError(f"edge {tensor} touches unknown layer {endpoint!r}")
        self.tensors.append(tensor)

    def out_edges(self, name: str) -> list[Tensor]:
        return [t for t in self.tensors if t.src == name]

    def in_edges(self, name: str) -> list[Tensor]:
        return [t for t in self.tensors if t.dst == name]

    def succs(self, name: str) -> list[str]:
        return sorted({t.dst for t in self.out_edges(name)})

    def preds(self, name: str) -> list[str]:
        return sorted({t.src for t in self.in_edges(name)})

    def sources(self) -> list[str]:
        return [n for n in self.order if not self.in_edges(n)]

    def sinks(self) -> list[str]:
        return [n for n in self.order if not self.out_edges(n)]

    def layer(self, name: str) -> Layer:
        return self.layers[name]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        return (self.layers[n] for n in self.order)

    def copy(self) -> "LayerGraph":
        return LayerGraph(
            [Layer(l.name, l.memory, l.time, l.params, l.members) for l in self],
            list(self.tensors),
            self.name,
        )

    # -- quantities the cost model needs ---------------------------------

    def total_time(self) -> float:
        """Latency of the unpartitioned service, i.e. ``sum of t_li`` in Eq. (6)."""
        return sum(l.time for l in self)

    def peak_memory(self) -> float:
        return max((l.memory for l in self), default=0.0)

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; raises on cycles so a malformed profile fails loudly."""
        indeg = {n: len({t.src for t in self.in_edges(n)}) for n in self.layers}
        ready = [n for n in self.order if indeg[n] == 0]
        out: list[str] = []
        while ready:
            ready.sort()
            node = ready.pop(0)
            out.append(node)
            for nxt in self.succs(node):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if len(out) != len(self.layers):
            cyclic = sorted(set(self.layers) - set(out))
            raise ValueError(f"layer graph {self.name!r} has a cycle over {cyclic}")
        return out

    def describe(self) -> str:
        lines = [f"{self.name}: {len(self)} nodes, {len(self.tensors)} edges"]
        for name in self.topological_order():
            l = self.layers[name]
            folded = "" if len(l.members) == 1 else f"  <-{list(l.members)}"
            lines.append(
                f"  {name:<28} M={l.memory / 2**20:8.2f}MiB t={l.time * 1e3:8.3f}ms "
                f"P={l.params / 2**20:8.2f}MiB{folded}"
            )
        return "\n".join(lines)
