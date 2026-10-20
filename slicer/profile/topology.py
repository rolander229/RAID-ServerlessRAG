"""Consolidate operator profiles into layer and service profiles (Eq. 1-3).

The paper gives three consolidation rules and no way to tell which applies: a
sequential chain takes the maximum operator memory and the sum of times,
parallel branches take the *sum* of the operands that are live at the same
position and the maximum time there, and a hybrid takes the maximum of the two.
What decides it here is where the parallelism is actually visible.

Inside a leaf module it is not. Eager PyTorch runs one branch at a time, so an
operator's nesting depth is a property of the Python call stack, not of the data
flow, and grouping operators by depth would invent positions the hardware never
had. Within a layer this module therefore applies Eq. (1).

Between modules the graph does show it: :mod:`slicer.models.graphify` keeps a
separate edge per operand of a fusion, because a cut above a ``cat`` has to ship
each branch, and those edges are exactly the :math:`m_{ij}` of Eq. (2).
:func:`layer_profile` takes the maximum of the two readings, which is Eq. (3)
for the mixed case, and it is why the profile a slice inherits never forgets
that a fused operand was live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from ..partition.graph import LayerGraph

if TYPE_CHECKING:  # imported for names only; this module never touches torch
    from .probe import LayerRecord, OpRecord


@dataclass(frozen=True)
class Profile:
    """A layer's consolidated footprint: Eq. (1)-(3) applied to one node."""

    memory: float
    time: float

    @property
    def seconds(self) -> float:
        return self.time / 1e9


def chain(ops: Sequence["OpRecord"]) -> Profile:
    """Eq. (1): the largest operand, and every operator's time."""
    return Profile(max((r.out for r in ops), default=0.0), sum(r.ns for r in ops))


def branches(rows: Sequence[Sequence["OpRecord"]]) -> Profile:
    """Eq. (2) over explicitly separated branches.

    ``rows[i][j]`` is the ``j``-th operator of branch ``i``, so position ``j``
    bills the sum of its operands -- they are all live for the fusion that joins
    the branches -- and charges the maximum of their durations.
    """
    if not rows:
        return Profile(0.0, 0.0)
    memory = time = 0.0
    for position in range(max(len(branch) for branch in rows)):
        column = [branch[position] for branch in rows if position < len(branch)]
        memory = max(memory, sum(r.out for r in column))
        time += max(r.ns for r in column)
    return Profile(memory, time)


def hybrid(serial: Profile, parallel: Profile) -> Profile:
    """Eq. (3)."""
    return Profile(max(serial.memory, parallel.memory), serial.time + parallel.time)


def layer_profile(graph: LayerGraph, name: str, record: "LayerRecord") -> Profile:
    """One node's footprint: operators within the layer, operands across edges.

    Memory takes the wider of the two readings, which is Eq. (3) for the mixed
    case. Time does not: Eq. (1) sums the operator durations, but the layer's own
    span is measured directly and covers the host work *between* its operators as
    well as the interval before the next module, and on a GPU that is most of a
    small layer -- the :math:`t_c` of Eq. (1) is the smaller number there. The
    operator sum is still reported next to it, as the check that no call is being
    billed twice.
    """
    inside = chain(record.ops)
    operands = sum(edge.bytes for edge in graph.in_edges(name))
    return Profile(max(inside.memory, operands), record.ns)


def dominant(ops: Iterable["OpRecord"], share: float = 0.8) -> list["OpRecord"]:
    """The operators covering ``share`` of a layer's memory, largest first.

    Sec. II-B: "arranges the memory footprint of operators in each layer in
    ascending order and chooses the top n operators responsible for 80% of the
    memory consumption". Selection is by memory because that is what the paper
    says, which leaves a layer whose cost is all time and no bytes -- an in-place
    activation -- without dominant operators; its time is still counted at
    service level.
    """
    rows = sorted(ops, key=lambda r: r.out)
    total = sum(r.out for r in rows)
    if total <= 0:
        return []
    picked: list["OpRecord"] = []
    covered = 0.0
    for record in reversed(rows):
        picked.append(record)
        covered += record.out
        if covered >= share * total:
            break
    return picked


def dominant_operators(layers: Iterable["LayerRecord"],
                       share: float = 0.8) -> list["OpRecord"]:
    """A service's dominant operator set, one row per family; the paper expects <15."""
    families: dict[str, "OpRecord"] = {}
    for layer in layers:
        for record in dominant(layer.ops, share):
            families.setdefault(record.family, record)
    return sorted(families.values(), key=lambda r: -r.out)
