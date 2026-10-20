"""Graph simplification: node and edge elimination (Sec. II-C, Fig. 6)."""

from __future__ import annotations

from .graph import Layer, LayerGraph, Tensor

# Alg. 1 folds two successive layers when their footprints differ by at most 5%.
SIMILARITY_THRESHOLD = 0.05


def _footprint_gap(a: Layer, b: Layer) -> float:
    if a.memory <= 0 or b.memory <= 0:
        return 0.0
    return abs(a.memory - b.memory) / max(a.memory, b.memory)


def node_elimination(graph: LayerGraph, threshold: float = SIMILARITY_THRESHOLD) -> tuple[LayerGraph, int]:
    """Fold ``lj`` into its successor when it has one in-edge, one out-edge and a similar footprint.

    Returns the rewritten graph and the number of nodes folded away. ``lj`` is absorbed by its
    successor, so the surviving boundary edge is ``lj``'s *input* tensor: a split there still
    cuts the same place, while the near-duplicate cut between ``lj`` and its successor is gone.
    """
    out = graph.copy()
    folded = 0
    for name in list(out.order):
        layer = out.layers.get(name)
        if layer is None:
            continue  # already absorbed earlier in this pass
        ins, outs = out.in_edges(name), out.out_edges(name)
        if len(ins) != 1 or len(outs) != 1:
            continue
        succ_name = outs[0].dst
        if succ_name == name or succ_name not in out.layers:
            continue
        succ = out.layers[succ_name]
        # A successor with several producers is a join point: folding into it would attribute
        # one branch's memory to a node that concurrently holds every branch's activations.
        if len(out.in_edges(succ_name)) != 1:
            continue
        if _footprint_gap(layer, succ) > threshold:
            continue

        members = layer.members + succ.members
        merged = Layer(
            name=succ_name,
            memory=max(succ.memory, layer.memory),  # Eq. (1): a chain peaks at its largest layer
            time=succ.time + layer.time,  # Eq. (1): ... and costs the sum of them
            params=succ.params + layer.params,
            members=members,
        )
        out.layers[succ_name] = merged

        # Drop both edges touching ``name`` (the input edge is recreated, the internal one is
        # gone by design: that near-duplicate cut point is what elimination removes).
        out.tensors = [t for t in out.tensors if t.src != name and t.dst != name]
        out.add_tensor(Tensor(ins[0].src, succ_name, ins[0].bytes))
        del out.layers[name]
        out.order.remove(name)
        folded += 1
    return out, folded


def edge_elimination(graph: LayerGraph) -> tuple[LayerGraph, int]:
    """Merge parallel edges that share source and destination into one (Fig. 6b).

    The merged edge carries every tensor that crosses the boundary, so its volume is the sum.
    """
    out = graph.copy()
    merged: dict[tuple[str, str], Tensor] = {}
    leftovers: list[Tensor] = []
    removed = 0
    for tensor in out.tensors:
        key = (tensor.src, tensor.dst)
        if key in merged:
            merged[key] = Tensor(key[0], key[1], merged[key].bytes + tensor.bytes)
            removed += 1
        else:
            merged[key] = tensor
    out.tensors = [merged[k] for k in dict.fromkeys((t.src, t.dst) for t in out.tensors)]
    return out, removed


def simplify(
    graph: LayerGraph, threshold: float = SIMILARITY_THRESHOLD, max_rounds: int = 100
) -> tuple[LayerGraph, dict]:
    """Iterate node then edge elimination to a fix point, giving ``G'`` of Alg. 1 lines 2-6."""
    current = graph
    stats = {"rounds": 0, "nodes_folded": 0, "edges_merged": 0}
    for _ in range(max_rounds):
        after_nodes, folded = node_elimination(current, threshold)
        after_edges, merged = edge_elimination(after_nodes)
        stats["rounds"] += 1
        stats["nodes_folded"] += folded
        stats["edges_merged"] += merged
        if len(after_edges) == len(current) and len(after_edges.tensors) == len(current.tensors):
            return after_edges, stats
        current = after_edges
    return current, stats
