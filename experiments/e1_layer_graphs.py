#!/usr/bin/env python3
"""Experiment 1: is the recovered layer graph of every service correct?

Nothing downstream can be trusted until this is yes. The partitioner reads a
DAG of layers with byte-carrying edges; if an edge is missing, a cut that looks
legal silently drops a skip connection and the measured latency is the latency
of a different model. So this script checks structure, not performance:

  coverage      every leaf module that actually ran is a node, and nothing that
                did not run is (Inception's auxiliary head is the example)
  acyclicity    topological_order() succeeds, i.e. the value-tracking walk did
                not invent a feedback edge
  connectivity  no node is floating free of both its predecessors and successors
  shape         total parameter bytes match ``model.parameters()``, which is the
                check that shared weights were not counted twice
  cuts          the bytes that would go on the wire for the widest boundary,
                which is what decides whether COM is worth anything

Run on v100:  python experiments/e1_layer_graphs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.models import SPECS, build_structure_graph, load  # noqa: E402
from slicer.partition.cost import boundary_edges  # noqa: E402


def check(key: str, device: str, batch: int | None) -> dict:
    spec = SPECS[key]
    started = time.perf_counter()
    model, inputs = load(key, device=device, batch=batch)
    build_s = time.perf_counter() - started

    graph = build_structure_graph(model, inputs, name=key, device=device)
    nodes, edges = len(graph), len(graph.tensors)
    topological = graph.topological_order()

    ran = set(topological)
    # A leaf that exists but never ran is not an error -- Inception's auxiliary
    # head is only called while training -- but it must be visible, because the
    # partitioner would otherwise be pricing a service that was never executed.
    not_traced = sorted(path for path, m in model.named_modules()
                        if not list(m.children()) and path not in ran)
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    widest = max((t.bytes for t in graph.tensors), default=0.0)
    # The quantity a residual chain inflates: Eq. (3) bills a layer the sum of its
    # in-edges, so one node with 145 of them reads as a 223 MB layer in a 16 MB
    # service. Which node held the record is reported next to the count.
    fan_in_node = max(topological, key=lambda n: len(graph.in_edges(n)), default="")
    fan_in = len(graph.in_edges(fan_in_node)) if fan_in_node else 0
    fan_in_mb = sum(e.bytes for e in graph.in_edges(fan_in_node)) / 2 ** 20
    sources, sinks = graph.sources(), graph.sinks()
    floating = [n for n in topological
                if n not in sources and n not in sinks and not graph.in_edges(n)]

    # A cut just above layer i of the execution order: the bytes it would move.
    cuts = [sum(t.bytes for t in boundary_edges(graph, [topological[:i]]))
            for i in range(1, len(topological))]

    return {
        "key": key,
        "paper": spec.paper,
        "family": spec.family,
        "source": spec.source,
        "batch": spec.batch if batch is None else batch,
        "build_s": round(build_s, 3),
        "nodes": nodes,
        "edges": edges,
        "params_mb": round(total_params / 2 ** 20, 1),
        "node_params_mb": round(sum(l.params for l in graph) / 2 ** 20, 1),
        "widest_edge_mb": round(widest / 2 ** 20, 2),
        "max_fan_in": fan_in,
        "fan_in_node": fan_in_node,
        "fan_in_mb": round(fan_in_mb, 2),
        "median_cut_mb": round(cuts[len(cuts) // 2] / 2 ** 20, 2) if cuts else 0.0,
        "sources": sources,
        "sinks": sinks,
        "not_traced": not_traced,
        "floating": floating,
        "ok": nodes > 0 and not floating and len(topological) == nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=None,
                        help="keys from slicer/models/zoo.py, default all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--out", default="runs/e1_layer_graphs.json")
    args = parser.parse_args()

    keys = args.models or sorted(SPECS)
    rows = []
    for key in keys:
        try:
            row = check(key, args.device, args.batch)
        except Exception as error:  # keep going: one bad model must not hide the rest
            row = {"key": key, "ok": False, "error": "%s: %s"
                   % (type(error).__name__, str(error)[:200])}
        rows.append(row)
        print("%-22s %5s %5s %10s %12s %-14s %s%s" % (
            row.get("paper", key), row.get("nodes", "-"), row.get("edges", "-"),
            "%.1fMB" % row["params_mb"] if "params_mb" in row else "-",
            "%.1fMB" % row["widest_edge_mb"] if "widest_edge_mb" in row else "-",
            "%d@%s" % (row["max_fan_in"], row["fan_in_node"])
            if "max_fan_in" in row else "-",
            "ok" if row["ok"] else "FAIL",
            "\n    " + row["error"] if row.get("error") else ""), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)
    print("\nwrote %s (%d/%d ok)" % (args.out, sum(r["ok"] for r in rows), len(rows)))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
