#!/usr/bin/env python3
"""P4: what a cut costs to cross, and how much of that AE takes off (Fig. 3, Fig. 13b/13c).

Eq. (5) bills a split service ``sum(tc)`` over the tensors that cross a cut, and
this experiment measures one of those terms rather than predicting it: a real
activation of a real service, from one end of a channel to the other, shipped
whole and shipped coded. Two questions are read off the result, and they are the
paper's two.

**How much of the hop is the payload** (Fig. 8). A pipe carries the bytes
themselves, which is what an HTTP reply or a Redis ``SET`` is built on; shared
memory carries only ``(slot, length)`` and leaves the payload in a tmpfs region
both ends map. The gap between the two columns is the saving COM offers, and it is
the number :mod:`slicer.communicate.channel` exists to produce.

**How much of the hop is compression** (Fig. 3, Fig. 13b/13c). Fig. 7's codec falls
on the payload by the ratio it is fit to, so the wire term should shrink like
``1/R`` while the encode and decode terms do not shrink at all. The sweep over
``--ratios`` is looking for where the total turns around, which is the shape Fig. 3
draws; the ablations read from the same column what taking the codec away costs.

What is *not* measured here, said out loud because it bounds the result:

* One machine, and one process. The writer's thread and the reader's thread are the
  two slices, so no process startup or page-table work is in any column -- which is
  also why the pipe's read copies land on the *writer's* clock: a 12 MB payload
  cannot finish being written until somebody drains it. Shared memory has no such
  coupling, so its read copy is the reader's own. The columns say which is which
  rather than pretending they match.
* The reader is one thread for the whole sweep, not one per hop: ATen charges a
  freshly spawned thread its first convolution, which read as a flat 4--13 ms on a
  codec that decodes in 0.7 ms. That cost is real for a cold instance and is
  reported as ``cold_ms``; it is not what a warm cut pays.
* The external store the paper reaches through (Redis, or a function's HTTP reply)
  is not installed on this node, so the pipe stands in for it; see
  ``issues/comm-no-external-store.md``.
* The payload always crosses on host memory: an activation read off a CUDA service
  is copied out before it is shipped, so the hop measured for ``--device cuda`` is
  still a CPU-side hop. What a GPU-resident cut costs is P5's question.
* Reconstruction error is the relative L2 distance between an activation and its
  reconstruction. The paper reports the accuracy the service loses, which needs a
  task metric per model; that belongs to P5/P6, and a number claimed here would be
  a different quantity.

Run on v100: ``python experiments/e4_communication.py``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from slicer.communicate import ae  # noqa: E402
from slicer.communicate.channel import Arena, open_pipe, open_shm, tmpfs_root  # noqa: E402
from slicer.models import SPECS, build_structure_graph, load  # noqa: E402
from slicer.models.graphify import Namer, leaf_modules, module_paths  # noqa: E402

MB = 2 ** 20


def _channel_axis(tensor: torch.Tensor) -> int:
    """Which axis Fig. 7's encoder should thin: torch's own convention says.

    A 4-D activation is NCHW, so its channels are at axis 1; a sequence or a node
    feature matrix keeps them last. The choice does not change the ratio -- a
    ``1x1`` convolution from *C* to *C/R* costs ``1/R`` of the bytes on whichever
    axis it runs -- but it changes what the codec is allowed to mix, so the row
    reports it.
    """
    return 1 if tensor.dim() == 4 else -1


def capture_activations(model, inputs, names: set[str]) -> dict:
    """The output tensor of each named leaf module, from one real forward.

    Read off a module's *return value* rather than out of the graph, because that is
    the buffer a cut would put on the wire, and it lives only while the hook holds
    it. Copied to host memory so a CUDA service and a CPU one ship the same bytes.
    """
    namer = Namer(module_paths(model))
    taken: dict[str, torch.Tensor] = {}

    def hook(module, args, kwargs, output):
        name = namer(module)
        if name in names and name not in taken and isinstance(output, torch.Tensor):
            taken[name] = output.detach().to("cpu").contiguous().clone()
        return output

    handles = [m.register_forward_hook(hook, with_kwargs=True)
               for m in leaf_modules(model)]
    try:
        with torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return taken


def _pair(transport: str, nbytes: int):
    """A (sender, receiver, teardown) triple for one hop of ``nbytes``."""
    if transport == "pipe":
        sender, receiver = open_pipe()
        return sender, receiver, lambda: None
    if transport == "shm":
        # Two slots, so writing the next message never waits for the reader of the
        # last one: with a single slot the protocol is what gets measured, not the
        # transport. See ShmEndpoint.
        arena = Arena.of(slots=2, slot_bytes=nbytes)
        return (*open_shm(arena, range(2)), arena.destroy)
    raise ValueError("unknown transport %r" % (transport,))


def _reader_side(receiver, coded_shape, dtype, codec, original, out: list,
                 repeats: int) -> None:
    """Consume ``repeats`` messages on one thread, timing each hop on its own clock.

    One thread for the whole sweep rather than one per hop, because ATen charges a
    fresh thread its first convolution: a codec that decodes in 0.7 ms measured
    4--13 ms when every hop got a new reader, and that flat term was the thread
    pool, not Fig. 7. The first sample is kept as ``cold_ms`` anyway -- a cold
    instance is a real thing a serverless service pays for -- but it is not what a
    warm cut costs, so it is not what the minimum is taken over.
    """
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            before_reads = receiver.received.reads
            payload = receiver.recv()
            unpacked = time.perf_counter()
            coded = torch.frombuffer(bytearray(payload), dtype=dtype).view(coded_shape)
            row = {"read_ms": (unpacked - started) * 1e3, "bytes": len(payload),
                   "reads": receiver.received.reads - before_reads,
                   "decode_ms": 0.0, "relative_error": 0.0}
            if codec is not None:
                decoded = time.perf_counter()
                back = codec.decode(coded)
                row["decode_ms"] = (time.perf_counter() - decoded) * 1e3
                row["relative_error"] = float(
                    (back - original).norm() / (original.norm() or 1.0))
            out.append(row)
    except BaseException as error:  # re-raised by the caller after the join
        out.append(error)


def measure_hop(tensor: torch.Tensor, codec, transport: str, repeats: int) -> dict:
    """The four terms of one ``t_c``: encode, serialise, wire, decode.

    Reported as the fastest of ``repeats`` hops, because Eq. (5) prices an idle
    service's cut; ``repeats`` goes out with the row so a reader knows what the
    minimum was taken over. The whole hop is taken from one repeat rather than
    stitching per-term minima, so a row is a hop that happened.
    """
    dtype = tensor.dtype
    coded_shape = tuple(int(s) for s in (tensor if codec is None
                                         else codec.encode(tensor)).shape)
    nbytes = 1
    for size in coded_shape:
        nbytes *= size
    nbytes *= torch.tensor([], dtype=dtype).element_size()
    sender, receiver, teardown = _pair(transport, nbytes)
    read_rows: list = []
    peer = threading.Thread(target=_reader_side,
                            args=(receiver, coded_shape, dtype, codec, tensor,
                                  read_rows, max(repeats, 1)))
    peer.daemon = True
    peer.start()
    wrote: list = []
    try:
        for _ in range(max(repeats, 1)):
            started = time.perf_counter()
            coded = tensor if codec is None else codec.encode(tensor)
            encode_ms = (time.perf_counter() - started) * 1e3
            started = time.perf_counter()
            payload = coded.numpy().tobytes()
            pack_ms = (time.perf_counter() - started) * 1e3
            started = time.perf_counter()
            before_writes = sender.sent.writes
            sender.send(payload)
            wrote.append({
                "encode_ms": encode_ms,
                "pack_ms": pack_ms,
                "wire_ms": (time.perf_counter() - started) * 1e3,
                "writes": sender.sent.writes - before_writes,
            })
    finally:
        peer.join(timeout=120)
    if peer.is_alive():
        raise RuntimeError("%s hop blocked; the endpoints deadlocked" % transport)
    if len(read_rows) < len(wrote):
        raise RuntimeError("the reader finished short: %d of %d hops"
                           % (len(read_rows), len(wrote)))
    failed = next((row for row in read_rows if isinstance(row, BaseException)), None)
    if failed is not None:
        raise failed
    sender.close()
    receiver.close()
    teardown()

    rows = []
    for sent, read in zip(wrote, read_rows):
        terms = {**sent, **read}
        # What the channel carries, as opposed to what the codec adds: Fig. 8's
        # claim is about these three, Fig. 3's trade-off is between them and the
        # other two.
        terms["ship_ms"] = terms["pack_ms"] + terms["wire_ms"] + terms["read_ms"]
        terms["hop_ms"] = terms["ship_ms"] + terms["encode_ms"] + terms["decode_ms"]
        rows.append(terms)
    best = min(rows, key=lambda row: row["hop_ms"])
    return {
        "transport": transport,
        "ratio": codec.ratio if codec is not None else 1.0,
        "raw_mb": round(tensor.numel() * tensor.element_size() / MB, 3),
        "wire_mb": round(best["bytes"] / MB, 3),
        "encode_ms": round(best["encode_ms"], 4),
        "pack_ms": round(best["pack_ms"], 4),
        "wire_ms": round(best["wire_ms"], 4),
        "read_ms": round(best["read_ms"], 4),
        "decode_ms": round(best["decode_ms"], 4),
        "ship_ms": round(best["ship_ms"], 4),
        "hop_ms": round(best["hop_ms"], 4),
        # The reader's very first hop, before its thread has run a convolution.
        "cold_ms": round(read_rows[0]["read_ms"] + read_rows[0]["decode_ms"], 4),
        # The writer's syscall count: a pipe's whole buffer is 64 KB, so a 12 MB
        # payload is 192 writes, and Fig. 8 is a claim about exactly that.
        "writes": best["writes"],
        "reads": best["reads"],
        "relative_error": round(best["relative_error"], 6),
        "repeats": repeats,
    }


def run(key: str, device: str, batch: int | None, ratios, transports, edges: int,
        repeats: int, steps: int) -> list[dict]:
    """Every (edge, ratio, transport) for one service's widest cuts."""
    spec = SPECS[key]
    model, inputs = load(key, device=device, batch=batch)
    graph = build_structure_graph(model, inputs, name=key, device=device)
    # The widest edges are what a cut chokes on, and Fig. 3 draws a service by its
    # worst hop.
    chosen = sorted(graph.tensors, key=lambda t: -t.bytes)[:edges]
    activations = capture_activations(model, inputs, {edge.dst for edge in chosen})
    rows = []
    for edge in chosen:
        tensor = activations.get(edge.dst)
        if tensor is None or not tensor.numel():
            print("  %s: no activation captured" % edge.dst, flush=True)
            continue
        axis = _channel_axis(tensor)
        for ratio in ratios:
            codec = None if ratio <= 1 else ae.fit(tensor, ratio, channel_axis=axis,
                                                   steps=steps)
            for transport in transports:
                try:
                    row = measure_hop(tensor, codec, transport, repeats)
                except Exception as error:  # one bad hop must not end the service
                    print("  %s r=%g %s: %s: %s" % (edge.dst, ratio, transport,
                                                    type(error).__name__, error),
                          flush=True)
                    continue
                row.update(key=key, paper=spec.paper, family=spec.family,
                           device=device,
                           batch=spec.batch if batch is None else batch,
                           layer=edge.dst, edge_bytes=edge.bytes,
                           channel_axis=axis,
                           fit_loss=codec.accounting()["loss"] if codec else None)
                rows.append(row)
                print("  %-24s %-5s r=%-4g hop=%8.3fms ship=%7.3fms wire=%6.2fMB "
                      "err=%.4f cold=%7.3fms" % (
                          edge.dst, transport, row["ratio"], row["hop_ms"],
                          row["ship_ms"], row["wire_mb"], row["relative_error"],
                          row["cold_ms"]), flush=True)
        activations[edge.dst] = None  # a 12 MB activation is the working set here
    return rows


def _report(rows: list[dict]) -> None:
    """Fig. 3's shape: the terms of one hop against the compression ratio."""
    print("\n%-18s %-22s %-5s %4s %8s %7s %7s %8s %7s %8s %8s %7s" % (
        "model", "layer", "xport", "R", "wire/MB", "enc/ms", "wire/ms", "read/ms",
        "dec/ms", "ship/ms", "hop/ms", "err"))
    for row in rows:
        print("%-18s %-22s %-5s %4g %8.2f %7.3f %7.3f %8.3f %7.3f %8.3f %8.3f %7.4f" % (
            row["paper"], row["layer"], row["transport"], row["ratio"],
            row["wire_mb"], row["encode_ms"], row["wire_ms"], row["read_ms"],
            row["decode_ms"], row["ship_ms"], row["hop_ms"], row["relative_error"]))

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["paper"], row["layer"], row["transport"]), []).append(row)
    print("\n== where coding starts to pay (Fig. 3) ==")
    print("hop is what a slice's latency pays; ship is the three terms Eq. (5) bills"
          " as t_c, i.e. the transport alone. The two turn around at different R"
          " because the codec's own arithmetic does not shrink with its payload.")
    for key in sorted(grouped):
        curve = sorted(grouped[key], key=lambda r: r["ratio"])
        cheapest = min(curve, key=lambda r: r["hop_ms"])
        shipped = min(curve, key=lambda r: r["ship_ms"])
        raw = next((r for r in curve if r["ratio"] <= 1.0), None)
        saving = (1.0 - cheapest["hop_ms"] / raw["hop_ms"]) if raw and raw["hop_ms"] else 0.0
        bytes_off = (1.0 - curve[-1]["wire_mb"] / raw["wire_mb"]) if raw else 0.0
        print("%-18s %-22s %-5s best R=%-4g hop=%7.3fms (%.1f%% off no-codec), "
              "best R for ship=%g  wire %.2f->%.2fMB  hop/ms %s" % (
                  key[0], key[1], key[2], cheapest["ratio"], cheapest["hop_ms"],
                  100.0 * saving, shipped["ratio"], raw["wire_mb"] if raw else 0.0,
                  curve[-1]["wire_mb"],
                  " ".join("%g:%.2f" % (r["ratio"], r["hop_ms"]) for r in curve)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=None,
                        help="keys from slicer/models/zoo.py, default all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1, 8, 16, 32, 64])
    parser.add_argument("--transports", nargs="+", default=["pipe", "shm"],
                        choices=["pipe", "shm"])
    parser.add_argument("--edges", type=int, default=2,
                        help="widest cross-slice edges to ship, per service")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000,
                        help="Adam steps used to fit Fig. 7's codec; see ae.fit")
    parser.add_argument("--out", default="runs/e4_communication.json")
    args = parser.parse_args()

    if tmpfs_root() != "/dev/shm":
        print("note: no /dev/shm here, so the shm column measures a file in %s"
              % tmpfs_root(), flush=True)
    keys = args.models or sorted(SPECS)
    rows = []
    for key in keys:
        started = time.time()
        try:
            made = run(key, args.device, args.batch, args.ratios, args.transports,
                       args.edges, args.repeats, args.steps)
        except Exception as error:  # keep going: one bad model must not hide the rest
            made = []
            print("%-22s FAIL %s: %s" % (key, type(error).__name__, error), flush=True)
        rows += made
        print("%-22s %2d rows  %.1fs" % (key, len(made), time.time() - started),
              flush=True)

    _report(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
