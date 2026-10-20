"""Service profile (Sec. II-B): the per-layer vectors M and T of Eq. (4).

One call to :func:`profile_model` produces what the partitioner consumes: a
:class:`~slicer.partition.graph.LayerGraph` whose nodes carry a measured memory
footprint and a measured duration, plus the operator-level rows the predictors
are trained on. The two passes have to agree on layer names, which they do
because both drive the same :class:`~slicer.models.graphify.Namer`;
:func:`~slicer.profile.probe.measure_forward` raises rather than guessing when a
model calls a module a different number of times per pass, since a shape-dependent
call count is the one thing that silently mislabels a profile.
"""

from __future__ import annotations

import ctypes
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..models import zoo
from ..models.graphify import build_structure_graph, move_tensors
from ..partition.graph import LayerGraph
from .probe import (LayerRecord, OpRecord, attach_peaks, calibrate, correct,
                    measure_forward, median_repeats, request_size, scale)
from .topology import Profile, dominant, layer_profile


@dataclass
class OpSample:
    """One row of the ``<X, s, p> -> <m, t>`` training table.

    ``memory`` is the byte model: what the call's own storages weigh, computed
    from shapes. ``peak`` is the measurement: how far the allocator's high-water
    mark rose while it ran. Only the second one is a quantity a predictor has to
    learn, since the first is a formula its features already determine.
    """

    model: str
    device: str
    batch: int
    size: int  # s: request elements
    op: str
    family: str
    layer: str
    params: float  # p: bytes of parameters in the enclosing layer
    memory: float  # m: bytes
    ns: float  # t
    in_bytes: float = 0.0  # not one of the paper's features; see regress.py
    peak: float | None = None  # measured m, CUDA only


@dataclass
class LayerSample:
    model: str
    device: str
    batch: int
    size: int
    layer: str
    params: float
    memory: float  # consolidated from operators, Eq. (1)-(3)
    peak: float | None  # independently measured by the allocator, CUDA only
    returned: float  # bytes of the value the layer handed back, i.e. an edge's worth
    ns: float  # the layer's span plus the interval before the next module, from the
               # pass that times layers: calibrated, then scaled to the forward


@dataclass
class ServiceProfile:
    key: str
    paper: str
    device: str
    batch: int
    size: int
    graph: LayerGraph
    profiles: dict[str, Profile] = field(default_factory=dict)
    operators: list[OpSample] = field(default_factory=list)
    layers: list[LayerSample] = field(default_factory=list)
    forward_ns: float = 0.0
    peak_bytes: float | None = None
    dominant_count: int = 0
    stray_ops: int = 0
    raw_layer_ns: float = 0.0  # layer spans before the instrumentation is subtracted
    operator_ns: float = 0.0  # operator self time summed over layers
    layer_ns: float = 0.0  # layer spans summed, before they are scaled to the forward
    timing_factor: float = 1.0  # forward_ns / layer_ns, applied to every t_l
    peak_ratio: float = 0.0  # median measured peak / Eq. (1)-(3) footprint
    calibration: tuple = ()  # (ns per layer span, ns per operator span) subtracted

    def vectors(self) -> tuple[list[float], list[float]]:
        """Eq. (4): ``M = {M_1..M_L}`` and ``T = {t_1..t_L}`` in run order."""
        rows = [self.profiles[name] for name in self.graph.order if name in self.profiles]
        return ([r.memory for r in rows], [r.seconds for r in rows])


_OVERHEAD: dict[str, tuple[float, float]] = {}


def overhead(device: str) -> tuple[float, float]:
    """Calibrated once per device and process: it is a property of the probe."""
    if device not in _OVERHEAD:
        _OVERHEAD[device] = calibrate(device)
    return _OVERHEAD[device]


def profile_model(model: torch.nn.Module, inputs: Any, key: str = "",
                  device: str = "cpu", warmup: int = 3, repeats: int = 5,
                  graph: LayerGraph | None = None,
                  on_pass: Callable[[str], None] | None = None) -> ServiceProfile:
    """Structure once, then three instrumented forwards -- in that order.

    ``graph`` lets a caller reuse an extraction it has already paid for; the
    structure pass is pure Python tracing and dominates the wall clock of this
    function for the wide models.

    The three forwards are :func:`~slicer.profile.probe.measure_forward`'s
    ``operators``, ``layers`` and ``memory`` passes, and they are separate because
    each observer spoils the other two's readings: the allocator counters cost as
    much host time as a small layer does, and the operator observer costs a
    ``__torch_function__`` dispatch per call inside every layer span. So the
    operator rows come from the first, Eq. (4)'s ``T`` from the second, and the
    measured footprints Table I is fitted against from the third, joined by
    position. The second pass also bills each layer the interval up to the next
    module, so the vector covers the whole forward rather than its leaf calls; what
    is left over is the profiler's own overhang, and that is removed by scaling the
    vector onto the uninstrumented forward, with
    :attr:`ServiceProfile.timing_factor` reporting how far it had to move.

    ``on_pass`` is called with the name of each pass as it begins, for a caller that
    prints progress; nothing about the readings depends on it.
    """
    batch = _batch_of(inputs)
    size = request_size(inputs)
    spec_key = key or model.__class__.__name__
    note = on_pass or (lambda label: None)
    if graph is None:
        graph = build_structure_graph(model, inputs, name=spec_key, device=device)
    order = graph.order

    runs: dict[str, dict] = {}

    def pass_(watch: str) -> dict:
        collected = []
        note("%s: %d warmup + %d timed forwards" % (watch, warmup, repeats))
        for index in range(warmup + repeats):
            layers, _ = measure_forward(model, inputs, device=device,
                                        expect_order=order, watch=watch)
            if index >= warmup:
                collected.append(layers)
        runs[watch] = collected
        return median_repeats(collected)

    per_layer, per_op = overhead(device)
    # The operator rows: which layer each call ran in, what it materialised, and
    # its own time. Only this pass sees the calls, so only this pass can say.
    medians = correct(pass_("operators"), per_layer, per_op)
    # Eq. (4)'s t_l comes from the pass that watches nothing but the leaf modules,
    # since the operator observer's own dispatch cost would otherwise be baked
    # into every span it reports. Its layer records are otherwise identical.
    spans = pass_("layers")
    before = {name: record.ns for name, record in spans.items()}
    spans = correct(spans, per_layer, 0.0)
    for name, record in medians.items():
        record.ns = spans[name].ns
    if torch.device(device).type == "cuda":
        # Bytes are deterministic, so one pass of the reader is enough; the peaks
        # in it are what Table I is fitted against.
        note("memory: 1 forward")
        footprint, _ = measure_forward(model, inputs, device=device,
                                       expect_order=order, watch="memory")
        medians = attach_peaks(medians, median_repeats([footprint]))
    note("uninstrumented forward: %d timed" % repeats)
    forward_ns = _plain_latency(model, inputs, device, repeats)
    note("service peak RSS")
    peak_bytes = _service_peak(model, inputs, device)
    samples = ServiceProfile(spec_key, _paper_name(spec_key), device, batch, size,
                             graph, forward_ns=forward_ns,
                             peak_bytes=peak_bytes,
                             stray_ops=sum(r.billed for r in runs["operators"][-1]),
                             calibration=overhead(device))
    samples.raw_layer_ns = sum(before.values())
    samples.layer_ns = sum(record.ns for record in medians.values())
    # Span arithmetic cannot close on the forward from both sides at once, so the
    # vector is stretched to the one number that is measured exactly.
    if samples.layer_ns:
        samples.timing_factor = scale(medians, forward_ns)

    ratios: list[float] = []
    for name, record in medians.items():
        profile = layer_profile(graph, name, record)
        samples.profiles[name] = profile
        samples.operator_ns += sum(operator.ns for operator in record.ops)
        if record.peak and profile.memory:
            ratios.append(record.peak / profile.memory)
        samples.dominant_count += len(dominant(record.ops))
        samples.layers.append(LayerSample(
            spec_key, device, batch, size, name, record.params,
            profile.memory, record.peak, record.returned, record.ns))
        for operator in record.ops:
            samples.operators.append(_op_row(operator, spec_key, device, batch, size,
                                             record.params))
    if ratios:
        samples.peak_ratio = sorted(ratios)[len(ratios) // 2]
    _fill(graph, samples.profiles)
    return samples


def _run(model: torch.nn.Module, inputs: Any, device: str) -> Any:
    moved = move_tensors(inputs, device)
    with torch.no_grad():
        return model(**moved) if isinstance(moved, dict) else model(*moved)


def _plain_latency(model: torch.nn.Module, inputs: Any, device: str,
                   repeats: int) -> float:
    """Median unsplit forward time, on a forward nothing is instrumenting."""
    sync = torch.cuda.synchronize if torch.device(device).type == "cuda" else (lambda: None)
    for _ in range(3):
        _run(model, inputs, device)
    sync()
    spans = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _run(model, inputs, device)
        sync()
        spans.append(float(time.perf_counter_ns() - started))
    spans.sort()
    return spans[len(spans) // 2]


def _service_peak(model: torch.nn.Module, inputs: Any, device: str) -> float | None:
    """High-water live bytes of one forward.

    Deliberately a separate pass from the instrumented ones: the tracer's own
    bookkeeping is integers, but the per-operator synchronizes change when the
    allocator frees, so this is measured on a plain forward.
    """
    if torch.device(device).type != "cuda":
        return _service_peak_cpu(model, inputs)
    _run(model, inputs, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    _run(model, inputs, device)
    torch.cuda.synchronize()
    return float(max(torch.cuda.max_memory_allocated(device) - base, 0.0))


def _resident_bytes() -> float | None:
    """This process's resident set, from ``/proc``; None where procfs is not there."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return pages * float(resource.getpagesize())


def _trim_heap() -> None:
    """Hand freed arena pages back to the OS so the next reading starts clean."""
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError, TypeError):
        pass


def _service_peak_cpu(model: torch.nn.Module, inputs: Any) -> float | None:
    """CPU's stand-in for the allocator's high-water mark: the resident rise of a forward.

    glibc does not return the pages a freed activation used, so the resident set
    *after* a forward is that forward's high-water, and the delta against the set
    before it is the same "bytes the request adds on top of the weights" quantity
    :func:`_service_peak` reads on CUDA. Two things follow, and both are why this
    stays a service-level number rather than a per-layer one: it cannot be
    attributed to the layer that took the bytes, and the trim before it only
    returns pages the heap holds *entirely* free, so a forward that fragments its
    arenas is read slightly high.
    """
    _trim_heap()
    base = _resident_bytes()
    if base is None:
        return None
    _run(model, inputs, "cpu")
    return max(_resident_bytes() - base, 0.0)


def _fill(graph: LayerGraph, profiles: dict[str, Profile]) -> None:
    for name, profile in profiles.items():
        layer = graph.layer(name)
        layer.memory = profile.memory
        layer.time = profile.seconds


def _op_row(record: OpRecord, model: str, device: str, batch: int, size: int,
            params: float) -> OpSample:
    return OpSample(model, device, batch, size, record.op, record.family, record.layer,
                    params, record.out, record.ns, record.inp, record.peak)


def _batch_of(inputs: Any) -> int:
    rows = [t.size(0) for t in _tensors(inputs) if t.dim()]
    return max(rows, default=1)


def _tensors(inputs: Any) -> list[torch.Tensor]:
    if isinstance(inputs, dict):
        inputs = tuple(inputs.values())
    return [t for t in inputs if isinstance(t, torch.Tensor)]


def _paper_name(key: str) -> str:
    try:
        return zoo.get(key).paper
    except KeyError:
        return key


def load_and_profile(key: str, device: str = "cpu", batch: int | None = None,
                     dtype=None, warmup: int = 3, repeats: int = 5) -> ServiceProfile:
    """Build a benchmark service from the zoo and profile it in one step."""
    model, inputs = zoo.load(key, device=device, dtype=dtype, batch=batch)
    return profile_model(model, inputs, key=key, device=device,
                         warmup=warmup, repeats=repeats)


def to_dict(profile: ServiceProfile) -> dict:
    """JSON-ready form; the operator rows are dropped, they go to a CSV."""
    memory, time_vector = profile.vectors()
    return {
        "key": profile.key,
        "paper": profile.paper,
        "device": profile.device,
        "batch": profile.batch,
        "size": profile.size,
        "nodes": len(profile.graph),
        "forward_ms": round(profile.forward_ns / 1e6, 3),
        "timing_factor": round(profile.timing_factor, 3),
        "peak_mb": None if profile.peak_bytes is None
        else round(profile.peak_bytes / 2 ** 20, 2),
        "dominant_ops": profile.dominant_count,
        "stray_ops": profile.stray_ops,
        "layers": [{"name": row.layer, "params_mb": round(row.params / 2 ** 20, 4),
                    "memory_mb": round(row.memory / 2 ** 20, 4),
                    "peak_mb": None if row.peak is None else round(row.peak / 2 ** 20, 4),
                    "returned_mb": round(row.returned / 2 ** 20, 4),
                    "time_ms": round(row.ns / 1e6, 4)} for row in profile.layers],
        "M": [round(value, 3) for value in memory],
        "T": [round(value, 6) for value in time_vector],
    }
