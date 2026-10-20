"""Measure the memory and duration of every layer and operator of one service.

This is the Service Profiler's measurement pass (Sec. II-B). It is a second
forward rather than an extension of
:func:`slicer.models.graphify.build_structure_graph`, because the structure pass
has to hold a reference to every tensor it tags and therefore inflates peak
memory: a pass that reports memory cannot be that pass. This one keeps integers
only, and borrows the tracer's naming so its rows join to the graph by name.

And on a GPU it is two more forwards as well, for the same reason one layer down:
reading the allocator's counters costs more host time than a small layer runs, and
installing the operator observer costs a layer's worth of dispatch, so the pass that
times layers sees nothing else. :class:`_Probe` names the three passes ``layers``,
``operators`` and ``memory``, and :func:`attach_peaks` joins their rows by position.

Three measurements per layer, and the gap between them is the point:

``out``
    Bytes the layer's operators materialised, consolidated per layer in
    :mod:`slicer.profile.topology`. This is the paper's own construction -- the
    layer profile "can be aggregated according to the layer topology" -- and the
    only reading available on both devices, so it is what reaches the partitioner.
``peak``
    CUDA only: the highest live-byte count seen while the layer ran, minus what
    was live when it started, and the same rise attributed to each operator
    inside it. An independent reading of the same quantity as ``out``, so
    :mod:`experiments.e2_service_profiling` can report how well Eq. (1)-(3) track
    what the allocator actually had to provide -- and can fit a predictor on a
    target that is a measurement rather than a formula.
``ns``
    Self time: a span minus the spans the call opened. Each span is a CUDA event
    pair resolved after one synchronize at the end of the forward, because a
    synchronizing span boundary would serialise the queue and measure the
    profiler rather than the service (see :class:`_Timer`). The storages a call
    is charged are those none of its children allocated. Measured over 3960
    operator rows of three services, no operator frame ever closed inside another
    -- eager PyTorch decomposes a composite below the ``__torch_function__``
    boundary, so ``nn.functional.conv2d`` is one frame and not two -- which makes
    both subtractions inert there and load-bearing for whatever does nest.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.overrides import TorchFunctionMode

from ..models.graphify import (Namer, OwnerCharge, _iter_tensors,
                               install_leaf_hooks, module_paths, move_tensors)


@dataclass
class OpRecord:
    """One torch-level function call."""

    op: str
    layer: str
    ns: float  # self time, i.e. own span minus the calls it opened
    out: float  # bytes of storages the call allocated, excluding any it returned back
    inp: float = 0.0  # bytes of the storages handed to it
    peak: float | None = None  # rise in the allocator's high-water mark, CUDA only

    @property
    def family(self) -> str:
        return op_family(self.op)


@dataclass
class LayerRecord:
    """One leaf-module call: the unit Eq. (1)-(3) consolidate operators over."""

    name: str
    params: float  # resident bytes, settled after the forward: see _Probe.settle_weights
    ns: float
    returned: float  # bytes of the value the module handed back, i.e. what a cut here ships
    ops: list[OpRecord] = field(default_factory=list)
    peak: float | None = None  # measured live-byte high-water mark, CUDA only
    billed: int = 0  # operators that ran outside every leaf module and were
                    # charged to this one as their consumer
    gap: float = 0.0  # time between this span and the next leaf module's start,
                     # already inside ns -- see :meth:`_Probe.bill_gaps`

_FAMILIES = (
    ("conv", ("conv1d", "conv2d", "conv3d", "conv_transpose", "convolution")),
    ("rnn", ("lstm", "gru")),
    ("attention", ("attention",)),
    ("matmul", ("matmul", "mm", "linear", "gemm", "dot")),
    ("pool", ("max_pool", "avg_pool", "adaptive_avg_pool", "interpolate")),
    ("norm", ("batch_norm", "layer_norm", "group_norm")),
    ("softmax", ("softmax", "log_softmax")),
    ("dropout", ("dropout",)),
    ("tanh", ("tanh", "sigmoid")),
    ("relu", ("relu", "gelu", "leaky_relu", "silu")),
    ("cat", ("cat", "stack", "concat")),
    ("reshape", ("permute", "transpose", "contiguous", "view", "reshape",
                 "expand", "squeeze", "unsqueeze", "movedim", "flatten")),
    ("add", ("add", "rsub", "sub")),
    ("mul", ("mul", "div", "mean", "sum", "var", "pow", "exp", "neg")),
    ("embed", ("embedding",)),
)


def op_family(name: str) -> str:
    """Map a torch function name onto one of the paper's operator categories.

    The table is searched in this order because the keys are substrings: the aten
    name of a convolution is ``_convolution``, ``expand`` must not be read as an
    exponent, ``addmm`` is a matmul rather than an addition, and
    ``multi_head_attention_forward`` is neither of the two ``mul`` matches inside
    its name.
    """
    base = name.rsplit(".", 1)[-1].lower()
    for family, keys in _FAMILIES:
        if any(key in base for key in keys):
            return family
    return "other"


def _pointers(obj: Any) -> set:
    """Storage addresses behind every tensor reachable from ``obj``.

    ``id()`` of a storage will not do: ``untyped_storage()`` builds a fresh Python
    wrapper on every call, so the device pointer is the only thing that identifies
    a buffer. The tensors are alive while their call is being closed, which is
    what keeps those pointers from being recycled underneath the comparison.
    ``StorageImpl`` outlives the tensor that owns it, so the address is stable.
    """
    return {t.untyped_storage().data_ptr() for t in _iter_tensors(obj)}


_QUERIES = frozenset("""dim size shape numel dtype device layout stride strides
                       storage_offset nbytes element_size get_device data_ptr
                       is_contiguous is_floating_point is_cuda is_leaf
                       requires_grad untyped_storage __get__""".split())
# These reach __torch_function__ like any other call -- ``BatchNorm2d`` asks its
# input ``dim()`` -- and cost a few hundred nanoseconds of host time while moving
# no bytes. They are not operators, and a family table would file them as one.
# ``__get__`` is that same category under a different name: a tensor property
# (``.data``, ``.T``, ``.dtype``) arrives as the descriptor's getter, which keeps
# its own name and loses the property's, and 7032 of the 7168 rows the first
# full CUDA sweep filed under "other" were exactly this.


def _materialised(obj: Any, already: set) -> tuple[float, set]:
    """Bytes in ``obj`` over storages not in ``already``, and which those were.

    :func:`output_bytes` in set form, so a frame can hand its allocations down
    to its parent and the parent does not bill the same buffer a second time.
    """
    fresh = set(already)
    total = 0.0
    for tensor in _iter_tensors(obj):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in fresh:
            continue
        fresh.add(pointer)
        total += float(storage.nbytes())
    return total, fresh - already


def output_bytes(obj: Any, operands: Any = ()) -> float:
    """Bytes the call materialised: its own storages, minus any it shares with an operand.

    ``transpose`` returns a tensor over the same storage, ``contiguous`` on an
    already contiguous tensor returns the tensor itself, and an in-place
    ``ReLU`` writes into and hands back its input. All three moved no data, and
    the fact they have in common is a buffer the call was already given -- which
    is why the comparison is against the operands rather than through ``_base``,
    a field an in-place op never sets. Comparing the *storage* pointer and not
    the tensor pointer is what makes a slice free as well: a view with a non-zero
    offset is still somebody else's allocation.
    """
    return _materialised(obj, _pointers(operands))[0]




def input_bytes(obj: Any) -> float:
    """Bytes handed to a call, views included: an operand has to be resident."""
    return sum(float(t.untyped_storage().nbytes()) for t in _iter_tensors(obj))


def request_size(inputs: Any) -> int:
    """Element count of a request: the ``s`` of the paper's ``<X, s, p>``."""
    if isinstance(inputs, dict):
        inputs = tuple(inputs.values())
    return sum(t.numel() for t in _iter_tensors(inputs))


class _Timer:
    """Spans that measure the device without draining the queue.

    Closing a span with a ``synchronize`` makes wall time equal device time, and
    makes every later kernel wait for the Python that follows. Measured on a V100
    over resnet50 at b=1, whose 158 leaf layers run an uninstrumented forward in
    4.65 ms: a synchronize per *layer* puts the sum of layer spans at 8.31 ms, and
    one per *operator* at 22.97 ms. An event pair per span records into the stream
    and leaves it running, so the same sum reads 7.66 ms; what remains is host time
    spent inside the layer, which a slice really does pay and :func:`calibrate`
    measures. The one synchronize of the pass happens in :meth:`finish`.

    ``watch=False`` is a timer that is not running. A pass that exists to read the
    allocator has no business recording events, and recording them is what would
    make the numbers of the pass that does time them wrong.
    """

    __slots__ = ("device", "stream", "watching")

    def __init__(self, device: torch.device, watch: bool = True) -> None:
        self.device = device if device.type == "cuda" else None
        self.watching = watch
        self.stream = (None if self.device is None
                       else torch.cuda.current_stream(device))

    def start(self) -> Any:
        if not self.watching:
            return None
        return self._event() if self.device is not None else time.perf_counter_ns()

    def end(self, start: Any) -> Any:
        if not self.watching:
            return None
        return self._event() if self.device is not None else time.perf_counter_ns()

    def elapsed(self, start: Any, stop: Any) -> float:
        """Nanoseconds between two handles; only valid after :meth:`finish`."""
        if start is None:
            return 0.0
        if self.device is None:
            return float(stop - start)
        return start.elapsed_time(stop) * 1e6

    def finish(self) -> None:
        if self.device is not None and self.watching:
            torch.cuda.synchronize(self.device)

    def _event(self) -> Any:
        event = torch.cuda.Event(enable_timing=True)
        # The stream is resolved once for the pass: record() without one looks it
        # up on every call, which measured 8.36 us against 2.61 us for the same
        # record handed the stream to write into.
        event.record(self.stream)
        return event


class _LiveBytes:
    """CUDA allocator reader; inert on the CPU, where no such counter exists.

    Inert unless ``watch`` is set, and that is not a performance nicety: in torch
    2.5 ``torch.cuda.memory_allocated`` is ``memory_stats()["...current"]``, and
    building that dictionary measured 98.8 us per call on this box -- three times
    the whole of a resnet50 layer. A pass that reads it cannot be the pass that
    times anything.
    """

    def __init__(self, device: torch.device, watch: bool = True) -> None:
        self.device = device if device.type == "cuda" else None
        self.reading = watch and self.device is not None

    def entry(self) -> float:
        if not self.reading:
            return 0.0
        torch.cuda.reset_peak_memory_stats(self.device)
        return float(torch.cuda.memory_allocated(self.device))

    def high_water(self, base: float) -> float | None:
        if not self.reading:
            return None
        return float(max(torch.cuda.max_memory_allocated(self.device) - base, 0.0))

    def peek(self) -> float | None:
        """The current high-water mark, leaving it standing for the next reader.

        Only ever read between one :meth:`entry` and the next, and the mark is
        monotone there, so the difference between two reads is what ran in
        between -- which is how an operator gets a peak without resetting the
        counter its layer is still reading.
        """
        if not self.reading:
            return None
        return float(torch.cuda.max_memory_allocated(self.device))


class _Span:
    """One open frame: where it began, what it opened, and what to bill.

    ``ns`` cannot be computed when a frame closes -- an event's elapsed time is
    only known once the device has run past it -- so the closed spans and their
    nesting are kept here and resolved in one pass by :meth:`_Probe.resolve`.
    """

    __slots__ = ("name", "start", "stop", "base", "layer", "child_bytes",
                 "children", "record", "inclusive")

    def __init__(self, name: str, start: Any, base: float = 0.0,
                 layer: int | None = None) -> None:
        self.name = name
        self.start = start
        self.stop: Any = None
        self.base = base
        self.layer = layer
        self.child_bytes: set = set()
        self.children: list["_Span"] = []
        self.record: Any = None
        self.inclusive = 0.0


class _Probe(TorchFunctionMode):
    """One instrumented forward, in one of three kinds.

    A span is either a leaf-module call, which carries an index into
    :attr:`layers`, or a torch function, which carries the bytes it materialised.
    Layers are charged their call plus the interval up to the next one, so the
    layer vector sums to the forward; operators are charged their own time minus
    the calls they opened, and their own bytes for the same reason: the buffer is
    the one the innermost call allocated, and the function that merely returned it
    again has already been billed.

    ``watch`` selects what a pass observes, and there are three because each of
    the three readings disturbs the other two:

    ``"layers"``
        Leaf hooks and event pairs only -- no ``__torch_function__`` mode, so no
        operator dispatch cost lands inside the spans being timed. This pass owns
        Eq. (4)'s ``t_l``.
    ``"operators"``
        Hooks, mode and events. This pass owns the operator rows: their self time,
        their bytes, and which layer they ran in. Its layer spans are wider than
        the ``"layers"`` pass's by the mode's own per-call cost, so they are
        discarded.
    ``"memory"``
        Hooks, mode and the allocator counters, no events. In torch 2.5 a counter
        read costs about as much host time as a small layer runs for
        (:class:`_LiveBytes`), which is the whole reason it is a pass of its own.
        This pass owns ``peak``, and :func:`attach_peaks` joins it to the rows the
        other two produced.
    """

    PASSES = ("layers", "operators", "memory")

    def __init__(self, device: torch.device, watch: str,
                 charge: OwnerCharge) -> None:
        super().__init__()
        if watch not in self.PASSES:
            raise ValueError("unknown pass %r, expected one of %s" % (watch, self.PASSES))
        self.charge = charge
        self.watch = watch
        self.namer = Namer()
        self.timer = _Timer(device, watch=watch != "memory")
        self.live = _LiveBytes(device, watch=watch == "memory")
        self.layers: list[LayerRecord] = []
        self.stack: list[_Span] = []
        self.spans: list[_Span] = []
        self.records: list[OpRecord] = []
        self.pending: list[OpRecord] = []
        self.hushed = False
        self.peak_seen: float = self.live.peek() or 0.0

    @contextlib.contextmanager
    def hush(self):
        """Stop observing while the probe measures itself.

        Sizing a call means calling torch functions on it, so without this the
        profile reports a tensor's ``numel`` as one of the service's operators:
        499 of resnet50's first 710 recorded calls were the probe's own.
        """
        was, self.hushed = self.hushed, True
        try:
            yield
        finally:
            self.hushed = was

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if self.hushed:
            return func(*args, **(kwargs or {}))
        span = _Span(getattr(func, "__name__", repr(func)), self.timer.start())
        self.stack.append(span)
        try:
            out = func(*args, **(kwargs or {}))
        except Exception:
            self.stack.pop()
            raise
        with self.hush():
            self.close(out, (args, kwargs or {}))
        return out

    def on_pre(self, module, args, kwargs):
        with self.hush():
            name = self.namer(module)
            self.charge.leaf(module, name)
            self.layers.append(LayerRecord(name, 0.0, 0.0, 0.0))
            index = len(self.layers) - 1
            base = self.live.entry()
            self.peak_seen = base
            self.claim_pending(index)
        self.stack.append(_Span(self.layers[index].name, self.timer.start(),
                                base, index))

    def on_post(self, module, args, kwargs, output):
        span = self.stack.pop()
        span.stop = self.timer.end(span.start)
        self.spans.append(span)
        layer = self.layers[span.layer]
        span.record = layer
        with self.hush():
            layer.returned = output_bytes(output, args)
            layer.peak = self.live.high_water(span.base)
        self.adopt(span)

    def close(self, result: Any, operands: Any) -> None:
        """Self bytes: the storages none of the calls this one opened allocated.

        Self time is not decided here; the span is parked on :attr:`spans` and
        resolved once the forward has run out.
        """
        span = self.stack.pop()
        span.stop = self.timer.end(span.start)
        self.spans.append(span)
        high = self.live.peek()
        peak = None
        if high is not None:
            peak, self.peak_seen = max(high - self.peak_seen, 0.0), high
        out, fresh = _materialised(result, _pointers(operands) | span.child_bytes)
        inp = input_bytes(operands)
        index = next((i for i in range(len(self.stack) - 1, -1, -1)
                      if self.stack[i].layer is not None), None)
        if (out or inp) and span.name not in _QUERIES:
            record = OpRecord(span.name, "" if index is None
                              else self.layers[self.stack[index].layer].name,
                              0.0, out, inp, peak)
            span.record = record
            self.records.append(record)
            if index is None:
                self.pending.append(record)
            else:
                self.layers[self.stack[index].layer].ops.append(record)
        self.adopt(span, fresh)

    def adopt(self, span: _Span, fresh: set = frozenset()) -> None:
        """Hand a closed span to whoever opened it: its bytes and its time."""
        if self.stack:
            self.stack[-1].child_bytes |= fresh
            self.stack[-1].children.append(span)

    def resolve(self) -> None:
        """Device time for every span, innermost calls first.

        One synchronize for the whole pass: an event's elapsed time is only known
        once the stream has run past it. Spans were appended as they closed, so a
        child is always earlier in the list than the parent that opened it and the
        subtraction is a single pass.
        """
        self.timer.finish()
        for span in self.spans:
            span.inclusive = self.timer.elapsed(span.start, span.stop)
        for span in self.spans:
            if span.record is None or span.layer is not None:
                continue
            # An operator is charged only itself, or a composite and the aten call
            # under it would each be billed for the same kernels.
            span.record.ns = max(span.inclusive - sum(child.inclusive
                                                      for child in span.children),
                                 0.0)
        self.bill_gaps()

    def bill_gaps(self) -> None:
        """Charge the time between two leaf modules to the one that ran first.

        Hand-written services do work outside every leaf module -- ``x @ self.adj``
        in :class:`~slicer.models.custom.GraphConv`, the attention mask in BERT --
        and on a measured service that work is most of the wall clock: without it
        the four leaf layers of gcn1 account for 0.16 ms of a 0.54 ms forward. An
        operator pass names those calls, but the pass that times layers deliberately
        does not see operators, so what it bills instead is the interval between two
        spans -- the same event arithmetic, one level up. Summed with the spans this
        is the whole instrumented forward, which is why :func:`scale` is left with
        only the profiler's own overhang to remove.
        """
        ordered = [span for span in self.spans if span.layer is not None]
        for index, span in enumerate(ordered):
            gap = 0.0
            if index + 1 < len(ordered):
                gap = max(self.timer.elapsed(span.stop, ordered[index + 1].start), 0.0)
            span.record.gap = gap
            span.record.ns = span.inclusive + gap

    def claim_pending(self, index: int) -> None:
        """Bill operators that ran outside every leaf module to their consumer.

        Hand-written services do work at module level -- ``relu(A @ self.linear(x))
        `` in :class:`~slicer.models.custom.GraphConv` -- and the graph has no node
        for a non-leaf module. :mod:`slicer.models.graphify` resolves the same
        situation structurally by contracting such an operator into its consumer,
        and in execution order the consumer is the next leaf module to run. What
        that cannot see is an operator whose result outlives the next call, where
        the time lands one node early; the bytes are unaffected.
        """
        layer = self.layers[index]
        for record in self.pending:
            record.layer = layer.name
            layer.ops.append(record)
        layer.billed += len(self.pending)
        self.pending = []

    def flush(self) -> None:
        """Operators after the last leaf call belong to it rather than nowhere."""
        if self.pending and self.layers:
            self.claim_pending(len(self.layers) - 1)

    def settle_weights(self) -> None:
        """Put every container's bytes on the leaf call that had to hold them.

        :class:`~slicer.models.graphify.OwnerCharge` can only tell where a hidden
        weight ran once the forward is over, so :meth:`on_pre` leaves the field
        empty and this fills it in.
        """
        for layer in self.layers:
            layer.params = self.charge.credits.get(layer.name, 0.0)


def measure_forward(model: torch.nn.Module, inputs: Any, device: str = "cpu",
                    expect_order: Sequence[str] | None = None, watch: str = "operators"):
    """One instrumented forward: ``(layers, operators)`` in execution order.

    ``watch`` picks the pass; see :class:`_Probe` for what each one owns and why
    they cannot share a forward. A ``"layers"`` pass reports empty ``ops`` lists
    and zero peaks, a ``"memory"`` pass reports zero durations.
    """
    dev = torch.device(device)
    model.eval().to(dev)
    inputs = move_tensors(inputs, device)
    probe = _Probe(dev, watch, OwnerCharge(model))
    probe.namer = Namer(module_paths(model))
    handles = install_leaf_hooks(model, probe.on_pre, probe.on_post, probe.charge)
    # Installing the mode is not free even for the calls it then declines to
    # record, so the pass that times layers stays out of it entirely.
    observing = contextlib.nullcontext() if watch == "layers" else probe
    try:
        with observing, torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    probe.resolve()
    probe.flush()
    probe.settle_weights()
    names = [layer.name for layer in probe.layers]
    # Compared as lists: a caller holding a tuple would otherwise fail the check
    # without any name actually differing, and _first_difference has nothing to
    # report for that, which reads like shape-dependent control flow and is not.
    if expect_order is not None and names != list(expect_order):
        raise RuntimeError(
            "layer names diverged from the structure pass at %s; the join between"
            " measured profiles and the graph is only valid while every module is"
            " called the same number of times, which shape-dependent control flow"
            " breaks" % _first_difference(expect_order, names))
    return probe.layers, probe.records


def _first_difference(expected: list[str], actual: list[str]) -> str:
    for index, name in enumerate(expected):
        if index >= len(actual) or actual[index] != name:
            return "%s vs %s (position %d)" % (name, actual[index:index + 1], index)
    return "extra %s" % actual[len(expected):len(expected) + 1]


class _Null(torch.nn.Module):
    """A layer that runs no operator: the probe's own cost, isolated."""

    def forward(self, x):
        return x


class _OneOp(torch.nn.Module):
    """A layer that runs exactly one operator."""

    def forward(self, x):
        return torch.relu(x)


def calibrate(device: str = "cpu", spans: int = 64, repeats: int = 5) -> tuple[float, float]:
    """Cost of the instrumentation itself, per layer span and per operator span.

    A span is not free even when it only records two events: the probe pays a
    hook pair and the byte accounting on the way out for a layer, and two
    ``__torch_function__`` dispatches plus that same accounting for an operator.
    On a service whose layers are a few tens of microseconds of kernel apiece that
    host time lands *inside* the span it is timing -- the device runs out of
    submitted work and waits for the next one -- so it has to be subtracted rather
    than ignored. Stacking no-op layers measures the first term in the pass that
    reports durations, and stacking one-op layers measures the second as the
    difference in the pass that reports operators. :func:`correct` does the
    subtracting so the partitioner sees the service rather than the profiler.

    The layer term is measured over a whole stack, gaps included, which is what
    makes it the right thing to take off a :meth:`_Probe.bill_gaps` span: a stack
    of no-op layers has no work between its modules, so whatever the pass bills
    there is the boundary itself.
    """
    import torch.nn as nn

    def stack(factory, watch: str) -> float:
        model = nn.Sequential(*[factory() for _ in range(spans)]).to(device)
        inputs = (torch.zeros(1, device=torch.device(device)),)
        runs = []
        for index in range(3 + repeats):
            layers, _ = measure_forward(model, inputs, device=device, watch=watch)
            if index >= 3:
                runs.append([layer.ns for layer in layers])
        return _median([value for run in runs for value in run])

    # The layer term comes from the pass that reports durations and the operator
    # term from the pass that reports operators, or the subtraction removes a
    # cost the span never paid and leaves one it did.
    return (stack(_Null, "layers"),
            max(stack(_OneOp, "operators") - stack(_Null, "operators"), 0.0))


def correct(medians: dict[str, LayerRecord], per_layer: float,
            per_op: float) -> dict[str, LayerRecord]:
    """Subtract the measured instrumentation cost from a median profile."""
    for record in medians.values():
        for operator in record.ops:
            operator.ns = max(operator.ns - per_op, 0.0)
        record.ns = max(record.ns - per_layer - per_op * len(record.ops), 0.0)
    return medians


def scale(medians: dict[str, LayerRecord], forward_ns: float) -> float:
    """Stretch the layer vector until it sums to the forward, and report by how much.

    :meth:`_Probe.bill_gaps` already hands every interval between two leaf modules
    to the module that preceded it, so the spans sum to the *instrumented* forward.
    What remains is the difference between that and the uninstrumented one, and it
    pulls in both directions: on a launch-bound service the probe's own Python runs
    inside the windows it is timing, so the sum is high (resnet50 at b=1: 1.40x),
    and on a GPU-bound one the queue runs ahead of the host and the sum is low. Both
    are corrections to a single quantity that is measured exactly -- an uninstrumented
    forward -- so the vector is scaled onto it. Each layer keeps its share of the
    forward, which is what the partitioner needs to price a cut, and the profiler
    stops claiming that a sum of spans is a wall clock.
    """
    total = sum(record.ns for record in medians.values())
    if total <= 0.0 or forward_ns <= 0.0:
        return 1.0
    factor = forward_ns / total
    for record in medians.values():
        record.ns *= factor
    return factor


def median_repeats(runs: list[list[LayerRecord]]) -> dict[str, LayerRecord]:
    """Per-layer median over repeats; the shared box makes a single pass noise.

    Memory is deterministic and so needs no statistic, but time is not: this
    machine is multi-tenant, and a pre-empted forward would otherwise set the
    profile the partitioner consumes.
    """
    for run in runs[1:]:
        if [r.name for r in run] != [r.name for r in runs[0]]:
            raise RuntimeError("repeat forwards disagreed on the layer sequence")
    medians: dict[str, LayerRecord] = {}
    for column in range(len(runs[0])):
        rows = [run[column] for run in runs]
        peaks = [row.peak for row in rows if row.peak is not None]
        medians[rows[0].name] = LayerRecord(
            rows[0].name, rows[0].params,
            _median([r.ns for r in rows]), _median([r.returned for r in rows]),
            _merge_ops(rows), _median(peaks) if peaks else None, rows[0].billed,
            gap=_median([r.gap for r in rows]))
    return medians


def attach_peaks(medians: dict[str, LayerRecord],
                 source: dict[str, LayerRecord]) -> dict[str, LayerRecord]:
    """Copy the measured footprints of the memory pass onto the timed one.

    Matched by position, and the position is checked rather than trusted: two
    forwards of the same module agree on its layer sequence and on the operators
    of each layer unless something shape-dependent is going on, and a peak that
    lands on the wrong layer's duration is worse than no peak at all.
    """
    for name, record in medians.items():
        other = source.get(name)
        if other is None or [r.op for r in other.ops] != [r.op for r in record.ops]:
            raise RuntimeError(
                "the memory pass and the timing pass disagree about %s: %s vs %s"
                % (name, [r.op for r in record.ops],
                   "<missing>" if other is None else [r.op for r in other.ops]))
        record.peak = other.peak
        for timed, measured in zip(record.ops, other.ops):
            timed.peak = measured.peak
    return medians


def _merge_ops(rows: list[LayerRecord]) -> list[OpRecord]:
    """Median time per operator slot, keeping the first pass's identity."""
    groups: dict[tuple[int, str], list[OpRecord]] = {}
    for row in rows:
        for position, record in enumerate(row.ops):
            groups.setdefault((position, record.op), []).append(record)
    out: list[OpRecord] = []
    for (position, _), bucket in sorted(groups.items()):
        head = bucket[0]
        peaks = [r.peak for r in bucket if r.peak is not None]
        out.append(OpRecord(head.op, head.layer, _median([r.ns for r in bucket]),
                            head.out, head.inp, _median(peaks) if peaks else None))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0 if ordered else 0.0
