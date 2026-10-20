"""Run the slices of a plan as functions, on a machine that has no OpenFaaS.

Sec. II-C's execution model is one function per vertical slice: each has its own
vCPU count, its own memory quota, its own server, and the only thing they share is
the message a cut leaves behind. The paper gets that from fourteen servers and a
queue daemon; this module gets it from processes, and the four knobs map as follows.

``eta``
    ``os.sched_setaffinity`` down to that many cores plus ``torch.set_num_threads``,
    so a slice with two vCPUs cannot borrow the box it is sitting on.
``alloc``
    ``RLIMIT_AS`` lowered to the process's own base plus the quota. The base is
    measured in the child rather than assumed: every function pays for the
    framework it imports, and on a platform that hands out 1792 MB per vCPU that
    floor is most of a small function's quota. The ceiling goes on *after* the child
    has recorded its program and released the weights above its own reach, because
    recording is a service-wide act that no single slice's quota can contain.
placement
    two slices on one server talk through ``/dev/shm``, which is the real thing COM
    claims. A pair on different servers cannot be given a real NIC here, so their
    hop is the same shared-memory copy with a floor of ``bytes / bandwidth`` from
    Table II put underneath it: the consumer pads a wait it already pays, and
    :attr:`Round.link_s` says how much of the total that added. The copy is
    measured and the link is a lower bound, so the two are reported apart.
invocation
    :meth:`Deployment.load` pushes a campaign through the service with a fixed
    number of requests in flight, which is the closed loop the concurrency decision
    settled on: the offered load is whatever the service can absorb, and the queueing
    a saturated service shows up as is the tail Fig. 10b measures. The paper's
    diurnal PAI arrival trace (Fig. 9) is not generated here, so throughput is
    reported rather than imposed.

Two rules about lanes are worth stating because nothing else in the file explains
them. Slices are **relabelled** into a topological order of the dependency graph
before anything starts, and a worker waits on its predecessors in increasing rank:
that is what makes a non-contiguous plan -- a slice feeding two later ones, a
residual add crossing a cut against layer order -- safe to run, since no slice ever
blocks on a higher rank, rank 0 has nothing to wait for by construction, and
progress follows by induction. And **request values ride the same lanes** as
activations, from the entry slice outwards: nobody *wrote* a request tensor, so
:meth:`~slicer.faas.program.Program.slices` bills no lane for one, yet a slice far
down can read it, and only the slice the controller hands the request to has it.

Run: ``python -m slicer.faas.runtime --model gcn1 --slices 2`` for a smoke test.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import multiprocessing as mp
import os
import resource
import shutil
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence
import traceback

import torch

from ..communicate import tensor
from ..communicate.channel import Arena, open_shm_fifo, tmpfs_root
from ..models import zoo
from ..models.graphify import move_tensors
from ..partition.cost import PlatformConfig
from .program import Program, record, _fill, _iter_tensors

MB = 2 ** 20
PAGE = float(os.sysconf("SC_PAGE_SIZE"))

#: Written at the head of every lane payload: the monotonic instant of the send.
#: Shared clock across processes, so a consumer can tell a hop from a queue.
_STAMP = struct.Struct("!Q")

#: A lane message that carries no values: the request is over and the service closes.
STOP: dict = {}

#: How long a slice may take to answer, and how often the parent looks to see whether
#: it still can. :meth:`Deployment._await` needs both: a child killed by the kernel
#: never gets to send the error, and an unbounded ``recv()`` on its pipe is a sweep
#: that stops producing output without ever failing.
_WAIT_S = 1800.0
_POLL_S = 0.5


@dataclass(frozen=True)
class Slice:
    """One function as the deployment states it: its layers, its knobs, its server."""

    index: int
    layers: tuple[str, ...]
    eta: int = 1
    alloc_mb: float = 0.0
    server: int = 0


@dataclass(frozen=True)
class Lane:
    """A named producer-to-consumer queue, with the slots it owns and its link.

    Ranks, not plan indices: a lane is addressed by the order the slices start in,
    which is the only order in which waiting on a peer is safe.
    """

    producer: int
    consumer: int
    slots: tuple[int, ...]
    link_bytes_per_s: float = 0.0  # zero when the two ends share a server


@dataclass(frozen=True)
class _Job:
    """Everything a child needs to become one slice, and nothing it cannot pickle.

    A spawned process inherits no descriptor that ``multiprocessing`` did not dupe
    for it, so the lanes are named by path rather than handed over open.
    """

    key: str
    device: str
    batch: int | None
    seed: int
    groups: tuple[tuple[str, ...], ...]
    index: int  # position in the plan, which is what the partitioner numbered
    rank: int  # topological position, which is what names the lanes
    entry: bool  # the one rank the controller hands requests to
    terminal: bool  # the one rank whose last step produces the program's outputs
    inbound: tuple[tuple[int, tuple[int, ...]], ...]  # (peer rank, keys to await)
    outbound: tuple[tuple[int, tuple[int, ...]], ...]  # (peer rank, keys to send)
    lanes: tuple[Lane, ...]
    arena_path: str
    arena_slots: int
    slot_bytes: int
    lane_root: str
    tag: str
    cores: tuple[int, ...]
    eta: int
    quota_bytes: float
    #: The two ends of one pipe, kept apart because a one-slice plan is its own entry
    #: *and* its own terminal: handed a single field, that process would read a
    #: connection it only opened for writing.
    feedback_in: Any = None   # entry: a request has left the service
    feedback_out: Any = None  # terminal: this request finished my last step


@dataclass
class Round:
    """One request through the chain, and what it cost."""

    seconds: float
    result: Any = None
    per_slice: tuple[dict, ...] = ()
    compute_s: float = 0.0  # summed spans: the work the split version did
    makespan_s: float = 0.0  # the longest single slice's span
    wire_bytes: float = 0.0
    transport_s: float = 0.0  # the transports' own time: syscalls and memcpys
    link_s: float = 0.0  # what the modelled link added on top

    def summary(self) -> str:
        return ("%.3f ms  %d slices  spans=%s  wire=%.2f MB in %.3f ms (+%.3f ms link)"
                % (self.seconds * 1e3, len(self.per_slice),
                   " ".join("%.2f" % (row["span_s"] * 1e3) for row in self.per_slice),
                   self.wire_bytes / MB, self.transport_s * 1e3, self.link_s * 1e3))


@dataclass
class Load:
    """A closed-loop campaign: what each of its requests took, and the tail of that."""

    latencies_s: tuple[float, ...]
    per_slice: tuple[dict, ...] = ()
    seconds: float = 0.0  # wall time of the whole campaign, startup excluded
    wire_bytes: float = 0.0

    @property
    def throughput(self) -> float:
        return len(self.latencies_s) / self.seconds if self.seconds else 0.0

    def percentile(self, q: float) -> float:
        """The ``q``-th percentile in seconds, by nearest rank: Fig. 10b's measure."""
        if not self.latencies_s:
            return 0.0
        ordered = sorted(self.latencies_s)
        position = min(len(ordered) - 1, max(0, int(round(q / 100.0 * len(ordered))) - 1))
        return ordered[position]

    def summary(self) -> str:
        return ("%d requests  mean %.3f ms  p50 %.3f  p95 %.3f  p99 %.3f  %.2f rps"
                % (len(self.latencies_s),
                   1e3 * sum(self.latencies_s) / max(1, len(self.latencies_s)),
                   1e3 * self.percentile(50), 1e3 * self.percentile(95),
                   1e3 * self.percentile(99), self.throughput))


def _return_to_os() -> None:
    """Hand back what the interpreter dropped *and* what glibc kept after that.

    Releasing a slice's weights is two steps, not one. Clearing the parameter tables
    only drops the last reference, and a freed block is not the operating system's
    again until glibc unmaps it -- which it does only at the top of an arena, so a
    block taken from the middle of a model load stays resident however dead it is.
    Without both, a function billed at 900 MB holds the whole service, and
    ``RLIMIT_AS`` then bounds an address space nobody is using instead of the memory
    somebody is.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # a libc without the symbol costs fidelity, not correctness
        pass


def _resident() -> tuple[float, float]:
    """This process's resident and virtual bytes: the base a quota is added to."""
    try:
        with open("/proc/self/statm") as handle:
            fields = handle.read().split()
        return float(fields[1]) * PAGE, float(fields[0]) * PAGE
    except (OSError, IndexError, ValueError):
        return (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0,
                float("inf"))


class _LaneEndpoint:
    """One end of a :class:`Lane`, telling the transport's time from the link's.

    A message carries the monotonic instant it was written. ``CLOCK_MONOTONIC`` is
    the same clock in every process on the box, so the consumer can tell how long
    the payload has been in flight from how long *it* has been blocked -- and the
    two are not the same thing: a slice that waits 200 ms for a producer that was
    computing has seen a 200 ms wait and a 0 ms hop. Without the stamp, an emulated
    link would always look free, because the queueing swallows it.

    The floor is then applied to the flight time, which is what a NIC would have
    cost, and only the shortfall is slept: ``link_s`` is the extra the modelled
    server-to-server link added, never less than zero and never more than the bytes
    divided by the rate.
    """

    def __init__(self, lane: Lane, sender=None, receiver=None) -> None:
        self.lane = lane
        self.sender = sender
        self.receiver = receiver
        self.moved_bytes = 0.0
        self.write_s = 0.0
        self.read_s = 0.0
        self.wait_s = 0.0
        self.link_s = 0.0

    def send(self, values: dict[int, Any]) -> None:
        payload = _STAMP.pack(time.monotonic_ns()) + tensor.pack(values)
        self.moved_bytes += len(payload)
        started = time.perf_counter()
        self.sender.send(payload)
        self.write_s += time.perf_counter() - started

    def recv(self) -> dict[int, Any]:
        started = time.perf_counter()
        payload = self.receiver.recv()
        self.wait_s += time.perf_counter() - started
        rate = self.lane.link_bytes_per_s
        if rate:
            in_flight = (time.monotonic_ns() - _STAMP.unpack_from(payload, 0)[0]) / 1e9
            floor = len(payload) / rate
            if floor > in_flight:  # the copy arrived early; the link would not have
                time.sleep(floor - in_flight)
                self.link_s += floor - in_flight
        unpacked = time.perf_counter()
        values = tensor.unpack(payload[_STAMP.size:])
        self.read_s += time.perf_counter() - unpacked
        return values

    def close(self) -> None:
        for endpoint in (self.sender, self.receiver):
            if endpoint is not None:
                endpoint.close()


def _lane_plan(program: Program, cases: Sequence[Any], servers: Sequence[int],
               link_bytes_per_s: float) -> tuple[dict, dict]:
    """Which values each lane carries, keyed by (producer, consumer) plan index.

    Activations come from :attr:`~slicer.faas.program.SliceProgram.outbound`. Request
    values are the case Sec. II-C never spells out: a slice that reads the request
    without writing it has no producer to hear from, so the entry slice -- the one
    the controller feeds -- forwards it, and the lane appears anyway.
    """
    entry = min((case.index for case in cases),
                key=lambda index: cases[index].steps[0] if cases[index].steps
                else 1 << 30)
    lanes: dict[tuple[int, int], set[int]] = {}
    rates: dict[tuple[int, int], float] = {}

    def add(producer: int, consumer: int, keys: Iterable[int]) -> None:
        if producer == consumer:
            return
        lane = (producer, consumer)
        lanes.setdefault(lane, set()).update(keys)
        if servers[producer] != servers[consumer]:
            rates[lane] = link_bytes_per_s

    for case in cases:
        if case.index == entry:
            continue
        arrived = {key for _, keys in case.inbound for key in keys}
        wanted = [key for key in case.request if key in program.inputs and key not in arrived]
        if wanted:
            add(entry, case.index, wanted)
    for case in cases:
        for peer, keys in case.outbound:
            if keys:
                add(case.index, peer, keys)  # the peer is who receives, not who sent
    return lanes, rates


def _rank_order(cases: Sequence[Any], lanes: dict[tuple[int, int], set[int]]
                ) -> dict[int, int]:
    """The plan renumbered so every producer precedes the slices it feeds.

    Rank 0 is the entry: it owns the program's first step, so it reads the request
    and nothing anybody else wrote, and every lane carrying a request value starts
    at it. That is what makes the wait order safe rather than merely acyclic.
    """
    feeds: dict[int, set[int]] = {case.index: set() for case in cases}
    for (producer, consumer), keys in lanes.items():
        if keys:
            feeds[consumer].add(producer)
    entry = min((case.index for case in cases),
                key=lambda index: cases[index].steps[0] if cases[index].steps
                else 1 << 30)
    if feeds[entry]:
        # The entry is where the request enters the service, so a slice that has to
        # wait before it can compute anything cannot be it. A plan can still be a DAG
        # and hit this: it means the layer order and the slice order disagree, which
        # is the one thing the rank relabelling cannot repair. Saying so here is what
        # keeps a sweep from hanging on a deadlocked handshake -- a straddle is only
        # deployable when every producer of the moved node sits in an earlier slice,
        # which is what :func:`slicer.partition.baselines._straddle` checks.
        raise ValueError("slice %d owns the first step but waits on slices %s, so no"
                         " request could be answered"
                         % (entry, sorted(feeds[entry])))
    order: list[int] = [entry]
    placed = {entry}
    while len(order) < len(cases):
        for case in cases:
            if case.index in placed or not feeds[case.index] <= placed:
                continue
            order.append(case.index)
            placed.add(case.index)
            break
        else:
            stuck = sorted(index for index in feeds if index not in placed)
            raise ValueError("slices %s wait on each other in a ring" % stuck)
    return {index: rank for rank, index in enumerate(order)}


def _serve(job: _Job, control) -> None:
    """One slice's whole life: become the function, then answer requests.

    The wrapper exists because a slice that dies is a *result*, not a hang: an
    experiment that walks twelve services cannot be resumed by hand when a quota is
    refused, and the default failure mode of a pipelined parent is to wait forever on
    a child that is already gone.
    """
    try:
        _run_slice(job, control)
    except BaseException:  # noqa: BLE001 - reported, then the process exits anyway
        try:
            control.send(("error", traceback.format_exc()))
        except OSError:
            pass


def _run_slice(job: _Job, control) -> None:
    torch.manual_seed(job.seed)
    if job.cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(job.cores))
    torch.set_num_threads(max(1, job.eta))

    model, inputs = zoo.load(job.key, device=job.device, batch=job.batch, seed=job.seed)
    program = record(model, inputs, name=job.key, device=job.device)
    case = program.slices(job.groups)[job.index]
    # A recorded step holds the tensors it read *by value*, so every slice's steps pin
    # every weight in the service; this child runs only its own slice, and the rest of
    # the capture is what keeps the unreleased weights alive.
    program.steps = ()
    freed = program.release(case.keeps)
    _return_to_os()
    resident, virtual = _resident()
    if job.quota_bytes:
        # Only lowering a soft limit is free, and this is the last moment the child
        # has room: the recording above needed the whole service's activations.
        #
        # The ceiling is the child's *virtual* size plus the quota, not its resident
        # set plus the quota: RLIMIT_AS bounds address space, and a process that has
        # imported torch has reserved several times what it holds, so a limit drawn
        # from RSS refuses the next mmap before the function has done any work.
        resource.setrlimit(resource.RLIMIT_AS,
                           (int(virtual + job.quota_bytes), resource.RLIM_INFINITY))

    arena = Arena.attach(job.arena_path, slots=job.arena_slots,
                         slot_bytes=job.slot_bytes)
    lanes: dict[tuple[int, int], _LaneEndpoint] = {}
    for lane in job.lanes:
        if job.rank not in (lane.producer, lane.consumer):
            continue
        sender, receiver = open_shm_fifo(
            arena, "%s-s%d-c%d" % (job.tag, lane.producer, lane.consumer),
            lane.slots, root=job.lane_root)
        lanes[(lane.producer, lane.consumer)] = _LaneEndpoint(
            lane, sender if lane.producer == job.rank else None,
            receiver if lane.consumer == job.rank else None)

    out = {rank: tuple(keys) for rank, keys in job.outbound}
    control.send(("ready", {"rank": job.rank, "index": job.index, "entry": job.entry,
                            "eta": job.eta, "cores": len(job.cores),
                            "base_bytes": resident, "virtual_bytes": virtual,
                            "quota_bytes": job.quota_bytes,
                            "freed_bytes": freed, "resident_bytes": _resident()[0],
                            "layers": len(case.layers), "steps": len(case.steps)}))
    rows: list[dict] = []
    latencies: list[float] = []  # the entry's, one per request that left the service
    #: Wall seconds per :meth:`Deployment.sweep` campaign, so a service that ran four
    #: windows reports four throughputs rather than one blended figure.
    windows: list[float] = []
    admitted: deque = deque()  # monotonic instants of the entry's live requests
    state: dict = {}

    def close_down() -> None:
        for rank in sorted(out):
            lanes[(job.rank, rank)].send(STOP)

    def serve_one(payload: Any) -> bool:
        """One request through this slice. ``False`` once a lane says the service is done."""
        clock = time.perf_counter()
        env: dict[int, Any] = {}
        if payload is not None:
            env.update(dict(zip(program.inputs,
                                _iter_tensors(move_tensors(payload, job.device)))))
        for peer, _ in job.inbound:
            values = lanes[(peer, job.rank)].recv()
            if not values:
                return False
            env.update(values)
        busy = time.perf_counter()
        case.run(env)
        busy = time.perf_counter() - busy
        span = time.perf_counter() - clock
        for rank in sorted(out):
            lanes[(job.rank, rank)].send({key: env[key] for key in out[rank]})
        mine = [lane for lane in lanes.values() if lane.sender is not None]
        rows.append({"span_s": span, "busy_s": busy,
                     # Bytes are counted where they are written; a lane's two ends
                     # each pay their own copy, and the reader's wait is the only
                     # place the modelled link's cost can show up.
                     "moved_bytes": sum(lane.moved_bytes for lane in mine),
                     "transport_s": sum(lane.write_s + lane.read_s
                                        for lane in lanes.values()),
                     "link_s": sum(lane.link_s for lane in lanes.values()),
                     "wait_s": sum(lane.wait_s for lane in lanes.values()),
                     "resident_bytes": _resident()[0]})
        for lane in lanes.values():
            lane.moved_bytes = 0.0
            lane.write_s = lane.read_s = lane.wait_s = lane.link_s = 0.0
        if case.result:
            state["result"] = _fill(program.template, env)
        if job.terminal and job.feedback_out is not None:
            job.feedback_out.send("done")
        return True

    def campaign(count: int, window: int, payload: Any) -> None:
        """``count`` requests, never more than ``window`` admitted but unfinished.

        The window is the whole of the concurrency model: once it is full, the next
        admission waits for a completion, and that wait is the backpressure a loaded
        service shows as tail latency. A queue-free closed loop, in other words --
        which is also why :attr:`Load.latencies` starts at admission and not at
        arrival, since arrival pacing *is* the window here.

        The stamp goes on *before* the entry computes its own step. The entry is a
        function, not a load generator that stands outside the service, so a request
        that is measured from after that step has finished has had its whole first hop
        removed from its own latency -- which on a one-slice plan reported 0.016 ms for
        a 1.55 ms request.
        """
        issued = 0
        started = time.perf_counter()
        while issued < count or admitted:
            while issued < count and len(admitted) < window:
                admitted.append(time.monotonic_ns())
                serve_one(payload)  # the entry waits on no lane, so no STOP can come
                issued += 1
            job.feedback_in.recv()
            latencies.append((time.monotonic_ns() - admitted.popleft()) / 1e9)
        windows.append(time.perf_counter() - started)

    while True:
        if job.entry:
            kind, *rest = control.recv()
            if kind == "stop":
                close_down()
                break
            if kind == "load":
                campaign(int(rest[0]), max(1, int(rest[1])), rest[2])
            elif not serve_one(rest[0]):
                break
        elif not serve_one(None):
            close_down()
            break

    control.send(("report", {
        "rank": job.rank, "index": job.index, "entry": job.entry,
        "terminal": job.terminal, "eta": job.eta, "cores": len(job.cores),
        "quota_bytes": job.quota_bytes, "freed_bytes": freed,
        "layers": len(case.layers), "steps": len(case.steps), "requests": len(rows),
        # One entry per request, never a mean: a campaign's tail is the thing Fig. 10b
        # measures, and averaging here would put the loss where the measurement is.
        **{field: [row[field] for row in rows] for field in
           ("span_s", "busy_s", "wait_s", "transport_s", "link_s", "moved_bytes",
            "resident_bytes")},
        "latencies_s": latencies, "campaign_s": windows, "result": state.get("result")}))
    # The report is waited for, not just sent: a CPU tensor leaves a process as a file
    # descriptor served by *this* process's resource sharer, so exiting here would
    # pull the socket out from under a parent that has read the header but not the
    # bytes, and its recv() fails with a missing address rather than a short read.
    try:
        control.recv()
    except (EOFError, OSError):
        pass
    for end in (job.feedback_in, job.feedback_out):
        if end is not None:
            end.close()
    for lane in lanes.values():
        lane.close()
    arena.destroy()


#: What a slice reports once per request, so a campaign boundary is a position in
#: each of these rather than a message the slice has to be told about.
_PER_REQUEST = ("span_s", "busy_s", "wait_s", "transport_s", "link_s", "moved_bytes",
                "resident_bytes")


def _window(row: dict, bounds: slice) -> dict:
    """One slice's report, cut down to the requests of one campaign."""
    cut = dict(row)
    for field_name in _PER_REQUEST:
        cut[field_name] = row[field_name][bounds]
    cut["requests"] = len(cut["span_s"])
    return cut


@dataclass
class Deployment:
    """A plan laid out as processes: the lanes between them and the caller's handle."""

    key: str
    slices: tuple[Slice, ...]
    ranks: dict[int, int]
    lanes: dict[tuple[int, int], tuple[int, ...]]
    open_lanes: tuple[Lane, ...] = ()
    arena: Arena | None = None
    lane_root: str = ""
    tag: str = ""
    entry_rank: int = 0
    terminal_index: int = 0  # the slice whose last step yields the program's outputs
    queue_depth: int = 2  # messages a lane can hold, so the cap :meth:`load` respects
    program: Any = None  # the parent's recording, which :meth:`baseline` reuses
    #: The request the parent loaded to build the recording. :meth:`load` takes its
    #: payload from the caller, and an experiment that hands it a second
    #: :func:`slicer.models.zoo.load` would pay for a 10 GB model twice.
    inputs: Any = None
    processes: dict = field(default_factory=dict)  # rank -> process
    controls: dict = field(default_factory=dict)  # rank -> Connection
    _inbox: dict = field(default_factory=dict)  # rank -> message read early
    ready: tuple = ()
    drained: bool = False  # one campaign per deployment: the reports have been taken
    closed: bool = False

    @classmethod
    def build(cls, key: str, groups: Sequence[Sequence[str]],
              etas: Sequence[int] | None = None, servers: Sequence[int] | None = None,
              allocs: Sequence[float] | None = None, device: str = "cpu",
              batch: int | None = None, seed: int = 0,
              platform: PlatformConfig | None = None, root: str | None = None,
              mb_per_vcpu: float = 1792.0, queue_depth: int = 2) -> "Deployment":
        """Cut the service, name every queue, and size the arena over the widest lane.

        A slot holds one message, and a message is everything one producer sends one
        consumer for this request, so the arena is sized by a lane's total rather than
        by its largest single value: a slice ships its frontier in one write.

        ``queue_depth`` is how many messages a lane can carry at once, and so caps the
        concurrency :meth:`load` may ask for: a slot is only released after the reader
        has copied it out, so a lane with two slots cannot hold four requests.

        ``allocs`` is what the plan says a slice costs; where it is silent the
        deployment gives the smallest quota that still affords that many vCPUs,
        which is Eq. (6)'s constraint read in the other direction.
        """
        plan = tuple(tuple(names) for names in groups)
        platform = platform or PlatformConfig()
        count = len(plan)
        etas = [max(1, int(etas[index])) if etas and index < len(etas) else 1
                for index in range(count)]
        servers = ([int(servers[index]) if index < len(servers) else 0
                    for index in range(count)] if servers else [0] * count)
        allocs = ([float(allocs[index]) if index < len(allocs) else etas[index] * mb_per_vcpu
                   for index in range(count)] if allocs
                  else [eta * mb_per_vcpu for eta in etas])
        slices = tuple(Slice(index, plan[index], etas[index], allocs[index], servers[index])
                       for index in range(count))

        model, inputs = zoo.load(key, device=device, batch=batch, seed=seed)
        program = record(model, inputs, name=key, device=device)
        cases = program.slices(plan)
        raw, rates = _lane_plan(program, cases, servers,
                                platform.crosshost_bandwidth_bytes_per_s)
        ranks = _rank_order(cases, raw)
        fed = {ranks[consumer] for (producer, consumer), keys in raw.items() if keys}
        asleep = sorted(index for index, rank in ranks.items() if rank and rank not in fed)
        if asleep:
            # A slice with nothing to wait for would run once and then block on a
            # lane nobody will ever write, so the deployment says so now.
            raise ValueError("slice %s is fed by no lane, so no request would wake it"
                             % asleep[0])
        lanes = {(ranks[producer], ranks[consumer]): keys
                 for (producer, consumer), keys in raw.items()}
        rates = {(ranks[producer], ranks[consumer]): rate
                 for (producer, consumer), rate in rates.items()}
        widest = max((sum(program.sizes.get(key, 0.0) for key in keys)
                      for keys in lanes.values()), default=4096.0)
        depth = max(2, int(queue_depth))
        specs = tuple(Lane(lane[0], lane[1],
                           tuple(range(depth * position, depth * position + depth)),
                           rates.get(lane, 0.0))
                      for position, lane in enumerate(sorted(lanes)))
        root = root or tmpfs_root()
        terminal = max((case.index for case in cases if case.result),
                       key=lambda index: ranks[index], default=max(ranks.values()))
        return cls(key=key, slices=slices, ranks=ranks, program=program,
                   inputs=inputs,
                   lanes={lane: tuple(sorted(keys)) for lane, keys in lanes.items()},
                   open_lanes=specs, entry_rank=min(ranks.values()),
                   terminal_index=terminal, queue_depth=depth,
                   lane_root=os.path.join(root, "slicer-%d" % os.getpid()),
                   tag="p%d" % os.getpid(),
                   arena=Arena.of(slots=depth * len(specs) or 2,
                                  slot_bytes=int(widest) + 4096, root=root))

    def core_blocks(self, total: int | None = None) -> dict[int, tuple[int, ...]]:
        """A disjoint core block per slice, counted over the whole deployment.

        The emulated servers do not get to share cores, even though the plan puts
        them far apart: one host runs all of them, and two OpenMP teams pinned to
        the same threads spin against each other, which reads as a thirty-fold
        slowdown rather than as a placement effect. A plan that asks for more
        threads than the box has wraps, and :meth:`start` reports the overlap.
        """
        available = total or len(os.sched_getaffinity(0))
        blocks: dict[int, tuple[int, ...]] = {}
        cursor = 0
        for spec in self.slices:
            rank = self.ranks[spec.index]
            blocks[rank] = tuple((cursor + offset) % available
                                 for offset in range(spec.eta))
            cursor += spec.eta
        return blocks

    def _drain(self) -> None:
        """Take whatever the slices have already sent, so no message is lost.

        :meth:`_await` is asked for one rank's report, but a campaign fails by having
        *some other* rank die, and a parent that reads strictly in rank order waits on
        the first rank's deadline while the answer to "what went wrong" is sitting in
        the second one's buffer.
        """
        for rank, control in self.controls.items():
            if rank in self._inbox or not control.poll(0.0):
                continue
            try:
                self._inbox[rank] = control.recv()
            except (EOFError, OSError):
                # A closed pipe is *readable*, which is not the same as informative:
                # the rank is gone, and only the process table can say so.
                self._inbox[rank] = ("closed", None)

    def _await(self, rank: int, want: str) -> dict:
        """The next message from one slice, or the reason it will not send one.

        A child can die without getting to speak -- a quota it cannot allocate under,
        a shape the framework refuses -- and a parent that calls ``recv()`` on the
        remains waits out the deadline rather than reporting the failure. So the wait
        is polled against every slice's pipe and against every process, and a process
        is the authority on whether more is coming.
        """
        deadline = time.monotonic() + _WAIT_S
        while True:
            self._drain()
            # Another slice's failure is the answer to this one's silence, so it is
            # reported rather than left in the inbox for a turn that never comes.
            for other, message in self._inbox.items():
                if other != rank and message[0] in ("error", "closed"):
                    # "closed" says the child is gone but not how, and the two ways it
                    # can go need different fixes: a Python exception travels with a
                    # traceback, a signal does not, so the exit code is the only sign
                    # of a quota that killed a process rather than refusing it.
                    code = self.processes[other].exitcode
                    raise RuntimeError("slice %d cannot serve this request: %s%s"
                                       % (self.slices_of_rank(other).index,
                                          message[1] or "its pipe is closed",
                                          "" if code is None
                                          else " (exit code %s)" % code))
            held = self._inbox.pop(rank, None)
            if held is not None:
                kind, *rest = held
                if kind == want:
                    return rest[0]
                if kind == "error":
                    raise RuntimeError("slice %d died: %s"
                                       % (self.slices_of_rank(rank).index, rest[0]))
                if kind == "closed":
                    raise RuntimeError("slice %d closed its control pipe before it"
                                       " owed %r" % (rank, want))
                raise RuntimeError("slice %d owed %r and sent %r" % (rank, want, kind))
            for other, process in self.processes.items():
                if not process.is_alive() and other not in self._inbox:
                    raise RuntimeError(
                        "slice %d exited with code %s before it owed %r"
                        % (self.slices_of_rank(other).index, process.exitcode, want))
            if time.monotonic() > deadline:
                raise TimeoutError("slice %d has not sent %r in %ds"
                                   % (rank, want, int(_WAIT_S)))
            time.sleep(_POLL_S)

    def slices_of_rank(self, rank: int) -> Slice:
        return next(spec for spec in self.slices if self.ranks[spec.index] == rank)

    def start(self, device: str = "cpu", batch: int | None = None, seed: int = 0,
              mb_per_vcpu: float = 1792.0,
              on_ready: Callable[[int, int, dict], None] | None = None) -> "Deployment":
        """Spawn the functions, and wait until each one says it holds its weights.

        ``on_ready(index, total, row)`` is called as each slice reports in, for a
        caller that prints progress. Spawning is sequential and each child loads the
        whole service first, so this is the slowest step of a deployment and the one
        a silent terminal most needs a name for.
        """
        context = mp.get_context("spawn")
        blocks = self.core_blocks()
        inbound: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        outbound: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for (producer, consumer), keys in self.lanes.items():
            inbound.setdefault(consumer, []).append((producer, keys))
            outbound.setdefault(producer, []).append((consumer, keys))
        os.makedirs(self.lane_root, exist_ok=True)
        # A completion travels from the last slice back to the first: the entry is the
        # only process that knows how many requests are live, and that is what a fixed
        # concurrency window is made of. The parent keeps no end of it.
        reader, writer = context.Pipe(duplex=False)
        last = self.ranks[self.terminal_index]
        for spec in self.slices:
            rank = self.ranks[spec.index]
            job = _Job(key=self.key, device=device, batch=batch, seed=seed,
                       groups=tuple(s.layers for s in self.slices), index=spec.index,
                       rank=rank, entry=rank == self.entry_rank,
                       terminal=spec.index == self.terminal_index,
                       inbound=tuple(sorted(inbound.get(rank, []))),
                       outbound=tuple(sorted(outbound.get(rank, []))),
                       lanes=self.open_lanes, arena_path=self.arena.path,
                       arena_slots=self.arena.slots, slot_bytes=self.arena.slot_bytes,
                       lane_root=self.lane_root, tag=self.tag, cores=blocks[rank],
                       eta=spec.eta,
                       quota_bytes=max(spec.alloc_mb, spec.eta * mb_per_vcpu) * MB,
                       feedback_in=reader if rank == self.entry_rank else None,
                       feedback_out=writer if rank == last else None)
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_serve, args=(job, child), daemon=True)
            process.start()
            self.processes[rank] = process
            self.controls[rank] = parent
            # Wait before forking the next one. A child transiently holds the *whole*
            # service -- it loads the model, records the forward, and only then frees
            # what its slice does not keep -- so spawning eight slices of a 10 GB
            # transformer at once asks for 86 GB of weights that exist nine times over.
            # One at a time keeps the peak at one load plus the shards already settled.
            row = self._await(rank, "ready")
            self.ready += (row,)
            if on_ready is not None:
                on_ready(len(self.ready), len(self.slices), row)
        reader.close()
        writer.close()
        return self

    def _finish(self) -> tuple[dict, ...]:
        """End the campaign and take the one report each slice owes."""
        self.controls[self.entry_rank].send(("stop", None))
        self.drained = True
        rows: list[dict] = []
        for rank in sorted(self.controls):
            row = self._await(rank, "report")
            rows.append(row)
            # Only now is this slice's result the caller's to keep, so only now may it
            # go away and take its file descriptors with it.
            self.controls[rank].send(("ack", None))
        return tuple(rows)

    def load(self, inputs: Any, requests: int = 100, concurrency: int = 1) -> Load:
        """Run ``requests`` through the service with at most ``concurrency`` in flight.

        Fig. 10b measures a 95th percentile, and a percentile needs a population: one
        request at a time gives the mean and nothing else. The window is closed-loop,
        so the offered load is whatever the service can absorb -- which on one box is
        far below the paper's 250--1250 requests/second, and :attr:`Load.throughput`
        is where that difference becomes visible rather than silent.
        """
        if concurrency > self.queue_depth:
            raise ValueError("a lane carries %d messages, so the window cannot be %d"
                             % (self.queue_depth, concurrency))
        started = time.perf_counter()
        self.controls[self.entry_rank].send(("load", int(requests), int(concurrency),
                                             inputs))
        rows = self._finish()
        entry = next(row for row in rows if row["entry"])
        return Load(latencies_s=tuple(entry["latencies_s"]), per_slice=rows,
                    seconds=time.perf_counter() - started,
                    wire_bytes=sum(sum(row["moved_bytes"]) for row in rows))

    def sweep(self, inputs: Any, campaigns: Sequence[tuple[int, int]]
              ) -> tuple[Load, ...]:
        """Several closed-loop windows through one already-started service.

        Starting a deployment is the expensive part -- every slice loads the whole
        service before it frees the layers it does not keep -- so a concurrency curve
        that paid for a start per point would be measuring the loader. The slices need
        no notice of the curve: a request enters only through the entry, and the entry
        admits the next campaign once the last one has left the terminal, so every rank
        serves exactly ``requests`` per campaign and the reports split by position.
        """
        for _, concurrency in campaigns:
            if concurrency > self.queue_depth:
                raise ValueError("a lane carries %d messages, so the window cannot be %d"
                                 % (self.queue_depth, concurrency))
        for requests, concurrency in campaigns:
            self.controls[self.entry_rank].send(("load", int(requests),
                                                 int(concurrency), inputs))
        rows = self._finish()
        entry = next(row for row in rows if row["entry"])
        cursor, loads = 0, []
        for position, (requests, concurrency) in enumerate(campaigns):
            bounds = slice(cursor, cursor + int(requests))
            loads.append(Load(
                latencies_s=tuple(entry["latencies_s"][bounds]),
                per_slice=tuple(_window(row, bounds) for row in rows),
                seconds=entry["campaign_s"][position],
                wire_bytes=sum(sum(row["moved_bytes"][bounds]) for row in rows)))
            cursor += int(requests)
        return tuple(loads)

    def invoke(self, inputs: Any) -> Round:
        """One request through the chain, timed from the caller's own clock."""
        load = self.load(inputs, requests=1, concurrency=1)
        rows = tuple({"rank": row["rank"], "index": row["index"],
                      "span_s": row["span_s"][0], "busy_s": row["busy_s"][0],
                      "moved_bytes": row["moved_bytes"][0],
                      "transport_s": row["transport_s"][0],
                      "link_s": row["link_s"][0],
                      "resident_bytes": row["resident_bytes"][0]}
                     for row in load.per_slice if row["requests"])
        return Round(seconds=load.seconds,
                     result=next((row["result"] for row in load.per_slice
                                  if row["result"] is not None), None),
                     per_slice=rows,
                     compute_s=sum(row["span_s"] for row in rows),
                     makespan_s=max((row["span_s"] for row in rows), default=0.0),
                     wire_bytes=load.wire_bytes,
                     transport_s=sum(row["transport_s"] for row in rows),
                     link_s=sum(row["link_s"] for row in rows))

    def unsplit(self, inputs: Any, device: str = "cpu", threads: int | None = None
                ) -> Round:
        """The whole service in this process: a numerical reference, not a baseline.

        Deliberately not a deployment -- no lane, no fork, no quota -- because a
        channel's cost would otherwise show up as a reason to split. But it runs with
        the parent's own thread count and the box's whole address space, so its
        seconds are *not* Fig. 10's ``Unsplit``: that one is a function like every
        other, with an ``eta`` and a quota, and it is built as
        ``Deployment.build(key, [order], [eta])`` -- a one-slice plan through the same
        path. Use this for ``allclose`` against a split result, and never as the
        denominator of a latency ratio.
        """
        started = time.perf_counter()
        if threads:
            torch.set_num_threads(threads)
        clock = time.perf_counter()
        result = self.program.run(move_tensors(inputs, device))
        span = time.perf_counter() - clock
        return Round(seconds=time.perf_counter() - started, result=result,
                     per_slice=({"rank": -1, "index": 0, "span_s": span,
                                 "moved_bytes": 0.0, "transport_s": 0.0, "link_s": 0.0,
                                 "resident_bytes": _resident()[0]},),
                     compute_s=span)

    def close(self) -> None:
        """Close the service: the stop signal travels down the lanes requests did."""
        if self.closed:
            return
        self.closed = True
        if self.controls and not self.drained:
            try:
                self.controls[self.entry_rank].send(("stop", None))
            except OSError:
                pass
        for process in self.processes.values():
            process.join(timeout=60)
            if process.is_alive():  # a slice wedged in a link cannot hold the box
                process.terminate()
        for control in self.controls.values():
            control.close()
        self.controls = {}
        if self.arena is not None:
            self.arena.destroy()
        shutil.rmtree(self.lane_root, ignore_errors=True)


def equal_layers(order: Sequence[str], slices: int) -> list[list[str]]:
    """``Uniform``: the same number of layers in every slice, whatever they cost."""
    step = -(-len(order) // slices)
    return [list(order[start:start + step]) for start in range(0, len(order), step)] or [[]]


def main(argv: Sequence[str | None] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gcn1")
    parser.add_argument("--slices", type=int, default=2)
    parser.add_argument("--eta", type=int, default=1)
    parser.add_argument("--servers", type=int, default=1,
                        help="spread the slices over this many emulated servers")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="closed-loop window; capped by the lane depth")
    args = parser.parse_args(argv)

    model, inputs = zoo.load(args.model, device="cpu")
    program = record(model, inputs, name=args.model)
    groups = equal_layers(program.graph.order, args.slices)
    servers = [index % args.servers for index in range(len(groups))]
    print("%s: %d layers, %d slices, %d servers"
          % (args.model, len(program.graph.order), len(groups), args.servers), flush=True)
    service = Deployment.build(args.model, groups, [args.eta] * len(groups), servers,
                               queue_depth=max(2, args.concurrency))
    service.start()
    for row in service.ready:
        print("  slice %d: %d layers, %d steps, base %.0f MB, quota %.0f MB, freed %.0f MB"
              % (row["index"], row["layers"], row["steps"], row["base_bytes"] / MB,
                 row["quota_bytes"] / MB, row["freed_bytes"] / MB), flush=True)
    want = service.unsplit(inputs)
    load = service.load(inputs, requests=args.requests, concurrency=args.concurrency)
    print("  window=%d  %s" % (args.concurrency, load.summary()), flush=True)
    for row in load.per_slice:
        print("  slice %d  busy %.3f ms  span %.3f ms  wait %.3f ms  transport %.3f ms"
              "  peak %.0f MB"
              % (row["index"], 1e3 * _mean(row["busy_s"]), 1e3 * _mean(row["span_s"]),
                 1e3 * _mean(row["wait_s"]), 1e3 * _mean(row["transport_s"]),
                 max(row["resident_bytes"], default=0.0) / MB), flush=True)
    back = [t for t in _iter_tensors(next(r["result"] for r in load.per_slice
                                           if r["result"] is not None))]
    same = all(torch.allclose(a, b, atol=1e-4)
               for a, b in zip(back, _iter_tensors(want.result)))
    print("  matches the unsplit service: %s  (unsplit %.3f ms, wire %.2f MB)"
          % (same, want.seconds * 1e3, load.wire_bytes / MB), flush=True)
    service.close()
    return 0 if same else 1


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    sys.exit(main())
