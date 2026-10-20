"""Message channels between slices (Sec. II-D, Fig. 8).

A cut service moves one activation per edge, from the process that produced it
to the process that consumes it, and the paper bills that hop two ways: through
an external store (Redis, or a function's HTTP reply), and through *shared
memory plus a message channel*, where the payload lands in a tmpfs region both
processes map and the channel carries only a header. Fig. 8 is that choice, and
Fig. 3 measures how much of the hop the second option removes.

Both transports here are one-way endpoints, because that is what a slice has: an
inbox from its predecessor and an outbox to its successor.

:class:`PipeEndpoint` pushes the payload through the kernel's pipe buffer, which
is the byte stream an HTTP or Redis hop is built on. A 12 MB tensor crosses a
64 KB pipe in 192 writes, and every one of them is a syscall whose time the
receiver pays for as latency.

:class:`ShmEndpoint` copies the payload into a slot of an :class:`Arena` in
``/dev/shm`` and writes only ``(slot, nbytes)`` to a pipe; the reader returns a
credit for the slot once it is done with it, so the writer can already be
filling the next one. That overlap is the reason the arena has more than one
slot -- with a single slot the receiver's copy-out would serialise behind the
sender's next write.

``/dev/shm`` is Linux's tmpfs. Where it is absent -- the macOS editing mirror --
``root`` can name any directory and the code still runs, just on disk rather
than in memory, so the correctness tests are portable and the timings are not.

:func:`open_fifo` and :func:`open_shm_fifo` are the same two transports over a
*named* pipe, which is what a slice running in its own process needs: a spawned
child inherits only the descriptors ``multiprocessing`` dupes for itself, so the
rendezvous has to be a path. :meth:`Arena.attach` is their counterpart on the
payload side -- the creating process keeps the unlink, the attaching one keeps
only its own mapping.
"""

from __future__ import annotations

import mmap
import os
import struct
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

__all__ = ["Arena", "Channel", "PipeEndpoint", "ShmEndpoint",
           "open_pipe", "open_shm", "mkfifo", "open_fifo", "open_shm_fifo",
           "pipe_chunk", "tmpfs_root"]

#: ``(slot, nbytes)``, the whole of what a shared-memory hop puts on the wire.
HEADER = struct.Struct("!II")

#: Linux's default pipe capacity, 16 pages; used when the real one is unreadable.
DEFAULT_PIPE_CHUNK = 65536

GETPIPE_SZ = 1032  # linux/asm-generic/fcntl.h; absent on other platforms


def tmpfs_root() -> str:
    """Where a shared region should live: the tmpfs the kernel never pages out."""
    return "/dev/shm" if os.path.isdir("/dev/shm") else os.getcwd()


@dataclass
class Stats:
    """What an endpoint did, so an experiment can bill payload and syscalls."""

    payload_bytes: int = 0
    writes: int = 0
    reads: int = 0


def count_writes(fd: int, data) -> int:
    """Write ``data`` out, returning how many ``os.write`` calls it took.

    The count is the measurement: Fig. 8's claim is about syscalls per hop, and
    a pipe that is short on buffer space splits a payload without anyone
    noticing unless somebody counts.
    """
    view = memoryview(data)
    calls = 0
    offset = 0
    while offset < len(view):
        offset += os.write(fd, view[offset:])
        calls += 1
    return calls


def read_exactly(fd: int, size: int) -> tuple[bytes, int]:
    """Read ``size`` bytes and report how many ``os.read`` calls they cost."""
    chunks = []
    got = 0
    calls = 0
    while got < size:
        block = os.read(fd, size - got)
        if not block:
            raise EOFError("peer closed with %d of %d bytes read" % (got, size))
        chunks.append(block)
        got += len(block)
        calls += 1
    return b"".join(chunks), calls


class Channel:
    """One endpoint of a one-way transfer of buffers."""

    def __init__(self) -> None:
        self.sent = Stats()
        self.received = Stats()

    def send(self, payload: bytes) -> None:
        raise NotImplementedError("this endpoint only receives")

    def recv(self) -> bytes:
        raise NotImplementedError("this endpoint only sends")

    def borrow(self):
        """Hold the next payload without copying it, if the transport can."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "Channel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PipeEndpoint(Channel):
    """Fig. 8's control group: the payload itself goes through the pipe."""

    def __init__(self, read_fd: int | None, write_fd: int | None,
                 chunk: int | None = None) -> None:
        super().__init__()
        self._r = read_fd
        self._w = write_fd
        sizing = write_fd if write_fd is not None else read_fd
        self._chunk = chunk or pipe_chunk(sizing)
        self._closed = False

    def send(self, payload: bytes) -> None:
        if self._w is None:
            raise NotImplementedError("this endpoint only receives")
        size = len(payload)
        calls = count_writes(self._w, HEADER.pack(size, 0))
        view = memoryview(payload)
        offset = 0
        while offset < size:
            calls += count_writes(self._w, view[offset:offset + self._chunk])
            offset += self._chunk
        self.sent.writes += calls
        self.sent.payload_bytes += size

    def recv(self) -> bytes:
        if self._r is None:
            raise NotImplementedError("this endpoint only sends")
        packed, calls = read_exactly(self._r, HEADER.size)
        size, _ = HEADER.unpack(packed)
        payload, extra = read_exactly(self._r, size)
        self.received.reads += calls + extra
        self.received.payload_bytes += size
        return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # A FIFO hands both directions the same descriptor, so the second close
        # of a pair is not an error to swallow -- it is the same fd.
        for fd in {self._r, self._w}:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def pipe_chunk(fd: int) -> int:
    """How much one ``os.write`` on this pipe carries before it blocks."""
    try:
        size = os.fcntl(fd, GETPIPE_SZ, 0)
    except (OSError, AttributeError):
        return DEFAULT_PIPE_CHUNK
    return max(int(size), 4096)


def open_pipe() -> tuple[PipeEndpoint, PipeEndpoint]:
    """A sender and a receiver at the two ends of one pipe."""
    read_fd, write_fd = os.pipe()
    return PipeEndpoint(None, write_fd), PipeEndpoint(read_fd, None)


class Arena:
    """A fixed set of slots in a shared file -- the payload store of Fig. 8.

    Owns the file and unlinks it, so a caller can build several queues over one
    arena, which is what a slice with several outgoing edges does.
    """

    def __init__(self, path: str, slots: int, slot_bytes: int, *, create: bool = True) -> None:
        self.path = path
        self.slots = slots
        self.slot_bytes = slot_bytes
        self._owns = create
        if create:
            # O_EXCL because a queue's slots are uninitialised: an arena left
            # behind by a crashed run would hand its stale bytes to a reader.
            self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            self._fd = os.open(path, os.O_RDWR)
        try:
            if create:
                os.ftruncate(self._fd, slots * slot_bytes)
            self.region = mmap.mmap(self._fd, slots * slot_bytes)
        except BaseException:
            self.destroy()
            raise

    @classmethod
    def of(cls, slots: int, slot_bytes: int, *, root: str | None = None,
           name: str | None = None) -> "Arena":
        path = os.path.join(root or tmpfs_root(), "slicer-%s" % (name or uuid.uuid4().hex[:12]))
        return cls(path, slots, slot_bytes)

    @classmethod
    def attach(cls, path: str, slots: int, slot_bytes: int) -> "Arena":
        """Map an arena another process created, which is all a second slice can do.

        The mapping is the whole of the sharing: both processes see the same pages,
        so the slot a producer writes is the slot the consumer reads, and neither
        copies. The creator unlinks the name, so an attached mapping stays alive on
        a path no directory entry has.
        """
        return cls(path, slots, slot_bytes, create=False)

    def slot(self, index: int) -> memoryview:
        start = index * self.slot_bytes
        return memoryview(self.region)[start:start + self.slot_bytes]

    def destroy(self) -> None:
        try:
            self.region.close()
        except (AttributeError, ValueError):
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        if not self._owns:
            return
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self) -> "Arena":
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()


class ShmEndpoint(Channel):
    """Fig. 8's optimisation: payload in the arena, header in the pipe.

    A slot is held by whoever may write it. The sender starts with the whole
    ring -- the arena is empty, so every slot is free -- and a receiver returns
    a slot only after the payload's view has been released. The ring is the
    endpoint's bookkeeping, so a caller that keeps both ends in one process must
    not let the sender write while a view is still open: the credits, not the
    bytes, are what the two ends agree on.
    """

    def __init__(self, arena: Arena, read_fd: int | None, write_fd: int | None,
                 credit_read_fd: int | None, credit_write_fd: int | None,
                 free: Sequence[int] = ()) -> None:
        super().__init__()
        self.arena = arena
        self._r = read_fd
        self._w = write_fd
        self._credit_r = credit_read_fd
        self._credit_w = credit_write_fd
        self._free = list(free)
        self._holds: list[int] = []
        self._closed = False

    def send(self, payload: bytes) -> None:
        if self._w is None:
            raise NotImplementedError("this endpoint only receives")
        if len(payload) > self.arena.slot_bytes:
            raise ValueError("%d B does not fit a %d B slot"
                             % (len(payload), self.arena.slot_bytes))
        slot = self._await_credit()
        view = memoryview(payload)
        target = self.arena.slot(slot)
        target[:len(view)] = view
        calls = count_writes(self._w, HEADER.pack(slot, len(payload)))
        self.sent.writes += calls
        self.sent.payload_bytes += len(payload)

    def recv(self) -> bytes:
        with self.borrow() as view:
            return bytes(view)

    @contextmanager
    def borrow(self):
        """Yield the payload as a view of the arena, then hand the slot back."""
        if self._r is None:
            raise NotImplementedError("this endpoint only sends")
        packed, calls = read_exactly(self._r, HEADER.size)
        slot, size = HEADER.unpack(packed)
        self.received.reads += calls
        self.received.payload_bytes += size
        self._holds.append(slot)
        try:
            yield memoryview(self.arena.slot(slot))[:size]
        finally:
            self._lend_credit()

    def _await_credit(self) -> int:
        while not self._free:
            packed, calls = read_exactly(self._credit_r, HEADER.size)
            self.received.reads += calls
            self._free.append(HEADER.unpack(packed)[0])
        return self._free.pop(0)

    def _lend_credit(self) -> None:
        while self._holds:
            slot = self._holds.pop(0)
            self._free.append(slot)
            if self._credit_w is not None:
                calls = count_writes(self._credit_w, HEADER.pack(slot, 0))
                self.sent.writes += calls

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in {self._r, self._w, self._credit_r, self._credit_w}:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def open_shm(arena: Arena, slots: Sequence[int]
             ) -> tuple[ShmEndpoint, ShmEndpoint]:
    """A sender and a receiver over ``slots`` of ``arena``.

    Two pipes: the data pipe carries ``(slot, nbytes)`` one way, the credit pipe
    carries the slot back the other. ``slots`` partitions an arena between
    directions, so a queue owns only its own indices and a slice can send on one
    queue while receiving on another that shares the mapping.
    """
    if not slots:
        raise ValueError("a queue needs at least one slot")
    data_r, data_w = os.pipe()
    credit_r, credit_w = os.pipe()
    sender = ShmEndpoint(arena, None, data_w, credit_r, None, free=slots)
    receiver = ShmEndpoint(arena, data_r, None, None, credit_w)
    return sender, receiver


def mkfifo(name: str, root: str | None = None) -> str:
    """The path of a named pipe called ``name``, creating it if it is not there.

    A queue a slice has to reach across a process boundary cannot be an anonymous
    ``os.pipe()``: a spawned child inherits only the descriptors ``multiprocessing``
    dupes for its own use, so the rendezvous has to be a name. ``name`` is the whole
    address, and both ends deriving it from the queue they belong to is what lets
    the runtime start a child with a plan and nothing else.
    """
    path = os.path.join(root or tmpfs_root(), "slicer-%s.fifo" % name)
    try:
        os.mkfifo(path, 0o600)
    except FileExistsError:
        pass
    return path


def _fifo_fd(path: str) -> int:
    """Open one end of a FIFO for both directions, so the open never waits.

    An ordinary ``O_WRONLY`` open blocks until a reader arrives, and a runtime that
    starts its slices in dependency order would deadlock in the first ``open`` if
    every process waited for its predecessor. ``O_RDWR`` on a FIFO is the standard
    way to get a descriptor that is ready immediately; the endpoint handed the
    descriptor still only reads or only writes, so the direction is a property of
    the endpoint rather than of the file.
    """
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        os.set_blocking(fd, True)  # the writes must block for backpressure
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_fifo(name: str, *, root: str | None = None) -> tuple[PipeEndpoint, PipeEndpoint]:
    """The cross-process form of :func:`open_pipe`: two ends of one named pipe.

    Each end gets its own descriptor for the FIFO, so closing one does not pull the
    other's file out from under it.
    """
    path = mkfifo(name, root)
    return PipeEndpoint(None, _fifo_fd(path)), PipeEndpoint(_fifo_fd(path), None)


def open_shm_fifo(arena: Arena, name: str, slots: Sequence[int],
                  *, root: str | None = None) -> tuple[ShmEndpoint, ShmEndpoint]:
    """The cross-process form of :func:`open_shm`.

    Two FIFOs, because the credit ring has to survive the parent handing a slot
    back to a child that is already filling the next one.
    """
    if not slots:
        raise ValueError("a queue needs at least one slot")
    data = mkfifo("%s-data" % name, root)
    credit = mkfifo("%s-credit" % name, root)
    sender = ShmEndpoint(arena, None, _fifo_fd(data), _fifo_fd(credit), None,
                         free=slots)
    receiver = ShmEndpoint(arena, _fifo_fd(data), None, None, _fifo_fd(credit))
    return sender, receiver
