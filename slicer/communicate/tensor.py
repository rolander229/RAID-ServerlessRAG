"""Move a layer's output across a process boundary (Sec. II-C, Fig. 4).

A cut service ships one value per edge, from the process that produced it to the
process that consumes it, and in MOPAR that value is a tensor: the output of the
producer layer, which the consumer layer takes as its input. :func:`torch.save`
would do this, but it wraps the bytes in a zip archive and a pickle of the
tensor's metadata, so a 12 MB activation crosses the wire with a cost that
belongs to the serialiser rather than to the model -- and COM's whole claim
(Fig. 3) is about how many *bytes* a hop moves.

The format here is therefore flat and tiny:

``kind``
    one byte, naming what follows: a tensor, a tuple or list of the same, a
    dict, or anything else pickled.
``tensor``
    dtype name, then the shape, then the contiguous bytes themselves.

So a tensor crosses as its bytes plus a header of a few tens of them, and the
shape stays exact, which is what a latency comparison between transports needs
to be able to say. The one escape hatch is ``OTHER``: a layer fed by an ``int``
(a GCN's index, a tokenizer's offset) ships as a pickle of it, which is small
and correct, and a layer fed by a custom object is not this repository's
problem -- the value will fail to pickle, and the error will name it.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any

import torch

__all__ = ["pack", "unpack"]

TENSOR, TUPLE, LIST, DICT, OTHER = range(5)

#: Counts and dimensions share one width: a GCN's rows are indexed by node count,
#: and the length of a pickled payload is bounded by nothing but the object.
_LEN = struct.Struct("!I")


def _put(out: bytearray, value: int) -> None:
    out += _LEN.pack(value)


def _pack_one(value: Any, out: bytearray) -> None:
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous()
        out.append(TENSOR)
        # ``str(dtype)`` is the documented spelling -- ``torch.float32`` -- and every
        # dtype is reachable from it as an attribute of the module, which ``name`` is
        # not guaranteed to be.
        name = str(raw.dtype).rpartition(".")[2].encode("ascii")
        _put(out, len(name))
        out += name
        shape = tuple(raw.shape)
        _put(out, len(shape))
        for dim in shape:
            _put(out, dim)
        out += raw.numpy().tobytes()
        return
    if isinstance(value, (tuple, list)):
        out.append(TUPLE if isinstance(value, tuple) else LIST)
        _put(out, len(value))
        for item in value:
            _pack_one(item, out)
        return
    if isinstance(value, dict):
        out.append(DICT)
        _put(out, len(value))
        for key, item in value.items():
            _pack_one(key, out)
            _pack_one(item, out)
        return
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    out.append(OTHER)
    _put(out, len(blob))
    out += blob


def pack(value: Any) -> bytes:
    """Serialise a layer's output -- a tensor, or the tuple/dict around one."""
    out = bytearray()
    _pack_one(value, out)
    return bytes(out)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.view = memoryview(data)
        self.offset = 0

    def byte(self) -> int:
        value = self.view[self.offset]
        self.offset += 1
        return value

    def count(self) -> int:
        (value,) = _LEN.unpack_from(self.view, self.offset)
        self.offset += _LEN.size
        return value

    def blob(self, size: int) -> bytes:
        chunk = self.view[self.offset:self.offset + size]
        self.offset += size
        return chunk.tobytes()

    def tensor(self) -> torch.Tensor:
        dtype = getattr(torch, self.blob(self.count()).decode("ascii"))
        shape = tuple(self.count() for _ in range(self.count()))
        size = torch.Size(shape)
        itemsize = torch.tensor([], dtype=dtype).element_size()
        payload = self.blob(size.numel() * itemsize)
        if size.numel() == 0:  # an empty buffer has no dtype of its own to read
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(payload), dtype=dtype).view(shape)

    def one(self) -> Any:
        kind = self.byte()
        if kind == TENSOR:
            return self.tensor()
        if kind in (TUPLE, LIST):
            items = [self.one() for _ in range(self.count())]
            return tuple(items) if kind == TUPLE else items
        if kind == DICT:
            return {self.one(): self.one() for _ in range(self.count())}
        return pickle.loads(self.blob(self.count()))


def unpack(data: bytes) -> Any:
    """The inverse of :func:`pack`, with tensors on the CPU and ready to be moved."""
    reader = _Reader(data)
    value = reader.one()
    if reader.offset != len(reader.view):
        raise ValueError("trailing bytes after the payload")
    return value
