"""Recover a model's layer DAG by watching one real forward pass.

MOPAR's partitioner never sees ``nn.Module`` objects: it consumes a
:class:`~slicer.partition.graph.LayerGraph` whose nodes carry memory and time and
whose edges carry the number of bytes that cross the cut. Getting those edges
right is what this module does, and the obvious approaches all fail for part of
the benchmark:

``torch.fx.symbolic_trace``
    Traces torchvision CNNs fine, but not the models whose forward contains
    data-dependent control flow or ``torch.ops`` calls -- the LSTM/GRU stacks,
    the graph convolutions with their dense adjacency, and HuggingFace-style
    transformers either raise or silently mis-trace.
``torch.jit.trace``
    Same coverage problem, plus it freezes the branch taken.
Hooks on ``nn.Module`` calls only
    Always works, but loses exactly the edges that matter: the residual ``add``
    in ResNet and ConvNeXt and the three-way ``cat`` in Inception are plain
    functions, not modules, so a hook-only view of ResNet is a chain and the
    partitioner cannot see that the block input skips over the block. A cut
    placed on such an edge would silently drop the skip connection.

So this module watches *values* rather than *calls*. Every tensor is tagged with
the node that produced it, where a node is either a leaf module call or a
torch-level function (``add``, ``cat``, ``matmul``, ...). Each node records which
node its operands came from, and a post-pass then assigns every torch function to
the layer that has to run it -- the first layer to read its output. Cutting the
resulting groups against each other yields module-to-module edges that keep both
skip connections and fusions.

Four invariants make that correct, and the first two were learned from a wrong
edge count rather than from reading:

* **Provenance is read when a value is consumed, not after the forward.**
  Deferred resolution reads the final producer map, and in-place writers break
  it: ``out += identity`` in a ResNet bottleneck, and ``nn.ReLU(inplace=True)``
  which torchvision uses 49 times in ResNet-50, both overwrite the map entry of a
  tensor they are about to read. Measured, before this was fixed: 108 edges
  instead of 157 with 49 ReLUs surfacing as graph sources; then 157 edges with no
  skip edges at all, because the same masking ate the operand of every ``add``.
* **A node is numbered before its function runs.** A torch function can call
  torch functions, so numbering after the call labels the innermost child.
* **An edge carries the size of the buffer a cut puts on the wire, not of the
  fused value.** A torch function is free work the partitioner cannot place on
  either side of a cut, so it runs with whichever layer first reads its result,
  and a cut above *that* layer ships each of its operands separately: ``cat`` of
  three branches costs three edges sized by the branches, not one sized by the
  concatenation. That is the conservative reading and it matches Eq. (4), which
  sums the transfer time over the tensors crossing the cut.
* **A value stops exporting its history at the first layer that consumes it.**
  Expanding operands recursively instead turns a residual chain into an
  accumulation: each ``add`` re-exports every module it has ever passed through,
  so the Transformer's last ``norm`` carried 145 in-edges of 6.16 MB each and
  Eq. (3) read a 223 MB layer out of a 16 MB service. Those older buffers are dead
  -- the newest accumulator already holds their sum -- so once a layer has read a
  value, everything downstream sees just that layer as its producer. What stays
  expanded is a fusion no layer has read yet, which is what keeps Inception's
  ``cat`` of two ``cat``s at four branch edges.

The capture keeps a reference to every tensor it tags, because ``id()`` is only
stable while the object lives, so peak memory here is above a plain forward.
Nothing downstream reads it: memory and time come from the service profiler's own
pass, and this module produces structure only.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.overrides import TorchFunctionMode

from ..partition.graph import Layer, LayerGraph, Tensor


def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    """Yield every tensor reachable from a (possibly nested) argument or result."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def move_tensors(obj: Any, device: str, dtype=None) -> Any:
    """Recursively place the tensors of an example request, leaving dtypes alone.

    Shared with :func:`slicer.models.zoo.load` so a model and its request cannot
    disagree about where they live. Integer inputs (token ids, masks) keep their
    dtype: casting them would hand an embedding lookup a float index.
    """
    if isinstance(obj, torch.Tensor):
        out = obj.to(device)
        return out.to(dtype) if dtype is not None and out.is_floating_point() else out
    if isinstance(obj, dict):
        return {k: move_tensors(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_tensors(v, device, dtype) for v in obj)
    if isinstance(obj, list):
        return [move_tensors(v, device, dtype) for v in obj]
    return obj


class Namer:
    """Turns a module into its layer name, disambiguating repeated calls.

    Shared with :mod:`slicer.profile.probe`, which measures a second forward and
    joins its rows to this graph by name. The call counter is the only state the
    two passes have to agree on, so it lives here rather than inside the tracer.
    """

    def __init__(self, paths: dict[int, str] | None = None) -> None:
        self.paths = paths or {}
        self.used: dict[str, int] = {}

    def __call__(self, module) -> str:
        base = self.paths.get(id(module), type(module).__name__)
        count = self.used.get(base, 0)
        self.used[base] = count + 1
        return base if count == 0 else "%s#%d" % (base, count)


def module_paths(model: torch.nn.Module) -> dict[int, str]:
    return {id(m): path for path, m in model.named_modules() if path}


def leaf_modules(model: torch.nn.Module) -> list:
    """Modules with no children: the units a layer profile is measured on."""
    return [m for m in model.modules() if not list(m.children())]


def _direct_bytes(module: torch.nn.Module) -> float:
    return sum(float(p.numel()) * p.element_size()
               for p in module.parameters(recurse=False))


class _Frame:
    """One open container call: the leaf it has billed its weights to, if any."""

    __slots__ = ("module", "first_leaf")

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.first_leaf: str | None = None


class OwnerCharge:
    """Put every parameter byte of a model on a leaf module *call*.

    A layer graph is measured on leaves, and ``parameters()`` on a leaf is just its
    own tensors, so a weight held by a *container* belongs to no node. The common
    case is a block that keeps a vector on itself; the fatal one is
    ``nn.MultiheadAttention``, which with ``need_weights=False`` runs a functional
    fast path that enters none of its own submodules -- so a BERT block's fused QKV
    matrix, 48 MB of it, and its ``out_proj`` besides are both invisible to a leaf
    view. The first version of this billed 3329 MB of a 4866 MB model and HyPAD
    sized every slice of a transformer against two thirds of what it has to hold.

    Two rules settle it, the same one
    :meth:`slicer.profile.probe._Probe.bill_strays` uses for such a module's *work*:

    * a call's own bytes belong to the first leaf executed inside it, because hidden
      work and hidden weights run with whoever reads them;
    * a container that executed no leaf hands its bytes -- its own, plus those of
      children that never ran -- to the next leaf executed anywhere.

    Neither is known while the forward is still going, so amounts accumulate in
    :attr:`credits`, keyed by the unique name of a leaf call, and the caller reads
    them once it is over. Every byte under a module that ran lands on exactly one
    call. Bytes under one that never ran land on the first leaf of the nearest
    ancestor that did, which is where the process holds them for the service's whole
    life; only a model that executes nothing at all leaves its weights uncharged.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._own = {id(m): _direct_bytes(m) for m in model.modules()}
        self._parent: dict[int, int] = {}
        for module in model.modules():
            for child in module.children():
                self._parent[id(child)] = id(module)
        self._subtree: dict[int, float] = {}
        for module in reversed(list(model.modules())):
            # modules() is pre-order, so a child's total is ready before its parent's.
            self._subtree[id(module)] = self._own[id(module)] + sum(
                self._subtree.get(id(child), 0.0) for child in module.children())
        # Only a subtree with bytes in it can charge or orphan anything, so the rest
        # needs no frame and no hook.
        self.tracked = [m for m in model.modules()
                        if self._subtree.get(id(m)) and list(m.children())]
        self.stack: list[_Frame] = []
        self.credits: dict[str, float] = {}
        self.awaiting: list[float] = []  # bytes of containers still waiting for a leaf
        self.ran: set[int] = set()

    def _note(self, module: torch.nn.Module) -> None:
        """Mark a module, and everything holding it, as having run.

        The holders are marked because a container can be *iterated* instead of
        called: ``for blk in self.blocks`` never fires a hook on the ``ModuleList``,
        yet its children ran, and reading it as unrun would bill its whole subtree a
        second time where it is held.
        """
        key = id(module)
        while key is not None and key not in self.ran:
            self.ran.add(key)
            key = self._parent.get(key)

    def enter(self, module: torch.nn.Module) -> None:
        """A tracked container is starting."""
        self._note(module)
        self.stack.append(_Frame(module))

    def leaf(self, module: torch.nn.Module, name: str) -> None:
        """A leaf call is starting, and it is the first one for every open frame."""
        self._note(module)
        total = self._own.get(id(module), 0.0) + sum(self.awaiting)
        self.awaiting.clear()
        for frame in reversed(self.stack):
            if frame.first_leaf is not None:
                break  # an inner frame already saw this call, so every outer one did
            frame.first_leaf = name
            total += self._own.get(id(frame.module), 0.0)
        self.credits[name] = self.credits.get(name, 0.0) + total

    def exit(self, module: torch.nn.Module) -> None:
        """A tracked container is over; whatever inside it never ran is billed now."""
        frame = self.stack.pop()
        missing = sum(self._subtree.get(id(child), 0.0) for child in module.children()
                      if id(child) not in self.ran)
        if frame.first_leaf is None:
            # Nothing ran inside, so neither its own bytes nor these were credited.
            orphan = self._own.get(id(module), 0.0) + missing
            if orphan:
                self.awaiting.append(orphan)
        elif missing:
            self.credits[frame.first_leaf] += missing


def install_leaf_hooks(model: torch.nn.Module, on_pre, on_post,
                       charge: OwnerCharge) -> list:
    """Hook every leaf module call, plus every container with parameters under it.

    The leaf hooks are the layer profile; the container hooks only maintain
    :class:`OwnerCharge`'s stack, since a leaf cannot see a weight held above it.
    ``on_pre`` has to hand the charge the name it minted (``charge.leaf(module,
    name)``), which is what the credits are keyed by. Returns handles the caller
    must remove.
    """
    leaves = leaf_modules(model)
    handles = [m.register_forward_pre_hook(on_pre, with_kwargs=True) for m in leaves]
    handles += [m.register_forward_hook(on_post, with_kwargs=True) for m in leaves]
    handles += [m.register_forward_pre_hook(
        lambda mod, args, kwargs: charge.enter(mod), with_kwargs=True)
        for m in charge.tracked]
    handles += [m.register_forward_hook(
        lambda mod, args, kwargs, out: charge.exit(mod), with_kwargs=True)
        for m in charge.tracked]
    return handles


class _Node:
    """A captured producer: one leaf module call, or one torch-level function.

    ``operands`` lists, for each input value the node read, the node that produced
    it and the tensor that was read -- the latter being the buffer a cut above this
    node would have to put on the wire. Resolving those producers to *layers* is
    the post-pass's job (:func:`_cut`), because it needs to know which layer reads
    which value first and that is only visible once the forward has run.

    Weights are not carried here: a module call's footprint comes from
    :attr:`OwnerCharge.credits` once the forward is over.
    """

    __slots__ = ("kind", "name", "operands")

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind  # "module" or "op"
        self.name = name
        self.operands: list[tuple[int, torch.Tensor]] = []


class _Capture(TorchFunctionMode):
    """Tag every tensor with its producing node for the duration of one forward.

    Leaf modules are hooked from outside (see :func:`build_structure_graph`)
    because ``__torch_function__`` cannot see a module call. Both feeds append to
    one node list, so a function sitting between two modules resolves like any
    other edge.
    """

    def __init__(self, charge: OwnerCharge) -> None:
        super().__init__()
        self.nodes: list[_Node] = []
        self.producer: dict[int, int] = {}  # id(tensor) -> node index, latest write
        self.objects: dict[int, torch.Tensor] = {}  # id -> tensor, keeps ids stable
        self.module_calls: list[int] = []
        self.pending: tuple | None = None  # the leaf call whose hook is open
        self.group: dict[int, int] = {}  # op node -> the module node that runs it, from _cut
        self.namer = Namer()
        self.charge = charge

    def _merge(self, node: _Node, tensors: Iterable[torch.Tensor]) -> None:
        """Record which node each of ``node``'s input values came from, once each."""
        known = {producer_id for producer_id, _ in node.operands}
        for tensor in tensors:
            producer_id = self.producer.get(id(tensor))
            if producer_id is None or producer_id in known:
                continue
            known.add(producer_id)
            node.operands.append((producer_id, tensor))

    def _tag(self, node_id: int, outputs: Any) -> None:
        for tensor in _iter_tensors(outputs):
            self.objects[id(tensor)] = tensor
            self.producer[id(tensor)] = node_id

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        node = _Node("op", getattr(func, "__name__", repr(func)))
        # Read provenance before calling: an in-place function overwrites the map
        # entry of one of its own operands.
        self._merge(node, _iter_tensors(args))
        self._merge(node, _iter_tensors(kwargs))
        self.nodes.append(node)
        node_id = len(self.nodes) - 1
        outputs = func(*args, **kwargs)
        self._tag(node_id, outputs)
        self.on_call(node, node_id, func, args, kwargs, outputs)
        return outputs

    def on_call(self, node: _Node, node_id: int, func, args, kwargs,
                outputs) -> None:
        """Hook for an observer that wants the call itself, not just its provenance.

        :class:`_Node` records *which* values a call read; a replay of the model
        needs the call as an instruction -- the callable, the non-tensor arguments,
        and which output is which. Only :mod:`slicer.faas.program` wants that, and it
        pays for nothing here: an empty method is not a dispatch.
        """

    def on_pre(self, module, args, kwargs):
        name = self.namer(module)
        self.charge.leaf(module, name)
        node = _Node("module", name)
        self._merge(node, _iter_tensors(args))
        self._merge(node, _iter_tensors(kwargs))
        self.nodes.append(node)
        self.module_calls.append(len(self.nodes) - 1)
        self.pending = (node, self.module_calls[-1], module, args, kwargs)

    def on_post(self, module, args, kwargs, output):
        node_id = self.module_calls[-1]
        self._tag(node_id, output)
        node, index, subject, inputs, named = self.pending
        self.on_call(node, index, subject, inputs, named, output)


def _cut(capture: _Capture) -> list[Tensor]:
    """Edges of the layer DAG: what a cut between two layers puts on the wire.

    A torch function is not a layer the partitioner can place, so it runs with the
    first layer that reads its output -- :meth:`_Node.operands` of that layer is
    walked back through such functions, and every layer reached stops the walk,
    because from there on the value's older buffers have been merged into one that
    is the only thing crossing the next cut. Byte counts are summed per producer,
    since a layer can read two separate buffers from the same one.

    The walk also decides where each torch function *runs*, which is the same
    information read the other way round: it lands in ``capture.group``, and a
    replay of the service (:mod:`slicer.faas.program`) executes a fused call in the
    slice that owns the layer it was folded into.
    """
    nodes = capture.nodes
    group = capture.group  # filled here so a replay of the model can reuse the mapping
    edges: list[Tensor] = []
    for module_id in capture.module_calls:
        shipped: dict[int, float] = {}
        stack = list(nodes[module_id].operands)
        while stack:
            producer_id, value = stack.pop()
            producer = nodes[producer_id]
            if producer.kind == "module":
                source = producer_id
            else:
                source = group.get(producer_id, -1)
                if source < 0:
                    group[producer_id] = module_id
                    stack.extend(producer.operands)
                    continue
            if source != module_id:
                shipped[source] = shipped.get(source, 0.0) + float(
                    value.numel() * value.element_size())
        for source, nbytes in shipped.items():
            edges.append(Tensor(src=nodes[source].name, dst=nodes[module_id].name,
                                bytes=nbytes))
    return edges


def build_structure_graph(model: torch.nn.Module, inputs, name: str = "",
                          device: str = "cpu") -> LayerGraph:
    """Profile-free layer graph: nodes and byte-carrying edges, no timings.

    ``inputs`` may be a tuple of positional arguments or a dict of keyword
    arguments. Nodes come out carrying ``params`` only; ``memory`` and ``time``
    stay zero until the service profiler fills them, so
    :meth:`~slicer.partition.graph.LayerGraph.total_time` on a fresh graph is
    deliberately 0 rather than a guess.
    """
    was_training = model.training
    model.eval().to(device)
    inputs = move_tensors(inputs, device)

    capture = _Capture(OwnerCharge(model))
    capture.namer = Namer(module_paths(model))
    handles = install_leaf_hooks(model, capture.on_pre, capture.on_post,
                           capture.charge)
    try:
        with capture, torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    if was_training:
        model.train()

    layers: list[Layer] = []
    credits = capture.charge.credits
    for node_id in capture.module_calls:
        node = capture.nodes[node_id]
        layers.append(Layer(name=node.name, memory=0.0, time=0.0,
                            params=credits.get(node.name, 0.0)))
    return LayerGraph(layers, _cut(capture), name or type(model).__name__)
