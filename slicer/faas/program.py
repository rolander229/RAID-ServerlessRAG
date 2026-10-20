"""Run one slice's part of a service, which means knowing which calls are its own.

The partitioner decides where to cut; something still has to *execute* the answer.
Neither of the tools that normally does this works on the paper's benchmark:
``torch.fx`` cannot trace the LSTM/GRU stacks, the graph convolutions or the
transformer blocks (see :mod:`slicer.models.graphify`), and ``torch.jit`` freezes
the branch it happened to take. So this module records rather than traces: it
watches one real forward, keeps every call -- the leaf modules and the torch
functions between them -- as a step with its arguments, and replays a *subset* of
the steps as one slice.

Which subset is the whole question, and it is answered once: :func:`_cut` folds
every torch function into the layer that has to run it, because a fused call cannot
be placed on either side of a cut. Read that map the other way round, every step has
an owner, and a slice's steps are the ones it owns in the order they ran. The split
this executes is therefore the split Eq. (4) billed, which is the only reason a
measured latency and a predicted one would mean the same thing.

Values are named by integer keys instead of held by reference, so a step's
arguments are a template with :class:`Ref` placeholders. Replay fills them from the
request's environment and drops a value as soon as the last step that reads it has
run. That last part is not an optimisation: a slice that kept every intermediate
would peak at the whole service's footprint, and the memory quota it is billed
against -- the quantity every price in this repository multiplies by time -- would
then measure the runtime rather than the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch

from ..models.graphify import (Namer, OwnerCharge, _Capture, _cut, _iter_tensors,
                               install_leaf_hooks, module_paths, move_tensors)
from ..partition.graph import Layer, LayerGraph

__all__ = ["Program", "Ref", "SliceProgram", "Step", "record", "run_chain"]


@dataclass(frozen=True)
class Ref:
    """One recorded value, read by key out of the request's environment."""

    key: int


def _walk(template: Any, leaf: Callable[[Any], Any],
          container: Callable[[type, Any], Any]) -> Any:
    """Apply ``leaf`` to every non-container and rebuild every container around it.

    A recorded argument is a tree of tensors, numbers, strings, ``torch.Size`` and
    keyword flags; only the containers are worth recursing into, and a container
    that will not take the rebuilt items -- a ``defaultdict`` with a factory, say --
    comes back as a plain list or dict rather than raising.
    """
    if isinstance(template, (list, tuple)):
        items = [_walk(value, leaf, container) for value in template]
        try:
            return container(type(template), items)
        except TypeError:
            return items
    if isinstance(template, dict):
        items = {k: _walk(v, leaf, container) for k, v in template.items()}
        if type(template) is dict:
            return items
        try:
            return container(type(template), items)
        except TypeError:
            return items
    return leaf(template)


def _fill(template: Any, env: Mapping[int, Any]) -> Any:
    """Replace every :class:`Ref` in a recorded argument structure with its value."""
    return _walk(template, lambda value: env[value.key] if isinstance(value, Ref) else value,
                 lambda cls, items: cls(items))


def _keys_of(template: Any) -> tuple[int, ...]:
    """Every key a template reads, in the order it reads them."""
    found: list[int] = []

    def leaf(value: Any) -> Any:
        if isinstance(value, Ref):
            found.append(value.key)
        return value

    _walk(template, leaf, lambda cls, items: items)
    return tuple(found)


def _tensors_of(template: Any) -> list[torch.Tensor]:
    """Every tensor a template holds by value rather than by key: weights, buffers."""
    found: list[torch.Tensor] = []

    def leaf(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            found.append(value)
        return value

    _walk(template, leaf, lambda cls, items: items)
    return found


@dataclass
class Step:
    """A recorded call: one leaf module's forward, or one torch function."""

    layer: str | None  # the node's name for a module call, None for a fused op
    func: Any  # the leaf module itself, or the torch callable
    args: Any
    kwargs: Any
    writes: tuple[int, ...] = ()  # value keys, in the order the result yields tensors
    reads: tuple[int, ...] = ()  # value keys the call consumed
    node: int = -1  # index into the capture's node list, where the owner came from

    def run(self, env: dict[int, Any]) -> None:
        out = self.func(*_fill(self.args, env), **_fill(self.kwargs, env))
        for key, value in zip(self.writes, _iter_tensors(out)):
            env[key] = value


class _Recorder(_Capture):
    """The provenance capture of :func:`~slicer.models.graphify.build_structure_graph`, plus the calls."""

    def __init__(self, charge: OwnerCharge, inputs: Any) -> None:
        super().__init__(charge)
        self.steps: list[Step] = []
        self.keys: dict[int, int] = {}  # id(tensor) -> value key, once per object
        self.count = 0
        # The request *is* the program's input, so its tensors are named before the
        # forward starts; anything else is either produced by a step or a constant.
        for tensor in _iter_tensors(inputs):
            self.keys[id(tensor)] = self._fresh()
            self.objects[id(tensor)] = tensor  # so a slot can be sized for it later

    def _fresh(self) -> int:
        self.count += 1
        return self.count - 1

    def _key(self, tensor: torch.Tensor) -> int:
        found = self.keys.get(id(tensor))
        if found is None:
            found = self.keys[id(tensor)] = self._fresh()
            self.objects[id(tensor)] = tensor  # a key is only stable while it lives
        return found

    def _template(self, obj: Any) -> Any:
        def leaf(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            if self.producer.get(id(value)) is None and id(value) not in self.keys:
                return value  # a weight or a buffer: the same object every request
            return Ref(self._key(value))

        return _walk(obj, leaf, lambda cls, items: cls(items))

    def on_call(self, node, node_id, func, args, kwargs, outputs) -> None:
        template = self._template(args)
        named = {name: self._template(value) for name, value in (kwargs or {}).items()}
        self.steps.append(Step(
            layer=node.name if node.kind == "module" else None,
            func=func, args=template, kwargs=named,
            writes=tuple(self._key(t) for t in _iter_tensors(outputs)),
            reads=_keys_of(template) + _keys_of(named), node=node_id))


@dataclass
class SliceProgram:
    """One vertical slice as an executable: its steps, whose values it sends, what it holds.

    A slice names its steps by position in the program and holds them by value, so a
    caller that keeps only this can drop the whole program: the steps of the other
    slices hold the weights they read, and until they go the memory a quota is
    supposed to measure is still resident.
    """

    index: int
    layers: tuple[str, ...]
    steps: tuple[int, ...]  # positions in the program, which is what :attr:`drop` keys
    owned: tuple[Step, ...] = ()  # the same steps, by value, in that order
    inbound: tuple[tuple[int, tuple[int, ...]], ...] = ()  # (peer slice, keys) to await
    outbound: tuple[tuple[int, tuple[int, ...]], ...] = ()  # (peer slice, keys) to send
    request: tuple[int, ...] = ()  # keys filled from the arriving request
    result: tuple[int, ...] = ()  # keys the service returns, when they are made here
    keeps: frozenset[int] = frozenset()  # id() of every weight and buffer read here
    drop: Mapping[int, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @torch.no_grad()  # once per request, not once per step: the graph is built either
    def run(self, env: dict[int, Any]) -> None:
        """Execute the slice, freeing every value as soon as nothing here reads it.

        ``env`` arrives holding the request's tensors and the inbound activations and
        leaves holding the outbound ones -- plus the result, on the last slice.
        Everything else is gone, so what the process is resident with is this
        slice's footprint rather than the service's.
        """
        keep = set(self.result) | {k for _, keys in self.outbound for k in keys}
        for position, step in zip(self.steps, self.owned):
            step.run(env)
            for key in self.drop.get(position, ()):
                if key not in keep:
                    env.pop(key, None)


@dataclass
class Program:
    """A service as recorded steps, and the arithmetic that cuts them."""

    graph: LayerGraph
    steps: tuple[Step, ...]
    inputs: tuple[int, ...]  # the request's value keys, in argument order
    results: tuple[int, ...]  # the keys of the value the whole service returns
    template: Any = None  # the returned structure, with refs for its tensors
    model: Any = None  # kept so a slice can drop the weights it does not own
    owners: tuple[str | None, ...] = ()  # per step: the layer whose slice runs it
    sizes: Mapping[int, float] = field(default_factory=dict)  # value key -> bytes

    @torch.no_grad()
    def run(self, inputs: Any) -> Any:
        """Replay every step: the fidelity check the runtime leans on."""
        env = dict(zip(self.inputs, _iter_tensors(inputs)))
        for step in self.steps:
            step.run(env)
        return _fill(self.template, env)

    def slices(self, groups: Iterable[Iterable[str]]) -> list[SliceProgram]:
        """Cut the program at the layer groups a partition plan named.

        A value crosses from the slice that wrote it to the slice that reads it, and
        a reader takes it from the *latest* writer before it -- which is what lets an
        in-place update, a residual add into a buffer another slice allocated, keep
        one key and still land on the right side of a cut.
        """
        plan = [tuple(names) for names in groups]
        home: dict[str, int] = {}
        for index, names in enumerate(plan):
            for name in names:
                if name in home:
                    raise ValueError("layer %r is claimed by two slices" % name)
                home[name] = index
        unknown = [name for name in self.graph.order if name not in home]
        if unknown:
            raise ValueError("the plan does not cover %s" % ", ".join(unknown[:5]))
        where = tuple(home[owner] if owner is not None else -1 for owner in self.owners)
        # A step whose owner no slice claimed has nowhere to run, so an empty part is
        # a real possibility -- but only one part per part of the plan, hence the
        # count of groups rather than of layers.
        mine: dict[int, list[int]] = {index: [] for index in range(len(plan))}
        for position, owner in enumerate(self.owners):
            if owner is not None:
                mine[home[owner]].append(position)

        writer: dict[int, int] = {}
        inbound: dict[int, dict[int, set[int]]] = {i: {} for i in mine}
        outbound: dict[int, dict[int, set[int]]] = {i: {} for i in mine}
        for position, step in enumerate(self.steps):
            reader = where[position]
            for key in step.reads:
                source = writer.get(key)
                if source is None:
                    continue  # the request's own tensor, or a weight: no slice wrote it
                from_slice = where[source]
                if reader >= 0 and from_slice >= 0 and from_slice != reader:
                    inbound[reader].setdefault(from_slice, set()).add(key)
                    outbound[from_slice].setdefault(reader, set()).add(key)
            for key in step.writes:
                writer[key] = position
        out: list[SliceProgram] = []
        for index in sorted(mine):
            steps = tuple(mine[index])
            made = set(steps)
            shipped = {k for keys in outbound[index].values() for k in keys}
            result = tuple(key for key in self.results if writer.get(key, -1) in made)
            last: dict[int, int] = {}
            for position in steps:
                for key in self.steps[position].reads + self.steps[position].writes:
                    last[key] = position
            drop: dict[int, list[int]] = {}
            for key, position in last.items():
                if key in shipped or key in result:
                    continue  # still needed after the last step that touches it
                drop.setdefault(position, []).append(key)
            out.append(SliceProgram(
                index=index,
                layers=tuple(name for name in self.graph.order if home[name] == index),
                steps=steps,
                inbound=tuple((peer, tuple(sorted(keys)))
                              for peer, keys in sorted(inbound[index].items())),
                outbound=tuple((peer, tuple(sorted(keys)))
                               for peer, keys in sorted(outbound[index].items())),
                request=tuple(key for key in self.inputs
                              if any(key in self.steps[p].reads for p in steps)),
                result=result,
                keeps=self._keeps(steps),
                owned=tuple(self.steps[p] for p in steps),
                drop={position: tuple(keys) for position, keys in drop.items()}))
        return out

    def _keeps(self, steps: tuple[int, ...]) -> frozenset[int]:
        """Every value the slice reads that it was not handed: weights and buffers.

        A leaf module's own parameters are not arguments, so they are absent from
        the templates -- the module is there, and its weights go with it.
        """
        keep: set[int] = set()
        for position in steps:
            step = self.steps[position]
            for value in (step.args, step.kwargs):
                keep.update(id(t) for t in _tensors_of(value))
            if isinstance(step.func, torch.nn.Module):
                keep.update(id(p) for p in step.func.parameters(recurse=False))
        return frozenset(keep)

    def release(self, keep: Iterable[int]) -> float:
        """Drop every parameter and buffer this slice does not own, returning bytes.

        A child process that holds the whole model holds its whole weight file, and
        then its memory quota says nothing about the slice. The steps only ever call
        the leaves they own and read the tensors they kept, so the rest is dead
        -- and freeing it is what makes ``RLIMIT_AS`` a test rather than a formality.
        """
        keep = set(keep)
        freed = 0.0
        for module in self.model.modules():
            for table in (module._parameters, module._buffers):
                for name, value in list(table.items()):
                    if value is None or id(value) in keep:
                        continue
                    freed += float(value.numel() * value.element_size())
                    table[name] = None
        return freed


def record(model: torch.nn.Module, inputs: Any, name: str = "",
           device: str = "cpu") -> Program:
    """One forward as an executable program, with :func:`_cut`'s graph beside it.

    The graph and the steps come out of the same pass, so a node name means the same
    call in both, which is what lets a plan written against one run against the
    other.
    """
    was_training = model.training
    model.eval().to(device)
    inputs = move_tensors(inputs, device)
    capture = _Recorder(OwnerCharge(model), inputs)
    capture.namer = Namer(module_paths(model))
    handles = install_leaf_hooks(model, capture.on_pre, capture.on_post, capture.charge)
    try:
        with capture, torch.no_grad():
            result = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    if was_training:
        model.train()

    layers = [Layer(name=capture.nodes[i].name, memory=0.0, time=0.0,
                    params=capture.charge.credits.get(capture.nodes[i].name, 0.0))
              for i in capture.module_calls]
    graph = LayerGraph(layers, _cut(capture), name or type(model).__name__)
    owners = [step.layer if step.layer is not None
              else capture.nodes[capture.group[step.node]].name
              if step.node in capture.group else None
              for step in capture.steps]
    # A call no layer folds in is read by nobody but the caller: it runs where it
    # already sits, with the layer that owns the value around it.
    following = None
    for position in range(len(owners) - 1, -1, -1):
        if owners[position] is None:
            owners[position] = following
        else:
            following = owners[position]
    preceding = next((owner for owner in owners if owner is not None), None)
    for position in range(len(owners)):
        if owners[position] is None:
            owners[position] = preceding
        else:
            preceding = owners[position]
    return Program(graph=graph, steps=tuple(capture.steps),
                   inputs=tuple(capture.keys[id(t)] for t in _iter_tensors(inputs)),
                   results=tuple(capture._key(t) for t in _iter_tensors(result)),
                   template=capture._template(result), model=model,
                   owners=tuple(owners),
                   sizes={key: float(value.numel() * value.element_size())
                          for ident, key in capture.keys.items()
                          for value in (capture.objects.get(ident),)
                          if value is not None})


def run_chain(program: Program, groups: Iterable[Iterable[str]],
              inputs: Any) -> Any:
    """Run the service as a chain of slices, one environment each, values shipped across.

    The single-process form of what :mod:`slicer.faas.runtime` does with processes and
    channels, so that a test can say what the runtime is meant to guarantee: a cut
    moves values and memory around, and does not change what the service computes.
    """
    cases = program.slices(groups)
    envs: list[dict[int, Any]] = [{} for _ in cases]
    request = dict(zip(program.inputs, _iter_tensors(inputs)))
    for case, env in zip(cases, envs):
        for key in case.request:
            env[key] = request[key]
        for peer, keys in case.inbound:
            for key in keys:
                env[key] = envs[peer][key]
        case.run(env)
    merged: dict[int, Any] = {}
    for env in envs:
        merged.update(env)
    return _fill(program.template, merged)
