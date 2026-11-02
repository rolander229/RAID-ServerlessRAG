"""Export one weight shard per slice, so a slice server loads only its own tensors.

The partitioner decides where to cut; :mod:`slicer.faas.program` decides what to
ship across a cut; something still has to put the *weights* of each slice into a
file the slice's own process can load. :func:`export_shards` walks the recorded
program, asks every slice which weight and buffer tensors its steps read (the
``keeps`` of :class:`~slicer.faas.program.SliceProgram`), and writes those
tensors -- and nothing else -- to ``runs/weights/{model_key}/slice_i.pt``.

Two rules the export has to respect:

1. **Buffers ship with the shard.** A shard is a ``state_dict`` subset, not a
   parameter list: BatchNorm's ``running_mean``/``running_var``/``num_batches_tracked``
   are buffers a leaf module reads inside its forward, so dropping them would
   silently turn eval-mode BN into batch statistics.
2. **Tied weights must land in one shard.** A shared tensor (Qwen-family models
   tie ``lm_head.weight`` to ``model.embed_tokens.weight``) is one object under
   several names; if two slices each need it, neither shard can be loaded alone
   without dragging the peer's tensor along. The export refuses such a plan --
   the fix is to cut so that the tied pair stays in one slice (the K8s plan's
   slice-0/3 precedent). resnet50 has no tying, so this path is dormant there.

The self-check at the bottom is the load-bearing claim: a *fresh* model whose
every parameter and buffer has first been scrubbed to garbage, re-filled only
from the shard files, cut and chained, computes the original monolith forward to
within 1e-4. If a shard missed a tensor the model needed, the chain would run on
garbage and the diff would say so.

Run: python -m slicer.models.weight_shard --model resnet50 --slices 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from . import zoo
from .graphify import _iter_tensors
from ..faas.program import Program, record, run_chain
from ..partition.cost import TimingModel
from ..partition.hypad import vertical_partition

MB = 2 ** 20

# Qwen-family tied weights: lm_head.weight is the *same object* as
# model.embed_tokens.weight. Export requires the tied pair to live in the
# slice that owns embed_tokens (see module docstring, rule 2). resnet50
# (the phase-A validation target) does not exercise this path.
TIED_NOTE = ("tied weights (e.g. lm_head <-> embed_tokens on Qwen-family models) "
             "must be claimed by one slice; this export refuses plans that split them")


def _qualified_tables(model: torch.nn.Module) -> dict[int, tuple[list[str], torch.Tensor]]:
    """Map ``id(tensor)`` to ``(qualified names, tensor)`` for every table entry.

    ``model.state_dict()`` detaches, so its values are new objects and cannot be
    matched against the id()s a recorded program kept. The module tables hold the
    live objects themselves, which is exactly what :meth:`Program.release` matches.
    A tied weight appears once here, under every name that points at it.
    """
    found: dict[int, list[str]] = {}
    for module_name, module in model.named_modules():
        prefix = module_name + "." if module_name else ""
        for name, value in module._parameters.items():
            if value is not None:
                found.setdefault(id(value), []).append(prefix + name)
        for name, value in module._buffers.items():
            if value is not None:
                found.setdefault(id(value), []).append(prefix + name)
    tables: dict[int, tuple[list[str], torch.Tensor]] = {}
    for tensor_id, names in found.items():
        tables[tensor_id] = (sorted(names), _lookup_tensor(model, names[0]))
    return tables


def _lookup_tensor(model: torch.nn.Module, qualified: str) -> torch.Tensor:
    """Resolve ``conv.weight``-style paths against the live module tables."""
    module_name, _, leaf = qualified.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    for table in (module._parameters, module._buffers):
        value = table.get(leaf)
        if value is not None:
            return value
    raise KeyError(qualified)


def _slice_state_ids(model: torch.nn.Module, case) -> tuple[set[int], int]:
    """Ids of every table tensor this slice reads, plus constants held by value.

    ``case.keeps`` are id()s of tensors the slice's steps read: leaf-module
    parameters caught by :meth:`Program._keeps`, and weights/buffers that reach a
    step's arguments by value (BatchNorm's running statistics arrive inside the
    fused ``F.batch_norm`` call, so they land here too). The leaf modules the
    slice calls contribute the rest of their tables, which is what pulls in
    ``num_batches_tracked`` -- a buffer no step ever passes on, but eval-mode
    correctness of the module still presumes.
    """
    ids = set(case.keeps)
    for step in case.owned:
        if isinstance(step.func, torch.nn.Module):
            for table in (step.func._parameters, step.func._buffers):
                for value in table.values():
                    if value is not None:
                        ids.add(id(value))
    tables = _qualified_tables(model)
    known = {tensor_id for tensor_id in ids if tensor_id in tables}
    return known, len(ids) - len(known)


def _check_tied(tables: dict[int, tuple[list[str], torch.Tensor]],
                ids_by_slice: list[set[int]]) -> None:
    """Refuse a plan that claims one tensor in two shards (module docstring, rule 2)."""
    owners: dict[int, list[int]] = {}
    for index, ids in enumerate(ids_by_slice):
        for tensor_id in ids:
            owners.setdefault(tensor_id, []).append(index)
    split = {tensor_id: slices for tensor_id, slices in owners.items() if len(slices) > 1}
    if split:
        detail = {tables[t][0]: s for t, s in split.items()}
        raise ValueError("%s: %s" % (TIED_NOTE, detail))


def export_shards(prog: Program, expanded_parts: Iterable[Iterable[str]],
                  out_dir: str | Path) -> dict:
    """Write ``slice_i.pt`` per slice and a ``shard_manifest.json`` beside them.

    Each file is a plain ``torch.save`` dict: the slice's ``state`` (a state_dict
    subset keyed by qualified module path) plus the layer list the manifest's
    weights-index needs. Returns the manifest, which also records the tensors a
    slice reads that live in no module table (constants inside step templates).
    """
    parts = [list(part) for part in expanded_parts]
    cases = prog.slices(parts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = _qualified_tables(prog.model)
    ids_by_slice: list[set[int]] = []
    files: list[dict] = []
    for index, (case, part) in enumerate(zip(cases, parts)):
        ids, constants = _slice_state_ids(prog.model, case)
        ids_by_slice.append(ids)
        state: dict[str, torch.Tensor] = {}
        for tensor_id in sorted(ids):
            names, tensor = tables[tensor_id]
            for name in names:  # a tied tensor exports under every name that reads it
                state[name] = tensor.detach().clone()
        path = out_dir / ("slice_%d.pt" % index)
        torch.save({"model_key": prog.graph.name,
                    "slice_id": index,
                    "layers": list(part),
                    "state": state}, path)
        files.append({"slice_id": index, "file": path.name,
                      "layers_count": len(part), "tensors": len(state),
                      "unique_tensors": len(ids),
                      "bytes": path.stat().st_size,
                      "mb": round(path.stat().st_size / MB, 2),
                      "constants_held_by_value": constants})
        print("[shard] slice %d: %d tensors (%d unique), %.2f MB, %d layers (%s..%s)"
              % (index, len(state), len(ids), files[-1]["mb"], len(part),
                 part[0] if part else "-", part[-1] if part else "-"))

    _check_tied(tables, ids_by_slice)

    manifest = {"model_key": prog.graph.name, "expanded_parts": parts,
                "shards": files, "slices_count": len(parts)}
    (out_dir / "shard_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                 encoding="utf-8")
    return manifest


def load_shard(model: torch.nn.Module, path: str | Path) -> list[str]:
    """Fill a model's tables from one shard file; every shard name must exist.

    ``copy_`` keeps the loaded values where the recorded steps already point:
    the steps hold weight objects by value, so filling the tables in place is
    what the executing program then reads.
    """
    # A shard holds only strings and tensors, so the restricted loader is both
    # safer and warning-free (torch >= 2.4 flips the default the other way).
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    loaded: list[str] = []
    for name, tensor in state.items():
        target = _lookup_tensor(model, name)
        if tuple(target.shape) != tuple(tensor.shape) or target.dtype != tensor.dtype:
            raise ValueError("shard %s: %r shape/dtype mismatch (%s vs %s)"
                             % (path, name, tuple(target.shape), tuple(tensor.shape)))
        target.data.copy_(tensor.to(target.device))
        loaded.append(name)
    return loaded


def scrub(model: torch.nn.Module) -> None:
    """Fill every parameter and buffer with deterministic garbage, in place.

    The verification needs a model whose correctness can only come from the
    shard files. Values are benign (no zeros in a running variance, no NaNs) so
    the recorded program stays executable while being numerically wrong.
    """
    with torch.no_grad():
        for module in model.modules():
            for table in (module._parameters, module._buffers):
                for name, value in table.items():
                    if value is None:
                        continue
                    if value.is_floating_point():
                        value.fill_(0.1 if table is module._buffers else 0.5)
                    else:
                        value.fill_(0)


def expand_parts(prog: Program, slices: int) -> list[list[str]]:
    """The same plan derivation capsule/bridge.py uses: DP cut, then unfold members."""
    groups, simplified, _stats = vertical_partition(prog.graph, timing=TimingModel(),
                                                    slices=slices)
    return [[name for node in group for name in simplified.layer(node).members]
            for group in groups]


def verify(model_key: str, slices: int, device: str = "cpu",
           out_root: str | Path = "runs") -> dict:
    """record -> vertical_partition -> export -> reload-from-shards -> chained run.

    Returns the acceptance record: per-slice tensor counts, the chained-vs-
    monolith max diff, and which files carried it.
    """
    model, inputs = zoo.load(model_key, device=device)
    prog = record(model, inputs, name=model_key, device=device)
    parts = expand_parts(prog, slices)

    out_dir = Path(out_root) / "weights" / model_key
    manifest = export_shards(prog, parts, out_dir)

    # A fresh copy of the service, scrubbed so nothing but the shard files can
    # make it compute the right answer, then recorded and cut the same way.
    model2, inputs2 = zoo.load(model_key, device=device)
    scrub(model2)
    prog2 = record(model2, inputs2, name=model_key, device=device)
    for entry in manifest["shards"]:
        names = load_shard(model2, out_dir / entry["file"])
        print("[shard] slice %d: loaded %d tensors from %s"
              % (entry["slice_id"], len(names), entry["file"]))

    with torch.no_grad():
        monolith = model(*inputs)
    chained = run_chain(prog2, parts, inputs2)
    diffs = [(a.float() - b.float()).abs().max().item()
             for a, b in zip(_iter_tensors(monolith), _iter_tensors(chained))]
    max_diff = max(diffs) if diffs else 0.0
    report = {"model_key": model_key, "slices": slices, "device": device,
              "max_diff": max_diff, "per_output_max_diff": diffs,
              "shards": manifest["shards"], "out_dir": str(out_dir)}
    print("[shard] chained vs monolith max-diff = %.3e (tolerance 1e-4) -> %s"
          % (max_diff, "PASS" if max_diff <= 1e-4 else "FAIL"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export per-slice weight shards and verify them.")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--json", default=None, help="write the acceptance report here")
    args = parser.parse_args(argv)
    report = verify(args.model, args.slices, args.device, args.out_root)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["max_diff"] <= 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
