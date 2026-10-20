"""Numerical equivalence check: staged ViT == monolith ViT.

Why this exists
---------------
Every cost/latency number in the ViT study is downstream of one assumption: that
running the 8 stage operators in sequence produces *exactly* what the original
``vit_l_16`` forward produces. If the split silently drops the positional
embedding, mis-slices ``encoder.layers``, or forgets ``encoder.ln``, the pipeline
still runs and still produces plausible-looking latencies -- and the whole
trade-off analysis becomes fiction.

So this script does not check "the shapes line up". It rebuilds the full model
and compares the staged output tensor against it element-wise, and it also walks
the parameter inventory both ways to prove that the union of the shard
state_dicts is a bijection onto the original model's parameters (nothing lost,
nothing duplicated, nothing silently shared).

Three checks, in increasing strength:

  A. **Inventory**: set(stage param names) partitions set(full param names),
     with no overlap and no residue, and the byte totals agree.
  B. **Tensor identity**: every stage param is bit-identical (allclose at
     exact tolerance) to the corresponding full-model param.
  C. **Forward equivalence**: staged forward vs the real ``vit_l_16`` forward
     on the same input, max abs diff. This is the check that catches a wrong
     ``ln`` placement or a dropped pos_embedding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from workloads import pipe_e  # noqa: E402


def _named_params(obj, prefix=""):
    out = {}
    for k, v in obj.state_dict().items():
        out[prefix + k] = v
    return out


def _layer_names():
    """The real child names of ``encoder.layers``.

    They are **not** numeric: torchvision builds the encoder as an
    ``nn.Sequential`` whose children are explicitly named ``encoder_layer_0``
    .. ``encoder_layer_23``. ``_carve`` re-homes each slice into an
    ``nn.ModuleList``, which renumbers the blocks locally, so the global name
    has to be looked up positionally from the model rather than guessed.
    """
    from torchvision.models import vit_l_16
    with torch.device("meta"):
        m = vit_l_16(weights=None)
    return [n for n, _ in m.encoder.layers.named_children()]


def _global_key(idx, local_key):
    """Map a stage-local state_dict key onto the full model's name.

    Needed because a stage's ``state_dict`` is numbered *locally*: every stage's
    first block is ``blocks.0``, so a naive union over stages reports a
    collision on ``blocks.0.ln_1.weight`` in all eight shards. The local keys
    are correct for the shard files (``stage_module`` rebuilds the same local
    structure); they are simply not globally unique names.

    The mapping is the inverse of the slicing ``_carve`` performs, written out
    independently here so that an accounting bug in ``_carve`` cannot hide
    behind the same bug in the check. Numerical truth is established by [C]
    against the real ``vit_l_16.forward``; [A]/[B] are accounting.
    """
    bpg = pipe_e.BLOCKS_PER_GROUP
    base = idx * bpg
    if local_key.startswith("blocks."):
        rest = local_key.split(".", 1)[1]
        j, tail = rest.split(".", 1)
        return "encoder.layers.%s.%s" % (_LAYER_NAMES[base + int(j)], tail)
    if idx == 0:
        if local_key == "pos_embedding":
            return "encoder.pos_embedding"
        if local_key.startswith("conv_proj.") or local_key == "class_token":
            return local_key
    if idx == pipe_e.GROUPS - 1:
        if local_key.startswith("ln."):
            return "encoder." + local_key
        if local_key.startswith("head."):
            return "heads." + local_key
    raise KeyError("no global mapping for stage%d key %r" % (idx, local_key))


_LAYER_NAMES = None


def check_inventory():
    global _LAYER_NAMES
    from torchvision.models import ViT_L_16_Weights, vit_l_16
    _LAYER_NAMES = _layer_names()
    full = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).eval()
    full_sd = _named_params(full)

    # map each full-model parameter to the stage that owns it
    owner = {}
    for i in range(pipe_e.GROUPS):
        stage = pipe_e._carve(full, i)
        for k in stage.state_dict():
            g = _global_key(i, k)
            if g in owner:
                print("  !! DUPLICATE  %-40s stage%d and stage%d"
                      % (g, owner[g], i))
                return None, None
            owner[g] = i

    missing = sorted(set(full_sd) - set(owner))
    extra = sorted(set(owner) - set(full_sd))

    print("  full model params      : %d" % len(full_sd))
    print("  params covered by shards: %d" % len(owner))
    if missing:
        print("  !! LOST (in model, not in any stage):")
        for k in missing[:20]:
            print("       %s" % k)
    if extra:
        print("  !! PHANTOM (in a stage, not in model):")
        for k in extra[:20]:
            print("       %s" % k)

    n_full = sum(v.numel() for v in full_sd.values()) * 4 / 1048576.0
    n_union = sum(full_sd[k].numel() for k in owner) * 4 / 1048576.0
    print("  full bytes             : %.2f MB" % n_full)
    print("  union of shard bytes   : %.2f MB" % n_union)
    ok = not missing and not extra and abs(n_full - n_union) < 1e-6
    return full, (ok, owner)


def check_tensor_identity(full, owner):
    """Each shard param must be bit-identical to its full-model counterpart."""
    full_sd = _named_params(full)
    bad = 0
    n = 0
    for i in range(pipe_e.GROUPS):
        stage = pipe_e._carve(full, i).eval()
        for k, v in stage.state_dict().items():
            ref = full_sd[_global_key(i, k)]
            n += 1
            if not torch.equal(v, ref):
                d = (v.float() - ref.float()).abs().max().item()
                print("  !! MISMATCH stage%d %s  maxdiff=%.3e" % (i, k, d))
                bad += 1
    if not bad:
        print("  all %d shard tensors bit-identical to the monolith" % n)
    return bad == 0


def check_forward_equivalence(full, batch=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, 3, pipe_e.RES, pipe_e.RES)

    with torch.no_grad():
        ref = full(x)

    # staged forward, straight from the shard files (the real deployment path)
    h = x
    for i in range(pipe_e.GROUPS):
        st = pipe_e.stage_module(i)
        with torch.no_grad():
            h = st(h)
    got = h

    print("  monolith out shape : %s" % (tuple(ref.shape),))
    print("  staged   out shape : %s" % (tuple(got.shape),))
    if ref.shape != got.shape:
        print("  !! SHAPE MISMATCH")
        return False

    d = (ref - got).abs().max().item()
    rel = d / max(ref.abs().max().item(), 1e-12)
    print("  max abs diff       : %.3e" % d)
    print("  max rel diff       : %.3e" % rel)

    # a real correctness gate: the argmax class must match, and logits must be
    # the same to float32 accumulation order (which is not bitwise for this
    # depth of matmul chain, hence the tolerance rather than torch.equal).
    same_argmax = torch.equal(ref.argmax(1), got.argmax(1))
    print("  argmax agreement   : %s" % ("yes" if same_argmax else "NO"))
    ok = same_argmax and d < 1e-3
    return ok


def main():
    print("=" * 62)
    print("ViT-L/16 staged-vs-monolith verification")
    print("=" * 62)

    print("\n[A] parameter inventory (partition, no loss, no duplication)")
    full, inv = check_inventory()
    if full is None:
        print("\nVERDICT: FAIL (duplicate ownership)")
        return 1
    ok_a, owner = inv
    print("  -> %s" % ("PASS" if ok_a else "FAIL"))

    print("\n[B] tensor identity")
    ok_b = check_tensor_identity(full, owner)
    print("  -> %s" % ("PASS" if ok_b else "FAIL"))

    print("\n[C] forward equivalence (shard files -> full model)")
    ok_c = check_forward_equivalence(full)
    print("  -> %s" % ("PASS" if ok_c else "FAIL"))

    print("\n" + "=" * 62)
    ok = ok_a and ok_b and ok_c
    print("VERDICT: %s" % ("PASS -- the split is exactly equivalent"
                          if ok else "FAIL -- do not trust downstream numbers"))
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
