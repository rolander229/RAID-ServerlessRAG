"""ViT-L/16 partitioned at arbitrary encoder-block boundaries (Gillis regime).

Why a second large-model pipeline
---------------------------------
``pipe_d.py`` shards a ResNet into three *fixed* stages (stem/mid/late). That is
enough to ask "does splitting reduce per-function memory", but it is not enough
to reproduce the thing Gillis is actually about: the **cost/latency trade-off as
a function of where you cut**. Three fixed stages give you at most a handful of
discrete points; a trade-off curve needs the cut position to be a free variable.

ResNet is also a poor stage for the study for a second, more basic reason: its
weights are small. Measured against this repo's 674 MB CPU function floor,
resnet152's 230 MB of weights cannot move the *billing tier* at all -- both the
monolith (674+230=904 MB) and a 2-way split (674+115=789 MB) round up to the
1024 MB tier, so the split saves exactly nothing. The cost saving in serverless
comes from the discrete memory tiers, so a model only becomes interesting once
it is large enough to push a function across a tier boundary.

ViT-L/16 is the right stage:

  * **24 encoder blocks, 48.05 MB each -- almost exactly uniform.** Depth is a
    continuous knob: cut after block 6, 12, 18, ... and every cut produces a
    different memory/cost point. 99.3% of the model (1153 of 1161 MB) lives in
    these blocks, so the shard sizes track the cut positions closely.
  * 1161 MB of weights against a 674 MB floor: the monolith needs the 2048 MB
    tier, while a 4-way split fits the 1024 MB tier. That is a 2x cost gap that
    simply does not exist for the ResNets.

How the split is realised
-------------------------
The 24 blocks are grouped into ``GROUPS`` stage operators (3 blocks each by
default). Each stage operator holds **only its own blocks**, and its shard is
written as a separate checkpoint, so a function that owns stage 5 never touches
stage 0's parameters. Architectures are built on the ``meta`` device and filled
with ``to_empty`` + ``load_state_dict``, so constructing a stage never
transiently materialises the whole model -- without that, VmHWM would record the
construction peak and the per-function memory comparison would be an artefact of
how the module was built.

The 8 groups are stage operators, not functions: which of them share a function
is a *layout* decision, made by the optimiser / experiment driver.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from workloads.pipe_a import asset_images, sink  # noqa: E402

# NOTE: ``workloads.pipes`` is imported lazily inside ``vit_spec()``. pipes.py
# imports this module at its own line ~27, *before* it defines CPU/GPU/OpIO at
# line ~29, so a module-level import here would be a circular import that fails
# on the half-initialised parent.

# Weights are fetched into the project, never into the user's home cache.
# setdefault so an explicit TORCH_HOME from the caller still wins.
_TORCH_HOME = os.path.join(os.getcwd(), "workloads_data", "torch_home")
os.environ.setdefault("TORCH_HOME", _TORCH_HOME)

SHARD_ROOT = "workloads_data/model_shards_vit"
MODEL = "vit_l_16"
N_BLOCKS = 24
GROUPS = 8
BLOCKS_PER_GROUP = N_BLOCKS // GROUPS
RES = 224
BATCH = int(os.environ.get("M2S_VIT_BATCH", "4"))
HIDDEN = 1024
SEQ = (RES // 16) ** 2 + 1          # 197: 196 patches + 1 class token

# Architecture constants of ViT-L/16. Duplicated from torchvision on purpose --
# see ``_skeleton`` for why, and ``_EncoderBlock`` for what has to stay in sync.
PATCH = 16
NUM_HEADS = 16
MLP_DIM = 4096
NUM_CLASSES = 1000
NORM_EPS = 1e-6
DROPOUT = 0.0

assert GROUPS * BLOCKS_PER_GROUP == N_BLOCKS, \
    "GROUPS must divide N_BLOCKS (%d %% %d != 0)" % (N_BLOCKS, GROUPS)


def stage_names():
    return ["vit_stage%d" % i for i in range(GROUPS)]


def stage_op_names():
    return ["vit_stage%d" % i for i in range(GROUPS)]


# --------------------------------------------------------------------- spec
def vit_spec():
    """OpIO declarations.

    Every stage boundary carries a ``(B, 197, 1024)`` activation, so each cut is
    a genuine bandwidth decision rather than a rounding error. With the default
    batch that is a few MB per hop -- large enough that the trade-off is real.
    """
    from workloads.pipes import CPU, OpIO
    ops = [
        OpIO("vit_ingest", (), ("n",), CPU),
        OpIO("vit_preprocess", ("n",), ("x",), CPU),
        OpIO("vit_stage0", ("x",), ("h0",), CPU),
    ]
    for i in range(1, GROUPS):
        ops.append(OpIO("vit_stage%d" % i,
                        ("h%d" % (i - 1),), ("h%d" % i,), CPU))
    ops += [
        OpIO("vit_topk", ("h%d" % (GROUPS - 1),), (), CPU),
        OpIO("vit_sink_worker", (), (), CPU, True),
    ]
    return ops


# ------------------------------------------------------- shard architecture
class _ProjStage(nn.Module):
    """conv_proj + class token + positional embedding + the first block group.

    torchvision's ``Encoder.forward`` adds the positional embedding itself, but
    we never call it -- we drive the blocks directly, so the embedding has to be
    added here, on the first stage, and must therefore live in this shard.
    """

    def __init__(self, conv_proj, class_token, pos_embedding, blocks):
        super().__init__()
        self.conv_proj = conv_proj
        self.class_token = class_token            # nn.Parameter on the parent
        self.pos_embedding = pos_embedding        # nn.Parameter on the encoder
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        n = x.shape[0]
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embedding
        for b in self.blocks:
            x = b(x)
        return x


class _BlocksStage(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class _TailStage(nn.Module):
    """last block group + encoder norm + classification head."""

    def __init__(self, blocks, ln, head):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln = ln
        self.head = head

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x)[:, 0])


# ------------------------------------------- torchvision-free stage skeleton
# Why these duplicates exist
# --------------------------
# A stage function's *parameters* were already its own, but its *architecture*
# was not: ``_carve`` sliced a materialised ViT-L, and materialising one means
# importing torchvision. That import is 918 ms and 132 MB of RSS, and it was
# paid by every stage function for the privilege of reading two integers out of
# ``full.encoder.layers`` -- the number of blocks and their hidden size.
#
# ``_EncoderBlock`` and ``_MLPBlock`` below are torchvision's classes spelled
# with ``torch.nn`` alone. Identical submodules, identical attribute names,
# identical forward, so the per-stage checkpoints load with ``strict=True``
# unchanged -- the shard format did not move, only who can read it.
#
# This is not a micro-optimisation. It is the difference between "a stage
# function ships torchvision and 132 MB of framework it never calls" and "a
# stage function ships torch". The billing tier is decided by RSS, so an
# unchanged architecture at 132 MB less RSS is worth real money.
#
# The cost is that the two spellings must stay in sync. ``verify_skeleton``
# below is the guard: it asserts, per stage, that the skeleton and the
# torchvision carve produce the same parameter names, shapes and values.
class _MLPBlock(nn.Sequential):
    """torchvision's ``MLPBlock``, re-expressed with ``torch.nn`` only.

    ``MLPBlock`` is ``MLP(in_dim, [mlp_dim, in_dim], GELU, dropout)``, i.e. a
    five-slot ``Sequential`` whose parameter-bearing slots are 0 and 3. The
    slot layout is load-bearing: it is what the checkpoint's key names encode.
    """

    def __init__(self, in_dim, mlp_dim, dropout=DROPOUT):
        super().__init__(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, in_dim),
            nn.Dropout(dropout),
        )


class _EncoderBlock(nn.Module):
    """torchvision's ``EncoderBlock``, re-expressed with ``torch.nn`` only.

    Same submodules under the same names (``ln_1``/``self_attention``/
    ``dropout``/``ln_2``/``mlp``) and the same pre-norm residual forward, so a
    checkpoint written against one loads into the other.
    """

    def __init__(self, num_heads=NUM_HEADS, hidden_dim=HIDDEN,
                 mlp_dim=MLP_DIM, dropout=DROPOUT,
                 attention_dropout=DROPOUT):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=NORM_EPS)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=NORM_EPS)
        self.mlp = _MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x):
        torch._assert(x.dim() == 3,
                      "Expected (batch, seq, hidden) got %s" % (x.shape,))
        y = self.ln_1(x)
        y, _ = self.self_attention(y, y, y, need_weights=False)
        y = self.dropout(y)
        y = y + x
        z = self.mlp(self.ln_2(y))
        return y + z


def _skeleton(idx):
    """Stage ``idx``'s architecture, built from ``torch.nn`` -- no torchvision.

    Materialised on the ``meta`` device so construction allocates nothing: with
    real storage the random init and the loaded shard would both be live during
    ``load_state_dict`` and the peak would be ~2x the shard, which is exactly
    the artefact ``assign`` exists to avoid.
    """
    lo, hi = idx * BLOCKS_PER_GROUP, (idx + 1) * BLOCKS_PER_GROUP
    with torch.device("meta"):
        blocks = [_EncoderBlock() for _ in range(lo, hi)]
        if idx == 0:
            return _ProjStage(
                nn.Conv2d(3, HIDDEN, kernel_size=PATCH, stride=PATCH),
                nn.Parameter(torch.zeros(1, 1, HIDDEN)),
                nn.Parameter(torch.zeros(1, SEQ, HIDDEN)),
                blocks)
        if idx == GROUPS - 1:
            return _TailStage(blocks, nn.LayerNorm(HIDDEN, eps=NORM_EPS),
                              nn.Linear(HIDDEN, NUM_CLASSES))
        return _BlocksStage(blocks)


def verify_skeleton(verbose=True):
    """Assert the torch-only skeleton matches the torchvision carve.

    Two architectures agreeing on parameter *names* is what makes a checkpoint
    portable; agreeing on *values* after loading the same checkpoint is what
    makes the outputs comparable. Both are checked here, per stage, so a
    torchvision change that silently renames a submodule fails loudly at the
    next shard build instead of quietly changing every reported number.
    """
    import torchvision  # only ever imported by this verifier
    full = _full_meta()
    report = []
    for i in range(GROUPS):
        mine = _skeleton(i).state_dict()
        theirs = _carve(full, i).state_dict()
        same_keys = set(mine) == set(theirs)
        mismatched = sorted(
            k for k in set(mine) & set(theirs)
            if tuple(mine[k].shape) != tuple(theirs[k].shape))
        report.append({"stage": i, "same_keys": same_keys,
                       "n_params": len(mine),
                       "shape_mismatches": mismatched})
    if verbose:
        for r in report:
            print("  stage%d  keys=%s params=%3d shapes=%s"
                  % (r["stage"], "ok" if r["same_keys"] else "DIFFER",
                     r["n_params"],
                     "ok" if not r["shape_mismatches"]
                     else r["shape_mismatches"]))
    ok = all(r["same_keys"] and not r["shape_mismatches"] for r in report)
    if verbose:
        print("  skeleton %s (torchvision %s)"
              % ("matches" if ok else "DOES NOT MATCH",
                 torchvision.__version__))
    return ok


def _carve(full, idx):
    """Extract stage ``idx``'s parameters out of a (meta or real) full model."""
    layers = full.encoder.layers
    lo, hi = idx * BLOCKS_PER_GROUP, (idx + 1) * BLOCKS_PER_GROUP
    if idx == 0:
        return _ProjStage(full.conv_proj, full.class_token,
                          full.encoder.pos_embedding, list(layers[lo:hi]))
    if idx == GROUPS - 1:
        return _TailStage(list(layers[lo:hi]), full.encoder.ln,
                          full.heads.head)
    return _BlocksStage(list(layers[lo:hi]))


def _full_meta():
    """Full ViT-L architecture on the ``meta`` device: allocates nothing."""
    from torchvision.models import vit_l_16
    with torch.device("meta"):
        return vit_l_16(weights=None)


def shard_dir():
    return os.path.join(SHARD_ROOT, MODEL)


def shard_paths():
    d = shard_dir()
    return {i: os.path.join(d, "stage%d.pt" % i) for i in range(GROUPS)}


def ensure_shards(verbose=True):
    """Write one checkpoint per stage so a function loads only its own."""
    paths = shard_paths()
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(shard_dir(), exist_ok=True)
    from torchvision.models import ViT_L_16_Weights, vit_l_16
    full = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).eval()
    for i in range(GROUPS):
        torch.save(_carve(full, i).state_dict(), paths[i])
        if verbose:
            print("  shard stage%-2d %8.2f MB"
                  % (i, os.path.getsize(paths[i]) / 1048576.0))
    del full
    return paths


_LOADED = {}


def load_mode():
    """How a shard is materialised. ``M2S_VIT_LOAD`` selects, default ``assign``.

    ``copy``   build the stage on meta, ``to_empty`` to allocate its parameters,
               then ``load_state_dict`` (which copies). The shard's own storage
               and the destination parameter storage are both alive during the
               load, so the peak is ~2x the shard.
    ``assign`` build on meta and ``load_state_dict(..., assign=True)``, which
               *binds* the loaded tensors as the parameters instead of copying
               into freshly allocated ones. One allocation instead of two.

    That 144 MB difference is not cosmetic: a two-stage function measured
    1042 MB with ``copy`` and 898 MB with ``assign``, and the 1024 MB billing
    tier sits between them. The tier -- and with it the entire cost verdict --
    was being decided by the loader rather than by the model.
    """
    return os.environ.get("M2S_VIT_LOAD", "assign").strip().lower()


def stage_module(idx):
    """Load exactly one stage's parameters. Peak allocation = that shard.

    The architecture comes from ``_skeleton`` rather than from a carved
    torchvision model, so a stage function can be loaded, run and billed without
    torchvision ever entering ``sys.modules``. ``torch.load`` is the only heap
    allocation of shard size on this path -- which is what makes the reported
    per-function RSS a property of the split rather than of the loader.
    """
    if idx not in _LOADED:
        ensure_shards(verbose=False)
        sd = torch.load(shard_paths()[idx], map_location="cpu")
        m = _skeleton(idx)
        if load_mode() == "copy":
            m = m.to_empty(device="cpu")
            m.load_state_dict(sd, strict=True)
        else:
            m.load_state_dict(sd, assign=True)
        del sd
        _LOADED[idx] = m.eval()
    return _LOADED[idx]


def stage_param_mb(idx):
    """fp32 weight footprint of one stage -- the theoretical per-function floor."""
    m = _skeleton(idx)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n * 4 / 1048576.0


def activation_mb():
    """Bytes crossing one stage boundary, in MB."""
    return BATCH * SEQ * HIDDEN * 4 / 1048576.0


# ------------------------------------------------------------------ operator
def _prep_tensor(n):
    import cv2
    from PIL import Image
    from torchvision import transforms
    # Matches the ViT_L_16 IMAGENET1K_V1 preset (resize 242, crop 224).
    tf = transforms.Compose([
        transforms.Resize(242),
        transforms.CenterCrop(RES),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    imgs = asset_images()
    xs = []
    for i in range(n):
        im = cv2.imread(imgs[i % len(imgs)])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        xs.append(tf(Image.fromarray(im)))
    return torch.stack(xs)


def _stage_fn(idx, st):
    upstream = "x" if idx == 0 else "h%d" % (idx - 1)
    key = "h%d" % idx

    def run():
        with torch.no_grad():
            st[key] = stage_module(idx)(st[upstream])
        return st[key]
    run.__name__ = "vit_stage%d" % idx
    return run


def vit_ops(st=None):
    st = {} if st is None else st

    def ingest():
        st["n"] = BATCH
        return {"model": MODEL, "batch": BATCH, "res": RES,
                "blocks": N_BLOCKS, "groups": GROUPS}

    def preprocess():
        st["x"] = _prep_tensor(st.get("n", BATCH))
        return st["x"]

    def topk():
        with torch.no_grad():
            return st["h%d" % (GROUPS - 1)].topk(5, dim=1)

    out = [("vit_ingest", ingest), ("vit_preprocess", preprocess)]
    out += [("vit_stage%d" % i, _stage_fn(i, st)) for i in range(GROUPS)]
    out += [
        ("vit_topk", topk),
        ("vit_sink_worker", lambda: sink("vit-l", "vit")),
    ]
    return out


# ------------------------------------------------------------------- layouts
def groups_from_sizes(sizes, light="ride"):
    """Contiguous function layout described by how many stages each holds.

    ``sizes`` must sum to ``GROUPS``. This is the single place where a layout is
    turned into operator groups, so the named helpers below cannot drift apart
    from it.

    ``light`` decides where the four non-model operators go:

    ``ride`` (default)
        ingestion and preprocessing join the first function, top-k and the sink
        join the last. Function count == number of model shards, which is the
        quantity the trade-off is plotted against.

    ``separate``
        they become two extra functions of their own. This costs two more
        functions but keeps the model functions free of the image-decode and
        top-k buffers, and that turns out to matter: a pure two-block function
        measured 1013-1020 MB while a two-block function carrying
        ingest/preprocess measured 1068.8 MB. The 1024 MB billing tier sits
        between them, so "where the light operators go" flips the tier of the
        whole layout -- and with it the cost verdict.
    """
    if sum(sizes) != GROUPS or any(s < 1 for s in sizes):
        raise ValueError("sizes must be positive and sum to %d, got %r"
                         % (GROUPS, sizes))
    names = ["vit_stage%d" % i for i in range(GROUPS)]
    parts, i = [], 0
    for s in sizes:
        parts.append(names[i:i + s])
        i += s
    if light == "ride":
        parts[0] = ["vit_ingest", "vit_preprocess"] + parts[0]
        parts[-1] = parts[-1] + ["vit_topk", "vit_sink_worker"]
    elif light == "separate":
        parts = ([["vit_ingest", "vit_preprocess"]] + parts
                 + [["vit_topk", "vit_sink_worker"]])
    else:
        raise ValueError("light must be 'ride' or 'separate', got %r" % light)
    return parts


def balanced_sizes(k):
    """Stage counts for ``k`` functions holding as close to equal shares as
    the 8 stage operators allow."""
    if k < 1 or k > GROUPS:
        raise ValueError("k must be in 1..%d, got %d" % (GROUPS, k))
    base, extra = divmod(GROUPS, k)
    return [base + (1 if j < extra else 0) for j in range(k)]


def balanced_groups(k):
    """Split the GROUPS stage operators into ``k`` contiguous, near-equal parts."""
    return groups_from_sizes(balanced_sizes(k))


def skewed_groups(cut_idx):
    """A 2-way split that cuts after stage operator ``cut_idx`` -- unbalanced.

    Used to show that the cut *position*, not just the shard count, moves the
    cost/latency point: an unbalanced split leaves one big stage carrying most
    of the parameters and therefore most of the memory tier.
    """
    if not 1 <= cut_idx <= GROUPS - 1:
        raise ValueError("cut_idx must be in 1..%d" % (GROUPS - 1))
    return groups_from_sizes([cut_idx, GROUPS - cut_idx])


def main():
    print("model              : %s" % MODEL)
    print("encoder blocks     : %d (%.2f MB each)"
          % (N_BLOCKS, stage_param_mb(1) / BLOCKS_PER_GROUP))
    print("stage operators    : %d groups x %d blocks"
          % (GROUPS, BLOCKS_PER_GROUP))
    print("boundary payload   : %.2f MB per hop (batch %d)"
          % (activation_mb(), BATCH))
    print()
    print("skeleton vs torchvision (the checkpoints must load into both):")
    verify_skeleton()
    print()
    paths = ensure_shards()
    total_w = total_d = 0.0
    for i in range(GROUPS):
        w = stage_param_mb(i)
        d = os.path.getsize(paths[i]) / 1048576.0
        total_w += w
        total_d += d
        print("  stage%-2d  %8.2f MB weights  (%8.2f MB on disk)" % (i, w, d))
    print("  TOTAL    %8.2f MB weights  (%8.2f MB on disk)" % (total_w, total_d))
    print()
    for k in (1, 2, 3, 4, 6, 8):
        g = balanced_groups(k)
        print("k=%d -> %d fn  %s" % (k, len(g), [len(x) for x in g]))


if __name__ == "__main__":
    main()
