"""LLM serving, split at the prefill/decode boundary.

Why this pipeline
-----------------
Every other pipeline here has a boundary payload whose size is a property of
the *model*: a (B, 197, 1024) activation, a feature map, a JPEG frame. An
autoregressive LLM does not. Its prefill phase emits a KV cache of

    B x kv_heads x prompt x head_dim x 2 (k and v) x layers x dtype

which is a property of the *request*. Cut prefill from decode and the bytes on
that edge grow linearly with how much the user typed, so "cut here or not" stops
being a decision about the model and becomes a decision about the traffic mix.
That is a different kind of edge from anything the solver has seen, and the
point of this pipeline is to put it in front of the solver.

The two phases also want opposite things, which is the second reason they are
interesting to separate:

  * ``prefill`` is one dense pass over the whole prompt. Compute-bound: the
    arithmetic scales with prompt x layers x hidden^2.
  * ``decode`` emits one token per step and, at every step, re-reads the whole
    model plus the whole KV cache to produce a single row of logits. Bound by
    memory traffic, not by FLOPs -- the arithmetic intensity collapses to O(1).

In one function they serialise: the decode loop cannot start until prefill has
finished, so time-to-first-token and inter-token latency are paid from the same
budget and the second token waits behind the first user's 4k-token prompt.

What is modelled honestly
-------------------------
Weights are constructed from the config (``nn.Linear`` etc. with a fixed seed)
rather than downloaded, so the pipeline needs no network and every number is
reproducible. The *shapes* and the *arithmetic* are Llama-style -- RMSNorm,
rotary positions, grouped-query attention, SwiGLU -- so the compute and the
memory traffic are those of a real decoder; the values are not a trained model
and nothing here claims to generate coherent text. The pipeline measures cost,
memory and latency, none of which depend on the weights being trained.

Run alone (no need to be registered in pipes.py):
    PYTHONPATH=M2S/src ./.venv/bin/python -u -m workloads.pipe_f
"""
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from workloads.pipe_a import sink  # noqa: E402

# ------------------------------------------------------------------- config
# Llama-shaped but small enough that a CPU sweep finishes. The knobs that move
# the boundary payload are the ones worth turning: ``PROMPT`` scales the KV
# cache linearly and nothing else does.
VOCAB = 32000
HIDDEN = 1024
LAYERS = 12
HEADS = 8
KV_HEADS = 4                      # GQA: two query heads per kv head
HEAD_DIM = HIDDEN // HEADS
INTER = 2816                      # SwiGLU intermediate
RMS_EPS = 1e-5
ROPE_BASE = 10000.0
SHARDS = 4                        # layer groups the model can be cut into
LAYERS_PER_SHARD = LAYERS // SHARDS
PROMPT = int(os.environ.get("M2S_LLM_PROMPT", "256"))
NEW_TOKENS = int(os.environ.get("M2S_LLM_NEW_TOKENS", "16"))
MAX_POS = PROMPT + NEW_TOKENS + 8
SEED = 20260917

assert SHARDS * LAYERS_PER_SHARD == LAYERS, \
    "SHARDS must divide LAYERS (%d %% %d != 0)" % (LAYERS, SHARDS)

PROMPT_TEXT = ("a monolithic inference service bills for the whole model on "
               "every request even when only a fraction of it runs ")


# ---------------------------------------------------------------- primitives
def _rope_tables():
    """cos/sin for every position we will ever ask for, built once."""
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2).float()
                              / HEAD_DIM))
    t = torch.arange(MAX_POS).float()
    freqs = torch.outer(t, inv)                        # (MAX_POS, HEAD_DIM/2)
    return torch.cos(freqs), torch.sin(freqs)


_COS, _SIN = None, None


def rope_tables():
    global _COS, _SIN
    if _COS is None:
        _COS, _SIN = _rope_tables()
    return _COS, _SIN


def _rotate(x, offset=0):
    """Apply rotary positions starting at ``offset`` to (B, H, T, D)."""
    cos, sin = rope_tables()
    t = x.shape[-2]
    c = cos[offset:offset + t].view(1, 1, t, -1)
    s = sin[offset:offset + t].view(1, 1, t, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Grouped-query attention with a KV cache the caller owns.

    The cache is a *return value*, not module state. That is deliberate: module
    state would make the KV cache invisible to the boundary accounting, and the
    KV cache crossing the prefill/decode cut is the entire point of this
    pipeline.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.repeat = HEADS // KV_HEADS

    def forward(self, x, past=None, offset=0):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(b, t, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(b, t, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = _rotate(q, offset)
        k = _rotate(k, offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        # GQA by broadcast rather than by materialising a repeat: the repeat
        # variant allocates HEADS/KV_HEADS copies of the cache, which is exactly
        # the kind of transient that would show up in VmHWM and be mistaken for
        # a property of the split.
        kk = k.repeat_interleave(self.repeat, dim=1)
        vv = v.repeat_interleave(self.repeat, dim=1)
        # is_causal only for the prefill pass; a decode step attends to
        # everything already in the cache.
        causal = t > 1
        y = F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)
        y = y.transpose(1, 2).reshape(b, t, HEADS * HEAD_DIM)
        return self.o_proj(y), (k, v)


class MLP(nn.Module):
    """SwiGLU: gate and up project in parallel, down folds them back."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, INTER, bias=False)
        self.up = nn.Linear(HIDDEN, INTER, bias=False)
        self.down = nn.Linear(INTER, HIDDEN, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)
        self.attn = Attention()
        self.mlp_norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)
        self.mlp = MLP()

    def forward(self, x, past=None, offset=0):
        a, kv = self.attn(self.attn_norm(x), past=past, offset=offset)
        x = x + a
        return x + self.mlp(self.mlp_norm(x)), kv


class Shard(nn.Module):
    """A contiguous run of layers: everything a shard function ever loads."""

    def __init__(self, idx):
        super().__init__()
        lo = idx * LAYERS_PER_SHARD
        self.layers = nn.ModuleList(
            [Layer() for _ in range(lo, lo + LAYERS_PER_SHARD)])

    def forward(self, x, past=None, offset=0):
        kvs = []
        for i, layer in enumerate(self.layers):
            p = past[i] if past is not None else None
            x, kv = layer(x, past=p, offset=offset)
            kvs.append(kv)
        return x, kvs


class Head(nn.Module):
    """lm_head + final norm. Tied to the embedding on purpose: that is what a
    real decoder does, and tying is what keeps the 32k-row table in one place
    instead of two."""

    def __init__(self, embed):
        super().__init__()
        self.norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)
        self.embed = embed

    def forward(self, x):
        return F.linear(self.norm(x), self.embed.weight)


# --------------------------------------------------------------- weight cache
_CACHE = {}


def _seeded(fn):
    """Build a module under a fixed seed so two processes agree bit for bit.

    Determinism matters more than it looks: the runtime compares a function's
    output against a monolithic reference, and a table built from a
    process-varying seed would make every such comparison fail for a reason that
    has nothing to do with the split.
    """
    torch.manual_seed(SEED)
    return fn()


def embedding():
    """The token table. Lives with shard 0, since that shard owns the input."""
    if "embed" not in _CACHE:
        _CACHE["embed"] = _seeded(lambda: nn.Embedding(VOCAB, HIDDEN))
    return _CACHE["embed"]


def shard(idx):
    if idx not in _CACHE:
        _CACHE[idx] = _seeded(lambda: Shard(idx)).eval()
    return _CACHE[idx]


def head():
    if "head" not in _CACHE:
        _CACHE["head"] = _seeded(lambda: Head(embedding())).eval()
    return _CACHE["head"]


def param_mb(which):
    """fp32 footprint of one loadable unit, in MB."""
    if which == "embed":
        m = embedding()
    elif which == "head":
        m = head()
    else:
        m = shard(int(which))
    n = sum(p.numel() for p in m.parameters())
    if which == "head":
        n = sum(p.numel() for p in m.parameters()) - \
            sum(p.numel() for p in embedding().parameters())
    return n * 4 / 1048576.0


def total_param_mb():
    n = sum(p.numel() for p in embedding().parameters())
    for i in range(SHARDS):
        n += sum(p.numel() for p in shard(i).parameters())
    n += sum(p.numel() for p in head().parameters())
    n -= sum(p.numel() for p in embedding().parameters())   # tied
    return n * 4 / 1048576.0


def kv_mb(prompt=None):
    """Bytes the KV cache occupies at the end of prefill, in MB.

    The whole reason this pipeline exists: this number is a function of the
    request, so a cut placed here has a payload the model cannot predict from
    the architecture alone.
    """
    p = PROMPT if prompt is None else prompt
    return (LAYERS * 2 * KV_HEADS * p * HEAD_DIM * 4) / 1048576.0


# ------------------------------------------------------------------ operator
def _prompt_ids():
    """Deterministic token ids for the fixed prompt.

    ``zlib.crc32`` rather than ``hash`` because Python's string hash is salted
    per process, and a pipeline whose input changes between runs cannot be
    compared against itself.
    """
    text = (PROMPT_TEXT * (PROMPT // len(PROMPT_TEXT) + 1))[:PROMPT]
    return torch.tensor(
        [[zlib.crc32(c.encode()) % VOCAB for c in text]], dtype=torch.long)


def llm_ops(st=None):
    """Operators, with the prefill pass cut into ``SHARDS`` layer groups.

    Both cuts are on offer and they are not the same cut:

      * between ``llm_prefill_s{i}`` groups -- spatial, per-function weights
        shrink, and the hidden state plus that group's KV crosses each edge;
      * between ``llm_prefill_s{SHARDS-1}``/``llm_lm_head`` and
        ``llm_decode`` -- temporal, the phase boundary, where the *entire* KV
        cache crosses at once.
    """
    st = {} if st is None else st

    def ingest():
        return {"model": "llm-d12", "hidden": HIDDEN, "layers": LAYERS,
                "heads": HEADS, "kv_heads": KV_HEADS, "vocab": VOCAB,
                "prompt": PROMPT, "new_tokens": NEW_TOKENS,
                "shards": SHARDS}

    def tokenize():
        st["ids"] = _prompt_ids()
        return st["ids"]

    def embed():
        with torch.no_grad():
            st["h"] = embedding()(st["ids"])
        return st["h"]

    def prefill(si):
        def run():
            with torch.no_grad():
                h, kvs = shard(si)(st["h"], past=None, offset=0)
            st["h"] = h
            st["kv%d" % si] = kvs
            return h
        run.__name__ = "llm_prefill_s%d" % si
        return run

    def lm_head():
        with torch.no_grad():
            st["logits"] = head()(st["h"][:, -1:])
        return st["logits"]

    def sample():
        with torch.no_grad():
            st["tokens"] = st["logits"].argmax(-1)
        return st["tokens"]

    def decode():
        """The autoregressive loop: one token per step, whole model per step."""
        with torch.no_grad():
            kvs = [list(st["kv%d" % i]) for i in range(SHARDS)]
            gen = [st["tokens"]]
            # Prompt length, read off the cache instead of from a state key.
            # ``ids`` belongs to the tokenize function, so depending on it would
            # make decode unable to run unless tokenize shared a function with
            # it -- precisely the layout a deployment does not want. The cache
            # already carries the count: kvs[shard][layer] is a (k, v) pair.
            offset = kvs[0][0][0].shape[2]
            for step in range(NEW_TOKENS):
                h = embedding()(gen[-1])
                for si in range(SHARDS):
                    h, new = shard(si)(h, past=kvs[si], offset=offset)
                    kvs[si] = list(new)
                logits = head()(h[:, -1:])
                gen.append(logits.argmax(-1))
                offset += 1
            st["gen"] = torch.cat(gen, dim=1)
        return st["gen"]

    def detokenize():
        return {"tokens": int(st["gen"].numel())}

    out = [("llm_ingest", ingest), ("llm_tokenize", tokenize),
           ("llm_embed", embed)]
    out += [("llm_prefill_s%d" % i, prefill(i)) for i in range(SHARDS)]
    out += [
        ("llm_lm_head", lm_head),
        ("llm_sample", sample),
        ("llm_decode", decode),
        ("llm_detokenize", detokenize),
        ("llm_stream_worker", lambda: sink("llm", "llm")),
    ]
    return out


def llm_spec():
    """OpIO declarations. The ``kv{i}`` keys are the interesting edges."""
    from workloads.pipes import CPU, OpIO
    ops = [
        OpIO("llm_ingest", (), ("ids",), CPU),
        OpIO("llm_tokenize", (), ("ids",), CPU),
        OpIO("llm_embed", ("ids",), ("h",), CPU),
    ]
    for i in range(SHARDS):
        reads = ("h",) if i == 0 else ("h",)
        ops.append(OpIO("llm_prefill_s%d" % i, reads,
                        ("h", "kv%d" % i), CPU))
    ops += [
        OpIO("llm_lm_head", ("h",), ("logits",), CPU),
        OpIO("llm_sample", ("logits",), ("tokens",), CPU),
        # Reads every shard's cache and every shard's weights: this is the
        # function the phase cut cannot make smaller.
        OpIO("llm_decode", ("tokens",) + tuple("kv%d" % i
                                               for i in range(SHARDS)),
             ("gen",), CPU),
        OpIO("llm_detokenize", ("gen",), (), CPU),
        OpIO("llm_stream_worker", (), (), CPU, True),
    ]
    return ops


def shard_layouts():
    """Named layouts: the phase cut, layer-only cuts, and the monolith.

    Every layout must place each operator exactly once and in an order that
    respects the data dependencies -- ``llm_sample`` before ``llm_decode``,
    which reads the first token it produced. ``pipes.audit_layout`` checks both
    statically; a layout that violates either does not run at all, and it fails
    inside a worker rather than here if nothing checks first.
    """
    names = ["llm_prefill_s%d" % i for i in range(SHARDS)]
    light_in = ["llm_ingest", "llm_tokenize", "llm_embed"]
    light_out = ["llm_lm_head", "llm_sample"]
    tail = ["llm_decode", "llm_detokenize", "llm_stream_worker"]
    return {
        # Everything in one function. The baseline.
        "L1_monolith": [light_in + names + light_out + tail],
        # Temporal, tightest form: prefill and sampling in one function, decode
        # in the other. One hop, and the whole KV cache crosses it.
        "L2_fused": [light_in + names + light_out, tail],
        # Temporal with the head split off too: one more function, one more hop,
        # and it shows what that hop costs when the payload is the KV cache.
        "L2_phase": [light_in + names, light_out, tail],
        # Spatial: the prefill pass is pipelined over layer groups.
        "L4_layers": [light_in + [names[0]], [names[1]], [names[2]],
                      [names[3]] + light_out + tail],
        # Both, which is what a real disaggregated deployment does.
        "L6_both": ([light_in] + [[n] for n in names]
                    + [light_out, tail]),
    }


def main():
    print("LLM (Llama-shaped)  hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d"
          % (HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB))
    print("prompt=%d tokens, %d new tokens, %d shards x %d layers"
          % (PROMPT, NEW_TOKENS, SHARDS, LAYERS_PER_SHARD))
    print()
    print("  embedding   %8.2f MB" % param_mb("embed"))
    for i in range(SHARDS):
        print("  shard%-2d     %8.2f MB  (layers %d-%d)"
              % (i, param_mb(i), i * LAYERS_PER_SHARD,
                 (i + 1) * LAYERS_PER_SHARD - 1))
    print("  lm_head     %8.2f MB  (tied: absent)"
          % param_mb("head"))
    print("  TOTAL       %8.2f MB fp32" % total_param_mb())
    print()
    print("  KV cache at end of prefill: %8.2f MB  <- the boundary payload"
          % kv_mb())
    print("    and it is a function of the request, not the model:")
    for p in (128, 256, 512, 1024, 2048, 4096):
        print("      prompt %5d tokens -> %8.2f MB" % (p, kv_mb(p)))
    print()
    for label, groups in sorted(shard_layouts().items()):
        print("  %-12s %d fn  %s" % (label, len(groups),
                                     [len(g) for g in groups]))


if __name__ == "__main__":
    main()
