"""DLRM-style recommendation, with the embedding table sharded by hash range.

Why this pipeline
-----------------
Every large model the solver has seen so far is sharded the same way: a sequence
of layers is cut into contiguous runs, and a function owns the layers in its run.
DLRM breaks that pattern. Its bulk is not layers at all -- it is a single
embedding table whose size is chosen by the catalogue, not by the arithmetic --
and the table admits a shard that is *not* contiguous in the operator graph: any
hash range is as good as any other, and every range is needed by every request.

That makes the boundary structure different in kind:

  * ``pipe_e``'s cut ships an activation that is the same size at every cut.
  * here, each table shard answers for the ids that hash into its range and
    returns a **partial sum** of shape (batch, dim). The payload is therefore
    O(batch x dim) no matter how many lookups were performed, and it does not
    grow with the number of shards. What grows with the shard count is the
    fan-out: k shards means k partial sums to combine and k tables resident.

The regime is also the one Gillis is really about, and the one a dense model
cannot reach: the table does not fit in one function. At the default 8M rows x
64 dims the table alone is 2.05 GB fp32, so the monolith needs the 3072 MB tier
while a four-way shard fits the 1024 MB tier -- a 3x tier gap that exists because
of the *table*, not because of the model's depth.

Honesty notes
-------------
  * Embeddings are randomly initialised from a fixed seed; this is an untrained
    table, so the pipeline measures memory, transfer and time, not recall. The
    shapes and the access pattern are those of a real DLRM.
  * The dense and top MLPs are small on purpose. DLRM's cost is the table, and
    a pipeline that buried that under a large MLP would misrepresent it.

Run alone:
    PYTHONPATH=M2S/src ./.venv/bin/python -u -m workloads.pipe_g
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from workloads.pipe_a import sink  # noqa: E402

# ------------------------------------------------------------------- config
ROWS = int(os.environ.get("M2S_DLRM_ROWS", str(8 * 1024 * 1024)))
DIM = 64                          # embedding width
FIELDS = 26                       # categorical fields per request
DENSE_IN = 13                     # continuous features
BOTTOM = 512                      # bottom MLP output width
TOP = 256                         # top MLP hidden width
BATCH = int(os.environ.get("M2S_DLRM_BATCH", "4096"))
SHARDS = int(os.environ.get("M2S_DLRM_SHARDS", "4"))
SEED = 20260917

assert ROWS % SHARDS == 0, "SHARDS must divide ROWS"
ROWS_PER_SHARD = ROWS // SHARDS


# --------------------------------------------------------------- weight cache
_CACHE = {}


def _seeded(fn):
    torch.manual_seed(SEED)
    return fn()


def table(i):
    """Shard ``i``'s slice of the embedding table.

    Built from a seed that depends only on ``i``, so every process materialises
    the same numbers without shipping them -- the alternative, a checkpoint per
    shard, would put 2 GB of file I/O in the way of every measurement.
    """
    if i not in _CACHE:
        torch.manual_seed(SEED + 1000 + i)
        _CACHE[i] = nn.Embedding(ROWS_PER_SHARD, DIM)
    return _CACHE[i]


def dense_net():
    if "dense" not in _CACHE:
        _CACHE["dense"] = _seeded(lambda: nn.Sequential(
            nn.Linear(DENSE_IN, BOTTOM), nn.ReLU(),
            nn.Linear(BOTTOM, TOP), nn.ReLU())).eval()
    return _CACHE["dense"]


def top_net():
    if "top" not in _CACHE:
        _CACHE["top"] = _seeded(lambda: nn.Sequential(
            nn.Linear(TOP + DIM, TOP), nn.ReLU(),
            nn.Linear(TOP, 1))).eval()
    return _CACHE["top"]


def table_mb(i=None):
    rows = ROWS if i is None else ROWS_PER_SHARD
    return rows * DIM * 4 / 1048576.0


def total_param_mb():
    dense = sum(p.numel() for p in dense_net().parameters())
    top = sum(p.numel() for p in top_net().parameters())
    return (ROWS * DIM + dense + top) * 4 / 1048576.0


def partial_mb():
    """Bytes one shard returns: (batch, dim) fp32, independent of shard count."""
    return BATCH * DIM * 4 / 1048576.0


# ------------------------------------------------------------------ operator
def _ids():
    """Deterministic (batch, fields) ids, spread over the whole table."""
    torch.manual_seed(SEED + 7)
    return torch.randint(0, ROWS, (BATCH, FIELDS), dtype=torch.long)


def dlrm_ops(st=None):
    st = {} if st is None else st

    def ingest():
        return {"model": "dlrm", "rows": ROWS, "dim": DIM, "fields": FIELDS,
                "batch": BATCH, "shards": SHARDS}

    def dense_features():
        torch.manual_seed(SEED + 11)
        st["dense"] = torch.rand(BATCH, DENSE_IN)
        return st["dense"]

    def sparse_ids():
        st["ids"] = _ids()
        return st["ids"]

    def bottom_mlp():
        with torch.no_grad():
            st["d"] = dense_net()(st["dense"])
        return st["d"]

    def lookup(si):
        """Rows whose hash falls in this shard's range -> a partial sum.

        The mask is what makes this a *shard* rather than a duplicate: an id
        outside ``[lo, hi)`` contributes nothing. Because the reduction happens
        here, the bytes that leave are (batch, dim) regardless of how many ids
        landed in the range -- the fan-out costs functions, not bytes.
        """
        def run():
            ids = st["ids"]
            lo, hi = si * ROWS_PER_SHARD, (si + 1) * ROWS_PER_SHARD
            m = ((ids >= lo) & (ids < hi)).unsqueeze(-1).to(torch.float32)
            with torch.no_grad():
                emb = table(si)((ids - lo).clamp_(0, ROWS_PER_SHARD - 1))
            st["part%d" % si] = (emb * m).sum(1)
            return st["part%d" % si]
        run.__name__ = "dlrm_embed_s%d" % si
        return run

    def combine():
        with torch.no_grad():
            s = st["part0"]
            for i in range(1, SHARDS):
                s = s + st["part%d" % i]
            st["s"] = s
            st["x"] = torch.cat([st["d"], F.normalize(s, dim=1)], dim=1)
        return st["x"]

    def top_mlp():
        with torch.no_grad():
            st["logits"] = top_net()(st["x"])
        return st["logits"]

    def topk():
        with torch.no_grad():
            return st["logits"].squeeze(-1).topk(min(8, BATCH))

    out = [("dlrm_ingest", ingest), ("dlrm_dense_features", dense_features),
           ("dlrm_sparse_ids", sparse_ids), ("dlrm_bottom_mlp", bottom_mlp)]
    out += [("dlrm_embed_s%d" % i, lookup(i)) for i in range(SHARDS)]
    out += [
        ("dlrm_combine", combine),
        ("dlrm_top_mlp", top_mlp),
        ("dlrm_topk", topk),
        ("dlrm_log_worker", lambda: sink("dlrm", "dlrm")),
    ]
    return out


def dlrm_spec():
    """OpIO declarations. ``part{i}`` is the fan-in the combine op pays for."""
    from workloads.pipes import CPU, OpIO
    ops = [
        OpIO("dlrm_ingest", (), (), CPU),
        OpIO("dlrm_dense_features", (), ("dense",), CPU),
        OpIO("dlrm_sparse_ids", (), ("ids",), CPU),
        OpIO("dlrm_bottom_mlp", ("dense",), ("d",), CPU),
    ]
    for i in range(SHARDS):
        ops.append(OpIO("dlrm_embed_s%d" % i, ("ids",), ("part%d" % i,), CPU))
    ops += [
        OpIO("dlrm_combine", ("d",) + tuple("part%d" % i
                                            for i in range(SHARDS)),
             ("x", "s"), CPU),
        OpIO("dlrm_top_mlp", ("x",), ("logits",), CPU),
        OpIO("dlrm_topk", ("logits",), (), CPU),
        OpIO("dlrm_log_worker", (), (), CPU, True),
    ]
    return ops


def shard_layouts():
    """Named layouts. The knob is how many tables live in one function.

    Splitting the table is the only way to shrink the per-function footprint:
    the dense path is kilobytes, and every shard needs the same ids and returns
    the same-shaped partial. So a layout here is really a statement about table
    residency, and the interesting axis is 1-table-per-function versus
    2-tables-per-function.

    The lookups have to run *before* the combine that consumes their partials,
    which is why the dense head is a leading function rather than sharing one
    with the combine. A layout that ordered these the other way would look fine
    on paper and fail in a worker; ``pipes.audit_layout`` is what says so.

    Each operator appears exactly once -- an operator belongs to one function,
    so "run the head in every lookup function" is not expressible here, and is
    not what a deployment does either.
    """
    lookups = ["dlrm_embed_s%d" % i for i in range(SHARDS)]
    head = ["dlrm_ingest", "dlrm_dense_features", "dlrm_sparse_ids",
            "dlrm_bottom_mlp"]
    tail = ["dlrm_combine", "dlrm_top_mlp", "dlrm_topk", "dlrm_log_worker"]
    return {
        # One function holds every table shard: the table does not fit.
        "G1_monolith": [head + lookups + tail],
        # One table per function: the shard count is the function count, and
        # the model functions carry nothing but their own slice of the table.
        "G1plusK_table": [head] + [[n] for n in lookups] + [tail],
        # Same table split, but the dense head rides with the first shard, so
        # there is one function fewer and one hop fewer.
        "G1plusK_head_fused": ([head + [lookups[0]]]
                               + [[n] for n in lookups[1:]] + [tail]),
        # Two tables per function: half the functions to pay for the init of,
        # at the price of two tables resident in the same place.
        "G2_table": [head] + [lookups[i:i + 2] for i in range(0, SHARDS, 2)] \
                    + [tail],
    }


def main():
    print("DLRM-like  rows=%d dim=%d fields=%d batch=%d dense_in=%d"
          % (ROWS, DIM, FIELDS, BATCH, DENSE_IN))
    print("           table sharded %d ways, %d rows per shard"
          % (SHARDS, ROWS_PER_SHARD))
    print()
    print("  embedding table   %9.2f MB fp32  (%d shards x %.2f MB)"
          % (table_mb(), SHARDS, table_mb(0)))
    print("  dense + top MLPs  %9.2f MB"
          % ((total_param_mb() - table_mb()), ))
    print("  TOTAL             %9.2f MB fp32" % total_param_mb())
    print()
    print("  boundary payload per lookup shard: %.3f MB  (batch x dim, fp32)"
          % partial_mb())
    print("    -> independent of lookup count and of shard count; only the")
    print("       number of partials to combine (= shards) depends on layout")
    print()
    for label, groups in sorted(shard_layouts().items()):
        print("  %-16s %d fn  %s" % (label, len(groups), [len(g) for g in groups]))


if __name__ == "__main__":
    main()
