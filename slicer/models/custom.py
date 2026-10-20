"""The benchmark services the paper names but does not publish.

Of the twelve models in Section III only four have downloadable weights (VGG,
ResNet, Inception and ConvNeXt, all in torchvision). The other four CPU services
and the four transformer services are cited to papers that never released an
implementation, so they are reconstructed here from the architecture the paper
describes and the reference it gives:

  LSTM+CNN  [3]  CNN feature encoder over a 1-D input, then an LSTM stack.
  GRU+CNN   [6]  same skeleton with GRU cells (Dey & Salem's gate variants).
  GCN1/GCN2 [10] graph convolution over a fixed adjacency (Henaff et al.), the
                 image-as-graph setting, so vertices are pixels and no dataset
                 is needed to build the graph.
  BERT-1.3B / BERT-3.0B / DisBERT / Transformer-2.6B  [5], [31], [41]
                 transformer stacks sized to the advertised parameter count.

None of these are the authors' networks, so what is reproduced is the *resource
shape* -- per-layer footprint, arithmetic cost and boundary sizes -- not their
accuracy. That is the honest reading of the paper's own CPU experiments, which
report memory utilisation and latency only; the single accuracy claim in the
paper is the autoencoder loss on ChatGLM, and that one does use real weights.

Random initialisation is therefore used throughout: footprint and cost are
functions of shape and dtype, not of weight values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- recurrent services -----------------------------------------------------


class CnnRnn(nn.Module):
    """CNN encoder feeding a recurrent stack: the paper's LSTM+CNN / GRU+CNN.

    The 1-D convolution front end plays the role of the character/patch encoder
    in [3]; the recurrent stack then reads the resulting sequence and the head
    classifies the final hidden state. The shape is what makes the service
    interesting to partition: the conv side is compute-bound with shrinking
    activations while the recurrent side is sequential with a constant per-step
    footprint, so the memory profile steps between the two halves.
    """

    def __init__(self, in_channels: int = 1, channels: tuple[int, ...] = (64, 128, 256),
                 hidden: int = 512, layers: int = 2, cell: str = "lstm",
                 classes: int = 10) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        last = in_channels
        for out in channels:
            convs += [nn.Conv1d(last, out, 5, stride=2, padding=2),
                      nn.BatchNorm1d(out), nn.ReLU()]
            last = out
        self.cnn = nn.Sequential(*convs)
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = recurrent(last, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).transpose(1, 2)
        # LSTM returns a (sequence, (h, c)) pair and the classifier only ever
        # reads the state, so stopping here is what the service actually does.
        out = self.rnn(features)
        state = out[1][0] if isinstance(out, tuple) else out
        return self.fc(state[-1])


# -- graph services ---------------------------------------------------------


def lattice_adjacency(side: int = 28, diagonal: bool = True) -> torch.Tensor:
    """Row-major pixel lattice as a symmetric adjacency with self loops.

    [10] runs graph convolution over images by treating each pixel as a vertex
    joined to its spatial neighbours. Deriving the graph from the input geometry
    means GCN1/GCN2 need no dataset and their layer shapes follow from ``side``.
    """
    n = side * side
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    adjacency = torch.zeros(n, n)
    for r in range(side):
        for c in range(side):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    adjacency[r * side + c, rr * side + cc] = 1.0
    return adjacency


def normalise(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation with self loops: D^-1/2 (A + I) D^-1/2."""
    padded = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
    degree = padded.sum(1).pow(-0.5)
    return degree.unsqueeze(1) * padded * degree.unsqueeze(0)


class GraphConv(nn.Module):
    """One graph convolution: ``relu(Â X W)`` for a fixed dense Â."""

    def __init__(self, adjacency: torch.Tensor, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.adjacency @ self.linear(x))


class GraphNet(nn.Module):
    """Stack of graph convolutions: the paper's GCN1 and GCN2 services."""

    def __init__(self, adjacency: torch.Tensor, dims: tuple[int, ...],
                 classes: int = 10) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            GraphConv(adjacency, a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Linear(dims[-1], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(-2))


# -- transformer services ---------------------------------------------------


class BertStack(nn.Module):
    """BERT-shaped encoder: embeddings, N transformer blocks, pooler, head.

    Written on ``nn.TransformerEncoderLayer`` rather than HuggingFace because the
    sizes the paper names (BERT-1.3B, BERT-3.0B) were never released by the
    cited work, so the configuration is built to a parameter count here and the
    library only gets in the way. The layer-internal structure is identical.
    """

    def __init__(self, vocab: int = 30522, hidden: int = 2048, layers: int = 24,
                 heads: int = 32, ffn: int = 8192, seq: int = 128,
                 task_head: int = 2, discrete: int = 0) -> None:
        super().__init__()
        self.width = hidden
        self.token = nn.Embedding(vocab, hidden)
        self.position = nn.Embedding(seq, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(hidden, heads, ffn, dropout=0.0,
                                       batch_first=True, activation="gelu")
            for _ in range(layers)
        )
        self.discrete = DisHead(hidden, codes=discrete) if discrete else nn.Identity()
        self.pool = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, task_head)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.norm(self.token(ids) + self.position(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(torch.tanh(self.pool(self.discrete(x[:, 0]))))


class DisHead(nn.Module):
    """Vector-quantised bottleneck on the pooled representation (DisBERT, [41]).

    The service differs from BERT-base by a discrete latent layer between the
    encoder and the similarity head, which puts a hard cut into the graph: the
    quantiser's output has no gradient path and, more to the point for the
    partitioner, a much smaller boundary tensor than the stream it sits on.
    """

    def __init__(self, hidden: int, codes: int = 512, dim: int = 256) -> None:
        super().__init__()
        self.project = nn.Linear(hidden, dim)
        self.codebook = nn.Parameter(torch.randn(codes, dim) * 0.02)
        self.restore = nn.Linear(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.project(x)
        nearest = torch.cdist(z, self.codebook).argmin(1)
        return self.restore(self.codebook[nearest])


class InceptionTransformer(nn.Module):
    """Vision transformer whose feed-forward block is a parallel-branch mixture.

    Transformer-2.6B is attributed to [31], "Inception Transformer", i.e. a ViT
    with the MLP replaced by Inception-style parallel branches. That single
    change matters for partitioning: the branches widen the graph at every block,
    so cuts have more than one candidate edge and the residual stream changes the
    per-layer footprint profile the DP searches over.
    """

    def __init__(self, image: int = 224, patch: int = 16, hidden: int = 2048,
                 layers: int = 48, heads: int = 32, branches: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, hidden, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        side = image // patch
        self.pos = nn.Parameter(torch.zeros(1, side * side + 1, hidden))
        self.blocks = nn.ModuleList(
            _InceptionBlock(hidden, heads, branches) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])


class _InceptionBlock(nn.Module):
    """Pre-norm attention block with an Inception (multi-branch) feed-forward."""

    def __init__(self, hidden: int, heads: int, branches: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        split = hidden // 2
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, split), nn.GELU())
             for _ in range(branches - 1)]
            + [nn.Conv1d(hidden, split, 3, padding=1)]
        )
        self.mix = nn.Linear(split * len(self.branches), hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        outs = [b(h.transpose(1, 2)).transpose(1, 2) if isinstance(b, nn.Conv1d)
                else b(h) for b in self.branches]
        return x + self.mix(torch.cat(outs, dim=-1))
