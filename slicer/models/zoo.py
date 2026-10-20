"""Registry of the paper's benchmark services, and how to get each one running.

One entry per model in Sections III-A and III-B, keyed by a short id that shows
up in ``runs/`` paths and in docs/results.md. Each entry knows the three things
the rest of the pipeline needs: how to build the module, what a representative
request looks like at a given batch size, and where its weights come from.

Weight policy. ``TORCH_HOME`` is pinned to ``weights/`` inside this repository so
a download cannot land in ``~/.cache`` or on the system disk, which is already at
91% full (docs/setup.md). Only the four torchvision CNNs have public checkpoints;
everything else is constructed with random initialisation, for the reason given
in :mod:`slicer.models.custom`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARED_TORCH_HOME = "/storage/xuedh_work2/models"
LOCAL_TORCH_HOME = os.path.join(REPO_ROOT, "weights")
TORCH_HOME = SHARED_TORCH_HOME if os.path.isdir(os.path.join(SHARED_TORCH_HOME, "hub")) else LOCAL_TORCH_HOME
os.environ.setdefault("TORCH_HOME", TORCH_HOME)

from .custom import BertStack, CnnRnn, GraphNet, InceptionTransformer  # noqa: E402
from .custom import lattice_adjacency, normalise  # noqa: E402
from .graphify import move_tensors  # noqa: E402

FAMILIES = ("CNN", "RNN", "GCN", "Transformer", "LLM")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    paper: str  # name as it appears in the paper
    family: str  # one of FAMILIES
    source: str  # "torchvision", "constructed", "huggingface"
    build: Callable[[], Any]
    inputs: Callable[[int], Any]  # batch size -> example request
    batch: int = 1
    pretrained: bool = False  # True only where public weights exist and are used
    cite: str = ""  # reference number in the paper
    note: str = ""

    def __repr__(self) -> str:  # keeps tracebacks free of closure addresses
        return "ModelSpec(%s=%s, %s)" % (self.key, self.paper, self.source)


def _torchvision(name: str) -> Callable[[], Any]:
    def build():
        import torchvision
        return getattr(torchvision.models, name)(weights="DEFAULT")
    return build


def _images(size: int) -> Callable[[int], Any]:
    def build(batch: int):
        import torch
        return (torch.randn(batch, 3, size, size),)
    return build


def _ids(seq: int, vocab: int = 30522) -> Callable[[int], Any]:
    def build(batch: int):
        import torch
        return (torch.randint(0, vocab, (batch, seq)),)
    return build


def _signal(length: int) -> Callable[[int], Any]:
    def build(batch: int):
        import torch
        return (torch.randn(batch, 1, length),)
    return build


def _nodes(count: int, features: int) -> Callable[[int], Any]:
    def build(batch: int):
        import torch
        # A graph service's request *is* the graph, so a batch here is a batch of
        # graphs sharing the adjacency -- which is what [10]'s image-as-graph
        # setting does when it classifies a mini-batch of pictures.
        return (torch.randn(batch, count, features),)
    return build


_ADJACENCY: dict[int, Any] = {}


def adjacency(nodes: int):
    """Normalised pixel-lattice adjacency for a square lattice of ``nodes``."""
    import torch
    if nodes not in _ADJACENCY:
        side = int(round(nodes ** 0.5))
        if side * side != nodes:
            raise ValueError("%d is not a square lattice, so it has no pixel graph" % nodes)
        _ADJACENCY[nodes] = normalise(lattice_adjacency(side)).to(torch.float32)
    return _ADJACENCY[nodes]


SPECS: dict[str, ModelSpec] = {}


def _add(*specs: ModelSpec) -> None:
    for spec in specs:
        if spec.key in SPECS:
            raise ValueError("duplicate model key %r" % spec.key)
        SPECS[spec.key] = spec


_add(
    # The four CNNs of Fig. 10/11: paper architectures, public weights.
    ModelSpec("vgg16", "VGG", "CNN", "torchvision",
              _torchvision("vgg16"), _images(224), cite="[32]"),
    ModelSpec("resnet50", "ResNet", "CNN", "torchvision",
              _torchvision("resnet50"), _images(224), cite="[9]"),
    ModelSpec("inception_v3", "Inception", "CNN", "torchvision",
              _torchvision("inception_v3"), _images(299), cite="[34]",
              note="Inception V3 requires 299x299; the paper's other CNNs run at "
                   "224. Eval-mode output is a tuple, handled by the profiler."),
    ModelSpec("convnext_tiny", "ConvNeXt", "CNN", "torchvision",
              _torchvision("convnext_tiny"), _images(224), cite="[18]"),
    # Reconstructed services, see slicer/models/custom.py for the rationale.
    ModelSpec("lstm_cnn", "LSTM+CNN", "RNN", "constructed",
              lambda: CnnRnn(cell="lstm"), _signal(4096), batch=8,
              cite="[3]", note="1-D CNN front end, then a 2-layer LSTM."),
    ModelSpec("gru_cnn", "GRU+CNN", "RNN", "constructed",
              lambda: CnnRnn(cell="gru"), _signal(4096), batch=8,
              cite="[6]", note="Same skeleton with GRU cells."),
    ModelSpec("gcn1", "GCN1", "GCN", "constructed",
              lambda: GraphNet(adjacency(784), (3, 32, 64, 32)),
              _nodes(784, 3), cite="[10]",
              note="Three graph convolutions over the 28x28 pixel lattice."),
    ModelSpec("gcn2", "GCN2", "GCN", "constructed",
              lambda: GraphNet(adjacency(1024), (3, 64, 128, 256, 128, 64)),
              _nodes(1024, 3), cite="[10]",
              note="Five graph convolutions over the 32x32 pixel lattice."),
    ModelSpec("bert_1_3b", "BERT-1.3B", "Transformer", "constructed",
              lambda: BertStack(hidden=2048, layers=24, heads=32, ffn=8192),
              _ids(128), cite="[5]",
              note="BERT-Large scaled to the advertised 1.3B parameters; the "
                   "cited work has no public checkpoint at that size."),
    ModelSpec("bert_3_0b", "BERT-3.0B", "Transformer", "constructed",
              lambda: BertStack(hidden=3072, layers=24, heads=32, ffn=12288),
              _ids(128), cite="[5]"),
    ModelSpec("disbert", "DisBERT", "Transformer", "constructed",
              lambda: BertStack(hidden=768, layers=12, heads=12, ffn=3072,
                                task_head=1, discrete=512),
              _ids(128), batch=8, cite="[41]",
              note="BERT-base encoder plus the discrete latent bottleneck of [41]."),
    ModelSpec("inception_transformer", "Transformer-2.6B", "Transformer",
              "constructed",
              lambda: InceptionTransformer(hidden=2048, layers=72, heads=32),
              _images(224), cite="[31]",
              note="ViT blocks whose MLP is a four-branch Inception mixture; 72 "
                   "layers is what lands on the advertised 2.6B parameters."),
)


OFFICIAL_8 = (
    "vgg16",
    "resnet50",
    "inception_v3",
    "convnext_tiny",
    "lstm_cnn",
    "gru_cnn",
    "gcn1",
    "gcn2",
)


def get(key: str) -> ModelSpec:
    try:
        return SPECS[key]
    except KeyError:
        raise KeyError("unknown model %r; available: %s"
                       % (key, ", ".join(sorted(SPECS)))) from None


def load(key: str, device: str = "cpu", dtype=None,
         batch: int | None = None, seed: int = 0):
    """Build a service plus one representative request.

    ``batch`` is overridden by the profiler when it sweeps input sizes; every
    layer here is shape-transparent, so the module does not change, only the
    request does. ``seed`` fixes the random initialisation so two runs of the
    same experiment see the same weights.
    """
    import torch

    spec = get(key)
    random.seed(seed)
    torch.manual_seed(seed)
    model = spec.build()
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    model.to(device)
    inputs = spec.inputs(spec.batch if batch is None else batch)
    return model, move_tensors(inputs, device, dtype)


def summary(key: str) -> str:
    """One-line description used by experiment logs and docs/results.md."""
    spec = get(key)
    return "%-22s %-11s %-12s cite=%-6s pretrained=%s" % (
        spec.paper, spec.family, spec.source, spec.cite, spec.pretrained)
