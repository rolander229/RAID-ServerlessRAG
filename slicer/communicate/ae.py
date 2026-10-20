"""The cross-slice tensor codec (Sec. II-D, Fig. 7).

MOPAR's third module exists because a cut has to ship an activation, and on the
paper's numbers that ship is most of the split service's latency: Fig. 3 sweeps
the compression ratio and reads a communication time that keeps falling, and the
ablations of Fig. 13b/13c take the codec away and watch the saving go with it.

Fig. 7's codec is deliberately small -- one 2D convolution in the encoder, one in
the decoder -- so that its own runtime cannot be confused with the transfer it
saves. Working in that shape makes the compression ratio exact rather than
approximate: a :math:`1\\times1` convolution from *C* channels to *C/R* encodes a
tensor into :math:`1/R` of its bytes whatever its spatial shape, and the decoder
inverts it. The ratio is therefore the knob the paper turns, and :math:`t_c` of
Eq. (5) is what it turns.

Two properties matter for the reproduction and are stated here rather than in a
comment at the call site:

* **The fit is per edge, on that edge's own activation.** A codec trained on one
  layer's output is not reused for another, and Fig. 3's ConvNeXt sweep is a
  sweep over one tensor's family.
* **Compression trades bytes for arithmetic.** The bytes fall by *R*; the encode
  and decode times do not, so a hop is only worth coding when the transfer was
  the expensive part. :meth:`Codec.accounting` reports both sides of that
  comparison, which is what lets an experiment publish a negative result instead
  of a missing one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


def _plan(rank: int, channel_axis: int) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """How to view an activation as ``(N, C, H, W)``, and how to undo that.

    A cut edge of VGG carries a feature map with its channels at axis 1; one of
    BERT or an RNN carries a sequence with its channels last, so the axis is
    theirs to declare and not this module's to guess. The permutation moves that
    axis to position 1 and leaves the remaining axes in their own order, so
    nothing is reordered beyond what a convolution requires; ``added`` counts the
    singleton spatial axes the view needs and ``inverse`` undoes the permutation.
    """
    if rank < 1:
        raise ValueError("a scalar has no channels to compress")
    if rank == 1:
        return (), 0, ()  # a lone feature vector is reshaped, not permuted
    axis = channel_axis % rank
    others = [a for a in range(rank) if a != axis]
    perm = tuple(others[:1] + [axis] + others[1:])
    return perm, 4 - rank, tuple(sorted(range(rank), key=perm.index))


def _view(tensor: torch.Tensor, perm: tuple[int, ...], added: int) -> torch.Tensor:
    """``tensor`` as a 4-D ``(N, C, H, W)`` map, without copying it."""
    out = tensor
    if out.dim() == 1:
        return out.reshape(1, out.shape[0], 1, 1)
    out = out.permute(perm)
    for _ in range(added):
        out = out.unsqueeze(-1)
    return out


def _unview(coded: torch.Tensor, shape: Sequence[int], perm: tuple[int, ...],
            added: int, inverse: tuple[int, ...]) -> torch.Tensor:
    """Undo :func:`_view`, back to ``shape``."""
    out = coded
    for _ in range(added):
        out = out.squeeze(-1)
    if len(shape) == 1:
        return out.reshape(shape)
    return out.permute(inverse)


class Autoencoder(nn.Module):
    """Fig. 7: ``encoder = Conv2d(C, C//R, 1)``, ``decoder = Conv2d(C//R, C, 1)``."""

    def __init__(self, channels: int, latent: int) -> None:
        super().__init__()
        self.channels = channels
        self.latent = latent
        self.encoder = nn.Conv2d(channels, latent, kernel_size=1)
        self.decoder = nn.Conv2d(latent, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class Codec:
    """A fitted codec for one cross-slice tensor, with its own cost accounting."""

    net: Autoencoder
    ratio: float
    shape: tuple[int, ...]
    perm: tuple[int, ...] = (0, 1, 2, 3)
    added: int = 0
    inverse: tuple[int, ...] = (0, 1, 2, 3)
    steps: int = 0
    loss: float = float("nan")
    relative_error: float = float("nan")  # ||x - decode(encode(x))|| / ||x||
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    #: Bytes the hop carried before and after coding; ratio = before / after.
    raw_bytes: int = 0
    coded_bytes: int = 0

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ship this: the latent map, in the ``(N, C', H, W)`` view it was fit on."""
        with torch.no_grad():
            return self.net.encoder(self.view(tensor))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.restore(self.net.decoder(latent))

    def view(self, tensor: torch.Tensor) -> torch.Tensor:
        """The ``(N, C, H, W)`` view the convolutions see."""
        return _view(tensor, self.perm, self.added)

    def restore(self, coded: torch.Tensor) -> torch.Tensor:
        """Undo :meth:`view` for a map with the original channel count."""
        return _unview(coded, self.shape, self.perm, self.added, self.inverse)

    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(tensor))

    def accounting(self) -> dict:
        return {
            "ratio": self.ratio,
            "raw_bytes": self.raw_bytes,
            "coded_bytes": self.coded_bytes,
            "byte_reduction": (self.raw_bytes / self.coded_bytes) if self.coded_bytes else 1.0,
            "encode_ms": self.encode_ms,
            "decode_ms": self.decode_ms,
            "codec_ms": self.encode_ms + self.decode_ms,
            "relative_error": self.relative_error,
            "loss": self.loss,
            "steps": self.steps,
        }


def _best_of(fn, warmup: int = 2, repeats: int = 5, cuda: bool = False) -> float:
    """Seconds of the fastest of ``repeats`` calls, device synchronised."""
    def sync():
        if cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
    sync()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - started)
    return best


def fit(tensor: torch.Tensor, ratio: float = 8.0, *, channel_axis: int = 1,
        steps: int = 1000, lr: float = 1e-2, seed: int = 0) -> Codec:
    """Train Fig. 7's codec on one activation, down to ``1/ratio`` of its bytes.

    ``tensor`` is what the cut would otherwise ship, in its natural shape, and
    ``channel_axis`` says where its feature dimension is: 1 for a ``Conv2d``
    output, -1 for a sequence. The target is the input itself -- an autoencoder,
    not a distillation -- and the loss is MSE, because the paper's accuracy claim
    is ultimately about what the activation still has to say.

    The two defaults are a convergence measurement, not a preference. Fig. 7's
    codec has no non-linearity, so it is a rank-``C/R`` linear map trained by
    Adam: at ``200`` steps and ``lr=1e-3`` the narrow bottleneck is still near its
    random initialisation and reconstructs a GCN1 activation at relative error
    0.56 -- a codec that discards the signal, which would make every ratio on
    Fig. 3's curve meaningless. At ``1000`` steps and ``1e-2`` the same fit reaches
    0.08, and the error still falls with more steps (0.046 at 4000). Callers who
    want the cheaper fit can ask for it, but the experiment's own rows should come
    from a converged one.
    """
    if ratio < 1:
        raise ValueError("a ratio below 1 inflates the tensor")
    shape = tuple(int(s) for s in tensor.shape)
    perm, added, inverse = _plan(len(shape), channel_axis)
    channels = shape[channel_axis % len(shape)] if len(shape) > 1 else 1
    latent = max(1, int(round(channels / ratio)))

    torch.manual_seed(seed)
    net = Autoencoder(channels, latent).to(tensor.device)
    view = _view(tensor.detach(), perm, added).to(tensor.device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    last = float("nan")
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        error = loss_fn(net(view), view)
        error.backward()
        optimiser.step()
        last = float(error.detach())

    with torch.no_grad():
        hidden = net.encoder(view).detach()
        back = net.decoder(hidden)
        denominator = float(view.norm()) or 1.0
        relative = float((view - back).norm()) / denominator
    net.eval()
    with torch.no_grad():
        encode_s = _best_of(lambda: net.encoder(view), cuda=view.is_cuda)
        decode_s = _best_of(lambda: net.decoder(hidden), cuda=view.is_cuda)

    codec = Codec(
        net=net,
        ratio=channels / latent,
        shape=shape,
        perm=perm,
        added=added,
        inverse=inverse,
        steps=steps,
        loss=last,
        relative_error=relative,
        encode_ms=encode_s * 1e3,
        decode_ms=decode_s * 1e3,
        raw_bytes=view.numel() * view.element_size(),
        coded_bytes=hidden.numel() * hidden.element_size(),
    )
    return codec
