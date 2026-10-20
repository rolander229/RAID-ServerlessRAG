"""Gillis-style pipeline: one large CNN, partitioned across functions.

Why this pipeline exists
------------------------
Every other pipeline in this repo is small enough that a single function can
comfortably hold it on a V100, so a layout optimiser has no *reason* to split
them -- and indeed the old solver either fused everything or collapsed to a
monolith. Gillis (ICDCS'21) describes the opposite regime: a serverless
function has a bounded memory ceiling, so a large model cannot be served by one
function at all and must be partitioned, each function holding only its own
layers.

This pipeline makes that regime explicit and, more importantly, *parameterised*:

  * three real torchvision ResNet sizes -- resnet18 / resnet50 / resnet152 --
    are the "model scale" axis. This is what lets the study answer the natural
    objection "a V100 has 32 GB, why split at all?": because the binding
    constraint is the *function's* memory ceiling, not the GPU's capacity, and
    where that constraint bites depends on model size.
  * the model is cut into three stage shards (stem / mid / late) and each
    function loads ONLY its own shard, so per-function memory genuinely
    excludes the other stages' parameters.
  * stage architectures are built under the ``meta`` device and then filled by
    ``load_state_dict(assign=True)``, so construction never transiently
    materialises the whole model. Without that, VmHWM would record the
    construction peak and the "memory reduction from splitting" claim would be
    an artefact of how the module was built rather than a property of the split.

Honesty note: the shard files are an implementation of "each function only
holds its own layers". A real FaaS deployment would ship each stage as its own
package; we emulate that by writing one checkpoint per stage and loading exactly
one per function.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from workloads.pipe_a import asset_images, sink  # noqa: E402
from workloads.util import DEV  # noqa: E402

SHARD_ROOT = "workloads_data/model_shards"
STAGES = ("stem", "mid", "late")
RES = 224
BATCH = 8

# (block, groups, base_width, [ (inplanes, planes, blocks, stride) x4 ])
# These reproduce torchvision's own architectures exactly, so the pretrained
# checkpoints load with strict=True.
#
# Subtlety worth stating: ``base_width`` widens only the *internal* 3x3 of each
# Bottleneck. It does NOT change the block's output channels, so the inplanes
# chain is the same for resnet50/152/wide_resnet101_2 -- what changes is the
# channel count of conv2. Getting this wrong produces a shape mismatch that only
# shows up at load time (which is how it was caught).
_SPECS = {
    "resnet18": ("BasicBlock", 1, 64, [
        (64, 64, 2, 1), (64, 128, 2, 2), (128, 256, 2, 2), (256, 512, 2, 2)]),
    "resnet50": ("Bottleneck", 1, 64, [
        (64, 64, 3, 1), (256, 128, 4, 2), (512, 256, 6, 2), (1024, 512, 3, 2)]),
    "resnet152": ("Bottleneck", 1, 64, [
        (64, 64, 3, 1), (256, 128, 8, 2), (512, 256, 36, 2),
        (1024, 512, 3, 2)]),
    "wide_resnet101_2": ("Bottleneck", 1, 128, [
        (64, 64, 3, 1), (256, 128, 4, 2), (512, 256, 23, 2),
        (1024, 512, 3, 2)]),
}
MODELS = tuple(_SPECS)

_current = [os.environ.get("M2S_LARGE_MODEL", "resnet152")]

# Which device the stage functions run on.
#
# This is the crux of the "is splitting necessary?" question, so it is an
# explicit knob rather than a constant:
#
#   cpu (default) -- the Gillis regime. The model lives in the function's own
#       process, so the FaaS memory ceiling really is the binding constraint and
#       a large model cannot be served by one function at all. This is also the
#       honest way to reproduce ICDCS'21, whose functions are pure CPU.
#   gpu -- parameters live in GPU memory, which is a *separate pool* that a
#       Lambda-style RSS ceiling does not observe. On a 32 GB V100 nothing
#       forces a split for memory reasons; whatever splitting is worth there has
#       to be argued on latency/concurrency grounds instead.
_current_device = [os.environ.get("M2S_LARGE_DEVICE", "cpu").lower()]


def set_device(dev: str) -> None:
    dev = dev.lower()
    if dev not in ("cpu", "gpu"):
        raise KeyError("M2S_LARGE_DEVICE must be cpu or gpu, got %r" % dev)
    _current_device[0] = dev


def current_device() -> str:
    return _current_device[0]


def torch_device():
    return DEV if _current_device[0] == "gpu" else torch.device("cpu")


def set_model(name: str) -> None:
    if name not in _SPECS:
        raise KeyError("unknown model %r (have %s)" % (name, MODELS))
    if name != _current[0]:
        for k in [k for k in _LOADED if k.startswith(name + "/")]:
            _LOADED.pop(k)
        _current[0] = name


def current_model() -> str:
    return _current[0]


def _weights(name):
    """torchvision's weight-enum names are camel-cased, not the builder name."""
    from torchvision import models
    enum = {
        "resnet18": "ResNet18_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet152": "ResNet152_Weights",
        "wide_resnet101_2": "Wide_ResNet101_2_Weights",
        "resnext101_64x4d": "ResNeXt101_64X4D_Weights",
    }[name]
    return getattr(models, enum).DEFAULT


def _block_cls(block_name: str):
    from torchvision.models.resnet import BasicBlock, Bottleneck
    return {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}[block_name]


def _make_layer(block, inplanes, planes, blocks, stride=1, groups=1,
                base_width=64):
    """Reproduce ``torchvision.models.resnet._make_layer`` exactly."""
    outplanes = planes * block.expansion
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes))
    layers = [block(inplanes, planes, stride, downsample, groups, base_width)]
    for _ in range(1, blocks):
        layers.append(block(outplanes, planes, 1, None, groups, base_width))
    return nn.Sequential(*layers)


def _layer_spec(name, idx):
    return _SPECS[name][3][idx]


def _block_args(name):
    return {"groups": _SPECS[name][1], "base_width": _SPECS[name][2]}


class _Stem(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _make_layer(_block_cls(_SPECS[name][0]),
                                  *_layer_spec(name, 0), **_block_args(name))

    def forward(self, x):
        return self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))


class _Mid(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.layer2 = _make_layer(_block_cls(_SPECS[name][0]),
                                  *_layer_spec(name, 1), **_block_args(name))
        self.layer3 = _make_layer(_block_cls(_SPECS[name][0]),
                                  *_layer_spec(name, 2), **_block_args(name))

    def forward(self, x):
        return self.layer3(self.layer2(x))


class _Late(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.layer4 = _make_layer(_block_cls(_SPECS[name][0]),
                                  *_layer_spec(name, 3), **_block_args(name))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * _block_cls(_SPECS[name][0]).expansion, 1000)

    def forward(self, x):
        return self.fc(self.avgpool(self.layer4(x)).flatten(1))


_CLS = {"stem": _Stem, "mid": _Mid, "late": _Late}


def stage_module_arch(name, model):
    """Architecture on the ``meta`` device: no memory is allocated."""
    with torch.device("meta"):
        return _CLS[name](model)


def shard_dir(model=None):
    return os.path.join(SHARD_ROOT, model or current_model())


def shard_paths(model=None):
    d = shard_dir(model)
    return {s: os.path.join(d, s + ".pt") for s in STAGES}


def ensure_shards(model=None, verbose=True):
    """Write one checkpoint per stage so a function can load only its own."""
    model = model or current_model()
    paths = shard_paths(model)
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(shard_dir(model), exist_ok=True)

    from torchvision import models
    full = getattr(models, model)(weights=_weights(model)).eval()
    src = {
        "stem": nn.ModuleDict({"conv1": full.conv1, "bn1": full.bn1,
                               "layer1": full.layer1}),
        "mid": nn.ModuleDict({"layer2": full.layer2, "layer3": full.layer3}),
        "late": nn.ModuleDict({"layer4": full.layer4, "fc": full.fc}),
    }
    for s in STAGES:
        torch.save(src[s].state_dict(), paths[s])
        if verbose:
            print("  shard %-5s %7.2f MB"
                  % (s, os.path.getsize(paths[s]) / 1048576.0))
    del full, src
    return paths


_LOADED = {}


def stage_module(name, model=None):
    """Load exactly one stage's parameters onto the GPU.

    The architecture is built on ``meta`` (zero allocation) and then
    materialised with ``to_empty``, which reserves storage *without* running
    the random initialiser. This matters for the measurement: a normal
    construction would touch every parameter with random values before the
    checkpoint overwrote it, and that transient would show up in VmHWM and
    blur the per-function memory comparison the whole pipeline exists to make.
    """
    model = model or current_model()
    tag = "%s/%s@%s" % (model, name, current_device())
    if tag not in _LOADED:
        ensure_shards(model, verbose=False)
        m = stage_module_arch(name, model).to_empty(device="cpu")
        sd = torch.load(shard_paths(model)[name], map_location="cpu")
        m.load_state_dict(sd, strict=True)
        _LOADED[tag] = m.to(torch_device()).eval()
        del sd
    return _LOADED[tag]


def stage_param_mb(name, model=None):
    """fp32 weight footprint of one stage -- the theoretical per-function floor."""
    m = stage_module_arch(name, model or current_model())
    n = sum(p.numel() for p in m.parameters())
    del m
    return n * 4 / 1048576.0


def _prep_tensor(n):
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(RES + 8),
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
    return torch.stack(xs).to(torch_device())


def largemodel_ops(st=None):
    st = {} if st is None else st

    def ingest():
        st["n"] = BATCH
        return {"model": current_model(), "batch": BATCH, "res": RES}

    def preprocess():
        st["x"] = _prep_tensor(st.get("n", BATCH))
        return st["x"]

    def stem():
        with torch.no_grad():
            st["t1"] = stage_module("stem")(st["x"])
        return st["t1"]

    def mid():
        with torch.no_grad():
            st["t2"] = stage_module("mid")(st["t1"])
        return st["t2"]

    def late():
        with torch.no_grad():
            st["logits"] = stage_module("late")(st["t2"])
        return st["logits"]

    def topk():
        with torch.no_grad():
            return st["logits"].topk(5, dim=1)

    return [
        ("large_input_ingest", ingest),
        ("large_preprocess", preprocess),
        ("large_stem_stage", stem),
        ("large_mid_stage", mid),
        ("large_late_stage", late),
        ("large_topk_postprocess", topk),
        ("large_result_sink_worker", lambda: sink("large-model", "large")),
    ]


def main():
    print("model scale axis: resnet18 / resnet50 / resnet152")
    for model in MODELS:
        print("\n%s" % model)
        paths = ensure_shards(model)
        total_param = 0.0
        total_disk = 0.0
        for s in STAGES:
            mb = stage_param_mb(s, model)
            disk = os.path.getsize(paths[s]) / 1048576.0
            total_param += mb
            total_disk += disk
            print("  %-5s %8.2f MB fp32 weights   (%6.2f MB on disk)"
                  % (s, mb, disk))
        print("  TOTAL %8.2f MB fp32 weights   (%6.2f MB on disk)"
              % (total_param, total_disk))


if __name__ == "__main__":
    main()
