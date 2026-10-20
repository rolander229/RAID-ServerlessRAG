"""torchvision model zoo, constructed on first use instead of at import.

Why the factories instead of module-scope imports
-------------------------------------------------
``torchvision`` costs ~918 ms to import and ~131 MB of RSS, and this repo needs
it for exactly one reason: five pipelines want a torchvision model or a
``transforms`` preset. The ViT pipeline does not. A ViT encoder stage is a stack
of ``nn.Module`` blocks whose parameters come from a per-stage checkpoint; it
never touches ``torchvision`` at all.

With module-scope imports that distinction is invisible. Every function imports
``pipes``, ``pipes`` imports this module, so every function paid the 918 ms --
including the six ViT stage functions that cannot use a torchvision model even
if they wanted to. Measured on ViT-L/16, cold start was 2314 ms per function of
which 918 ms was a framework the function never calls, and the per-function RSS
floor was 674 MB where 512 MB is available from the torch stack alone.

Deferring to first use makes both costs proportional to what the function
actually runs. Every public name keeps its signature -- this is a load-order
change, not an API change -- but callers must treat these as calls, not as
module-level constants.
"""
from .util import DEV

_CACHE = {}
_MODELS = None
_PRE = None
_NAMES = None


def _get(key, factory):
    if key not in _CACHE:
        m = factory()
        m.to(DEV)
        m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _models():
    """``torchvision.models``, imported once, on the first construction."""
    global _MODELS
    if _MODELS is None:
        import torchvision.models as M
        _MODELS = M
    return _MODELS


def detector():
    # ``ssdlite320_mobilenet_v3_large`` lives in ``models.detection`` and is not
    # re-exported at the ``models`` top level, so it gets its own import.
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
    w = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    return _get("ssdlite", lambda: ssdlite320_mobilenet_v3_large(weights=w))


def backbone():
    M = _models()
    w = M.ResNet18_Weights.DEFAULT
    return _get("resnet18", lambda: M.resnet18(weights=w))


def small_cnn():
    M = _models()
    w = M.MobileNet_V3_Small_Weights.DEFAULT
    return _get("mbv3s", lambda: M.mobilenet_v3_small(weights=w))


def tiny_cnn():
    M = _models()
    w = M.ShuffleNet_V2_X0_5_Weights.DEFAULT
    return _get("shuff", lambda: M.shufflenet_v2_x0_5(weights=w))


def cls_transform():
    """The ImageNet classification preset, built on first use."""
    global _PRE
    if _PRE is None:
        from torchvision import transforms as T
        _PRE = T.Compose([
            T.Resize(232),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _PRE


def det_transform():
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights)
    return SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()


def label(i):
    global _NAMES
    if _NAMES is None:
        _NAMES = _models().ResNet18_Weights.DEFAULT.meta["categories"]
    return _NAMES[i] if 0 <= i < len(_NAMES) else str(i)
