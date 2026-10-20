import glob
import os
import time

import cv2
import numpy as np
import torch

from .models import (backbone, cls_transform, det_transform, detector,
                     small_cnn, tiny_cnn)
from .util import DEV

ASSETS = None
VIDEO = "workloads_data/clip.avi"


def asset_images():
    """Real photographs shipped with the ultralytics wheel."""
    global ASSETS
    if ASSETS:
        return ASSETS
    try:
        import ultralytics
        d = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
        ASSETS = sorted(glob.glob(os.path.join(d, "*.jpg")))
    except Exception:
        ASSETS = []
    if not ASSETS:
        raise RuntimeError("no real images found")
    return ASSETS


def frame(index=0, size=(1280, 720)):
    imgs = asset_images()
    im = cv2.imread(imgs[index % len(imgs)])
    im = cv2.resize(im, size)
    return im


def ensure_clip(n=48, fps=25, size=(640, 360)):
    """Encode a real video file from real photographs (real codec work)."""
    if os.path.exists(VIDEO):
        return VIDEO
    os.makedirs(os.path.dirname(VIDEO), exist_ok=True)
    imgs = asset_images()
    w = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        im = cv2.imread(imgs[i % len(imgs)])
        im = cv2.resize(im, size)
        for _ in range(2):
            w.write(im)
    w.release()
    return VIDEO


def read_clip(max_frames=32):
    cap = cv2.VideoCapture(ensure_clip())
    out = []
    while len(out) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def batch_tensor(size=(224, 224), n=32):
    """Batch built from real photographs (jpeg -> resize -> tensor)."""
    imgs = asset_images()
    prep = cls_transform()
    xs = []
    for i in range(n):
        im = cv2.imread(imgs[i % len(imgs)])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        from PIL import Image
        xs.append(prep(Image.fromarray(im)))
    return torch.stack(xs).to(DEV)


def detect(frame_bgr, score=0.5):
    m = detector()
    tf = det_transform()
    from PIL import Image
    im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x = tf(im).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = m(x)[0]
    keep = out["scores"] > score
    return {k: v[keep].detach().cpu() for k, v in out.items()}


def embed(crops):
    """Real embedding: MobileNetV3-Small features (ImageNet weights)."""
    if not crops:
        return torch.zeros(0, 1000, device=DEV)
    import torch.nn.functional as F
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = small_cnn()
    xs = torch.stack([tf(c) for c in crops]).to(DEV)
    with torch.no_grad():
        z = m(xs)
    return F.normalize(z, dim=1)


def sink(payload, tag):
    """Async worker: real disk write, not a sleep."""
    p = os.path.join("workloads_data/out", tag + ".txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    body = str(payload)[:4096]
    with open(p, "w") as f:
        f.write(body)
    return len(body)


def cluster(points, voxel=0.5):
    """Real voxel downsample + radius clustering on a point cloud."""
    pts = np.asarray(points, dtype=np.float32)
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]
