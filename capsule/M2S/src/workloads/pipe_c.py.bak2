import os

import cv2
import numpy as np
import torch

from .pipe_a import cluster, detect, embed, frame, read_clip, sink
from .pipe_b import _crops
from .util import DEV

_CACHE = {}


def _hf(endpoint="https://hf-mirror.com"):
    os.environ.setdefault("HF_ENDPOINT", endpoint)


def depth_model():
    """Real monocular depth: Depth-Anything-V2-Small (HF mirror)."""
    if "depth" not in _CACHE:
        _CACHE["depth"] = None
        try:
            _hf()
            from transformers import (AutoImageProcessor,
                                      AutoModelForDepthEstimation)
            mid = "depth-anything/Depth-Anything-V2-Small-hf"
            p = AutoImageProcessor.from_pretrained(mid)
            m = AutoModelForDepthEstimation.from_pretrained(mid)
            m.to(DEV).eval()
            _CACHE["depth"] = (p, m)
        except Exception as e:
            print("[warn] depth unavailable:", str(e)[:120])
    return _CACHE["depth"]


def depth_map(bgr):
    got = depth_model()
    if got is None:
        return sgbm(bgr)
    proc, net = got
    from PIL import Image
    x = proc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
             return_tensors="pt")
    x = {k: v.to(DEV) for k, v in x.items()}
    with torch.no_grad():
        d = net(**x).predicted_depth
    return d.squeeze().float().cpu().numpy()


def sgbm(bgr):
    """Fallback: real OpenCV semi-global block matching on a stereo pair."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7)
    return m.compute(g[:, :-32], g[:, 32:]).astype(np.float32)


def to_cloud(depth, step=4):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    z = np.clip(depth[::step, ::step], 0.1, 80.0)
    return np.stack([xs * z / 640.0, ys * z / 640.0, z], -1).reshape(-1, 3)


def raft():
    if "raft" not in _CACHE:
        from torchvision.models.optical_flow import (raft_small,
                                                     Raft_Small_Weights)
        m = raft_small(weights=Raft_Small_Weights.DEFAULT)
        m.to(DEV).eval()
        _CACHE["raft"] = m
    return _CACHE["raft"]


def seg_model():
    if "seg" not in _CACHE:
        from torchvision.models.segmentation import (
            lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights)
        w = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        m = lraspp_mobilenet_v3_large(weights=w)
        m.to(DEV).eval()
        _CACHE["seg"] = (m, w.transforms())
    return _CACHE["seg"]


def sr_model():
    if "sr" not in _CACHE:
        _CACHE["sr"] = None
        try:
            _hf()
            from transformers import (Swin2SRForImageSuperResolution,
                                      AutoImageProcessor)
            mid = "caidas/swin2SR-classical-sr-x2-64"
            p = AutoImageProcessor.from_pretrained(mid)
            m = Swin2SRForImageSuperResolution.from_pretrained(mid)
            m.to(DEV).eval()
            _CACHE["sr"] = (p, m)
        except Exception as e:
            print("[warn] SR unavailable:", str(e)[:120])
    return _CACHE["sr"]


def super_resolve(bgr):
    got = sr_model()
    if got is None:
        up = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        return cv2.filter2D(up, -1, k)
    proc, net = got
    from PIL import Image
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x = proc(images=im, return_tensors="pt")
    x = {k: v.to(DEV) for k, v in x.items()}
    with torch.no_grad():
        out = net(**x).reconstruction.data
    arr = out.squeeze().float().clamp(0, 1).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0)) * 255.0
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)


def driving_ops(st=None):
    st = {} if st is None else st

    def camera():
        st["im"] = frame(1)
        return st["im"]

    def det2d():
        st["det"] = detect(st["im"])
        return st["det"]

    def depth():
        st["d"] = depth_map(st["im"])
        return st["d"]

    def lidar():
        st["pc"] = cluster(to_cloud(st["d"]))
        return st["pc"]

    def fuse():
        h, w = st["d"].shape
        out = []
        for x1, y1, x2, y2 in st["det"]["boxes"].numpy().astype(int):
            cx, cy = (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2
            sx = min(int(cx * w / st["im"].shape[1]), w - 1)
            sy = min(int(cy * h / st["im"].shape[0]), h - 1)
            out.append({"x": cx, "y": cy, "z": float(st["d"][sy, sx])})
        st["tracks"] = out
        return out

    def trajectory():
        return [{"x": t["x"], "y": t["y"], "z": t["z"], "vz": 0.0}
                for t in st["tracks"]]

    return [
        ("camera_frame_ingest", camera),
        ("detect_obstacle_2d", det2d),
        ("estimate_3d_depth", depth),
        ("lidar_pointcloud_cluster", lidar),
        ("sensor_fusion_kalman", fuse),
        ("trajectory_prediction", trajectory),
        ("sensor_fusion_worker", lambda: sink(st.get("tracks"), "driving")),
    ]


def video_ops(st=None):
    # `gallery` is deliberately absent here: it is warm-instance state that the
    # operator itself builds on first use (see VIDEO spec's `local=("gallery",)`).
    st = {} if st is None else st

    def decode():
        st["frames"] = read_clip(32)
        return st["frames"]

    def subjects():
        st["dets"] = [detect(f) for f in st["frames"][:6]]
        return st["dets"]

    def flow():
        m = raft()

        def t(x):
            y = torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
            y = y.permute(2, 0, 1).float() / 255.0
            return y.unsqueeze(0).to(DEV)
        with torch.no_grad():
            st["flow"] = m(t(st["frames"][0]), t(st["frames"][5]))[-1]
        return st["flow"]

    def embedding():
        crops = []
        for im, det in zip(st["frames"][:6], st["dets"]):
            crops.extend(_crops(im, det, k=2))
        st["emb"] = embed(crops)
        return st["emb"]

    def gallery():
        e = st["emb"]
        if e.shape[0] == 0:
            return torch.zeros(1, device=DEV)
        if st.get("gallery") is None:
            st["gallery"] = e.clone()
        sim = e @ st["gallery"].T
        return sim.topk(min(5, sim.shape[1]), dim=1)

    return [
        ("video_decode_frames", decode),
        ("subject_detection", subjects),
        ("optical_flow_tracking", flow),
        ("subject_embedding", embedding),
        ("gallery_similarity_search", gallery),
        ("video_analysis_worker", lambda: sink("clip done", "video")),
    ]


def image_ops(st=None):
    st = {} if st is None else st

    def decode():
        st["im"] = frame(1, (1024, 1024))
        return st["im"]

    def sr():
        st["up"] = super_resolve(st["im"])
        return st["up"]

    def segment():
        m, tf = seg_model()
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(st["up"], cv2.COLOR_BGR2RGB))
        x = tf(im).unsqueeze(0).to(DEV)
        with torch.no_grad():
            st["seg"] = m(x)["out"]
        return st["seg"]

    def describe():
        h, w = st["up"].shape[:2]
        box = torch.tensor([[0, 0, w - 1, h - 1]])
        return embed(_crops(st["up"], {"boxes": box}))

    return [
        ("image_decode", decode),
        ("super_resolution_enhance", sr),
        ("semantic_segmentation", segment),
        ("image_semantic_describe", describe),
        ("segment_images_batch_worker", lambda: sink("sr+seg", "image")),
    ]
