import cv2
import torch
import torch.nn.functional as F

from .models import backbone
from .pipe_a import batch_tensor, detect, embed, frame, sink
from .util import DEV


def _crops(im, det, k=8, size=(224, 224)):
    from PIL import Image
    out = []
    for b in det["boxes"][:k].numpy().astype(int):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        c = im[max(y1, 0):y2, max(x1, 0):x2]
        if c.size:
            c = cv2.resize(c, size)
            out.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
    return out


def dnn_ops(st=None):
    st = {} if st is None else st
    fusion = torch.nn.Linear(1000, 512).to(DEV).eval()

    def preprocess():
        st["x"] = batch_tensor(n=32)
        return st["x"]

    def feature():
        with torch.no_grad():
            st["f"] = backbone()(st["x"])
        return st["f"]

    def head():
        with torch.no_grad():
            st["h"] = fusion(F.relu(st["f"]))
        return st["h"]

    def postprocess():
        with torch.no_grad():
            s = F.softmax(st["h"], dim=1)
        return s.topk(5, dim=1)

    return [
        ("preprocess_batch_tensors", preprocess),
        ("resnet_feature_backbone", feature),
        ("multimodal_fusion_head", head),
        ("postprocess_logits", postprocess),
        ("dispatch_batch_sink_worker", lambda: sink("batch=32", "dnn")),
    ]


def recsys_ops(st=None):
    emb = torch.nn.Embedding(10000, 128).to(DEV)
    ranker = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(256, 1)).to(DEV).eval()
    st = {} if st is None else st

    def ctx():
        return torch.arange(64, device=DEV)

    def recall():
        with torch.no_grad():
            items = emb(torch.arange(4096, device=DEV))
            q = emb(torch.tensor([7, 19, 23], device=DEV)).mean(0)
            st["items"] = items
            st["top"] = (items @ q).topk(50)
        return st["top"]

    def feats():
        with torch.no_grad():
            u = emb(ctx()[:8]).mean(0)
            i = st["items"][st["top"][1][:1]].mean(0)
            st["x"] = torch.cat([u, i])
        return st["x"]

    def rank():
        with torch.no_grad():
            st["s"] = ranker(st["x"].view(1, -1))
        return st["s"]

    def topk():
        with torch.no_grad():
            return st["s"].topk(1)

    return [
        ("parse_user_context", ctx),
        ("recall_candidate_items", recall),
        ("extract_dense_sparse_features", feats),
        ("deepfm_neural_forward", rank),
        ("score_ranking_topk", topk),
        ("record_behavior_log_worker", lambda: sink("user=7", "recsys")),
    ]


def traffic_ops(st=None):
    st = {} if st is None else st

    def capture():
        st["im"] = frame(0)
        return st["im"]

    def vehicle():
        st["det"] = detect(st["im"])
        return st["det"]

    def stats():
        return {"vehicles": int(len(st["det"]["boxes"]))}

    return [
        ("traffic_frame_capture", capture),
        ("vehicle_detection", vehicle),
        ("plate_roi_crop", lambda: _crops(st["im"], st["det"])),
        ("license_plate_recognition", lambda: embed(_crops(st["im"],
                                                          st["det"]))),
        ("traffic_statistics", stats),
        ("traffic_analysis_worker", lambda: sink(stats(), "traffic")),
    ]
