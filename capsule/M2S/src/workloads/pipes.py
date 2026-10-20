"""Unified pipeline registry with declared operator I/O.

Before this module the operator list lived in ``runner.PIPELINES``, the data
dependencies lived in ``solver_eval.edge_names()``, the operator bodies lived in
``pipe_b.py``/``pipe_c.py``, and the sizes were literals inside closures -- four
places to edit for one new pipeline, and no place that said *which values cross
an operator boundary*.

Here each operator declares ``reads``/``writes`` against a shared state dict.
Data-dependency edges are then derived, and cutting the graph at a function
boundary yields exactly the set of keys that must be serialised.

``device`` values mirror the CUDA-event measurements in
``results/real_profiling_results.json``.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipe_b import dnn_ops, recsys_ops, traffic_ops  # noqa: E402
from workloads.pipe_c import (driving_ops, image_ops,  # noqa: E402
                              video_ops)
from workloads.pipe_d import (current_device as largemodel_device,  # noqa: E402
                              largemodel_ops)
from workloads.pipe_e import vit_ops, vit_spec  # noqa: E402
from workloads.pipe_f import llm_ops, llm_spec  # noqa: E402
from workloads.pipe_g import dlrm_ops, dlrm_spec  # noqa: E402

GPU = "GPU"
CPU = "CPU"


@dataclass(frozen=True)
class OpIO:
    """One operator: identity plus its state-dict contract.

    ``reads``/``writes`` describe *data edges*: values that must be shipped if
    the producer and the consumer end up in different functions.

    ``local`` describes *function-local mutable state*: warm-instance memory
    that survives across invocations of one function (e.g. a retrieval gallery
    built on the first request and reused afterwards). Such a key is read and
    written, and it looks exactly like a data edge in a monolith -- but it must
    NEVER cross a boundary: shipping it would pay a pointless serialisation cost
    and destroy the very caching benefit that makes the split worthwhile.
    Keeping the two categories apart is what lets the solver decide "cut here or
    not" on the real edges only.
    """

    name: str
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    device: str
    is_worker: bool = False
    local: Tuple[str, ...] = ()


class TracingState(dict):
    """State dict that records which keys an operator actually touches.

    Declared I/O (``OpIO.reads``/``writes``) is what decides which values cross
    a function boundary. If a declaration is incomplete the op works fine in a
    monolith and raises KeyError the moment it is split -- exactly the failure
    observed with ``sensor_fusion_kalman``, which reads ``st["im"]`` for its
    shape but had only ``d`` and ``det`` declared. This class makes that class
    of bug detectable instead of silent.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read_keys: Set[str] = set()
        self.written_keys: Set[str] = set()

    def __getitem__(self, k):
        self.read_keys.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read_keys.add(k)
        return super().get(k, default)

    def __setitem__(self, k, v):
        self.written_keys.add(k)
        super().__setitem__(k, v)


@dataclass
class Pipeline:
    key: str
    title: str
    ops: List[OpIO]
    fns: Dict[str, Callable[[], object]]
    state: dict = field(default_factory=dict)

    def names(self) -> List[str]:
        return [o.name for o in self.ops]

    def by_name(self, name: str) -> OpIO:
        for o in self.ops:
            if o.name == name:
                return o
        raise KeyError(name)

    def index(self, name: str) -> int:
        return self.names().index(name)

    def edges(self) -> List[Tuple[str, str]]:
        """Derive (producer, consumer) edges from reads/writes declarations."""
        last_writer: Dict[str, str] = {}
        out: List[Tuple[str, str]] = []
        for o in self.ops:
            for r in o.reads:
                w = last_writer.get(r)
                if w is not None and (w, o.name) not in out:
                    out.append((w, o.name))
            for k in o.writes:
                last_writer[k] = o.name
        return out

    def devices(self) -> Dict[str, str]:
        return {o.name: o.device for o in self.ops}

    def crossing(self, upper: List[str], lower: List[str]) -> List[str]:
        """State keys written upstream and read downstream: must be shipped.

        Keys that either side keeps as function-local state are excluded -- they
        are not data edges, they are warm-instance memory.
        """
        wrote: Set[str] = set()
        pinned: Set[str] = set()
        for n in upper:
            o = self.by_name(n)
            wrote.update(o.writes)
            pinned.update(o.local)
        need: Set[str] = set()
        for n in lower:
            o = self.by_name(n)
            need.update(o.reads)
            pinned.update(o.local)
        return sorted((wrote & need) - pinned)

    def run_ops(self, names: List[str]) -> Dict[str, object]:
        """Execute a contiguous run of operators against the shared state."""
        produced = {}
        for n in names:
            out = self.fns[n]()
            produced[n] = out
        return produced


DNN = [
    OpIO("preprocess_batch_tensors", (), ("x",), GPU),
    OpIO("resnet_feature_backbone", ("x",), ("f",), GPU),
    OpIO("multimodal_fusion_head", ("f",), ("h",), GPU),
    OpIO("postprocess_logits", ("h",), (), GPU),
    OpIO("dispatch_batch_sink_worker", (), (), CPU, True),
]

RECSYS = [
    OpIO("parse_user_context", (), (), GPU),
    OpIO("recall_candidate_items", (), ("items", "top"), GPU),
    OpIO("extract_dense_sparse_features", ("items", "top"), ("x",), GPU),
    OpIO("deepfm_neural_forward", ("x",), ("s",), GPU),
    OpIO("score_ranking_topk", ("s",), (), GPU),
    OpIO("record_behavior_log_worker", (), (), CPU, True),
]

TRAFFIC = [
    OpIO("traffic_frame_capture", (), ("im",), CPU),
    OpIO("vehicle_detection", ("im",), ("det",), GPU),
    OpIO("plate_roi_crop", ("im", "det"), (), CPU),
    OpIO("license_plate_recognition", ("im", "det"), (), GPU),
    OpIO("traffic_statistics", ("det",), (), CPU),
    OpIO("traffic_analysis_worker", ("det",), (), CPU, True),
]

DRIVING = [
    OpIO("camera_frame_ingest", (), ("im",), CPU),
    OpIO("detect_obstacle_2d", ("im",), ("det",), GPU),
    OpIO("estimate_3d_depth", ("im",), ("d",), CPU),
    OpIO("lidar_pointcloud_cluster", ("d",), ("pc",), CPU),
    OpIO("sensor_fusion_kalman", ("im", "d", "det"), ("tracks",), CPU),
    OpIO("trajectory_prediction", ("tracks",), (), CPU),
    OpIO("sensor_fusion_worker", ("tracks",), (), CPU, True),
]

VIDEO = [
    OpIO("video_decode_frames", (), ("frames",), CPU),
    OpIO("subject_detection", ("frames",), ("dets",), GPU),
    OpIO("optical_flow_tracking", ("frames",), ("flow",), GPU),
    OpIO("subject_embedding", ("frames", "dets"), ("emb",), GPU),
    # `gallery` is warm-instance state: built on the first request and reused.
    OpIO("gallery_similarity_search", ("emb", "gallery"), ("gallery",), GPU,
         local=("gallery",)),
    OpIO("video_analysis_worker", (), (), CPU, True),
]

IMAGE = [
    OpIO("image_decode", (), ("im",), CPU),
    OpIO("super_resolution_enhance", ("im",), ("up",), CPU),
    OpIO("semantic_segmentation", ("up",), ("seg",), GPU),
    OpIO("image_semantic_describe", ("up",), (), GPU),
    OpIO("segment_images_batch_worker", (), (), CPU, True),
]

# The Gillis regime: each stage holds only its own parameters. `t1`/`t2` are
# large activation tensors, so every cut here is a genuine bandwidth decision
# and not a rounding error.
#
# The declared device is computed from the M2S_LARGE_DEVICE knob rather than
# fixed, because it changes which constraint binds: a CPU-resident stage is
# subject to the function's RSS ceiling, a GPU-resident one is not.
def large_spec():
    dev = GPU if largemodel_device() == "gpu" else CPU
    return [
        OpIO("large_input_ingest", (), ("n",), CPU),
        OpIO("large_preprocess", ("n",), ("x",), dev),
        OpIO("large_stem_stage", ("x",), ("t1",), dev),
        OpIO("large_mid_stage", ("t1",), ("t2",), dev),
        OpIO("large_late_stage", ("t2",), ("logits",), dev),
        OpIO("large_topk_postprocess", ("logits",), (), dev),
        OpIO("large_result_sink_worker", (), (), CPU, True),
    ]

FACTORIES = [
    ("dnn_pipeline", "Deep Neural Network (DNN) batch inference", dnn_ops, DNN),
    ("recsys_deepfm", "Deep Recommendation (DeepFM ranking)", recsys_ops,
     RECSYS),
    ("pipeline_traffic", "Smart Traffic (SSDLite vehicle detection)",
     traffic_ops, TRAFFIC),
    ("pipeline_driving", "Autonomous Driving (det + monocular depth)",
     driving_ops, DRIVING),
    ("pipeline_video", "Video Analytics (decode + RAFT flow + retrieval)",
     video_ops, VIDEO),
    ("pipeline_image", "Image Processing (SR + LRASPP segmentation)",
     image_ops, IMAGE),
    ("largemodel_partition", "Large CNN partitioned by stage (Gillis regime)",
     largemodel_ops, large_spec),
    ("vit_partition", "ViT-L/16 partitioned at encoder-block boundaries",
     vit_ops, vit_spec),
    # Two pipelines that break assumptions the seven above share.
    #
    # ``llm_serve`` has a boundary payload that is a function of the *request*
    # (the KV cache scales with prompt length), so "cut here or not" stops
    # being a question about the architecture. ``dlrm_ranking`` shards a table
    # rather than a sequence of layers, so a shard is a hash range and the
    # reduction -- not the model -- decides the payload size. Both are
    # representative of what actually runs on serverless today, and both put a
    # different shape of edge in front of the solver.
    ("llm_serve", "LLM serving split at the prefill/decode boundary",
     llm_ops, llm_spec),
    ("dlrm_ranking", "DLRM ranking with a hash-range sharded embedding table",
     dlrm_ops, dlrm_spec),
]


def keys() -> List[str]:
    return [k for k, _, _, _ in FACTORIES]


def build(key: str, st: Optional[dict] = None) -> Pipeline:
    for k, title, factory, spec in FACTORIES:
        if k == key:
            state = TracingState() if st is None else st
            ops = list(spec() if callable(spec) else spec)
            return Pipeline(key=k, title=title, ops=ops,
                            fns=dict(factory(state)), state=state)
    raise KeyError("unknown pipeline %r (have %s)" % (key, keys()))


def validate(key: str) -> List[str]:
    """Run each operator once and report I/O the declaration missed.

    A missing declaration is invisible in a monolith and fatal once the
    pipeline is split, so this check gates every new pipeline.

    What this cannot see is a dependency that *only* exists once the pipeline is
    cut. Every operator here runs against one shared state, so a key an operator
    reads is present whether or not the layout would have carried it across.
    ``audit_layout`` is the other half of the check and the two are not
    interchangeable.
    """
    p = build(key)
    problems: List[str] = []
    for o in p.ops:
        before_reads = set(p.state.read_keys)
        before_writes = set(p.state.written_keys)
        try:
            p.run_ops([o.name])
        except Exception as e:
            problems.append("%s: raised %s: %s"
                            % (o.name, type(e).__name__, str(e)[:90]))
            continue
        declared = set(o.reads) | set(o.writes) | set(o.local)
        touched = set(p.state.read_keys) - before_reads
        undeclared_read = touched - declared
        if undeclared_read:
            problems.append("%s: reads undeclared keys %s"
                            % (o.name, sorted(undeclared_read)))
        new_writes = set(p.state.written_keys) - before_writes
        undeclared_write = new_writes - declared
        if undeclared_write:
            problems.append("%s: writes undeclared keys %s"
                            % (o.name, sorted(undeclared_write)))
    return problems


def audit_layout(ops, groups) -> List[str]:
    """Can this layout actually run? Checked statically, before any process.

    ``validate`` runs every operator against one shared state, so a key an
    operator reads is always there and a dependency that only *exists* once the
    pipeline is cut cannot show up. Consequences are not subtle: ``llm_decode``
    took the prompt length from ``ids``, which the tokenize function owns. Every
    layout ran in one function; every split died on the first request with a
    missing key, and the failure surfaced as a worker EOF rather than as
    anything that named the dependency.

    A function may read a key if it was written earlier in the same function or
    by a function that runs earlier. ``local`` keys are exempt by construction:
    they are function-local state that must never cross a boundary.

    Also checks the layout places every operator exactly once -- an operator
    belongs to one function, so a layout that drops or duplicates one has
    silently changed the pipeline rather than partitioned it.
    """
    by_name = {o.name: o for o in ops}
    producer = {}
    for i, o in enumerate(ops):
        for k in o.writes:
            producer.setdefault(k, i)

    problems: List[str] = []
    flat = [n for g in groups for n in g]
    missing = [o.name for o in ops if o.name not in flat]
    dup = sorted({n for n in flat if flat.count(n) > 1})
    if missing:
        problems.append("layout drops operator(s) %s" % missing)
    if dup:
        problems.append("layout places operator(s) more than once: %s" % dup)

    available: Set[str] = set()
    for fi, group in enumerate(groups):
        wrote_here: Set[str] = set()
        for name in group:
            o = by_name.get(name)
            if o is None:
                problems.append("function %d: unknown operator %r" % (fi, name))
                continue
            exempt = set(o.local)
            for k in o.reads:
                if k in exempt or k in wrote_here or k in available:
                    continue
                p = producer.get(k)
                if p is None:
                    problems.append(
                        "function %d: %s reads %r, which no operator writes"
                        % (fi, name, k))
                else:
                    problems.append(
                        "function %d: %s reads %r, written by %s in a later "
                        "function" % (fi, name, k, ops[p].name))
            wrote_here |= set(o.writes)
            available |= set(o.writes)
    return problems


def main():
    do_validate = "--validate" in sys.argv
    bad = 0
    for k in keys():
        p = build(k)
        dev = p.devices()
        ngpu = sum(1 for v in dev.values() if v == GPU)
        print("%-18s ops=%d  gpu=%d cpu=%d  edges=%d"
              % (k, len(p.ops), ngpu, len(dev) - ngpu, len(p.edges())))
        if do_validate:
            for prob in validate(k):
                bad += 1
                print("    !! %s" % prob)
        else:
            for a, b in p.edges():
                print("     %-32s -> %s" % (a, b))
        print()
    if do_validate:
        print("VALIDATION %s (%d problem(s))"
              % ("FAILED" if bad else "OK", bad))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
