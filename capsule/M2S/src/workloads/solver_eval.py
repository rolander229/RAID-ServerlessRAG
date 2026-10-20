"""Feed the REAL measured profiles into the existing OptimalPartitionSolver
and check whether the designed algorithm still produces sensible plans.

Run:  PYTHONPATH=M2S/src .venv/bin/python -m workloads.solver_eval
"""
import json
import sys
import os

sys.path.insert(0, "M2S/src")

from profiler.optimizer import OptimalPartitionSolver, CommunicationCostModel

REAL = "results/real_profiling_results.json"
OLD_FUNCS = {"dnn_pipeline": 4, "recsys_deepfm": 6, "pipeline_traffic": 4,
             "pipeline_driving": 7, "pipeline_video": 4, "pipeline_image": 4}

# data-dependency edges (source, target) as actually implemented


def edge_names():
    return {
        "dnn_pipeline": [
            ("preprocess_batch_tensors", "resnet_feature_backbone"),
            ("resnet_feature_backbone", "multimodal_fusion_head"),
            ("multimodal_fusion_head", "postprocess_logits"),
            ("postprocess_logits", "dispatch_batch_sink_worker"),
        ],
        "recsys_deepfm": [
            ("parse_user_context", "recall_candidate_items"),
            ("recall_candidate_items", "extract_dense_sparse_features"),
            ("extract_dense_sparse_features", "deepfm_neural_forward"),
            ("parse_user_context", "extract_dense_sparse_features"),
            ("deepfm_neural_forward", "score_ranking_topk"),
            ("score_ranking_topk", "record_behavior_log_worker"),
        ],
        "pipeline_traffic": [
            ("traffic_frame_capture", "vehicle_detection"),
            ("vehicle_detection", "plate_roi_crop"),
            ("plate_roi_crop", "license_plate_recognition"),
            ("vehicle_detection", "traffic_statistics"),
            ("traffic_statistics", "traffic_analysis_worker"),
        ],
        "pipeline_driving": [
            ("camera_frame_ingest", "detect_obstacle_2d"),
            ("camera_frame_ingest", "estimate_3d_depth"),
            ("detect_obstacle_2d", "sensor_fusion_kalman"),
            ("estimate_3d_depth", "lidar_pointcloud_cluster"),
            ("estimate_3d_depth", "sensor_fusion_kalman"),
            ("lidar_pointcloud_cluster", "sensor_fusion_kalman"),
            ("sensor_fusion_kalman", "trajectory_prediction"),
            ("trajectory_prediction", "sensor_fusion_worker"),
        ],
        "pipeline_video": [
            ("video_decode_frames", "subject_detection"),
            ("video_decode_frames", "optical_flow_tracking"),
            ("video_decode_frames", "subject_embedding"),
            ("subject_detection", "subject_embedding"),
            ("subject_embedding", "gallery_similarity_search"),
            ("gallery_similarity_search", "video_analysis_worker"),
        ],
        "pipeline_image": [
            ("image_decode", "super_resolution_enhance"),
            ("super_resolution_enhance", "semantic_segmentation"),
            ("super_resolution_enhance", "image_semantic_describe"),
            ("semantic_segmentation", "segment_images_batch_worker"),
            ("image_semantic_describe", "segment_images_batch_worker"),
        ],
    }


# compression factor: what a real wire format would carry instead of raw bytes
COMPRESS = {
    "traffic_frame_capture": 0.04, "camera_frame_ingest": 0.04,
    "image_decode": 0.04, "video_decode_frames": 0.04,
    "super_resolution_enhance": 0.04,
    "preprocess_batch_tensors": 0.04,
    "estimate_3d_depth": 0.05,
    "semantic_segmentation": 0.15,
    "license_plate_recognition": 0.5, "subject_embedding": 0.5,
    "plate_roi_crop": 0.04,
}


def build_report(b, compress=False):
    ops = {}
    payload = {}
    for o in b["operators"]:
        if "avg_ms" not in o:
            continue
        dev = "GPU_Inference" if o["device"] == "GPU" else (
            "CPU_Heavy" if o["avg_ms"] > 50 else "CPU_Light")
        ops[o["operator"]] = {
            "avg_latency_ms": o["avg_ms"],
            "peak_cpu_memory_mb": max(64.0, o["payload_mb"] * 4 + 32),
            "peak_gpu_memory_mb": o["gpu_peak_extra_mb"],
            "compute_device": dev,
            "is_async_candidate": False,
        }
        mb = o["payload_mb"]
        if compress:
            mb = mb * COMPRESS.get(o["operator"], 0.05)
        payload[o["operator"]] = mb
    matrix = []
    for s, t in edge_names()[b["benchmark_id"]]:
        if s in payload:
            matrix.append({"source_operator": s, "target_operator": t,
                           "payload_size_mb": round(payload[s], 4)})
    return {"operators": ops, "data_passing_matrix": matrix,
            "pipeline_name": b["benchmark_id"],
            "benchmark_id": b["benchmark_id"]}


def summarize(plan):
    return [(f["logical_name"].replace("-lambda", "").replace(
        "-fused-pipeline", ""), f["operators"]) for f in plan.fused_functions]


def main():
    data = json.loads(open(REAL).read())
    print("%-18s %-28s %-28s" % ("benchmark", "RAW bytes -> plan",
                                 "compressed -> plan"))
    print("-" * 78)
    rows = []
    for b in data["benchmarks"]:
        out = {}
        for mode, comp in (("raw", False), ("zip", True)):
            rep = build_report(b, compress=comp)
            plan = OptimalPartitionSolver().solve(rep)
            out[mode] = plan
        r, z = out["raw"], out["zip"]
        print("%-18s fns=%d  comm-red=%6.2f%%      fns=%d  comm-red=%6.2f%%"
              % (b["benchmark_id"], r.total_fused_faas_functions,
                 r.metrics["communication_volume_reduction_percent"],
                 z.total_fused_faas_functions,
                 z.metrics["communication_volume_reduction_percent"]))
        rows.append((b["benchmark_id"], r, z))

    print("\n=== RAW-byte plan detail (what the current rule actually decides) ===")
    for bid, r, z in rows:
        print("\n%s  (old fabricated plan had %d functions)"
              % (bid, OLD_FUNCS[bid]))
        for name, ops in summarize(r):
            print("   [raw] %-34s = %s" % (name, ops))
        for name, ops in summarize(z):
            print("   [zip] %-34s = %s" % (name, ops))

    print("\n=== is the claimed benefit real? transfer-cost model check ===")
    for bid, r, z in rows:
        mr = r.metrics
        mz = z.metrics
        print("%-18s raw: saved %.1f MB -> claimed %.1f ms | "
              "compressed: saved %.1f MB -> claimed %.1f ms"
              % (bid, mr["eliminated_network_transfer_mb"],
                 mr["baseline_e2e_latency_ms"] - mr["optimized_e2e_latency_ms"],
                 mz["eliminated_network_transfer_mb"],
                 mz["baseline_e2e_latency_ms"] - mz["optimized_e2e_latency_ms"]))


if __name__ == "__main__":
    main()
