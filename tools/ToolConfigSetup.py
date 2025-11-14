#!/usr/bin/env python3
"""
k8s_nodes_to_json.py

Query Kubernetes nodes and save summary hardware/resource info to JSON.

Usage:
    pip install kubernetes
    python k8s_nodes_to_json.py --output nodes_info.json

Optional:
    --include-pods    : count scheduled pods per node (may be slower)
"""

import json
import argparse
import re
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

# -------- helpers --------
def parse_quantity(q: str):
    """
    Parse k8s resource quantity strings into a numeric value.
    Returns:
      - For CPU: number of cores (float) (e.g. "500m" -> 0.5, "2" -> 2.0)
      - For memory/storage: bytes (int) (e.g. "128974848Ki" -> bytes)
      - For plain ints: int
    This is a best-effort parser for common suffixes (Ki, Mi, Gi, Ti, Pi, Ei, m).
    """
    if q is None:
        return None
    q = str(q)
    # CPU milli cores
    if q.endswith("m") and re.match(r"^\d+m$", q):
        return float(q[:-1]) / 1000.0
    # binary SI suffixes for memory/storage
    m = re.match(r"^([0-9.]+)(Ei|Pi|Ti|Gi|Mi|Ki)$", q)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        mapping = {
            "Ki": 1024 ** 1,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
        }
        return int(val * mapping[unit])
    # decimal SI (rare in K8s resource output but handle)
    m2 = re.match(r"^([0-9.]+)(k|M|G|T|P|E)$", q)
    if m2:
        val = float(m2.group(1))
        unit = m2.group(2)
        mapping = {
            "k": 10**3,
            "M": 10**6,
            "G": 10**9,
            "T": 10**12,
            "P": 10**15,
            "E": 10**18,
        }
        return int(val * mapping[unit])
    # plain integer
    if re.match(r"^[0-9]+$", q):
        return int(q)
    # fallback: try float
    try:
        return float(q)
    except ValueError:
        return q


def safe_get(d, key, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default


def node_to_dict(v1node, include_pods=False, core_v1_api=None):
    meta = v1node.metadata
    status = v1node.status
    spec = v1node.spec

    info = {
        "name": meta.name,
        "uid": meta.uid,
        "creationTimestamp": meta.creation_timestamp.isoformat() if meta.creation_timestamp else None,
        "labels": meta.labels or {},
        "annotations": meta.annotations or {},
        "addresses": {},
        "taints": [],
        "capacity": {},
        "allocatable": {},
        "nodeInfo": {},
        "conditions": {},
    }

    # addresses
    if status and status.addresses:
        for a in status.addresses:
            info["addresses"][a.type] = a.address

    # taints
    if spec and spec.taints:
        for t in spec.taints:
            info["taints"].append({
                "key": t.key,
                "value": t.value,
                "effect": t.effect
            })

    # capacity and allocatable: convert known values
    if status:
        cap = status.capacity or {}
        alloc = status.allocatable or {}
        for k, v in cap.items():
            parsed = parse_quantity(v)
            info["capacity"][k] = parsed
        for k, v in alloc.items():
            parsed = parse_quantity(v)
            info["allocatable"][k] = parsed

    # nodeInfo
    if status and status.node_info:
        ni = status.node_info
        info["nodeInfo"] = {
            "machineID": ni.machine_id,
            "systemUUID": ni.system_uuid,
            "bootID": ni.boot_id,
            "kernelVersion": ni.kernel_version,
            "osImage": ni.os_image,
            "containerRuntimeVersion": ni.container_runtime_version,
            "kubeletVersion": ni.kubelet_version,
            "kubeProxyVersion": ni.kube_proxy_version,
            "operatingSystem": ni.operating_system,
            "architecture": ni.architecture
        }

    # conditions
    if status and status.conditions:
        for c in status.conditions:
            info["conditions"][c.type] = {
                "status": c.status,
                "reason": c.reason,
                "message": c.message,
                "lastHeartbeatTime": c.last_heartbeat_time.isoformat() if c.last_heartbeat_time else None,
                "lastTransitionTime": c.last_transition_time.isoformat() if c.last_transition_time else None
            }

    # detect GPUs: any capacity/allocatable keys containing 'gpu'
    gpu_entries = {}
    for k, v in (status.capacity or {}).items():
        if "gpu" in k.lower():
            gpu_entries[k] = parse_quantity(v)
    if gpu_entries:
        info["gpus"] = gpu_entries

    # optionally count pods scheduled to node
    if include_pods and core_v1_api is not None:
        try:
            field_selector = f"spec.nodeName={meta.name},status.phase!=Succeeded,status.phase!=Failed"
            pods = core_v1_api.list_pod_for_all_namespaces(field_selector=field_selector)
            info["runningPodsCount"] = len(pods.items)
            # optionally capture pod names (commented out to keep output small)
            # info["pods"] = [p.metadata.name for p in pods.items]
        except ApiException as e:
            info["runningPodsCount"] = None
            info.setdefault("_errors", []).append(f"pod-list-error:{str(e)}")

    return info


def get_k8s_res_info(select_node: Optional[str] = None) -> ToolResponse:
    config.load_kube_config_from_dict()
    core_v1 = client.CoreV1Api()
    try:
        nodes = core_v1.list_node().items
    except ApiException as e:
        print("Error listing nodes:", e)
        return
    out = {"collectedAt": None, "nodeCount": len(nodes), "nodes": []}
    import datetime
    out["collectedAt"] = datetime.datetime.utcnow().isoformat() + "Z"

    for n in nodes:
        nd = node_to_dict(n, include_pods=False, core_v1_api=None)
        out["nodes"].append(nd)
    
    kube_tool_contents = "Kubernetes Nodes Resource Info:"
    if select_node:
        out["nodes"] = [node for node in out["nodes"] if node["name"] == select_node]
        kube_tool_contents = "Kubernetes Node Resource Info for node: " + select_node
    
    

    try:
        return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = f"{kube_tool_contents}\n{json.dumps(out, indent=2)}"
                ),
            ],
        )
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = f"Error generating response: {str(e)}"
                ),
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Kubernetes node hardware/resource info to JSON")
    parser.add_argument("--output", "-o", default="k8s_nodes_info.json", help="Output JSON file")
    parser.add_argument("--kubeconfig", "-k", default=None, help="Path to kubeconfig (optional)")
    parser.add_argument("--include-pods", action="store_true", help="Include count of scheduled pods per node")

    args = parser.parse_args()

    # load kube config (prefer explicit kubeconfig, then default, then in-cluster)
    try:
        if args.kubeconfig:
            config.load_kube_config(config_file=args.kubeconfig)
        else:
            try:
                config.load_kube_config()
            except Exception:
                # fallback to in-cluster
                config.load_incluster_config()
    except Exception as e:
        print("Failed to load kube config (tried kubeconfig/default/in-cluster). Error:", e)
        

    core_v1 = client.CoreV1Api()

    try:
        nodes = core_v1.list_node().items
    except ApiException as e:
        print("Error listing nodes:", e)


    out = {"collectedAt": None, "nodeCount": len(nodes), "nodes": []}
    import datetime
    out["collectedAt"] = datetime.datetime.utcnow().isoformat() + "Z"

    for n in nodes:
        nd = node_to_dict(n, include_pods=args.include_pods, core_v1_api=core_v1 if args.include_pods else None)
        out["nodes"].append(nd)

    # write JSON
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(nodes)} nodes to {args.output}")

