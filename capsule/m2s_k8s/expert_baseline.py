"""K8s expert baseline: independently derive the expected artifacts.

Given the slice manifest (and optionally the Architect blueprint), this module
derives what a K8s-expert human would produce -- per-slice resource numbers,
env contract, service topology, artifact inventory -- using ONLY:

  * the slice manifest,
  * capsule.agent_tools.mapping_rules (constants + scalar math),
  * its OWN document-building logic.

Deliberate independence (the "自己抄自己" guard): expert_baseline never imports
capsule.k8s_render nor capsule.agent_tools.k8s_render_tool.  The renderer and
this module are two code paths over the same published mapping rules, so
comparing their outputs is a real cross-check (plan principle 3: one math,
two uses) rather than a file diff of a module against itself.

Roles:
  * runner records `compare_expected_vs_render()` output in report.json;
  * the resnet50 derivation is locked as tests/golden/
    resnet50_k3_expert_baseline.json after human review (golden);
  * M3' scoring compares pipeline products against this baseline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from capsule.agent_tools.mapping_rules import (
    ANCHOR_NAME, DEFAULT_INGRESS_PATH, ENV_OUT_PEERS, ENV_SLICE_ID,
    ENV_TRANSPORT, HEALTHZ_PATH, LABEL_ALLOC_MB, LABEL_MODEL, LABEL_ROLE,
    LABEL_SLICE_ID, REQUIRED_MOUNTS, SERVICE_PORT, SHM_SIZE_LIMIT,
    LAMBDA_MB_PER_VCPU, TORCH_BASE_MI, HEADROOM, TRANSPORT_MODES, INLINE_MAX_MB,
    expected_cpu_requests_m, expected_memory_limit_mi, expected_memory_requests_mi,
    expected_out_peers_env, k8s_name, slice_role,
)

DEFAULT_NAMESPACE = "mopar-dev"
DEFAULT_IMAGE = "harbor.example/mopar/slice-runner:0.1"
DEFAULT_DIFF_IMAGE = "harbor.example/mopar/difftest:0.1"
DEFAULT_DIFF_SAMPLES = 100
DEFAULT_DIFF_TOLERANCE = "1e-4"
DEFAULT_ACTIVATIONS_PVC = "mopar-activations"
DEFAULT_WEIGHTS_PVC = "mopar-weights"
DEFAULT_ACTIVATIONS_SIZE = "20Gi"
DEFAULT_WEIGHTS_SIZE = "20Gi"

FILE_PVC_ACTIVATIONS = "10-pvc-activations.yaml"
FILE_PVC_WEIGHTS = "11-pvc-weights.yaml"
FILE_PLAN_META = "20-plan-meta-configmap.yaml"
FILE_INGRESS = "40-ingress.yaml"
FILE_DIFFTEST = "50-difftest-job.yaml"


def default_blueprint(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical blueprint a K8s expert would fix for a CPU slice chain."""
    model_key = str(manifest.get("model_key", "model"))
    return {
        "model_key": model_key,
        "namespace": DEFAULT_NAMESPACE,
        "image": DEFAULT_IMAGE,
        "topology": {"mode": "linear-chain",
                     "entry_slice": _entry_slice(manifest)},
        "transport": {"mode": "dual", "inline_max_mb": INLINE_MAX_MB,
                      "activations_pvc": DEFAULT_ACTIVATIONS_PVC,
                      "activations_size": DEFAULT_ACTIVATIONS_SIZE},
        "storage": {"weights_pvc": DEFAULT_WEIGHTS_PVC,
                    "weights_size": DEFAULT_WEIGHTS_SIZE},
        "quota": {"lambda_mb_per_vcpu": LAMBDA_MB_PER_VCPU,
                  "torch_base_mi": TORCH_BASE_MI, "headroom": HEADROOM},
        "gpu": {"required": False, "resource_name": "nvidia.com/v100"},
        "difftest": {"samples": DEFAULT_DIFF_SAMPLES,
                     "tolerance": DEFAULT_DIFF_TOLERANCE,
                     "image": DEFAULT_DIFF_IMAGE},
        "notes": "canonical expert baseline blueprint (CPU linear chain)",
    }


def _entry_slice(manifest: Dict[str, Any]) -> int:
    for s in manifest.get("slices", []):
        if not (s.get("inbound_peers") or []):
            return s.get("slice_id", 0)
    return 0


def _slice_names(manifest: Dict[str, Any]) -> Dict[Any, str]:
    model_key = str(manifest.get("model_key", "model"))
    return {s.get("slice_id"): k8s_name(s.get("name") or f"{model_key}_slice_{s.get('slice_id')}")
            for s in manifest.get("slices", [])}


def _fmt_mb(value: float) -> str:
    v = float(value)
    return str(int(v)) if v.is_integer() else str(v)


# --------------------------------------------------------------------------
# derivation (own code path -- shares constants, not document builders)
# --------------------------------------------------------------------------
def derive_expected(manifest: Dict[str, Any],
                    blueprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Derive the expected K8s artifacts for a manifest, expert-style."""
    bp = blueprint or default_blueprint(manifest)
    namespace = bp["namespace"]
    model_key = str(manifest.get("model_key", "model"))
    model = k8s_name(model_key)
    lam = bp["quota"]["lambda_mb_per_vcpu"]
    torch_base = bp["quota"]["torch_base_mi"]
    headroom = bp["quota"]["headroom"]
    names = _slice_names(manifest)

    per_slice: List[Dict[str, Any]] = []
    for s in manifest.get("slices", []):
        sid = s.get("slice_id")
        dep_name = names[sid]
        role = slice_role(s)
        alloc = float(s["alloc_mb"])
        footprint = float(s["footprint_mb"])
        limit_mi = expected_memory_limit_mi(alloc, torch_base, headroom)
        cpu_m = expected_cpu_requests_m(alloc, lam)
        req_mi = expected_memory_requests_mi(footprint)
        per_slice.append({
            "slice_id": sid,
            "deployment_name": dep_name,
            "service_name": dep_name,
            "role": role,
            "labels": {
                LABEL_MODEL: model,
                LABEL_SLICE_ID: str(sid),
                LABEL_ROLE: role,
                LABEL_ALLOC_MB: _fmt_mb(alloc),
            },
            "namespace": namespace,
            "image": bp["image"],
            "replicas": 1,
            "service_port": SERVICE_PORT,
            "limits_memory": f"{limit_mi}Mi",
            "limits_memory_mi": limit_mi,
            "requests_cpu": f"{cpu_m}m",
            "requests_cpu_m": cpu_m,
            "requests_memory": f"{req_mi}Mi",
            "requests_memory_mi": req_mi,
            "env": {
                ENV_SLICE_ID: str(sid),
                ENV_OUT_PEERS: expected_out_peers_env(s, names, namespace),
                ENV_TRANSPORT: "dual",
            },
            "mounts": sorted(REQUIRED_MOUNTS),
            "probes": {"path": HEALTHZ_PATH, "port": SERVICE_PORT},
            "shm_size_limit": SHM_SIZE_LIMIT,
            "layers_count": len(s.get("layers") or []),
            "outbound_peers": list(s.get("outbound_peers") or []),
            "inbound_peers": list(s.get("inbound_peers") or []),
        })

    entry = manifest.get("slices", [{}])[0]
    ingress = None
    if entry.get("trigger_type") == "api_gateway":
        ingress = {
            "file": FILE_INGRESS,
            "name": f"{model}-ingress",
            "path": entry.get("path") or DEFAULT_INGRESS_PATH,
            "path_type": "Prefix",
            "backend_service": names[entry.get("slice_id", 0)],
            "backend_port": SERVICE_PORT,
        }

    inventory = {
        "deployments": len(per_slice),
        "services": len(per_slice),
        "ingress": ingress,
        "ingress_count_expected": (1 if ingress else 0),
        "pvcs": [
            {"file": FILE_PVC_ACTIVATIONS, "name": bp["transport"]["activations_pvc"],
             "access_modes": ["ReadWriteMany"],
             "storage": bp["transport"]["activations_size"], "role": "activations"},
            {"file": FILE_PVC_WEIGHTS, "name": bp["storage"]["weights_pvc"],
             "access_modes": ["ReadOnlyMany"],
             "storage": bp["storage"]["weights_size"], "role": "weights"},
        ],
        "configmaps": [
            {"file": FILE_PLAN_META, "name": "mopar-plan-meta",
             "keys": ["model_key", "slices_count", "lambda_mb_per_vcpu",
                      "anchor_name", "torch_base_mi", "headroom",
                      "manifest_sha256", "generated_at", "plan_source", "gpu"]},
        ],
        "jobs": [
            {"file": FILE_DIFFTEST, "name": f"mopar-{model}-difftest",
             "endpoint": f"{names[entry.get('slice_id', 0)]}.{namespace}.svc:{SERVICE_PORT}",
             "samples": bp["difftest"]["samples"],
             "tolerance": bp["difftest"]["tolerance"], "restart_policy": "Never"},
        ],
    }

    return {
        "model_key": model_key,
        "namespace": namespace,
        "image": bp["image"],
        "per_slice": per_slice,
        "inventory": inventory,
        "rules_fingerprint": {
            "anchor_name": ANCHOR_NAME,
            "lambda_mb_per_vcpu": lam,
            "torch_base_mi": torch_base,
            "headroom": headroom,
            "transport_modes": list(TRANSPORT_MODES),
            "inline_max_mb": INLINE_MAX_MB,
        },
    }


# --------------------------------------------------------------------------
# comparison against what actually got rendered
# --------------------------------------------------------------------------
def compare_expected_vs_render(expected: Dict[str, Any],
                               render_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mismatch list between the expert derivation and the render summary
    (per-slice numbers extracted from rendered YAML).  Empty list = agreement."""
    mismatches: List[Dict[str, Any]] = []
    exp_by_id = {p["slice_id"]: p for p in expected["per_slice"]}
    got_by_id = {p["slice_id"]: p for p in render_summary.get("per_slice", [])}
    for sid, exp in exp_by_id.items():
        got = got_by_id.get(sid)
        if got is None:
            mismatches.append({"slice_id": sid, "field": "slice",
                               "expected": "present", "rendered": "missing"})
            continue
        for field in ("limits_memory", "requests_cpu", "requests_memory",
                      "out_peers_env" if "out_peers_env" in got else "env"):
            exp_field = {"out_peers_env": "env"}.get(field, field)
            exp_val = exp.get(exp_field) if exp_field != "env" else exp["env"][ENV_OUT_PEERS]
            got_val = got.get(field)
            if exp_val != got_val:
                mismatches.append({"slice_id": sid, "field": field,
                                   "expected": exp_val, "rendered": got_val})
    for sid in sorted(set(got_by_id) - set(exp_by_id)):
        mismatches.append({"slice_id": sid, "field": "slice",
                           "expected": "absent", "rendered": "present"})
    return mismatches
