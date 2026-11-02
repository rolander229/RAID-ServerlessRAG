"""Shared mapping constants & scalar rules for the M2' K8s pipeline.

Single source of truth for the numbers and platform rules that all three
consumers must agree on (metrics-review-k8s.md "验收器 = 评分器" principle):

  * capsule/agent_tools/k8s_render_tool.py  (wraps capsule.k8s_render.render)
  * capsule/agent_tools/kube_validate_tool.py  (D4/D5 gate checks)
  * capsule/m2s_k8s/expert_baseline.py  (independent expert derivation)

This module holds ONLY constants and scalar math functions.  It deliberately
does NOT build K8s documents -- each consumer keeps its own document-building
code path so the validator cannot rubber-stamp a bug it shares with the
producer (the "自己抄自己" trap).

Anchors (confirmed 2026-11-02):
  * CPU functions  -> AWS Lambda anchor  lambda = 1769 MB/vCPU
  * limits.memory  -> ceil((alloc_mb + torch_base_mi) * headroom) Mi
                      (metrics-review-k8s.md section 3.4 three-number split:
                       the declared alloc stays the billing number, the limit
                       is deliberately generous for fault tolerance)
  * requests.cpu   -> ceil(alloc_mb / lambda) m
  * requests.memory-> ceil(footprint_mb) Mi  (declared demand)
"""

from __future__ import annotations

import math
import re
from typing import Dict, Tuple

# --- quota anchors (metrics-review-k8s.md section 4 + K0 renderer defaults) ---
LAMBDA_MB_PER_VCPU = 1769          # AWS Lambda official MB-per-vCPU anchor
ANCHOR_NAME = "aws-lambda-1769"
TORCH_BASE_MI = 600                # torch runtime memory floor
HEADROOM = 1.3                     # limit headroom multiplier
LIMIT_TOLERANCE_MI = 16            # D4 tolerance on limits.memory vs expected
REQUEST_MEM_TOLERANCE_MI = 16      # D4 tolerance on requests.memory vs ceil(footprint)

# --- platform constants (slice-manifest-to-k8s-mapping-draft.md + K0) ---
SERVICE_PORT = 8080
HEALTHZ_PATH = "/healthz"
DEFAULT_INGRESS_PATH = "/predict"
SHM_SIZE_LIMIT = "256Mi"           # /dev/shm emptyDir (Memory medium) sizeLimit
INLINE_MAX_MB = 2                  # tensor transport dual-track inline threshold
TRANSPORT_MODES = ("inline", "pvc", "dual")
GPU_RESOURCE_DEFAULT = "nvidia.com/v100"
GPU_RESOURCE_RE = re.compile(r"^nvidia\.com/[a-z0-9][a-z0-9-]*$")
PLAN_SOURCE_DP = "dp_solve"
PLAN_SOURCE_HARDCODED = "hardcoded"

# --- label conventions (mopar.io/* reserved prefix) ---
LABEL_MODEL = "mopar.io/model"
LABEL_SLICE_ID = "mopar.io/slice-id"
LABEL_ROLE = "mopar.io/role"
LABEL_ALLOC_MB = "mopar.io/alloc-mb"
RESERVED_LABEL_PREFIXES = ("mopar.io/", "app.kubernetes.io/")
ENV_SLICE_ID = "MOPAR_SLICE_ID"
ENV_OUT_PEERS = "MOPAR_OUT_PEERS"
ENV_TRANSPORT = "MOPAR_TRANSPORT"
MOPAR_ENV_PREFIX = "MOPAR_"

# --- required volume mount points (D4 "PVC 挂载点齐备") ---
REQUIRED_MOUNTS = {
    "/dev/shm": {"name": "shm", "readOnly": False},
    "/models": {"name": "weights", "readOnly": True},
    "/activations": {"name": "activations", "readOnly": False},
}

# --- K8s name grammar (DNS-1123 subdomain / label value) ---
DNS1123_SUBDOMAIN_RE = re.compile(
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
)
LABEL_VALUE_RE = re.compile(r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$")

FILE_INGRESS = "40-ingress.yaml"
FILE_DIFFTEST = "50-difftest-job.yaml"


# --------------------------------------------------------------------------
# scalar math (shared by renderer wrapper, validator and expert baseline)
# --------------------------------------------------------------------------
def expected_memory_limit_mi(alloc_mb: float,
                             torch_base_mi: int = TORCH_BASE_MI,
                             headroom: float = HEADROOM) -> int:
    """D4 expected limits.memory in Mi = ceil((alloc + torch floor) * headroom)."""
    return math.ceil((float(alloc_mb) + torch_base_mi) * headroom)


def expected_cpu_requests_m(alloc_mb: float,
                            lambda_mb_per_vcpu: int = LAMBDA_MB_PER_VCPU) -> int:
    """D4 expected requests.cpu in millicores = ceil(alloc / lambda)."""
    return math.ceil(float(alloc_mb) / lambda_mb_per_vcpu)


def expected_memory_requests_mi(footprint_mb: float) -> int:
    """D4 expected requests.memory in Mi = ceil(declared footprint)."""
    return math.ceil(float(footprint_mb))


def k8s_name(name: str) -> str:
    """manifest slice names use '_'; K8s object names are DNS-1123 ('-' only)."""
    return str(name).replace("_", "-").lower()


def valid_dns1123_subdomain(name: str) -> bool:
    if not isinstance(name, str) or not name or len(name) > 253:
        return False
    return bool(DNS1123_SUBDOMAIN_RE.match(name))


def valid_label_value(value: str) -> bool:
    if not isinstance(value, str) or len(value) > 63:
        return False
    return bool(LABEL_VALUE_RE.match(value))


# --------------------------------------------------------------------------
# K8s quantity parsers (validator needs to compare rendered vs expected)
# --------------------------------------------------------------------------
# K8s quantity suffixes, exact spellings per k8s.io/apimachinery ParseQuantity
# (binary: Ki/Mi/Gi/Ti/Pi/Ei; decimal SI: k/M/G/T/P/E; bare = plain bytes).
# Table value = how many Mi one unit represents.
_MI_PER = {
    "Ki": 1 / 1024,
    "Mi": 1.0,
    "Gi": 1024.0,
    "Ti": 1024.0 ** 2,
    "Pi": 1024.0 ** 3,
    "Ei": 1024.0 ** 4,
    "k": 1e3 / (1024.0 ** 2),
    "M": 1e6 / (1024.0 ** 2),
    "G": 1e9 / (1024.0 ** 2),
    "T": 1e12 / (1024.0 ** 2),
    "P": 1e15 / (1024.0 ** 2),
    "E": 1e18 / (1024.0 ** 2),
    "": 1 / (1024.0 ** 2),
}


def parse_memory_mi(quantity: str) -> int:
    """Parse a K8s memory quantity to an integer Mi count (raises ValueError)."""
    if isinstance(quantity, (int, float)) and not isinstance(quantity, bool):
        return int(math.ceil(float(quantity) / (1024.0 ** 2)))  # raw bytes
    text = str(quantity).strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)$", text)
    if not m:
        raise ValueError(f"not a K8s memory quantity: {quantity!r}")
    value, suffix = m.groups()
    if suffix not in _MI_PER:
        raise ValueError(f"unsupported memory suffix {suffix!r} in {quantity!r}")
    return math.ceil(float(value) * _MI_PER[suffix])


def parse_cpu_m(quantity: str) -> int:
    """Parse a K8s CPU quantity to integer millicores ('1' -> 1000, '500m' -> 500)."""
    text = str(quantity).strip()
    if text.endswith("m"):
        return int(float(text[:-1]))
    return int(float(text) * 1000)


def peer_service_dns(slice_name: str, namespace: str,
                     port: int = SERVICE_PORT) -> str:
    """Expected Service DNS for an outbound peer env value."""
    return f"{k8s_name(slice_name)}.{namespace}.svc:{port}"


def expected_out_peers_env(slice: Dict, name_by_id: Dict, namespace: str,
                           port: int = SERVICE_PORT) -> str:
    """Expected MOPAR_OUT_PEERS value for one manifest slice entry."""
    peers = slice.get("outbound_peers") or []
    return ",".join(
        f"{k8s_name(name_by_id[p])}.{namespace}.svc:{port}" for p in peers
    )


def slice_role(slice: Dict) -> str:
    """orchestrator/worker role exactly as the renderer derives it."""
    return slice.get("type") or ("orchestrator" if slice.get("slice_id") == 0 else "worker")


def resource_audit(slice: Dict) -> Tuple[int, int, int]:
    """(limits_mi, cpu_m, requests_mi) the D4 gate expects for a manifest slice."""
    alloc = float(slice["alloc_mb"])
    footprint = float(slice["footprint_mb"])
    return (
        expected_memory_limit_mi(alloc),
        expected_cpu_requests_m(alloc),
        expected_memory_requests_mi(footprint),
    )
