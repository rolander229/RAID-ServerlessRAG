"""K0 renderer: slice_manifest.json -> deterministic Kubernetes YAML.

Batch K0 of the M2S->K8s adaptation plan (m2s-k8s-adaptation-plan.md): a pure,
deterministic translator from the MOPAR slice manifest (capsule/bridge.py
output) to native K8s resources.  Field mapping follows
slice-manifest-to-k8s-mapping-draft.md; the quota math follows
metrics-review-k8s.md section 3.4 (three separated numbers, generous limit):

    limits.memory = ceil((alloc_mb + torch_base_mi) * headroom) Mi
    requests.memory = ceil(footprint_mb) Mi          (declared demand)
    requests.cpu  = ceil(alloc_mb / lambda_mb_per_vcpu) m   (AWS anchor 1769)

Resource numbers come from this module only -- the K8s Engineer agent may
overlay names/annotations but never invent numbers (plan principle 1).

Determinism contract: same (manifest, params) -> byte-identical output.
yaml.safe_dump_all is pinned (sort_keys=False, default_flow_style=False,
width=120).  The generation timestamp is part of RenderParams, so the wall
clock never enters render(); the CLI fills it.  manifest_sha256 is the sha256
of the canonical JSON dump (sort_keys, compact separators) of the manifest.

Linear-chain gate (capsule/slice_server.py contract): every slice must have
<= 1 inbound and <= 1 outbound edge, non-empty layers and alloc_mb > 0.
Violations raise ValueError -- the renderer refuses, it does not guess.

Plan-source marker (plan-meta ConfigMap): bridge.py emits an explicit
"plan_source" field ("dp_solve" for HyPAD output, "hardcoded" for the
hand-written qwen3 plan).  For older manifests without the field, the
fallback heuristic is: empty stats dict + "qwen" in model_key -> hardcoded.

GPU reservation: v0 renders CPU slices only.  If the model looks like a GPU
LLM (or the manifest sets "requires_gpu": true) the renderer warns on stderr
and stamps gpu: "reserved" into the plan-meta ConfigMap instead of rendering
nvidia.com/* resources.

CLI:
    python -m capsule.k8s_render --manifest tests/fixtures/resnet50_manifest.json \
        --out runs/k0_render_resnet50 --namespace mopar-dev \
        --image harbor.example/mopar/slice-runner:0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

PLAN_SOURCE_DP = "dp_solve"
PLAN_SOURCE_HARDCODED = "hardcoded"
ANCHOR_NAME = "aws-lambda-1769"
GPU_MODEL_HINTS = ("qwen", "chatglm", "llama", "gpt")

SERVICE_PORT = 8080
HEALTHZ_PATH = "/healthz"
DEFAULT_INGRESS_PATH = "/predict"
SHM_SIZE_LIMIT = "256Mi"

INGRESS_FILE = "40-ingress.yaml"


@dataclass(frozen=True)
class RenderParams:
    """Everything the renderer needs besides the manifest.

    ``image`` and ``generated_at`` are required on purpose: an imageless or
    timestampless render cannot be applied or audited, and the timestamp must
    be pinned for byte-level determinism (tests pass a fixed one).
    """

    image: str
    generated_at: str
    namespace: str = "mopar-dev"
    lambda_mb_per_vcpu: int = 1769
    torch_base_mi: int = 600
    headroom: float = 1.3
    activations_pvc: str = "mopar-activations"
    weights_pvc: str = "mopar-weights"
    activations_size: str = "20Gi"
    weights_size: str = "20Gi"
    diff_image: str = "harbor.example/mopar/difftest:0.1"
    diff_samples: int = 100
    diff_tolerance: str = "1e-4"
    gpu_resource: str = "nvidia.com/v100"
    gpu_per_slice: int = 0  # reserved: v0 renders CPU slices only


def _warn(msg: str) -> None:
    print(f"[k8s_render] WARNING: {msg}", file=sys.stderr)

class _ForceQuotedStr(str):
    """String scalar that must serialize quoted.

    PyYAML (YAML 1.1) writes 1e-4 unquoted and re-reads it as a string, but
    Go yaml parsers (yaml.v2, i.e. kubectl/client-go) resolve 1e-4 as a
    NUMBER, and env.value requires a string -- kubectl apply would reject
    the manifest.  Anything numeric-looking goes through this class.
    """


def _quoted_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="'")


yaml.SafeDumper.add_representer(_ForceQuotedStr, _quoted_representer)


def k8s_name(name: str) -> str:
    """manifest names use ``_``; K8s object names are DNS-1123 (``-`` only)."""
    return str(name).replace("_", "-").lower()


def _fmt_mb(value: float) -> str:
    """128.0 -> "128", 128.5 -> "128.5" (K8s label/annotation audit values)."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _require_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    return float(value)


def _plan_source(manifest: Dict[str, Any]) -> str:
    explicit = manifest.get("plan_source")
    if explicit in (PLAN_SOURCE_DP, PLAN_SOURCE_HARDCODED):
        return str(explicit)
    stats = manifest.get("stats")
    model = str(manifest.get("model_key", ""))
    if not stats and "qwen" in model:
        return PLAN_SOURCE_HARDCODED
    return PLAN_SOURCE_DP


def _needs_gpu(manifest: Dict[str, Any]) -> bool:
    if "requires_gpu" in manifest:
        return bool(manifest["requires_gpu"])
    model = str(manifest.get("model_key", "")).lower()
    return any(hint in model for hint in GPU_MODEL_HINTS)


def _validate_linear_chain(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    slices = manifest.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("manifest.slices must be a non-empty list")
    known_ids = set()
    for s in slices:
        if not isinstance(s, dict):
            raise ValueError(f"manifest.slices entries must be dicts, got {type(s).__name__}")
        sid = s.get("slice_id")
        known_ids.add(sid)
        if not isinstance(s.get("layers"), list) or not s["layers"]:
            raise ValueError(f"slice {sid!r}: layers is empty -- nothing to deploy")
        alloc = _require_number(s.get("alloc_mb"), f"slice {sid!r}.alloc_mb")
        if alloc <= 0:
            raise ValueError(f"slice {sid!r}: alloc_mb must be > 0, got {alloc}")
        footprint = _require_number(s.get("footprint_mb"), f"slice {sid!r}.footprint_mb")
        if footprint <= 0:
            raise ValueError(f"slice {sid!r}: footprint_mb must be > 0, got {footprint}")
    for s in slices:
        sid = s.get("slice_id")
        for key in ("inbound_peers", "outbound_peers"):
            peers = s.get(key) or []
            if len(peers) > 1:
                raise ValueError(
                    f"slice {sid!r}: {key} has {len(peers)} edges -- linear-chain "
                    "contract (<=1 in / <=1 out per slice, capsule/slice_server.py) violated"
                )
            for peer in peers:
                if peer not in known_ids:
                    raise ValueError(f"slice {sid!r}: {key} references unknown slice_id {peer!r}")
    return slices


def _labels(model: str, sid: Any = None, role: str = None, alloc_mb: float = None) -> Dict[str, str]:
    labels = {"mopar.io/model": model}
    if sid is not None:
        labels["mopar.io/slice-id"] = str(sid)
    if role is not None:
        labels["mopar.io/role"] = role
    if alloc_mb is not None:
        labels["mopar.io/alloc-mb"] = _fmt_mb(alloc_mb)
    return labels


def _probe(delay: int, period: int) -> Dict[str, Any]:
    return {
        "httpGet": {"path": HEALTHZ_PATH, "port": SERVICE_PORT},
        "initialDelaySeconds": delay,
        "periodSeconds": period,
    }


def _slice_volume_sections(params: RenderParams) -> List[Dict[str, Any]]:
    mounts = [
        {"name": "shm", "mountPath": "/dev/shm"},
        {"name": "weights", "mountPath": "/models", "readOnly": True},
        {"name": "activations", "mountPath": "/activations"},
    ]
    volumes = [
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": SHM_SIZE_LIMIT}},
        {"name": "weights", "persistentVolumeClaim": {"claimName": params.weights_pvc, "readOnly": True}},
        {"name": "activations", "persistentVolumeClaim": {"claimName": params.activations_pvc}},
    ]
    return mounts, volumes


def _deployment_doc(s: Dict[str, Any], params: RenderParams, model: str, name_by_id: Dict[Any, str]) -> Dict[str, Any]:
    sid = s["slice_id"]
    dep_name = name_by_id[sid]
    role = s.get("type") or ("orchestrator" if sid == 0 else "worker")
    alloc = float(s["alloc_mb"])
    footprint = float(s["footprint_mb"])

    limit_mi = math.ceil((alloc + params.torch_base_mi) * params.headroom)
    request_mi = math.ceil(footprint)
    cpu_m = math.ceil(alloc / params.lambda_mb_per_vcpu)

    peers = s.get("outbound_peers") or []
    out_peers = ",".join(f"{name_by_id[p]}.{params.namespace}.svc:{SERVICE_PORT}" for p in peers)

    mounts, volumes = _slice_volume_sections(params)
    labels = _labels(model, sid=sid, role=role, alloc_mb=alloc)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": dep_name,
            "namespace": params.namespace,
            "labels": labels,
            "annotations": {"mopar.io/lambda-mb-per-vcpu": str(params.lambda_mb_per_vcpu)},
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"mopar.io/model": model, "mopar.io/slice-id": str(sid)}},
            "template": {
                "metadata": {"labels": dict(labels)},
                "spec": {
                    "containers": [
                        {
                            "name": "slice",
                            "image": params.image,
                            "ports": [{"name": "http", "containerPort": SERVICE_PORT}],
                            "env": [
                                {"name": "MOPAR_SLICE_ID", "value": str(sid)},
                                {"name": "MOPAR_OUT_PEERS", "value": out_peers},
                                {"name": "MOPAR_TRANSPORT", "value": "dual"},
                            ],
                            "resources": {
                                "requests": {"cpu": f"{cpu_m}m", "memory": f"{request_mi}Mi"},
                                "limits": {"memory": f"{limit_mi}Mi"},
                            },
                            "volumeMounts": mounts,
                            "readinessProbe": _probe(5, 10),
                            "livenessProbe": _probe(15, 20),
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


def _service_doc(s: Dict[str, Any], params: RenderParams, model: str, name_by_id: Dict[Any, str]) -> Dict[str, Any]:
    sid = s["slice_id"]
    role = s.get("type") or ("orchestrator" if sid == 0 else "worker")
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name_by_id[sid],
            "namespace": params.namespace,
            "labels": _labels(model, sid=sid, role=role),
        },
        "spec": {
            "type": "ClusterIP",
            "selector": {"mopar.io/model": model, "mopar.io/slice-id": str(sid)},
            "ports": [{"name": "http", "port": SERVICE_PORT, "targetPort": SERVICE_PORT}],
        },
    }


def _pvc_doc(name: str, access_mode: str, size: str, role: str, params: RenderParams, model: str) -> Dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "namespace": params.namespace,
            "labels": {"mopar.io/model": model, "mopar.io/role": role},
        },
        "spec": {
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": size}},
        },
    }


def _plan_meta_doc(manifest: Dict[str, Any], params: RenderParams, model: str, plan_source: str, gpu_meta: str) -> Dict[str, Any]:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "mopar-plan-meta",
            "namespace": params.namespace,
            "labels": {"mopar.io/model": model},
        },
        "data": {
            "model_key": str(manifest.get("model_key", "")),
            "slices_count": str(len(manifest["slices"])),
            "lambda_mb_per_vcpu": str(params.lambda_mb_per_vcpu),
            "anchor_name": ANCHOR_NAME,
            "torch_base_mi": str(params.torch_base_mi),
            "headroom": str(params.headroom),
            "manifest_sha256": sha,
            "generated_at": params.generated_at,
            "plan_source": plan_source,
            "gpu": gpu_meta,
        },
    }


def _difftest_job_doc(params: RenderParams, model: str, model_key: str, slice0_name: str, slices_count: int) -> Dict[str, Any]:
    labels = {"mopar.io/model": model, "mopar.io/role": "difftest"}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"mopar-{model}-difftest",
            "namespace": params.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": dict(labels)},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "difftest",
                            "image": params.diff_image,
                            "env": [
                                {"name": "MOPAR_ENDPOINT", "value": f"{slice0_name}.{params.namespace}.svc:{SERVICE_PORT}"},
                                {"name": "MOPAR_MODEL", "value": model_key},
                                {"name": "MOPAR_SLICES_COUNT", "value": str(slices_count)},
                                {"name": "MOPAR_DIFF_SAMPLES", "value": str(params.diff_samples)},
                                {"name": "MOPAR_DIFF_TOLERANCE", "value": _ForceQuotedStr(params.diff_tolerance)},
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "512Mi"},
                                "limits": {"memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"name": "activations", "mountPath": "/activations"},
                                {"name": "weights", "mountPath": "/models", "readOnly": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "activations", "persistentVolumeClaim": {"claimName": params.activations_pvc}},
                        {"name": "weights", "persistentVolumeClaim": {"claimName": params.weights_pvc}},
                    ],
                },
            },
        },
    }


def _dump(docs: List[Dict[str, Any]]) -> str:
    return yaml.safe_dump_all(
        docs,
        sort_keys=False,
        default_flow_style=False,
        width=120,
        explicit_start=True,
    )


def render(manifest: Dict[str, Any], params: RenderParams) -> Dict[str, str]:
    """Translate a slice manifest into {file_name: yaml_text} (deterministic)."""
    if params.lambda_mb_per_vcpu <= 0:
        raise ValueError(f"lambda_mb_per_vcpu must be > 0, got {params.lambda_mb_per_vcpu}")
    if params.headroom <= 0:
        raise ValueError(f"headroom must be > 0, got {params.headroom}")
    if params.torch_base_mi < 0:
        raise ValueError(f"torch_base_mi must be >= 0, got {params.torch_base_mi}")

    slices = _validate_linear_chain(manifest)
    model_key = str(manifest.get("model_key", "model"))
    model = k8s_name(model_key)
    name_by_id = {}
    for s in slices:
        raw_name = s.get("name") or f"{model_key}_slice_{s.get('slice_id')}"
        name_by_id[s["slice_id"]] = k8s_name(raw_name)

    plan_source = _plan_source(manifest)
    needs_gpu = _needs_gpu(manifest)
    if needs_gpu:
        _warn(
            f"model {model_key!r} looks like a GPU LLM (gpu_resource={params.gpu_resource}) "
            "but v0 renders CPU-only slices; stamping gpu: reserved into plan-meta"
        )
    if params.gpu_per_slice:
        _warn(f"gpu_per_slice={params.gpu_per_slice} requested but v0 does not render GPU slices; ignoring")
    gpu_meta = "reserved" if needs_gpu else "not-required"

    files: Dict[str, str] = {}

    files["10-pvc-activations.yaml"] = _dump(
        [_pvc_doc(params.activations_pvc, "ReadWriteMany", params.activations_size, "activations", params, model)]
    )
    files["11-pvc-weights.yaml"] = _dump(
        [_pvc_doc(params.weights_pvc, "ReadOnlyMany", params.weights_size, "weights", params, model)]
    )
    files["20-plan-meta-configmap.yaml"] = _dump([_plan_meta_doc(manifest, params, model, plan_source, gpu_meta)])

    for s in slices:
        dep_name = name_by_id[s["slice_id"]]
        files[f"30-{dep_name}.yaml"] = _dump(
            [
                _deployment_doc(s, params, model, name_by_id),
                _service_doc(s, params, model, name_by_id),
            ]
        )

    slice0 = slices[0]
    if slice0.get("trigger_type") == "api_gateway":
        path = slice0.get("path") or DEFAULT_INGRESS_PATH
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{model}-ingress",
                "namespace": params.namespace,
                "labels": {"mopar.io/model": model},
            },
            "spec": {
                "rules": [
                    {
                        "http": {
                            "paths": [
                                {
                                    "path": path,
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": name_by_id[slice0["slice_id"]],
                                            "port": {"number": SERVICE_PORT},
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        }
        files[INGRESS_FILE] = _dump([ingress])

    files["50-difftest-job.yaml"] = _dump(
        [_difftest_job_doc(params, model, model_key, name_by_id[slice0["slice_id"]], len(slices))]
    )
    return files


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m capsule.k8s_render",
        description="K0 renderer: slice_manifest.json -> deterministic Kubernetes YAML",
    )
    parser.add_argument("--manifest", required=True, help="path to slice_manifest.json")
    parser.add_argument("--out", required=True, help="output directory for the rendered YAML files")
    parser.add_argument("--namespace", default="mopar-dev")
    parser.add_argument("--image", required=True, help="slice-runner container image")
    parser.add_argument("--lambda-mb-per-vcpu", type=int, default=1769, help="AWS Lambda anchor MB/vCPU")
    parser.add_argument("--torch-base-mi", type=int, default=600, help="torch runtime memory floor in Mi")
    parser.add_argument("--headroom", type=float, default=1.3, help="limit headroom multiplier")
    parser.add_argument("--activations-pvc", default="mopar-activations")
    parser.add_argument("--weights-pvc", default="mopar-weights")
    parser.add_argument("--activations-size", default="20Gi")
    parser.add_argument("--weights-size", default="20Gi")
    parser.add_argument("--diff-image", default="harbor.example/mopar/difftest:0.1", help="differential-test Job image")
    parser.add_argument("--diff-samples", type=int, default=100, help="differential test input count N")
    parser.add_argument("--diff-tolerance", default="1e-4", help="differential test max-tensor-diff acceptance")
    parser.add_argument("--generated-at", default="", help="ISO-8601 UTC stamp; defaults to wall clock now")
    args = parser.parse_args(argv)

    generated_at = args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    params = RenderParams(
        image=args.image,
        generated_at=generated_at,
        namespace=args.namespace,
        lambda_mb_per_vcpu=args.lambda_mb_per_vcpu,
        torch_base_mi=args.torch_base_mi,
        headroom=args.headroom,
        activations_pvc=args.activations_pvc,
        weights_pvc=args.weights_pvc,
        activations_size=args.activations_size,
        weights_size=args.weights_size,
        diff_image=args.diff_image,
        diff_samples=args.diff_samples,
        diff_tolerance=args.diff_tolerance,
    )
    try:
        files = render(manifest, params)
    except ValueError as exc:
        print(f"[k8s_render] gate rejected the manifest: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        target = out_dir / fname
        target.write_text(text, encoding="utf-8")
        print(f"[k8s_render] wrote {target} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
