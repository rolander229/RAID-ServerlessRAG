"""k8s_render_tool: the K0 renderer wrapped as a callable agent tool.

Design (m2s-k8s-adaptation-plan.md, principle 1): resource numbers come from
the deterministic renderer only -- the K8s Engineer agent calls this tool and
may never invent numbers.  The tool therefore:

  1. validates the Architect's blueprint (missing/ill-typed fields -> a clear
     error naming the offending path, not a stack trace),
  2. translates blueprint -> capsule.k8s_render.RenderParams,
  3. calls capsule.k8s_render.render() (unchanged K0 code path),
  4. returns {file_name: yaml_text} plus a rendering summary that the runner
     embeds in report.json and the Validator cross-checks.

Tool surface is deliberately CrewAI-compatible: plain callable class with
``name`` / ``description`` attributes and a ``run(**kwargs) -> dict`` method
(swapping in crewai.tools.BaseTool later only adds a pydantic args_schema).

The blueprint schema lives here (BLUEPRINT_FIELDS) and is mirrored by
capsule/m2s_k8s/roles.py (Architect prompt) and the FakeLLM canned response.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from capsule.k8s_render import RenderParams, render

# blueprint field -> (required?, type or tuple of types, description)
BLUEPRINT_FIELDS: Dict[str, tuple] = {
    "model_key": (True, str, "model identifier, must match manifest.model_key"),
    "namespace": (True, str, "target K8s namespace"),
    "image": (True, str, "slice-runner container image"),
    "topology": (True, dict, "{mode: 'linear-chain', entry_slice: int}"),
    "transport": (True, dict, "{mode: inline|pvc|dual, inline_max_mb, activations_pvc, activations_size}"),
    "storage": (True, dict, "{weights_pvc, weights_size}"),
    "quota": (True, dict, "{lambda_mb_per_vcpu, torch_base_mi, headroom} -- rendering anchors"),
    "gpu": (True, dict, "{required: bool, resource_name: str}"),
    "difftest": (True, dict, "{samples: int, tolerance: str, image: str}"),
    "node_affinity": (False, (dict, type(None)), "optional node affinity for GPU slices"),
    "notes": (False, str, "free-text architecture notes"),
}

TRANSPORT_MODES = ("inline", "pvc", "dual")


class BlueprintError(ValueError):
    """Raised when the Architect blueprint misses or mistypes a field."""


def validate_blueprint(blueprint: Any) -> Dict[str, Any]:
    """Structural validation with per-path error messages (no LLM guessing)."""
    if not isinstance(blueprint, dict):
        raise BlueprintError(f"blueprint must be a JSON object, got {type(blueprint).__name__}")
    for field, (required, typ, desc) in BLUEPRINT_FIELDS.items():
        if field not in blueprint:
            if required:
                raise BlueprintError(f"blueprint.{field} is missing -- required ({desc})")
            continue
        if isinstance(typ, tuple):
            if not isinstance(blueprint[field], typ):
                raise BlueprintError(
                    f"blueprint.{field} must be one of {[t.__name__ for t in typ]}, "
                    f"got {type(blueprint[field]).__name__}"
                )
        elif not isinstance(blueprint[field], typ):
            raise BlueprintError(
                f"blueprint.{field} must be {typ.__name__}, got {type(blueprint[field]).__name__}"
            )
    # nested required keys
    nested = {
        "topology": ("mode", "entry_slice"),
        "transport": ("mode", "inline_max_mb", "activations_pvc", "activations_size"),
        "storage": ("weights_pvc", "weights_size"),
        "quota": ("lambda_mb_per_vcpu", "torch_base_mi", "headroom"),
        "gpu": ("required", "resource_name"),
        "difftest": ("samples", "tolerance", "image"),
    }
    for parent, keys in nested.items():
        for key in keys:
            if key not in blueprint[parent]:
                raise BlueprintError(f"blueprint.{parent}.{key} is missing")
    if blueprint["topology"]["mode"] != "linear-chain":
        raise BlueprintError(
            f"blueprint.topology.mode must be 'linear-chain' (capsule/slice_server.py "
            f"contract), got {blueprint['topology']['mode']!r}"
        )
    mode = blueprint["transport"]["mode"]
    if mode not in TRANSPORT_MODES:
        raise BlueprintError(
            f"blueprint.transport.mode must be one of {TRANSPORT_MODES}, got {mode!r}"
        )
    inline_max = blueprint["transport"]["inline_max_mb"]
    if isinstance(inline_max, bool) or not isinstance(inline_max, (int, float)) or inline_max <= 0:
        raise BlueprintError(
            f"blueprint.transport.inline_max_mb must be a number > 0, got {inline_max!r}"
        )
    quota = blueprint["quota"]
    lam = quota["lambda_mb_per_vcpu"]
    if isinstance(lam, bool) or not isinstance(lam, int) or lam <= 0:
        raise BlueprintError(
            f"blueprint.quota.lambda_mb_per_vcpu must be an int > 0, got {lam!r}"
        )
    if not isinstance(quota["torch_base_mi"], int) or quota["torch_base_mi"] < 0:
        raise BlueprintError(
            f"blueprint.quota.torch_base_mi must be an int >= 0, got {quota['torch_base_mi']!r}"
        )
    headroom = quota["headroom"]
    if isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or headroom <= 0:
        raise BlueprintError(
            f"blueprint.quota.headroom must be a number > 0, got {headroom!r}"
        )
    samples = blueprint["difftest"]["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise BlueprintError(
            f"blueprint.difftest.samples must be an int > 0, got {samples!r}"
        )
    return blueprint


def load_manifest(manifest: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Accept a path to slice_manifest.json or an in-memory dict."""
    if isinstance(manifest, dict):
        return manifest
    path = Path(manifest)
    if not path.exists():
        raise FileNotFoundError(f"manifest file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest {path} is not valid JSON: {exc}") from exc


class K8sRenderTool:
    """Callable tool: (manifest, blueprint, params) -> {file_name: yaml_text} + summary."""

    name = "k8s_render_tool"
    description = (
        "Deterministically render slice_manifest.json + Architect blueprint into "
        "Kubernetes YAML documents. Input: manifest (path or dict), blueprint (dict), "
        "optional params {generated_at, diff_image, ...}. Returns "
        "{'files': {name: yaml_text}, 'summary': {...}}. Resource numbers are "
        "computed here and may not be edited downstream (overlay-only boundary)."
    )

    def run(self,
            manifest: Union[str, Path, Dict[str, Any]],
            blueprint: Dict[str, Any],
            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        blueprint = validate_blueprint(blueprint)
        manifest = load_manifest(manifest)
        params = dict(params or {})

        manifest_model = str(manifest.get("model_key", ""))
        if manifest_model and manifest_model != blueprint["model_key"]:
            raise BlueprintError(
                f"blueprint.model_key {blueprint['model_key']!r} does not match "
                f"manifest.model_key {manifest_model!r}"
            )

        # generated_at must be pinned by the caller for byte-level determinism;
        # the tool only falls back to the wall clock when the runner allows it.
        generated_at = params.get("generated_at") or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        render_params = RenderParams(
            image=blueprint["image"],
            generated_at=generated_at,
            namespace=blueprint["namespace"],
            lambda_mb_per_vcpu=blueprint["quota"]["lambda_mb_per_vcpu"],
            torch_base_mi=blueprint["quota"]["torch_base_mi"],
            headroom=float(blueprint["quota"]["headroom"]),
            activations_pvc=blueprint["transport"]["activations_pvc"],
            weights_pvc=blueprint["storage"]["weights_pvc"],
            activations_size=blueprint["transport"]["activations_size"],
            weights_size=blueprint["storage"]["weights_size"],
            diff_image=blueprint["difftest"]["image"],
            diff_samples=blueprint["difftest"]["samples"],
            diff_tolerance=blueprint["difftest"]["tolerance"],
            gpu_resource=blueprint["gpu"]["resource_name"],
        )

        warnings: list[str] = []
        if blueprint["transport"]["mode"] != "dual":
            warnings.append(
                f"blueprint.transport.mode={blueprint['transport']['mode']!r} but the K0 "
                "renderer pins MOPAR_TRANSPORT=dual for every slice; overlay env "
                "MOPAR_TRANSPORT if a single track is required"
            )
        if blueprint["gpu"]["required"]:
            warnings.append(
                f"blueprint.gpu.required=true: v0 renders CPU-only slices "
                f"({blueprint['gpu']['resource_name']} not allocated); "
                "plan-meta stamps gpu: reserved"
            )
        if blueprint["transport"]["inline_max_mb"] != 2:
            warnings.append(
                f"blueprint.transport.inline_max_mb={blueprint['transport']['inline_max_mb']!r} "
                "differs from the 2MB plan anchor; adapters/clients must agree"
            )

        try:
            files = render(manifest, render_params)
        except ValueError as exc:
            # linear-chain gate / bad alloc / unknown peer -> interception point
            raise ValueError(f"k8s_render_tool rejected the manifest: {exc}") from exc

        summary = _summarize(manifest, blueprint, files, generated_at, warnings)
        return {"files": files, "summary": summary}


def _summarize(manifest: Dict[str, Any], blueprint: Dict[str, Any],
                files: Dict[str, str], generated_at: str,
                warnings: list) -> Dict[str, Any]:
    """Per-slice numbers EXTRACTED from the rendered YAML (provenance: the
    actual output, not a re-derivation of the formula) + inventory, for
    report.json and downstream cross-checks."""
    per_slice = []
    dep_by_id = {}
    for fname, text in sorted(files.items()):
        for doc in yaml.safe_load_all(text):
            if isinstance(doc, dict) and doc.get("kind") == "Deployment":
                labels = doc.get("metadata", {}).get("labels", {})
                dep_by_id[labels.get("mopar.io/slice-id")] = doc
    for s in manifest.get("slices", []):
        doc = dep_by_id.get(str(s.get("slice_id")))
        resources = (((doc or {}).get("spec", {}).get("template", {})
                      .get("spec", {}).get("containers") or [{}])[0]
                     .get("resources", {}))
        peers_env = ""
        if doc:
            for env in doc["spec"]["template"]["spec"]["containers"][0].get("env", []):
                if env.get("name") == "MOPAR_OUT_PEERS":
                    peers_env = env.get("value", "")
        per_slice.append({
            "slice_id": s.get("slice_id"),
            "name": s.get("name"),
            "layers_count": len(s.get("layers") or []),
            "alloc_mb": float(s["alloc_mb"]),
            "footprint_mb": float(s["footprint_mb"]),
            "limits_memory": resources.get("limits", {}).get("memory"),
            "requests_cpu": resources.get("requests", {}).get("cpu"),
            "requests_memory": resources.get("requests", {}).get("memory"),
            "out_peers_env": peers_env,
            "outbound_peers": s.get("outbound_peers") or [],
            "inbound_peers": s.get("inbound_peers") or [],
        })
    return {
        "model_key": manifest.get("model_key"),
        "namespace": blueprint["namespace"],
        "image": blueprint["image"],
        "generated_at": generated_at,
        "transport": dict(blueprint["transport"]),
        "quota": dict(blueprint["quota"]),
        "file_names": sorted(files),
        "file_count": len(files),
        "total_bytes": sum(len(t.encode("utf-8")) for t in files.values()),
        "per_slice": per_slice,
        "warnings": warnings,
        "tool": "k8s_render_tool",
    }


# convenience: parse a multi-doc yaml stream back into documents (used by tests
# and the validator when it receives rendered text rather than files dict)
def parse_stream(text: str):
    return [d for d in yaml.safe_load_all(text) if d is not None]
