"""Canned FakeLLM responses: the resnet50 golden path, offline.

``build_fake_responses(manifest, generated_at)`` simulates what a competent
LLM would emit for each of the six tasks when the pipeline runs without a
real model:

  * the blueprint is hand-written here (literal canon numbers, exactly as an
    architect would type them) -- deliberately NOT derived through
    expert_baseline.derive_expected, so the pipeline test still exercises the
    expert-baseline cross-check as an independent path;
  * adapter SPEC['layers'] is filled from the manifest (a real developer
    would copy them from the context the runner provides);
  * the engineer/validator tasks emit the exact tool-call JSON contract.

For other models, build your own responses dict and pass it to FakeLLM.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from capsule.agent_tools.mapping_rules import k8s_name


def _entry_slice(manifest: Dict[str, Any]) -> int:
    for s in manifest.get("slices", []):
        if not (s.get("inbound_peers") or []):
            return s.get("slice_id", 0)
    return 0


def _dep_file(manifest: Dict[str, Any], sid: int) -> str:
    for s in manifest.get("slices", []):
        if s.get("slice_id") == sid:
            dep = k8s_name(s.get("name") or
                           f"{manifest.get('model_key', 'model')}_slice_{sid}")
            return f"30-{dep}.yaml"
    raise KeyError(f"slice_id {sid} not in manifest")


def _blueprint_text(manifest: Dict[str, Any]) -> str:
    blueprint = {
        "model_key": manifest.get("model_key"),
        "namespace": "mopar-dev",
        "image": "harbor.example/mopar/slice-runner:0.1",
        "topology": {"mode": "linear-chain",
                     "entry_slice": _entry_slice(manifest)},
        "transport": {"mode": "dual", "inline_max_mb": 2,
                      "activations_pvc": "mopar-activations",
                      "activations_size": "20Gi"},
        "storage": {"weights_pvc": "mopar-weights", "weights_size": "20Gi"},
        "quota": {"lambda_mb_per_vcpu": 1769, "torch_base_mi": 600,
                  "headroom": 1.3},
        "gpu": {"required": False, "resource_name": "nvidia.com/v100"},
        "difftest": {"samples": 100, "tolerance": "1e-4",
                     "image": "harbor.example/mopar/difftest:0.1"},
        "notes": ("canned FakeLLM blueprint: CPU linear chain, dual-track "
                  "tensor transport (<=2MB inline, >2MB activations PVC), "
                  "AWS-Lambda-1769 quota anchors, no GPU for this manifest"),
    }
    return "```file: blueprint.json\n" + \
        json.dumps(blueprint, ensure_ascii=False, indent=2) + "\n```"


def _adapter_text(manifest: Dict[str, Any], s: Dict[str, Any]) -> str:
    sid = s.get("slice_id")
    layers = list(s.get("layers") or [])
    entry = s.get("trigger_type") == "api_gateway"
    is_first = sid == 0
    is_last = not (s.get("outbound_peers") or [])
    if is_first:
        expects, produces = "client_json", "b64_tensor"
        role_note = ("Entry slice: adapts the client JSON request into the "
                     "slice_server /invoke payload (b64 tensor + shape).")
        input_body = (
            '    if "x_b64" not in payload or "shape" not in payload:\n'
            '        raise ValueError("entry slice requires x_b64 and shape")\n'
            '    return {"x_b64": payload["x_b64"], "shape": list(payload["shape"])}'
        )
        output_body = "    return result"
    elif is_last:
        expects, produces = "b64_tensor", "client_json"
        role_note = ("Final slice: shapes the last-hop result back into a "
                     "client-friendly JSON response.")
        input_body = "    return payload"
        output_body = (
            '    if isinstance(result, dict) and "y_b64" in result:\n'
            '        return {"pred_b64": result["y_b64"],\n'
            '                "shape": list(result.get("shape", [])),\n'
            '                "slice_id": SPEC["slice_id"]}\n'
            "    return result"
        )
    else:
        expects, produces = "b64_tensor", "b64_tensor"
        role_note = "Intermediate slice: pure tensor relay through /invoke."
        input_body = "    return payload"
        output_body = "    return result"
    return (
        f'```file: adapters/slice_{sid}_adapter.py\n'
        f'"""Slice adapter for {s.get("name")} (M2\' Developer role, FakeLLM canned).\n'
        f'\n'
        f'{role_note}\n'
        f'Model quirks: BN running buffers ride in the weight shard; tied-weights\n'
        f'constraints (lm_head/embed_tokens) do not apply to this manifest.\n'
        f'"""\n'
        f'\n'
        f'SPEC = {{\n'
        f'    "slice_id": {sid},\n'
        f'    "entry": {entry},\n'
        f'    "expects": "{expects}",\n'
        f'    "produces": "{produces}",\n'
        f'    "layers": {layers!r},\n'
        f'}}\n'
        f'\n'
        f'\n'
        f'def adapt_input(payload: dict) -> dict:\n'
        f'    """Client/hop payload -> slice_server /invoke body."""\n'
        f'{input_body}\n'
        f'\n'
        f'\n'
        f'def adapt_output(result: dict) -> dict:\n'
        f'    """Slice result -> next hop or client."""\n'
        f'{output_body}\n'
        f'```\n'
    )


def build_fake_responses(manifest: Dict[str, Any],
                         generated_at: Optional[str] = None) -> Dict[str, str]:
    """task_id -> canned LLM response for the whole six-task pipeline."""
    stamp = generated_at or "2026-11-03T00:00:00Z"
    adapters = "\n".join(_adapter_text(manifest, s)
                         for s in manifest.get("slices", []))
    render_call = (
        "```json\n" +
        json.dumps({
            "tool": "k8s_render_tool",
            "args": {
                "manifest_ref": "state.manifest",
                "blueprint_ref": "state.blueprint.json",
                "params": {"generated_at": stamp},
            },
        }, indent=2)
        + "\n```"
    )
    overlay = (
        "```json\n" +
        json.dumps({
            "overlays": [
                {
                    "file": _dep_file(manifest, _entry_slice(manifest)),
                    "target_kind": "Deployment",
                    "patch": {
                        "annotations": {"mopar.io/pipeline": "m2s-m2prime"},
                        "append_env": [{"name": "M2S_ADAPTER_SET",
                                        "value": "fake-canned-1"}],
                    },
                    "reason": ("provenance annotation + adapter-set selector on "
                               "the entry slice; resource numbers untouched"),
                }
            ]
        }, indent=2)
        + "\n```"
    )
    validate_call = (
        "```json\n" +
        json.dumps({"tool": "kube_validate_tool", "args": {}}, indent=2)
        + "\n```"
    )
    signoff = (
        "```json\n" +
        json.dumps({
            "verdict": "PASS",
            "d4": {"pass": True,
                   "notes": "limits/requests recomputed by the validator from the "
                            "manifest; numbers come only from k8s_render_tool"},
            "d5": {"pass": True,
                   "notes": "one Deployment+Service per slice; single Ingress on "
                            "the entry slice; MOPAR_OUT_PEERS matches Service DNS"},
            "consistency": {"pass": True, "missing_adapters": []},
            "notes": "canned FakeLLM sign-off (the runner's mechanical gate is "
                     "authoritative, not this verdict)",
        }, indent=2)
        + "\n```"
    )
    return {
        "architect_blueprint": _blueprint_text(manifest),
        "developer_adapters": adapters,
        "engineer_render": render_call,
        "engineer_overlay": overlay,
        "validator_checks": validate_call,
        "validator_signoff": signoff,
    }
