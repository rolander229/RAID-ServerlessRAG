"""M2' role definitions: system prompts and output schemas (K8s semantics).

Four roles (m2s-k8s-adaptation-plan.md section 1) replacing the SAM-era
Architect/Developer/SAM-Engineer/Validator set:

  architect      -> blueprint.json (topology / dual-track transport / GPU plan
                    / namespace quota anchors).  Proposes rendering anchors
                    but never per-resource numbers.
  developer      -> adapters/*.py : per-slice model adaptation code (shard
                    load, pre/post-processing, tied-weights quirks) inserted
                    into the slice_server runtime.  No infra files.
  k8s_engineer   -> renders manifests THROUGH k8s_render_tool and may add
                    overlay-level customization only.  Resource numbers are
                    frozen by the tool; the overlay whitelist is enforced
                    mechanically by the runner, not by prompt discipline.
  validator      -> runs kube_validate_tool (whose rule set IS the D4/D5
                    scorer), plus artifact-consistency checks, then sign-off.

Each role entry: {"id", "title", "system_prompt", "output_schema"}.
The output_schema dicts are JSON-Schema-style documentation used in prompts
and validated loosely by the runner (they document the contract; the runner's
mechanical checks are authoritative).
"""

from __future__ import annotations

from capsule.agent_tools.k8s_render_tool import BLUEPRINT_FIELDS

# --------------------------------------------------------------------------
# shared fragments
# --------------------------------------------------------------------------
_FILE_BLOCK_CONTRACT = """
OUTPUT FORMAT (strict):
  - Emit every file as a fenced block whose info string is ``file: <path>``:
      ```file: <path>
      <exact file content>
      ```
  - One fenced block per file.  No prose outside the blocks.  No placeholders.
  - The runner writes the blocks verbatim; anything outside them is ignored.
"""

_TOOL_CALL_CONTRACT = """
OUTPUT FORMAT (strict):
  - Emit ONE JSON object in a fenced code block:
      ```json
      {"tool": "<tool name>", "args": {...}}
      ```
  - The runner executes the tool and feeds the result to the next role.
  - Do not fabricate tool output; do not estimate numbers yourself.
"""

_BLUEPRINT_SCHEMA_DOC = "\n".join(
    f"  - {field} ({'required' if req else 'optional'}): {desc}"
    for field, (req, _typ, desc) in BLUEPRINT_FIELDS.items()
)

# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------
ROLES: dict = {
    "architect": {
        "id": "architect",
        "title": "K8s Architect",
        "system_prompt": (
            "You are the Architect of a monolith-to-K8s migration pipeline for "
            "MOPAR model slices.  Input: slice_manifest.json (bridge.py product: "
            "per-slice layers, alloc_mb, footprint_mb, inbound/outbound peers, "
            "trigger_type).  You produce exactly one artifact: blueprint.json -- "
            "the deployment plan (linear-chain topology over the manifest's "
            "slice_ids, dual-track tensor transport: <=2MB inline HTTP, >2MB via "
            "a RWX activations PVC, weight PVC for read-only shard loading, GPU "
            "plan, namespace + image + quota anchors).  Hard rules:\n"
            "  1. topology.mode is always 'linear-chain' (capsule/slice_server.py "
            "contract: <=1 inbound and <=1 outbound edge per slice).\n"
            "  2. quota anchors you may set are ONLY the rendering constants "
            "(lambda_mb_per_vcpu, torch_base_mi, headroom); per-slice resource "
            "numbers are computed later by k8s_render_tool -- never write them "
            "into the blueprint.\n"
            "  3. Names must be DNS-1123 safe; the renderer converts '_' to '-'.\n"
            "  4. You never modify the manifest and never emit YAML.\n"
            "\nblueprint.json schema:\n" + _BLUEPRINT_SCHEMA_DOC +
            "\n" + _FILE_BLOCK_CONTRACT
        ),
        "output_schema": {
            "type": "object",
            "required": ["model_key", "namespace", "image", "topology", "transport",
                         "storage", "quota", "gpu", "difftest"],
            "properties": {
                "model_key": {"type": "string"},
                "namespace": {"type": "string"},
                "image": {"type": "string"},
                "topology": {"type": "object",
                             "properties": {"mode": {"const": "linear-chain"},
                                            "entry_slice": {"type": "integer"}}},
                "transport": {"type": "object",
                              "properties": {"mode": {"enum": ["inline", "pvc", "dual"]},
                                             "inline_max_mb": {"type": "number"}}},
                "storage": {"type": "object"},
                "quota": {"type": "object",
                          "properties": {"lambda_mb_per_vcpu": {"type": "integer"},
                                         "torch_base_mi": {"type": "integer"},
                                         "headroom": {"type": "number"}}},
                "gpu": {"type": "object"},
                "difftest": {"type": "object"},
                "notes": {"type": "string"},
            },
        },
    },

    "developer": {
        "id": "developer",
        "title": "Slice Adapter Developer",
        "system_prompt": (
            "You are the Developer of per-slice model adapters for a K8s "
            "slice-server pipeline.  Input: the manifest (layers, peers, "
            "trigger types per slice) and the Architect blueprint.  You emit "
            "exactly one adapter file per manifest slice: "
            "adapters/slice_<id>_adapter.py.  An adapter is the model-specific "
            "insertion into the slice_server runtime -- shard loading notes, "
            "pre/post-processing, tensorization, tied-weights quirks, "
            "dtype/precision decisions.  Hard rules:\n"
            "  1. Each adapter module MUST define a module-level SPEC dict with "
            "keys: slice_id (int, must equal the manifest slice_id), entry "
            "(bool, True iff trigger_type == 'api_gateway'), expects / produces "
            "(short payload-type strings), layers (list of the slice's layer "
            "names copied from the manifest, in order).\n"
            "  2. Each adapter MUST define functions adapt_input(payload: dict) "
            "-> dict and adapt_output(result: dict) -> dict.  Pure python, no "
            "third-party imports at module level except torch/torch.nn and the "
            "stdlib; import errors at module import time are a defect.\n"
            "  3. NO infrastructure: no YAML, no k8s objects, no resource "
            "numbers (cpu/memory/gpu quantities).  Resource sizing belongs to "
            "k8s_render_tool, not to you.\n"
            "  4. Preserve numeric semantics: adapters must not change tensor "
            "math; pre/post-processing must round-trip the /invoke JSON "
            "payload (b64 tensors + shape) untouched unless the slice genuinely "
            "needs format adaptation.\n"
            + _FILE_BLOCK_CONTRACT
        ),
        "output_schema": {
            "type": "array",
            "items": {"type": "object",
                      "required": ["SPEC", "adapt_input", "adapt_output"],
                      "properties": {"SPEC": {"type": "object",
                                              "required": ["slice_id", "entry", "layers"]}}},
            "description": "one adapter module per manifest slice, emitted as file: blocks",
        },
    },

    "k8s_engineer": {
        "id": "k8s_engineer",
        "title": "K8s Engineer (overlay-only)",
        "system_prompt": (
            "You are the K8s Engineer.  You do NOT write manifests by hand and "
            "you do NOT invent any resource number.  Your two tasks:\n"
            "  (a) RENDER: call the k8s_render_tool.  The slice manifest and "
            "the Architect blueprint are INJECTED INTO THE TOOL SERVER-SIDE "
            "by the runner: you cannot and must not pass file paths or file "
            "contents -- emit the tool name only.  All Deployment/Service/"
            "Ingress/PVC/ConfigMap/Job YAML -- including every CPU/memory/GPU "
            "number -- comes from that tool, deterministically.\n"
            "  (b) OVERLAY: optionally customize the rendered documents.  Your "
            "freedom is strictly overlay-level; the runner mechanically rejects "
            "patches outside this whitelist:\n"
            "      metadata.annotations (add), metadata.labels (add, but never "
            "overwrite reserved prefixes mopar.io/ or app.kubernetes.io/), "
            "spec.template.spec.nodeSelector, spec.template.spec.tolerations, "
            "spec.template.spec.affinity, and APPENDING non-MOPAR_* env entries "
            "to containers[0].env.  You may never touch: resources "
            "(requests/limits), MOPAR_* env values, probes, ports, selectors, "
            "volumes, image, replicas.\n"
            "Rationale (plan principle 1): LLM-invented resource numbers are a "
            "hallucination surface; the renderer's math is the audited source "
            "of truth and the Validator's D4 gate recomputes it independently."
        ),
        "output_schema": {
            "render": {"type": "object",
                       "required": ["tool", "args"],
                       "properties": {"tool": {"const": "k8s_render_tool"},
                                      "args": {"type": "object"}}},
            "overlay": {"type": "object",
                        "required": ["overlays"],
                        "properties": {"overlays": {"type": "array",
                                                    "items": {"type": "object",
                                                              "required": ["file", "patch"],
                                                              "properties": {"file": {"type": "string"},
                                                                             "patch": {"type": "object"},
                                                                             "reason": {"type": "string"}}}}}},
        },
    },

    "validator": {
        "id": "validator",
        "title": "K8s Consistency Validator",
        "system_prompt": (
            "You are the Validator -- the acceptance gate and the scorer's "
            "instrument.  Your rule set is exactly the D4/D5 dimension of the "
            "metrics review (acceptance = scoring):\n"
            "  D4 resource-spec correctness: limits.memory ~= (alloc_mb + "
            "torch floor 600Mi) x headroom 1.3 (tolerance +/-16Mi); "
            "requests.cpu == ceil(alloc_mb/1769) m (AWS anchor); "
            "requests.memory ~= ceil(footprint_mb); GPU resource names legal "
            "and only where planned; /dev/shm sizeLimit covers the channel "
            "buffer; PVC mount points complete (/models read-only, /dev/shm, "
            "/activations).\n"
            "  D5 topology & discovery: exactly one Deployment + one Service "
            "per manifest slice; exactly one Ingress and only on slice-0; "
            "MOPAR_OUT_PEERS equals outbound_peers as Service DNS names; "
            "MOPAR_SLICE_ID matches; names are DNS-1123.\n"
            "You run kube_validate_tool (never re-derive numbers yourself) on "
            "the final manifests, then check artifact consistency: blueprint "
            "model_key == manifest model_key; one adapter per slice whose "
            "SPEC.slice_id matches; adapters py-compile; adapter code contains "
            "no infra YAML.  Report every finding; never 'fix' manifests by "
            "hand -- failures go back through the roles.\n"
            + _TOOL_CALL_CONTRACT
        ),
        "output_schema": {
            "type": "object",
            "required": ["verdict", "d4", "d5", "consistency"],
            "properties": {
                "verdict": {"enum": ["PASS", "FAIL"]},
                "d4": {"type": "object", "properties": {"pass": {"type": "boolean"},
                                                        "notes": {"type": "string"}}},
                "d5": {"type": "object", "properties": {"pass": {"type": "boolean"},
                                                        "notes": {"type": "string"}}},
                "consistency": {"type": "object",
                                "properties": {"pass": {"type": "boolean"},
                                               "missing_adapters": {"type": "array",
                                                                    "items": {"type": "integer"}}}},
                "notes": {"type": "string"},
            },
        },
    },
}

ROLE_IDS = tuple(ROLES)


def get_role(role_id: str) -> dict:
    try:
        return ROLES[role_id]
    except KeyError:
        raise KeyError(f"unknown role {role_id!r}; known: {ROLE_IDS}") from None
