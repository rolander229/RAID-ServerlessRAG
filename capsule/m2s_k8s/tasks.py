"""M2' task graph: six sequential tasks over four roles.

Task kinds (runner mechanics):
  files  -> LLM emits ``file: <path>`` fenced blocks; runner writes them
  tool   -> LLM emits {"tool": ..., "args": ...}; runner executes the tool
  json   -> LLM emits one JSON object; runner validates and records it

Order is strictly sequential (linear-chain pipeline); each task receives a
context summary of its predecessors' outputs.
"""

from __future__ import annotations

MANIFEST_SUMMARY_FIELDS = (
    "model_key, slices_count, total_layers, per-slice (slice_id, name, "
    "layers_count, alloc_mb, footprint_mb, trigger_type, inbound/outbound peers)"
)

TASKS: list = [
    {
        "id": "architect_blueprint",
        "role": "architect",
        "kind": "files",
        "title": "Architect: blueprint.json",
        "instruction": (
            "Design the K8s deployment blueprint for the manifest below.\n"
            "Write exactly one file: blueprint.json (the object itself, pretty-"
            "printed JSON).  Derive topology strictly from the manifest's "
            "inbound/outbound peers; entry_slice = the slice with no inbound "
            "peers.  namespace, image and PVC names/size are yours to fix; the "
            "quota anchors default to the AWS-Lambda-1769 canon "
            "(lambda_mb_per_vcpu=1769, torch_base_mi=600, headroom=1.3) unless "
            "the platform brief says otherwise.  GPU plan must state "
            "required=false for CPU-only manifests.  Set difftest to "
            "samples=100 tolerance=1e-4 (RQ3 acceptance).\n\n"
            "MANIFEST SUMMARY:\n{manifest_summary}\n"
        ),
    },
    {
        "id": "developer_adapters",
        "role": "developer",
        "kind": "files",
        "title": "Developer: per-slice adapters",
        "instruction": (
            "Write the per-slice adapter modules for the manifest below, under "
            "the blueprint already produced.\n"
            "Exactly one file per slice: adapters/slice_<slice_id>_adapter.py.  "
            "Copy each slice's `layers` list verbatim into SPEC['layers'].  "
            "slice 0 (entry) adapts the client JSON request into the /invoke "
            "payload; intermediate slices pass tensors through; the last slice "
            "adapts the final result back to a client-friendly response.  Note "
            "model-specific quirks in comments (tied weights, BN buffers, "
            "dtype).  No infra YAML, no resource numbers.\n\n"
            "MANIFEST SUMMARY:\n{manifest_summary}\n\n"
            "BLUEPRINT (blueprint.json):\n{blueprint}\n"
        ),
    },
    {
        "id": "engineer_render",
        "role": "k8s_engineer",
        "kind": "tool",
        "title": "K8s Engineer: render manifests via k8s_render_tool",
        "instruction": (
            "Render the K8s manifests for the manifest below using the "
            "blueprint produced earlier.  Call k8s_render_tool -- do NOT write "
            "YAML yourself, do NOT compute any resource number.\n"
            'Emit: {{"tool": "k8s_render_tool", "args": {{"manifest_ref": '
            '"<path or literal manifest>", "blueprint_ref": "<path or literal '
            'blueprint>", "params": {{"generated_at": "<ISO-8601 UTC>"}}}}}}\n'
            "The runner resolves manifest_ref/blueprint_ref to the pipeline "
            "artifacts and pins generated_at for byte-level determinism.\n\n"
            "MANIFEST SUMMARY:\n{manifest_summary}\n\n"
            "BLUEPRINT (blueprint.json):\n{blueprint}\n"
        ),
    },
    {
        "id": "engineer_overlay",
        "role": "k8s_engineer",
        "kind": "json",
        "title": "K8s Engineer: bounded overlay",
        "instruction": (
            "Decide the overlay for the rendered manifests below.  You may "
            "only patch: metadata.annotations, metadata.labels (non-reserved "
            "keys), spec.template.spec.nodeSelector / tolerations / affinity, "
            "and APPENDING non-MOPAR_* env entries.  Resource numbers, "
            "MOPAR_* env, probes, ports, selectors, volumes, image and "
            "replicas are frozen (the runner rejects such patches).\n"
            "Emit one JSON object:\n"
            '{{"overlays": [{{"file": "<rendered file name>", "patch": <JSON '
            'merge patch>, "reason": "<why>"}}]}}\n'
            "Use an empty overlays list when nothing needs customization.\n\n"
            "RENDER SUMMARY (per-slice numbers extracted from rendered YAML):\n"
            "{render_summary}\n\n"
            "RENDERED FILE NAMES:\n{render_files}\n"
        ),
    },
    {
        "id": "validator_checks",
        "role": "validator",
        "kind": "tool",
        "title": "Validator: kube_validate_tool on final manifests",
        "instruction": (
            "Validate the final manifests below (after overlay application) "
            "against the manifest and blueprint.  Call kube_validate_tool -- "
            "its rule set is the D4/D5 gate; do not re-derive anything.\n"
            'Emit: {{"tool": "kube_validate_tool", "args": {{}}}}  (the runner '
            "supplies the rendered files, manifest and blueprint).\n\n"
            "MANIFEST SUMMARY:\n{manifest_summary}\n\n"
            "BLUEPRINT (blueprint.json):\n{blueprint}\n\n"
            "FINAL MANIFEST FILES:\n{render_files}\n"
        ),
    },
    {
        "id": "validator_signoff",
        "role": "validator",
        "kind": "json",
        "title": "Validator: consistency checks and sign-off",
        "instruction": (
            "Produce the final verdict.  Check artifact consistency: blueprint "
            "model_key vs manifest; one adapter per manifest slice with "
            "SPEC.slice_id matching (adapters listed below; the runner "
            "py-compiles them); validator tool findings reproduced below.  "
            "Emit one JSON object:\n"
            '{{"verdict": "PASS"|"FAIL", "d4": {{"pass": bool, "notes": str}}, '
            '"d5": {{"pass": bool, "notes": str}}, "consistency": {{"pass": '
            'bool, "missing_adapters": [int]}}, "notes": str}}\n\n'
            "MANIFEST SUMMARY:\n{manifest_summary}\n\n"
            "ADAPTER FILES:\n{adapter_files}\n\n"
            "KUBE_VALIDATE RESULT:\n{validation}\n\n"
            "EXPERT-BASELINE CROSS-CHECK (independent derivation vs render):\n"
            "{expert_check}\n"
        ),
    },
]


def get_task(task_id: str) -> dict:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise KeyError(f"unknown task {task_id!r}; known: {[t['id'] for t in TASKS]}")


def task_marker(task: dict, turn: int) -> str:
    """Machine-readable marker the runner prepends to every user message;
    FakeLLM keys its canned responses off this."""
    return f"# m2s-k8s task: {task['id']} role: {task['role']} turn: {turn}"
