"""M2' sequential pipeline runner: manifest -> agents -> k8s_migrated/.

Orchestrates the four roles (capsule/m2s_k8s/roles.py) over six sequential
tasks (tasks.py) with file-handover artifacts, replacing the CrewAI-era
capsule/M2S/src/main.py for the K8s target.  The LLM is behind the
``LLMClient.chat(messages) -> str`` interface (capsule/m2s_k8s/llm.py):
FakeLLM for offline/CI runs, ArkLLM for the real Ark base -- swapping in
CrewAI later means re-implementing this file's task loop on top of a Crew,
not changing the tools, roles or contracts.

Output contract (k8s_migrated/, per m2s-k8s-adaptation-plan.md):

    <out_dir>/
      blueprint.json       Architect artifact
      manifests/*.yaml     rendered (and overlaid) K8s documents
      adapters/*.py        per-slice adapter code (Developer)
      report.json          per-task input/output digests, checks, timings

Mechanical enforcement (not prompt discipline):
  * resource numbers only from k8s_render_tool -- the engineer's overlay
    patches are whitelist-checked (OVERLAY_KEYS) and anything touching
    resources / MOPAR_* env / probes / ports / volumes is rejected;
  * broken manifests (non-linear chain, bad alloc, unknown peers, invalid
    JSON) are intercepted at manifest load or by the render tool's gate --
    recorded as ``interceptions`` in report.json, remaining tasks skipped;
  * adapters are py-compiled and their SPEC.slice_id must cover the manifest.

CLI::

    python -m capsule.m2s_k8s.runner --manifest tests/fixtures/resnet50_manifest.json \
        --out runs/m2s_fake_resnet50 --fake-llm --generated-at 2026-11-03T00:00:00Z
    python -m capsule.m2s_k8s.runner --check-llm
Exit codes: 0 pass, 1 pipeline ran but validation/verdict failed, 2 intercepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from capsule.agent_tools.k8s_render_tool import K8sRenderTool, load_manifest, validate_blueprint
from capsule.agent_tools.kube_validate_tool import KubeValidateTool
from capsule.m2s_k8s.expert_baseline import compare_expected_vs_render, derive_expected
from capsule.m2s_k8s.llm import CONFIG_GUIDANCE, ArkLLM, FakeLLM, LLMClient
from capsule.m2s_k8s.roles import ROLES
from capsule.m2s_k8s.tasks import TASKS, task_marker

REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVED_LABEL_PREFIXES = ("mopar.io/", "app.kubernetes.io/")
MOPAR_ENV_PREFIX = "MOPAR_"
FILE_BLOCK_RE = re.compile(r"```file:\s*(\S+)\s*\n(.*?)```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*```", re.DOTALL)
SPEC_SLICE_ID_RE = re.compile(r"slice_id\s*\"?\s*[:=]\s*\"?(\d+)")

# overlay patch vocabulary -- the ONLY keys an engineer patch may contain
OVERLAY_KEYS = {
    "annotations",      # -> metadata.annotations (merged)
    "labels",           # -> metadata.labels (merged, reserved prefixes rejected)
    "pod_annotations",  # -> spec.template.metadata.annotations (merged)
    "pod_labels",       # -> spec.template.metadata.labels (merged, reserved rejected)
    "node_selector",    # -> spec.template.spec.nodeSelector (merged)
    "tolerations",      # -> spec.template.spec.tolerations (replaced)
    "affinity",         # -> spec.template.spec.affinity (replaced)
    "append_env",       # -> containers[0].env append (MOPAR_* rejected)
}


class PipelineError(RuntimeError):
    """A task produced unusable output -- the pipeline records and aborts."""


class OverlayError(PipelineError):
    """An overlay patch left the whitelist."""


# --------------------------------------------------------------------------
# LLM output parsing
# --------------------------------------------------------------------------
def parse_file_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract (path, content) from ``file:`` fenced blocks."""
    return [(m.group(1).strip(), m.group(2)) for m in FILE_BLOCK_RE.finditer(text)]


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse the LLM's JSON object: direct, else last fenced json block."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    candidates = JSON_BLOCK_RE.findall(text)
    for raw in reversed(candidates):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise PipelineError("no parseable JSON object in the response "
                        f"(head: {stripped[:200]!r})")


# --------------------------------------------------------------------------
# overlay application (whitelist-enforced)
# --------------------------------------------------------------------------
def _merge(dst: Dict[str, Any], extra: Dict[str, Any], what: str) -> None:
    if not isinstance(extra, dict):
        raise OverlayError(f"overlay {what} must be an object, got {type(extra).__name__}")
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v, what)
        else:
            dst[k] = v


def _labels_guard(extra: Dict[str, Any], file: str) -> None:
    for key in extra:
        if any(key.startswith(p) for p in RESERVED_LABEL_PREFIXES):
            raise OverlayError(
                f"{file}: overlay label {key!r} touches a reserved prefix "
                f"{RESERVED_LABEL_PREFIXES} -- renderer-owned labels are frozen")


def apply_overlay(files: Dict[str, str], file: str, patch: Dict[str, Any],
                  target_kind: Optional[str] = None) -> int:
    """Apply one whitelisted overlay patch in place; returns docs patched."""
    if file not in files:
        raise OverlayError(f"overlay targets unknown file {file!r}; "
                           f"rendered files: {sorted(files)}")
    if not isinstance(patch, dict):
        raise OverlayError(f"{file}: overlay patch must be an object")
    illegal = sorted(set(patch) - OVERLAY_KEYS)
    if illegal:
        raise OverlayError(
            f"{file}: overlay patch key(s) {illegal} outside the whitelist "
            f"{sorted(OVERLAY_KEYS)} -- resource numbers and renderer-owned "
            "fields are frozen (plan principle 1)")

    docs = [d for d in yaml.safe_load_all(files[file]) if d is not None]
    if target_kind is None:
        if len(docs) != 1:
            raise OverlayError(f"{file}: contains {len(docs)} documents -- overlay "
                               "must declare target_kind")
        targets = docs
    else:
        targets = [d for d in docs if d.get("kind") == target_kind]
        if not targets:
            raise OverlayError(f"{file}: no document with kind {target_kind!r}")

    for doc in targets:
        if "annotations" in patch:
            meta = doc.setdefault("metadata", {}).setdefault("annotations", {})
            _merge(meta, patch["annotations"], "annotations")
        if "labels" in patch:
            _labels_guard(patch["labels"], file)
            _merge(doc.setdefault("metadata", {}).setdefault("labels", {}),
                   patch["labels"], "labels")
        if "pod_annotations" in patch:
            tpl = doc.setdefault("spec", {}).setdefault("template", {})
            _merge(tpl.setdefault("metadata", {}).setdefault("annotations", {}),
                   patch["pod_annotations"], "pod_annotations")
        if "pod_labels" in patch:
            _labels_guard(patch["pod_labels"], file)
            tpl = doc.setdefault("spec", {}).setdefault("template", {})
            _merge(tpl.setdefault("metadata", {}).setdefault("labels", {}),
                   patch["pod_labels"], "pod_labels")
        if "node_selector" in patch:
            pspec = doc.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})
            _merge(pspec.setdefault("nodeSelector", {}), patch["node_selector"], "node_selector")
        if "tolerations" in patch:
            doc["spec"]["template"].setdefault("spec", {})["tolerations"] = patch["tolerations"]
        if "affinity" in patch:
            doc["spec"]["template"].setdefault("spec", {})["affinity"] = patch["affinity"]
        if "append_env" in patch:
            entries = patch["append_env"]
            if not isinstance(entries, list):
                raise OverlayError(f"{file}: append_env must be a list of env objects")
            containers = doc["spec"]["template"].setdefault("spec", {}).setdefault("containers", [])
            if not containers:
                raise OverlayError(f"{file}: append_env found no container")
            env = containers[0].setdefault("env", [])
            known = {e.get("name") for e in env}
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("name"):
                    raise OverlayError(f"{file}: append_env entry needs a name: {entry!r}")
                if str(entry["name"]).startswith(MOPAR_ENV_PREFIX):
                    raise OverlayError(
                        f"{file}: append_env {entry['name']!r} uses the reserved "
                        f"{MOPAR_ENV_PREFIX} prefix -- pipeline env is renderer-owned")
                if entry["name"] in known:
                    raise OverlayError(f"{file}: env {entry['name']!r} already exists")
                env.append({"name": entry["name"], "value": str(entry.get("value", ""))})
                known.add(entry["name"])

    files[file] = yaml.safe_dump_all(
        docs, sort_keys=False, default_flow_style=False, width=120, explicit_start=True,
    )
    return len(targets)


# --------------------------------------------------------------------------
# adapter consistency (Validator's mechanical part)
# --------------------------------------------------------------------------
def check_adapters(adapter_files: Dict[str, str], manifest: Dict[str, Any]) -> List[str]:
    """py-compile every adapter; SPEC.slice_id must match manifest slices 1:1."""
    problems: List[str] = []
    found: Dict[int, str] = {}
    for path, content in sorted(adapter_files.items()):
        try:
            compile(content, path, "exec")
        except SyntaxError as exc:
            problems.append(f"adapter {path}: SyntaxError: {exc}")
            continue
        m = SPEC_SLICE_ID_RE.search(content)
        if not m:
            problems.append(f"adapter {path}: no SPEC.slice_id found")
            continue
        sid = int(m.group(1))
        if sid in found:
            problems.append(f"adapter {path}: slice_id {sid} already covered by {found[sid]}")
        found[sid] = path
        if "def adapt_input" not in content or "def adapt_output" not in content:
            problems.append(f"adapter {path}: missing adapt_input/adapt_output hook")
        if "resources" in content and re.search(r"\b(limits|requests)\s*[:=]", content):
            problems.append(f"adapter {path}: looks like it declares K8s resources -- "
                            "adapters must not size pods")
    for s in manifest.get("slices", []):
        sid = s.get("slice_id")
        if sid not in found:
            problems.append(f"manifest slice {sid}: no adapter file")
    return problems


# --------------------------------------------------------------------------
# context digests (what each role sees of its predecessors)
# --------------------------------------------------------------------------
def _manifest_summary(manifest: Dict[str, Any], full_layers: bool = False) -> str:
    lines = [f"model_key={manifest.get('model_key')} slices_count={manifest.get('slices_count')} "
             f"total_layers={manifest.get('total_layers')} "
             f"plan_source={manifest.get('plan_source', 'unspecified')}"]
    for s in manifest.get("slices", []):
        layers = s.get("layers") or []
        line = (f"- slice {s.get('slice_id')} name={s.get('name')} "
                f"alloc_mb={s.get('alloc_mb')} footprint_mb={s.get('footprint_mb')} "
                f"trigger_type={s.get('trigger_type')} type={s.get('type', 'auto')} "
                f"in={s.get('inbound_peers', [])} out={s.get('outbound_peers', [])} "
                f"layers({len(layers)})")
        if full_layers:
            line += f"={layers!r}"
        lines.append(line)
    return "\n".join(lines)


def _render_files_summary(files: Dict[str, str]) -> str:
    return "\n".join(f"- {name} ({len(text.encode('utf-8'))} bytes)"
                     for name, text in sorted(files.items()))


def _digest(obj: Any, limit: int = 1600) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + f"...[truncated {len(text)} chars]"


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
class _TaskRecorder:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def start(self, task: Dict[str, Any]) -> Dict[str, Any]:
        step = {"task_id": task["id"], "role": task["role"], "kind": task["kind"],
                "status": "running", "input_summary": "", "output_summary": "",
                "duration_ms": None, "error": None}
        self.steps.append(step)
        return step

    def finish(self, step, status: str, input_summary: str, output_summary: str,
                error: Optional[str] = None):
        step["status"] = status
        step["input_summary"] = input_summary
        step["output_summary"] = output_summary
        step["error"] = error


def _safe_out_path(out_dir: Path, rel_path: str) -> Path:
    """Resolve an artifact path under out_dir, refusing traversal."""
    rel = rel_path.lstrip("/")
    if ".." in Path(rel).parts:
        raise PipelineError(f"artifact path escapes out_dir: {rel_path!r}")
    return out_dir / rel


def _write_artifact(out_dir: Path, rel_path: str, content: str) -> Path:
    target = _safe_out_path(out_dir, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _load_manifest_or_intercept(manifest_path: Union[str, Path],
                                recorder: _TaskRecorder) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        manifest = load_manifest(manifest_path)
        if not isinstance(manifest, dict) or not manifest.get("slices"):
            raise ValueError("manifest must be an object with a non-empty 'slices' list")
        return manifest, None
    except Exception as exc:  # noqa: BLE001 -- interception point
        err = f"manifest load failed: {type(exc).__name__}: {exc}"
        step = recorder.start({"id": "manifest_load", "role": "runner", "kind": "io"})
        recorder.finish(step, "intercepted", str(manifest_path), "", err)
        return None, err


def run(manifest_path: Union[str, Path],
        out_dir: Union[str, Path],
        llm_client: LLMClient,
        *,
        generated_at: Optional[str] = None,
        run_kubeconform: bool = True) -> Dict[str, Any]:
    """Execute the six-task pipeline; returns the report dict (also written
    as <out_dir>/report.json).  Never raises for content problems -- those
    become interceptions inside the report; only true IO/bugs raise."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = time.perf_counter()
    recorder = _TaskRecorder()

    report: Dict[str, Any] = {
        "schema_version": "m2s-k8s-report/1",
        "generated_at": generated_at,
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir),
        "llm": llm_client.describe(),
        "tasks": recorder.steps,
        "interceptions": [],
        "validation": None,
        "expert_baseline": None,
        "signoff": None,
        "artifacts": {},
        "summary": {"pass": False, "status": "pending"},
    }

    manifest, err = _load_manifest_or_intercept(manifest_path, recorder)
    if manifest is None:
        report["interceptions"].append({"where": "manifest_load", "what": err})
        report["summary"] = {"pass": False, "status": "intercepted",
                             "reason": err}
        _finalize_report(report, out_dir, time.perf_counter() - started)
        return report

    state: Dict[str, Any] = {
        "manifest": manifest,
        "manifest_path": Path(manifest_path),
        "blueprint": None,
        "adapters": {},          # rel path -> content
        "files": None,           # rendered {file: yaml}
        "render_summary": None,
        "overlays_applied": 0,
        "validation": None,
    }
    exit_status = "pending"

    for task in TASKS:
        step = recorder.start(task)
        t0 = time.perf_counter()
        input_summary = _task_input_summary(task, state, generated_at)
        try:
            _run_task(task, state, llm_client, generated_at, run_kubeconform)
        except PipelineError as exc:
            step["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            recorder.finish(step, "intercepted", input_summary, "", str(exc))
            report["interceptions"].append({"where": task["id"], "what": str(exc)})
            exit_status = "intercepted"
            break
        except Exception as exc:  # noqa: BLE001 -- record, do not crash the run
            step["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            detail = f"{type(exc).__name__}: {exc}"
            recorder.finish(step, "failed", input_summary, "", detail)
            report["interceptions"].append(
                {"where": task["id"], "what": detail,
                 "traceback": traceback.format_exc(limit=6)})
            exit_status = "failed"
            break
        else:
            step["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            recorder.finish(step, "ok", input_summary,
                            _task_output_summary(task, state))
    else:
        exit_status = "completed"

    skipped = [t["id"] for t in TASKS
               if t["id"] not in {s["task_id"] for s in recorder.steps}]

    # persist artifacts
    try:
        if state["blueprint"] is not None:
            _write_artifact(out_dir, "blueprint.json",
                            json.dumps(state["blueprint"], ensure_ascii=False,
                                       indent=2, sort_keys=True) + "\n")
            report["artifacts"]["blueprint.json"] = "blueprint.json"
        for rel, content in state["adapters"].items():
            _write_artifact(out_dir, rel, content)
        if state["adapters"]:
            report["artifacts"]["adapters"] = sorted(state["adapters"])
        if state["files"]:
            for fname, text in sorted(state["files"].items()):
                _write_artifact(out_dir, f"manifests/{fname}", text)
            report["artifacts"]["manifests"] = sorted(state["files"])
    except OSError as exc:
        report["interceptions"].append({"where": "artifact_write", "what": str(exc)})
        exit_status = "failed"

    report["validation"] = state["validation"]
    report["signoff"] = state.get("signoff")
    report["expert_baseline"] = state.get("expert_check")
    report["tasks"] = recorder.steps
    for s in recorder.steps:
        if s["status"] == "running":
            s["status"] = "skipped"
            skipped.append(s["task_id"])
    if skipped:
        report["skipped_tasks"] = skipped

    # final gate
    validation_ok = bool(state["validation"] and state["validation"].get("ok"))
    verdict = report.get("signoff") or {}
    verdict_pass = verdict.get("verdict") == "PASS"
    adapter_ok = not state.get("adapter_problems")
    summary_pass = exit_status == "completed" and validation_ok and verdict_pass and adapter_ok
    report["summary"] = {
        "pass": summary_pass,
        "status": exit_status,
        "validation_ok": validation_ok,
        "verdict": verdict.get("verdict"),
        "adapter_problems": state.get("adapter_problems") or [],
        "interception_count": len(report["interceptions"]),
        "model_key": manifest.get("model_key"),
        "slices_count": manifest.get("slices_count", len(manifest.get("slices", []))),
    }
    _finalize_report(report, out_dir, time.perf_counter() - started)
    return report


# --------------------------------------------------------------------------
# per-task glue
# --------------------------------------------------------------------------
def _task_input_summary(task, state, generated_at) -> str:
    manifest = state["manifest"]
    kind = task["kind"]
    if task["id"] == "architect_blueprint":
        return f"manifest digest:\n{_manifest_summary(manifest)}"
    if task["id"] == "developer_adapters":
        return (f"manifest digest:\n{_manifest_summary(manifest, full_layers=True)}\n\n"
                f"blueprint:\n{_digest(state['blueprint'])}")
    if task["id"] == "engineer_render":
        return (f"manifest + blueprint -> k8s_render_tool; generated_at pinned {generated_at}")
    if task["id"] == "engineer_overlay":
        return (f"rendered files:\n{_render_files_summary(state['files'])}\n\n"
                f"render summary:\n{_digest(state['render_summary'])}")
    if task["id"] == "validator_checks":
        return (f"final manifests (after {state['overlays_applied']} overlay patch(es)):\n"
                f"{_render_files_summary(state['files'])}")
    return (f"validation:\n{_digest(state['validation'])}\n\n"
            f"adapters:\n{sorted(state['adapters'])}\n\n"
            f"expert check:\n{_digest(state.get('expert_check'), 800)}")


def _task_output_summary(task, state) -> str:
    if task["id"] == "architect_blueprint":
        return f"blueprint.json validated; namespace={state['blueprint']['namespace']} " \
               f"image={state['blueprint']['image']}"
    if task["id"] == "developer_adapters":
        return f"{len(state['adapters'])} adapter(s): {sorted(state['adapters'])}"
    if task["id"] == "engineer_render":
        rs = state["render_summary"]
        return (f"{rs['file_count']} files, {rs['total_bytes']} bytes, "
                f"per-slice numbers extracted from rendered YAML")
    if task["id"] == "engineer_overlay":
        return f"{state['overlays_applied']} overlay patch(es) applied"
    if task["id"] == "validator_checks":
        v = state["validation"]
        return (f"errors={v['error_count']} warnings={v['warning_count']} "
                f"kubeconform={v['kubeconform']['status']} ok={v['ok']}")
    return f"verdict={state.get('signoff', {}).get('verdict')}"


def _build_messages(task, state, generated_at) -> List[Dict[str, str]]:
    role = ROLES[task["role"]]
    ctx = {
        "manifest_summary": _manifest_summary(state["manifest"],
                                              full_layers=(task["id"] == "developer_adapters")),
        "blueprint": _digest(state["blueprint"]),
        "render_summary": _digest(state["render_summary"]),
        "render_files": _render_files_summary(state["files"] or {}),
        "adapter_files": _digest(sorted(state["adapters"]), 400),
        "validation": _digest(state["validation"]),
        "expert_check": _digest(state.get("expert_check"), 800),
    }
    user_content = task["instruction"].format(**ctx)
    user_content += f"\nRUN PARAMS: generated_at={generated_at}\n"
    return [
        {"role": "system", "content": role["system_prompt"]},
        {"role": "user", "content": f"{task_marker(task, turn=1)}\n\n{user_content}"},
    ]


def _run_task(task, state, llm_client, generated_at, run_kubeconform) -> None:
    messages = _build_messages(task, state, generated_at)
    response = llm_client.chat(messages)

    if task["id"] == "architect_blueprint":
        blocks = parse_file_blocks(response)
        bp_blocks = [(p, c) for p, c in blocks if Path(p).name == "blueprint.json"]
        if len(bp_blocks) != 1:
            raise PipelineError(
                f"Architect must emit exactly one 'file: blueprint.json' block, "
                f"got {len(bp_blocks)} in {[p for p, _ in blocks]}")
        try:
            blueprint = json.loads(bp_blocks[0][1])
        except json.JSONDecodeError as exc:
            raise PipelineError(f"blueprint.json is not valid JSON: {exc}") from exc
        validate_blueprint(blueprint)  # raises BlueprintError (a ValueError) -> interception
        if blueprint["model_key"] != state["manifest"].get("model_key"):
            raise PipelineError(
                f"blueprint.model_key {blueprint['model_key']!r} != manifest "
                f"model_key {state['manifest'].get('model_key')!r}")
        state["blueprint"] = blueprint
        return

    if task["id"] == "developer_adapters":
        blocks = parse_file_blocks(response)
        if not blocks:
            raise PipelineError("Developer emitted no 'file:' blocks")
        expected = {s.get("slice_id") for s in state["manifest"]["slices"]}
        seen_ids = set()
        for rel, content in blocks:
            if not rel.startswith("adapters/") or not rel.endswith(".py"):
                raise PipelineError(f"Developer file outside adapters/*.py: {rel!r}")
            m = SPEC_SLICE_ID_RE.search(content)
            if not m:
                raise PipelineError(f"adapter {rel}: SPEC.slice_id not found")
            sid = int(m.group(1))
            if sid not in expected:
                raise PipelineError(f"adapter {rel}: SPEC.slice_id {sid} not in manifest")
            if sid in seen_ids:
                raise PipelineError(f"adapter {rel}: duplicate slice_id {sid}")
            seen_ids.add(sid)
            state["adapters"][rel] = content
        missing = expected - seen_ids
        if missing:
            raise PipelineError(f"Developer missed adapter(s) for slice(s) {sorted(missing)}")
        state["adapter_problems"] = check_adapters(state["adapters"], state["manifest"])
        if state["adapter_problems"]:
            raise PipelineError("adapter consistency failed: "
                                + "; ".join(state["adapter_problems"]))
        return

    if task["id"] == "engineer_render":
        call = parse_json_object(response)
        if call.get("tool") != "k8s_render_tool":
            raise PipelineError(
                f"engineer_render must call k8s_render_tool, got {call.get('tool')!r}")
        args = call.get("args") or {}
        params = dict(args.get("params") or {})
        params.setdefault("generated_at", generated_at)
        # resolve refs -- the LLM never passes inline resource numbers
        manifest_arg = args.get("manifest_ref", args.get("manifest"))
        blueprint_arg = args.get("blueprint_ref", args.get("blueprint"))
        manifest = state["manifest"] if manifest_arg in (None, "state.manifest") \
            else load_manifest(manifest_arg)
        blueprint = state["blueprint"] if blueprint_arg in (None, "state.blueprint.json") \
            else json.loads(Path(blueprint_arg).read_text(encoding="utf-8"))
        try:
            result = K8sRenderTool().run(manifest, blueprint, params)
        except ValueError as exc:  # render gate / blueprint validation -> interception
            raise PipelineError(str(exc)) from exc
        state["files"] = result["files"]
        state["render_summary"] = result["summary"]
        expected_bp = derive_expected(state["manifest"], state["blueprint"])
        state["expert_check"] = {
            "mismatches": compare_expected_vs_render(expected_bp, result["summary"]),
            "expected_inventory": {
                "deployments": expected_bp["inventory"]["deployments"],
                "services": expected_bp["inventory"]["services"],
                "ingress_count_expected": expected_bp["inventory"]["ingress_count_expected"],
            },
        }
        return

    if task["id"] == "engineer_overlay":
        payload = parse_json_object(response)
        overlays = payload.get("overlays")
        if overlays is None:
            raise PipelineError('overlay response must be {"overlays": [...]}')
        applied = 0
        for ov in overlays:
            applied += apply_overlay(
                state["files"],
                ov.get("file"),
                ov.get("patch") or {},
                ov.get("target_kind"),
            )
        state["overlays_applied"] = applied
        return

    if task["id"] == "validator_checks":
        call = parse_json_object(response)
        if call.get("tool") != "kube_validate_tool":
            raise PipelineError(
                f"validator_checks must call kube_validate_tool, got {call.get('tool')!r}")
        state["validation"] = KubeValidateTool().run(
            files=state["files"],
            manifest=state["manifest"],
            blueprint=state["blueprint"],
            run_kubeconform=run_kubeconform,
        )
        return

    if task["id"] == "validator_signoff":
        verdict = parse_json_object(response)
        verdict["adapter_problems"] = state.get("adapter_problems") or []
        verdict["expert_baseline_mismatches"] = (state.get("expert_check") or {}).get("mismatches")
        verdict["validation_ok"] = bool(state["validation"] and state["validation"]["ok"])
        for key in ("verdict", "d4", "d5", "consistency"):
            if key not in verdict:
                raise PipelineError(f"sign-off JSON missing key {key!r}")
        state["signoff"] = verdict
        return

    raise PipelineError(f"unhandled task kind: {task['id']}")


def _finalize_report(report: Dict[str, Any], out_dir: Path, elapsed_s: float) -> None:
    report["elapsed_s"] = round(elapsed_s, 3)
    target = out_dir / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _check_llm() -> int:
    """Connectivity probe: env presence + a tokenless GET /models (never a
    chat completion).  Real keys are provisioned by the parent session."""
    status = ArkLLM.env_status()
    print("[check-llm] Ark environment probe (no chat completion is sent)")
    for name, info in status.items():
        if name == "ARK_API_KEY":
            print(f"  {name}: {'SET' if info['set'] else 'NOT SET'} ({info['hint']})")
        elif name == "ARK_BASE_URL":
            print(f"  {name}: {'custom: ' + str(os.getenv('ARK_BASE_URL')) if info['set'] else 'default ' + info['default']}")
        else:
            print(f"  {name}: {'SET' if info['set'] else 'NOT SET'}")
    client = ArkLLM()
    missing = client.missing_config()
    if missing:
        print(f"\n[check-llm] MISSING: {', '.join(missing)}")
        print(CONFIG_GUIDANCE)
        return 1
    url = f"{client.base_url}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {client.api_key}"},
                                 method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"\n[check-llm] GET {url} -> HTTP {resp.status}")
            print(f"[check-llm] body head: {body[:300]}")
            print("[check-llm] endpoint reachable; credentials accepted.")
            return 0
    except urllib.error.HTTPError as exc:
        print(f"\n[check-llm] GET {url} -> HTTP {exc.code} ({exc.reason})")
        if exc.code in (401, 403):
            print("[check-llm] The endpoint answered but REJECTED the key -- "
                  "regenerate the dedicated ARK_API_KEY via arkcli.")
        return 1
    except urllib.error.URLError as exc:
        print(f"\n[check-llm] GET {url} failed: {exc.reason}")
        print("[check-llm] Endpoint unreachable from this host (network/egress).")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m capsule.m2s_k8s.runner",
        description="M2' four-agent K8s migration pipeline (sequential runner)")
    parser.add_argument("--manifest", help="path to slice_manifest.json")
    parser.add_argument("--out", help="output directory (k8s_migrated contract)")
    parser.add_argument("--fake-llm", action="store_true",
                        help="run offline with canned responses (resnet50 golden path)")
    parser.add_argument("--check-llm", action="store_true",
                        help="probe ARK_* configuration and endpoint, then exit")
    parser.add_argument("--generated-at", default=None,
                        help="ISO-8601 UTC stamp pinned into renders/report")
    parser.add_argument("--no-kubeconform", action="store_true",
                        help="skip the level-2 kubeconform schema pass")
    args = parser.parse_args(argv)

    if args.check_llm:
        return _check_llm()

    if not args.manifest or not args.out:
        parser.error("--manifest and --out are required (or use --check-llm)")

    if args.fake_llm:
        from capsule.m2s_k8s.fake_responses import build_fake_responses
        manifest = load_manifest(args.manifest)
        llm: LLMClient = FakeLLM(build_fake_responses(manifest, args.generated_at))
    else:
        llm = ArkLLM()
        missing = llm.missing_config()
        if missing:
            print(f"[runner] Ark not configured: missing {', '.join(missing)}")
            print(CONFIG_GUIDANCE)
            print("\n[runner] For an offline run use --fake-llm.")
            return 2

    report = run(args.manifest, args.out, llm,
                 generated_at=args.generated_at,
                 run_kubeconform=not args.no_kubeconform)
    summary = report["summary"]
    print(f"[runner] status={summary['status']} pass={summary['pass']} "
          f"out={args.out} elapsed={report['elapsed_s']}s")
    for step in report["tasks"]:
        print(f"[runner]   {step['task_id']:<20} {step['role']:<12} "
              f"{step['status']:<11} {step['duration_ms']}ms")
    for it in report["interceptions"]:
        print(f"[runner] INTERCEPTED at {it['where']}: {it['what']}")
    if report.get("validation"):
        v = report["validation"]
        print(f"[runner] validation: errors={v['error_count']} warnings={v['warning_count']} "
              f"kubeconform={v['kubeconform']['status']}")
    if summary["pass"]:
        return 0
    return 1 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
