"""kube_validate_tool: two-level validator for rendered K8s YAML (D4/D5 gate).

Level 1 (self-written, always runs -- this is the D4/D5 checker, i.e. the
scorer's rule set per metrics-review-k8s.md "验收器 = 评分器"):

  * syntax:      PyYAML stream parse, kind/apiVersion/metadata sanity,
                 DNS-1123 names, label-value grammar;
  * structure:   Deployment/Service/Ingress/PVC/ConfigMap/Job required fields,
                 selector<->pod-label matching, cross-doc service references,
                 probes, volume mounts;
  * D4 (needs the slice manifest): limits.memory ~= ceil((alloc+torch_base)*
                 headroom) +/- 16Mi, requests.cpu == ceil(alloc/1769) m,
                 requests.memory ~= ceil(footprint) +/- 16Mi, GPU resource
                 name legality (v0: none expected), shm sizeLimit, required
                 mount points, audit labels;
  * D5 (needs the manifest): one Deployment+Service per slice, exactly one
                 Ingress and only on slice-0, MOPAR_OUT_PEERS matches
                 outbound_peers Service DNS, MOPAR_SLICE_ID matches.

Level 2 (best effort): if a kubeconform binary exists (repo ``bin/`` first,
then $PATH) the files are schema-checked with it.  When the binary is absent
or cannot run (offline / TLS interception), the result records
``status: "skipped"`` with the reason -- level 1 alone is still authoritative.

Output contract (dict)::

    {"ok": bool, "error_count": int, "warning_count": int,
     "findings": [{"code", "severity", "file", "kind", "name", "message"}...],
     "checked": {"files": int, "docs": int, "kinds": {...}},
     "kubeconform": {"status": "pass"|"fail"|"skipped"|"error", "detail": str,
                     "bin": str|null}}
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from capsule.agent_tools.mapping_rules import (
    ANCHOR_NAME, ENV_OUT_PEERS, ENV_SLICE_ID, ENV_TRANSPORT, FILE_INGRESS,
    GPU_RESOURCE_RE, HEALTHZ_PATH, LABEL_ALLOC_MB, LABEL_SLICE_ID, LIMIT_TOLERANCE_MI,
    MOPAR_ENV_PREFIX, REQUEST_MEM_TOLERANCE_MI, REQUIRED_MOUNTS, SERVICE_PORT,
    SHM_SIZE_LIMIT, expected_cpu_requests_m, expected_memory_limit_mi,
    expected_memory_requests_mi, expected_out_peers_env, k8s_name,
    parse_cpu_m, parse_memory_mi, slice_role, valid_dns1123_subdomain,
    valid_label_value,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KUBECONFORM_SEARCH = (REPO_ROOT / "bin" / "kubeconform",)
# vendored offline schemas (kubernetes-json-schema master-standalone-strict,
# fetched 2026-11-03; keeps level-2 usable on hosts without GitHub egress).
# The suffix grammar strips ".k8s.io": ingress networking.k8s.io/v1 ->
# ingress-networking-v1.json.  -strict is satisfied because the vendored
# files ARE the strict variants.
KUBECONFORM_SCHEMA_DIR = REPO_ROOT / "bin" / "kubeconform-schemas"
KUBECONFORM_SCHEMA_TEMPLATE = str(
    KUBECONFORM_SCHEMA_DIR / "{{.ResourceKind}}{{.KindSuffix}}.json"
)


class KubeValidateTool:
    """Callable tool: rendered YAML (+ optional manifest/blueprint) -> findings."""

    name = "kube_validate_tool"
    description = (
        "Validate Kubernetes YAML documents (single file, directory, dict of "
        "texts, or multi-doc stream). Level 1: syntax + structural + D4/D5 "
        "resource/topology rules against the slice manifest. Level 2: "
        "kubeconform schema check when a binary is available. Returns "
        "{'ok', 'findings', 'checked', 'kubeconform'}."
    )

    def run(self,
            files: Optional[Dict[str, str]] = None,
            yaml_dir: Union[str, Path, None] = None,
            text: Optional[str] = None,
            manifest: Union[str, Path, Dict[str, Any], None] = None,
            blueprint: Optional[Dict[str, Any]] = None,
            run_kubeconform: bool = True,
            kubeconform_bin: Optional[str] = None) -> Dict[str, Any]:
        docs: List[Dict[str, Any]] = []          # {"file", "index", "doc"}
        findings: List[Dict[str, Any]] = []
        checked: Dict[str, Any] = {"files": 0, "docs": 0, "kinds": {}}

        sources: Dict[str, str] = {}
        if files:
            sources.update({str(k): v for k, v in files.items()})
        if yaml_dir is not None:
            ydir = Path(yaml_dir)
            if not ydir.is_dir():
                findings.append(_finding("V-IN-001", "error", str(ydir),
                                         "input", "yaml_dir is not a directory"))
            else:
                for p in sorted(ydir.glob("*.yaml")):
                    sources[p.name] = p.read_text(encoding="utf-8")
                for p in sorted(ydir.glob("*.yml")):
                    sources.setdefault(p.name, p.read_text(encoding="utf-8"))
        if text is not None:
            sources["<stream>"] = text
        if not sources:
            findings.append(_finding("V-IN-002", "error", None, "input",
                                     "no input given: pass files=, yaml_dir= or text="))

        manifest_dict = None
        if manifest is not None:
            manifest_dict = _load_manifest(manifest, findings)
        namespace = (blueprint or {}).get("namespace")

        # ---------------- level 1a: syntax ----------------
        for fname in sorted(sources):
            checked["files"] += 1
            try:
                stream = [d for d in yaml.safe_load_all(sources[fname]) if d is not None]
            except yaml.YAMLError as exc:
                findings.append(_finding("V-SYN-001", "error", fname, "stream",
                                         f"YAML parse error: {exc}"))
                continue
            for idx, doc in enumerate(stream):
                checked["docs"] += 1
                entry = {"file": fname, "index": idx, "doc": doc}
                docs.append(entry)
                _check_doc_basics(entry, findings)

        kinds = {}
        for e in docs:
            kinds[e["doc"].get("kind", "?")] = kinds.get(e["doc"].get("kind", "?"), 0) + 1
        checked["kinds"] = kinds

        # ---------------- level 1b: structure + cross-doc ----------------
        services = {d["doc"].get("metadata", {}).get("name"): d
                    for d in docs if d["doc"].get("kind") == "Service"}
        deployments = [d for d in docs if d["doc"].get("kind") == "Deployment"]
        for e in docs:
            kind = e["doc"].get("kind")
            if kind == "Deployment":
                _check_deployment(e, findings)
            elif kind == "Service":
                _check_service(e, deployments, findings)
            elif kind == "Ingress":
                _check_ingress(e, services, findings)
            elif kind == "PersistentVolumeClaim":
                _check_pvc(e, findings)
            elif kind == "ConfigMap":
                _check_configmap(e, findings)
            elif kind == "Job":
                _check_job(e, findings)

        # ---------------- level 1c: D4/D5 against the manifest ----------------
        if manifest_dict is not None:
            _check_d4_d5(manifest_dict, docs, findings,
                         namespace=namespace,
                         quota=(blueprint or {}).get("quota") or {})

        # ---------------- level 2: kubeconform (best effort) ----------------
        kc = {"status": "skipped", "detail": "not requested", "bin": None}
        if run_kubeconform:
            kc = _run_kubeconform(sources, kubeconform_bin)
            if kc["status"] == "fail":
                findings.append(_finding("V-KC-001", "error", None, "schema",
                                         f"kubeconform schema errors: {kc['detail']}"))
            elif kc["status"] == "error":
                # infrastructure problem (offline / schema fetch blocked), NOT
                # a document defect: warning only, level-1 stays authoritative
                findings.append(_finding("V-KC-002", "warning", None, "schema",
                                         f"kubeconform could not complete: {kc['detail']}"))
        ok = not any(f["severity"] == "error" for f in findings)
        return {
            "ok": ok,
            "error_count": sum(1 for f in findings if f["severity"] == "error"),
            "warning_count": sum(1 for f in findings if f["severity"] == "warning"),
            "findings": findings,
            "checked": checked,
            "kubeconform": kc,
            "tool": "kube_validate_tool",
            "anchor": ANCHOR_NAME,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _finding(code: str, severity: str, file, kind_or_ctx, message: str,
             name=None) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "file": file,
            "context": kind_or_ctx, "name": name, "message": message}


def _load_manifest(manifest, findings):
    if isinstance(manifest, dict):
        return manifest
    p = Path(manifest)
    if not p.exists():
        findings.append(_finding("V-IN-003", "error", str(p), "manifest",
                                 "manifest file not found"))
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append(_finding("V-IN-004", "error", str(p), "manifest",
                                 f"manifest is not valid JSON: {exc}"))
        return None


def _meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = doc.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _check_doc_basics(entry, findings):
    doc = entry["doc"]
    fname, idx = entry["file"], entry["index"]
    where = f"{fname}[{idx}]"
    if not isinstance(doc, dict):
        findings.append(_finding("V-SYN-002", "error", fname, where,
                                 f"document {idx} is not a mapping"))
        return
    kind = doc.get("kind")
    if not kind or not isinstance(kind, str):
        findings.append(_finding("V-SYN-003", "error", fname, where,
                                 "missing string field 'kind'"))
        return
    api = doc.get("apiVersion")
    if not api or not isinstance(api, str):
        findings.append(_finding("V-SYN-004", "error", fname, where,
                                 f"{kind}: missing string field 'apiVersion'"))
        return
    meta = _meta(doc)
    name = meta.get("name")
    if not name or not isinstance(name, str):
        findings.append(_finding("V-SYN-005", "error", fname, where,
                                 f"{kind}: missing metadata.name"))
        return
    if not valid_dns1123_subdomain(name):
        findings.append(_finding("V-NAME-001", "error", fname, where,
                                 f"{kind}/{name}: metadata.name is not a valid "
                                 "DNS-1123 subdomain (lowercase alnum, '-', '.')",
                                 name=name))
    for key, value in (meta.get("labels") or {}).items():
        if not valid_label_value(str(value)):
            findings.append(_finding("V-NAME-002", "error", fname, where,
                                     f"{kind}/{name}: label {key}={value!r} is not a "
                                     "valid label value (<=63 chars, alnum - _ .)",
                                     name=name))


def _containers(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return spec.get("template", {}).get("spec", {}).get("containers") or []


def _check_deployment(entry, findings):
    doc, fname = entry["doc"], entry["file"]
    name = _meta(doc).get("name")
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        findings.append(_finding("V-DEP-001", "error", fname, "Deployment",
                                 f"{name}: missing spec"))
        return
    if not isinstance(spec.get("replicas"), int) or spec["replicas"] < 1:
        findings.append(_finding("V-DEP-002", "error", fname, "Deployment",
                                 f"{name}: spec.replicas must be an int >= 1"))
    selector = spec.get("selector", {})
    match = selector.get("matchLabels") if isinstance(selector, dict) else None
    pod_labels = spec.get("template", {}).get("metadata", {}).get("labels")
    if not isinstance(match, dict) or not match:
        findings.append(_finding("V-DEP-003", "error", fname, "Deployment",
                                 f"{name}: spec.selector.matchLabels missing/empty"))
    if not isinstance(pod_labels, dict) or not pod_labels:
        findings.append(_finding("V-DEP-004", "error", fname, "Deployment",
                                 f"{name}: pod template labels missing/empty"))
    elif isinstance(match, dict) and not set(match.items()) <= set(pod_labels.items()):
        findings.append(_finding("V-DEP-005", "error", fname, "Deployment",
                                 f"{name}: selector.matchLabels {match} is not a subset "
                                 f"of pod template labels {pod_labels}"))
    containers = _containers(spec)
    if not containers:
        findings.append(_finding("V-DEP-006", "error", fname, "Deployment",
                                 f"{name}: no containers in pod template"))
        return
    for c in containers:
        cname = c.get("name", "?")
        if not c.get("image"):
            findings.append(_finding("V-DEP-007", "error", fname, "Deployment",
                                     f"{name}/{cname}: container image missing"))
        if not isinstance(c.get("resources"), dict):
            findings.append(_finding("V-DEP-008", "error", fname, "Deployment",
                                     f"{name}/{cname}: resources block missing"))
        for probe in ("readinessProbe", "livenessProbe"):
            if probe not in c:
                findings.append(_finding("V-DEP-009", "warning", fname, "Deployment",
                                         f"{name}/{cname}: {probe} missing "
                                         f"(/healthz:{SERVICE_PORT} is the renderer contract)"))


def _check_service(entry, deployments, findings):
    doc, fname = entry["doc"], entry["file"]
    name = _meta(doc).get("name")
    spec = doc.get("spec", {})
    selector = spec.get("selector")
    if not isinstance(selector, dict) or not selector:
        findings.append(_finding("V-SVC-001", "error", fname, "Service",
                                 f"{name}: spec.selector missing/empty"))
        return
    ports = spec.get("ports") or []
    if not ports or not isinstance(ports[0].get("port"), int):
        findings.append(_finding("V-SVC-002", "error", fname, "Service",
                                 f"{name}: spec.ports[0].port missing"))
    matched = []
    for dep in deployments:
        pod_labels = dep["doc"].get("spec", {}).get("template", {}).get("metadata", {}).get("labels") or {}
        if set(selector.items()) <= set(pod_labels.items()):
            matched.append(dep)
    if not matched:
        findings.append(_finding("V-SVC-003", "error", fname, "Service",
                                 f"{name}: selector {selector} matches no Deployment pod template"))
    elif len(matched) > 1:
        names = sorted(_meta(d["doc"]).get("name", "?") for d in matched)
        findings.append(_finding("V-SVC-004", "error", fname, "Service",
                                 f"{name}: selector {selector} matches multiple Deployments {names}"))
    else:
        dep = matched[0]
        dep_name = _meta(dep["doc"]).get("name")
        cports = {p.get("containerPort")
                  for c in _containers(dep["doc"].get("spec", {}))
                  for p in (c.get("ports") or [])}
        for p in ports:
            if p.get("targetPort") is not None and p.get("targetPort") not in cports:
                findings.append(_finding("V-SVC-005", "error", fname, "Service",
                                         f"{name}: targetPort {p.get('targetPort')} not exposed by "
                                         f"Deployment {dep_name} (containerPorts={sorted(cports)})"))
            if p.get("port") is not None and p.get("port") not in cports and p.get("targetPort") is None:
                findings.append(_finding("V-SVC-006", "error", fname, "Service",
                                         f"{name}: port {p.get('port')} does not match any containerPort "
                                         f"of Deployment {dep_name}"))


def _check_ingress(entry, services, findings):
    doc, fname = entry["doc"], entry["file"]
    name = _meta(doc).get("name")
    spec = doc.get("spec", {})
    rules = spec.get("rules") or []
    if not rules:
        findings.append(_finding("V-ING-001", "error", fname, "Ingress",
                                 f"{name}: spec.rules empty"))
        return
    for r in rules:
        for path in ((r.get("http") or {}).get("paths") or []):
            backend = path.get("backend", {}).get("service", {})
            svc_name = backend.get("name")
            if not svc_name:
                findings.append(_finding("V-ING-002", "error", fname, "Ingress",
                                         f"{name}: path {path.get('path')!r} backend.service.name missing"))
            elif svc_name not in services:
                findings.append(_finding("V-ING-003", "error", fname, "Ingress",
                                         f"{name}: backend service {svc_name!r} does not exist"))
            else:
                svc_ports = {p.get("port") for p in services[svc_name]["doc"].get("spec", {}).get("ports", [])}
                if backend.get("port", {}).get("number") not in svc_ports:
                    findings.append(_finding("V-ING-004", "error", fname, "Ingress",
                                             f"{name}: backend port {backend.get('port')} not served by {svc_name}"))


def _check_pvc(entry, findings):
    doc, fname = entry["doc"], entry["file"]
    name = _meta(doc).get("name")
    spec = doc.get("spec", {})
    if not spec.get("accessModes"):
        findings.append(_finding("V-PVC-001", "error", fname, "PVC", f"{name}: accessModes missing"))
    storage = spec.get("resources", {}).get("requests", {}).get("storage")
    if not storage:
        findings.append(_finding("V-PVC-002", "error", fname, "PVC",
                                 f"{name}: resources.requests.storage missing"))
    else:
        try:
            parse_memory_mi(storage)
        except ValueError as exc:
            findings.append(_finding("V-PVC-003", "error", fname, "PVC",
                                     f"{name}: bad storage quantity: {exc}"))


def _check_configmap(entry, findings):
    doc, fname = entry["doc"], entry["file"]
    if not isinstance(doc.get("data"), dict):
        findings.append(_finding("V-CM-001", "error", fname, "ConfigMap",
                                 f"{_meta(doc).get('name')}: data must be a mapping of strings"))


def _check_job(entry, findings):
    doc, fname = entry["doc"], entry["file"]
    name = _meta(doc).get("name")
    spec = doc.get("spec", {})
    tpl = spec.get("template", {}).get("spec", {})
    if tpl.get("restartPolicy") != "Never":
        findings.append(_finding("V-JOB-001", "error", fname, "Job",
                                 f"{name}: template.spec.restartPolicy must be 'Never'"))
    if "backoffLimit" not in spec:
        findings.append(_finding("V-JOB-002", "warning", fname, "Job",
                                 f"{name}: backoffLimit not set (renderer pins 0)"))
    for c in tpl.get("containers") or []:
        if not c.get("image"):
            findings.append(_finding("V-JOB-003", "error", fname, "Job",
                                     f"{name}/{c.get('name', '?')}: container image missing"))


# --------------------------------------------------------------------------
# D4 / D5 against the manifest
# --------------------------------------------------------------------------
def _check_d4_d5(manifest: Dict[str, Any], docs, findings, namespace, quota):
    slices = manifest.get("slices") or []
    if not slices:
        findings.append(_finding("V-MF-001", "error", None, "manifest",
                                 "manifest.slices is empty -- nothing to check"))
        return
    torch_base = int(quota.get("torch_base_mi", 600))
    headroom = float(quota.get("headroom", 1.3))
    lam = int(quota.get("lambda_mb_per_vcpu", 1769))

    name_by_id = {s.get("slice_id"): k8s_name(s.get("name") or
                  f"{manifest.get('model_key', 'model')}_slice_{s.get('slice_id')}")
                  for s in slices}
    dep_by_id, svc_by_id = {}, {}
    ingresses = []
    for e in docs:
        doc, fname = e["doc"], e["file"]
        kind = doc.get("kind")
        sid = _meta(doc).get("labels", {}).get(LABEL_SLICE_ID)
        if kind == "Deployment":
            dep_by_id.setdefault(sid, []).append((e, doc))
        elif kind == "Service":
            svc_by_id.setdefault(sid, []).append((e, doc))
        elif kind == "Ingress":
            ingresses.append((e, doc))

    # -- D5: exactly one Deployment + one Service per slice -------------------
    for s in slices:
        sid = s.get("slice_id")
        sid_key = str(sid)
        role = slice_role(s)
        deps = dep_by_id.get(sid_key, [])
        if not deps:
            findings.append(_finding("V-TOP-001", "error", None, "D5",
                                     f"slice {sid}: no Deployment found"))
        elif len(deps) > 1:
            findings.append(_finding("V-TOP-002", "error", None, "D5",
                                     f"slice {sid}: {len(deps)} Deployments, expected exactly 1"))
        svcs = svc_by_id.get(sid_key, [])
        if not svcs:
            findings.append(_finding("V-TOP-003", "error", None, "D5",
                                     f"slice {sid}: no Service found"))
        elif len(svcs) > 1:
            findings.append(_finding("V-TOP-004", "error", None, "D5",
                                     f"slice {sid}: {len(svcs)} Services, expected exactly 1"))
        for e, doc in deps:
            _check_d4(doc, s, fname=e["file"], torch_base=torch_base,
                      headroom=headroom, lam=lam, findings=findings)
        for e, doc in svcs:
            _labels_audit(doc, s, e["file"], findings)
        # D5: env contract on the (single) deployment
        if deps:
            entry0, doc0 = deps[0]
            _check_env_contract(doc0, s, name_by_id,
                                namespace or _meta(doc0).get("namespace") or "default",
                                e_file=entry0["file"], findings=findings)

    extra_dep_sids = sorted(set(dep_by_id) - {str(s.get("slice_id")) for s in slices},
                            key=lambda x: (x is None, x))
    if extra_dep_sids:
        findings.append(_finding("V-TOP-005", "error", None, "D5",
                                 f"Deployments reference slice-ids not in the manifest: {extra_dep_sids}"))

    # -- D5: exactly one Ingress, only on slice-0 -----------------------------
    entry_slice = slices[0]
    entry_name = name_by_id.get(entry_slice.get("slice_id"))
    if len(ingresses) > 1:
        files = sorted(e["file"] for e, _ in ingresses)
        findings.append(_finding("V-ING-010", "error", None, "D5",
                                 f"{len(ingresses)} Ingress documents found ({files}); "
                                 "linear chain exposes exactly one entry"))
    for e, doc in ingresses:
        backend_names = []
        for r in doc.get("spec", {}).get("rules") or []:
            for path in (r.get("http") or {}).get("paths") or []:
                backend_names.append(path.get("backend", {}).get("service", {}).get("name"))
        for bn in backend_names:
            if bn != entry_name:
                findings.append(_finding("V-ING-011", "error", e["file"], "D5",
                                         f"Ingress backend {bn!r} must be the entry slice "
                                         f"service {entry_name!r} (slice-0 only)"))
    if not ingresses and entry_slice.get("trigger_type") == "api_gateway":
        findings.append(_finding("V-ING-012", "error", None, "D5",
                                 f"manifest slice-0 trigger_type=api_gateway but no Ingress "
                                 f"rendered (expected {FILE_INGRESS})"))


def _check_d4(doc: Dict[str, Any], slice: Dict[str, Any], fname, torch_base,
              headroom, lam, findings):
    name = _meta(doc).get("name")
    sid = slice.get("slice_id")
    alloc = float(slice["alloc_mb"])
    footprint = float(slice["footprint_mb"])
    containers = _containers(doc.get("spec", {}))
    if not containers:
        return  # V-DEP-006 already reported
    c = containers[0]
    res = c.get("resources") or {}
    req = res.get("requests") or {}
    lim = res.get("limits") or {}

    # D4: limits.memory ~= ceil((alloc + torch floor) * headroom) +/- 16Mi
    expected_lim = expected_memory_limit_mi(alloc, torch_base, headroom)
    lim_mem = lim.get("memory")
    if lim_mem is None:
        findings.append(_finding("V-RES-001", "error", fname, "D4",
                                 f"slice {sid} ({name}): limits.memory missing; expected "
                                 f"~{expected_lim}Mi (alloc {alloc:g} + floor {torch_base}Mi "
                                 f"x headroom {headroom})"))
    else:
        try:
            got = parse_memory_mi(lim_mem)
            if abs(got - expected_lim) > LIMIT_TOLERANCE_MI:
                findings.append(_finding("V-RES-001", "error", fname, "D4",
                                         f"slice {sid} ({name}): limits.memory {lim_mem} = {got}Mi, "
                                         f"expected ~{expected_lim}Mi = (alloc {alloc:g}Mi + floor "
                                         f"{torch_base}Mi) x headroom {headroom} (tolerance "
                                         f"+/-{LIMIT_TOLERANCE_MI}Mi)"))
        except ValueError as exc:
            findings.append(_finding("V-RES-001", "error", fname, "D4",
                                     f"slice {sid} ({name}): unparseable limits.memory {lim_mem!r}: {exc}"))

    # D4: requests.cpu == ceil(alloc / lambda) m (exact)
    expected_cpu = expected_cpu_requests_m(alloc, lam)
    cpu = req.get("cpu")
    if cpu is None:
        findings.append(_finding("V-RES-002", "error", fname, "D4",
                                 f"slice {sid} ({name}): requests.cpu missing; expected "
                                 f"{expected_cpu}m = ceil({alloc:g}/{lam}) ({lam} MB/vCPU anchor)"))
    else:
        try:
            got = parse_cpu_m(cpu)
            if got != expected_cpu:
                findings.append(_finding("V-RES-002", "error", fname, "D4",
                                         f"slice {sid} ({name}): requests.cpu {cpu} = {got}m, expected "
                                         f"{expected_cpu}m = ceil(alloc {alloc:g} / {lam} MB-per-vCPU)"))
        except ValueError as exc:
            findings.append(_finding("V-RES-002", "error", fname, "D4",
                                     f"slice {sid} ({name}): unparseable requests.cpu {cpu!r}: {exc}"))

    # D4: requests.memory ~= ceil(footprint) +/- 16Mi
    expected_req = expected_memory_requests_mi(footprint)
    req_mem = req.get("memory")
    if req_mem is None:
        findings.append(_finding("V-RES-003", "warning", fname, "D4",
                                 f"slice {sid} ({name}): requests.memory missing; expected "
                                 f"~{expected_req}Mi (footprint {footprint:g})"))
    else:
        try:
            got = parse_memory_mi(req_mem)
            if abs(got - expected_req) > REQUEST_MEM_TOLERANCE_MI:
                findings.append(_finding("V-RES-003", "error", fname, "D4",
                                         f"slice {sid} ({name}): requests.memory {req_mem} = {got}Mi, "
                                         f"expected ~{expected_req}Mi (footprint {footprint:g}Mi, "
                                         f"tolerance +/-{REQUEST_MEM_TOLERANCE_MI}Mi)"))
        except ValueError as exc:
            findings.append(_finding("V-RES-003", "error", fname, "D4",
                                     f"slice {sid} ({name}): unparseable requests.memory {req_mem!r}: {exc}"))

    # D4: GPU resources -- v0 renders CPU-only
    gpu_keys = sorted(k for k in list(lim) + list(req) if k.startswith("nvidia.com/"))
    if gpu_keys:
        bad = [k for k in gpu_keys if not GPU_RESOURCE_RE.match(k)]
        if bad:
            findings.append(_finding("V-RES-004", "error", fname, "D4",
                                     f"slice {sid} ({name}): illegal GPU resource name(s) {bad} "
                                     "(expected nvidia.com/<model> e.g. nvidia.com/v100)"))
        else:
            findings.append(_finding("V-RES-004", "warning", fname, "D4",
                                     f"slice {sid} ({name}): GPU resource(s) {gpu_keys} present; the K0 "
                                     "v0 renderer is CPU-only -- verify this is intentional"))

    # D4: shm emptyDir (Memory) sizeLimit
    volumes = doc.get("spec", {}).get("template", {}).get("spec", {}).get("volumes") or []
    vol_by_name = {v.get("name"): v for v in volumes}
    shm = vol_by_name.get("shm")
    if shm is None:
        findings.append(_finding("V-RES-005", "error", fname, "D4",
                                 f"slice {sid} ({name}): shm volume missing (/dev/shm channel buffer)"))
    else:
        empty = shm.get("emptyDir") or {}
        if empty.get("medium") != "Memory":
            findings.append(_finding("V-RES-005", "error", fname, "D4",
                                     f"slice {sid} ({name}): shm emptyDir.medium must be 'Memory'"))
        if empty.get("sizeLimit") != SHM_SIZE_LIMIT:
            findings.append(_finding("V-RES-005", "warning", fname, "D4",
                                     f"slice {sid} ({name}): shm sizeLimit {empty.get('sizeLimit')!r} "
                                     f"!= renderer constant {SHM_SIZE_LIMIT}"))

    # D4: required mount points + dangling volume references
    mounts = {m.get("mountPath"): m for m in (c.get("volumeMounts") or [])}
    for path, want in REQUIRED_MOUNTS.items():
        m = mounts.get(path)
        if m is None:
            findings.append(_finding("V-MNT-001", "error", fname, "D4",
                                     f"slice {sid} ({name}): mount {path} missing "
                                     f"(volume {want['name']}, readOnly={want['readOnly']})"))
        else:
            if want["readOnly"] and not m.get("readOnly"):
                findings.append(_finding("V-MNT-002", "error", fname, "D4",
                                         f"slice {sid} ({name}): mount {path} must be readOnly"))
            if m.get("name") != want["name"]:
                findings.append(_finding("V-MNT-003", "warning", fname, "D4",
                                         f"slice {sid} ({name}): mount {path} uses volume "
                                         f"{m.get('name')!r}, renderer constant is {want['name']!r}"))
    for m in c.get("volumeMounts") or []:
        if m.get("name") not in vol_by_name:
            findings.append(_finding("V-MNT-004", "error", fname, "D4",
                                     f"slice {sid} ({name}): volumeMount {m.get('name')!r} "
                                     "references a non-existent volume"))


def _labels_audit(doc: Dict[str, Any], slice: Dict[str, Any], fname, findings):
    labels = _meta(doc).get("labels") or {}
    sid = slice.get("slice_id")
    if labels.get(LABEL_SLICE_ID) != str(sid):
        findings.append(_finding("V-LBL-001", "error", fname, "D5",
                                 f"slice {sid}: label {LABEL_SLICE_ID}="
                                 f"{labels.get(LABEL_SLICE_ID)!r} != manifest slice_id {sid!r}"))


def _check_env_contract(doc: Dict[str, Any], slice: Dict[str, Any], name_by_id,
                        namespace, e_file, findings):
    sid = slice.get("slice_id")
    c = (_containers(doc.get("spec", {})) or [{}])[0]
    env = {e.get("name"): e.get("value") for e in c.get("env") or []}
    if env.get(ENV_SLICE_ID) != str(sid):
        findings.append(_finding("V-ENV-001", "error", e_file, "D5",
                                 f"slice {sid}: env {ENV_SLICE_ID}={env.get(ENV_SLICE_ID)!r} "
                                 f"!= manifest slice_id {sid!r}"))
    expected_peers = expected_out_peers_env(slice, name_by_id, namespace)
    if env.get(ENV_OUT_PEERS) != expected_peers:
        findings.append(_finding("V-ENV-002", "error", e_file, "D5",
                                 f"slice {sid}: env {ENV_OUT_PEERS}={env.get(ENV_OUT_PEERS)!r} "
                                 f"!= expected {expected_peers!r} (outbound_peers "
                                 f"{slice.get('outbound_peers')}, Service DNS with namespace "
                                 f"{namespace!r})"))
    if ENV_TRANSPORT not in env:
        findings.append(_finding("V-ENV-003", "warning", e_file, "D5",
                                 f"slice {sid}: env {ENV_TRANSPORT} missing (dual-track transport switch)"))


# --------------------------------------------------------------------------
# level 2: kubeconform
# --------------------------------------------------------------------------
def _find_kubeconform(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for candidate in KUBECONFORM_SEARCH:
        if candidate.exists():
            return str(candidate)
    which = shutil.which("kubeconform")
    return which


def _run_kubeconform(sources: Dict[str, str], explicit_bin: Optional[str]) -> Dict[str, Any]:
    bin_path = _find_kubeconform(explicit_bin)
    if not bin_path:
        return {"status": "skipped",
                "detail": "kubeconform binary not found (looked in repo bin/ and $PATH); "
                          "level-1 checks remain authoritative",
                "bin": None}
    with tempfile.TemporaryDirectory(prefix="kubeconform-") as tmp:
        tmpdir = Path(tmp)
        for fname, text in sources.items():
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", fname)
            (tmpdir / safe).write_text(text, encoding="utf-8")
        env = dict(os.environ)
        cmd = [bin_path, "-strict"]
        if KUBECONFORM_SCHEMA_DIR.is_dir() and any(KUBECONFORM_SCHEMA_DIR.glob("*.json")):
            cmd += ["-schema-location", KUBECONFORM_SCHEMA_TEMPLATE]
        cmd += ["-summary", str(tmpdir)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120, env=env, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "detail": "kubeconform timed out after 120s", "bin": bin_path}
        except OSError as exc:
            return {"status": "error", "detail": f"kubeconform failed to run: {exc}", "bin": bin_path}
        output = (proc.stdout + proc.stderr).strip()
        if "failed downloading schema" in output:
            return {"status": "error",
                    "detail": "schema download blocked (offline host); "
                              "vendor schemas into bin/kubeconform-schemas/ "
                              "or accept level-1 as the sole gate",
                    "bin": bin_path}
        if proc.returncode == 0:
            return {"status": "pass", "detail": output or "all documents valid", "bin": bin_path}
        if proc.returncode == 1:
            return {"status": "fail", "detail": output, "bin": bin_path}
        # 2+: internal error
        return {"status": "error",
                "detail": f"kubeconform exited {proc.returncode}: {output}",
                "bin": bin_path}
