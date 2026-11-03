"""M2'-1 acceptance: tool layer (k8s_render_tool + kube_validate_tool).

Positive: resnet50 fixture rendered through the tool validates clean against
the manifest (D4/D5) and structurally.  Negative: deliberately sabotaged
documents must produce the SPECIFIC finding codes, proving the validator is
a real gate and not a rubber stamp.
"""

import json
import sys
import unittest
from pathlib import Path

import yaml

from capsule.agent_tools.k8s_render_tool import (
    BlueprintError, K8sRenderTool, load_manifest, validate_blueprint,
)
from capsule.agent_tools.kube_validate_tool import KubeValidateTool
from capsule.m2s_k8s.runner import OverlayError, apply_overlay

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "resnet50_manifest.json"

BLUEPRINT = {
    "model_key": "resnet50",
    "namespace": "mopar-dev",
    "image": "harbor.example/mopar/slice-runner:0.1",
    "topology": {"mode": "linear-chain", "entry_slice": 0},
    "transport": {"mode": "dual", "inline_max_mb": 2,
                  "activations_pvc": "mopar-activations", "activations_size": "20Gi"},
    "storage": {"weights_pvc": "mopar-weights", "weights_size": "20Gi"},
    "quota": {"lambda_mb_per_vcpu": 1769, "torch_base_mi": 600, "headroom": 1.3},
    "gpu": {"required": False, "resource_name": "nvidia.com/v100"},
    "difftest": {"samples": 100, "tolerance": "1e-4",
                 "image": "harbor.example/mopar/difftest:0.1"},
}
PARAMS = {"generated_at": "2026-11-03T00:00:00Z"}


def render_fixture():
    return K8sRenderTool().run(load_manifest(FIXTURE), BLUEPRINT, PARAMS)


def codes(findings):
    return {f["code"] for f in findings}


class TestRenderTool(unittest.TestCase):
    def test_renders_expected_inventory(self):
        out = render_fixture()
        self.assertEqual(out["summary"]["file_count"], 8)
        self.assertIn("40-ingress.yaml", out["files"])
        self.assertIn("50-difftest-job.yaml", out["files"])
        self.assertEqual(len(out["summary"]["per_slice"]), 3)
        # numbers extracted from the rendered YAML, not recomputed
        slice0 = out["summary"]["per_slice"][0]
        self.assertEqual(slice0["limits_memory"], "947Mi")
        self.assertEqual(slice0["requests_cpu"], 1)
        self.assertEqual(slice0["requests_memory"], "56Mi")

    def test_deterministic_with_pinned_generated_at(self):
        a = render_fixture()["files"]
        b = render_fixture()["files"]
        self.assertEqual(a, b)

    def test_blueprint_missing_field_is_a_clear_error(self):
        bad = {k: v for k, v in BLUEPRINT.items() if k != "quota"}
        with self.assertRaises(BlueprintError) as ctx:
            validate_blueprint(bad)
        self.assertIn("blueprint.quota", str(ctx.exception))

    def test_blueprint_bad_quota_value_is_a_clear_error(self):
        bad = json.loads(json.dumps(BLUEPRINT))
        bad["quota"]["lambda_mb_per_vcpu"] = 0
        with self.assertRaises(BlueprintError) as ctx:
            validate_blueprint(bad)
        self.assertIn("lambda_mb_per_vcpu", str(ctx.exception))

    def test_blueprint_mode_mismatch_is_a_clear_error(self):
        bad = json.loads(json.dumps(BLUEPRINT))
        bad["topology"]["mode"] = "mesh"
        with self.assertRaises(BlueprintError) as ctx:
            validate_blueprint(bad)
        self.assertIn("linear-chain", str(ctx.exception))

    def test_model_key_mismatch_rejected(self):
        bad_bp = json.loads(json.dumps(BLUEPRINT))
        bad_bp["model_key"] = "vgg16"
        with self.assertRaises(BlueprintError) as ctx:
            K8sRenderTool().run(load_manifest(FIXTURE), bad_bp, PARAMS)
        self.assertIn("model_key", str(ctx.exception))

    def test_render_gate_intercepts_fanout_manifest(self):
        manifest = load_manifest(FIXTURE)
        manifest["slices"][0]["outbound_peers"] = [1, 2]  # broken chain
        with self.assertRaises(ValueError) as ctx:
            K8sRenderTool().run(manifest, BLUEPRINT, PARAMS)
        self.assertIn("linear-chain", str(ctx.exception))


class TestValidateCleanRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rendered = render_fixture()
        cls.result = KubeValidateTool().run(
            files=cls.rendered["files"], manifest=load_manifest(FIXTURE),
            blueprint=BLUEPRINT, run_kubeconform=True)

    def test_clean_render_has_zero_errors(self):
        self.assertTrue(self.result["ok"],
                        msg=f"unexpected findings: {self.result['findings']}")
        self.assertEqual(self.result["error_count"], 0)
        self.assertEqual(self.result["checked"]["docs"], 11)

    def test_kubeconform_best_effort_never_crashes(self):
        self.assertIn(self.result["kubeconform"]["status"],
                      ("pass", "skipped", "error"))

    def test_checked_kinds_histogram(self):
        kinds = self.result["checked"]["kinds"]
        self.assertEqual(kinds.get("Deployment"), 3)
        self.assertEqual(kinds.get("Service"), 3)
        self.assertEqual(kinds.get("Ingress"), 1)
        self.assertEqual(kinds.get("PersistentVolumeClaim"), 2)
        self.assertEqual(kinds.get("ConfigMap"), 1)
        self.assertEqual(kinds.get("Job"), 1)


class TestSabotage(unittest.TestCase):
    """Negative controls: each sabotage -> one specific D4/D5 finding."""

    def validate(self, files, manifest=None):
        return KubeValidateTool().run(
            files=files, manifest=manifest if manifest is not None else load_manifest(FIXTURE),
            blueprint=BLUEPRINT, run_kubeconform=False)

    def _sabotaged(self, mutate_deployment):
        rendered = render_fixture()["files"]
        for fname in list(rendered):
            docs = [d for d in yaml.safe_load_all(rendered[fname]) if d]
            hit = False
            for d in docs:
                if d.get("kind") == "Deployment":
                    mutate_deployment(d)
                    hit = True
            if hit:
                rendered[fname] = yaml.safe_dump_all(
                    docs, sort_keys=False, default_flow_style=False,
                    width=120, explicit_start=True)
                break
        return rendered

    def test_wrong_limits_memory_flagged_with_expected_value(self):
        def bump(d):
            d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] = "512Mi"
        result = self.validate(self._sabotaged(bump))
        res = [f for f in result["findings"] if f["code"] == "V-RES-001"]
        self.assertEqual(len(res), 1)
        self.assertIn("947Mi", res[0]["message"])
        self.assertIn("tolerance", res[0]["message"])
        self.assertFalse(result["ok"])

    def test_wrong_requests_cpu_flagged(self):
        def bump(d):
            d["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"] = "2"
        result = self.validate(self._sabotaged(bump))
        self.assertIn("V-RES-002", codes(result["findings"]))
        self.assertIn("1769", next(f for f in result["findings"]
                                   if f["code"] == "V-RES-002")["message"])

    def test_wrong_requests_memory_flagged(self):
        def bump(d):
            d["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["memory"] = "2Gi"
        result = self.validate(self._sabotaged(bump))
        self.assertIn("V-RES-003", codes(result["findings"]))

    def test_ingress_on_worker_flagged(self):
        rendered = render_fixture()["files"]
        worker_svc = "resnet50-slice-1"
        rendered["41-ingress-worker.yaml"] = yaml.safe_dump_all([{
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {"name": "resnet50-worker-ingress", "namespace": "mopar-dev",
                         "labels": {"mopar.io/model": "resnet50"}},
            "spec": {
                "rules": [
                    {
                        "http": {
                            "paths": [
                                {
                                    "path": "/worker",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": worker_svc,
                                            "port": {"number": 8080},
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        }], sort_keys=False, explicit_start=True)
        result = self.validate(rendered)
        self.assertIn("V-ING-010", codes(result["findings"]))
        self.assertIn("V-ING-011", codes(result["findings"]))
        msg = next(f for f in result["findings"] if f["code"] == "V-ING-011")["message"]
        self.assertIn("slice-0", msg)

    def test_bad_dns_name_flagged(self):
        def rename(d):
            d["metadata"]["name"] = "Resnet50_Slice0"
        result = self.validate(self._sabotaged(rename))
        self.assertIn("V-NAME-001", codes(result["findings"]))
        self.assertIn("DNS-1123", next(f for f in result["findings"]
                                       if f["code"] == "V-NAME-001")["message"])

    def test_broken_yaml_flagged(self):
        result = self.validate({"broken.yaml": "deployment: [unclosed"})
        self.assertIn("V-SYN-001", codes(result["findings"]))
        self.assertIn("YAML parse error",
                      next(f for f in result["findings"]
                           if f["code"] == "V-SYN-001")["message"])

    def test_missing_shm_volume_flagged(self):
        def drop(d):
            spec = d["spec"]["template"]["spec"]
            spec["volumes"] = [v for v in spec["volumes"] if v["name"] != "shm"]
        result = self.validate(self._sabotaged(drop))
        self.assertIn("V-RES-005", codes(result["findings"]))

    def test_wrong_out_peers_env_flagged(self):
        def bump(d):
            for env in d["spec"]["template"]["spec"]["containers"][0]["env"]:
                if env["name"] == "MOPAR_OUT_PEERS":
                    env["value"] = "wrong-svc.mopar-dev.svc:9999"
        result = self.validate(self._sabotaged(bump))
        self.assertIn("V-ENV-002", codes(result["findings"]))
        self.assertIn("resnet50-slice-1.mopar-dev.svc:8080",
                      next(f for f in result["findings"]
                           if f["code"] == "V-ENV-002")["message"])

    def test_gpu_resource_flagged_on_cpu_render(self):
        def add(d):
            d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = 1
        result = self.validate(self._sabotaged(add))
        self.assertIn("V-RES-004", codes(result["findings"]))

    def test_dangling_volume_mount_flagged(self):
        def bump(d):
            d["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append(
                {"name": "ghost", "mountPath": "/ghost"})
        result = self.validate(self._sabotaged(bump))
        self.assertIn("V-MNT-004", codes(result["findings"]))

    def test_service_selector_without_deployment_flagged(self):
        rendered = render_fixture()["files"]
        rendered["35-orphan-svc.yaml"] = yaml.safe_dump_all([{
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": "orphan", "namespace": "mopar-dev"},
            "spec": {"selector": {"mopar.io/model": "resnet50", "mopar.io/slice-id": "9"},
                     "ports": [{"port": 8080}]},
        }], explicit_start=True)
        result = self.validate(rendered)
        self.assertIn("V-SVC-003", codes(result["findings"]))

    def test_manifest_without_ingress_flagged(self):
        manifest = load_manifest(FIXTURE)
        rendered = render_fixture()["files"]
        rendered.pop("40-ingress.yaml")
        result = self.validate(rendered, manifest)
        self.assertIn("V-ING-012", codes(result["findings"]))

    def test_structural_only_check_without_manifest(self):
        """Without the manifest only syntax/structure checks run."""
        result = self.validate(render_fixture()["files"], manifest=None)
        self.assertTrue(result["ok"])


class TestOverlayWhitelist(unittest.TestCase):
    def setUp(self):
        self.files = render_fixture()["files"]
        self.target = "30-resnet50-slice-0.yaml"

    def test_legal_overlay_applies(self):
        n = apply_overlay(self.files, self.target, {
            "annotations": {"mopar.io/pipeline": "m2s"},
            "append_env": [{"name": "EXTRA", "value": "1"}],
        }, target_kind="Deployment")
        self.assertEqual(n, 1)
        doc = next(d for d in yaml.safe_load_all(self.files[self.target])
                   if d and d.get("kind") == "Deployment")
        self.assertEqual(doc["metadata"]["annotations"]["mopar.io/pipeline"], "m2s")
        envs = {e["name"] for e in doc["spec"]["template"]["spec"]["containers"][0]["env"]}
        self.assertIn("EXTRA", envs)

    def test_resources_patch_rejected(self):
        with self.assertRaises(OverlayError) as ctx:
            apply_overlay(self.files, self.target,
                          {"resources": {"limits": {"memory": "16Gi"}}},
                          target_kind="Deployment")
        self.assertIn("whitelist", str(ctx.exception))

    def test_mopar_env_rejected(self):
        with self.assertRaises(OverlayError) as ctx:
            apply_overlay(self.files, self.target,
                          {"append_env": [{"name": "MOPAR_TRANSPORT", "value": "inline"}]},
                          target_kind="Deployment")
        self.assertIn("MOPAR_", str(ctx.exception))

    def test_reserved_label_rejected(self):
        with self.assertRaises(OverlayError) as ctx:
            apply_overlay(self.files, self.target,
                          {"labels": {"mopar.io/alloc-mb": "99999"}},
                          target_kind="Deployment")
        self.assertIn("reserved", str(ctx.exception))

    def test_unknown_file_rejected(self):
        with self.assertRaises(OverlayError):
            apply_overlay(self.files, "99-nope.yaml", {"annotations": {}})


if __name__ == "__main__":
    unittest.main()
