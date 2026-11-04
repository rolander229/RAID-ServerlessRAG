"""M2'-3 acceptance: FakeLLM offline pipeline on resnet50 + interception.

Positive: the full six-task pipeline produces the complete k8s_migrated/
artifact tree, kube_validate passes everything, report.json is complete and
the expert-baseline cross-check agrees.

Negative: broken manifests (invalid JSON / fanout chain / bad alloc) must be
intercepted by the runner or the render tool, recorded in report.json, with
downstream tasks skipped and nothing half-written.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from capsule.m2s_k8s.expert_baseline import compare_expected_vs_render, derive_expected
from capsule.m2s_k8s.fake_responses import build_fake_responses
from capsule.m2s_k8s.llm import FakeLLM, LLMError
from capsule.m2s_k8s.runner import run
from capsule.agent_tools.k8s_render_tool import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "resnet50_manifest.json"
GOLDEN_BASELINE = REPO_ROOT / "tests" / "golden" / "resnet50_k3_expert_baseline.json"
STAMP = "2026-11-03T00:00:00Z"


def fake_run(tmp, manifest_path=FIXTURE, responses=None):
    manifest = load_manifest(manifest_path)
    llm = FakeLLM(responses if responses is not None
                  else build_fake_responses(manifest, STAMP))
    report = run(manifest_path, tmp / "k8s_migrated", llm, generated_at=STAMP,
                 run_kubeconform=True)
    return report


class TestGoldenPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="m2s-golden-")
        cls.tmp = Path(cls._tmp.name)
        cls.report = fake_run(cls.tmp)
        cls.out = cls.tmp / "k8s_migrated"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_pipeline_passes(self):
        self.assertTrue(self.report["summary"]["pass"],
                        msg=json.dumps(self.report["interceptions"], indent=2))
        self.assertEqual(self.report["summary"]["status"], "completed")

    def test_artifact_tree_complete(self):
        self.assertTrue((self.out / "blueprint.json").is_file())
        self.assertTrue((self.out / "report.json").is_file())
        adapters = sorted((self.out / "adapters").glob("*.py"))
        manifests = sorted((self.out / "manifests").glob("*.yaml"))
        self.assertEqual([p.name for p in adapters],
                         ["slice_0_adapter.py", "slice_1_adapter.py", "slice_2_adapter.py"])
        self.assertEqual(len(manifests), 9)
        self.assertIn("40-ingress.yaml", [p.name for p in manifests])

    def test_manifests_are_valid_yaml_and_validate_clean(self):
        from capsule.agent_tools.kube_validate_tool import KubeValidateTool
        files = {p.name: p.read_text(encoding="utf-8")
                 for p in sorted((self.out / "manifests").glob("*.yaml"))}
        result = KubeValidateTool().run(files=files, manifest=load_manifest(FIXTURE),
                                        blueprint=self.report and json.loads(
                                            (self.out / "blueprint.json").read_text()),
                                        run_kubeconform=True)
        self.assertTrue(result["ok"], msg=result["findings"])
        self.assertEqual(result["error_count"], 0)

    def test_report_has_all_six_tasks_with_io_summaries(self):
        ids = [s["task_id"] for s in self.report["tasks"]]
        self.assertEqual(ids, ["architect_blueprint", "developer_adapters",
                               "engineer_render", "engineer_overlay",
                               "validator_checks", "validator_signoff"])
        for step in self.report["tasks"]:
            self.assertEqual(step["status"], "ok", msg=step)
            self.assertTrue(step["input_summary"], msg=step["task_id"])
            self.assertTrue(step["output_summary"], msg=step["task_id"])
            self.assertIsNotNone(step["duration_ms"])

    def test_report_records_validation_and_expert_agreement(self):
        self.assertTrue(self.report["validation"]["ok"])
        self.assertEqual(self.report["validation"]["error_count"], 0)
        self.assertIsNotNone(self.report["expert_baseline"])
        self.assertEqual(self.report["expert_baseline"]["mismatches"], [])
        self.assertEqual(self.report["signoff"]["verdict"], "PASS")
        self.assertEqual(self.report["llm"]["kind"], "FakeLLM")
        self.assertGreater(self.report["elapsed_s"], 0)

    def test_report_records_overlay_and_adapter_provenance(self):
        overlay_steps = [s for s in self.report["tasks"] if s["task_id"] == "engineer_overlay"]
        self.assertIn("1 overlay patch", overlay_steps[0]["output_summary"])
        adapters = (self.out / "adapters" / "slice_1_adapter.py").read_text(encoding="utf-8")
        self.assertIn("SPEC", adapters)
        self.assertIn("slice_id", adapters)

    def test_rerun_is_deterministic_in_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="m2s-rerun-") as other:
            other = Path(other)
            fake_run(other, responses=build_fake_responses(
                load_manifest(FIXTURE), STAMP))
            for rel in ("blueprint.json", "manifests/30-resnet50-slice-0.yaml",
                        "adapters/slice_0_adapter.py"):
                a = (self.out / rel).read_text(encoding="utf-8")
                b = (other / "k8s_migrated" / rel).read_text(encoding="utf-8")
                self.assertEqual(a, b, msg=rel)


class TestExpertBaselineGolden(unittest.TestCase):
    def test_derivation_matches_reviewed_golden(self):
        golden = json.loads(GOLDEN_BASELINE.read_text(encoding="utf-8"))
        expected = derive_expected(load_manifest(FIXTURE), None)
        self.assertEqual(expected, golden["expected"])

    def test_render_agrees_with_independent_baseline(self):
        from capsule.agent_tools.k8s_render_tool import K8sRenderTool
        manifest = load_manifest(FIXTURE)
        rendered = K8sRenderTool().run(
            manifest, json.loads(json.dumps(derive_expected(manifest)["per_slice"] and {
                # blueprint compatible with the baseline's canonical choices
                "model_key": "resnet50", "namespace": "mopar-dev",
                "image": "harbor.example/mopar/slice-runner:0.1",
                "topology": {"mode": "linear-chain", "entry_slice": 0},
                "transport": {"mode": "dual", "inline_max_mb": 2,
                              "activations_pvc": "mopar-activations",
                              "activations_size": "20Gi"},
                "storage": {"weights_pvc": "mopar-weights", "weights_size": "20Gi"},
                "quota": {"lambda_mb_per_vcpu": 1769, "torch_base_mi": 600, "headroom": 1.3},
                "gpu": {"required": False, "resource_name": "nvidia.com/v100"},
                "difftest": {"samples": 100, "tolerance": "1e-4",
                             "image": "harbor.example/mopar/difftest:0.1"},
            })), {"generated_at": STAMP})
        mismatches = compare_expected_vs_render(derive_expected(manifest),
                                                rendered["summary"])
        self.assertEqual(mismatches, [])


class TestInterception(unittest.TestCase):
    def _broken_manifest(self, mutate):
        manifest = load_manifest(FIXTURE)
        mutate(manifest)
        # mkdtemp (not the context manager): the file must outlive this method
        tmp = Path(tempfile.mkdtemp(prefix="m2s-broken-"))
        p = tmp / "broken_manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        return p

    def _run_and_assert_intercepted(self, manifest_path, where, expect_downstream_absent):
        with tempfile.TemporaryDirectory(prefix="m2s-intercept-") as tmp:
            tmp = Path(tmp)
            manifest = load_manifest(manifest_path)
            report = fake_run(tmp, manifest_path=manifest_path,
                              responses=build_fake_responses(manifest, STAMP))
            self.assertFalse(report["summary"]["pass"])
            self.assertEqual(report["summary"]["status"], "intercepted")
            hits = [i for i in report["interceptions"] if i["where"] == where]
            self.assertTrue(hits, msg=report["interceptions"])
            out = tmp / "k8s_migrated"
            self.assertFalse((out / expect_downstream_absent).exists())
            skipped = report.get("skipped_tasks", [])
            self.assertIn("validator_signoff", skipped)
            return report

    def test_invalid_json_manifest_intercepted_at_load(self):
        with tempfile.TemporaryDirectory(prefix="m2s-badjson-") as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            with tempfile.TemporaryDirectory() as out:
                llm = FakeLLM({})
                report = run(p, Path(out) / "k8s_migrated", llm, generated_at=STAMP)
            self.assertEqual(report["summary"]["status"], "intercepted")
            self.assertEqual(report["interceptions"][0]["where"], "manifest_load")
            self.assertFalse(llm.calls)  # no LLM call was wasted

    def test_fanout_manifest_intercepted_at_render(self):
        p = self._broken_manifest(
            lambda m: m["slices"][0].__setitem__("outbound_peers", [1, 2]))
        report = self._run_and_assert_intercepted(p, "engineer_render",
                                                  "manifests/50-difftest-job.yaml")
        self.assertIn("linear-chain", report["interceptions"][0]["what"])

    def test_zero_alloc_manifest_intercepted_at_render(self):
        p = self._broken_manifest(
            lambda m: m["slices"][1].__setitem__("alloc_mb", 0))
        report = self._run_and_assert_intercepted(p, "engineer_render",
                                                  "manifests/50-difftest-job.yaml")
        self.assertIn("alloc_mb", report["interceptions"][0]["what"])

    def test_unknown_peer_manifest_intercepted_at_render(self):
        p = self._broken_manifest(
            lambda m: m["slices"][0].__setitem__("outbound_peers", [7]))
        report = self._run_and_assert_intercepted(p, "engineer_render",
                                                  "manifests/50-difftest-job.yaml")
        self.assertIn("unknown slice_id", report["interceptions"][0]["what"])

    def test_developer_missing_adapter_intercepted(self):
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        # drop the whole slice-2 adapter block -> Developer must be stopped
        responses["developer_adapters"] = re.sub(
            r"```file: adapters/slice_2_adapter\.py\n.*?```", "",
            responses["developer_adapters"], flags=re.DOTALL)
        with tempfile.TemporaryDirectory(prefix="m2s-missad-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertEqual(report["summary"]["status"], "intercepted")
        self.assertEqual(report["interceptions"][-1]["where"], "developer_adapters")
        self.assertIn("missed adapter", report["interceptions"][-1]["what"])

    def test_engineer_hand_written_yaml_rejected(self):
        """The engineer may not bypass the render tool."""
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        responses["engineer_render"] = (
            '```json\n{"tool": "yaml_emit", "args": {}}\n```')
        with tempfile.TemporaryDirectory(prefix="m2s-rogue-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertEqual(report["interceptions"][-1]["where"], "engineer_render")
        self.assertIn("k8s_render_tool", report["interceptions"][-1]["what"])

    def test_validator_must_use_its_tool(self):
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        responses["validator_checks"] = (
            '```json\n{"tool": "eyeball", "args": {}}\n```')
        with tempfile.TemporaryDirectory(prefix="m2s-badval-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertEqual(report["interceptions"][-1]["where"], "validator_checks")

    def test_fake_llm_without_canned_response_is_recorded(self):
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        del responses["engineer_overlay"]
        with tempfile.TemporaryDirectory(prefix="m2s-nocanned-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertEqual(report["summary"]["status"], "failed")
        self.assertIn("FakeLLM has no canned response",
                      report["interceptions"][-1]["what"])


class TestServerSideArgInjection(unittest.TestCase):
    """RQ2 data point (2026-11-03 real-LLM first run): deepseek-flash
    hallucinated {"manifest_ref": "manifest.json"} -> FileNotFoundError.
    Regression: the runner must inject canonical state regardless of what the
    LLM passes, record the override, and still complete the pipeline."""

    def test_hallucinated_manifest_path_is_overridden(self):
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        responses["engineer_render"] = (
            '```json\n{"tool": "k8s_render_tool", "args": '
            '{"manifest_ref": "manifest.json", '
            '"params": {"generated_at": "1999-01-01T00:00:00Z"}}}\n```')
        with tempfile.TemporaryDirectory(prefix="m2s-halluc-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertTrue(report["summary"]["pass"], msg=report["interceptions"])
        overrides = report["llm_arg_overrides"]
        self.assertEqual([o["arg"] for o in overrides],
                         ["manifest_ref", "params"])
        self.assertEqual(overrides[0]["provided"], "manifest.json")
        self.assertIn("overridden", overrides[0]["action"])
        # determinism: the runner's stamp won, not the hallucinated one
        self.assertEqual(report["generated_at"], STAMP)

    def test_clean_tool_call_records_no_overrides(self):
        manifest = load_manifest(FIXTURE)
        responses = build_fake_responses(manifest, STAMP)
        with tempfile.TemporaryDirectory(prefix="m2s-cleanargs-") as tmp:
            report = fake_run(Path(tmp), responses=responses)
        self.assertTrue(report["summary"]["pass"])
        self.assertEqual(report["llm_arg_overrides"], [])


if __name__ == "__main__":
    unittest.main()
