"""K0 golden-file acceptance tests for capsule.k8s_render (k8s-prep batch K0).

The golden file tests/golden/resnet50_k3.yaml is the byte-level lock of the
full render of tests/fixtures/resnet50_manifest.json under PARAMS below.  The
first generation was human-reviewed against
slice-manifest-to-k8s-mapping-draft.md before locking:
    limits.memory = ceil((alloc_mb + 600) * 1.3) Mi   (128 -> 947Mi)
    requests.cpu  = ceil(alloc_mb / 1769) m           (128 -> 1m)
    requests.memory = ceil(footprint_mb) Mi           (55.63 -> 56Mi)
"""

import json
import unittest
from pathlib import Path

import yaml

from capsule.k8s_render import RenderParams, render

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "resnet50_manifest.json"
GOLDEN = REPO_ROOT / "tests" / "golden" / "resnet50_k3.yaml"

PARAMS = RenderParams(
    image="harbor.example/mopar/slice-runner:0.1",
    generated_at="2026-11-02T00:00:00Z",
    namespace="mopar-dev",
    diff_image="harbor.example/mopar/difftest:0.1",
)


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def concat(files):
    return "".join(f"# file: {name}\n{text}" for name, text in files.items())


def docs_of(text):
    return list(yaml.safe_load_all(text))


def deployment_of(files, fname):
    for doc in docs_of(files[fname]):
        if doc["kind"] == "Deployment":
            return doc
    raise AssertionError(f"no Deployment in {fname}")


def env_of(files, fname):
    container = deployment_of(files, fname)["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e["value"] for e in container["env"]}


def resources_of(files, fname):
    return deployment_of(files, fname)["spec"]["template"]["spec"]["containers"][0]["resources"]


class K8sRenderGoldenTest(unittest.TestCase):
    def test_golden_file_byte_identical(self):
        files = render(load_fixture(), PARAMS)
        self.assertEqual(concat(files), GOLDEN.read_text(encoding="utf-8"))

    def test_deterministic_two_runs(self):
        first = render(load_fixture(), PARAMS)
        second = render(load_fixture(), PARAMS)
        self.assertEqual(list(first.keys()), list(second.keys()))
        for name in first:
            self.assertEqual(first[name], second[name], f"file {name} differs between runs")

    def test_resource_formula(self):
        manifest = {
            "model_key": "alpha",
            "slices_count": 2,
            "slices": [
                {"slice_id": 0, "name": "alpha_slice_0", "layers": ["conv1"], "alloc_mb": 128,
                 "footprint_mb": 100, "inbound_peers": [], "outbound_peers": [1],
                 "trigger_type": "api_gateway", "path": "/predict", "type": "orchestrator"},
                {"slice_id": 1, "name": "alpha_slice_1", "layers": ["fc"], "alloc_mb": 128,
                 "footprint_mb": 100.4, "inbound_peers": [0], "outbound_peers": [],
                 "trigger_type": "event", "type": "worker"},
            ],
        }
        files = render(manifest, RenderParams(image="img:1", generated_at="T"))
        res0 = resources_of(files, "30-alpha-slice-0.yaml")
        self.assertEqual(res0["limits"]["memory"], "947Mi")  # ceil((128+600)*1.3) = ceil(946.4)
        self.assertEqual(res0["requests"]["cpu"], "1m")      # ceil(128/1769) = 1
        self.assertEqual(res0["requests"]["memory"], "100Mi")
        res1 = resources_of(files, "30-alpha-slice-1.yaml")
        self.assertEqual(res1["requests"]["memory"], "101Mi")  # ceil(100.4)

    def test_gate_rejects_fanout_and_bad_alloc(self):
        with self.subTest("two outbound edges"):
            bad = load_fixture()
            bad["slices"][0]["outbound_peers"] = [1, 2]
            with self.assertRaises(ValueError):
                render(bad, PARAMS)
        with self.subTest("two inbound edges"):
            bad = load_fixture()
            bad["slices"][2]["inbound_peers"] = [0, 1]
            with self.assertRaises(ValueError):
                render(bad, PARAMS)
        with self.subTest("alloc_mb = 0"):
            bad = load_fixture()
            bad["slices"][1]["alloc_mb"] = 0
            with self.assertRaises(ValueError):
                render(bad, PARAMS)
        with self.subTest("empty layers"):
            bad = load_fixture()
            bad["slices"][0]["layers"] = []
            with self.assertRaises(ValueError):
                render(bad, PARAMS)

    def test_topology_ingress_only_on_slice0(self):
        manifest = load_fixture()
        files = render(manifest, PARAMS)
        self.assertEqual(len(manifest["slices"]), 3)
        # exactly one Ingress, bound to the slice-0 Service
        self.assertEqual([k for k in files if k.startswith("40-ingress")], ["40-ingress.yaml"])
        ingress = docs_of(files["40-ingress.yaml"])[0]
        self.assertEqual(ingress["kind"], "Ingress")
        backend = ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]
        self.assertEqual(backend["name"], "resnet50-slice-0")
        self.assertEqual(backend["port"]["number"], 8080)
        # every slice has its own Deployment+Service file
        slice_files = sorted(k for k in files if k.startswith("30-"))
        self.assertEqual(
            slice_files,
            ["30-resnet50-slice-0.yaml", "30-resnet50-slice-1.yaml", "30-resnet50-slice-2.yaml"],
        )
        self.assertEqual(env_of(files, "30-resnet50-slice-0.yaml")["MOPAR_OUT_PEERS"],
                         "resnet50-slice-1.mopar-dev.svc:8080")
        self.assertEqual(env_of(files, "30-resnet50-slice-0.yaml")["MOPAR_TRANSPORT"], "dual")
        self.assertEqual(env_of(files, "30-resnet50-slice-2.yaml")["MOPAR_OUT_PEERS"], "")

    def test_plan_source_marker(self):
        files = render(load_fixture(), PARAMS)
        meta = docs_of(files["20-plan-meta-configmap.yaml"])[0]["data"]
        self.assertEqual(meta["plan_source"], "dp_solve")
        qwen = {
            "model_key": "qwen3_1_7b",
            "slices_count": 2,
            "stats": {},
            "slices": [
                {"slice_id": 0, "name": "qwen3_1_7b_slice_0", "layers": ["embed"],
                 "alloc_mb": 3072, "footprint_mb": 2250, "inbound_peers": [],
                 "outbound_peers": [1], "trigger_type": "api_gateway", "path": "/chat"},
                {"slice_id": 1, "name": "qwen3_1_7b_slice_1", "layers": ["lm_head"],
                 "alloc_mb": 2048, "footprint_mb": 1440, "inbound_peers": [0],
                 "outbound_peers": [], "trigger_type": "event"},
            ],
        }
        files = render(qwen, RenderParams(image="img:1", generated_at="T"))
        meta = docs_of(files["20-plan-meta-configmap.yaml"])[0]["data"]
        self.assertEqual(meta["plan_source"], "hardcoded")
        self.assertEqual(meta["gpu"], "reserved")


if __name__ == "__main__":
    unittest.main()
