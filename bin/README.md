# bin/ — vendored binaries (research use)

## kubeconform v0.8.0 (linux-amd64)

Level-2 schema validator for `capsule/agent_tools/kube_validate_tool.py`.

- Source: https://github.com/yannh/kubeconform/releases/download/v0.8.0/kubeconform-linux-amd64.tar.gz
- sha256 (official CHECKSUMS): `9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883`
  (see `kubeconform.sha256`)
- dell-65 TLS caveat: the host's egress path intercepts TLS with a private CA
  (curl exit 60), so the tarball was fetched with `-k` and verified against the
  release's CHECKSUMS file fetched through the same channel. Integrity of the
  transfer is verified; trust anchors through a MITM proxy are a documented
  residual risk for a research-only validation binary.

## kubeconform-schemas/

Vendored offline schemas (kubernetes-json-schema `master-standalone-strict`,
fetched 2026-11-03) so level-2 validation works on hosts without GitHub
egress: `persistentvolumeclaim-v1`, `configmap-v1`, `deployment-apps-v1`,
`service-v1`, `ingress-networking-v1`, `job-batch-v1`. Note the naming rule:
group suffix strips `.k8s.io` (ingress networking.k8s.io/v1 ->
`ingress-networking-v1.json`), and the local `-schema-location` template is
`{{.ResourceKind}}{{.KindSuffix}}.json` (NOT `.NormalizedType`, which local
templates do not support).

The tool passes `-schema-location bin/kubeconform-schemas/{{.ResourceKind}}{{.KindSuffix}}.json`
when this directory contains schemas and falls back to the default remote
location otherwise; when schema downloads fail it reports
`kubeconform.status = "error"` + warning finding `V-KC-002` (infrastructure
problem, not a document defect) and level-1 checks remain the gate.
