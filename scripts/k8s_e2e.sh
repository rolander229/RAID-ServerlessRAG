#!/usr/bin/env bash
# MOPAR K8s 一键端到端（E1/E2/E4）
#
#   切分(slicer) → 渲染(k8s_render) → 清理残留 → 部署 → 等待就绪 → 集群内探针 → 采集与成绩单
#
# 用法:
#   scripts/k8s_e2e.sh <model> <slices> <lambda_mb_per_vcpu> [options]
#
# 必填环境:
#   --image <harbor>/<repo>:<tag>   slice-runner 镜像（或设置 MOPAR_IMAGE）
#
# 选项:
#   --namespace NS            目标命名空间（默认 mopar-<model>）
#   --node-selector k=v       钉节点（如 kubernetes.io/hostname=dell-52）
#   --mem-limit-mi N          内存限额覆盖（默认按公式；C2 后建议 1024）
#   --requests-cpu X          调度用 CPU 申报（默认 100m）
#   --weights-host-path P     权重 hostPath（默认 /var/mopar/weights）
#   --skip-deploy             只做切分+渲染（本地验证，不碰集群）
#   --keep                    保留 tester Pod（默认清理）
#   --out DIR                 产物目录（默认 build/e2e/<model>）
#
# 产物:
#   <out>/manifest.json · <out>/manifests/*.yaml · <out>/a4_logs/*.jsonl · <out>/summary.md
set -euo pipefail

MODEL=${1:?用法: k8s_e2e.sh <model> <slices> <lambda> [options]}
SLICES=${2:?缺少 <slices>}
LAMBDA=${3:?缺少 <lambda>}
shift 3

IMAGE="${MOPAR_IMAGE:-}"
NAMESPACE="mopar-${MODEL}"
NODE_SELECTOR=""
MEM_LIMIT_MI=""
REQUESTS_CPU="100m"
WEIGHTS_HOST_PATH="/var/mopar/weights"
SKIP_DEPLOY=0
KEEP=0
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)            IMAGE="$2"; shift 2 ;;
    --namespace)        NAMESPACE="$2"; shift 2 ;;
    --node-selector)    NODE_SELECTOR="$2"; shift 2 ;;
    --mem-limit-mi)     MEM_LIMIT_MI="$2"; shift 2 ;;
    --requests-cpu)     REQUESTS_CPU="$2"; shift 2 ;;
    --weights-host-path) WEIGHTS_HOST_PATH="$2"; shift 2 ;;
    --skip-deploy)      SKIP_DEPLOY=1; shift ;;
    --keep)             KEEP=1; shift ;;
    --out)              OUT="$2"; shift 2 ;;
    -h|--help)          sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
OUT="${OUT:-${ROOT}/build/e2e/${MODEL}}"
MAN_DIR="${OUT}/manifests"
LOG_DIR="${OUT}/a4_logs"
mkdir -p "$MAN_DIR" "$LOG_DIR"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

step() { echo; echo "=== [$1/7] $2 ==="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
  [[ -n "$IMAGE" ]] || die "缺少 --image（或 MOPAR_IMAGE）"
  command -v kubectl >/dev/null || die "kubectl 不可用"
  echo "[ctx] $(kubectl config current-context 2>/dev/null || echo '(no context)')"
fi

# ---------------------------------------------------------------- [1/7] 切分
step 1 "切分 ${MODEL}（K=${SLICES}, λ=${LAMBDA}）"
"$PY" - "$MODEL" "$SLICES" "$LAMBDA" "${OUT}/manifest.json" <<'PC'
import json, sys
from capsule.bridge import SliceCapsuleBridge
model, slices, lam, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
m = SliceCapsuleBridge().partition_model(model, slices_count=slices, mb_per_vcpu=lam)
with open(out, "w") as fh:
    json.dump(m, fh, indent=2)
print("slices:", m["slices_count"], "| alloc:", [s["alloc_mb"] for s in m["slices"]],
      "| max_diff:", m["max_tensor_diff"])
PC

# ---------------------------------------------------------------- [2/7] 渲染
step 2 "渲染 K8s manifests（meta-load / hostPath / 无激活 PVC）"
RENDER=(--manifest "${OUT}/manifest.json" --out "$MAN_DIR" --namespace "$NAMESPACE"
        --image "${IMAGE:-placeholder:local}" --requests-cpu "$REQUESTS_CPU"
        --weights-mode hostPath --weights-host-path "$WEIGHTS_HOST_PATH"
        --activations-mode omit --meta-load --generated-at "$STAMP")
[[ -n "$NODE_SELECTOR" ]] && RENDER+=(--node-selector "$NODE_SELECTOR")
[[ -n "$MEM_LIMIT_MI" ]] && RENDER+=(--mem-limit-mi "$MEM_LIMIT_MI")
"$PY" -m capsule.k8s_render "${RENDER[@]}" | tail -1

if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
  echo; echo "（--skip-deploy）切分与渲染完成，产物: $OUT"; exit 0
fi

# ------------------------------------------------------- [3/7] 清理（幂等 E4）
step 3 "清理旧代 Deployment/RS/Pod（label mopar.io/model=${MODEL}）"
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"
kubectl delete deploy,rs,pod -n "$NAMESPACE" -l "mopar.io/model=${MODEL}" --ignore-not-found --wait=true | tail -1

# ---------------------------------------------------------------- [4/7] 部署
step 4 "部署（跳过占位差分 Job）"
for f in "${MAN_DIR}"/*.yaml; do
  [[ "$f" == *difftest* ]] && continue
  kubectl apply -n "$NAMESPACE" -f "$f" >/dev/null
done
kubectl get deploy -n "$NAMESPACE" -l "mopar.io/model=${MODEL}" --no-headers | awk '{print "  deploy:", $1, $2}'

# ---------------------------------------------------------------- [5/7] 等待
step 5 "等待四片就绪（rollout 300s/片）"
for n in $(seq 0 $((SLICES - 1))); do
  kubectl rollout status "deploy/${MODEL}-slice-${n}" -n "$NAMESPACE" --timeout=300s | tail -1
done
kubectl get pods -n "$NAMESPACE" -l "mopar.io/model=${MODEL}" --no-headers | awk '{print "  pod:", $1, $2, $3}'

# ---------------------------------------------------------------- [6/7] 探针
step 6 "集群内差分探针（链 vs 单体）"
TESTER="mopar-e2e-tester"
kubectl delete pod "$TESTER" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
kubectl run "$TESTER" -n "$NAMESPACE" --image="$IMAGE" --restart=Never --overrides="{
  \"spec\": {\"nodeName\": \"${NODE_SELECTOR##*=}\",
    \"containers\": [{\"name\": \"t\", \"image\": \"$IMAGE\",
      \"command\": [\"sleep\", \"1800\"],
      \"env\": [{\"name\": \"TORCH_HOME\", \"value\": \"/models\"}],
      \"resources\": {\"limits\": {\"memory\": \"3Gi\"}},
      \"volumeMounts\": [{\"name\": \"w\", \"mountPath\": \"/models\"},
                        {\"name\": \"p\", \"mountPath\": \"/app/plan\"}]}],
    \"volumes\": [{\"name\": \"w\", \"hostPath\": {\"path\": \"${WEIGHTS_HOST_PATH}\", \"type\": \"DirectoryOrCreate\"}},
                 {\"name\": \"p\", \"configMap\": {\"name\": \"mopar-plan\"}}]}}" >/dev/null
kubectl wait --for=condition=Ready "pod/$TESTER" -n "$NAMESPACE" --timeout=180s | tail -1
set +e
PROBE_OUT=$(kubectl exec "$TESTER" -n "$NAMESPACE" -- \
  python -m capsule.slice_server --model "$MODEL" \
  --probe "http://${MODEL}-slice-0.${NAMESPACE}.svc:8080/invoke" \
  --plan "/app/plan/${MODEL}_manifest.json" --weights-dir "/models/${MODEL}" --device cpu 2>&1 | tail -3)
PROBE_RC=$?
set -e
echo "$PROBE_OUT"
[[ "$PROBE_RC" -eq 0 ]] && echo "$PROBE_OUT" | grep -q "PASS" || die "探针未通过（见上）"

# ---------------------------------------------------------------- [7/7] 采集
step 7 "逐跳日志采集 + 成绩单"
for n in $(seq 0 $((SLICES - 1))); do
  POD=$(kubectl get pods -n "$NAMESPACE" -l "mopar.io/model=${MODEL},mopar.io/slice-id=${n}" \
        -o jsonpath='{.items[0].metadata.name}')
  kubectl exec "$POD" -n "$NAMESPACE" -- sh -c 'cat /tmp/logs/*.jsonl 2>/dev/null' \
    > "${LOG_DIR}/${MODEL}_slice${n}.jsonl" || true
done
[[ "$KEEP" -eq 1 ]] || kubectl delete pod "$TESTER" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true

"$PY" - "$OUT" "$MODEL" "$NAMESPACE" "$STAMP" <<'PC'
import glob, json, sys
from pathlib import Path
out, model, ns, stamp = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
manifest = json.loads((out / "manifest.json").read_text())
hops = {}
for f in sorted(glob.glob(str(out / "a4_logs" / "*.jsonl"))):
    for line in Path(f).read_text().splitlines():
        try: e = json.loads(line)
        except Exception: continue
        if e.get("status") != "ok": continue
        hid = e.get("slice_id"); hops.setdefault(hid, []).append(e)
def avg(vals): return round(sum(vals) / len(vals), 1) if vals else 0.0
rows = []
for h in sorted(hops):
    es = hops[h]
    rows.append("| %d | %.1f | %.1f | %.3f |" % (h, avg([e["hop_ms"] for e in es]),
                                                  avg([e["busy_ms"] for e in es]),
                                                  avg([e["moved_mb"] for e in es])))
doc = ["# K8s 端到端成绩单 — %s" % model, "",
       "- 命名空间: %s · 生成: %s" % (ns, stamp),
       "- 切分: alloc=%s MB" % [s["alloc_mb"] for s in manifest["slices"]],
       "- 探针: PASS（链 vs 单体 max-diff ≤ 1e-4）", "",
       "| 跳 | hop_ms | busy_ms | moved_mb |", "|---|---|---|---|"] + rows
(out / "summary.md").write_text("\n".join(doc) + "\n")
print("成绩单:", out / "summary.md")
PC

echo
echo "✅ E2E 完成: $OUT"
