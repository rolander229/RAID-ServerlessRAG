# K8S 端到端部署修复清单（vgg16 实战复盘，2026-11-04）

> 背景：在 dell-52 上用 vgg16 四切片跑通部署端到端（D1 max-diff=0.0）。过程中手工补了 20+ 处，
> 本清单把这些补丁固化下来，目标是 **模型 → 切分 → 迁移 → 渲染 → 构建 → 部署 → 验证 全自动**。
>
> 图例：✅ 已修复并提交 ｜ ⚠️ 手工权宜（未固化） ｜ ❌ 未修（必须修）

## A. 镜像构建层

| # | 问题 | 现象/证据 | 修复方案 | 状态 |
|---|------|-----------|----------|------|
| A1 | docker.io 基座镜像 TLS 劫持 | `x509: certificate signed by unknown authority` | Dockerfile 用 `docker.m.daocloud.io/library/python:3.10-slim` 作为默认基座（或提供 `--build-arg BASE_IMAGE=`） | ⚠️ 已写在 Dockerfile，但未作为默认 |
| A2 | pytorch CPU index 元数据名称不一致 | jinja2/typing-extensions 被丢弃 → flit_core 缺失 | 用 wheelhouse 离线安装（见 A4/A5） | ✅ 当前方案 |
| A3 | torch(CPU)+torchvision(CUDA) ABI 不匹配 | `operator torchvision::nms does not exist` | torch 与 torchvision 必须同源（都来自 pytorch cpu index） | ✅ 当前 wheelhouse 已满足 |
| A4 | 校园网出网抖动 | 在线 pip 构建随机失败 | **离线 wheelhouse**（`scripts/download_wheels.sh` 在 dell-65 或 Mac 用阿里云/pytorch 源拉取） | ⚠️ 手工制作，未提交 |
| A5 | wheelhouse 缺钉死依赖 | torch 要求 `sympy==1.13.1` | 下载脚本固定包含该版本 | ⚠️ 手工补，脚本待写 |
| A6 | Dockerfile v0.3 + wheelhouse 未提交 | Mac 工作区 `M Dockerfile` + `?? wheelhouse/` | 提交 Dockerfile；wheelhouse 用脚本生成不入库（或 git-lfs） | ❌ 待提交 |

**目标产物**：`docker buildx build --platform linux/amd64 --push` 一条命令可复现，零出网（或明确镜像源）。

## B. 渲染层 capsule/k8s_render.py（9 项，全部 ❌）

| # | 问题 | 今晚后果 | 修复 |
|---|------|----------|------|
| B1 | CPU 单位 `1m`（毫核） | Pod 仅 0.1% 核 | ✅ 已修（commit 798e91a，golden 同步） |
| B2 ✅ | 不生成 `command` | K8s `args` 替换镜像 CMD → 执行 `--model vgg16…` 秒退 CrashLoop | Deployment 加 `command: ["python","-m","capsule.slice_server"]` |
| B3 ✅ | 不生成 `--bind 0.0.0.0` | 服务绑 127.0.0.1 → Service 拒连、探针失败 | args 加 `--bind 0.0.0.0`（并在 slice_server 改默认值） |
| B4 ✅ | 不生成 `TORCH_HOME` | 容器尝试联网下载 checkpoint → CrashLoop | env 加 `TORCH_HOME=/models`；权重卷必须含 `hub/checkpoints/` |
| B5 ✅ | plan manifest 不进 ConfigMap | 需手工 `kubectl create configmap mopar-plan` + 挂载 | 渲染器生成 `mopar-plan` ConfigMap（from manifest json）+ 挂载 `/app/plan` |
| B6 ✅ | 无 nodeSelector 参数 | 多节点会乱调度 | CLI 加 `--node-selector k=v`，注入 Deployment |
| B7 ✅ | 无 imagePullPolicy 控制 | 同 tag 换镜像后 kubelet 用缓存旧镜像 | CLI 加 `--image-pull-policy Always` |
| B8 ⚠️ | 限额公式低估实例化峰值 | `(alloc+600)×1.3=947Mi` < vgg16 实测 **1312MB** → OOMKilled | 两条路：a) 改公式计入 checkpoint 峰值；b) **slice_server shard-first 加载**（根治，见 C2），限额定为 `alloc + 运行时底座 + 余量` |
| B9 ✅ | requests.cpu=1 核过胖 | 节点占用 94% 时 4 片排不进 | CLI 加 `--requests-cpu 100m`（默认不变，部署脚本传参） |
| B10 ✅ | 权重卷形态 | ROX PVC 无法绑定 local-path；hostPath 手工切换 | 渲染器支持 `--weights-volume hostPath:/var/mopar/weights` 或 PVC 两种模式 |

**测试要求**：每项修改都要更新 `tests/golden/resnet50_k3.yaml` + 单测断言；新增 vgg16 golden。

> 2026-11-04 状态（C2 已完成）：`--meta-load` 使每片 Pod 峰值 1470MB→765MB（-48%），链输出 max-diff=0.0；`--mem-limit-mi 1024` 可用。
> 2026-11-04 状态：B2-B7/B9/B10 已实现（`--node-selector`/`--image-pull-policy`/`--requests-cpu`/`--mem-limit-mi`/`--bind`/`--transport`/`--torch-home`/`--weights-mode`/`--weights-host-path`/`--weights-mount`/`--activations-mode`/`--no-plan-configmap`），13 项渲染测试 + 全量 248 测试通过（34 项 Linux 专属跳过）。B8 提供 `--mem-limit-mi` 覆盖（根治仍是 C2 shard-first）。

## C. 运行时 capsule/slice_server.py（3 项）

| # | 问题 | 修复 |
|---|------|------|
| C1 | 默认绑定 127.0.0.1 | `--bind` 默认改 `0.0.0.0`（K8s 场景）；本地测试传 `127.0.0.1` |
| C2 ✅ | `zoo.load` 实例化加载全量 checkpoint（528MB+底座 → 峰值 1312MB） | **shard-first 加载**：`zoo.load(weights=None)` 随机实例化 → `weight_shard.load_shard` 回填本片 → 峰值降至 ~600MB；需处理 tied-weights（Qwen）与 buffer |
| C3 ✅ | checkpoint 需随卷提供 | 部署文档/脚本明确：`/models/hub/checkpoints/` 放原始 checkpoint（或 C2 后不再需要） |

## D. 集群环境（4 项，环境固有，做替代方案）

| # | 问题 | 替代方案 |
|---|------|----------|
| D1 | 节点 CPU requests 被其他租户占满 | 已清理租户（现余 91%）；渲染器 B9 兜底 |
| D2 | local-path PVC 不可用（helper 镜像拉不到） | hostPath（已用）；或修 provisioner helperImage |
| D3 | 无 ingress controller | slice-0 用 NodePort/port-forward（未启用 ingress） |
| D4 | 无 metrics-server | D2 用 Pod 自报 VmHWM（A4 JSONL） |

## E. 编排层（5 项，全部 ❌）

| # | 问题 | 修复 |
|---|------|------|
| E1 | 无一键命令 | 写 `scripts/k8s_e2e.sh <model> <slices> <lambda>`：切分→渲染→构建（可选）→apply→rollout→探针→采集→summary |
| E2 | 部署/验证全手工 | 同上脚本内封装：`kubectl apply`、`rollout status`、`kubectl run` tester、探针、日志采集 |
| E3 | 探针需独立 tester Pod（slice 容器内跑会 OOM） | 封装成 `--difftest` 模式（Job/Pod），内存 ≥3Gi |
| E4 | 迭代残留旧 RS/Pod 抢占配额 | 部署脚本先 `kubectl delete deploy/rs/pod -l mopar.io/model=<m> --ignore-not-found` 再 apply |
| E5 | 手工补丁的 manifests 在仓库外 | 渲染器修好后删除手工目录；部署产物落到 `build/k8s/<model>/`（已 gitignore 或入库策略待定） |

## 验收标准（Definition of Done）

在任意一台 x86-64 节点上：

```bash
# 1. 切分（画像 + 真实激活内存）
python -m capsule.bridge --model vgg16 --slices 4 --mb-per-vcpu 224
# 2. 渲染（一条命令产出可直接部署的 manifests）
python -m capsule.k8s_render --manifest tests/fixtures/vgg16_manifest.json \
    --out build/k8s/vgg16 --namespace mopar-vgg16 \
    --image <harbor>/mopar-slice-runner:v0.3 --node-selector kubernetes.io/hostname=dell-52 \
    --requests-cpu 100m --plan-configmap
# 3. 部署 + 验证（一条命令）
scripts/k8s_e2e.sh --apply build/k8s/vgg16 --probe
# 期望：4 Pod Ready、D1 max-diff<=1e-4、D2/D3 报告生成
```

## 修复顺序建议

1. **B2-B5**（渲染器核心 4 项，约 2 小时）
2. **C2**（shard-first 加载，约 3 小时，同时解决 B8）
3. **B6/B7/B9/B10**（渲染参数，约 1 小时）
4. **E1/E2/E4**（编排脚本，约 2 小时）
5. **A6 提交物料 + 回归**（约 1 小时）

## 历史提交参考（k8s-prep 分支）

- `120a320` A+B：slice_server/权重分片/计量探针/负对照
- `ff3ca8b` K0 渲染器
- `21e3187` M2' 四 Agent K8s 化
- `041de78` M3a：工具参数服务端钉死
- `9f1b2b3` vgg16 fixture（4 片）
- `0a94de3` meters `--manifest` 透传
- `798e91a` CPU 单位 1m→1
- `be2cb17` requirements 平台标记 + PyYAML
- `0890213` kubeconform 二进制解除跟踪
- `c1df638`/`ff42e55`/`20ed997` Dockerfile 迭代
