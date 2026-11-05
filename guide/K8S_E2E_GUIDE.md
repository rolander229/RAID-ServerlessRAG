# MOPAR 全流程指南：切分 → 封装 → K8s 部署

> 面向后来者的实操文档。所有命令均在本仓库实测通过（Linux 集群 + macOS 开发机）。
> 配套材料：[MODEL_E2E_TASKS.md](./MODEL_E2E_TASKS.md)（接下来的部署任务清单）、`K8S_E2E_FIXLIST.md`（历史修复清单，仓库根）、`scripts/k8s_e2e.sh`（一键脚本）。

---

## 0. 这套系统是什么

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐
│   slicer    │    │     capsule      │    │          K8s 集群            │
│  拆分引擎    │ →  │   封装迁移引擎    │ →  │  真实分布式切片服务           │
│ 画像+HyPAD   │    │ 四Agent迁移+渲染  │    │ Deployment/Service/探针/计费  │
└─────────────┘    └──────────────────┘    └─────────────────────────────┘
   manifest.json      K8s manifests + 适配器        4 片 Running + D1=0.0
```

三阶段的分工：

| 阶段 | 输入 | 输出 | 关键点 |
|------|------|------|--------|
| **切分**（slicer） | 模型 + 样本输入 | `slice_manifest.json`（每片层列表/配额/peer） | 确定性算法（画像 → 消约 → HyPAD DP） |
| **封装**（capsule） | manifest | K8s YAML + 适配器 + 报告 | LLM 只做决策；**资源数字必须来自确定性渲染器** |
| **部署**（K8s） | manifests + 镜像 + 权重 | 4 Pod Running + 探针 PASS | 一键脚本完成全部编排 |

---

## 1. 环境准备

### 1.1 开发机（macOS / Linux 均可，切分与渲染不需要 GPU）

```bash
# 仓库
git clone git@github.com:rolander229/RAID-ServerlessRAG.git
cd RAID-ServerlessRAG && git checkout k8s-prep

# Python 3.10 虚拟环境（uv 最快）
uv venv --python 3.10
uv pip install --python .venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 预训练权重缓存（CNN 类模型需要；constructed 模型不需要）
export TORCH_HOME=/path/to/torch-home   # 内有 hub/checkpoints/vgg16-*.pth 等
```

> `requirements.txt` 中 `triton` 带 `sys_platform == "linux"` 标记（macOS 自动跳过）。

### 1.2 部署机（Linux 集群节点）

- K8s ≥ 1.28、kubectl 可用；节点已装 containerd
- 权重卷：`/var/mopar/weights/`（hostPath 形态，见 §6.1）
- 访问镜像仓库的凭证（harbor）

---

## 2. 快速开始（一条命令）

```bash
# 前提：已构建镜像（§5）且推送到 harbor
scripts/k8s_e2e.sh vgg16 4 224 \
  --image gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.4 \
  --namespace mopar-vgg16 \
  --node-selector kubernetes.io/hostname=dell-52 \
  --mem-limit-mi 1024
```

脚本自动完成 7 步：**切分 → 渲染 → 幂等清理 → 部署 → 等待就绪 → 集群内探针 → 采集与成绩单**。
产物在 `build/e2e/<model>/`：`manifest.json`、`manifests/`、`a4_logs/`、`summary.md`。

实测：**从空命名空间 60 秒到验证通过**。

---

## 3. 切分（slicer）

### 3.1 原理速览

```
模型 + 样本输入
   │
   ├─ ① 画像 profile_model：逐层的激活内存 + 耗时 + 参数（真实测量）
   ├─ ② 消约 eliminate：内存落差 < θ 的相邻层合并（缩小搜索空间）
   ├─ ③ HyPAD DP：在简化图上求「成本最优的连续分组」
   │      成本 = alloc/1024 × (tp+ta+tc) × 内存单价 + 通信单价 × tc
   │      约束：时延预算、每片配额下限（128MB）、η 并行度上限
   └─ ④ 输出 slice_manifest.json（每片 layers / alloc_mb / footprint / peers）
```

### 3.2 关键参数（三个旋钮）

| 参数 | 默认 | 作用 | 调参直觉 |
|------|------|------|----------|
| **λ**（`mb_per_vcpu`） | 1769（AWS 锚） | 内存↔算力汇率 | **λ 越小 → 内存越“贵” → 切得越细**（448 → 更多片；224 → 更多） |
| **θ**（`similarity`） | 0.20 | 消约合并阈值 | θ 越小 → 保留更多候选切点 → 理论可切更细（但 vgg16 这类平滑曲线对 θ 不敏感） |
| **K**（`slices`） | 调用方指定 | 强制片数 | `vertical_partition(K)` 是“强制恰好 K 组”；自由求解用 `partition()` |

> 注意：`partition_model(..., slices_count=K)` 走的是**固定 K** 路径；HyPAD 的自由求解（自动定片数）是 `partition()`，其片数由成本模型+时延预算决定。两者都基于同一份画像。

### 3.3 产物 manifest（字段）

```json
{
  "model_key": "vgg16",
  "plan_source": "dp_solve",          // dp_solve | hardcoded（qwen3 手工计划）
  "slices": [
    {"slice_id": 0, "name": "vgg16_slice_0", "layers": ["features.0", "..."],
     "layers_count": 24, "alloc_mb": 128.0, "footprint_mb": 41.4,
     "inbound_peers": [], "outbound_peers": [1],
     "trigger_type": "api_gateway", "path": "/predict", "type": "orchestrator"}
  ],
  "expanded_parts": [["features.0", "..."], ["..."]],
  "max_tensor_diff": 0.0
}
```

### 3.4 两个必须知道的坑

1. **`record()` 快速路径不采集激活内存**（节点 `memory=0.0`）→ footprint 退化为纯权重 → 全触 128MB 地板 → 永远 1 片。
   已修复：`bridge.partition_model` 内部接 `profile_model` 填真实激活内存。**自己写脚本时务必用画像链**：
   ```python
   graph = build_structure_graph(model, inputs)
   profile = profile_model(model, inputs, graph=graph)
   sized = profile.graph          # 带真实 memory/time 的图
   ```
2. **“切不切”由成本模型决定，不由模型大小决定**：小模型（resnet50 ~100MB）在 λ=1769 下判“不切”是**正确结论**（单片 128MB 就够）。想强制切：降 λ 或给固定 K。

---

## 4. 封装（capsule）

### 4.1 四 Agent 迁移（真 LLM）

```
manifest → Architect(蓝图) → Developer(适配器) → K8s Engineer(渲染+overlay) → Validator(校验)
```

- LLM 只决定“做什么”（命名/拓扑/适配逻辑）；**资源数字一律来自 `k8s_render`**（服务端注入，LLM 无法篡改）。
- 运行：`python -m capsule.m2s_k8s.runner --manifest ... --out ...`；`--fake-llm` 仅用于离线调试。
- 产物：`k8s_migrated/{blueprint.json, manifests/, adapters/, report.json}`。

### 4.2 确定性渲染器 `k8s_render`

```bash
python -m capsule.k8s_render \
  --manifest tests/fixtures/vgg16_manifest.json \
  --out build/k8s/vgg16 --namespace mopar-vgg16 \
  --image <harbor>/mopar-slice-runner:v0.4 \
  --node-selector kubernetes.io/hostname=dell-52 \
  --requests-cpu 100m --mem-limit-mi 1024 \
  --weights-mode hostPath --weights-host-path /var/mopar/weights \
  --activations-mode omit --meta-load --bind 0.0.0.0 --transport inline
```

关键开关（每个都对应一个真实故障）：

| 开关 | 作用 | 不设的后果 |
|------|------|-----------|
| `--meta-load` | slice_server 走 shard-first（C2） | 每 Pod 峰值 1312MB，限额需 3Gi |
| `--bind 0.0.0.0` | 服务监听所有网卡 | 绑 127.0.0.1 → Service 拒连、探针失败 |
| `--torch-home /models` | 容器内预训练权重查找路径 | 联网下载 checkpoint → 失败 |
| `--requests-cpu 100m` | 调度申报调小 | 每片 1 核 → 拥挤节点排不进 |
| `--mem-limit-mi` | 峰值覆盖 | 947Mi 撞破 vgg16 实例化峰值 → OOMKilled |
| `--node-selector` | 钉节点 | 多节点乱调度 |
| `--weights-mode hostPath` | 权重卷形态 | PVC 在某些集群不可用（helper 镜像拉取失败） |
| `--meta-load` + `--activations-mode omit` | 免 528MB checkpoint | 峰值高/需搬运大文件 |

### 4.3 产物体检

```bash
kubectl apply --dry-run=client -f build/k8s/vgg16/   # 语法+字段
kubeconform -strict -summary build/k8s/vgg16/        # 结构校验（仓库 bin/kubeconform）
```
D4/D5 指标由 `kube_validate_tool` 执行：limits 与 alloc+地板+余量一致（±16Mi）、cpu 折算正确、拓扑线性、plan 挂载完整。

---

## 5. 镜像构建（离线 wheelhouse，零出网）

```bash
# 1) 生成依赖轮子仓库（联网一次，可复用）
bash scripts/download_wheels.sh          # torch/torchvision(+cpu) + 依赖，钉死版本

# 2) 交叉构建并推送（Mac ARM → linux/amd64）
docker buildx build --platform linux/amd64 \
  -t gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.4 --push .
```

要点：
- 基座用 `docker.m.daocloud.io/library/python:3.10-slim`（docker.io 在校园网常被劫持）
- torch 与 torchvision **必须同源**（都用 +cpu 版本，否则 `operator torchvision::nms does not exist`）
- 离线安装：`pip install --no-index --find-links /wheelhouse ...`（构建零网络依赖）

---

## 6. 部署（K8s）

### 6.1 集群前置（一次性）

```bash
# 命名空间（脚本会自动建）
kubectl create namespace mopar-vgg16

# 权重目录（hostPath 形态，所有 Pod 钉同一节点）
#   /var/mopar/weights/
#   ├── vgg16/slice_{0..3}.pt          ← 切分产物（weight_shard 导出）
#   └── hub/checkpoints/vgg16-*.pth    ← 预训练 checkpoint（非 meta 模式才需要）
```

### 6.2 一键脚本（E 阶段）

```bash
scripts/k8s_e2e.sh <model> <K> <lambda> --image <img> --namespace mopar-<model> \
  --node-selector kubernetes.io/hostname=<node> --mem-limit-mi <N>
```

脚本内部（顺序即依赖）：

| 步骤 | 动作 |
|------|------|
| 1 | `partition_model` 切分（画像+DP），写 `manifest.json` |
| 2 | `k8s_render` 渲染（meta-load/hostPath/omit/PVC 条件渲染） |
| 3 | **幂等清理**：`kubectl delete deploy,rs,pod -l mopar.io/model=<model>` |
| 4 | `kubectl apply`（跳过占位差分 Job） |
| 5 | 逐片 `rollout status`（300s/片） |
| 6 | **集群内探针**：独立 tester Pod（3Gi）→ 全链 `--probe` |
| 7 | 逐 Pod 抓 A4 JSONL → `summary.md` |

### 6.3 验收指标

| 指标 | 采集方式 | 通过标准 |
|------|----------|----------|
| **D1 数值等价** | 集群内探针（链 vs 单体） | `max-diff ≤ 1e-4`（实测 0.0） |
| **D2 内存配额** | `healthz.rss_hwm_mb`（VmHWM 自报） | 峰值 < 限额；比率有区分度 |
| **D3 成本/延迟** | Pod 内 `/tmp/logs/*.jsonl`（hop_ms/busy_ms/moved_mb） | 逐跳单调、无超时 |
| **D4 资源规格** | kube_validate_tool（limits/cpu/GPU/挂载） | 0 errors |
| **D5 拓扑** | 交付文件自检（Service/Ingress/peers） | 线性链一致 |

### 6.4 故障排查表（今晚全部实测坑）

| 症状 | 根因 | 修复 |
|------|------|------|
| Pod Pending：`persistentvolumeclaim "mopar-weights" not found` | 引用了已删除的 PVC / 文本替换静默失败 | 改 hostPath 或用 `yaml.safe_dump` 结构化改 |
| Pod Pending：`Insufficient cpu` | requests.cpu 每片 1 核，节点被其他负载占满 | `--requests-cpu 100m` |
| CrashLoop：容器立即退出 | K8s `args` 替换镜像 CMD，缺 `command` | 渲染器加 `command: [python, -m, capsule.slice_server]`（已内置） |
| CrashLoop：`operator torchvision::nms does not exist` | torch CPU + torchvision CUDA ABI 不匹配 | 双轮子同源（+cpu）重建镜像 |
| CrashLoop：`Downloading vgg16-*.pth` | 无 TORCH_HOME/checkpoint | 权重卷放 checkpoint + `TORCH_HOME=/models`，或 `--meta-load` |
| CrashLoop：OOMKilled（exit 137） | 实例化峰值 1312MB > 947Mi 限额 | `--meta-load`（峰值 →785MB）或 `--mem-limit-mi 2048` |
| Service 连不通、探针失败 | 服务绑 127.0.0.1 | `--bind 0.0.0.0`（渲染器默认已发） |
| 改镜像后仍跑旧代码 | 节点缓存了同 tag 镜像 | `--image-pull-policy Always` 或换 tag（v0.3→v0.4） |
| 探针 137（OOM） | 探针在 slice 容器内跑，叠加单体模型 | 独立 tester Pod（脚本已内置） |
| `kubectl delete rs 65f75bd969` NotFound | 裸哈希 ≠ 全名 | 用全名或 `-l` 标签选择器 |
| Mac 上 `os.sched_getaffinity` 不存在 | 裸机运行时是 Linux 专属 | 测试已加 `skipUnless(platform=="Linux")` |

---

## 7. 适配其它模型（重点章节）

### 7.1 模型注册：`slicer/models/zoo.py` 的 `ModelSpec`

每个模型在 `SPECS` 里注册一条：

```python
ModelSpec(
    "my_cnn",                 # key（CLI/脚本用）
    "MyCNN-12M",              # 展示名
    "CNN",                    # 家族：CNN / Transformer / Recurrent / Graph / LLM
    "constructed" | "pretrained_torchvision" | "hf",
    lambda: build_my_cnn(),   # 构造函数（必须返回 nn.Module，eval 模式前向）
    inputs_fn,                # 代表性输入（形状即画像输入）
    batch=1, cite="[x]", note="...",
)
```

- **constructed**：代码现场构造（随机初始化），**零下载**，适合切分/画像/迁移验证（bert_3_0b、inception_transformer 都属于此类）。
- **pretrained_torchvision**：走 `torchvision.models.*(weights=DEFAULT)`（vgg16/resnet50），需要 `TORCH_HOME` 缓存。
- **hf**：HuggingFace 权重（chatglm3-6b、qwen3-1.7b），需要共享模型目录或下载。

注册后 `zoo.load(key)` 即可用，切分/渲染/部署全链路自动适配。

### 7.2 三类模型的差异

| 类型 | 代表 | 权重来源 | 部署要点 |
|------|------|----------|----------|
| **CNN**（torchvision） | vgg16 / resnet50 | `TORCH_HOME` 缓存 checkpoint | 非 meta 模式需把 checkpoint 放权重卷；**有 BN buffer**（见 7.3） |
| **constructed Transformer** | bert_3_0b（8 片）/ inception_transformer | 代码构造，**无需下载** | 直接 `--meta-load` 即可；内存曲线锯齿明显，天然多片 |
| **真 LLM**（HF） | qwen3-1.7b / chatglm3-6b | 共享模型目录（/storage/.../models） | GPU 资源 + P-a 形态（见 7.4）；**tied weights 约束**（见 7.3） |

### 7.3 分片与加载的两个硬约束

1. **BN buffer 不是 parameter**：`running_mean/var/num_batches_tracked` 在 `_buffers` 表里。
   `load_shard` 已同时覆盖 parameters 与 buffers；写新模型时**不要假设只有 parameters**。
2. **tied weights（Qwen3）**：`embed_tokens` 与 `lm_head` 共享同一张量。分片导出必须把它们放进**同一片**（`weight_shard` 的 tied 校验会拒绝拆散），部署时同一设备。

### 7.4 GPU 模型（LLM）的部署形态

| 形态 | 结构 | 适用 |
|------|------|------|
| **P-a 单 Pod 多卡**（推荐起步） | 1 Pod `limits.nvidia.com/<card>: 4`，进程内 `device_map` 按片分卡 | qwen3-1.7b / chatglm3-6b 的 4 卡流水线；与既有实验脚本 100% 兼容 |
| P-b 每 Pod 一片一卡 | 4 Pod × 1 卡，激活走网络 | 真正的“片即函数”；通信延迟劣化（跨 pod TCP） |

资源名注意：不同集群的 GPU 资源名不同（如 `nvidia.com/v100`、`nvidia.com/gpu`），渲染器用 `--gpu-resource` 指定（v0 仅预留）。

### 7.5 新模型适配 Checklist（逐步照做）

```
□ 1. 注册 ModelSpec（zoo.py）：key/家族/构造/输入形状
□ 2. 冒烟：python -c "from slicer.models import zoo; m,x=zoo.load('key'); print(m(x).shape)"
□ 3. 切分：partition_model('key', K, λ) → 检查 manifest（片数/alloc/层边界合理）
        - 若自由求解判“不切”：降 λ（448/224）或给固定 K 强制切
□ 4. 渲染：k8s_render --meta-load ... → kubeconform 通过
□ 5. 分片：weight_shard 导出（若模型有 tied weights，确认同片校验通过）
□ 6. 权重入卷：分片 +（非 meta 时）checkpoint 放 /var/mopar/weights/<key>/
□ 7. 一键部署：scripts/k8s_e2e.sh key K λ --image ... 
□ 8. 验收：探针 max-diff=0.0；healthz 峰值 < 限额；A4 日志逐跳正常
□ 9. 记录：模型类型/参数/片数/配额/峰值 写进运行记录（wiki/README）
```

### 7.6 两个现成实例

```bash
# ① constructed Transformer（零下载，多片天然）
scripts/k8s_e2e.sh bert_3_0b 8 448 --image <img> --namespace mopar-bert3b \
  --node-selector kubernetes.io/hostname=dell-52 --mem-limit-mi 2048

# ② 真 LLM（需 GPU + 权重目录；P-a 形态单独渲染 GPU 资源）
#    （v0 渲染器仅预留 GPU 字段，qwen3 形态见 m2s-k8s-adaptation-plan.md K2 节）
```

---

## 8. 命令速查

```bash
# 环境
uv venv --python 3.10 && uv pip install -r requirements.txt
export TORCH_HOME=/path/to/torch-home

# 切分（单模型，含画像）
python -c "from capsule.bridge import SliceCapsuleBridge as B; import json; \
  print(json.dumps(B().partition_model('vgg16', 4, mb_per_vcpu=224), indent=2))"

# 渲染
python -m capsule.k8s_render --manifest m.json --out out/ --namespace ns \
  --image IMG --meta-load --node-selector k=v --requests-cpu 100m --mem-limit-mi 1024 \
  --weights-mode hostPath --weights-host-path /var/mopar/weights --activations-mode omit

# 镜像
bash scripts/download_wheels.sh
docker buildx build --platform linux/amd64 -t IMG --push .

# 一键端到端
scripts/k8s_e2e.sh vgg16 4 224 --image IMG --namespace mopar-vgg16 \
  --node-selector kubernetes.io/hostname=dell-52 --mem-limit-mi 1024

# 验收
kubectl get pods -n mopar-vgg16
kubectl exec -n ns deploy/<model>-slice-0 -- python -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/healthz').read().decode())"
cat build/e2e/<model>/summary.md
```

---

## 9. 术语表

| 术语 | 含义 |
|------|------|
| alloc_mb | 切分器给每片的**计算配额**（权重+激活+下限圆整） |
| footprint_mb | 该片的实际模型数据需求（权重+峰值激活） |
| λ | 内存↔算力汇率（MB per vCPU），AWS 锚 1769 |
| θ | 消约合并阈值（相似度） |
| K | 强制片数（vertical_partition 入参） |
| shard | 权重分片文件 `slice_i.pt`（可从零还原全部参数） |
| meta 设备 / meta-load | 零字节骨架 + 按分片只物化本片（C2） |
| D1-D5 | 数值等价/内存/成本延迟/资源规格/拓扑五维验收 |
| 跳数税 | K8s HTTP 链相对裸机 /dev/shm 链的延迟开销（≈3×） |
