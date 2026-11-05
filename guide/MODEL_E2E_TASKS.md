# 模型端到端部署任务清单（Model E2E Tasks）

> 回答"接下来用这套工具部署哪些模型、怎么排期"。
> 工具用法见同目录 [K8S_E2E_GUIDE.md](./K8S_E2E_GUIDE.md)；历史修复见仓库根 `K8S_E2E_FIXLIST.md`。
> 状态标记：☐ 未开始 · ◐ 进行中 · ✅ 完成。日期基准：2026-11-05。

---

## 0. 资源基线与硬约束

| 项 | 值 | 对任务的影响 |
|----|-----|-------------|
| 部署节点 | **dell-52 单 worker**（dell-51 为带污点控制面，不收 Pod） | 所有片钉同一节点 |
| 内存空闲 | ~23 Gi（节点 32G，其他负载 ~9G） | 全部模型限额合计须 ≤ ~20Gi |
| CPU requests 空闲 | ~5.5 核 | 8 片 × 100m = 0.8 核，无压力 |
| 镜像 | `mopar-slice-runner:v0.4`（CPU torch + meta-load） | LLM/GPU 形态暂不在此列 |
| 权重卷 | hostPath `/var/mopar/weights/<model>/` | 每个模型部署前须推送分片 |
| 激活传输 | dual 模式：inline ≤2MB / file >2MB | **大 batch 会触发 file 传输** → 见前置工作 ③ |

---

## 1. 模型清单总表（v2 · 大 batch 版）

| # | 模型 | 参数量（体积） | 拉取/下载方式 | 预期切片数 | 预期输入 size（大 batch） | 状态 |
|---|------|---------------|--------------|-----------|--------------------------|------|
| 0 | **bert_1_3b** | 1.3B / 5.2 GiB | 无需下载（zoo 内 constructed，`BertStack` 现场构造） | **8** | ids `(32, 128)` ≈ 32KB | ☐ 已预检，待部署 |
| 1 | **BERT-Large** | 333M / 1.33 GiB | 无需下载（复用 `BertStack`，改 5 个超参注册） | **4**（可扩 8） | ids `(32, 128)` ≈ 32KB | ☐ |
| 2 | **ConvNeXt-Large** | 198M / 792 MB | 随机初始化（推荐，零下载）；可选 torchvision pretrained（787MB） | **4**（按 stage 切；stage-3 可扩 8） | image `(16, 3, 224, 224)` ≈ **9.6 MB** | ☐ |
| 3 | **ViT-Large/16** | 307M / 1.23 GiB | 新写标准 ViT 构造器（~40 行，零下载）；可选 torchvision `vit_l_16` pretrained（1.23GB） | **4**（可扩 6） | image `(16, 3, 224, 224)` ≈ **9.6 MB** | ☐ |
| 4 | **GPT-2-Medium** | 355M / 1.42 GiB | 新写 CausalStack 构造器（~1h，零下载）；可选 HF `gpt2-medium`（走 hf-mirror） | **4** | ids `(32, 128)` ≈ 32KB | ☐ |

> 体积均按 fp32 计。参照基线：vgg16 = 138M / 0.53 GiB（✅ 已完成 K=4 部署，batch=1）。

**为什么是这五个**：覆盖 文本/视觉 × 编码器/解码器 × CNN 代际对照 × 0.5-5Gi 尺寸带；每个都是"真多片"（经济性自然成立，非强制 K）；全部零下载即可跑（constructed）。

**资源核算**：bert_1_3b K=8 限额 ≈ 13Gi（batch 32 后 ~16Gi）；其余四个 K=4 各约 4-6Gi——任意两个可同时驻留，全部串行更稳。

---

## 2. 通用前置工作（三个工作项，阻塞所有模型）

### ① ☐ 分片导出 + 推送节点

constructed 模型的分片同样要在切分后落到节点卷。`scripts/k8s_e2e.sh` 增加：

```
[切分之后] export_shards(manifest) → build/e2e/<model>/shards/slice_*.pt
           scp → dell-52:/var/mopar/weights/<model>/
```

工作量：~30 分钟（脚本内加导出 python 块 + scp 步骤，`--skip-deploy` 也执行导出）。

### ② ☐ batch 贯通（batch 成为一等切分参数）

大 batch 下激活占比 10-30%，**切分边界随 batch 移动**——manifest 必须记录四元组：

```
partition_model(model, K, mb_per_vcpu=λ, batch=B)
  → manifest.serving_batch = B
  → 渲染器 args 注入 --batch B
  → 探针以同 batch 发请求（probe 与服务端一致，否则 D1 对照不成立）
```

工作量：~1 小时（bridge/渲染器/脚本三处小改 + golden 同步）。

### ③ ☐ 激活卷 hostPath（file 传输的跨 Pod 共享）

batch 16/32 的载荷超过 2MB inline 阈值 → dual 模式切到 file 传输 → 跨 Pod 需要共享激活目录。所有片钉同一节点时，**hostPath 就是事实上的 RWX**：

```
渲染器 --activations-mode hostPath --activations-host-path /var/mopar/activations
（与权重卷同款做法；避开被校园网卡死的 local-path PVC）
```

工作量：~30 分钟（渲染器 volume 分支 + 单测）。

> 完成标准：三个工作项各自带单测/golden，`--skip-deploy` 全链本地验证通过。

---

## 3. 各模型任务卡

### 3.1 ☐ bert_1_3b — K=8，batch=32（首选首发）

- **注册**：已在 zoo（`BertStack(hidden=768, layers=12, heads=12, ffn=3072, task_head=1, discrete=512)`，`_ids(128)`, batch=8 → 部署时 batch 覆盖为 32）
- **预检结论**（2026-11-05 实测）：K=8 alloc `[832, 640, 640, 704, 576, 640, 640, 640]` MB，diff 5.5e-7 ✅
- **部署命令**（前置三件套完成后）：
  ```bash
  scripts/k8s_e2e.sh bert_1_3b 8 448 --batch 32 \
    --image gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.4 \
    --namespace mopar-bert13b --node-selector kubernetes.io/hostname=dell-52 \
    --mem-limit-mi 2048
  ```
- **验收**：8 Pod Ready；探针 diff ≤1e-4；每跳 moved_mb 进入 MB 量级（file 传输生效）；节点内存不超 20Gi
- **风险**：slice-0（embedding+瓶颈）batch 32 下峰值可能上探——限额预留 2.5Gi

### 3.2 ☐ BERT-Large — K=4，batch=32

- **注册**：`BertStack(hidden=1024, layers=24, heads=16, ffn=4096, task_head=1, discrete=512)`，5 行改动 + golden/预检
- **价值**：与 1_3b/3_0b 构成 **0.3B → 1.3B → 3B 规模阶梯**（论文可画 alloc/切分数随规模的变化）
- **部署**：`scripts/k8s_e2e.sh bert_large 4 448 --batch 32 --mem-limit-mi 1536 ...`
- **风险**：无特殊；注意与 bert_1_3b 的 key 区分（建议 key `bert_large`）

### 3.3 ☐ ConvNeXt-Large — K=4，batch=16

- **注册**：torchvision `convnext_large` 构造（随机初始化；pretrained 可选），`_images(224)`
- **价值**：**CNN 代际对照**（vgg16 经典 vs ConvNeXt 现代，同为大尺寸）；stage 边界 [3,3,27,3] 是天然切点
- **部署**：`scripts/k8s_e2e.sh convnext_large 4 448 --batch 16 --mem-limit-mi 2048 ...`
- **风险**：depthwise 大通道激活较肥——batch 16 下每 Pod 峰值预估 ~1.5Gi，限额 2Gi

### 3.4 ☐ ViT-Large/16 — K=4，batch=16

- **注册**：新写标准 ViT 构造器（patch-embed + cls + 12×[LN→MHSA→LN→MLP] + head，~40 行），`_images(224)`
- **价值**：视觉 Transformer 对照（与 bert 家族构成"文本 vs 视觉 × 同为均匀层堆叠"）
- **部署**：`scripts/k8s_e2e.sh vit_large 4 448 --batch 16 --mem-limit-mi 2048 ...`
- **风险**：197 token 的 attention 矩阵 batch 16 下 ~13MB/层——profile 正常但 D2 峰值要盯

### 3.5 ☐ GPT-2-Medium — K=4，batch=32

- **注册**：**新写 CausalStack 构造器（~1h）**——causal-mask 自注意力 + GELU-MLP，hidden 1024 / 24 层 / heads 16 / ffn 4096 / vocab 50257
- **⚠️ 必须偏离原版一处**：embedding 与 lm_head 用**独立权重（untied）**。原版 tied 会让"首尾必须同片"与线性链矛盾 → 切分塌缩成 1 片。文档注明此偏离
- **价值**：因果解码器在切分链里的行为验证——**qwen3 GPU 化（P-a）的前置演习**
- **部署**：`scripts/k8s_e2e.sh gpt2_medium 4 448 --batch 32 --mem-limit-mi 2048 ...`
- **风险**：causal mask 的 `torch.where/triu` 是否被 graphify 正确录为步骤——先做 record 冒烟再注册

---

## 4. 执行顺序与里程碑

```
M1 ☐ 前置三件套（分片导出 / batch 贯通 / 激活卷 hostPath）
M2 ☐ bert_1_3b K=8 batch=32        ← 首发验收：8 片大 batch 全链 PASS
M3 ☐ BERT-Large K=4  ┐
    ConvNeXt-Large K=4 ┘           ← 规模阶梯 + CNN 代际（一个下午）
M4 ☐ ViT-Large K=4   ┐
    GPT-2-Medium K=4   ┘           ← 两个新构造器（一天）
M5 ☐ 数据汇总：6 模型（含 vgg16）D1-D3 对照表 + 论文数据点归档
```

每个里程碑的完成定义：`scripts/k8s_e2e.sh` 一条命令、探针 PASS、summary.md 产出、运行记录入库（wiki/实验记录）。

---

## 5. 明确排除的候选（避免后人重复踩）

| 候选 | 排除原因 |
|------|----------|
| 小模型强制 K（resnet50/inception_v3/convnext_tiny/lstm_cnn/gru_cnn/disbert） | "不该切却切"的经济性反例——**延后**，论文写作期如需负结果数据再批量跑（半天） |
| qwen3-1.7b / chatglm3-6b | 需 GPU P-a 形态 + 渲染器 GPU 字段（K2 工作）+ 自回归解码 ≠ 单次前向链——**GPU 里程碑另立** |
| U-Net / DeepLabV3 | 跳连接破坏线性链契约（inbound fan-in）——是 gate 负测试素材，非部署候选 |
| llama_1b_tiny | 因果解码器演习职能由 GPT-2-Medium 承担，不重复注册 |
| ALBERT-xxlarge | 全层共享权重（tied everywhere）→ 切分必塌缩 1 片——tied 约束的极端反例，仅作文档案例 |

---

## 6. 每个模型给论文贡献什么

| 模型 | D1 | D2 | D3 |
|------|----|----|----|
| vgg16（已有） | CNN 逐位一致（0.0） | 权重主导（batch 1 激活≈0） | 跳数税基线（batch 1，全 inline） |
| bert_1_3b | Transformer 跨片浮点漂移（e-7 量级） | 大 batch 激活占比可见 | 8 跳、file 传输生效、跳数税随 K |
| BERT-Large | 同家族规模缩放一致性 | 同上（更轻） | 4 跳对照 |
| ConvNeXt-Large | CNN（现代）等价性 | depthwise 激活形态 | 视觉载荷 9.6MB 传输 |
| ViT-Large | 视觉 Transformer 等价性 | attention 激活形态 | 同上 |
| GPT-2-Medium | 因果解码器等价性 | 词表 embedding 胖头 | LLM 化前的传输画像 |

---

*维护约定：每完成一个模型，把对应任务卡状态改为 ✅ 并回填实测数字（alloc 分布 / 峰值 RSS / 探针 diff / 各跳延迟），替换"预估"值。*
