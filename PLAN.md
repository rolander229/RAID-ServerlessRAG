# MOPAR 复现计划

复现对象：**MOPAR: A Model Partitioning Framework for Deep Learning Inference Services on
Serverless Platforms**（arXiv:2404.02445v1, IEEE Trans.）。

约束（用户给定）：

1. 不使用 docker、不使用 kubernetes，用**裸机物理机替身**替代 OpenFaaS。
2. 一切读写只发生在 `v100:/storage/xuedh_workspace/mopar-impl`，其余目录只读/可复制。
3. 尽量用原文所述模型，权重下载到本目录 `weights/`。
4. 难以解决的问题落盘到 `issues/`。
5. 4 张 Tesla V100-SXM2-32GB 可用（用户 2026-09-21 授权），故 GPU/LLM 一节列入正式范围。
6. **不复用 `/storage/xuedh_workspace/mono2faas` 的代码与结论**（用户判定其多为 trash；其自有文档记录过一条
   profiler 线是 `time.sleep()` 伪造的）。裸机运行时在本仓库从零实现；任何引用到的数字都必须在本仓库实测得到。

## 1. 论文要复现什么

MOPAR 由三个组件构成，对应三组可验证数字：

| 组件 | 论文位置 | 机制 | 判据 |
|---|---|---|---|
| **SP** 服务画像 | II-B | 算子/层/服务三级画像；取占该层内存 80% 的 dominant 算子；用 `<模型,输入尺寸,参数量> → <内存,耗时>` 回归 | Table I：层内存 RMSLE LR 0.156 / XGBoost 0.105 / RF 0.139；Fig 5：Conv2D 0.0319、MatMul 0.0427；耗时平均精度 >95% |
| **MPE** 划分引擎 | II-C, Alg.1 (HyPAD) | 节点消约（相邻层内存差 ≤5%）+ 边消约 → G′；DP 求垂直切点；对每片搜索并行度 η≤M/λ | 每模型 3–5 片（Fig 11）；MPE 使 CNN/RNN/GCN 延迟降 1.515×/1.403×/1.273×（Fig 13a）；资源效率平均 +27.62% |
| **COM** 通信优化 | II-D | ①AE（2D 卷积，编解码各一层）压缩 slice 间张量；②`/dev/shm` 式共享内存 + Linux pipe 消息通道 | 数据传输降 2.92×；ConvNeXt 延迟 1.29 s→0.605 s（2.13×）；共享内存使 CNN/RNN/GCN 延迟降 52.01%/40.34%/53.15%；精度损失 ~0.05% |

端到端（III-A）：12 模型、5 基线（Unsplit / Uniform / NonSplit / AlpaServe / Clockwork++）、
PAI 昼夜 trace（250–1250 rps，100 KB–100 MB），指标为 95 分位延迟、归一化内存利用率、
每请求成本。目标数字：相对 Unsplit 延迟降 5.47%/6.71%/9.31%（CNN/RNN/GCN）；相对其余 4 个
划分基线平均延迟降 19.19%；OpenFaaS 成本降 5.6×、Lambda 降 2.58×（Table III）。

GPU/LLM 部分（III-B，ChatGLM 6B/12B，Table IV/V）列为**stretch**，排在 CPU 主线之后。

## 2. OpenFaaS 的裸机替身

论文里 function 的本质属性只有三条与结论相关：**①每个 slice 是一个独立进程/实例，②实例有
内存配额（配额决定 vCPU 数，λ = MB/vCPU），③实例数随请求量自动增减**。容器与 k8s 只是载体。
因此替身保留这三条，替换掉载体：

| 论文概念 | 裸机实现 | 偏差说明 |
|---|---|---|
| Docker 容器 function | Python worker 进程，监听 UNIX socket，HTTP/1.1 收请求 | 无镜像层，冷启动≈import+加载权重；单独计时上报 |
| OpenFaaS gateway | `runtime/gateway.py`：路由、LB、请求计数 | 只实现同步调用路径 |
| 自动扩缩容 | `runtime/autoscale.py`：按队列深度增减副本，受全局 CPU/内存预算约束 | 无 scale-to-zero 冷启动收益，故**不声称**复现冷启动相关结论 |
| function 内存配额 | 子进程 `RLIMIT_AS`（CPU 推理）；GPU 用 `set_per_process_memory_fraction` | RLIMIT_AS 对 CUDA 进程过紧会误杀，故 GPU 走另一条路 |
| slice 水平切分为 η 个子片 | 按输出通道把该层参数切成 η 份，每份一个 worker，再做参数聚合（即 Eq.5 的 `ta`） | 真实分片，聚合耗时实测而非假设 |
| 14 台服务器集群 | 单机 80 核 / 251 GB 划分为若干「server」预算域，slice 按亲和性放置 | **同机模拟**：跨 server 的边用带宽模型计价，同 server 才走 `/dev/shm` |
| tmpfs 共享目录 | `/dev/shm`（本身即 tmpfs，无需 root 挂载） | 等价 |
| Linux pipe 消息通道 | `os.pipe()` 全双工一对 | 等价 |
| Redis 外部存储（对照组） | 本地 Redis（若不可用则退化为 TCP socket 中转） | 落 `issues/` 说明 |
| Lambda 定价 | `cm=1.667e-5 $/GB/s`，最小 128 MB，按 Eq.6 计费 | 直接照抄定价模型 |

## 3. 分阶段执行

- **P0 骨架**：包结构、`scripts/sync.sh`（本地编辑镜像 → v100，rsync 不带 `--delete`）、文档、venv。
- **P1 模型层图**：12 模型统一抽象（节点=层，边=张量 shape/dtype），优先复用
  `/storage/xuedh_workspace/models` 已有权重。
- **P2 SP**：`probe` 采集算子级内存/耗时 → 三种回归 → RMSLE 表（对齐 Table I）。**已完成，
  结论为「论文精度指标在实测显存上复现不出来」，见 `docs/results.md`。**

### 余下工作按论文实验章的 A/B 两段推进（用户 2026-09-21 指定，2026-09-22 收窄范围）

**范围只覆盖 §III-A 与 §III-B**（用户 2026-09-22：「所有实验只需要复现论文 3.1 和 3.2，
不用考虑其他无关实验」）。§I-B 的 Fig 2/Fig 3、§II 的 Fig 4–8、§III-C 的 Fig 13 消融
**不再是交付项**；SP（P2）与 COM（P4）作为 MPE 的前置和 MOPAR 的机制仍然要存在，但它们
那张扫描图不进结果表。

图的归属按 PDF 原文核对如下（此前把 Fig 8 记在 3.1 名下是错的，它属 §II-D 设计图）：

| 论文位置 | 图/表 | 性质 |
|---|---|---|
| §I-B Observations | Fig 2a–d、Fig 3a/b | 动机测量 ⇒ **超出范围** |
| §II-B/C/D 设计 | Fig 4、Fig 5、Fig 6、Fig 7、Fig 8 | Fig 5 是 SP 测量（P2 已做），其余为设计图 |
| **§III-A-1 Setup** | Table II、Fig 9 | 硬件与请求 trace |
| **§III-A-2 结果** | **Fig 10a/b/c、Table III** | 归一化内存利用率、OpenFaaS/Lambda P95、每请求成本 |
| **§III-A-3 Transformer** | **Fig 12a/b** | 四个 Transformer 的资源消耗与 latency |
| §III-C MPE 段引用 | Fig 11 | 每 slice 配额矩形 + 「3–5 片」；图在 §III-A 页、正文在 §III-C ⇒ 作为 3.1 切分方案的直接证据保留 |
| **§III-B** | **Table IV、Table V** | ChatGLM 系列 char/s、accuracy、communication ms |
| §III-C | Fig 13a/b/c | 消融 ⇒ **超出范围** |

所以 A 段本身就是端到端实验，没有微基准可退。

| 阶段 | 论文里对应的图/表 | 要交付的东西 | 依赖 |
|---|---|---|---|
| **P3 MPE 落地** | Fig 11（每模型 3–5 片） | 已完成：`profile/timing.py`、12 服务切分方案；λ 敏感性已扫（`runs/e3b_cut_price.json` 144 行：λ≥448 只有三个 Transformer 切得开，λ=1792 下 9/12 服务恒为一片，结论写在 `issues/hypad-cut-price.md`） | P2 |
| **P4 COM（降级为机制）** | — | 已完成：`/dev/shm` + `os.pipe()` 通道、AE 编解码；`runs/e4_communication.json` 的 240 行压缩比扫描留作附录。**负载强度**：论文未给开环到达率，按用户定的口径改为**闭环固定并发** C ∈ {1,2,4,8}，理由与后果写在 `issues/faas-bare-metal-fidelity.md` §3 | P3 |
| **P5 裸机运行时** | — | 已完成：`faas/runtime.py`：η→affinity+线程数、alloc→RLIMIT_AS、placement→车道/模型链路、拓扑重编号；四组配置数值与 unsplit 一致；闭环并发已实现（`Deployment.sweep`），且 `release()` 的按值持有缺陷已修（`ab6ef84`，见 `issues/` §7） | P4 |
| **P6 A 段端到端** | **Fig 10a/b/c、Table III、Fig 12、Fig 11** | 已完成：四个基线切分器（AlpaServe DP、Clockwork++ 放置、NonSplit 非连续 IP、Uniform 已有）；闭环固定并发取 P95；Table III 成本（$1.667e-5/GB/s、128MB 起价、每 100 RPS 内存）。产物 `runs/e5_end_to_end.json` 72 行 / 0 失败 / 18 次实际部署，读数写在 `docs/results.md`「端到端」一节 | P5 |

**P7 = 论文 §III-B「Evaluation with LLMs on GPUs」**，A 段全部落盘后才开：ChatGLM-6B、
ChatGLM2-6B/12B、ChatGLM3-6B/12B（权重下到 `weights/`），2 卡与 4 卡，Table IV 的 chars/s
与 Table V 的 accuracy（MMLU + C-Eval，zero-shot CoT）/communication ms，基线 NonSplit 与
Default（PyTorch 均分权重）。**已知偏差**：论文用 bf16，V100（sm_70）没有 bf16，只能 fp16；论文是 2 台 ×2×A40 48GB
（NVLink 112.5 GB/s），我们是**单机 4×V100-SXM2-32GB**：`nvidia-smi topo -m` 给的是 NV2
全互连，每卡 6 条 link × 25.78 GB/s = 单向 154.7 GB/s，所以片间通信不是「没有 NVLink 的
降级」，而是比论文更快、且四张卡之间不存在跨 server 那一跳；真正不等价的是显存量级
（48×2 vs 32×4）与 bf16。

每阶段结束在 v100 仓库提交一次；与论文数字的对照表持续写在 `docs/results.md`。

## 4. 判定「复现成功」的标准

不追求逐点重合（硬件从 Xeon 8269CY 换成 v100 的机器、且为单机模拟集群），追求**趋势与量级**：

1. SP：XGBoost 在层内存 RMSLE 上优于 RF 与 LR，且绝对值落在 0.1 量级。
   → **未达成（P2 已结案）**：实测 RF 0.5106 最好、XGBoost 0.5572 次之，量级差 5×；
   且同一批预测器在解析目标上 LR 给 0.0000，说明该指标不区分真实测量。详见 `docs/results.md`。
2. MPE：8 个非 Transformer 模型切出 3–5 片，且「MOPAR 延迟 < Unsplit」在多数模型上成立。
   → **两条都没复现出来，且是解析意义上的 0（A 段已结案）**：λ=1792 下 Eq.(6) 的延迟约束
   不许那 9 个服务被切开，六个方法返回同一份方案；「切出 3–5 片」只在三个 Transformer 上以
   3/8/8 片出现。延迟一列在**解析**口径下恒不成立（24 个服务-批次组合里延迟最小点全在 k=1），
   在 C=8 的**实测**口径下 4 个 Transformer 都不输。详见 `issues/hypad-cut-price.md`。
3. COM：共享内存相对外部存储、AE 相对无压缩，均带来可观且方向正确的延迟下降。
   → **降级为机制，不作交付图**：§III-A 没有 COM 的独立图，端到端就是它的载体。
   实测一次 hop 3.5–10.5 GB/s vs 模型 10 GB/s（`issues/faas-bare-metal-fidelity.md` §6），
   `link_s` 与 `moved_mb` 逐窗落在 e5 的每一行里。
4. 端到端：MOPAR 的成本/请求显著低于 Unsplit，且在划分类基线里占优。
   → **成本达成、延迟有条件**：4 个 Transformer 平均省 **1.78×**（单服务 1.00–1.92×），
   8 个非 Transformer 恒等（同一个方案，省 0%）；P95 相对四个划分基线 **+31.90%**，
   但同一个 Unsplit 方案两次独立读数差 −21%～+30%，所以只认跨服务的汇总，不认单服务符号。

达不到的项，逐条写进 `issues/`，注明是「环境差异」还是「实现未到位」。
