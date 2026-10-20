# 智能算法模型云原生服务化工具: Serverless 深度学习推理模型拆分与函数封装框架

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCU124-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-179%20Passed-brightgreen.svg)]()

> 本仓库包含两个引擎：**`slicer/`（拆分引擎：算子画像 + HyPAD 动态规划切分）** 与 **`capsule/`（封装引擎：云原生函数自动封装 + 六维封装准确率评测）**，由 `capsule/bridge.py` 端到端桥接。

---

## 🌟 核心理念与创新点

在无服务器计算（Serverless / FaaS，如 AWS Lambda、OpenFaaS）架构下，深度学习模型推理常受困于**内存配额与算力刚性绑定**以及**整模型峰值内存常驻浪费**的双重瓶颈。

**slicer** 拆分引擎提出了全新的模型切分与协同流水线范式：
1. **纵向异构切分 (Vertical Slicing)**：依据算子层资源相似度与内存足迹跳变，将模型图切分为多个低配额的连续子图（Slices），打破整模型峰值显存绑定；
2. **片内水平并行 (Horizontal Intra-Slice Parallelism)**：针对计算密集型切片，利用平台定价比例 $\lambda$ 凸搜索最优并发线程数 $\eta^*$；
3. **通信开销对冲 (COM Engine)**：结合单机 POSIX 共享内存零拷贝通道与跨网络自编码器（AutoEncoder）张量降维，彻底抵消切分引入的通信损耗；
4. **多 GPU 大语言模型流水线适配**：将 MPE 调度扩展至超大参数量模型（ChatGLM3-6B、Qwen3-1.7B），突破 PyTorch 权重绑定（Tied Word Embeddings）与 KV-Cache 带来的非对称负载瓶颈。

---

## 🚀 性能亮点概览

| 评测维度 | 代表模型 / 场景 | 基准策略 | 本方案表现 | 核心收益与突破 |
|---|---|---|---|---|
| **Serverless 内存配额削减** | **VGG-16 / ResNet-50** | NonSplit (整模型常驻) | 4 Slices / 3 Slices | **内存峰值降低 66.92%**，调用成本节省 38% |
| **Serverless 尾部延迟与 SLO** | 典型 CNN / RNN 服务 | Default 均匀分片 | 资源感知切分（slicer） | **SLO 达成率提升至 94.6%**（提升 23.4%） |
| **官方 GPU LLM 基准** | ChatGLM 家族 | Default 流水线 | 协同分片（slicer） | **通信耗时降低 18.96%**，精度损失仅为 **0.030%** |
| **真实 4× V100 实测 (12GB)** | **ChatGLM3-6B** | 4-GPU Default (45.75 char/s) | 4-GPU 本方案 (57.30 char/s) | **吞吐提升 +25.3%**，显存降至 2.6~3.5GB/卡 |
| **真实 4× V100 实测 (3.8GB)** | **Qwen3-1.7B** | 4-GPU Default (33.42 char/s) | 4-GPU 本方案 (39.11 char/s) | **吞吐提升 +17.0%**，通信气泡降低 35.7% |

---

## 📦 环境依赖安装与项目迁移指南 (Installation & Migration)

当需要将本项目完整迁移部署到一台新的服务器或计算节点时，请参照以下步骤构建隔离依赖环境。

### 1. 系统与 Python 前置要求
- **操作系统**：Linux (Ubuntu 20.04/22.04 LTS 或其它主流 Linux 发行版)；
- **Python 版本**：Python 3.10+ (推荐 Python 3.10.12)；
- **计算硬件**：
  - **CPU 模式**：任何多核 x86_64 处理器即可完整运行全部 Serverless 切分算法与单机推理验证；
  - **GPU 模式 (可选)**：NVIDIA GPU (支持 Compute Capability $\ge 7.0$，如 Tesla V100, A100, RTX 3090/4090, H100 等)。

### 2. 方式 A：使用 `uv` 极速安装 (强烈推荐，10x 提速与硬链接零冗余)

`uv` 是现代超高速 Python 包管理器。它不仅能秒级解析依赖，还能在同一文件系统下使用硬链接（Hardlink）共享包缓存，避免双倍磁盘占用：

```bash
# 1. 在新机器上安装 uv (若未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者通过已有 pip 安装: pip install uv

# 2. 进入项目根目录并创建虚拟运行环境
cd mopar-impl
uv venv .venv --python python3

# 3. 一键安装全部运行时依赖 (使用国内阿里云镜像源加速)
uv pip install --python .venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  -r requirements.txt
```

### 3. 方式 B：使用传统 `venv` 与标准 `pip` 安装

若新机器受限无法安装 `uv`，亦可采用原生标准流程：

```bash
cd mopar-impl
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

### 4. 关键迁移与存储注意事项 (Migration Tips)

- **大容量存储重定向 (防止根磁盘写满)**：
  PyTorch 与 CUDA 轮子（约 5.6GB）以及大语言模型权重（ChatGLM 12GB、Qwen 3.8GB）体积较大。若新机器系统盘根分区 `/` 较小，请在安装和运行前将缓存重定向到大容量数据盘（如 `/storage` 或 `/data`）：
  ```bash
  export UV_CACHE_DIR=/path/to/large_disk/.uvcache
  export HF_HOME=/path/to/large_disk/.cache/huggingface
  export HF_ENDPOINT=https://hf-mirror.com   # 启用 HuggingFace 国内官方镜像高速下载
  ```
- **多用户公共模型共享库**：
  在当前集群服务器上，大模型与视觉小模型权重已统一存放在公共共享目录 `/storage/xuedh_work2/models/`（含 `chatglm3-6b`、`qwen3-1.7b` 独立目录，以及 `hub/checkpoints/` 视觉模型库），团队所有成员均可直接免下载加载复用。
- **开箱即用的内置 CUDA 运行时**：
  [`requirements.txt`](requirements.txt) 指定的 `torch==2.5.1` 自带 CUDA 12.4 完整动态运行时。新服务器**无需安装系统级 CUDA Toolkit 或 nvcc**，只要机器具备正常的 NVIDIA 驱动即可直接驱动 GPU 多卡切分。
- **一键迁移完整性验证 (Smoke Test)**：
  安装完成后，执行单元回归测试：
  ```bash
  source .venv/bin/activate
  python -m unittest discover -s tests
  ```
  > 输出 **`Ran 179 tests ... OK (skipped=1)`** 即代表环境迁移 100% 成功！

---

## 📂 项目代码全景目录

```text
mopar-impl/
├── slicer/                         # 拆分引擎: 核心算法与系统实现
│   ├── models/                     # 12 个基准模型定义与 DAG 拓扑图抽取 (GraphExtractor)
│   ├── profile/                    # 算子/层/服务三级画像预测器 (Service Profiler)
│   ├── partition/                  # 切分决策核心: 消约 (Eliminate) 与 HyPAD 动态规划
│   ├── communicate/                # POSIX 共享内存零拷贝与自编码器 (AE) 压缩引擎
│   └── faas/                       # 高保真 Serverless 运行时仿真器 (RLIMIT 隔离)
├── capsule/                        # 封装引擎: 云原生函数封装与六维评测
│   ├── bridge.py                   # 拆分->封装->评测 端到端桥接入口
│   ├── evaluation/                 # 六维封装准确率评测器 + 独立专家基线
│   └── M2S/                        # 多智能体代码迁移流水线
├── experiments/                    # 论文图表与硬件基准一键复现入口
│   ├── e3c_official_fig10_slices.py # Figure 10 内存切片生成
│   ├── e5_official_ablation.py     # Figure 11 三项消融实验
│   ├── e6_gpu_llm.py               # Table 3/4 官方 GPU LLM 评测
│   ├── e6_chatglm3_real_partition.py # 真实 ChatGLM3-6B 4× V100 多卡流水线
│   └── e7_qwen3_partition.py       # 真实 Qwen3-1.7B 4× V100 多卡流水线
├── scripts/                        # 实用工具与辅助脚本
│   ├── trace_service.py            # 秒级查看任意模型在指定 λ 下的切分决策
│   ├── download_chatglm3.py        # ChatGLM3-6B 真实权重下载器
│   └── download_qwen3.py           # Qwen3-1.7B 真实权重下载器
├── runs/                           # 实验产出数据与高清绘图 (JSON / PNG)
├── tests/                          # 单元测试集 (179 个用例，100% 通过)
├── docs/                           # 技术文档体系（本地维护，不入库）
└── requirements.txt                # 项目 Python 依赖清单
```

---

## 🖥️ 评测硬件平台规格

- **物理 GPU**：4× NVIDIA Tesla V100-SXM2-32GB (总显存 128 GB，NVLink 300 GB/s 互联)；
- **物理 CPU**：80 vCPU (Intel Xeon Gold 6148 @ 2.40GHz)，251 GB 物理主机内存；
- **持久化挂载**：`/storage` 大容量独立盘 (2.2 TB 空闲空间，全量模型权重专用)；
- **软件环境**：Ubuntu Linux 22.04 LTS, PyTorch 2.5.1+cu124, CUDA 12.4, Transformers 5.17.0。
