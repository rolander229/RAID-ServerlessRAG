# Guide for RAG Serving

> **文档说明**：本文档最后更新于 **2026年5月18日**。这是一个持续维护的指南，后续内容会根据项目进展随时进行补充和调整，请留意获取最新版本。

- RAG (Retrieval-Augmented Generation，检索增强生成) 是一种结合了信息检索和生成式 AI 的技术，旨在为大语言模型 (LLM) 提供外部专属知识，从而减少幻觉并提高回答的准确性。其核心步骤通常包括文档切块 (Chunking)、向量化 (Embedding)、检索 (Retrieval) 和生成 (Generation)。
- **推荐阅读**：想要快速直观地了解 RAG 的基础概念与工作流，建议参考 [IBM: 什么是检索增强生成 (RAG)？](https://www.ibm.com/cn-zh/topics/retrieval-augmented-generation) 或者流行的 AI 框架文档，如 [Huggingface 的 RAG 概念指南](https://huggingface.co/blog/ngxson/make-your-own-rag)。
- **推荐阅读**：人工智能与深度学习部分的知识，推荐通过阅读经典教程如李沐老师的 [动手学深度学习 (Dive into Deep Learning)](https://zh.d2l.ai/) 以及其注意力机制相关章节，来快速理解信息检索和 AI 深度学习的基础原理。


## 初次实践
### Crater容器的创建
- 参考[Crater官方文档](https://raids-lab.github.io/crater/zh/docs/user/)

### RAG-Anything：云端 API 体验
在使用本地 GPU 构建复杂的 RAG 系统前，建议先通过调用云端 API 体验 RAG 的整体 pipeline。
- 初步尝试：在你的本地（PC）尝试使用 API Token 搭建一个 RAG 工作流，理解数据流在文档处理、向量搜索到最后大语言模型生成中的流转：
[RAG-Anything](https://github.com/HKUDS/RAG-Anything)

## 基于 Ayo 的 RAG 部署与优化
### 快速上手指南
本项目（基于 [Ayo](https://github.com/NetX-lab/Ayo) 框架）旨在帮助你从零开始在服务器/端侧上部署一套完整且经过高度优化的本地 RAG 系统。该系统也类似于[RAG-Anything](https://github.com/HKUDS/RAG-Anything)，提供了一套自定义的框架来帮助开发者搭建RAG工作流，同时更加高效地对其进行服务化部署。我们将一步步指导你完成操作：

#### 1. 准备环境与安装向量数据库
RAG 系统通常需要向量数据库来存储文档。本项目使用 PostgreSQL 搭配 `pgvector` 插件作为向量数据库。
执行以下命令安装依赖并配置数据库（也可参考并执行 `tools/pg_install.sh`）：

```bash
# 获取 pgvector 源码
git clone https://github.com/pgvector/pgvector.git 
cd pgvector

# 安装系统依赖
sudo apt update
sudo apt install build-essential libpq-dev postgresql-server-dev-all postgresql postgresql-contrib

# 编译并安装 pgvector
make PG_CONFIG=$(which pg_config)
sudo make install

# 确保 PostgreSQL 服务已启动
sudo service postgresql start

# 创建数据库用户并开启 vector 扩展
sudo -u postgres psql -c "CREATE USER asplos25 WITH PASSWORD '123456';"
sudo -u postgres psql -c "CREATE DATABASE database_asplos OWNER asplos25;"
sudo -u postgres psql database_asplos -c "CREATE EXTENSION vector;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE database_asplos TO asplos25;"
cd ..
```

#### 2. 安装框架与运行依赖
基于Crater镜像中已有的vLLM 0.8.5依赖，以下是一些额外需要安装的 Python 包：
```bash
sudo pip install asyncpg, pgvector, matplotlib, sentence_transformers
```



#### 3. 准备开源模型权重
全本地化的部署需要我们在本地加载大模型、向量模型以及重拍序模型。项目中提供了一个基于国内镜像的快速下载脚本。
运行命令下载常用模型（如 BAAI 的各类模型）：
```bash
python tools/download_hf_model.py
```

> 在Crater创建的容器中，已经预先下载好了 `Qwen2.5-0.5B` 模型权重，并放置在 `/data/Models/Qwen/Qwen2.5-0.5B/` 目录下。你也可以通过上述脚本下载到本地。

#### 4. 运行优化后的端到端 RAG 系统
- 一切准备就绪后，可以通过执行预先写好的 example，来体验经过依赖修剪、并行和 Pipeline 等优化的 RAG 工作流：
```bash
python optimized_embedding_ingestion_searching_reranking_llm.py
```
## 下一步
- 你可以尝试修改实例中的模型配置，来体验不同模型（如 Qwen2.5-0.5B）在 RAG 工作流中的表现，或者修改`num_cpu`, `num_gpu`等资源配置，来观察不同资源分配对稳定性、系统性能的影响。
- 进一步观察和分析实例中的Optimization策略（如依赖修剪、并行和 Pipeline）对系统性能的提升效果。
- 了解云计算、serverless的概念，为下一步进行迁移部署做好准备。


## 参考文献
```bibtex
@inproceedings{tan2025ayo,
title = {Towards End-to-End Optimization of LLM-based Applications with Ayo},
author = {Xin Tan and Yimin Jiang and Yitao Yang and Hong Xu},
booktitle = {Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
url = {https://doi.org/10.1145/3676641.3716278},
doi = {10.1145/3676641.3716278},
year = {2025}
}

@article{tan2024teola,
  title={Teola: Towards end-to-end optimization of llm-based applications},
  author={Tan, Xin and Jiang, Yimin and Yang, Yitao and Xu, Hong},
  journal={arXiv preprint arXiv:2407.00326},
  year={2024}
}
```