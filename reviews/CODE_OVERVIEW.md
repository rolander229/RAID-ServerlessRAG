# Ayo 项目代码速览

此文档对仓库中主要模块做快速说明，帮助理解各组件如何协同工作。

**概览**
- **Purpose**: 框架用于构建基于 DAG 的检索增强生成（RAG）流水线，结合 embedding、向量库、reranker、LLM 等引擎进行搜索与生成。
- **Entry scripts**: 根目录下有若干示例脚本（如 `optimized_embedding_ingestion_searching_reranking_llm.py`）用于构建与运行示例 DAG。

**应用入口**
- **APP**: [Ayo/app.py](Ayo/app.py) : 框架主入口。
  - **职责**: 管理引擎注册、启动/停止、接收并调度查询、执行 DAG 优化、以及监控查询状态与统计指标。
  - **关键函数**: `init()` 初始化（含 Ray），`register_engine()` 注册引擎并创建引擎调度器，`submit_query()` 接收 Query，`start()`/`stop()` 控制运行，内部通过 `graph_scheduler` 提交执行。

**配置**
- **EngineConfig / AppConfig**: [Ayo/configs/config.py](Ayo/configs/config.py)
  - **职责**: 轻量数据类，封装引擎与应用的配置信息（模型参数、资源、延迟配置等）。

**DAG 与节点**
- **DAG**: [Ayo/dags/dag.py](Ayo/dags/dag.py)
  - **职责**: 管理节点集合、建立拓扑、生成输入/输出节点、执行拓扑排序、应用优化 passes（`opt_pass`）。
  - **重要能力**: `register_nodes()`、`topological_sort()`、`optimize()`、`collect_outputs()` 等。
- **Node**: [Ayo/dags/node.py](Ayo/dags/node.py)
  - **职责**: 表示 DAG 中的单个计算元（或输入/输出节点），包含 IO schema、父子依赖、状态、分片与批处理元数据。
  - **关键属性**: `input_kwargs`、`output_names`、`input_shards_mapping`、`is_ready` 判定等。
- **Node commons**: [Ayo/dags/node_commons.py](Ayo/dags/node_commons.py)
  - **包含**: `NodeStatus`、`NodeAnnotation`、`NodeOps`、`NodeType`、`NodeIOSchema` 等枚举与数据结构。

**引擎实现（Ray Actor / 异步）**
- **BaseEngine**: [Ayo/engines/base_engine.py](Ayo/engines/base_engine.py)
  - **职责**: 抽象异步请求、队列与批处理的基本流程；子类需实现模型加载、提交请求、取批与处理批逻辑。
- **Engine Registry**: [Ayo/engines/engine_types.py](Ayo/engines/engine_types.py)
  - **职责**: 全局注册表，维护引擎类型到实现类与默认配置的映射（`ENGINE_REGISTRY`）。
- **EmbeddingEngine**: [Ayo/engines/embedder.py](Ayo/engines/embedder.py)
  - **职责**: 使用 sentence-transformers 加载 embedding 模型，接收文本批量请求，合并按查询分组并返回 embeddings（通过 Ray ObjectRef 通信）。
  - **特点**: 支持 batching、按 query 聚合请求、在 GPU/CPU 上 warm-up。
- **VectorDBEngine**: [Ayo/engines/vector_db.py](Ayo/engines/vector_db.py)
  - **职责**: 使用 `asyncpg` 与 `pgvector` 将向量写入 PostgreSQL 表并执行向量检索。
  - **实现细节**: 每个 query_id 使用独立表名（哈希生成），分别维护插入队列与搜索队列，批量插入与分组处理搜索请求。
- **RerankerEngine**: [Ayo/engines/reranker.py](Ayo/engines/reranker.py)
  - **职责**: 使用 CrossEncoder（sentence-transformers）对候选段落按 query 进行交叉编码打分，支持两种 batch 策略（`request` 与 `pair`）。
  - **输出**: 返回按 score 排序的 top-k 结果（Ray ObjectRef）。
- **LLMEngine**: [Ayo/engines/llm.py](Ayo/engines/llm.py)
  - **职责**: 基于 `vllm` 提供异步生成能力，支持分步 prefill（partial/full）与解码（decoding / partial decoding）。
  - **特性**: 维护生成流（generator）以支持部分预填充与并发的部分解码请求，结果通过调度器回调上报。

**模块（高层逻辑 -> 原语节点）**
- **BaseModule**: [Ayo/modules/base_module.py](Ayo/modules/base_module.py)
  - **职责**: 模块抽象，子类需实现 `to_primitive_nodes()` 将复合模块拆成 DAG 原语节点（`Node`）。
- **Embedding / Searching / Reranking 等模块**: [Ayo/modules/embedding.py](Ayo/modules/embedding.py), [Ayo/modules/searching.py](Ayo/modules/searching.py), [Ayo/modules/reranking.py](Ayo/modules/reranking.py)
  - **职责**: 将常用功能（文本 embedding、向量检索、reranking 等）封装成模块，返回若干 `Node` 以便组装到 DAG。
- **PromptTemplate**: [Ayo/modules/prompt_template.py](Ayo/modules/prompt_template.py)
  - **职责**: 提供常见 prompt 模板与占位符替换工具（如 RAG 模版、query expanding 等），方便 LLM 节点配置 prompt。

**优化 Pass（opt_pass）**
- **总体**: [Ayo/opt_pass](Ayo/opt_pass) 下包含若干优化策略，DAG 在提交执行前会应用用户启用的 passes。
- **Prefilling Split**: [Ayo/opt_pass/prefilling_split.py](Ayo/opt_pass/prefilling_split.py)
  - **目的**: 将 LLM 的 prefilling 拆成 partial-prefill 与 full-prefill，提前执行不依赖问题（或可提前计算）的那部分以减少总体延迟。
  - **实现**: 解析 prompt template 中的占位符，按占位符位置切分模板，创建 `node-partial` 与 `node-full` 并接回 DAG，移除原始 prefilling 节点。
- **其他 passes**: `stage_decomposition.py`、`decoding_pipeling.py`、`pruning_dependency.py` 等用于分解阶段、流水线解码与裁剪无关依赖，提升并行度与吞吐。

**调度器**
- **GraphScheduler / EngineScheduler**: 位于 [Ayo/schedulers](Ayo/schedulers)
  - **职责**: `EngineScheduler` 管理单种引擎的实例/队列（在 Ray Actor 中），`GraphScheduler` 负责将 DAG 的执行计划下发到各个 engine-scheduler 并收集结果（分布式执行协调）。

**查询与状态**
- **Query / QueryState**: [Ayo/queries/query.py](Ayo/queries/query.py)、[Ayo/queries/query_state.py](Ayo/queries/query_state.py)
  - **职责**: 封装单次查询的元数据、输入、DAG 副本、执行状态与最终结果缓存，以便跨 actor / 调度器协作追踪。

**工具与日志**
- **Logger/Utils**: [Ayo/logger.py](Ayo/logger.py)、[Ayo/utils.py](Ayo/utils.py)
  - **职责**: 提供统一日志配置与一些内部帮助函数。

**示例代码**
- 根目录 `optimized_embedding_ingestion_searching_reranking_llm.py`：示例程序，构建完整 DAG（embedding -> ingestion -> query embed -> search -> rerank -> llm prefill -> decoding），注册各引擎配置并运行 APP。

**数据流（简要）**
- 客户端创建 `Query`，填充 query inputs 并调用 `APP.submit_query()`。
- `APP` 在后台对 DAG 执行优化（可应用 `opt_pass`），随后将查询提交给 `GraphScheduler`。
- `GraphScheduler` 按 DAG 的依赖下发任务到不同类型的 `EngineScheduler`（各自对应引擎类型的 Ray actor）。
- 每种引擎按内部队列/批处理策略执行任务，完成后通过 scheduler 回调把结果写回并在需要时触发下游节点。
- 最终 `GraphScheduler` 收集输出节点结果，APP 通过 Query 的 `future` 返回结果给调用者。

---

如需我生成更详细的每个文件逐行说明（或将此文档拆成多节并加上调用图），我可以继续扩展或运行静态分析并生成可视化 DAG。