# 深入推理过程说明（针对 `optimized_embedding_ingestion_searching_reranking_llm.py`）

本文档详细描述示例脚本中构建的推理流水线（Embedding → Ingestion → QueryEmbedding → Search → Reranking → LLM Prefill → LLM Decoding）的执行过程、各节点职责、引擎交互、异步与批处理机制，以及优化 pass 对延迟/吞吐的影响。

**整体目标**
- 构建并运行一个基于 DAG 的检索增强生成（RAG）流程：先对文档做 embedding 并入库，再对 query 做 embedding 并检索候选文档，接着对候选进行 rerank，最后用 LLM 基于检索结果生成回答（prefill + decoding）。

---

**脚本主要部分回顾（高层）**
- `create_base_dag()`：构建 DAG 节点（`Node`），定义每个节点的 `io_schema`、`engine_type`、`op_type`、`anno`（batchable/splittable）与 config（如 `top_k`、prompt 等），并连成链路。
- `run_app(dag)`：
  - 初始化 `APP`（`APP.init()`，内部会启动 Ray 并创建 `GraphScheduler` actor）；
  - 通过 `app.register_engine(...)` 注册 embedding、vector_db、reranker、llm 的引擎配置；
  - `app.start()` 启动后台队列与监控任务；
  - 创建并并发提交示例 Query（通过 `app.submit_query()`），等待并打印返回结果；
  - 最后负责清理（`app.stop()`、`app.shutdown()`）。

---

## 推理数据流与控制流（逐步）

1. 客户端组装 `Query` 对象
   - `Query` 包含线程安全的 DAG 副本（deepcopy(dag)）与 `query_inputs`（在脚本中包含 `passages` 与 `queries`）。
   - 调用 `APP.submit_query(query)`：返回一个 `asyncio.Future` 给调用者，用于最终获取结果。

2. APP 入队与优化
   - `APP.submit_query()` 把 `QueryTask` 放入 `app.query_queue`，并记录到 `active_queries`。
   - `APP` 的后台 `_process_queue()` 循环取出任务并创建处理任务 `_handle_query()`。
   - `_handle_query()` 在 `query_semaphore` 控制下并发运行：调用 `optimize_dag()` 在线程池中执行 DAG 优化（`dag.optimize(enabled_passes)`），这会按配置的 `opt_pass` 顺序（例如 `PrefillingSpiltPass`、`StageDecompositionPass` 等）对 DAG 做结构改造以提升并行度或减少延迟。**（这里是在线的DAG优化阶段，之前离线阶段也有对DAG的优化）**

1. GraphScheduler 接管执行计划（详细）
   - **查询提交**：优化后，APP 调用 `graph_scheduler.submit_query.remote(query, config)`，将 `Query` 对象（包含优化后的 DAG）提交给 `GraphScheduler`（Ray actor）。
   
   - **QueryRunner 初始化**：
     - GraphScheduler 为每个查询创建一个 `QueryRunner` 实例，负责该查询的完整执行生命周期。
     - `QueryRunner.initialize()` 进行以下初始化：
       - 执行 DAG 的拓扑排序（`dag.topological_sort()`），确定节点执行顺序。
       - 将节点按类型分类：`input_nodes`（输入节点）、`compute_nodes`（计算节点）、`output_nodes`（输出节点）。
       - 标记所有 `input_nodes` 为 `COMPLETED` 状态，并将其 `input_values` 写入 `nodes_outputs`（结果存储）。
       - 将 `compute_nodes` 和 `output_nodes` 放入 `pending` 队列等待调度。
   
   - **DAG 执行循环**（`_process_dag`）：
     - GraphScheduler 启动异步任务 `_process_dag(runner)`，持续循环直到 DAG 完成（`dag.check_completion()`）。
     - 每轮迭代执行以下步骤：
       
       **(a)检查就绪节点**：
       - 遍历 `pending` 队列，调用 `check_node_ready(node)` 检查每个节点是否就绪。
       - 就绪条件根据节点类型不同：
         - 普通计算节点：所有父节点的 `status == COMPLETED`。
         - `LLM_DECODING` / `LLM_PARTIAL_DECODING`：对应的 prefilling 节点为 `RUNNING` 或 `COMPLETED` 即可（支持流水线）。
         - `LLM_FULL_PREFILLING`：要求 partial prefilling 父节点至少 `RUNNING`，其他父节点 `COMPLETED`。
       - 就绪的节点从 `pending` 移到 `ready` 队列。
       
       **(b)批量提交节点**：
       - 从 `ready` 队列取出所有节点，调用 `asyncio.gather(*[runner.submit_node(node) for node in nodes_to_submit])` 并行提交。
       - 提交后的节点移入 `running` 队列。
       
       **(c) 监控运行中节点**：
       - 检查 `running` 队列中的节点状态（通过 `query.query_state.get_node_status.remote(node.name)`）。
       - 若节点状态变为 `COMPLETED`，则：
         - 调用 `query_state.get_node_result.remote(node.name)` 获取结果并存入 `nodes_outputs[node.name]`。
         - 更新节点 `status = COMPLETED`，从 `running` 移除。
       - 若节点 `FAILED`，则标记为错误并终止 DAG 执行。
       
       **(d) 睡眠并重复**：
       - `await asyncio.sleep(0.01)` 后进入下一轮循环。
   
   - **节点提交详细流程**（`submit_node`）：
     - **INPUT 节点**：直接跳过（已在初始化时完成）。
     - **OUTPUT 节点**：从父节点复制结果到 `nodes_outputs`，标记 `COMPLETED`。
     - **AGGREGATOR 节点**（特殊计算节点）：
       - 在 GraphScheduler 内部直接处理（不提交给引擎），支持 `DUMMY`（无操作）、`MERGE`（合并字典/列表）、`TOP_K`（按 score 排序取前 k）等模式。
       - 完成后直接调用 `query_state.set_node_result.remote(node.name, result)`。
     - **普通 COMPUTE 节点**：
       1. 调用 `node.update_input_kwargs(nodes_outputs)` 从已完成的父节点获取输入数据。
       2. 调用 `prepare_engine_payload(node)` 准备引擎请求的 payload：
          - 使用 `TRANSFORMER_REGISTRY` 查找该 `engine_type` 的 transformer（如 `EmbeddingTransformer`、`VectorDBTransformer` 等）。
          - Transformer 将 `node.input_kwargs` 转换为引擎特定的参数格式（如 embedding 引擎需要 `texts`，reranker 需要 `query` + `passages`）。
       3. 创建 `EngineRequest` 对象：
          - `request_id = f"{query_id}::{node.name}"`（后续用于结果回传）
          - `payload` 为 transformer 生成的字典
       4. 获取对应 `engine_type` 的 `EngineScheduler` actor handle。
       5. 调用 `await scheduler.submit_request.remote(engine_request)` 提交到引擎调度器。
       6. 更新节点 `status = RUNNING`。

2. Embedding 节点（文档/查询 embedding）
   - `EmbeddingEngine`（Ray actor）：
     - 接收 `submit_request(request_id, query_id, texts)`；把请求放入 `request_queue`。
     - `_get_next_batch()` 将队列中的请求聚合（力求把同一 query 的请求合并优先以保持数据局部性），收集到 `batch_texts`，再放入 `batch_queue`。
     - `_process_batches()` 调用 `_compute_embeddings`（由 sentence-transformers 模型在 GPU/CPU 上进行批量推理），得到 numpy embeddings 并通过 `ray.put` 生成 `ObjectRef`，随后调用调度器的回调 `scheduler_ref.on_result` 把结果回传给调度器（GraphScheduler）。
   - 要点：
     - 批处理通过 `max_batch_size` 提升吞吐；
     - 对同一 query 的请求有聚合优先策略，降低跨 query 的混合；
     - 结果通过 Ray object store（`ray.put`）在 actor 间传递以避免序列化开销。

3. VectorDB 节点（Ingestion 与 Search）
   - `VectorDBEngine` 使用 `asyncpg` 与 `pgvector`：
     - Ingestion: 收到 embeddings 与文本后按 `query_id` 分组，将向量插入为单独的 PostgreSQL 表（表名基于 `query_id` 的 md5 哈希）并创建 HNSW 索引。
     - Search: 接收一组 `query_vectors`，在对应表上按向量相似度（pgvector 的 `<=>`）排序并返回 top-k 文档（转换为相似度 score）。
   - 要点：
     - 每个 query 使用独立表降低多租户或临时索引冲突；
     - 插入与查询使用不同队列（insert_queue / search_queue），各自有 batch 与异步循环；
     - 插入后通过 scheduler 的回调告知完成（通常下游节点依赖 `index_status`）。

4. Reranker 节点
   - `RerankerEngine` 使用 CrossEncoder（sentence-transformers 的 CrossEncoder）：
     - 接收 `submit_request(request_id, query_id, query, passages, top_k)`。
     - Batch 策略可选：`request`（逐请求批处理）或 `pair`（按 query-passage pair 批处理以提高模型并行度）。
     - 对传入的 passage 列表进行交叉编码打分，按 score 排序并返回 top_k（通过 `ray.put` 放入 object store 并用 `scheduler_ref.on_result` 回调）。
   - 要点：
     - Cross-encoder 精度高但计算量大，通常在 `top_k` 较小的候选上使用；
     - `pair` 策略有利于将更多 query-passage 对组合在一起做批量计算以提高吞吐，但需要管理部分结果累积逻辑。

5. LLM 节点（Prefilling 与 Decoding）
   - `LLMEngine` 基于 `vllm.AsyncLLMEngine`：
     - 支持多种操作：`LLM_PARTIAL_PREFILLING`（partial prefill）、`LLM_FULL_PREFILLING`（full prefill）、`LLM_DECODING`、以及 `LLM_PARTIAL_DECODING`。
     - Prefill 流程：把 prompt chunk(s) 提交到 vLLM 引擎的预填充接口。为了支持流水线与并发，框架将预填充拆为两个阶段：可提前计算的部分（partial）与需要最终上下文的部分（full）。
     - Decoding 流程（生成）：当 decoding 请求到来时，LLM engine 读取对应 `llm_internal_id` 下的流式生成器（`result_generator_tracker`），迭代结果流。若配置为 partial decoding，则返回部分输出即可。
   - 回传：生成完成或出现可用的部分结果后，通过 `ray.put` 或 `scheduler_ref.on_result` 把结果回送给 GraphScheduler。
   - 要点：
     - 通过维护生成流与局部状态（`partial_results`），支持把较长的 prompt 分段预填充并行处理；
     - 这对 RAG 场景有益——先把与检索无关或可提前计算的 prompt 部分预填充（partial），减小实时依赖的工作量。

6. EngineScheduler 详细工作机制
   - **职责**：`EngineScheduler` 是每类引擎（如 embedding、vector_db、reranker、llm）的资源管理与负载均衡器，一个 Ray actor 管理该类引擎的多个实例。
   
   - **初始化**（`__init__`）：
     - 接收 `EngineConfig`（包含引擎类、实例数、GPU/CPU 资源配置、模型配置等）。
     - 调用 `_create_engines()` 创建 `instances` 个引擎实例（Ray actors）：
       - 根据 `num_gpus`、`num_cpus`、`resources` 配置 Ray remote options。
11      - 将当前 EngineScheduler 的 actor handle 作为 `scheduler_ref` 传给每个引擎实例。
       - 等待每个引擎的 `is_ready()` 确认初始化完成（模型加载、warm-up等）。
     - 创建三个核心队列与任务：
       - `request_queue`：接收来自 GraphScheduler 的请求。
       - `pending_requests`：追踪已提交但未完成的请求。
       - `complete_queue`：存储引擎回传的完成结果。
       - 启动两个异步任务：`_submit_requests()` 和 `_process_results()`。
   
   - **请求提交流程**（`submit_request` 被 GraphScheduler 调用）：
     1. `EngineScheduler.submit_request(engine_request)` 将请求放入 `request_queue`。
     2. 后台任务 `_submit_requests()` 循环：
        - 从 `request_queue` 取出请求（异步等待，超时 0.05 秒避免空转）。
        - 调用调度策略 `_get_next_engine()`（默认 `RoundRobinStrategy`）选择一个引擎实例。
        - 调用 `await engine.submit_request.remote(request_id, query_id, **payload)`，将请求提交给选中的引擎 actor。
        - 将请求存入 `pending_requests[request_id]` 等待结果。
   
   - **结果回调处理**（`on_result` 被引擎 actor 调用）：
     1. 引擎完成计算后调用 `scheduler_ref.on_result.remote(request_id, query_id, result_ref)`。
     2. `EngineScheduler.on_result()` 执行：
        - 确保 `result_ref` 是 Ray ObjectRef（若不是则用 `ray.put` 包装）。
        - 从 `pending_requests` 中找到对应请求。
        - 将 `result_ref` 附加到请求的 `result_ref` 字段。
        - 将请求移入 `complete_queue`。
        - 从 `pending_requests` 中删除该请求。
   
   - **结果处理任务**（`_process_results`）：
     1. 循环从 `complete_queue` 取出完成的请求。
     2. 使用 `ray.get(result_ref)` 获取实际结果（在 executor 中执行避免阻塞事件循环）。
     3. 解析 `request_id`（格式为 `{query_id}::{node_name}`），提取节点名。
     4. 调用 `query.query_state.set_node_result.remote(node_name, result_ref)` 将结果写入查询状态。
     5. 从 `pending_requests` 清理该请求。
   
   - **动态扩缩容**（可选）：
     - `add_engine()`：动态添加新引擎实例。
     - `remove_engine(index)`：移除指定引擎实例并调整调度索引。
   
   - **关闭流程**（`shutdown`）：
     - 停止两个后台任务。
     - 调用所有引擎实例的 `shutdown.remote()`。
     - 清空引擎列表。

1. 完整调度链路图示
   ```
   APP.submit_query(query)
     └─> GraphScheduler.submit_query(query)
          └─> 创建 QueryRunner(query, dag, engine_schedulers)
               └─> QueryRunner.initialize()
                    ├─> 拓扑排序 DAG
                    ├─> 初始化 input_nodes（状态 COMPLETED）
                    └─> pending ← [compute_nodes, output_nodes]
               
               └─> QueryRunner._process_dag() 循环
                    ├─> 检查 ready 节点（check_node_ready）
                    ├─> submit_node(node) 并行提交
                    │    ├─> node.update_input_kwargs(nodes_outputs)
                    │    ├─> payload = prepare_engine_payload(node)  // 使用 Transformer
                    │    ├─> engine_request = EngineRequest(request_id, query_id, payload)
                    │    └─> EngineScheduler.submit_request(engine_request)
                    │         └─> request_queue.put(engine_request)
                    │
                    └─> 监控 running 节点状态，收集结果到 nodes_outputs
   
   EngineScheduler._submit_requests() 后台任务
     └─> 循环从 request_queue 取请求
          ├─> engine = _get_next_engine()  // RoundRobin 策略
          ├─> engine.submit_request.remote(request_id, query_id, **payload)
          └─> pending_requests[request_id] = request
   
   Engine (如 EmbeddingEngine)
     └─> 批处理 + 模型推理
          └─> scheduler_ref.on_result.remote(request_id, query_id, result_ref)
   
   EngineScheduler.on_result(request_id, query_id, result_ref)
     └─> request.result_ref = result_ref
          └─> complete_queue.put(request)
   
   EngineScheduler._process_results() 后台任务
     └─> 循环从 complete_queue 取完成请求
          ├─> result = ray.get(result_ref)
          ├─> node_name = request_id.split("::")[-1]
          └─> query.query_state.set_node_result.remote(node_name, result_ref)
   
   QueryRunner 的 _process_dag 循环
     └─> 检测到 node 状态变为 COMPLETED
          ├─> result = query_state.get_node_result(node_name)
          ├─> nodes_outputs[node_name] = result
          └─> 触发下游依赖节点进入 ready 队列
   
   DAG 完成后
     └─> query.status = QueryStatus.COMPLETED
          └─> APP._monitor_query_status() 检测到完成
               └─> query_task.future.set_result(node_results)
                    └─> 返回给用户
   ```

2.  关键设计要点

   - **三级队列架构**：
     - GraphScheduler 的 `pending` / `ready` / `running` 队列管理节点生命周期。
     - EngineScheduler 的 `request_queue` / `pending_requests` / `complete_queue` 管理请求生命周期。
     - Engine 内部的 `request_queue` / `batch_queue` 管理批处理。
   
   - **异步回调设计**：
     - 引擎不主动 pull 结果，而是通过 `scheduler_ref.on_result()` 回调推送结果。
     - EngineScheduler 作为中介，解耦引擎与 DAG 执行逻辑。
   
   - **负载均衡**：
     - EngineScheduler 通过调度策略（RoundRobin 等）在多个引擎实例间分配请求。
     - 未来可扩展为基于负载、延迟的动态调度。
   
   - **状态同步**：
     - 所有节点状态与结果通过 `query_state`（Ray actor）集中管理，保证分布式一致性。
     - GraphScheduler 轮询 `query_state` 检测节点完成，而非直接等待结果（解耦）。
   
   - **流水线优化**：
     - LLM 的 prefill/decoding 节点特殊处理：decoding 不必等 prefill 完全完成，只需 `RUNNING` 状态即可开始（支持 chunk-based 流水线）。

3. 结果收集与返回
   - 当 DAG 中所有叶子节点或输出节点完成后，GraphScheduler 把输出写入 `Query.query_state`，APP 的监视器 `_monitor_query_status` 会检测 `QueryStatus.COMPLETED`，随后用 `query_task.future.set_result(node_results)` 将结果返回给初始调用者。
   - APP 还维护超时与错误处理路径（`handle_timeout`, `handle_error`），并更新指标（如 avg latency、成功/失败计数）。

---

## 优化 Pass 对推理的影响（举例）

- `PrefillingSpiltPass`：把 LLM 的 prefill 分成 partial/full。效益：
  - 对于 RAG prompt，部分上下文（如 system prompt / instructions / document-independent模板片段）可能不依赖于检索结果或用户 query 中的某些字段，提前计算可缩短用户感知延迟；
  - 分段预填充允许把计算密集型预处理与在线依赖的部分分离，从而并行化 I/O（检索）与计算（prefill）。

- `StageDecompositionPass` / `LLMDecodingPipeliningPass` 等：
  - 把单个大步骤拆成多个阶段（分片/并行解码），增加流水线并发度，提升吞吐但可能增加总资源占用。

- `PruningDependencyPass`：裁剪 DAG 中与当前查询无关或冗余依赖，减少不必要计算。

综合影响：优化 pass 的目标通常在“降低尾延迟”和“提高吞吐”之间权衡。分离可提前计算的任务（partial prefill）对交互式延迟尤为有利；而把大量计算拆为并行子任务能提升总体吞吐。

---

## 关键实现与注意事项（工程角度）

- Ray 与异步混用：引擎为 Ray actor，但自身执行大量 asyncio 任务（批处理/队列）。通信多通过 `ray.get` / `ray.put` 以及 actor remote 调用回调。
- 数据传输效率：大量使用 `ray.put` 将中间大对象（embeddings、reranker scores、生成文本）放入 object store，避免重复序列化。
- 多租户/多 query 表设计：`VectorDBEngine` 为每个 `query_id` 创建独立表，方便隔离但需注意数据库表数量增长与清理策略。
- 批处理策略：embedding 与 reranker 都设计了复杂的 batching 策略（聚合同一 query 的请求、或把 query-passage pair 批量化），需要权衡延迟与吞吐。
- 错误与超时处理：APP 层统一监控查询状态并在超时或失败时清理、上报错误使得调用者可以可靠得到异常信息。
- Prompt 模板管理：脚本中通过 `replace_placeholders` 把占位符替换为实际字段名（例如把 prompt 中 `{question}` 替换为 `{queries}`），与 `PrefillingSpiltPass` 的占位符分析互补。

---

## 在本示例脚本中可观察的运行行为（实践）
- 启动：`APP.init()` 触发 Ray 初始化；`register_engine()` 会创建 `EngineScheduler`（remote actor），并等待其就绪。
- Warm-up：embedding、reranker 等引擎在初始化时会对模型做一次 warm-up（小 batch 的推理）。
- 并发查询：脚本用 `asyncio.gather` 并发提交查询，APP 的 `query_semaphore` 控制同时处理的上限。
- 日志与度量：APP 与引擎中均有日志输出（`get_logger`），APP 维护 `metrics`（总查询、超时、平均延迟等）。

---

## 建议与扩展方向
- 清理 VectorDB 表：当前为每个 `query_id` 创建表，需添加生命周期管理以避免长期积累。
- 动态资源调度：`EngineScheduler` 可根据负载与延迟动态扩容/缩容 engine instances。
- 更细粒度的 backpressure：在 APP 层与 engine 层之间引入更丰富的 backpressure 策略，避免队列过载。
- 增加 end-to-end 测试与基准脚本，量化优化 pass 的收益（延迟/吞吐 trade-off）。

---

文档已写入仓库：`DETAILED_INFERENCE.md`。如需我把其中某一节进一步展开为时序图、调用图或把部分伪代码改为实际的 sequence diagram，我可以继续补充。