from typing import List, Union
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node, NodeAnnotation, NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.modules.prompt_template import replace_placeholders, RAG_QUESTION_ANSWERING_PROMPT_TEMPLATE_STRING

def design_dag_from_monolith():
    # 1. 初始化 DAG
    dag = DAG(dag_id="monolith_to_dag_rag")

    # 2. 定义【特征提取】节点 (对应 MonolithicRAG.ingest 和 search 中的 encode)
    # 拆分原因：GPU 密集型，适合大规模批处理
    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str]},
            output_format={"passages_embeddings": List[List[float]]}
        ),
        op_type=NodeOps.EMBEDDING,
        anno=NodeAnnotation.BATCHABLE
    )

    # 3. 定义【数据入库】节点 (对应 MonolithicRAG.ingest 中的 copy_records_to_table)
    # 拆分原因：I/O 密集型，数据库连接管理
    ingestion_node = Node(
        name="Ingestion",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str], "passages_embeddings": List[List[float]]},
            output_format={"index_status": bool}
        ),
        op_type=NodeOps.VECTORDB_INGESTION
    )

    # 4. 定义【向量检索】节点 (对应 MonolithicRAG.search 中的 fetch)
    # 拆分原因：数据库索引查询，可独立扩展实例提高 QPS
    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={"queries_embeddings": List[List[float]], "index_status": bool},
            output_format={"search_results": List[List[str]]}
        ),
        op_type=NodeOps.VECTORDB_SEARCHING,
        config={"top_k": 20}
    )

    # 5. 定义【重排序】节点 (对应 MonolithicRAG.rerank)
    # 拆分原因：Cross-Encoder 极其耗时，通常需要与检索节点分开，甚至可以分解(Decomposition)
    reranking_node = Node(
        name="Reranking",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        io_schema=NodeIOSchema(
            input_format={"queries": str, "search_results": List[str]},
            output_format={"reranked_results": List[str]}
        ),
        op_type=NodeOps.RERANKING,
        config={"top_k": 2}
    )

    # 6. 定义【LLM 推理】节点 (对应 MonolithicRAG.generate)
    # 拆分原因：显存大户，通常拆分为 Prefilling (首字) 和 Decoding (增量) 以支持流水线并行
    llm_node = Node(
        name="LLM_Generation",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"queries": List[str], "reranked_results": List[List[str]]},
            output_format={"answer": str}
        ),
        op_type=NodeOps.LLM_GENERATE,
        config={
            "prompt_template": replace_placeholders(
                RAG_QUESTION_ANSWERING_PROMPT_TEMPLATE_STRING,
                question="queries",
                context="reranked_results"
            )
        }
    )

    # 7. 构建拓扑关系
    # (Embedding -> Ingestion) -> Search -> Reranking -> LLM
    embedding_node >> ingestion_node >> search_node >> reranking_node >> llm_node

    # 注册节点到 DAG
    dag.register_nodes(embedding_node, ingestion_node, search_node, reranking_node, llm_node)
    
    return dag
