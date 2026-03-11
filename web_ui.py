import os
import sys
import asyncio
import streamlit as st
import traceback
from typing import List

# 确保可以导入当前目录和 Ayo 库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_embedding_ingestion_searching_reranking_llm import create_base_dag, run_app
from Ayo.vis.vis_graph_interactive import visualize_dag_interactive
import streamlit.components.v1 as components
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType

# K8s 环境适配：允许通过环境变量注入数据库连接信息，而不是硬编码 localhost
# db_host = os.environ.get("POSTGRES_HOST", "localhost")
# db_port = int(os.environ.get("POSTGRES_PORT", 5432))
# vectordb_config.model_config["host"] = db_host
# vectordb_config.model_config["port"] = db_port

from Ayo.opt_pass.decoding_pipeling import LLMDecodingPipeliningPass
from Ayo.opt_pass.prefilling_split import PrefillingSpiltPass
from Ayo.opt_pass.pruning_dependency import PruningDependencyPass
from Ayo.opt_pass.reranking_decomposition import RerankingDecompositionPass
from Ayo.opt_pass.stage_decomposition import StageDecompositionPass

st.set_page_config(layout="wide", page_title="Ayo DAG Optimizer & Runner")

st.title("Ayo DAG Optimizer & App Runner")

output_dir = "./.output_cache"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_monolithic_rag_dag() -> DAG:
        """Create a single-node DAG that represents monolithic RAG."""
        monolithic_dag = DAG(dag_id="monolithic_rag_single_node", name="Monolithic RAG")

        monolithic_node = Node(
                name="MonolithicRAG",
                node_type=NodeType.COMPUTE,
                engine_type=EngineType.DUMMY,
                io_schema=NodeIOSchema(
                        input_format={"passages": List[str], "queries": List[str]},
                        output_format={"answer": str},
                ),
                op_type=NodeOps.LLM_FULL,
                config={
                        "description": "Embedding + ingestion + search + rerank + generation in one node"
                },
        )

        monolithic_dag.set_query_inputs(
                {
                        "passages": [
                                "AI is a branch of computer science...",
                                "Machine learning is a subset of AI...",
                                "OSDI is a conference about systems...",
                        ],
                        "queries": ["I want to know some system conferences."],
                }
        )
        monolithic_dag.register_nodes(monolithic_node)
        return monolithic_dag


def write_monolithic_to_dag_story_html(output_path: str) -> str:
        """Write a lightweight HTML card showing how monolithic RAG expands into a DAG."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
            background: #f8fafc;
            color: #0f172a;
        }
        .wrap {
            padding: 16px;
        }
        .title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .row {
            display: flex;
            align-items: center;
            gap: 12px;
            overflow-x: auto;
            padding-bottom: 8px;
        }
        .node {
            min-width: 140px;
            text-align: center;
            border: 2px solid #1d4ed8;
            border-radius: 10px;
            padding: 10px 8px;
            background: #dbeafe;
            font-size: 13px;
            font-weight: 600;
        }
        .mono {
            border-color: #7c3aed;
            background: #ede9fe;
            min-width: 180px;
        }
        .arrow {
            font-size: 20px;
            color: #334155;
            font-weight: 700;
        }
        .hint {
            margin-top: 10px;
            font-size: 12px;
            color: #334155;
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"title\">Monolithic RAG -> Decomposed DAG</div>
        <div class=\"row\">
            <div class=\"node mono\">MonolithicRAG</div>
            <div class=\"arrow\">=></div>
            <div class=\"node\">Embedding</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">Ingestion</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">QueryEmbedding</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">Search</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">Reranking</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">LLMPrefilling</div>
            <div class=\"arrow\">-></div>
            <div class=\"node\">LLMDecoding</div>
        </div>
        <div class=\"hint\">
            The left side hides all stages in one operator. The right side exposes each stage so optimization passes can rewrite, split, pipeline, or prune parts of the workflow.
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

# Initialize Session State
if 'dag' not in st.session_state:
    st.session_state.dag = None
if 'monolithic_initialized' not in st.session_state:
    st.session_state.monolithic_initialized = False
if 'original_html_path' not in st.session_state:
    st.session_state.original_html_path = None
if 'optimized_html_path' not in st.session_state:
    st.session_state.optimized_html_path = None
if 'monolithic_html_path' not in st.session_state:
    st.session_state.monolithic_html_path = None
if 'transform_story_html_path' not in st.session_state:
    st.session_state.transform_story_html_path = None

# Always prepare the monolithic single-node view when page loads.
if not st.session_state.monolithic_initialized:
    monolithic_dag = create_monolithic_rag_dag()
    monolithic_html = f"{output_dir}/monolithic_single_node.html"
    visualize_dag_interactive(monolithic_dag, monolithic_html)
    st.session_state.monolithic_html_path = monolithic_html
    st.session_state.monolithic_initialized = True

st.header("1. Monolithic RAG (Single Node)")
if st.session_state.monolithic_html_path and os.path.exists(st.session_state.monolithic_html_path):
    with open(st.session_state.monolithic_html_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    components.html(source_code, height=420, scrolling=True)
else:
    st.info("Failed to load monolithic view.")

st.sidebar.header("2. Select Passes For Decomposition & Optimization")

use_pruning = st.sidebar.checkbox("PruningDependencyPass", value=True)
use_stage = st.sidebar.checkbox("StageDecompositionPass", value=False)
use_reranking = st.sidebar.checkbox("RerankingDecompositionPass", value=True)
if use_reranking:
    num_sub_nodes = st.sidebar.number_input("Reranking num_sub_nodes", min_value=1, max_value=8, value=2)
else:
    num_sub_nodes = 2
use_prefilling = st.sidebar.checkbox("PrefillingSpiltPass", value=False)
use_pipelining = st.sidebar.checkbox("LLMDecodingPipeliningPass", value=False)

if st.sidebar.button("Decompose Monolithic -> Optimize DAG"):
    with st.spinner("Decomposing monolithic RAG and optimizing DAG..."):
        transform_story_html = f"{output_dir}/monolithic_to_dag_story.html"
        write_monolithic_to_dag_story_html(transform_story_html)

        # 2. Create original decomposed DAG
        dag = create_base_dag()
        original_html = f"{output_dir}/before_optimize.html"
        visualize_dag_interactive(dag, original_html)
        
        # 3. Collect selected passes
        passes = []
        if use_pruning:
            passes.append(PruningDependencyPass())
        if use_stage:
            passes.append(StageDecompositionPass())
        if use_reranking:
            passes.append(RerankingDecompositionPass(num_sub_nodes=num_sub_nodes))
        if use_prefilling:
            passes.append(PrefillingSpiltPass())
        if use_pipelining:
            passes.append(LLMDecodingPipeliningPass())
            
        # 4. Optimize DAG
        dag.optimize(passes)
        dag.set_dag_id("optimized_dag")
        
        # 5. Visualize optimized DAG
        optimized_html = f"{output_dir}/after_optimize.html"
        visualize_dag_interactive(dag, optimized_html)
        
        # Save to session state
        st.session_state.dag = dag
        st.session_state.transform_story_html_path = transform_story_html
        st.session_state.original_html_path = original_html
        st.session_state.optimized_html_path = optimized_html
        
        st.success("Monolithic DAG has been decomposed and optimized successfully!")

st.header("3. Decomposed DAG Visualization")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Optimization")
    if st.session_state.original_html_path and os.path.exists(st.session_state.original_html_path):
        with open(st.session_state.original_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650, scrolling=True)
    else:
        st.info("Select passes and click 'Decompose Monolithic -> Optimize DAG' in the sidebar.")

with col2:
    st.subheader("After Optimization")
    if st.session_state.optimized_html_path and os.path.exists(st.session_state.optimized_html_path):
        with open(st.session_state.optimized_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650, scrolling=True)
    else:
        st.info("Optimized DAG will appear after decomposition + optimization.")

# st.header("4. Monolithic -> DAG Mapping")

# if st.session_state.transform_story_html_path and os.path.exists(st.session_state.transform_story_html_path):
#     with open(st.session_state.transform_story_html_path, 'r', encoding='utf-8') as f:
#         source_code = f.read()
#     components.html(source_code, height=210, scrolling=False)
# else:
#     st.info("Click decomposition button to generate the mapping view.")

st.header("5. Run Application")
if st.button("Run App", type="primary"):
    if st.session_state.dag is None:
        st.error("Please generate and optimize the DAG first!")
    else:
        st.info("Running application... Check the terminal output for logs.")
        with st.spinner("Running app... (This may take a while depending on your engines/models)"):
            try:
                # 运行我们的 asyncio 逻辑
                # 因为 Streamlit 在它自己的事件循环里，我们可以直接用 asyncio.run
                results = asyncio.run(run_app(st.session_state.dag))
                st.success("App finished running successfully!")
                
                if results:
                    st.markdown("### Execution Results")
                    for i, res in enumerate(results):
                        with st.expander(f"Query {i} Output", expanded=True):
                            if "LLMDecoding" in res:
                                st.markdown("**LLMDecoding Final Answer:**")
                                st.info(res["LLMDecoding"])
                            else:
                                st.write(res)
                                
            except Exception as e:
                st.error(f"Error during runtime: {e}")
                st.code(traceback.format_exc())
