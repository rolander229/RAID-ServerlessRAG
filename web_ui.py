import os
import sys
import asyncio
import streamlit as st
from PIL import Image
import traceback

# 确保可以导入当前目录和 Ayo 库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_embedding_ingestion_searching_reranking_llm import create_base_dag, run_app, vectordb_config
from Ayo.vis.vis_graph import visualize_dag_with_compute_nodes_in_line
from Ayo.vis.vis_graph_interactive import visualize_dag_interactive
import streamlit.components.v1 as components

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

# Initialize Session State
if 'dag' not in st.session_state:
    st.session_state.dag = None
if 'original_html_path' not in st.session_state:
    st.session_state.original_html_path = None
if 'optimized_html_path' not in st.session_state:
    st.session_state.optimized_html_path = None

st.sidebar.header("1. Select Optimization Passes")

use_pruning = st.sidebar.checkbox("PruningDependencyPass", value=True)
use_stage = st.sidebar.checkbox("StageDecompositionPass", value=False)
use_reranking = st.sidebar.checkbox("RerankingDecompositionPass", value=True)
if use_reranking:
    num_sub_nodes = st.sidebar.number_input("Reranking num_sub_nodes", min_value=1, max_value=8, value=2)
else:
    num_sub_nodes = 2
use_prefilling = st.sidebar.checkbox("PrefillingSpiltPass", value=False)
use_pipelining = st.sidebar.checkbox("LLMDecodingPipeliningPass", value=False)

if st.sidebar.button("Generate & Optimize DAG"):
    with st.spinner("Creating and optimizing DAG..."):
        # 1. Create original DAG
        dag = create_base_dag()
        original_html = f"{output_dir}/before_optimize.html"
        visualize_dag_interactive(dag, original_html)
        
        # 2. Collect selected passes
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
            
        # 3. Optimize DAG
        dag.optimize(passes)
        dag.set_dag_id("optimized_dag")
        
        # 4. Visualize optimized DAG
        optimized_html = f"{output_dir}/after_optimize.html"
        visualize_dag_interactive(dag, optimized_html)
        
        # Save to session state
        st.session_state.dag = dag
        st.session_state.original_html_path = original_html
        st.session_state.optimized_html_path = optimized_html
        
        st.success("DAG created and optimized successfully!")

st.header("2. DAG Visualization (Interactive)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Optimization")
    if st.session_state.original_html_path and os.path.exists(st.session_state.original_html_path):
        with open(st.session_state.original_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650, scrolling=True)
    else:
        st.info("Click 'Generate & Optimize DAG' to see the visualization.")

with col2:
    st.subheader("After Optimization")
    if st.session_state.optimized_html_path and os.path.exists(st.session_state.optimized_html_path):
        with open(st.session_state.optimized_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650, scrolling=True)

st.header("3. Run Application")
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
