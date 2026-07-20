"""RerankingDecompositionPass

Motivation
----------
StageDecompositionPass decomposes EMBEDDING and VECTORDB_SEARCHING nodes into
parallel micro-batches but intentionally excludes RERANKING (it is in
``batchable_ops`` but *not* in ``splittable_ops``).  This leaves a sequential
bottleneck: the full set of top-K retrieved candidates must be scored by a
single cross-encoder call before LLM prefilling can start.

Optimization
------------
This pass splits the candidate pool (``search_results``) across *N* parallel
Reranking sub-nodes.  Each sub-node independently scores its shard against the
query.  A global TOP-K aggregator node then selects the best results from all
shards, preserving the original reranking semantics.

Timeline impact (top_k_search=20, num_sub_nodes=4 example)
-----------------------------------------------------------

Before this pass::

    [Search] ──► [Reranking (20 docs)] ──► [LLMPrefilling]

After this pass::

    [Search] ──► [Reranking-sub-0  (5 docs)] ─┐
             ├──► [Reranking-sub-1  (5 docs)] ─┤─► [Reranking-aggregator] ──► [LLMPrefilling]
             ├──► [Reranking-sub-2  (5 docs)] ─┤
             └──► [Reranking-sub-3  (5 docs)] ─┘

The reranking stage latency is reduced by approximately 1/N because all
sub-nodes are dispatched to the Reranker engine concurrently, and the engine
supports batched cross-encoder inference.

Interaction with other passes
------------------------------
* Should run **after** ``StageDecompositionPass`` so that the Search node has
  already been stabilised.
* Should run **before** ``PrefillingSpiltPass`` so that the prefilling split
  logic sees a clean ``Reranking-aggregator → LLMPrefilling`` edge.
"""

import copy
import math
from typing import List, Set

from Ayo.configs.model_config import AggMode
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.opt_pass.base_pass import OPT_Pass

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class RerankingDecompositionPass(OPT_Pass):
    """Decompose a Reranking node into *N* parallel sub-nodes with a TOP-K aggregator.

    Parameters
    ----------
    num_sub_nodes : int
        Number of parallel Reranking sub-nodes to create (default: 2).
        Setting this to 1 is a no-op.
    """

    def __init__(self, num_sub_nodes: int = 2):
        super().__init__(name="reranking_decomposition")
        self.configure(num_sub_nodes=num_sub_nodes)

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self, dag: DAG) -> DAG:
        """Execute the optimization pass.

        Args:
            dag: Input DAG.  Best results when ``StageDecompositionPass`` has
                 already been applied.

        Returns:
            DAG with Reranking nodes decomposed into parallel sub-nodes.
        """
        processed_nodes: Set[Node] = set()
        # Operate on a stable snapshot of the topological order
        to_process = dag.topological_sort()

        for node in to_process:
            if self._is_decomposable(node) and node not in processed_nodes:
                logger.info(f"[RerankingDecompositionPass] decomposing node: {node.name}")
                self._decompose(dag, node, processed_nodes)

        return dag

    def get_applicable_nodes(self, dag: DAG) -> List[Node]:
        """Return Reranking nodes eligible for decomposition."""
        return [n for n in dag.nodes if self._is_decomposable(n)]

    def validate_dag(self, dag: DAG) -> bool:
        return len(self.get_applicable_nodes(dag)) > 0

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _is_decomposable(self, node: Node) -> bool:
        """Check if this Reranking node is eligible.

        Conditions:
        * Operation type is RERANKING.
        * Node has not already been decomposed (prevents double-processing).
        * Node has ``search_results`` as an input field (sanity check).
        * Configured ``num_sub_nodes`` > 1.
        """
        return (
            node.op_type == NodeOps.RERANKING
            and not node.decomposed
            and "search_results" in node.input_names
            and self.get_config("num_sub_nodes", 2) > 1
        )

    def _get_total_candidates(self, node: Node, dag: DAG) -> int:
        """Resolve the number of retrieved candidates at compile time.

        Traversal strategy:
        1. Follow ``input_key_to_parent`` back to the node that produces
           ``search_results``, then walk through any AGGREGATOR nodes to reach
           the originating VECTORDB_SEARCHING node whose ``top_k`` config
           contains the answer.
        2. Fall back to ``dag.query_inputs["search_results"]`` if it exists.
        3. Return -1 when the size cannot be determined statically (the pass
           will still decompose using open-ended slices in that case).
        """
        source = node.input_key_to_parent.get("search_results")
        if source is not None:
            visited: Set[Node] = set()
            current = source
            while current is not None and current not in visited:
                visited.add(current)
                if current.op_type == NodeOps.VECTORDB_SEARCHING:
                    top_k = current.config.get("top_k")
                    if top_k is not None:
                        return int(top_k)
                # Walk up through aggregator nodes
                if current.op_type == NodeOps.AGGREGATOR and current.parents:
                    current = current.parents[0]
                else:
                    break

        # Fallback: check dag-level query inputs
        if "search_results" in dag.query_inputs:
            data = dag.query_inputs["search_results"]
            if isinstance(data, list):
                return len(data)

        return -1  # Cannot determine statically

    def _decompose(
        self,
        dag: DAG,
        node: Node,
        processed_nodes: Set[Node],
    ) -> None:
        """Replace *node* with N parallel sub-nodes and a TOP-K aggregator.

        Steps
        -----
        1. Determine shard boundaries.
        2. Create ``num_sub_nodes`` Reranking sub-nodes, each handling a slice
           of ``search_results`` (``queries`` is replicated, not sliced).
        3. Connect every original parent → every sub-node.
        4. Create a TOP-K aggregator that merges sub-node outputs.
        5. Redirect every child of *node* to the aggregator.
        6. Remove *node* from the DAG.
        """
        num_sub_nodes = self.get_config("num_sub_nodes", 2)

        # ------------------------------------------------------------------ #
        # 1. Determine shard boundaries                                        #
        # ------------------------------------------------------------------ #
        total_candidates = self._get_total_candidates(node, dag)

        if total_candidates != -1 and total_candidates < num_sub_nodes * 2:
            logger.info(
                f"[RerankingDecompositionPass] skipping {node.name}: "
                f"only {total_candidates} candidates for {num_sub_nodes} sub-nodes."
            )
            return

        if total_candidates == -1:
            # Fall back: use num_sub_nodes equal partitions without a known size.
            # Slices will be resolved at runtime by the engine's partial handling.
            micro_batch_size = None
            logger.warning(
                f"[RerankingDecompositionPass] cannot determine static candidate "
                f"count for {node.name}; using open-ended per-sub-node slices."
            )
        else:
            micro_batch_size = math.ceil(total_candidates / num_sub_nodes)

        # ------------------------------------------------------------------ #
        # 2. Build sub-nodes                                                   #
        # ------------------------------------------------------------------ #
        sub_nodes: List[Node] = []
        for i in range(num_sub_nodes):
            sub_io_schema = copy.deepcopy(node.io_schema)
            # Each sub-node gets a uniquely named output so the aggregator
            # can identify them via io_schema intersection.
            sub_io_schema.output_format = {
                f"{k}-sub-{i}": v
                for k, v in node.io_schema.output_format.items()
            }
            sub_config = copy.deepcopy(node.config)

            sub_node = Node(
                name=f"{node.name}-sub-{i}",
                op_type=node.op_type,
                engine_type=node.engine_type,
                node_type=node.node_type,
                io_schema=sub_io_schema,
                config=sub_config,
                anno=node.anno,
            )
            sub_node.decomposed = True

            # Shard search_results; leave queries un-sharded (broadcast).
            if micro_batch_size is not None:
                start = i * micro_batch_size
                end = min((i + 1) * micro_batch_size, total_candidates)
                sub_node.input_shards_mapping = {
                    "search_results": slice(start, end)
                }
            # If total_candidates is unknown we do not set input_shards_mapping;
            # the engine will process the full input (graceful degradation).

            dag.add_node(sub_node)
            sub_nodes.append(sub_node)

        # ------------------------------------------------------------------ #
        # 3. Connect every original parent → every sub-node                   #
        # ------------------------------------------------------------------ #
        # snapshot original_parents before any structural changes
        original_parents = list(node.parents)
        for parent in original_parents:
            for sub_node in sub_nodes:
                # add_child → add_parent auto-wires input_key_from_parents
                parent.add_child(sub_node)

        # ------------------------------------------------------------------ #
        # 4. Create TOP-K aggregator                                           #
        # ------------------------------------------------------------------ #
        agg_input_format = {
            f"{k}-sub-{i}": v
            for i in range(num_sub_nodes)
            for k, v in node.io_schema.output_format.items()
        }
        agg_io_schema = NodeIOSchema(
            input_format=agg_input_format,
            output_format=copy.deepcopy(node.io_schema.output_format),
        )

        top_k = node.config.get("top_k", 5)
        agg_node = Node(
            name=f"{node.name}-aggregator",
            op_type=NodeOps.AGGREGATOR,
            engine_type=node.engine_type,
            node_type=node.node_type,
            io_schema=agg_io_schema,
            config={"agg_mode": AggMode.TOP_K, "top_k": top_k},
        )
        dag.add_node(agg_node)

        for sub_node in sub_nodes:
            sub_node.add_child(agg_node)

        # ------------------------------------------------------------------ #
        # 5. Redirect original children → aggregator                          #
        # ------------------------------------------------------------------ #
        original_children = list(node.children)
        for child in original_children:
            # Connect aggregator → child.
            # add_parent auto-wires input_key_from_parents via io intersection.
            agg_node.add_child(child)

            # Clean up the stale mapping that pointed to the original node so
            # that update_input_kwargs / check_node_ready do not see a ghost
            # parent after remove_node cleans up child.parents.
            if node.name in child.input_key_from_parents:
                stale_key = child.input_key_from_parents.pop(node.name)
                # Refresh input_key_to_parent to point at the aggregator
                if stale_key in child.input_key_to_parent:
                    child.input_key_to_parent[stale_key] = agg_node

        # ------------------------------------------------------------------ #
        # 6. Remove original Reranking node                                    #
        # ------------------------------------------------------------------ #
        # dag.remove_node handles:
        #   parent.children.remove(node) for every parent of node
        #   child.parents.remove(node)   for every child still listed in node.children
        dag.remove_node(node.name)
        processed_nodes.add(node)

        logger.info(
            f"[RerankingDecompositionPass] {node.name} → "
            f"{num_sub_nodes} sub-nodes "
            f"(total_candidates={total_candidates}, "
            f"micro_batch_size={micro_batch_size})"
        )
