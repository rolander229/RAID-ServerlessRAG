"""PDDisaggregationPass

Motivation
----------
A monolithic LLM_FULL node combines prefilling (prompt processing) and
autoregressive decoding in a single compute step.  When multiple queries arrive
concurrently, the prefilling phase for one query can block the decoding phase of
another, leading to head-of-line blocking and poor GPU utilisation.

Prefill-Decode (PD) Disaggregation separates these two phases into distinct
pipeline stages so that:
* Prefilling nodes from many queries can be batched together (compute-bound,
  benefits from large batch sizes).
* Decoding nodes can be dispatched without waiting for a new full prefill to
  complete (memory-bandwidth-bound, benefits from low latency scheduling).

Transformation
--------------
Before this pass::

    [... upstream ...] ──► [LLM (LLM_FULL)] ──► [... downstream ...]

After this pass::

    [... upstream ...] ──► [LLMPrefilling (LLM_PREFILLING)] ──► [LLMDecoding (LLM_DECODING)] ──► [... downstream ...]

The pass:
1. Locates every ``LLM_FULL`` node in the DAG.
2. Creates a ``LLM_PREFILLING`` node that inherits the input schema and
   configuration of the original node and outputs a ``prefill_state`` boolean.
3. Creates a ``LLM_DECODING`` node that takes ``prefill_state`` as its sole
   input and emits the same outputs as the original node.
4. Re-wires parents → prefilling → decoding → children.
5. Removes the original ``LLM_FULL`` node.

Both new nodes share the same ``llm_internal_id`` (taken from the original node
config or generated fresh) so the LLM engine can correlate the two phases.

Interaction with other passes
------------------------------
* Should run **after** ``RerankingDecompositionPass`` and
  ``StageDecompositionPass`` so that the upstream graph is stable.
* Should run **before** ``PrefillingSpiltPass`` so that the prefilling split
  logic sees the new ``LLMPrefilling`` node.
* Should run **before** ``LLMDecodingPipeliningPass`` for the same reason.
"""

import copy
import uuid
from typing import List, Set

from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeAnnotation, NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.opt_pass.base_pass import OPT_Pass

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class PDDisaggregationPass(OPT_Pass):
    """Decompose each LLM_FULL node into a LLM_PREFILLING + LLM_DECODING pair.

    The two new nodes share the same ``llm_internal_id`` so that the LLM engine
    can associate the cached KV state from prefilling with the subsequent
    decoding request.
    """

    def __init__(self) -> None:
        super().__init__(name="pd_disaggregation")

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self, dag: DAG) -> DAG:
        """Execute the optimization pass.

        Args:
            dag: Input DAG to optimize.

        Returns:
            DAG with every LLM_FULL node replaced by a
            LLM_PREFILLING → LLM_DECODING pair.
        """
        processed_nodes: Set[Node] = set()

        # Operate on a stable snapshot; the loop modifies the DAG in-place.
        to_process = dag.topological_sort()

        for node in to_process:
            if self._is_disaggregatable(node) and node not in processed_nodes:
                logger.info(
                    f"[PDDisaggregationPass] disaggregating node: {node.name}"
                )
                self._disaggregate(dag, node, processed_nodes)

        return dag

    def get_applicable_nodes(self, dag: DAG) -> List[Node]:
        """Return LLM_FULL nodes eligible for disaggregation."""
        return [n for n in dag.nodes if self._is_disaggregatable(n)]

    def validate_dag(self, dag: DAG) -> bool:
        return len(self.get_applicable_nodes(dag)) > 0

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _is_disaggregatable(self, node: Node) -> bool:
        """Return True if *node* is a LLM_FULL node that has not been
        disaggregated yet."""
        return node.op_type == NodeOps.LLM_FULL and not node.decomposed

    def _disaggregate(
        self,
        dag: DAG,
        node: Node,
        processed_nodes: Set[Node],
    ) -> None:
        """Replace *node* with a LLM_PREFILLING → LLM_DECODING pair.

        Steps
        -----
        1. Resolve (or generate) the ``llm_internal_id`` that ties the two
           phases together inside the LLM engine.
        2. Build the ``LLMPrefilling`` node.
           * Input schema  = same as the original LLM_FULL node.
           * Output schema = ``{"prefill_state": bool}``.
        3. Build the ``LLMDecoding`` node.
           * Input schema  = ``{"prefill_state": bool}``.
           * Output schema = same as the original LLM_FULL node.
        4. Connect: every original parent → prefilling node.
        5. Connect: prefilling node → decoding node.
        6. Connect: decoding node → every original child (fix up
           ``input_key_from_parents`` / ``input_key_to_parent``).
        7. Remove the original LLM_FULL node.
        """

        # ------------------------------------------------------------------ #
        # 1. Resolve llm_internal_id                                           #
        # ------------------------------------------------------------------ #
        llm_internal_id: str = node.config.get(
            "llm_internal_id", str(uuid.uuid4())
        )

        shared_config = copy.deepcopy(node.config)
        shared_config["llm_internal_id"] = llm_internal_id

        # ------------------------------------------------------------------ #
        # 2. Build LLMPrefilling node                                          #
        # ------------------------------------------------------------------ #
        prefilling_io = NodeIOSchema(
            input_format=copy.deepcopy(node.io_schema.input_format),
            output_format={"prefill_state": bool},
        )
        prefilling_node = Node(
            name=f"{node.name}Prefilling",
            node_type=NodeType.COMPUTE,
            engine_type=EngineType.LLM,
            op_type=NodeOps.LLM_PREFILLING,
            io_schema=prefilling_io,
            anno=NodeAnnotation.BATCHABLE,
            config=copy.deepcopy(shared_config),
        )
        prefilling_node.decomposed = True
        dag.add_node(prefilling_node)

        # ------------------------------------------------------------------ #
        # 3. Build LLMDecoding node                                            #
        # ------------------------------------------------------------------ #
        decoding_io = NodeIOSchema(
            input_format={"prefill_state": bool},
            output_format=copy.deepcopy(node.io_schema.output_format),
        )
        decoding_node = Node(
            name=f"{node.name}Decoding",
            node_type=NodeType.COMPUTE,
            engine_type=EngineType.LLM,
            op_type=NodeOps.LLM_DECODING,
            io_schema=decoding_io,
            anno=NodeAnnotation.BATCHABLE,
            config=copy.deepcopy(shared_config),
        )
        decoding_node.decomposed = True
        dag.add_node(decoding_node)

        # ------------------------------------------------------------------ #
        # 4. Connect every original parent → prefilling node                  #
        # ------------------------------------------------------------------ #
        original_parents = list(node.parents)
        for parent in original_parents:
            parent.add_child(prefilling_node)

        # ------------------------------------------------------------------ #
        # 5. Connect prefilling → decoding                                     #
        # ------------------------------------------------------------------ #
        prefilling_node.add_child(decoding_node)

        # ------------------------------------------------------------------ #
        # 6. Redirect original children → decoding node                       #
        # ------------------------------------------------------------------ #
        original_children = list(node.children)
        for child in original_children:
            decoding_node.add_child(child)

            # Clean up stale mapping pointing at the original LLM_FULL node.
            if node.name in child.input_key_from_parents:
                stale_key = child.input_key_from_parents.pop(node.name)
                if stale_key in child.input_key_to_parent:
                    child.input_key_to_parent[stale_key] = decoding_node

        # ------------------------------------------------------------------ #
        # 7. Remove original LLM_FULL node                                     #
        # ------------------------------------------------------------------ #
        dag.remove_node(node.name)
        processed_nodes.add(node)

        logger.info(
            f"[PDDisaggregationPass] {node.name} → "
            f"{prefilling_node.name} → {decoding_node.name} "
            f"(llm_internal_id={llm_internal_id})"
        )
