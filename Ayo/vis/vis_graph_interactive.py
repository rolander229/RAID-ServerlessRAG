import networkx as nx
import json
from pyvis.network import Network

from Ayo.dags.dag import DAG
from Ayo.dags.node_commons import NodeType

def visualize_dag_interactive(dag: DAG, output_path: str = "interactive_dag.html"):
    """
    Make an interactive DAG visualization using pyvis.
    Nodes can be dragged, zoomed, and have tooltips.
    """
    if hasattr(dag, "_ensure_topo_sort"):
        dag._ensure_topo_sort()

    G = nx.DiGraph()

    node_type_colors = {
        NodeType.INPUT: "#90EE90",
        NodeType.COMPUTE: "#87CEFA",
        NodeType.OUTPUT: "#FA8072",
    }

    # Add nodes
    for node in dag.nodes:
        color = node_type_colors.get(node.node_type, "#D3D3D3")
        engine_str = node.engine_type.value if hasattr(node.engine_type, 'value') else str(node.engine_type)
        label = f"{node.name}\n[{engine_str}]"
        
        # Build tooltip title
        title_lines = [
            f"<b>Name:</b> {node.name}",
            f"<b>Node Type:</b> {node.node_type}",
            f"<b>Engine:</b> {engine_str}",
            f"<b>Op Type:</b> {node.op_type}"
        ]
        title = "<br>".join(title_lines)
        
        G.add_node(node.name, label=label, title=title, color=color, shape="box", font={"size": 16, "face": "arial"})

    # Add edges
    for node in dag.nodes:
        for child in node.children:
            G.add_edge(node.name, child.name, arrows="to")

    # Create Pyvis network
    net = Network(directed=True, height="600px", width="100%", bgcolor="white", font_color="black")
    net.from_nx(G)

    # Use a directed hierarchical layout resembling a DAG
    options = {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "edges": {
            "color": {"inherit": True},
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
                "roundness": 0.4
            }
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 250,
                "nodeSpacing": 100
            }
        },
        "physics": {
            "enabled": False  # Disable continuous bouncing, only rely on the initial layout or user dragging
        }
    }
    net.set_options(json.dumps(options))

    net.save_graph(output_path)
    return output_path
