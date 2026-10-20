"""The twelve benchmark services of Sec. III-A and their layer graphs.

``zoo`` is the registry (what the paper ran), ``custom`` holds the architectures
the paper did not publish, and ``graphify`` turns any of them into the
:class:`~slicer.partition.graph.LayerGraph` the partitioner consumes. Start with::

    from slicer.models import load, build_structure_graph
    model, inputs = load("resnet50", device="cuda")
    graph = build_structure_graph(model, inputs, name="resnet50")
"""

from .custom import BertStack, CnnRnn, GraphNet, InceptionTransformer
from .graphify import build_structure_graph
from .zoo import FAMILIES, SPECS, ModelSpec, get, load

__all__ = ["BertStack", "CnnRnn", "GraphNet", "InceptionTransformer",
           "build_structure_graph", "FAMILIES", "SPECS", "ModelSpec", "get", "load"]
