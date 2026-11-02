"""capsule.agent_tools -- tool layer for the M2' K8s migration pipeline.

Pure-Python, deterministic, LLM-free callables that the four agent roles
invoke through the runner.  Each tool is a plain class with ``name`` /
``description`` and a ``run(**kwargs) -> dict`` method -- the surface is
CrewAI-compatible (wrapping in crewai.tools.BaseTool later only adds a
pydantic args_schema).

  * mapping_rules.py      shared mapping constants & scalar rules (the single
                          source behind "验收器 = 评分器")
  * K8sRenderTool         capsule.k8s_render.render() wrapped as a tool:
                          (manifest, blueprint) -> {file: yaml} + summary
  * KubeValidateTool      two-level validator (self-written D4/D5 checks +
                          optional kubeconform schema pass)
"""

from capsule.agent_tools.k8s_render_tool import (
    BLUEPRINT_FIELDS,
    BlueprintError,
    K8sRenderTool,
    load_manifest,
    validate_blueprint,
)
from capsule.agent_tools.kube_validate_tool import KubeValidateTool

__all__ = [
    "BLUEPRINT_FIELDS",
    "BlueprintError",
    "K8sRenderTool",
    "KubeValidateTool",
    "load_manifest",
    "validate_blueprint",
]
