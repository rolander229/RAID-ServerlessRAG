"""capsule.m2s_k8s -- M2' four-agent K8s migration pipeline.

Modules:
  roles            four K8s-era role definitions (system prompts + schemas)
  tasks            six sequential tasks binding roles to tools/artifacts
  expert_baseline  independent expert derivation (scoring reference)
  llm              LLMClient / FakeLLM / ArkLLM (env-configured, no hardcoded keys)
  fake_responses   canned responses for offline pipeline runs
  runner           sequential pipeline -> k8s_migrated/ contract

CLI:
  python -m capsule.m2s_k8s.runner --manifest <slice_manifest.json> --out <dir> --fake-llm
  python -m capsule.m2s_k8s.runner --check-llm
"""

from capsule.m2s_k8s.expert_baseline import compare_expected_vs_render, derive_expected
from capsule.m2s_k8s.llm import ArkLLM, FakeLLM, LLMClient, LLMError

# NOTE: capsule.m2s_k8s.runner is intentionally NOT imported here so that
# `python -m capsule.m2s_k8s.runner` does not double-import itself.
# Use `from capsule.m2s_k8s.runner import run, PipelineError, OverlayError`.

__all__ = [
    "ArkLLM",
    "FakeLLM",
    "LLMClient",
    "LLMError",
    "compare_expected_vs_render",
    "derive_expected",
]
