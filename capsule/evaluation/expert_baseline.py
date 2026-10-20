"""
Independent Expert Ground-Truth Baselines for MOPAR-partitioned DNN Pipelines.

This module provides expert-authored reference designs that are INDEPENDENT of the
tool's output: stage groupings are defined by fixed name-pattern rules over the
model's layer universe, and the API/trigger/infrastructure contracts are specified
up front. The evaluator compares tool artifacts against these baselines instead of
comparing the tool against a copy of itself.

Partition similarity uses pairwise co-membership Jaccard (an MoJoFM-style
partition-distance proxy, IEEE TSE '07): two layers agree iff they are grouped
together in BOTH partitions.
"""
import re
from typing import Any, Callable, Dict, List, Tuple

# Ordered rules: first matching pattern assigns the layer to a named expert stage.
# Entries may be regex strings or callables (name -> bool).
EXPERT_PROFILES: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "stages": [
            ("stem", [r"^(conv1|bn1|relu|maxpool)$"]),
            ("stage1", [r"^layer1\."]),
            ("stage2", [r"^layer2\."]),
            ("stage3", [r"^layer3\."]),
            ("head", [r"^(layer4\.|avgpool|fc)"]),
        ],
        "api": [("POST", "/predict", 200)],
    },
    "vgg16": {
        "stages": [
            ("backbone", [r"^features\."]),
            ("head", [r"^(avgpool|flatten|classifier\.)"]),
        ],
        "api": [("POST", "/predict", 200)],
    },
    "inception_v3": {
        "stages": [
            ("stem", [r"^Conv2d_[1-4]"]),
            ("inception-mid", [r"^Mixed_[56]"]),
            ("head", [r"^(Mixed_7|avgpool|dropout|fc)"]),
        ],
        "api": [("POST", "/predict", 200)],
    },
    "convnext_tiny": {
        "stages": [
            ("stem", [r"^features\.[01]"]),
            ("mid-blocks", [r"^features\.[234]"]),
            ("late-blocks", [r"^features\.[567]"]),
            ("head", [r"^(avgpool|flatten|classifier\.)"]),
        ],
        "api": [("POST", "/predict", 200)],
    },
    "qwen3_1_7b": {
        "stages": [
            ("embedding+early-blocks", [lambda n: n in ("embed_tokens", "rotary_emb") or _qwen_layer(n) is not None and _qwen_layer(n) <= 3]),
            ("mid-blocks-1", [lambda n: _qwen_layer(n) is not None and 4 <= _qwen_layer(n) <= 11]),
            ("mid-blocks-2", [lambda n: _qwen_layer(n) is not None and 12 <= _qwen_layer(n) <= 19]),
            ("head", [lambda n: n in ("norm", "lm_head") or (_qwen_layer(n) is not None and _qwen_layer(n) >= 20)]),
        ],
        "api": [("POST", "/chat", 200)],
    },
}

# Fallback for models without a dedicated profile: single-stage monolith design.
_DEFAULT_PROFILE = {
    "stages": [("pipeline", [r".*"])],
    "api": [("POST", "/predict", 200)],
}


def _qwen_layer(name: str):
    m = re.match(r"^layers\.(\d+)(\.|$)", name)
    return int(m.group(1)) if m else None


def _stage_for(name: str, stages: List[Tuple[str, List]]) -> str:
    for stage_name, patterns in stages:
        for p in patterns:
            if callable(p):
                if p(name):
                    return stage_name
            elif re.search(p, name):
                return stage_name
    return "unassigned"


def expert_partition(layer_universe: List[str], model_key: str) -> List[List[str]]:
    """Assign every layer of the universe to an expert stage; return stage groups."""
    profile = EXPERT_PROFILES.get(model_key, _DEFAULT_PROFILE)
    groups: Dict[str, List[str]] = {}
    for layer in layer_universe:
        groups.setdefault(_stage_for(layer, profile["stages"]), []).append(layer)
    # Keep pipeline order: group layers by first appearance of their stage.
    ordered: List[List[str]] = []
    seen = set()
    for layer in layer_universe:
        stage = _stage_for(layer, profile["stages"])
        if stage not in seen:
            seen.add(stage)
            ordered.append(list(groups[stage]))
    return ordered


def partition_similarity(
    tool_parts: List[List[str]], expert_parts: List[List[str]]
) -> Tuple[float, Dict[str, Any]]:
    """Pairwise co-membership Jaccard between two partitions of the same layer set."""
    tool_map: Dict[str, int] = {}
    for i, part in enumerate(tool_parts):
        for layer in part:
            tool_map[layer] = i
    exp_map: Dict[str, int] = {}
    for i, part in enumerate(expert_parts):
        for layer in part:
            exp_map[layer] = i

    universe = sorted(set(tool_map) & set(exp_map))
    if len(universe) < 2:
        return 0.0, {"reason": "partition universe too small to compare", "universe_size": len(universe)}

    pair_sets = []
    for mapping in (tool_map, exp_map):
        groups: Dict[int, List[str]] = {}
        for layer in universe:
            groups.setdefault(mapping[layer], []).append(layer)
        pairs = set()
        for members in groups.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pairs.add((members[a], members[b]))
        pair_sets.append(pairs)

    inter = len(pair_sets[0] & pair_sets[1])
    union = len(pair_sets[0] | pair_sets[1])
    score = inter / union if union else 0.0
    return score, {
        "universe_size": len(universe),
        "tool_groups": len(set(tool_map[l] for l in universe)),
        "expert_groups": len(set(exp_map[l] for l in universe)),
        "agreeing_pairs": inter,
        "total_union_pairs": union,
        "metric": "pairwise co-membership Jaccard (MoJoFM-style, IEEE TSE '07)",
    }


def build_expert_spec(model_key: str, layer_universe: List[str]) -> Dict[str, Any]:
    """Expert-authored ground-truth specification, independent of tool output."""
    profile = EXPERT_PROFILES.get(model_key, _DEFAULT_PROFILE)
    method, path, expected_status = profile["api"][0]
    stage_groups = expert_partition(layer_universe, model_key)
    spec = {
        "application_name": f"expert-{model_key}-pipeline",
        "domain_type": "Expert Manual Reference Design for Micro-Partitioned Serverless Inference",
        "author": "independent expert baseline (fixed stage-pattern rules, see evaluation/expert_baseline.py)",
        "total_monolith_endpoints": 1,
        "total_encapsulated_functions": len(stage_groups),
        "functions": [
            {
                "name": f"{model_key}-entry",
                "method": method,
                "path": path,
                "expected_status": expected_status,
                "trigger": "api_gateway",
                "type": "orchestrator",
            }
        ] + [
            {
                "name": f"{model_key}-stage-worker-{i}",
                "method": None,
                "path": None,
                "trigger": "event",
                "type": "worker",
            }
            for i in range(1, len(stage_groups))
        ],
        "infrastructure": {
            "api_gateway": True,
            "cognito": False,
            "sqs_queues": [],
            "dynamodb_tables": [],
            "s3_buckets": ["slicer-activations-storage"],
        },
        "expert_partition": stage_groups,
    }
    return spec
