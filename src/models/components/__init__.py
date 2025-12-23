from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "GraphBatch",
    "GraphState",
    "GraphEnv",
    "GraphEmbedder",
    "GFlowNetEstimator",
    "GFlowNetActor",
    "GNNStateEncoder",
    "EdgeFrontierPolicy",
    "EdgeGATPolicy",
    "EdgeMLPMixerPolicy",
    "AnswerOnlyReward",
    "RewardOutput",
    "FeatureFusion",
    "FiLMLayer",
    "DDE",
    "PEConv",
    "DenseFeatureExtractor",
    "DeterministicHead",
    "EmbeddingProjector",
    "Retriever",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FeatureFusion": (".fusion", "FeatureFusion"),
    "FiLMLayer": (".fusion", "FiLMLayer"),
    "GraphBatch": (".gflownet_env", "GraphBatch"),
    "GraphEnv": (".gflownet_env", "GraphEnv"),
    "GraphState": (".gflownet_env", "GraphState"),
    "GFlowNetActor": (".gflownet_actor", "GFlowNetActor"),
    "GNNStateEncoder": (".gflownet_state_encoder", "GNNStateEncoder"),
    "GraphEmbedder": (".gflownet_embedder", "GraphEmbedder"),
    "GFlowNetEstimator": (".gflownet_estimator", "GFlowNetEstimator"),
    "EdgeFrontierPolicy": (".gflownet_policies", "EdgeFrontierPolicy"),
    "EdgeGATPolicy": (".gflownet_policies", "EdgeGATPolicy"),
    "EdgeMLPMixerPolicy": (".gflownet_policies", "EdgeMLPMixerPolicy"),
    "AnswerOnlyReward": (".gflownet_rewards", "AnswerOnlyReward"),
    "RewardOutput": (".gflownet_rewards", "RewardOutput"),
    "DDE": (".graph", "DDE"),
    "PEConv": (".graph", "PEConv"),
    "DenseFeatureExtractor": (".heads", "DenseFeatureExtractor"),
    "DeterministicHead": (".heads", "DeterministicHead"),
    "EmbeddingProjector": (".projections", "EmbeddingProjector"),
    "Retriever": (".retriever", "Retriever"),
}

if TYPE_CHECKING:  # pragma: no cover
    from .fusion import FeatureFusion, FiLMLayer
    from .gflownet_actor import GFlowNetActor
    from .gflownet_embedder import GraphEmbedder
    from .gflownet_env import GraphBatch, GraphEnv, GraphState
    from .gflownet_estimator import GFlowNetEstimator
    from .gflownet_state_encoder import GNNStateEncoder
    from .gflownet_policies import EdgeFrontierPolicy, EdgeGATPolicy, EdgeMLPMixerPolicy
    from .gflownet_rewards import AnswerOnlyReward, RewardOutput
    from .graph import DDE, PEConv
    from .heads import DenseFeatureExtractor, DeterministicHead
    from .projections import EmbeddingProjector
    from .retriever import Retriever


def __getattr__(name: str) -> Any:  # pragma: no cover
    spec = _LAZY_IMPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
