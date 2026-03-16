from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph_runtime import GraphObservation, GraphTopology


@dataclass(frozen=True)
class StartDistribution:
    candidate_nodes_abs: torch.Tensor
    candidate_graph_ids: torch.Tensor
    log_probs: torch.Tensor


@dataclass(frozen=True)
class ForwardActionDistribution:
    edge_logits: torch.Tensor
    edge_agent_batch: torch.Tensor
    edge_ids: torch.Tensor
    target_nodes: torch.Tensor
    out_degrees: torch.Tensor


@dataclass(frozen=True)
class PreparedSearchBatch:
    """Encoded batch payload shared across GFlowNet runtime code."""

    topology: GraphTopology
    observation: GraphObservation
    node_tokens: torch.Tensor
    question_tokens: torch.Tensor


@dataclass(frozen=True)
class HeuristicCache:
    node_log_heuristic: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedGFlowNetBatch(PreparedSearchBatch):
    heuristic_cache: HeuristicCache


__all__ = [
    "ForwardActionDistribution",
    "HeuristicCache",
    "PreparedGFlowNetBatch",
    "PreparedSearchBatch",
    "StartDistribution",
]
