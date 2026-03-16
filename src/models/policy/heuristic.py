from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.models.components.heuristic_heads import LearnedHeuristicHead
from src.models.configs import HeuristicConfig

from .heuristic_utils import (
    compute_embedding_log_heuristic,
    compute_topology_log_heuristic,
)
from .state import SearchState
from .types import ForwardActionDistribution, HeuristicCache, PreparedSearchBatch


StateFeatureBuilder = Callable[[PreparedSearchBatch, SearchState], torch.Tensor]


class TrajectoryHeuristic(nn.Module):
    def __init__(
        self,
        *,
        config: HeuristicConfig,
        learned_head: LearnedHeuristicHead | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self.learned_head = learned_head
        if self.uses_learned_head and self.learned_head is None:
            raise ValueError("learned heuristic requires a LearnedHeuristicHead.")
        if not self.uses_learned_head and self.learned_head is not None:
            raise ValueError(
                "learned_head should only be provided when heuristic.kind is 'learned'."
            )

    @property
    def config(self) -> HeuristicConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return float(self.config.beta) > 0.0

    @property
    def uses_learned_head(self) -> bool:
        return self.config.canonical_kind == "learned"

    def build_cache(self, prepared_batch: PreparedSearchBatch) -> HeuristicCache:
        if not self.enabled:
            return HeuristicCache(node_log_heuristic=None)
        if self.config.canonical_kind == "topology":
            return HeuristicCache(
                node_log_heuristic=compute_topology_log_heuristic(
                    topology=prepared_batch.topology,
                    observation=prepared_batch.observation,
                    restart_prob=float(self.config.topology_restart_prob),
                    num_iters=int(self.config.topology_num_iters),
                    eps=float(self.config.topology_eps),
                )
            )
        if self.config.canonical_kind == "embedding":
            return HeuristicCache(
                node_log_heuristic=compute_embedding_log_heuristic(
                    topology=prepared_batch.topology,
                    node_tokens=prepared_batch.node_tokens,
                    question_tokens=prepared_batch.question_tokens,
                    temperature=float(self.config.embedding_temperature),
                )
            )
        return HeuristicCache(node_log_heuristic=None)

    @staticmethod
    def _node_log_heuristic(
        *, node_ids: torch.Tensor, cache: HeuristicCache
    ) -> torch.Tensor:
        if cache.node_log_heuristic is None:
            return torch.zeros_like(node_ids, dtype=torch.float32)
        return cache.node_log_heuristic.index_select(0, node_ids)

    def _state_log_heuristic(
        self,
        *,
        state_features: torch.Tensor,
        question_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.learned_head is None:
            return torch.zeros(
                (int(state_features.size(0)),),
                device=state_features.device,
                dtype=torch.float32,
            )
        logits = self.learned_head(
            state_features=state_features,
            question_features=question_features,
        )
        return torch.nn.functional.logsigmoid(logits)

    def compute_start_bias(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        heuristic_cache: HeuristicCache,
        candidate_nodes_abs: torch.Tensor,
        candidate_graph_ids: torch.Tensor,
        build_state_features: StateFeatureBuilder,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                (int(candidate_nodes_abs.numel()),),
                device=candidate_nodes_abs.device,
                dtype=torch.float32,
            )
        if self.uses_learned_head:
            start_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=candidate_nodes_abs.view(-1, 1),
                done_mask=torch.zeros(
                    (int(candidate_nodes_abs.numel()), 1),
                    device=candidate_nodes_abs.device,
                    dtype=torch.bool,
                ),
                num_steps=torch.zeros(
                    (int(candidate_nodes_abs.numel()), 1),
                    device=candidate_nodes_abs.device,
                    dtype=torch.long,
                ),
            )
            state_features = build_state_features(prepared_batch, start_state).view(
                int(candidate_nodes_abs.numel()), -1
            )
            question_features = prepared_batch.question_tokens.index_select(
                0, candidate_graph_ids
            )
            return self._state_log_heuristic(
                state_features=state_features,
                question_features=question_features,
            )
        return self._node_log_heuristic(
            node_ids=candidate_nodes_abs,
            cache=heuristic_cache,
        )

    def compute_transition_bias(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        heuristic_cache: HeuristicCache,
        distribution: ForwardActionDistribution,
        state: SearchState,
        build_state_features: StateFeatureBuilder,
    ) -> torch.Tensor:
        if not self.enabled or int(distribution.edge_logits.numel()) == 0:
            return torch.zeros(
                (int(distribution.target_nodes.numel()),),
                device=distribution.target_nodes.device,
                dtype=torch.float32,
            )
        if self.uses_learned_head:
            child_num_steps = (
                state.flatten_num_steps().index_select(0, distribution.edge_agent_batch)
                + 1
            )
            child_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=distribution.target_nodes.view(-1, 1),
                done_mask=torch.zeros(
                    (int(distribution.target_nodes.numel()), 1),
                    device=distribution.target_nodes.device,
                    dtype=torch.bool,
                ),
                num_steps=child_num_steps.view(-1, 1),
            )
            child_features = build_state_features(prepared_batch, child_state).view(
                int(distribution.target_nodes.numel()), -1
            )
            child_graph_ids = prepared_batch.topology.graph_index_from_nodes(
                distribution.target_nodes
            )
            return self._state_log_heuristic(
                state_features=child_features,
                question_features=prepared_batch.question_tokens.index_select(
                    0, child_graph_ids
                ),
            )
        return self._node_log_heuristic(
            node_ids=distribution.target_nodes,
            cache=heuristic_cache,
        )


__all__ = ["TrajectoryHeuristic"]
