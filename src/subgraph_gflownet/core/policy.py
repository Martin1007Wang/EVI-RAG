from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping

import torch
from torch import nn

from src.graph import TrajectoryBatch

from .actor import SubgraphActionDistribution, SubgraphActor
from .encoder import SubgraphEncoder
from .reward import AdmissibleAnswerSet, SubgraphRewardModel, SubgraphTerminalReward
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch, SubgraphState
from .state_kernel import (
    analyze_subgraph_rollout_batch,
    analyze_subgraph_state,
    forward_valid_removable_edge_ids,
    initial_subgraph_state,
    initialize_subgraph_rollout_batch,
    transition_subgraph_rollout_batch,
)
from .subgraph_batch import SubgraphBatch, SubgraphBatchBuildOptions

SUBGRAPH_STATE_MODE = "subgraph"


class SubgraphPolicy(nn.Module):
    def __init__(
        self,
        *,
        state_mode: str,
        backbone: dict[str, Any],
        flow_head: dict[str, Any],
        state_encoder: dict[str, Any],
        actor: dict[str, Any],
        answer_reward: dict[str, Any],
        proposal_prior: dict[str, Any],
        max_steps: int,
    ) -> None:
        super().__init__()
        self.state_mode = SUBGRAPH_STATE_MODE
        if str(state_mode) != SUBGRAPH_STATE_MODE:
            raise ValueError("policy.state_mode must be 'subgraph'.")
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 for SubgraphPolicy.")
        self.encoder = SubgraphEncoder(
            backbone=backbone,
            state_encoder=state_encoder,
            flow_head=flow_head,
        )
        self.reward_model = SubgraphRewardModel(**answer_reward)
        self.actor = SubgraphActor(
            hidden_dim=int(backbone["hidden_dim"]),
            max_steps=self.max_steps,
            actor=actor,
            proposal_prior=proposal_prior,
        )

    def prepare_batch(
        self,
        batch: TrajectoryBatch,
        *,
        build_options: SubgraphBatchBuildOptions | None = None,
    ) -> SubgraphBatch:
        return self.encoder.prepare_batch(batch, build_options=build_options)

    @staticmethod
    def initial_state() -> SubgraphState:
        return initial_subgraph_state()

    def initialize_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphBatch,
        num_rollouts: int,
    ) -> SubgraphRolloutBatch:
        return initialize_subgraph_rollout_batch(
            batch=prepared_batch, num_rollouts=num_rollouts
        )

    def analyze_state(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        state: SubgraphState,
    ) -> SubgraphAnalysis:
        return analyze_subgraph_state(
            batch=prepared_batch, graph_idx=graph_idx, state=state
        )

    def analyze_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
    ) -> tuple[SubgraphAnalysis, ...]:
        return analyze_subgraph_rollout_batch(
            batch=prepared_batch, rollout_batch=rollout_batch
        )

    def admissible_answer_set(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> AdmissibleAnswerSet:
        return self.reward_model.admissible_answer_set(
            batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def count_gold_answers_in_graph(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        return self.reward_model.count_gold_answers_in_graph(
            batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def transition(
        self,
        *,
        rollout_batch: SubgraphRolloutBatch,
        chosen_actions: tuple[SubgraphAction, ...],
    ) -> SubgraphRolloutBatch:
        return transition_subgraph_rollout_batch(
            rollout_batch=rollout_batch,
            chosen_actions=chosen_actions,
        )

    def _encode_policy_state(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None,
    ) -> tuple[tuple[SubgraphAnalysis, ...], torch.Tensor]:
        if analyses is None:
            analyses = self.analyze_rollout_batch(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
            )
        state_features = self.encoder.encode_states(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        return analyses, state_features

    def encode_state_features(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | MappingABC[int, SubgraphAnalysis],
        state_indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder.encode_states(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
            state_indices=state_indices,
        )

    def compute_backward_log_prob(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        state: SubgraphState,
    ) -> float:
        removable_edge_ids = forward_valid_removable_edge_ids(
            batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
        )
        if not removable_edge_ids:
            raise RuntimeError(
                "Backward policy requires at least one forward-valid removable edge. "
                f"graph_idx={graph_idx} state={state.edge_ids}"
            )
        return -math.log(float(len(removable_edge_ids)))

    def backward_policy_name(self) -> str:
        return "uniform_forward_valid_edge_deletion"

    def compute_terminal_reward(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> SubgraphTerminalReward:
        return self.reward_model.compute_terminal_reward(
            batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def compute_terminal_log_reward(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[float, int, int, bool]:
        terminal_reward = self.compute_terminal_reward(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )
        return (
            float(terminal_reward.log_reward),
            int(terminal_reward.answer_set.count),
            int(terminal_reward.gold_answer_count),
            bool(terminal_reward.hit),
        )

    def compute_stop_log_reward(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[float, int, int, bool]:
        return self.compute_terminal_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def oracle_distance(
        self,
        *,
        prepared_batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> int:
        return self.reward_model.oracle_distance(
            batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def compute_log_flows(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None = None,
    ) -> torch.Tensor:
        _, state_features = self._encode_policy_state(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        return self.encoder.compute_log_flows(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            state_features=state_features,
        )

    def compute_log_flows_from_state_features(
        self,
        *,
        prepared_batch: SubgraphBatch,
        state_features: torch.Tensor,
        rollout_batch: SubgraphRolloutBatch | None = None,
        graph_ids: torch.Tensor | None = None,
        done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rollout_batch is not None:
            return self.encoder.compute_log_flows(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                state_features=state_features,
            )
        if graph_ids is None:
            raise ValueError(
                "compute_log_flows_from_state_features requires rollout_batch or graph_ids."
            )
        return self.encoder.compute_log_flows_for_graph_ids(
            prepared_batch=prepared_batch,
            graph_ids=graph_ids,
            state_features=state_features,
            done_mask=done_mask,
        )

    def build_state_action_distribution(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        flat_state_index: int,
        analysis: SubgraphAnalysis,
        state_feature: torch.Tensor,
        action_pruning: Mapping[str, Any] | None = None,
    ):
        return self.actor.build_state_distribution(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            flat_state_index=flat_state_index,
            analysis=analysis,
            state_feature=state_feature,
            action_pruning=action_pruning,
        )

    def build_action_distribution_from_state_features(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...],
        state_features: torch.Tensor,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphActionDistribution:
        return self.actor.build_action_distribution(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
            state_features=state_features,
            action_pruning=action_pruning,
        )

    def compute_action_distribution(
        self,
        *,
        prepared_batch: SubgraphBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None = None,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphActionDistribution:
        analyses, state_features = self._encode_policy_state(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        return self.build_action_distribution_from_state_features(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
            state_features=state_features,
            action_pruning=action_pruning,
        )

    @staticmethod
    def compute_target_log_probs(
        distribution: SubgraphActionDistribution,
    ) -> torch.Tensor:
        del distribution
        raise RuntimeError(
            "Flat action log-prob computation is not available in the strict "
            "hierarchical policy. The sampler computes staged log-probs directly."
        )


__all__ = ["SubgraphActionDistribution", "SubgraphPolicy", "SUBGRAPH_STATE_MODE"]
