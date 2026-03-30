from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.graph import TrajectoryBatch
from src.utils.segment_ops import segment_logsumexp_1d

from .actor import SubgraphActionDistribution, SubgraphActor
from .encoder import SubgraphEncoder
from .prepared_batch import SubgraphPreparedBatch
from .reward import SubgraphRewardModel
from .state import (
    SubgraphAction,
    SubgraphAnalysis,
    SubgraphRolloutBatch,
    SubgraphState,
    analyze_subgraph_rollout_batch,
    analyze_subgraph_state,
    initial_subgraph_state,
    initialize_subgraph_rollout_batch,
    transition_subgraph_rollout_batch,
)


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
        subgraph_reward: dict[str, Any],
        subgraph_proposal: dict[str, Any],
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
        self.reward_model = SubgraphRewardModel(
            max_steps=self.max_steps, **subgraph_reward
        )
        self.actor = SubgraphActor(
            hidden_dim=int(backbone["hidden_dim"]),
            max_steps=self.max_steps,
            actor=actor,
            subgraph_proposal=subgraph_proposal,
        )

    def prepare_batch(self, batch: TrajectoryBatch) -> SubgraphPreparedBatch:
        return self.encoder.prepare_batch(batch)

    @staticmethod
    def initial_state() -> SubgraphState:
        return initial_subgraph_state()

    def initialize_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        num_rollouts: int,
    ) -> SubgraphRolloutBatch:
        return initialize_subgraph_rollout_batch(
            prepared_batch=prepared_batch,
            num_rollouts=num_rollouts,
        )

    def analyze_state(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        state: SubgraphState,
    ) -> SubgraphAnalysis:
        return analyze_subgraph_state(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
        )

    def analyze_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
    ) -> tuple[SubgraphAnalysis, ...]:
        return analyze_subgraph_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
        )

    def count_gold_answers(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        return self.reward_model.count_gold_answers(
            prepared_batch=prepared_batch,
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
        prepared_batch: SubgraphPreparedBatch,
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

    def compute_expand_log_reward(
        self,
        *,
        current_analysis: SubgraphAnalysis,
        next_analysis: SubgraphAnalysis,
        prepared_batch: SubgraphPreparedBatch | None = None,
        graph_idx: int | None = None,
    ) -> float:
        return self.reward_model.compute_expand_log_reward(
            current_analysis=current_analysis,
            next_analysis=next_analysis,
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
        )

    def compute_stop_log_reward(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[float, int, bool]:
        return self.reward_model.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def oracle_distance(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> int:
        return self.reward_model.oracle_distance(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def compute_log_flows(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
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

    def compute_action_distribution(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None = None,
    ) -> SubgraphActionDistribution:
        analyses, state_features = self._encode_policy_state(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        return self.actor.build_action_distribution(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
            state_features=state_features,
        )

    @staticmethod
    def compute_target_log_probs(
        distribution: SubgraphActionDistribution,
    ) -> torch.Tensor:
        if int(distribution.logits.numel()) == 0:
            return distribution.logits
        lse, _ = segment_logsumexp_1d(
            values=distribution.logits,
            segment_ids=distribution.segment_ids,
            num_segments=int(distribution.flat_state_indices.numel()),
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        return distribution.logits - lse.index_select(0, distribution.segment_ids)

    def compute_proposal_bias(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        distribution: SubgraphActionDistribution,
        proposal_bias_scale: float,
    ) -> torch.Tensor:
        del prepared_batch
        return self.actor.compute_proposal_bias(
            distribution=distribution,
            proposal_bias_scale=proposal_bias_scale,
        )


__all__ = ["SubgraphActionDistribution", "SubgraphPolicy"]
