from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack
from src.weaver.objectives.subtb.batch import SubTBBatch
from src.weaver.policy import (
    BackwardScoringModel,
    ForwardPolicy,
    PolicyActionSpace,
    PolicyInput,
)
from src.weaver.reward import EvidenceStateScoreOutput


@dataclass(frozen=True, slots=True)
class ForwardSubTBPolicyScores:
    frontier_count: torch.Tensor
    log_flow: torch.Tensor
    stop_log_prob_by_state: torch.Tensor
    step_log_prob: torch.Tensor
    terminal_stop_logp_by_traj: torch.Tensor
    frontier_row_ids: torch.Tensor | None = None
    frontier_edge_ids: torch.Tensor | None = None
    frontier_log_prob: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class SubTBPolicyScores:
    frontier_count: torch.Tensor
    log_flow: torch.Tensor
    stop_log_prob_by_state: torch.Tensor
    step_log_prob: torch.Tensor
    backward_step_log_prob: torch.Tensor
    terminal_stop_logp_by_traj: torch.Tensor
    frontier_row_ids: torch.Tensor | None = None
    frontier_edge_ids: torch.Tensor | None = None
    frontier_log_prob: torch.Tensor | None = None


def score_forward_subtb_batch(
    *,
    batch: SubTBBatch,
    policy: ForwardPolicy,
    features: FeaturePack,
    policy_input: PolicyInput,
    graph_context: GraphContext,
    reward: EvidenceStateScoreOutput,
    action_space: PolicyActionSpace | None = None,
) -> ForwardSubTBPolicyScores:
    states = batch.states
    del action_space
    scoring_policy_input = policy_input
    action_space = (
        policy.prepare_action_space(
            state=states,
            graph_context=graph_context,
            policy_input=scoring_policy_input,
            training=True,
            scoring=True,
        )
    )
    output = policy(
        state=states,
        features=features,
        graph_context=graph_context,
        policy_input=scoring_policy_input,
        action_space=action_space,
        compute_log_flow=True,
    )
    rows = torch.arange(states.num_states, device=states.device)
    log_flow = output.require_log_flow_base().float() + reward.state_potential.detach().float()
    stop_log_prob_by_state = output.gather_log_prob(
        row_ids=rows,
        edge_ids=torch.full((states.num_states,), -1, dtype=torch.long, device=states.device),
    )
    frontier_count = torch.bincount(output.frontier.row_ids, minlength=states.num_states).float()
    step_log_prob = torch.zeros_like(batch.trajectories.edge_logp, dtype=torch.float32)
    if int(batch.step_parent_state_ids.numel()) > 0:
        step_log_prob[batch.step_traj_ids, batch.step_ids] = output.gather_log_prob(
            row_ids=batch.step_parent_state_ids,
            edge_ids=batch.step_edge_ids,
        )
    terminal_stop_logp_by_traj = stop_log_prob_by_state.index_select(0, batch.terminal_state_ids)
    return ForwardSubTBPolicyScores(
        frontier_count=frontier_count,
        log_flow=log_flow,
        stop_log_prob_by_state=stop_log_prob_by_state,
        step_log_prob=step_log_prob,
        terminal_stop_logp_by_traj=terminal_stop_logp_by_traj,
        frontier_row_ids=output.frontier.row_ids,
        frontier_edge_ids=output.frontier.edge_ids,
        frontier_log_prob=output.action_log_prob[states.num_states :],
    )


def score_backward_step_log_probs(
    *,
    batch: SubTBBatch,
    model: BackwardScoringModel,
    features: FeaturePack,
    graph_context: GraphContext,
) -> torch.Tensor:
    backward_step_logp = torch.zeros_like(batch.trajectories.edge_logp, dtype=torch.float32)
    if int(batch.step_traj_ids.numel()) == 0:
        return backward_step_logp
    policy_input = model.build_policy_input(features)
    output = model(
        child_state=batch.states.take(batch.step_child_state_ids),
        graph_context=graph_context,
        policy_input=policy_input,
    )
    gathered = output.gather_log_prob(
        row_ids=torch.arange(batch.step_child_state_ids.numel(), device=batch.states.device, dtype=torch.long),
        edge_ids=batch.step_edge_ids,
    )
    backward_step_logp[batch.step_traj_ids, batch.step_ids] = gathered
    return backward_step_logp


def combine_subtb_scores(
    *,
    forward_scores: ForwardSubTBPolicyScores,
    backward_step_log_prob: torch.Tensor,
) -> SubTBPolicyScores:
    return SubTBPolicyScores(
        frontier_count=forward_scores.frontier_count,
        log_flow=forward_scores.log_flow,
        stop_log_prob_by_state=forward_scores.stop_log_prob_by_state,
        step_log_prob=forward_scores.step_log_prob,
        backward_step_log_prob=backward_step_log_prob,
        terminal_stop_logp_by_traj=forward_scores.terminal_stop_logp_by_traj,
        frontier_row_ids=forward_scores.frontier_row_ids,
        frontier_edge_ids=forward_scores.frontier_edge_ids,
        frontier_log_prob=forward_scores.frontier_log_prob,
    )


__all__ = [
    "ForwardSubTBPolicyScores",
    "SubTBPolicyScores",
    "combine_subtb_scores",
    "score_backward_step_log_probs",
    "score_forward_subtb_batch",
]
