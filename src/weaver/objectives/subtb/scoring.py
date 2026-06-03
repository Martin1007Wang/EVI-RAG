from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack
from src.weaver.policy import ForwardPolicy, PolicyActionSpace, PolicyCache, uniform_backward_log_prob

from .batch import SubTBBatch


@dataclass(frozen=True, slots=True)
class SubTBPolicyScores:
    action_space: PolicyActionSpace
    action_count: torch.Tensor
    frontier_count: torch.Tensor
    log_flow: torch.Tensor
    stop_log_prob_by_state: torch.Tensor
    log_backward_by_state: torch.Tensor
    step_log_prob: torch.Tensor
    forward_prefix_by_traj: torch.Tensor
    backward_prefix_by_traj: torch.Tensor
    terminal_stop_logp_by_traj: torch.Tensor


def score_subtb_batch(
    *,
    batch: SubTBBatch,
    policy: ForwardPolicy,
    features: FeaturePack,
    cache: PolicyCache,
    graph_context: GraphContext,
) -> SubTBPolicyScores:
    states = batch.states
    action_space = policy.prepare_action_space(state=states, graph_context=graph_context)
    action_count = torch.bincount(action_space.frontier.row_ids, minlength=states.num_states)
    output = policy(
        state=states,
        features=features,
        graph_context=graph_context,
        cache=cache,
        action_space=action_space,
        compute_log_flow=True,
    )
    rows = torch.arange(states.num_states, device=states.device)
    log_flow = output.require_log_flow().float()
    stop_log_prob_by_state = output.gather_log_prob(
        row_ids=rows,
        edge_ids=torch.full((states.num_states,), -1, dtype=torch.long, device=states.device),
    )
    frontier_count = torch.bincount(output.frontier.row_ids, minlength=states.num_states).float()
    # Re-score recorded trajectory edges under the current policy parameters.
    # Replay rows have no historical policy logits, and no importance correction is applied.
    step_log_prob = torch.zeros_like(batch.trajectories.edge_logp, dtype=torch.float32)
    if int(batch.step_parent_state_ids.numel()) > 0:
        step_log_prob[batch.step_traj_ids, batch.step_ids] = output.gather_log_prob(
            row_ids=batch.step_parent_state_ids,
            edge_ids=batch.step_edge_ids,
        )
    log_backward_by_state = torch.zeros(states.num_states, dtype=torch.float32, device=states.device)
    non_root = states.edge_count.gt(0)
    if bool(non_root.any()):
        log_backward_by_state[non_root] = uniform_backward_log_prob(
            child_state=states.take(non_root.nonzero(as_tuple=True)[0]),
            graph_context=graph_context,
        )

    forward_prefix_by_traj = torch.cat(
        [
            step_log_prob.new_zeros((batch.trajectories.num_trajectories, 1)),
            torch.cumsum(step_log_prob, dim=1),
        ],
        dim=1,
    )
    backward_step_logp = torch.zeros_like(step_log_prob, dtype=torch.float32)
    if int(batch.trajectories.num_trajectories) > 0 and batch.trajectories.budget > 0:
        valid_children = batch.valid_steps
        child_state_ids = batch.prefix_state_ids[:, 1:]
        backward_step_logp[valid_children] = log_backward_by_state.index_select(0, child_state_ids[valid_children])
    backward_prefix_by_traj = torch.cat(
        [
            backward_step_logp.new_zeros((batch.trajectories.num_trajectories, 1)),
            torch.cumsum(backward_step_logp, dim=1),
        ],
        dim=1,
    )
    terminal_stop_logp_by_traj = stop_log_prob_by_state.index_select(0, batch.terminal_state_ids)
    return SubTBPolicyScores(
        action_space=action_space,
        action_count=action_count,
        frontier_count=frontier_count,
        log_flow=log_flow,
        stop_log_prob_by_state=stop_log_prob_by_state,
        log_backward_by_state=log_backward_by_state,
        step_log_prob=step_log_prob,
        forward_prefix_by_traj=forward_prefix_by_traj,
        backward_prefix_by_traj=backward_prefix_by_traj,
        terminal_stop_logp_by_traj=terminal_stop_logp_by_traj,
    )
