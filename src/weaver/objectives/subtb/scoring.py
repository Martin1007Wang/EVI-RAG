from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack
from src.weaver.policy import ForwardPolicy, PolicyActionSpace, PolicyInput, uniform_backward_log_prob
from src.weaver.reward import EvidenceStateScoreOutput

from .batch import SubTBBatch


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


def score_subtb_batch(
    *,
    batch: SubTBBatch,
    policy: ForwardPolicy,
    features: FeaturePack,
    policy_input: PolicyInput,
    graph_context: GraphContext,
    reward: EvidenceStateScoreOutput,
    action_space: PolicyActionSpace | None = None,
) -> SubTBPolicyScores:
    states = batch.states
    del policy_input
    # Training scoring must use a fresh PolicyInput from the current FeaturePack.
    # Reusing rollout/no_grad caches here can detach question/edge/relation tensors
    # and silently break gradient flow or retain an old graph across optimizer steps.
    scoring_policy_input = policy.build_policy_input(
        features,
        graph_context=graph_context,
    )
    action_space = policy.prepare_action_space(state=states, graph_context=graph_context) if action_space is None else action_space
    output = policy(
        state=states,
        features=features,
        graph_context=graph_context,
        policy_input=scoring_policy_input,
        action_space=action_space,
        compute_log_flow=True,
    )
    rows = torch.arange(states.num_states, device=states.device)
    log_flow_base = output.require_log_flow_base().float()
    log_flow = log_flow_base + reward.state_potential.detach().float()
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
    backward_step_logp = torch.zeros_like(step_log_prob, dtype=torch.float32)
    if int(batch.step_traj_ids.numel()) > 0:
        child_state_ids = batch.prefix_state_ids[batch.step_traj_ids, batch.step_ids + 1]
        non_root = states.edge_count.gt(0)
        log_backward_by_state = torch.zeros(states.num_states, dtype=torch.float32, device=states.device)
        if bool(non_root.any()):
            non_root_rows = non_root.nonzero(as_tuple=True)[0]
            log_backward_by_state[non_root_rows] = uniform_backward_log_prob(
                child_state=states.take(non_root_rows),
                graph_context=graph_context,
            )
        backward_step_logp[batch.step_traj_ids, batch.step_ids] = log_backward_by_state.index_select(0, child_state_ids)
    terminal_stop_logp_by_traj = stop_log_prob_by_state.index_select(0, batch.terminal_state_ids)
    return SubTBPolicyScores(
        frontier_count=frontier_count,
        log_flow=log_flow,
        stop_log_prob_by_state=stop_log_prob_by_state,
        step_log_prob=step_log_prob,
        backward_step_log_prob=backward_step_logp,
        terminal_stop_logp_by_traj=terminal_stop_logp_by_traj,
        frontier_row_ids=output.frontier.row_ids,
        frontier_edge_ids=output.frontier.edge_ids,
        frontier_log_prob=output.action_log_prob[states.num_states :],
    )
