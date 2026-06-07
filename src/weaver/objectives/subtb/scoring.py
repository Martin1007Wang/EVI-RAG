from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack
from src.weaver.policy import ForwardPolicy, PolicyActionSpace, PolicyInput
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
    backward_aux_logprob: torch.Tensor | None = None


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
    recorded_edge_ids_by_state = _recorded_edge_ids_by_state(batch=batch, device=states.device)
    scoring_policy_input = policy.build_policy_input(
        features,
        graph_context=graph_context,
    )
    action_space = (
        policy.prepare_action_space(
            state=states,
            graph_context=graph_context,
            policy_input=scoring_policy_input,
            training=True,
            recorded_edge_ids_by_state=recorded_edge_ids_by_state,
        )
        if action_space is None
        else action_space
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
    backward_step_logp = torch.zeros_like(batch.trajectories.edge_logp, dtype=torch.float32)
    backward_aux_logprob = torch.empty(0, dtype=torch.float32, device=states.device)
    if int(batch.step_traj_ids.numel()) > 0:
        backward_output = policy.score_backward(
            child_state=states.take(batch.step_child_state_ids),
            graph_context=graph_context,
            policy_input=scoring_policy_input,
            forward_output=type(output)(
                action_logits=output.action_logits,
                action_row_ids=output.action_row_ids,
                action_edge_ids=output.action_edge_ids,
                frontier=output.frontier,
                log_flow_base=output.log_flow_base,
                state_h=output.require_state_h().index_select(0, batch.step_child_state_ids),
            ),
        )
        backward_aux_logprob = backward_output.gather_log_prob(
            row_ids=torch.arange(batch.step_child_state_ids.numel(), device=states.device, dtype=torch.long),
            edge_ids=batch.step_edge_ids,
        )
        if bool(backward_aux_logprob.requires_grad):
            backward_step_logp = backward_step_logp.clone()
            backward_step_logp[batch.step_traj_ids, batch.step_ids] = backward_aux_logprob
            backward_step_logp = backward_step_logp.detach()
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
        backward_aux_logprob=backward_aux_logprob,
    )


def _recorded_edge_ids_by_state(*, batch: SubTBBatch, device: torch.device) -> torch.Tensor:
    width = int(batch.states.edge_capacity)
    if int(batch.step_parent_state_ids.numel()) == 0:
        return torch.full((batch.states.num_states, 0), -1, dtype=torch.long, device=device)
    slots = torch.full((batch.states.num_states, width), -1, dtype=torch.long, device=device)
    fill = torch.zeros((batch.states.num_states,), dtype=torch.long, device=device)
    for idx in range(int(batch.step_parent_state_ids.numel())):
        row = int(batch.step_parent_state_ids[idx].item())
        pos = int(fill[row].item())
        if pos >= width:
            continue
        edge_id = int(batch.step_edge_ids[idx].item())
        existing = slots[row, :pos]
        if bool(existing.eq(edge_id).any()):
            continue
        slots[row, pos] = edge_id
        fill[row] += 1
    return slots
