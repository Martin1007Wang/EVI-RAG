from __future__ import annotations

import torch

from src.weaver.policy import PolicyOutput
from src.weaver.rollout.buffer import RolloutBuffer
from src.weaver.rollout.executor import StepContext
from src.weaver.rollout.sampling import action_log_probs


def write_policy_diagnostics(
    *,
    buffer: RolloutBuffer,
    step_out: PolicyOutput,
    step_context: StepContext,
    num_graphs: int,
) -> None:
    """
    Write target-policy diagnostics.

    target_continue_prob is the summed action probability of all Expand(e).
    target_stop_prob is the action probability of Stop.

    Edge entropy is action-level entropy over Expand(e) actions.
    """
    device = step_out.stop_logits.device
    active = step_context.active_mask.to(device=device, dtype=torch.bool)
    can_expand = step_context.can_expand.to(device=device, dtype=torch.bool)

    stop_logp, edge_logp = action_log_probs(
        stop_logits=step_out.stop_logits,
        edge_logits=step_out.edge_logits,
        candidate_batch_ids=step_out.candidate_batch_ids,
        can_expand=can_expand,
        batch_size=int(num_graphs),
    )

    target_stop_prob = torch.where(
        torch.isfinite(stop_logp),
        stop_logp.exp(),
        stop_logp.new_zeros(num_graphs),
    )
    edge_prob = torch.where(
        torch.isfinite(edge_logp),
        edge_logp.exp(),
        edge_logp.new_zeros(edge_logp.shape),
    )
    target_continue_prob = step_out.stop_logits.new_zeros(num_graphs)
    if edge_prob.numel() > 0:
        target_continue_prob = torch.bincount(
            step_out.candidate_batch_ids.to(device=device, dtype=torch.long),
            weights=edge_prob,
            minlength=int(num_graphs),
        ).to(dtype=torch.float32)

    stop_log_pf = stop_logp

    edge_entropy, edge_entropy_valid = edge_entropy_by_graph(
        edge_log_probs=edge_logp,
        candidate_batch_ids=step_out.candidate_batch_ids,
        step_context=step_context,
        num_graphs=int(num_graphs),
        device=device,
    )

    buffer.write_policy_step_diagnostics(
        t=step_context.t,
        active=active,
        target_stop_prob=target_stop_prob.to(dtype=torch.float32),
        target_continue_prob=target_continue_prob.to(dtype=torch.float32),
        stop_log_pf=stop_log_pf.to(dtype=torch.float32),
        action_valid_mask=can_expand,
        stop_tb_valid_mask=can_expand,
        edge_action_entropy=edge_entropy,
        edge_action_entropy_valid_mask=edge_entropy_valid,
        budget_exhausted_mask=step_context.budget_exhausted.to(
            device=device,
            dtype=torch.bool,
        ),
    )


def edge_entropy_by_graph(
    *,
    edge_log_probs: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    step_context: StepContext,
    num_graphs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Action-level Expand-edge entropy -sum_e P(Expand(e)|s) log P(Expand(e)|s).
    """
    num_graphs = int(num_graphs)

    entropy = torch.zeros(num_graphs, dtype=torch.float32, device=device)

    valid = step_context.active_mask.to(device=device, dtype=torch.bool)
    valid = valid & step_context.has_candidate.to(device=device, dtype=torch.bool)
    valid = valid & ~step_context.budget_exhausted.to(device=device, dtype=torch.bool)

    if edge_log_probs.numel() == 0 or not bool(valid.any()):
        return entropy, valid

    edge_batch = candidate_batch_ids.to(device=device, dtype=torch.long)
    log_probs = edge_log_probs.to(device=device, dtype=torch.float32)
    probs = torch.where(
        torch.isfinite(log_probs),
        log_probs.exp(),
        log_probs.new_zeros(log_probs.shape),
    )
    contribution = torch.where(
        torch.isfinite(log_probs),
        -(probs * log_probs),
        torch.zeros_like(log_probs),
    )

    entropy = torch.bincount(
        edge_batch,
        weights=contribution,
        minlength=num_graphs,
    ).to(dtype=torch.float32)

    entropy = entropy * valid.to(dtype=torch.float32)
    return entropy, valid


__all__ = [
    "edge_entropy_by_graph",
    "write_policy_diagnostics",
]
