from __future__ import annotations

import torch

from src.graph.segments import scatter_log_softmax
from src.weaver.policy import PolicyOutput
from src.weaver.rollout.buffer import RolloutBuffer
from src.weaver.rollout.executor import StepContext
from src.weaver.rollout.sampling import (
    EXPAND_OPTION_INDEX,
    STOP_OPTION_INDEX,
    option_action_log_probs,
)


def write_policy_diagnostics(
    *,
    buffer: RolloutBuffer,
    step_out: PolicyOutput,
    step_context: StepContext,
    num_graphs: int,
) -> None:
    """
    Write target-policy diagnostics.

    target_continue_prob is option-level Expand probability.
    target_stop_prob is option-level Stop probability.

    Edge entropy is conditional entropy:
        H[P(edge | s, Expand)]
    """
    device = step_out.stop_logits.device
    active = step_context.active_mask.to(device=device, dtype=torch.bool)
    can_expand = step_context.can_expand.to(device=device, dtype=torch.bool)

    type_logp, edge_logp = option_action_log_probs(
        stop_logits=step_out.stop_logits,
        expand_logits=step_out.expand_logits,
        edge_logits=step_out.edge_logits,
        candidate_batch_ids=step_out.candidate_batch_ids,
        can_expand=can_expand,
        batch_size=int(num_graphs),
    )
    del edge_logp

    target_stop_prob = torch.where(
        torch.isfinite(type_logp[:, STOP_OPTION_INDEX]),
        type_logp[:, STOP_OPTION_INDEX].exp(),
        type_logp.new_zeros(num_graphs),
    )

    target_continue_prob = torch.where(
        torch.isfinite(type_logp[:, EXPAND_OPTION_INDEX]),
        type_logp[:, EXPAND_OPTION_INDEX].exp(),
        type_logp.new_zeros(num_graphs),
    )

    stop_log_pf = type_logp[:, STOP_OPTION_INDEX]

    edge_entropy, edge_entropy_valid = edge_entropy_by_graph(
        edge_logits=step_out.edge_logits,
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
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    step_context: StepContext,
    num_graphs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional edge entropy H[P(edge | s, Expand)] per graph.
    """
    num_graphs = int(num_graphs)

    entropy = torch.zeros(num_graphs, dtype=torch.float32, device=device)

    valid = step_context.active_mask.to(device=device, dtype=torch.bool)
    valid = valid & step_context.has_candidate.to(device=device, dtype=torch.bool)
    valid = valid & ~step_context.budget_exhausted.to(device=device, dtype=torch.bool)

    if edge_logits.numel() == 0 or not bool(valid.any()):
        return entropy, valid

    edge_batch = candidate_batch_ids.to(device=device, dtype=torch.long)
    logits = edge_logits.to(device=device, dtype=torch.float32)

    log_probs = scatter_log_softmax(
        logits,
        edge_batch,
        num_graphs,
    )

    probs = log_probs.exp()
    contribution = -(probs * log_probs)

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
