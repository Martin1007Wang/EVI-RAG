from __future__ import annotations

import torch

from src.graph.segments import scatter_log_softmax
from src.weaver.policy import CandidateEdges, PolicyStepOutput
from src.weaver.rollout.buffers import RolloutBuffer
from src.weaver.rollout.executor import budget_exhausted_mask, has_candidate
from src.weaver.rollout.sampling import (
    EXPAND_OPTION_INDEX,
    STOP_OPTION_INDEX,
    option_action_log_probs,
)


def write_policy_diagnostics(
    *,
    buffer: RolloutBuffer,
    step_out: PolicyStepOutput,
    active: torch.Tensor,
    t: int,
    num_graphs: int,
    remaining_budget: torch.Tensor,
) -> None:
    """
    Write target-policy diagnostics.

    target_continue_prob is option-level Expand probability.
    target_stop_prob is option-level Stop probability.

    Edge entropy is conditional entropy:
        H[P(edge | s, Expand)]
    """
    device = step_out.stop_logits.device
    active = active.to(device=device, dtype=torch.bool)

    has_edge = has_candidate(
        candidate_batch_index=step_out.candidates.batch_index,
        num_graphs=int(num_graphs),
        device=device,
    )

    exhausted = budget_exhausted_mask(
        remaining_budget,
        num_graphs=int(num_graphs),
        device=device,
    )

    can_expand = active & has_edge & ~exhausted

    type_logp, edge_logp = option_action_log_probs(
        stop_logits=step_out.stop_logits,
        expand_logits=step_out.expand_logits,
        candidates=step_out.candidates,
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
        candidates=step_out.candidates,
        active=active,
        has_candidate_edge=has_edge,
        remaining_budget=remaining_budget,
        num_graphs=int(num_graphs),
        device=device,
    )

    buffer.write_policy_step_diagnostics(
        t=t,
        active=active,
        target_stop_prob=target_stop_prob.to(dtype=torch.float32),
        target_continue_prob=target_continue_prob.to(dtype=torch.float32),
        stop_log_pf=stop_log_pf.to(dtype=torch.float32),
        action_valid_mask=can_expand,
        stop_tb_valid_mask=can_expand,
        edge_action_entropy=edge_entropy,
        edge_action_entropy_valid_mask=edge_entropy_valid,
        budget_exhausted_mask=exhausted,
    )


def edge_entropy_by_graph(
    *,
    candidates: CandidateEdges,
    active: torch.Tensor,
    has_candidate_edge: torch.Tensor,
    remaining_budget: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional edge entropy H[P(edge | s, Expand)] per graph.
    """
    num_graphs = int(num_graphs)

    entropy = torch.zeros(num_graphs, dtype=torch.float32, device=device)

    valid = active.to(device=device, dtype=torch.bool)
    valid = valid & has_candidate_edge.to(device=device, dtype=torch.bool)
    valid = valid & ~budget_exhausted_mask(
        remaining_budget,
        num_graphs=num_graphs,
        device=device,
    )

    if len(candidates) == 0 or not bool(valid.any()):
        return entropy, valid

    edge_batch = candidates.batch_index.to(device=device, dtype=torch.long)
    logits = candidates.expand_logits.to(device=device, dtype=torch.float32)

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