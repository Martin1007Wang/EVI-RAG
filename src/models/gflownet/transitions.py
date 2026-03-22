from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from .types import (
    ForwardActionDistribution,
    PreparedSearchBatch,
    SearchPolicyProtocol,
    SearchState,
)


@dataclass(frozen=True)
class ConstrainedPolicyStep:
    distribution: ForwardActionDistribution
    move_log_probs: torch.Tensor
    move_probs: torch.Tensor
    has_values: torch.Tensor


ConstrainedForwardStep = ConstrainedPolicyStep


def apply_forward_constraints(
    distribution: ForwardActionDistribution,
    *,
    state: SearchState,
    max_steps: int,
) -> ForwardActionDistribution:
    active_flat = ~state.flatten_done_mask()
    num_steps_flat = state.flatten_num_steps()
    at_horizon = active_flat & (num_steps_flat >= max_steps)
    edge_logits = distribution.edge_logits
    if int(edge_logits.numel()) > 0:
        neg_inf = torch.tensor(
            float("-inf"),
            device=edge_logits.device,
            dtype=edge_logits.dtype,
        )
        edge_agent_batch = distribution.edge_agent_batch
        edge_at_horizon = at_horizon.index_select(0, edge_agent_batch)
        submit_mask = (
            distribution.is_submit.to(dtype=torch.bool)
            if distribution.is_submit is not None
            else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
        )
        edge_logits = edge_logits.masked_fill(edge_at_horizon & (~submit_mask), neg_inf)
    return ForwardActionDistribution(
        edge_logits=edge_logits,
        edge_agent_batch=distribution.edge_agent_batch,
        edge_ids=distribution.edge_ids,
        target_nodes=distribution.target_nodes,
        out_degrees=distribution.out_degrees,
        is_submit=distribution.is_submit,
        current_log_f=distribution.current_log_f,
    )


def compute_constrained_policy_step(
    *,
    policy: SearchPolicyProtocol,
    prepared_batch: PreparedSearchBatch,
    state: SearchState,
    max_steps: int,
    disable_candidate_shortlist: bool = False,
    required_edge_ids: torch.Tensor | None = None,
) -> ConstrainedPolicyStep:
    distribution: ForwardActionDistribution
    if disable_candidate_shortlist:
        no_shortlist_fn = getattr(
            policy, "compute_forward_distribution_without_shortlist", None
        )
        if callable(no_shortlist_fn):
            distribution = cast(
                ForwardActionDistribution,
                no_shortlist_fn(prepared_batch, state),
            )
        else:
            try:
                distribution = cast(
                    ForwardActionDistribution,
                    policy.compute_forward_distribution(
                        prepared_batch,
                        state,
                        required_edge_ids=required_edge_ids,
                    ),
                )
            except TypeError as error:
                if "unexpected keyword argument" not in str(error):
                    raise
                distribution = cast(
                    ForwardActionDistribution,
                    policy.compute_forward_distribution(prepared_batch, state),
                )
    else:
        try:
            distribution = cast(
                ForwardActionDistribution,
                policy.compute_forward_distribution(
                    prepared_batch,
                    state,
                    required_edge_ids=required_edge_ids,
                ),
            )
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            distribution = cast(
                ForwardActionDistribution,
                policy.compute_forward_distribution(prepared_batch, state),
            )
    distribution = cast(ForwardActionDistribution, distribution)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=max_steps,
    )
    move_log_probs, _, has_values = policy.compute_move_log_probs(distribution)
    move_probs = (
        move_log_probs.exp() if int(move_log_probs.numel()) > 0 else move_log_probs
    )
    return ConstrainedPolicyStep(
        distribution=distribution,
        move_log_probs=move_log_probs,
        move_probs=move_probs.to(dtype=torch.float32),
        has_values=has_values,
    )


compute_constrained_forward_step = compute_constrained_policy_step


__all__ = [
    "ConstrainedForwardStep",
    "ConstrainedPolicyStep",
    "apply_forward_constraints",
    "compute_constrained_forward_step",
    "compute_constrained_policy_step",
]
