from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from .legality import apply_forward_legality
from .prefix_state import (
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
    """Compatibility wrapper around hard forward-legality masking."""

    return apply_forward_legality(
        distribution,
        state=state,
        max_steps=max_steps,
    )


def compute_constrained_policy_step(
    *,
    policy: SearchPolicyProtocol,
    prepared_batch: PreparedSearchBatch,
    state: SearchState,
    max_steps: int,
    required_edge_ids: torch.Tensor | None = None,
) -> ConstrainedPolicyStep:
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
