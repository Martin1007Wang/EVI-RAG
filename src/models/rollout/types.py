from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

STOP_REASON_ACTION = 1
STOP_REASON_MAX_STEPS_REACHED = 2
STOP_REASON_DEAD_END = 3


@dataclass(frozen=True)
class RolloutResult:
    """Immutable rollout tensor bundle used by training/eval pipelines."""

    log_pf_sum: torch.Tensor
    stop_nodes: torch.Tensor
    num_moves: torch.Tensor
    num_steps: torch.Tensor
    stop_reason: torch.Tensor
    actions: Optional[torch.Tensor] = None
    log_pf_steps: Optional[torch.Tensor] = None
    log_pb_steps: Optional[torch.Tensor] = None
    log_f_steps: Optional[torch.Tensor] = None
    stop_logprob_steps: Optional[torch.Tensor] = None
    state_nodes_steps: Optional[torch.Tensor] = None
    continue_valid_steps: Optional[torch.Tensor] = None
    stop_valid_steps: Optional[torch.Tensor] = None
    log_pb_sum: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None
    policy_metrics: Optional[dict[str, float]] = None


__all__ = [
    "STOP_REASON_ACTION",
    "STOP_REASON_MAX_STEPS_REACHED",
    "STOP_REASON_DEAD_END",
    "RolloutResult",
]
