from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

_ZERO = 0
_FLOAT_ZERO = 0.0

_DEFAULT_HARD_TARGET_BONUS = 10.0
_DEFAULT_MIN_LOG_REWARD = -10.0


@dataclass
class RewardOutput:
    log_reward: torch.Tensor
    success: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class GraphFusionReward(nn.Module):
    """Log-linear reward with hard target bonus.

    Expects `node_is_target` as the single source of truth for terminal hits.
    """

    def __init__(
        self,
        *,
        hard_target_bonus: float = _DEFAULT_HARD_TARGET_BONUS,
        min_log_reward: float = _DEFAULT_MIN_LOG_REWARD,
    ) -> None:
        super().__init__()
        self.hard_target_bonus = float(hard_target_bonus)
        self.min_log_reward = float(min_log_reward)
        if self.hard_target_bonus < float(_ZERO):
            raise ValueError("hard_target_bonus must be >= 0.")
        self._has_bonus = self.hard_target_bonus > float(_ZERO)
        # precompute to avoid per-forward log/exp allocations
        self._log_bonus = float(math.log1p(self.hard_target_bonus)) if self._has_bonus else float(_FLOAT_ZERO)

    def forward(
        self,
        *,
        node_ptr: torch.Tensor,
        stop_node_locals: torch.Tensor,
        dummy_mask: torch.Tensor | None = None,
        node_is_target: torch.Tensor | None = None,
        node_is_answer: torch.Tensor | None = None,
        **_,
    ) -> RewardOutput:
        device = node_ptr.device
        num_nodes_total = int(node_ptr[-1].detach().item()) if node_ptr.numel() > 0 else 0

        stop_globals, valid_stop = self._resolve_stop_nodes(
            stop_node_locals=stop_node_locals,
            node_ptr=node_ptr,
        )
        if dummy_mask is not None:
            dummy_mask = dummy_mask.to(device=device, dtype=torch.bool)
        # prefer unified naming; fallback to legacy node_is_answer
        if node_is_target is None:
            node_is_target = node_is_answer
        if node_is_target is None:
            raise ValueError("node_is_target is required for reward computation.")
        node_is_target = node_is_target.to(device=device, dtype=torch.bool)
        if int(node_is_target.numel()) != num_nodes_total:
            raise ValueError("node_is_target length mismatch with node_ptr.")

        hard_hit = node_is_target[stop_globals] & valid_stop
        if dummy_mask is not None:
            hard_hit = hard_hit & (~dummy_mask)
        base_log = torch.full(stop_globals.shape, self.min_log_reward, device=device, dtype=torch.float32)
        bonus_log = base_log.new_full(base_log.shape, self._log_bonus) if self._has_bonus else base_log.new_zeros(base_log.shape)
        log_reward = torch.where(hard_hit, bonus_log, base_log)
        success = hard_hit.to(dtype=torch.bool)
        return RewardOutput(
            log_reward=log_reward,
            success=success,
        )

    @staticmethod
    def _resolve_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stop_locals = stop_node_locals.to(dtype=torch.long)
        valid_stop = stop_locals >= _ZERO
        node_counts = node_ptr[1:] - node_ptr[:-1]
        if valid_stop.any():
            out_of_range = stop_locals >= node_counts
            if bool(out_of_range.any().detach().tolist()):
                raise ValueError("stop_node_locals out of range for node_ptr.")
        node_offset = node_ptr[:-1]
        stop_globals = node_offset + stop_locals.clamp(min=_ZERO)
        return stop_globals, valid_stop


__all__ = ["RewardOutput", "GraphFusionReward"]
