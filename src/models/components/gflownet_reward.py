from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch_scatter import scatter_max, scatter_min

_ZERO = 0
_ONE = 1
_FLOAT_ZERO = 0.0
_NEG_INF = float("-inf")

_DEFAULT_HARD_TARGET_BONUS = 10.0
_DEFAULT_MIN_LOG_REWARD = -10.0
_DEFAULT_STEP_BIAS = 0.0
_DEFAULT_POTENTIAL_WEIGHT = 0.0
_DEFAULT_POTENTIAL_GAMMA = 1.0
_POTENTIAL_UNREACHABLE_OFFSET = 1.0
_DEFAULT_POTENTIAL_SCHEDULE = "linear"
_DEFAULT_POTENTIAL_WEIGHT_END = 0.0


@dataclass
class RewardOutput:
    reward: torch.Tensor
    log_reward: torch.Tensor
    success: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class GraphFusionReward(nn.Module):
    """Log-linear reward with projected semantic guidance, hard masking, and potential shaping."""

    def __init__(
        self,
        *,
        hard_target_bonus: float = _DEFAULT_HARD_TARGET_BONUS,
        min_log_reward: float = _DEFAULT_MIN_LOG_REWARD,
        step_bias: float = _DEFAULT_STEP_BIAS,
        potential_weight: float = _DEFAULT_POTENTIAL_WEIGHT,
        potential_gamma: float = _DEFAULT_POTENTIAL_GAMMA,
        potential_unreachable_offset: float = _POTENTIAL_UNREACHABLE_OFFSET,
        potential_schedule: str = _DEFAULT_POTENTIAL_SCHEDULE,
        potential_weight_end: float = _DEFAULT_POTENTIAL_WEIGHT_END,
        potential_anneal_epochs: int | None = None,
        potential_pure_phase_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.hard_target_bonus = float(hard_target_bonus)
        self.min_log_reward = float(min_log_reward)
        self.step_bias = float(step_bias)
        self.potential_weight = float(potential_weight)
        self.potential_gamma = float(potential_gamma)
        self.potential_unreachable_offset = float(potential_unreachable_offset)
        self.potential_schedule = str(potential_schedule)
        self.potential_weight_end = float(potential_weight_end)
        self.potential_anneal_epochs = potential_anneal_epochs
        self.potential_pure_phase_ratio = potential_pure_phase_ratio
        if self.hard_target_bonus < float(_ZERO):
            raise ValueError("hard_target_bonus must be >= 0.")
        if self.potential_weight < float(_ZERO):
            raise ValueError("potential_weight must be >= 0.")
        if self.potential_weight != float(_ZERO) and self.potential_gamma != float(_ONE):
            raise ValueError("potential_gamma must be 1.0 when potential shaping is enabled.")
        if self.potential_unreachable_offset < float(_ZERO):
            raise ValueError("potential_unreachable_offset must be >= 0.")
        if self.potential_weight != float(_ZERO) or self.potential_weight_end != float(_ZERO):
            raise ValueError("Potential shaping is disabled; set potential_weight and potential_weight_end to 0.")

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        answer_node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
        node_min_dists: torch.Tensor | None = None,
        stop_node_locals: torch.Tensor,
        answer_hit: torch.Tensor,
        path_length: torch.Tensor | None = None,
        dummy_mask: torch.Tensor | None = None,
        **_,
    ) -> RewardOutput:
        num_graphs = int(node_ptr.numel() - 1)
        device = node_ptr.device

        stop_globals, valid_stop = self._resolve_stop_nodes(
            stop_node_locals=stop_node_locals,
            node_ptr=node_ptr,
        )
        node_is_start, node_is_answer = self._build_node_flags(
            num_nodes_total=int(node_ptr[-1].item()),
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            device=device,
        )
        invalid_nodes = self._build_invalid_nodes(
            edge_index=edge_index,
            node_is_start=node_is_start,
            node_is_answer=node_is_answer,
        )
        invalid_stop = invalid_nodes[stop_globals] | (~valid_stop)

        hard_hit = node_is_answer[stop_globals] & (~invalid_stop)
        log_reward = torch.where(
            hard_hit,
            torch.full_like(stop_globals, _FLOAT_ZERO, dtype=torch.float32),
            torch.full_like(stop_globals, self.min_log_reward, dtype=torch.float32),
        )
        if self.hard_target_bonus > float(_ZERO):
            bonus_log = torch.log(torch.full_like(log_reward, self.hard_target_bonus))
            hard_log = torch.logaddexp(torch.zeros_like(log_reward), bonus_log)
            log_reward = torch.where(hard_hit, hard_log, log_reward)
        if self.step_bias != float(_ZERO) and path_length is not None:
            length = path_length.to(device=log_reward.device, dtype=log_reward.dtype)
            log_reward = log_reward + (self.step_bias * length)
        if self.potential_weight != float(_ZERO):
            log_reward = log_reward + self._compute_potential_shaping(
                node_min_dists=node_min_dists,
                node_ptr=node_ptr,
                start_node_locals=start_node_locals,
                start_ptr=start_ptr,
                stop_globals=stop_globals,
                valid_stop=valid_stop,
                dummy_mask=dummy_mask,
                dtype=log_reward.dtype,
            )

        min_log = torch.full_like(log_reward, self.min_log_reward)
        log_reward = torch.where(invalid_stop, min_log, log_reward)
        reward = hard_hit.to(dtype=torch.float32)
        if dummy_mask is not None:
            dummy_mask = dummy_mask.to(device=device, dtype=torch.bool)
            log_reward = torch.where(dummy_mask, torch.full_like(log_reward, _NEG_INF), log_reward)
            reward = torch.where(dummy_mask, torch.zeros_like(reward), reward)
            answer_hit = torch.where(dummy_mask, torch.zeros_like(answer_hit), answer_hit)
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=answer_hit.to(dtype=torch.float32),
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
            if bool(out_of_range.any().item()):
                raise ValueError("stop_node_locals out of range for node_ptr.")
        node_offset = node_ptr[:-1]
        stop_globals = node_offset + stop_locals.clamp(min=_ZERO)
        return stop_globals, valid_stop

    @staticmethod
    def _build_node_flags(
        *,
        num_nodes_total: int,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if start_node_locals.numel() > _ZERO:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if answer_node_locals.numel() > _ZERO:
            node_is_answer[answer_node_locals] = True
        return node_is_start, node_is_answer

    @staticmethod
    def _build_invalid_nodes(
        *,
        edge_index: torch.Tensor,
        node_is_start: torch.Tensor,
        node_is_answer: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes_total = int(node_is_start.numel())
        neighbors = torch.zeros(num_nodes_total, device=edge_index.device, dtype=torch.bool)
        if edge_index.numel() > _ZERO:
            heads = edge_index[0]
            tails = edge_index[1]
            start_heads = node_is_start[heads]
            if start_heads.any():
                neighbors[tails[start_heads]] = True
            start_tails = node_is_start[tails]
            if start_tails.any():
                neighbors[heads[start_tails]] = True
        invalid = (node_is_start | neighbors) & (~node_is_answer)
        return invalid

    @staticmethod
    def _build_node_batch(node_ptr: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if num_graphs <= _ZERO:
            return torch.zeros((_ZERO,), device=node_ptr.device, dtype=torch.long)
        node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).clamp(min=_ZERO)
        graph_ids = torch.arange(num_graphs, device=node_ptr.device)
        return torch.repeat_interleave(graph_ids, node_counts)

    @staticmethod
    def _fill_unreachable_distances(
        node_min_dists: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
        *,
        unreachable_offset: float,
    ) -> torch.Tensor:
        reachable = node_min_dists >= _ZERO
        if not bool(reachable.any().item()):
            return node_min_dists
        reachable_vals = torch.where(reachable, node_min_dists, torch.zeros_like(node_min_dists))
        max_reach, _ = scatter_max(reachable_vals, node_batch, dim=0, dim_size=num_graphs)
        reach_counts = torch.bincount(node_batch[reachable], minlength=num_graphs)
        has_reach = reach_counts > _ZERO
        max_reach = torch.where(has_reach, max_reach, torch.zeros_like(max_reach))
        fill = max_reach + float(unreachable_offset)
        return torch.where(reachable, node_min_dists, fill[node_batch])

    @staticmethod
    def _compute_start_min_distance(
        dist: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).clamp(min=_ZERO)
        has_start = start_counts > _ZERO
        if start_node_locals.numel() == _ZERO:
            start_min = torch.zeros((num_graphs,), device=device, dtype=dist.dtype)
            return start_min, has_start
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=device), start_counts)
        start_locals = start_node_locals.to(device=device, dtype=torch.long, non_blocking=True)
        start_dist = dist.index_select(0, start_locals)
        start_min, _ = scatter_min(start_dist, graph_ids, dim=0, dim_size=num_graphs)
        start_min = torch.where(has_start, start_min, torch.zeros_like(start_min))
        return start_min, has_start

    def _compute_potential_shaping(
        self,
        *,
        node_min_dists: torch.Tensor | None,
        node_ptr: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        stop_globals: torch.Tensor,
        valid_stop: torch.Tensor,
        dummy_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if node_min_dists is None:
            raise ValueError("node_min_dists must be provided when potential shaping is enabled.")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            return torch.zeros((_ZERO,), device=node_ptr.device, dtype=dtype)
        node_min_dists = node_min_dists.to(device=node_ptr.device, dtype=dtype)
        node_batch = self._build_node_batch(node_ptr, num_graphs)
        dist = self._fill_unreachable_distances(
            node_min_dists,
            node_batch,
            num_graphs,
            unreachable_offset=self.potential_unreachable_offset,
        )
        start_min, has_start = self._compute_start_min_distance(
            dist,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
            num_graphs=num_graphs,
            device=node_ptr.device,
        )
        stop_dist = dist.index_select(0, stop_globals)
        stop_dist = torch.where(valid_stop, stop_dist, torch.zeros_like(stop_dist))
        phi_start = -start_min
        phi_stop = -stop_dist
        shaping = (self.potential_gamma * phi_stop - phi_start) * self.potential_weight
        valid = valid_stop & has_start
        if dummy_mask is not None:
            valid = valid & (~dummy_mask.to(device=node_ptr.device, dtype=torch.bool))
        return torch.where(valid, shaping, torch.zeros_like(shaping))


__all__ = ["RewardOutput", "GraphFusionReward"]
