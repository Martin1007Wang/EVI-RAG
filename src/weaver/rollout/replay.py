from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.rollout.trajectory import EXTERNAL_TERMINAL, TrajectoryBatch


@dataclass(slots=True)
class ReplaySource:
    metric_name: str = "val/rollout_union@8/recall"
    threshold: float = 0.8
    _best_metric: float = field(default=0.0, init=False, repr=False)
    _fraction: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.metric_name = str(self.metric_name)
        self.threshold = float(self.threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")

    def current_fraction(self) -> float:
        return self._fraction

    def update_from_validation(self, *, metric_value: float) -> float:
        self._best_metric = max(self._best_metric, float(metric_value))
        if self._best_metric < self.threshold:
            self._fraction = 1.0
        else:
            span = max(1.0 - self.threshold, 1.0e-6)
            self._fraction = max(0.0, min(1.0, (1.0 - self._best_metric) / span))
        return self._fraction

    def raw_trajectory_count(
        self,
        *,
        replay_context: ReplayContext,
        budget: int,
        replay_round: int = 0,
    ) -> int:
        _, edge_count, _ = self._select_bank_slice(
            replay_context=replay_context,
            replay_round=replay_round,
        )
        budget = int(budget)
        return int((edge_count.ge(0) & edge_count.le(budget)).sum().item())

    @torch.no_grad()
    def sample_trajectories(
        self,
        *,
        graph_context: GraphContext,
        target_context: TargetContext,
        replay_context: ReplayContext,
        budget: int,
        replay_round: int = 0,
    ) -> TrajectoryBatch:
        del target_context
        edge_ids, edge_count, priority = self._select_bank_slice(
            replay_context=replay_context,
            replay_round=replay_round,
        )
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be nonnegative.")
        edge_ids = edge_ids.to(device=graph_context.device, dtype=torch.long)
        edge_count = edge_count.to(device=graph_context.device, dtype=torch.long)
        priority = priority.to(device=graph_context.device, dtype=torch.float32)
        valid = edge_count.ge(0) & edge_count.le(budget)
        if not bool(valid.any()):
            return TrajectoryBatch.empty(device=graph_context.device, budget=budget)
        keep = _per_graph_replay_keep_mask(
            valid=valid,
            fraction=float(self._fraction),
        )
        if not bool(keep.any()):
            return TrajectoryBatch.empty(device=graph_context.device, budget=budget)
        graph_ids = torch.arange(
            int(graph_context.num_graphs),
            dtype=torch.long,
            device=graph_context.device,
        ).view(-1, 1).expand_as(edge_count)[keep]
        selected_edge_ids = edge_ids[keep].contiguous()
        selected_edge_count = edge_count[keep].contiguous()
        selected_priority = priority[keep].contiguous()
        order = torch.argsort(selected_priority, descending=True, stable=True)
        graph_ids = graph_ids.index_select(0, order)
        selected_edge_ids = selected_edge_ids.index_select(0, order)
        selected_edge_count = selected_edge_count.index_select(0, order)
        num = int(selected_edge_count.numel())
        output_edge_ids = torch.full((num, budget), -1, dtype=torch.long, device=graph_context.device)
        copy_width = min(int(selected_edge_ids.size(1)), budget)
        if copy_width > 0:
            output_edge_ids[:, :copy_width] = selected_edge_ids[:, :copy_width]
        return TrajectoryBatch(
            graph_ids=graph_ids,
            edge_ids=output_edge_ids,
            edge_logp=torch.zeros((num, budget), dtype=torch.float32, device=graph_context.device),
            edge_count=selected_edge_count,
            stop_reason=torch.full((num,), int(EXTERNAL_TERMINAL), dtype=torch.uint8, device=graph_context.device),
            stop_logp=torch.zeros((num,), dtype=torch.float32, device=graph_context.device),
            source=torch.ones((num,), dtype=torch.bool, device=graph_context.device),
        )

    @staticmethod
    def _select_bank_slice(
        *,
        replay_context: ReplayContext,
        replay_round: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variant = int(replay_round) % int(replay_context.edge_ids.size(1))
        return (
            replay_context.edge_ids[:, variant, :, :],
            replay_context.edge_count[:, variant, :],
            replay_context.priority[:, variant, :],
        )


def _per_graph_replay_keep_mask(*, valid: torch.Tensor, fraction: float) -> torch.Tensor:
    fraction = max(0.0, min(float(fraction), 1.0))
    keep = torch.zeros_like(valid)
    if fraction <= 0.0 or not bool(valid.any()):
        return keep
    if fraction >= 1.0:
        return valid.clone()
    for graph_id in range(int(valid.size(0))):
        slots = valid[graph_id].nonzero(as_tuple=False).flatten()
        if int(slots.numel()) == 0:
            continue
        keep_count = max(1, round(float(slots.numel()) * fraction))
        keep[graph_id, slots[:keep_count]] = True
    return keep


__all__ = ["ReplaySource"]
