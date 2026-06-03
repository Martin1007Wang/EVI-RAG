from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.rollout.trajectory import EXTERNAL_TERMINAL, TrajectoryBatch


@dataclass(frozen=True, slots=True)
class ReplaySource:
    anneal_steps: int | None = None

    def __post_init__(self) -> None:
        if self.anneal_steps is not None and int(self.anneal_steps) <= 0:
            raise ValueError("anneal_steps must be positive or None.")

    def replay_weight(self, *, global_step: int) -> float:
        if self.anneal_steps is None:
            return 1.0
        step = max(int(global_step), 0)
        return max(0.0, 1.0 - float(step) / float(self.anneal_steps))

    def raw_trajectory_count(
        self,
        *,
        replay_context: ReplayContext,
        budget: int,
        replay_round: int = 0,
    ) -> int:
        _, edge_count = self._select_bank_slice(
            replay_context=replay_context,
            budget=budget,
            replay_round=replay_round,
        )
        return int(edge_count.ge(0).sum().item())

    @torch.no_grad()
    def sample_trajectories(
        self,
        *,
        graph_context: GraphContext,
        target_context: TargetContext,
        replay_context: ReplayContext,
        budget: int,
        global_step: int = 0,
        replay_round: int = 0,
    ) -> TrajectoryBatch:
        del target_context
        edge_ids, edge_count = self._select_bank_slice(
            replay_context=replay_context,
            budget=budget,
            replay_round=replay_round,
        )
        edge_ids = edge_ids.to(
            device=graph_context.device,
            dtype=torch.long,
        )
        edge_count = edge_count.to(
            device=graph_context.device,
            dtype=torch.long,
        )
        valid = edge_count.ge(0)
        if not bool(valid.any()):
            return TrajectoryBatch.empty(device=graph_context.device, budget=budget)
        graph_ids = torch.arange(
            int(graph_context.num_graphs),
            dtype=torch.long,
            device=graph_context.device,
        ).view(-1, 1).expand_as(edge_count)[valid]
        edge_ids = edge_ids[valid].contiguous()
        edge_count = edge_count[valid].contiguous()
        keep_count = min(
            int(edge_count.numel()),
            int(self.replay_weight(global_step=global_step) * float(edge_count.numel())),
        )
        if keep_count <= 0:
            return TrajectoryBatch.empty(device=graph_context.device, budget=budget)
        graph_ids = graph_ids[:keep_count].contiguous()
        edge_ids = edge_ids[:keep_count].contiguous()
        edge_count = edge_count[:keep_count].contiguous()
        num = int(edge_count.numel())
        return TrajectoryBatch(
            graph_ids=graph_ids,
            edge_ids=edge_ids,
            edge_logp=torch.zeros((num, budget), dtype=torch.float32, device=graph_context.device),
            edge_count=edge_count,
            stop_reason=torch.full((num,), int(EXTERNAL_TERMINAL), dtype=torch.uint8, device=graph_context.device),
            stop_logp=torch.zeros((num,), dtype=torch.float32, device=graph_context.device),
            source=torch.ones((num,), dtype=torch.bool, device=graph_context.device),
        )

    @staticmethod
    def _select_bank_slice(
        *,
        replay_context: ReplayContext,
        budget: int,
        replay_round: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        budget = int(budget)
        if budget < 0 or budget >= int(replay_context.edge_ids.size(1)):
            raise ValueError(
                f"Replay bank supports budgets in [0, {int(replay_context.edge_ids.size(1)) - 1}], got {budget}."
            )
        variant = int(replay_round) % int(replay_context.edge_ids.size(2))
        return (
            replay_context.edge_ids[:, budget, variant, :, :budget],
            replay_context.edge_count[:, budget, variant, :],
        )


__all__ = ["ReplaySource"]
