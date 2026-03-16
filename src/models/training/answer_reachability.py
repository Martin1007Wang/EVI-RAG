from __future__ import annotations

import torch

from src.graph_runtime import TrajectoryBatch


def _build_answer_mask(batch: TrajectoryBatch) -> torch.Tensor:
    answer_mask = torch.zeros(
        (batch.num_nodes_total,), device=batch.node_ptr.device, dtype=torch.bool
    )
    if int(batch.a_local_indices.numel()) == 0:
        return answer_mask
    counts = batch.a_ptr[1:] - batch.a_ptr[:-1]
    offsets = batch.node_ptr[:-1].repeat_interleave(counts)
    absolute = batch.a_local_indices + offsets
    answer_mask.scatter_(0, absolute, True)
    return answer_mask


class AnswerReachabilityTrajectorySupervisor:
    def __init__(self, *, epsilon: float, failure_reward_mode: str) -> None:
        self.epsilon = float(epsilon)
        self.failure_reward_mode = str(failure_reward_mode)
        if self.failure_reward_mode not in {"constant", "graph_normalized"}:
            raise ValueError(
                "failure_reward_mode must be one of {'constant', 'graph_normalized'}."
            )

    def build_terminal_target_mask(self, *, batch: TrajectoryBatch) -> torch.Tensor:
        return _build_answer_mask(batch)

    def compute_terminal_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
        success_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.full(
            terminal_nodes.shape,
            fill_value=self.epsilon,
            device=terminal_nodes.device,
            dtype=torch.float32,
        )
        if self.failure_reward_mode == "graph_normalized":
            answer_counts = (batch.a_ptr[1:] - batch.a_ptr[:-1]).to(dtype=torch.float32)
            non_answer_counts = (
                (batch.node_ptr[1:] - batch.node_ptr[:-1]).to(dtype=torch.float32)
                - answer_counts
            ).clamp_min(1.0)
            graph_ids = (
                torch.arange(
                    batch.num_graphs,
                    device=terminal_nodes.device,
                    dtype=torch.long,
                )
                .unsqueeze(1)
                .expand_as(terminal_nodes)
            )
            rewards = self.epsilon / non_answer_counts.index_select(
                0, graph_ids.view(-1)
            ).view_as(terminal_nodes)
        rewards = torch.where(success_mask, torch.ones_like(rewards), rewards)
        return rewards, rewards.clamp_min(1.0e-12).log()


__all__ = [
    "AnswerReachabilityTrajectorySupervisor",
]
