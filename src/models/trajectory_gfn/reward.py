from __future__ import annotations

import torch

from src.models.environment.ops import build_node_membership_mask

from .batch import TrajectoryBatch


class TrajectoryReward:
    def __init__(
        self,
        *,
        epsilon: float,
        wrong_stop_reward_mode: str = "graph_normalized",
    ) -> None:
        self.epsilon = float(epsilon)
        self.wrong_stop_reward_mode = str(wrong_stop_reward_mode)
        if self.epsilon <= 0.0:
            raise ValueError("reward epsilon must be > 0.")
        if self.wrong_stop_reward_mode not in {"constant", "graph_normalized"}:
            raise ValueError(
                "wrong_stop_reward_mode must be one of {'constant', 'graph_normalized'}."
            )

    def build_target_mask(self, batch: TrajectoryBatch) -> torch.Tensor:
        return build_node_membership_mask(
            local_indices=batch.a_local_indices,
            ptr=batch.a_ptr,
            node_ptr=batch.node_ptr,
            num_nodes_total=batch.num_nodes_total,
            device=batch.node_ptr.device,
            field_name="a_local_indices",
        )

    def compute(
        self,
        *,
        batch: TrajectoryBatch,
        stop_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mask = self.build_target_mask(batch)
        target_mask = target_mask.to(device=stop_nodes.device, dtype=torch.bool)
        flat_stop = stop_nodes.reshape(-1)
        safe_stop = flat_stop.clamp(min=0, max=max(batch.num_nodes_total - 1, 0))
        hits = target_mask.index_select(0, safe_stop).view_as(stop_nodes)
        rewards = self._build_wrong_stop_rewards(batch=batch, stop_nodes=stop_nodes)
        rewards = torch.where(hits, torch.ones_like(rewards), rewards)
        log_rewards = rewards.log()
        return hits, rewards, log_rewards

    def _build_wrong_stop_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        stop_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if self.wrong_stop_reward_mode == "constant":
            return torch.full(
                stop_nodes.shape,
                fill_value=self.epsilon,
                device=stop_nodes.device,
                dtype=torch.float32,
            )
        target_mask = self.build_target_mask(batch)
        target_mask = target_mask.to(device=stop_nodes.device, dtype=torch.bool)
        node_batch = batch.node_batch.to(device=stop_nodes.device, dtype=torch.long)
        non_target = (~target_mask).to(dtype=torch.float32)
        non_target_counts = torch.zeros(
            (batch.num_graphs,), device=stop_nodes.device, dtype=torch.float32
        )
        non_target_counts.scatter_add_(0, node_batch, non_target)
        non_target_counts = non_target_counts.clamp(min=1.0)
        graph_ids = torch.arange(
            batch.num_graphs, device=stop_nodes.device, dtype=torch.long
        ).unsqueeze(1)
        graph_ids = graph_ids.expand_as(stop_nodes)
        return self.epsilon / non_target_counts.index_select(
            0, graph_ids.reshape(-1)
        ).reshape(stop_nodes.shape)
