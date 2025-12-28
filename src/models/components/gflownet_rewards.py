from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn
from torch_scatter import scatter_min


@dataclass
class RewardOutput:
    reward: torch.Tensor
    log_reward: torch.Tensor
    success: torch.Tensor
    semantic_score: torch.Tensor
    length_cost: torch.Tensor
    path_len: torch.Tensor
    shortest_len: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class GFlowNetReward(nn.Module):
    """Energy-based reward: outcome + semantic bonus - length penalty."""

    def __init__(
        self,
        *,
        success_reward: float = 1.0,
        failure_reward: float = 0.01,
        semantic_coef: float = 1.0,
        length_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.semantic_coef = float(semantic_coef)
        self.length_coef = float(length_coef)
        if self.success_reward <= 0.0:
            raise ValueError(f"success_reward must be positive; got {self.success_reward}.")
        if self.failure_reward <= 0.0:
            raise ValueError(f"failure_reward must be positive; got {self.failure_reward}.")
        if self.success_reward <= self.failure_reward:
            raise ValueError(
                f"success_reward must be greater than failure_reward; got {self.success_reward} <= {self.failure_reward}."
            )
        if self.semantic_coef < 0.0:
            raise ValueError(f"semantic_coef must be >= 0; got {self.semantic_coef}.")
        if self.length_coef < 0.0:
            raise ValueError(f"length_coef must be >= 0; got {self.length_coef}.")
        self.log_success = float(math.log(self.success_reward))
        self.log_failure = float(math.log(self.failure_reward))

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_scores: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        answer_hit: torch.Tensor,  # [B]
        pair_start_node_locals: torch.Tensor | None = None,
        pair_answer_node_locals: torch.Tensor | None = None,
        pair_shortest_lengths: torch.Tensor | None = None,
        start_node_hit: torch.Tensor | None = None,
        answer_node_hit: torch.Tensor | None = None,
        node_ptr: torch.Tensor | None = None,
        **_,
    ) -> RewardOutput:
        device = selected_mask.device
        answer_hit = answer_hit.to(device=device)
        num_graphs = int(answer_hit.numel())
        if num_graphs <= 0:
            empty = torch.zeros(0, device=device, dtype=torch.float32)
            return RewardOutput(
                reward=empty,
                log_reward=empty,
                success=empty,
                semantic_score=empty,
                length_cost=empty,
                path_len=empty,
                shortest_len=empty,
            )

        edge_batch = edge_batch.to(device=device, dtype=torch.long)
        mask_f = selected_mask.to(device=device, dtype=torch.float32)
        path_len = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        if mask_f.numel() > 0:
            path_len.index_add_(0, edge_batch, mask_f)

        if edge_scores is None:
            raise ValueError("edge_scores must be provided for semantic reward.")
        edge_scores = edge_scores.to(device=device, dtype=torch.float32).view(-1)
        if edge_scores.numel() != mask_f.numel():
            raise ValueError(
                f"edge_scores length {edge_scores.numel()} != selected_mask length {mask_f.numel()}."
            )
        semantic_sum = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        if mask_f.numel() > 0:
            semantic_weights = torch.sigmoid(edge_scores)
            semantic_sum.index_add_(0, edge_batch, mask_f * semantic_weights)
        semantic_score = semantic_sum / path_len.clamp(min=1.0)

        shortest_len = self._match_shortest_lengths(
            pair_start_node_locals=pair_start_node_locals,
            pair_answer_node_locals=pair_answer_node_locals,
            pair_shortest_lengths=pair_shortest_lengths,
            start_node_hit=start_node_hit,
            answer_node_hit=answer_node_hit,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=device,
        )
        hit_mask = answer_hit.to(dtype=torch.bool)
        if bool((hit_mask & (shortest_len < 0)).any().item()):
            raise ValueError("Missing pair shortest length for answer-hit trajectories.")
        shortest_len_f = shortest_len.to(dtype=path_len.dtype)
        length_cost = (path_len - shortest_len_f).clamp(min=0.0)
        semantic_score = torch.where(hit_mask, semantic_score, torch.zeros_like(semantic_score))
        length_cost = torch.where(hit_mask, length_cost, torch.zeros_like(length_cost))

        log_reward = torch.where(
            hit_mask,
            self.log_success + self.semantic_coef * semantic_score - self.length_coef * length_cost,
            torch.full((num_graphs,), self.log_failure, device=device, dtype=torch.float32),
        )

        reward = torch.exp(log_reward)
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=answer_hit.to(dtype=torch.float32),
            semantic_score=semantic_score,
            length_cost=length_cost,
            path_len=path_len,
            shortest_len=shortest_len.to(dtype=path_len.dtype),
        )

    @staticmethod
    def _match_shortest_lengths(
        *,
        pair_start_node_locals: torch.Tensor | None,
        pair_answer_node_locals: torch.Tensor | None,
        pair_shortest_lengths: torch.Tensor | None,
        start_node_hit: torch.Tensor | None,
        answer_node_hit: torch.Tensor | None,
        node_ptr: torch.Tensor | None,
        num_graphs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            pair_start_node_locals is None
            or pair_answer_node_locals is None
            or pair_shortest_lengths is None
            or start_node_hit is None
            or answer_node_hit is None
            or node_ptr is None
        ):
            raise ValueError("pair_* fields and node_ptr/start_node_hit/answer_node_hit are required for reward.")

        pair_start = pair_start_node_locals.to(device=device, dtype=torch.long).view(-1)
        pair_answer = pair_answer_node_locals.to(device=device, dtype=torch.long).view(-1)
        pair_lengths = pair_shortest_lengths.to(device=device, dtype=torch.long).view(-1)
        node_ptr = node_ptr.to(device=device, dtype=torch.long).view(-1)
        if pair_start.numel() == 0:
            return torch.full((num_graphs,), -1, device=device, dtype=torch.long)
        if pair_start.numel() != pair_answer.numel() or pair_start.numel() != pair_lengths.numel():
            raise ValueError("pair_start/answer/lengths size mismatch in reward.")

        pair_graph = torch.bucketize(pair_start, node_ptr[1:], right=True)
        if pair_graph.numel() > 0:
            if int(pair_graph.min().item()) < 0 or int(pair_graph.max().item()) >= num_graphs:
                raise ValueError("pair_graph out of range for reward computation.")
        pair_start_local = pair_start - node_ptr[pair_graph]
        pair_answer_local = pair_answer - node_ptr[pair_graph]

        start_hit = start_node_hit.to(device=device, dtype=torch.long).view(-1)
        answer_hit = answer_node_hit.to(device=device, dtype=torch.long).view(-1)
        if start_hit.numel() != num_graphs or answer_hit.numel() != num_graphs:
            raise ValueError("start_node_hit/answer_node_hit size mismatch in reward.")

        match = (pair_start_local == start_hit[pair_graph]) & (pair_answer_local == answer_hit[pair_graph])
        if not bool(match.any().item()):
            return torch.full((num_graphs,), -1, device=device, dtype=torch.long)

        match_graph = pair_graph[match]
        match_lengths = pair_lengths[match]
        if match_lengths.numel() == 0:
            return torch.full((num_graphs,), -1, device=device, dtype=torch.long)
        match_counts = torch.bincount(match_graph, minlength=num_graphs)
        sentinel = int(match_lengths.max().item()) + 1
        out = torch.full((num_graphs,), sentinel, device=device, dtype=match_lengths.dtype)
        shortest_len, _ = scatter_min(match_lengths, match_graph, dim=0, out=out)
        return torch.where(match_counts > 0, shortest_len, torch.full_like(shortest_len, -1))


__all__ = ["RewardOutput", "GFlowNetReward"]
