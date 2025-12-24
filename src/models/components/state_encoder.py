from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StateEncoderCache:
    question_tokens: torch.Tensor   # [B, H]
    node_tokens: torch.Tensor       # [N_total, H]
    node_batch: torch.Tensor        # [N_total]


class StateEncoder(nn.Module):
    """State = mean(active nodes) + question + step embedding."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        self.step_embeddings = nn.Embedding(self.max_steps + 1, self.hidden_dim)
        nn.init.constant_(self.step_embeddings.weight, 0.0)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def precompute(
        self,
        *,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        **_: torch.Tensor,
    ) -> StateEncoderCache:
        if question_tokens.dim() != 2 or question_tokens.size(-1) != self.hidden_dim:
            raise ValueError("question_tokens must be [B,H] with hidden_dim H.")
        num_graphs = int(question_tokens.size(0))
        if node_ptr.numel() != num_graphs + 1:
            raise ValueError("node_ptr length mismatch with question_tokens batch size.")
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_ptr.device),
            node_counts,
        )
        if node_batch.numel() != node_tokens.size(0):
            raise ValueError("node_batch length mismatch with node_tokens in encoder precompute.")
        return StateEncoderCache(
            question_tokens=question_tokens,
            node_tokens=node_tokens,
            node_batch=node_batch,
        )

    def encode_state(self, *, cache: StateEncoderCache, state: "GraphState") -> torch.Tensor:
        question_tokens = cache.question_tokens
        num_graphs = int(question_tokens.size(0))
        device = question_tokens.device
        dtype = question_tokens.dtype

        node_tokens = cache.node_tokens.to(device=device)
        node_batch = cache.node_batch.to(device=device)
        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        if active_nodes.numel() != node_tokens.size(0):
            raise ValueError("active_nodes length mismatch with node_tokens in encoder.")

        if bool(active_nodes.any().item()):
            active_sum = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=node_tokens.dtype)
            active_sum.index_add_(0, node_batch[active_nodes], node_tokens[active_nodes])
            active_count = torch.bincount(node_batch[active_nodes], minlength=num_graphs).clamp(min=1).to(device=device)
            active_mean = active_sum / active_count.unsqueeze(-1).to(dtype=active_sum.dtype)
        else:
            active_mean = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=node_tokens.dtype)

        step_counts = state.step_counts.to(device=device, dtype=torch.long).clamp(min=0, max=self.max_steps)
        remaining = (self.max_steps - step_counts).clamp(min=0, max=self.max_steps)
        step_emb = self.step_embeddings(remaining)

        state_tokens = active_mean.to(dtype=dtype) + question_tokens + step_emb.to(dtype=dtype)
        return self.norm(state_tokens)


__all__ = ["StateEncoder", "StateEncoderCache"]
