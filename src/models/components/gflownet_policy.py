from __future__ import annotations

import torch
from torch import nn


class EnergyEdgePolicy(nn.Module):
    """Energy-based edge policy with masked edge logits."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(
        self,
        *,
        edge_scores: torch.Tensor,
        valid_edges_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = valid_edges_mask.to(device=edge_scores.device, dtype=torch.bool)
        neg_inf = torch.finfo(edge_scores.dtype).min
        edge_logits = edge_scores.masked_fill(~valid_mask, neg_inf)
        return edge_logits


__all__ = ["EnergyEdgePolicy"]
