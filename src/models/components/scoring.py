from __future__ import annotations

import math

import torch
from torch import nn


class NodeFlowHead(nn.Module):
    """Question-conditioned node flow scorer used by the mainline GFlowNet."""

    def __init__(
        self,
        *,
        node_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("priority_head.num_layers must be >= 1.")
        self.q_proj = nn.Linear(node_dim, question_dim, bias=False)
        layers: list[nn.Module] = []
        in_dim = int(node_dim + question_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.residual = nn.Sequential(*layers)

    def forward(
        self, node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        bilinear = (question_features * self.q_proj(node_features)).sum(dim=-1)
        bilinear = bilinear / math.sqrt(question_features.size(-1))
        residual = self.residual(
            torch.cat((node_features, question_features), dim=-1)
        ).squeeze(-1)
        return bilinear + residual


__all__ = ["NodeFlowHead"]
