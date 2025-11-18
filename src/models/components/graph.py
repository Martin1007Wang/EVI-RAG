from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    """Simple mean aggregation used inside DDE."""

    def __init__(self) -> None:
        super().__init__(aggr="mean", node_dim=0)

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class DDE(nn.Module):
    """Directional diffusion encoder stacking forward + reverse PEConv layers."""

    def __init__(self, num_rounds: int = 2, num_reverse_rounds: int = 2) -> None:
        super().__init__()
        self.num_rounds = int(max(0, num_rounds))
        self.num_reverse_rounds = int(max(0, num_reverse_rounds))
        self.layers = nn.ModuleList([PEConv() for _ in range(self.num_rounds)])
        self.reverse_layers = nn.ModuleList([PEConv() for _ in range(self.num_reverse_rounds)])

    def forward(
        self,
        topic_one_hot: torch.Tensor,
        edge_index: torch.Tensor,
        reverse_edge_index: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if reverse_edge_index is None:
            reverse_edge_index = edge_index.flip(0)

        result: list[torch.Tensor] = []
        h = topic_one_hot
        for layer in self.layers:
            h = layer(edge_index, h)
            result.append(h)

        h_rev = topic_one_hot
        for layer in self.reverse_layers:
            h_rev = layer(reverse_edge_index, h_rev)
            result.append(h_rev)

        return result


__all__ = ["PEConv", "DDE"]
