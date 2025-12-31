from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

STRUCT_MODE_NONE = 0
STRUCT_MODE_DIFFUSION = 1
STRUCT_MODE_DISTANCE = 2
_MAX_DDE_ROUNDS = 4


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
        if self.num_rounds > _MAX_DDE_ROUNDS or self.num_reverse_rounds > _MAX_DDE_ROUNDS:
            raise ValueError(
                f"DDE supports at most {_MAX_DDE_ROUNDS} rounds per direction; "
                f"got num_rounds={self.num_rounds}, num_reverse_rounds={self.num_reverse_rounds}."
            )
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
        forward = self._apply_rounds(topic_one_hot, edge_index, self.layers, self.num_rounds)
        reverse = self._apply_rounds(topic_one_hot, reverse_edge_index, self.reverse_layers, self.num_reverse_rounds)
        return forward + reverse

    @staticmethod
    def _apply_rounds(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layers: nn.ModuleList,
        num_rounds: int,
    ) -> list[torch.Tensor]:
        if num_rounds == 0:
            return []
        h1 = layers[0](edge_index, x)
        if num_rounds == 1:
            return [h1]
        h2 = layers[1](edge_index, h1)
        if num_rounds == 2:
            return [h1, h2]
        h3 = layers[2](edge_index, h2)
        if num_rounds == 3:
            return [h1, h2, h3]
        h4 = layers[3](edge_index, h3)
        if num_rounds == 4:
            return [h1, h2, h3, h4]
        raise ValueError(f"num_rounds must be <= {_MAX_DDE_ROUNDS}, got {num_rounds}.")


__all__ = [
    "PEConv",
    "DDE",
    "STRUCT_MODE_NONE",
    "STRUCT_MODE_DIFFUSION",
    "STRUCT_MODE_DISTANCE",
]
