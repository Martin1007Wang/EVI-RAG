from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class EmbeddingProjector(nn.Module):
    """Lazy projection layer that still registers parameters for optimizers."""

    def __init__(self, output_dim: int, *, finetune: bool = False, activation: nn.Module | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LazyLinear(output_dim)]
        if activation is None:
            activation = nn.Tanh()
        layers.append(activation)
        self.network = nn.Sequential(*layers)
        if not finetune:
            self.freeze()

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.network(tensor)

    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


__all__ = ["EmbeddingProjector"]
