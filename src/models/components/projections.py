from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class EmbeddingProjector(nn.Module):
    """Deterministic projection MLP (Linear + activation)."""

    def __init__(
        self,
        output_dim: int,
        *,
        input_dim: Optional[int] = None,
        finetune: bool = False,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_dim is None or input_dim <= 0:
            raise ValueError("EmbeddingProjector requires a positive input_dim; lazy layers are disallowed.")
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
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
