from __future__ import annotations

import torch
from torch import nn


class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation (FiLM) conditioning."""

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.param_generator = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.param_generator(conditioning)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return (gamma + 1) * features + beta


class FeatureFusion(nn.Module):
    """Fuses semantic + structural features using concat or FiLM."""

    def __init__(self, *, fusion_method: str, semantic_dim: int, structure_dim: int) -> None:
        super().__init__()
        method = fusion_method.strip().lower()
        if method not in {"concat", "film"}:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        self.fusion_method = method
        self.semantic_dim = int(semantic_dim)
        self.structure_dim = int(structure_dim)
        if self.fusion_method == "film":
            self.film = FiLMLayer(structure_dim, semantic_dim)
            self.output_dim = semantic_dim
        else:
            self.film = None
            self.output_dim = semantic_dim + structure_dim

    def forward(self, semantic: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        if self.fusion_method == "concat":
            return torch.cat([semantic, structure], dim=-1)
        assert self.film is not None
        return self.film(semantic, structure)

    def get_output_dim(self) -> int:
        return self.output_dim


__all__ = ["FeatureFusion", "FiLMLayer"]
