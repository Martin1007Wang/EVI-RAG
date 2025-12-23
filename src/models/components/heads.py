from __future__ import annotations

import copy

from torch import nn


class DenseFeatureExtractor(nn.Module):
    """Two-layer MLP with dropout and configurable activations."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout_p: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if activation is None:
            raise ValueError("DenseFeatureExtractor requires a non-null activation module.")
        activation_0 = copy.deepcopy(activation)
        activation_1 = copy.deepcopy(activation)
        self.network = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            activation_0,
            nn.Dropout(dropout_p),
            nn.Linear(emb_dim, hidden_dim),
            activation_1,
            nn.Dropout(dropout_p),
        )

    def forward(self, inputs):
        return self.network(inputs)


class DeterministicHead(nn.Module):
    """Single-logit scorer built on top of DenseFeatureExtractor outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)

__all__ = ["DenseFeatureExtractor", "DeterministicHead"]
