from __future__ import annotations

import math

import torch
from torch import nn


def _build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(max(num_layers - 1, 0)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class NodeFlowHead(nn.Module):
    """Critic head over state features.

    The critic reads the already question-conditioned state feature and maps it to
    a single log-flow scalar. ``question_features`` is kept in the signature only
    for interface compatibility with older call sites.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        del question_dim, hidden_dim, num_layers, dropout
        super().__init__()
        self.critic = nn.Linear(int(node_dim), 1)

    def forward(
        self, node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        del question_features
        return self.critic(node_features).squeeze(-1)


class TransitionPolicyHead(nn.Module):
    """Detached actor head over recurrent state queries and edge keys."""

    def __init__(
        self,
        *,
        state_dim: int,
        relation_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        microbatch_size: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("transition_head.num_layers must be >= 1.")
        if microbatch_size < 1:
            raise ValueError("transition_head.microbatch_size must be >= 1.")
        self.microbatch_size = int(microbatch_size)
        self.actor_dim = int(state_dim)
        self.query_norm = nn.LayerNorm(self.actor_dim)
        self.edge_norm = nn.LayerNorm(int(state_dim + relation_dim))
        self.query_mlp = _build_mlp(
            input_dim=self.actor_dim,
            output_dim=self.actor_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.edge_mlp = _build_mlp(
            input_dim=int(state_dim + relation_dim),
            output_dim=self.actor_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    @staticmethod
    def _validate_inputs(
        *,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> None:
        if candidate_state_features.shape != current_state_features.shape:
            raise ValueError(
                "candidate_state_features must match current_state_features shape in TransitionPolicyHead."
            )
        if tuple(relation_features.shape[:-1]) != tuple(
            current_state_features.shape[:-1]
        ):
            raise ValueError(
                "relation_features batch shape must match current_state_features in TransitionPolicyHead."
            )

    def _forward_chunk(
        self,
        *,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        actor_query = self.query_mlp(
            self.query_norm(current_state_features.detach().to(dtype=torch.float32))
        )
        detached_relation_features = relation_features.detach().to(dtype=torch.float32)
        detached_candidate_features = candidate_state_features.detach().to(
            dtype=torch.float32
        )
        edge_key = self.edge_mlp(
            self.edge_norm(
                torch.cat(
                    (
                        detached_relation_features,
                        detached_candidate_features,
                    ),
                    dim=-1,
                )
            )
        )
        logits = (actor_query * edge_key).sum(dim=-1)
        return logits / math.sqrt(float(max(self.actor_dim, 1)))

    def forward(
        self,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            current_state_features=current_state_features,
            candidate_state_features=candidate_state_features,
            relation_features=relation_features,
        )
        batch_size = int(current_state_features.size(0))
        if batch_size <= self.microbatch_size:
            return self._forward_chunk(
                current_state_features=current_state_features,
                candidate_state_features=candidate_state_features,
                relation_features=relation_features,
            )
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, batch_size, self.microbatch_size):
            end = min(start + self.microbatch_size, batch_size)
            chunk_slice = slice(start, end)
            logits_chunks.append(
                self._forward_chunk(
                    current_state_features=current_state_features[chunk_slice],
                    candidate_state_features=candidate_state_features[chunk_slice],
                    relation_features=relation_features[chunk_slice],
                )
            )
        return torch.cat(logits_chunks, dim=0)


__all__ = ["NodeFlowHead", "TransitionPolicyHead"]
