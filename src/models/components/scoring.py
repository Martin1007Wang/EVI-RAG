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


class TransitionPolicyHead(nn.Module):
    """Question-conditioned transition scorer over candidate graph edges."""

    def __init__(
        self,
        *,
        state_dim: int,
        relation_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("transition_head.num_layers must be >= 1.")
        self.context_query = nn.Linear(
            int((2 * state_dim) + relation_dim + question_dim), question_dim
        )
        self.context_key = nn.Linear(question_dim, question_dim, bias=False)
        self.context_value = nn.Linear(question_dim, question_dim, bias=False)
        self.context_norm = nn.LayerNorm(question_dim)
        self.relation_context_proj = nn.Linear(relation_dim, question_dim, bias=False)
        input_dim = int((2 * state_dim) + relation_dim + (3 * question_dim))
        self.input_norm = nn.LayerNorm(input_dim)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
        question_features: torch.Tensor,
        question_context_features: torch.Tensor,
        question_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if question_context_features.dim() != 3:
            raise ValueError(
                "question_context_features must be 3D [N, L, d] in TransitionPolicyHead."
            )
        if (
            question_context_mask.dtype != torch.bool
            or question_context_mask.dim() != 2
        ):
            raise ValueError(
                "question_context_mask must be 2D bool in TransitionPolicyHead."
            )
        if tuple(question_context_mask.shape) != tuple(
            question_context_features.shape[:2]
        ):
            raise ValueError(
                "question_context_mask shape must match question_context_features in TransitionPolicyHead."
            )
        if bool((~question_context_mask).all(dim=1).any().item()):
            raise ValueError(
                "question_context_mask contains rows without valid tokens in TransitionPolicyHead."
            )
        context_query = self.context_query(
            torch.cat(
                (
                    current_state_features,
                    candidate_state_features,
                    relation_features,
                    question_features,
                ),
                dim=-1,
            )
        )
        context_key = self.context_key(
            question_context_features.to(dtype=torch.float32)
        )
        context_value = self.context_value(
            question_context_features.to(dtype=torch.float32)
        )
        attention_scores = torch.einsum(
            "bd,bld->bl", context_query.to(dtype=torch.float32), context_key
        )
        attention_scores = attention_scores / math.sqrt(float(context_key.size(-1)))
        attention_scores = attention_scores.masked_fill(
            ~question_context_mask, float("-inf")
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.where(
            torch.isfinite(attention_weights),
            attention_weights,
            torch.zeros_like(attention_weights),
        )
        question_context_summary = torch.einsum(
            "bl,bld->bd", attention_weights, context_value
        )
        question_context_summary = self.context_norm(
            question_context_summary + context_query.to(dtype=torch.float32)
        )
        relation_context_interaction = (
            question_context_summary
            * self.relation_context_proj(relation_features.to(dtype=torch.float32))
        )
        fused = torch.cat(
            (
                current_state_features,
                candidate_state_features,
                relation_features,
                question_features,
                question_context_summary.to(dtype=current_state_features.dtype),
                relation_context_interaction.to(dtype=current_state_features.dtype),
            ),
            dim=-1,
        )
        return self.mlp(self.input_norm(fused)).squeeze(-1)


__all__ = ["NodeFlowHead", "TransitionPolicyHead"]
