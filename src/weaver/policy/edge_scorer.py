from __future__ import annotations

import torch
from torch import nn


class QuestionConditionedEdgeScorer(nn.Module):
    """Question-conditioned scorer for candidate graph edges."""

    def __init__(self, *, hidden_dim: int, relation_lambda: float = 0.5) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim**-0.5
        self.relation_lambda = float(relation_lambda)
        self.marginal_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def score_alignment(
        self,
        *,
        question_h: torch.Tensor,
        edge_h: torch.Tensor,
        relation_h: torch.Tensor,
    ) -> torch.Tensor:
        relation_score = (question_h * relation_h).sum(dim=-1) * self.scale
        edge_score = (question_h * edge_h).sum(dim=-1) * self.scale
        return relation_score + self.relation_lambda * edge_score

    def score_state(
        self,
        *,
        state_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        """State-conditioned edge compatibility over [state_h, edge_h, state_h * edge_h]."""
        phi_state = self.marginal_mlp(
            torch.cat(
                [
                    state_h,
                    edge_h,
                    state_h * edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return phi_state


__all__ = ["QuestionConditionedEdgeScorer"]
