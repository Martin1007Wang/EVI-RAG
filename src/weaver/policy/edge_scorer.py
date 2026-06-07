from __future__ import annotations

import math

import torch
from torch import nn

Tensor = torch.Tensor


class QuestionConditionedEdgeScorer(nn.Module):
    """
    Edge action energy scorer.

    It exposes two terms:

      1. score_alignment(q, e):
           static query-edge alignment.
           Independent of state z, so it can be precomputed per (G, q).

      2. score_state(z, e):
           state-conditioned edge compatibility.
           Depends on current selected-edge state h_z.

    Final edge energy is assembled in FlowEstimator:

        L_e(z) = align_scale * score_alignment(q, e) + score_state(z, e)

    No frontier-internal normalization is performed here.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.hidden_dim = hidden_dim

        inner_dim = int(state_hidden_dim or min(hidden_dim, 256))

        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner_dim, 1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def score_alignment(
        self,
        *,
        question_h: Tensor,  # [E, H]
        edge_h: Tensor,  # [E, H]
    ) -> Tensor:  # [E]
        """
        Static query-edge alignment:

            a(q,e) = <q, h_e> / sqrt(H)

        This does not depend on state z.
        """
        if question_h.shape != edge_h.shape:
            raise ValueError("question_h and edge_h must have the same shape.")

        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        return (question_h.float() * edge_h.float()).sum(dim=-1) * scale

    def score_state(
        self,
        *,
        state_h: Tensor,  # [F, H]
        edge_h: Tensor,  # [F, H]
    ) -> Tensor:  # [F]
        """
        State-conditioned edge energy:

            c_theta(z,e) = MLP([h_z || h_e || h_z * h_e])
        """
        if state_h.shape != edge_h.shape:
            raise ValueError("state_h and edge_h must have the same shape.")

        x = torch.cat(
            [
                state_h.float(),
                edge_h.float(),
                state_h.float() * edge_h.float(),
            ],
            dim=-1,
        )
        return self.state_mlp(x).squeeze(-1)
