from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import init_xavier

from .frontier_encoder import FrontierEncoding
from .state_encoder import StateEncoding


class EdgeActionScorer(nn.Module):
    """
    State-conditioned frontier edge scorer.

    For each legal frontier transition (z, e), this module returns an edge
    action log-energy:

        Q_theta(z, e) = A_PLM(q, e) + Delta_theta(z, e)
        A_PLM(q, e) = semantic_prior_scale * <q_sem, rel_sem>

    If the training objective enforces flow consistency, this action energy can
    be interpreted as transition log-flow Q_theta(z, e). Without such a loss,
    it is only an edge logit / edge energy.

    Design principles:
    - every semantic dot product is computed only in raw L2 PLM space;
    - state, budget, and encoded edge information only enter through the
      learnable model-space residual;
    - initial behavior is relation-only PLM prior, with residual correction
      starting near zero.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        adapter_hidden_dim: int | None = None,
        init_adapter_scale: float = 0.1,
        adapter_final_init_scale: float = 0.0,
        semantic_prior_scale: float = 10.0,
        use_edge_logit_shift: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.semantic_prior_scale = float(semantic_prior_scale)
        if self.semantic_prior_scale <= 0.0:
            raise ValueError(f"semantic_prior_scale must be positive, got {semantic_prior_scale}.")

        adapter_hidden_dim = self.hidden_dim if adapter_hidden_dim is None else int(adapter_hidden_dim)
        if adapter_hidden_dim <= 0:
            raise ValueError(f"adapter_hidden_dim must be positive, got {adapter_hidden_dim}.")

        self.adapter_final_init_scale = float(adapter_final_init_scale)
        if self.adapter_final_init_scale < 0.0:
            raise ValueError(
                "adapter_final_init_scale must be non-negative, "
                f"got {adapter_final_init_scale}."
            )

        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, 1),
        )

        self.adapter_scale = nn.Parameter(torch.tensor(float(init_adapter_scale)))

        # Optional global expansion prior. It shifts every edge action energy
        # and therefore changes the STOP-vs-expand balance.
        self.edge_logit_shift = nn.Parameter(torch.tensor(0.0)) if use_edge_logit_shift else None

        self._reset_parameters()

    def forward(
        self,
        *,
        state: StateEncoding,
        frontier: FrontierEncoding,
    ) -> torch.Tensor:
        return self.score_tensors(
            state_h=state.state_h,
            row_ids=frontier.row_ids,
            edge_h=frontier.edge_h,
            query_h=frontier.query_h,
            rel_sem_h=frontier.rel_sem_h,
            query_sem_h=frontier.query_sem_h,
        )

    def semantic_prior(self, frontier: FrontierEncoding) -> torch.Tensor:
        return self.semantic_prior_scale * (frontier.query_sem_h * frontier.rel_sem_h).sum(dim=-1)

    def score_tensors(
        self,
        *,
        state_h: torch.Tensor,
        row_ids: torch.Tensor,
        edge_h: torch.Tensor,
        query_h: torch.Tensor,
        rel_sem_h: torch.Tensor,
        query_sem_h: torch.Tensor,
    ) -> torch.Tensor:
        if edge_h.numel() == 0:
            return edge_h.new_empty((0,))

        device = edge_h.device
        dtype = edge_h.dtype
        row_ids = row_ids.to(device=device, dtype=torch.long).view(-1)
        num_frontier = row_ids.numel()

        if edge_h.shape[0] != num_frontier:
            raise ValueError(
                "edge_h and row_ids must have the same first dimension, "
                f"got edge_h={edge_h.shape[0]}, row_ids={num_frontier}."
            )
        if query_h.shape[0] != num_frontier:
            raise ValueError(
                "query_h and row_ids must have the same first dimension, "
                f"got query_h={query_h.shape[0]}, row_ids={num_frontier}."
            )
        if rel_sem_h.shape[0] != num_frontier or query_sem_h.shape[0] != num_frontier:
            raise ValueError("semantic tensors must have one row per frontier action.")
        if edge_h.shape[-1] != self.hidden_dim:
            raise ValueError(f"edge_h last dim must be {self.hidden_dim}, got {edge_h.shape[-1]}.")
        if query_h.shape[-1] != self.hidden_dim:
            raise ValueError(f"query_h last dim must be {self.hidden_dim}, got {query_h.shape[-1]}.")

        state_h = state_h.to(device=device, dtype=dtype)
        query_h = query_h.to(device=device, dtype=dtype)
        edge_h = edge_h.to(device=device, dtype=dtype)

        if state_h.shape[-1] != self.hidden_dim:
            raise ValueError(f"state_h last dim must be {self.hidden_dim}, got {state_h.shape[-1]}.")
        if row_ids.max().item() >= state_h.shape[0]:
            raise ValueError("row_ids contains row id outside state_h.")

        state_edge_h = state_h.index_select(0, row_ids)
        semantic = self.semantic_prior_scale * (
            query_sem_h.to(device=device, dtype=dtype)
            * rel_sem_h.to(device=device, dtype=dtype)
        ).sum(dim=-1)
        residual = self.residual(
            state_h=state_edge_h,
            query_h=query_h,
            edge_h=edge_h,
        )
        edge_log_energy = semantic + residual

        if self.edge_logit_shift is not None:
            edge_log_energy = edge_log_energy + self.edge_logit_shift.to(
                device=device,
                dtype=dtype,
            )

        return edge_log_energy.view(-1)

    def residual(
        self,
        *,
        state_h: torch.Tensor,
        query_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual_head(
            torch.cat(
                [
                    query_h,
                    state_h,
                    edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        return self.adapter_scale.to(device=residual.device, dtype=residual.dtype) * residual

    def _reset_parameters(self) -> None:
        for module in self.residual_head.modules():
            if isinstance(module, nn.Linear):
                init_xavier(module)

        final = self.residual_head[-1]
        if isinstance(final, nn.Linear):
            if self.adapter_final_init_scale == 0.0:
                nn.init.zeros_(final.weight)
            else:
                init_xavier(final)
                with torch.no_grad():
                    final.weight.mul_(self.adapter_final_init_scale)
            nn.init.zeros_(final.bias)


EdgeFlowScorer = EdgeActionScorer


__all__ = ["EdgeActionScorer", "EdgeFlowScorer"]
