from __future__ import annotations

import torch
from torch import nn

_ZERO = 0
_ONE = 1
_TWO = 2
_HALF = 0.5


class BilinearStepScorer(nn.Module):
    def __init__(self, *, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= _ZERO:
            raise ValueError("dim must be > 0.")
        self.w_query_shift = nn.Linear(self.dim, self.dim, bias=False)
        self.scale = float(self.dim) ** -_HALF

    def forward(
        self,
        context: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tail.dim() == _TWO:
            return self._forward_edges(
                context=context,
                head=head,
                relation=relation,
                tail=tail,
                mask=mask,
            )
        if tail.dim() != 3:
            raise ValueError("tail must be 2D (edges) or 3D (B,K).")
        return self._forward_grouped(
            context=context,
            head=head,
            relation=relation,
            tail=tail,
            mask=mask,
        )

    def _forward_edges(
        self,
        *,
        context: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.dim() != _TWO or head.dim() != _TWO:
            raise ValueError("context and head must be [E, d].")
        if relation.dim() != _TWO or tail.dim() != _TWO:
            raise ValueError("relation and tail must be [E, d].")
        if (
            context.size(0) != head.size(0)
            or head.size(0) != relation.size(0)
            or head.size(0) != tail.size(0)
        ):
            raise ValueError("edge inputs must share the same first dimension.")
        rel_shifted = relation + self.w_query_shift(context)
        score = (head * rel_shifted * tail).sum(dim=-1) * self.scale
        if mask is not None:
            if mask.dim() != _ONE:
                raise ValueError("edge mask must be 1D.")
            score = score.masked_fill(mask == 0, torch.finfo(score.dtype).min)
        return score

    def _forward_grouped(
        self,
        *,
        context: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.dim() != _TWO or head.dim() != _TWO:
            raise ValueError("context and head must be [B, d].")
        if relation.dim() != 3 or tail.dim() != 3:
            raise ValueError("relation and tail must be [B, K, d].")
        if context.size(0) != head.size(0) or context.size(0) != tail.size(0):
            raise ValueError("grouped inputs must share the same batch size.")
        rel_shifted = relation + self.w_query_shift(context).unsqueeze(1)
        score = (head.unsqueeze(1) * rel_shifted * tail).sum(dim=-1) * self.scale
        if mask is not None:
            if mask.dim() != _TWO:
                raise ValueError("grouped mask must be [B, K].")
            score = score.masked_fill(mask == 0, torch.finfo(score.dtype).min)
        return score


__all__ = ["BilinearStepScorer"]
