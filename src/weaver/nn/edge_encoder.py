from __future__ import annotations

import torch
from torch import nn


class EdgeEncoder(nn.Module):
    """
    Encode a KG edge e=(u,r,v) as a model-space token for residual
    scoring and state summarization.

    Inputs are already projected into the same model space by FeatureEncoder:

        src_h: [E, H]
        rel_h: [E, H]
        dst_h: [E, H]

    This module is not used for raw PLM semantic similarity. All semantic
    dot products must be computed from FrontierEncoding.query_sem_h /
    rel_sem_h in the original L2-normalized PLM space.

    Edge representation:

        h_e = LN(w_src * h_u + w_rel * h_r + w_dst * h_v)

    The role weights provide a small structural inductive bias for residual
    features. They should not be interpreted as semantic prior weights.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        src_weight_init: float = 0.2,
        rel_weight_init: float = 1.0,
        dst_weight_init: float = 0.8,
        learn_role_weights: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        init = torch.tensor(
            [
                float(src_weight_init),
                float(rel_weight_init),
                float(dst_weight_init),
            ],
            dtype=torch.float32,
        )

        if bool(init.le(0).any()):
            raise ValueError("src_weight_init, rel_weight_init, and dst_weight_init " "must all be positive.")

        # Softmax logits. Initial normalized weights are proportional to
        # [src_weight_init, rel_weight_init, dst_weight_init].
        self.role_logits = nn.Parameter(init.log())
        self.role_logits.requires_grad_(bool(learn_role_weights))

        self.norm = nn.LayerNorm(self.hidden_dim)

    @property
    def role_weights(self) -> torch.Tensor:
        return self.role_logits.softmax(dim=0)

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> torch.Tensor:
        self._check_inputs(src_h=src_h, rel_h=rel_h, dst_h=dst_h)

        weights = self.role_weights.to(device=src_h.device, dtype=src_h.dtype)

        edge_h = weights[0] * src_h + weights[1] * rel_h + weights[2] * dst_h

        return self.norm(edge_h)

    def _check_inputs(
        self,
        *,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> None:
        if src_h.ndim != 2:
            raise ValueError(f"src_h must have shape [E, H], got {tuple(src_h.shape)}.")
        if rel_h.ndim != 2:
            raise ValueError(f"rel_h must have shape [E, H], got {tuple(rel_h.shape)}.")
        if dst_h.ndim != 2:
            raise ValueError(f"dst_h must have shape [E, H], got {tuple(dst_h.shape)}.")

        if src_h.shape != rel_h.shape or src_h.shape != dst_h.shape:
            raise ValueError(
                "src_h, rel_h, and dst_h must have the same shape: "
                f"src={tuple(src_h.shape)}, "
                f"rel={tuple(rel_h.shape)}, "
                f"dst={tuple(dst_h.shape)}."
            )

        if int(src_h.size(-1)) != self.hidden_dim:
            raise ValueError(f"last dim must be hidden_dim={self.hidden_dim}, " f"got {int(src_h.size(-1))}.")


__all__ = ["EdgeEncoder"]
