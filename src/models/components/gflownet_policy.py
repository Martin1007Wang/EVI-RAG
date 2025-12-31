from __future__ import annotations

from typing import Tuple
import math

import torch
from torch import nn


def _zero_init_last_linear(module: nn.Module) -> None:
    """Zero-init the last Linear layer so the policy starts near-uniform logits."""
    last_linear = None
    for sub in reversed(list(module.modules())):
        if isinstance(sub, nn.Linear):
            last_linear = sub
            break
    if last_linear is None:
        raise ValueError("Expected nn.Linear in module for zero-init, but found none.")
    nn.init.constant_(last_linear.weight, 0.0)
    if last_linear.bias is not None:
        nn.init.constant_(last_linear.bias, 0.0)


def _segment_softmax_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Compute softmax(logits) independently within each segment (graph)."""
    if logits.dim() != 1 or segment_ids.dim() != 1:
        raise ValueError("logits and segment_ids must be 1D tensors.")
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have the same length.")
    if logits.numel() == 0:
        return logits
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")

    device = logits.device
    dtype = logits.dtype
    neg_inf = torch.finfo(dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)

    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    denom = torch.zeros((num_segments,), device=device, dtype=dtype)
    denom.index_add_(0, segment_ids, exp)
    eps = torch.finfo(dtype).eps
    return exp / denom[segment_ids].clamp(min=eps)


class GFlowNetEdgePolicy(nn.Module):
    """Edge policy with local attention pooling (undirected)."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        self.edge_proj_base = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.edge_proj_dropout = nn.Dropout(dropout)
        self.attn_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self._attn_scale = float(math.sqrt(self.hidden_dim))

        self.edge_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        _zero_init_last_linear(self.edge_head)
        _zero_init_last_linear(self.stop_head)

    def compute_edge_base(self, edge_tokens: torch.Tensor) -> torch.Tensor:
        if edge_tokens.dim() != 2 or edge_tokens.size(-1) != self.hidden_dim:
            raise ValueError("edge_tokens must be [E,H] with hidden_dim H.")
        return self.edge_proj_base(edge_tokens)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        state_tokens: torch.Tensor,        # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        valid_edges_mask: torch.Tensor,    # [E_total]
        edge_base: torch.Tensor | None = None,
        *,
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        if state_tokens.dim() != 2 or state_tokens.size(-1) != self.hidden_dim:
            raise ValueError("state_tokens must be [B,H] with hidden_dim H.")
        if edge_tokens.dim() != 2 or edge_tokens.size(-1) != self.hidden_dim:
            raise ValueError("edge_tokens must be [E,H] with hidden_dim H.")
        if edge_batch.dim() != 1 or edge_batch.numel() != edge_tokens.size(0):
            raise ValueError("edge_batch must be [E_total] and aligned with edge_tokens.")
        if valid_edges_mask.shape != edge_batch.shape:
            raise ValueError("valid_edges_mask must have shape [E_total] aligned with edge_batch.")

        num_graphs = int(state_tokens.size(0))
        if num_graphs <= 0:
            raise ValueError("state_tokens must have positive batch size.")

        if edge_base is None:
            edge_base = self.compute_edge_base(edge_tokens)
        if edge_base.shape != edge_tokens.shape:
            raise ValueError("edge_base must match edge_tokens shape [E,H].")
        edge_repr = self.edge_proj_dropout(edge_base)

        candidate = valid_edges_mask.to(device=device, dtype=torch.bool)
        neg_inf = float(torch.finfo(edge_repr.dtype).min)
        edge_logits = torch.full((edge_repr.size(0),), neg_inf, device=device, dtype=edge_repr.dtype)

        if not bool(candidate.any().item()):
            state_out = self.state_norm(state_tokens.to(device=device, dtype=edge_repr.dtype))
            stop_logits = self.stop_head(state_out).squeeze(-1)
            return edge_logits, stop_logits, state_out

        cand_idx = torch.nonzero(candidate, as_tuple=False).view(-1)
        seg = edge_batch[cand_idx].to(device=device, dtype=torch.long)
        if seg.numel() != cand_idx.numel():
            raise ValueError("candidate indexing mismatch for edge_batch.")

        state_base = self.state_norm(state_tokens.to(device=device, dtype=edge_repr.dtype))
        q = self.attn_q(state_base)[seg]
        k = self.attn_k(edge_repr[cand_idx])
        v = self.attn_v(edge_repr[cand_idx])
        att_logits = (q * k).sum(dim=-1) / max(self._attn_scale, 1.0)
        att_w = _segment_softmax_1d(att_logits, seg, num_segments=num_graphs).to(dtype=edge_repr.dtype)

        context = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=edge_repr.dtype)
        context.index_add_(0, seg, att_w.unsqueeze(-1) * v)
        state_out = self.state_norm(state_tokens.to(device=device, dtype=edge_repr.dtype) + context)

        edge_in = torch.cat([state_out[seg], edge_repr[cand_idx]], dim=-1)
        edge_logits[cand_idx] = self.edge_head(edge_in).squeeze(-1)

        stop_logits = self.stop_head(state_out).squeeze(-1)
        return edge_logits, stop_logits, state_out


__all__ = ["GFlowNetEdgePolicy"]
