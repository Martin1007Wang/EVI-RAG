from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .fusion import FiLMLayer

if TYPE_CHECKING:  # pragma: no cover
    from .gflownet_env import GraphState


DIR_UNKNOWN = 0
DIR_FORWARD = 1
DIR_REVERSE = 2


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


class _EdgeMPNNLayer(nn.Module):
    """Edge-conditioned mean-aggregation layer (O(E))."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.src_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.hidden_dim:
            raise ValueError("x must be [N,H] with hidden_dim H.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be [2,E].")
        if edge_attr.dim() != 2 or edge_attr.size(-1) != self.hidden_dim:
            raise ValueError("edge_attr must be [E,H] with hidden_dim H.")
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError("edge_attr length must match edge_index num_edges.")

        if edge_index.numel() == 0:
            return self.norm(x)

        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        msg = self.src_proj(x[src]) + self.edge_proj(edge_attr)
        msg = F.gelu(msg)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        deg = torch.bincount(dst, minlength=x.size(0)).clamp(min=1).to(dtype=x.dtype).unsqueeze(-1)
        agg = agg / deg

        out = x + self.dropout(self.out_proj(agg))
        return self.norm(out)


@dataclass(frozen=True)
class GNNStateEncoderCache:
    node_ctx: torch.Tensor          # [N_total, H]
    edge_ctx: torch.Tensor          # [E_total, H]
    node_batch: torch.Tensor        # [N_total]
    edge_batch: torch.Tensor        # [E_total]
    question_tokens: torch.Tensor   # [B, H]
    start_ctx: torch.Tensor         # [B, H]


class GNNStateEncoder(nn.Module):
    """Graph (GNN) state encoder for path-mode GFlowNet.

    Precompute node contexts with an edge-conditioned MPNN on the batched retrieval graph, then
    encode each environment state via query-aware pooling over visited nodes + selected edges.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_start_state: bool = True,
        use_question_film: bool = True,
        use_direction_emb: bool = True,
        direction_vocab_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        self.num_layers = int(max(0, num_layers))
        self.use_start_state = bool(use_start_state)
        self.use_question_film = bool(use_question_film)
        self.use_direction_emb = bool(use_direction_emb)

        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.question_film: Optional[FiLMLayer] = FiLMLayer(self.hidden_dim, self.hidden_dim) if self.use_question_film else None

        self.direction_embeddings: Optional[nn.Embedding] = None
        if self.use_direction_emb:
            if int(direction_vocab_size) <= 0:
                raise ValueError(f"direction_vocab_size must be positive, got {direction_vocab_size}")
            self.direction_embeddings = nn.Embedding(int(direction_vocab_size), self.hidden_dim)
            nn.init.zeros_(self.direction_embeddings.weight)

        self.node_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.layers = nn.ModuleList([_EdgeMPNNLayer(self.hidden_dim, dropout=float(dropout)) for _ in range(self.num_layers)])

        # Zero-init to avoid overpowering retriever features at initialization.
        self.order_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.order_embeddings.weight, 0.0)
        self.step_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.step_embeddings.weight, 0.0)

        self.pool_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self._attn_scale = float(max(1.0, math.sqrt(self.hidden_dim)))

    def precompute(
        self,
        *,
        edge_index: torch.Tensor,      # [2, E_total]
        edge_batch: torch.Tensor,      # [E_total]
        node_ptr: torch.Tensor,        # [B+1]
        start_node_locals: torch.Tensor,  # [S_total]
        start_ptr: torch.Tensor,       # [B+1]
        node_tokens: torch.Tensor,      # [N_total, H]
        edge_tokens: torch.Tensor,      # [E_total, H]
        question_tokens: torch.Tensor,  # [B, H]
    ) -> GNNStateEncoderCache:
        device = node_tokens.device
        dtype = node_tokens.dtype
        node_ptr = node_ptr.to(device=device, dtype=torch.long).view(-1)
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("Graph batch must have positive num_graphs.")
        if question_tokens.shape != (num_graphs, self.hidden_dim):
            raise ValueError(
                f"question_tokens must be [B,H]=({num_graphs},{self.hidden_dim}), got {tuple(question_tokens.shape)}"
            )
        if node_tokens.dim() != 2 or node_tokens.size(-1) != self.hidden_dim:
            raise ValueError("node_tokens must be [N,H] with hidden_dim H.")
        if edge_tokens.dim() != 2 or edge_tokens.size(-1) != self.hidden_dim:
            raise ValueError("edge_tokens must be [E,H] with hidden_dim H.")
        if int(node_ptr[-1].item()) != int(node_tokens.size(0)):
            raise ValueError("node_ptr[-1] must equal total number of nodes in node_tokens.")
        if int(edge_index.size(1)) != int(edge_tokens.size(0)):
            raise ValueError("edge_tokens length must match edge_index num_edges.")
        edge_batch = edge_batch.to(device=device, dtype=torch.long).view(-1)
        if edge_batch.numel() != int(edge_tokens.size(0)):
            raise ValueError("edge_batch length mismatch with edge_tokens.")

        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)
        if node_batch.numel() != node_tokens.size(0):
            raise ValueError("node_batch length mismatch with node_tokens.")

        node_h = self.node_proj(node_tokens)
        if self.question_film is not None:
            node_h = self.question_film(node_h, question_tokens.to(device=device, dtype=dtype)[node_batch])

        edge_h = self.edge_proj(edge_tokens)

        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_index_aug = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if self.direction_embeddings is not None:
            fwd = self.direction_embeddings.weight[DIR_FORWARD].to(device=device, dtype=dtype)
            rev = self.direction_embeddings.weight[DIR_REVERSE].to(device=device, dtype=dtype)
            edge_attr_aug = torch.cat([edge_h + fwd, edge_h + rev], dim=0)
        else:
            edge_attr_aug = torch.cat([edge_h, edge_h], dim=0)

        x = node_h
        for layer in self.layers:
            x = layer(x, edge_index_aug, edge_attr_aug)
        node_ctx = x

        start_ctx = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        if self.use_start_state:
            start_nodes = start_node_locals.to(device=device, dtype=torch.long).view(-1)
            start_ptr = start_ptr.to(device=device, dtype=torch.long).view(-1)
            if start_ptr.numel() != num_graphs + 1:
                raise ValueError("start_ptr length mismatch with num_graphs.")
            start_counts = (start_ptr[1:] - start_ptr[:-1]).clamp(min=0)
            if start_nodes.numel() != int(start_counts.sum().item()):
                raise ValueError("start_node_locals length mismatch with start_ptr.")
            if start_nodes.numel() > 0:
                start_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), start_counts)
                start_ctx.index_add_(0, start_batch, node_ctx[start_nodes])
            denom = start_counts.clamp(min=1).to(dtype=dtype).unsqueeze(-1)
            start_ctx = start_ctx / denom

        return GNNStateEncoderCache(
            node_ctx=node_ctx,
            edge_ctx=edge_h,
            node_batch=node_batch,
            edge_batch=edge_batch,
            question_tokens=question_tokens.to(device=device, dtype=dtype),
            start_ctx=start_ctx,
        )

    def encode_state(
        self,
        *,
        cache: GNNStateEncoderCache,
        state: GraphState,
    ) -> torch.Tensor:
        num_graphs = int(cache.question_tokens.size(0))
        device = cache.question_tokens.device
        dtype = cache.question_tokens.dtype

        step_idx = state.step_counts.to(device=device, dtype=torch.long).clamp(min=0, max=self.max_steps + 1)
        base = cache.question_tokens + cache.start_ctx + self.step_embeddings(step_idx)

        ctx_nodes = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        visited = state.visited_nodes.to(device=device, dtype=torch.bool).view(-1)
        if bool(visited.any().item()):
            visited_idx = torch.nonzero(visited, as_tuple=False).view(-1)
            seg = cache.node_batch[visited_idx].to(device=device, dtype=torch.long)
            feat = cache.node_ctx[visited_idx]
            q = self.pool_q(base)[seg]
            k = self.node_k(feat)
            v = self.node_v(feat)
            att_logits = (q * k).sum(dim=-1) / self._attn_scale
            att_w = _segment_softmax_1d(att_logits, seg, num_segments=num_graphs).to(dtype=dtype)
            ctx_nodes.index_add_(0, seg, att_w.unsqueeze(-1) * v)

        ctx_edges = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        selected = state.selected_mask.to(device=device, dtype=torch.bool).view(-1)
        if bool(selected.any().item()):
            sel_idx = torch.nonzero(selected, as_tuple=False).view(-1)
            seg_e = cache.edge_batch[sel_idx].to(device=device, dtype=torch.long)
            feat_e = cache.edge_ctx[sel_idx]
            order_raw = state.selection_order[sel_idx].to(device=device, dtype=torch.long)
            order_idx = order_raw.clamp(min=-1, max=self.max_steps) + 1
            feat_e = feat_e + self.order_embeddings(order_idx)

            q_e = self.pool_q(base)[seg_e]
            k_e = self.edge_k(feat_e)
            v_e = self.edge_v(feat_e)
            att_logits_e = (q_e * k_e).sum(dim=-1) / self._attn_scale
            att_w_e = _segment_softmax_1d(att_logits_e, seg_e, num_segments=num_graphs).to(dtype=dtype)
            ctx_edges.index_add_(0, seg_e, att_w_e.unsqueeze(-1) * v_e)

        return self.state_norm(base + ctx_nodes + ctx_edges)


__all__ = ["GNNStateEncoder", "GNNStateEncoderCache"]
