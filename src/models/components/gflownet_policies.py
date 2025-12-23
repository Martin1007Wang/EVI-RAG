from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
from torch import nn

TYPE_SELECTED = 2


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


def _compute_lookahead_states(
    edge_repr_selected: torch.Tensor,
    edge_batch: torch.Tensor,
    selected_mask: torch.Tensor,
    context_tokens: torch.Tensor,
    *,
    graph_norm: nn.LayerNorm,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute phi(s) / phi(s') with a vectorized look-ahead aggregation."""
    device = edge_repr_selected.device
    hidden_dim = edge_repr_selected.size(-1)
    num_graphs = int(context_tokens.size(0))

    selected_mask_f = selected_mask.float().unsqueeze(-1)
    selected_sum = torch.zeros(num_graphs, hidden_dim, device=device)
    selected_sum.index_add_(0, edge_batch, edge_repr_selected * selected_mask_f)
    selected_count_raw = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)
    # Clamp only for division stability; keep the raw count for next_state to avoid double-counting at step0.
    selected_count = selected_count_raw.clamp(min=1.0).unsqueeze(-1)
    current_state = selected_sum / selected_count
    current_state = graph_norm(current_state + context_tokens)

    next_sum = selected_sum[edge_batch] + edge_repr_selected
    next_count = selected_count_raw[edge_batch].unsqueeze(-1) + 1.0
    next_state = next_sum / next_count
    next_state = graph_norm(next_state + context_tokens[edge_batch])
    return current_state, next_state


def _apply_order_embedding(
    edge_tokens: torch.Tensor,
    selection_order: Optional[torch.Tensor],
    *,
    max_steps: int,
    order_embeddings: nn.Embedding,
) -> torch.Tensor:
    if selection_order is None:
        return edge_tokens
    if selection_order.shape != edge_tokens.shape[:1]:
        raise ValueError("selection_order shape must match edge_tokens first dimension.")
    order_idx = selection_order.to(device=edge_tokens.device, dtype=torch.long).clamp(min=-1, max=max_steps) + 1
    return edge_tokens + order_embeddings(order_idx)


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


class EdgeMLPMixerPolicy(nn.Module):
    """轻量版策略：边级 MLP + 简单 token mixing，O(E)。"""

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        direction_vocab_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.type_embeddings = nn.Embedding(5, self.hidden_dim)
        self.order_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.order_embeddings.weight, 0.0)
        if int(direction_vocab_size) <= 0:
            raise ValueError(f"direction_vocab_size must be positive, got {direction_vocab_size}")
        self.direction_embeddings = nn.Embedding(int(direction_vocab_size), self.hidden_dim)
        nn.init.constant_(self.direction_embeddings.weight, 0.0)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        # FiLM-style调制：让 question_tokens 直接作用到每条边的表示上，避免“只靠结构”忽略语义。
        self.q_film = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        layers: list[nn.Module] = []
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.edge_mlp = nn.Sequential(*layers)
        self.lookahead_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.stop_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        _zero_init_last_linear(self.lookahead_head)
        _zero_init_last_linear(self.stop_proj)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        state_tokens: torch.Tensor,        # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        edge_direction: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        if state_tokens.shape != question_tokens.shape:
            raise ValueError("state_tokens must have the same shape as question_tokens.")

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
        if edge_direction is not None:
            if edge_direction.shape != edge_tokens.shape[:1]:
                raise ValueError("edge_direction shape must match edge_tokens length.")
            dir_idx = edge_direction.to(device=edge_tokens.device, dtype=torch.long).clamp(
                min=0,
                max=self.direction_embeddings.num_embeddings - 1,
            )
            edge_tokens = edge_tokens + self.direction_embeddings(dir_idx)
        edge_tokens = edge_tokens + self.q_film(question_tokens[edge_batch])
        selected_types = torch.full_like(selected_mask, TYPE_SELECTED, dtype=torch.long, device=device)
        type_embed_selected = self.type_embeddings(selected_types)
        edge_selected = self.edge_mlp(edge_tokens + type_embed_selected)

        current_state, next_state = _compute_lookahead_states(
            edge_selected,
            edge_batch,
            selected_mask,
            state_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        stop_input = torch.cat([current_state, state_tokens], dim=-1)
        stop_logits = self.stop_proj(stop_input).squeeze(-1)
        cls_out = current_state
        return edge_logits, stop_logits, cls_out


class EdgeFrontierPolicy(nn.Module):
    """局部前沿感知策略：利用当前节点和邻接边标记，避免全局 self-attention。"""

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.1,
        frontier_bonus: float = 0.5,
        direction_vocab_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.frontier_bonus = float(frontier_bonus)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.q_film = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 2),
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.lookahead_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.order_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.order_embeddings.weight, 0.0)
        if int(direction_vocab_size) <= 0:
            raise ValueError(f"direction_vocab_size must be positive, got {direction_vocab_size}")
        self.direction_embeddings = nn.Embedding(int(direction_vocab_size), self.hidden_dim)
        nn.init.constant_(self.direction_embeddings.weight, 0.0)
        _zero_init_last_linear(self.lookahead_head)
        _zero_init_last_linear(self.stop_head)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        state_tokens: torch.Tensor,        # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        frontier_mask: Optional[torch.Tensor] = None,  # [E_total]
        edge_direction: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        if state_tokens.shape != question_tokens.shape:
            raise ValueError("state_tokens must have the same shape as question_tokens.")

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
        if edge_direction is not None:
            if edge_direction.shape != edge_tokens.shape[:1]:
                raise ValueError("edge_direction shape must match edge_tokens length.")
            dir_idx = edge_direction.to(device=edge_tokens.device, dtype=torch.long).clamp(
                min=0,
                max=self.direction_embeddings.num_embeddings - 1,
            )
            edge_tokens = edge_tokens + self.direction_embeddings(dir_idx)
        edge_tokens = edge_tokens + self.q_film(question_tokens[edge_batch])
        candidate_mask = ~selected_mask
        if frontier_mask is None:
            frontier_mask = candidate_mask
            if current_tail is not None and edge_heads is not None and edge_tails is not None:
                cur = current_tail.to(device)
                heads = edge_heads.to(device)
                tails = edge_tails.to(device)
                frontier_mask = candidate_mask & (
                    (heads == cur[edge_batch]) | (tails == cur[edge_batch])
                )
        else:
            frontier_mask = frontier_mask.to(device) & candidate_mask

        aux_selected = torch.stack(
            [
                torch.zeros_like(candidate_mask, dtype=torch.float, device=device),
                frontier_mask.float(),
            ],
            dim=-1,
        )
        expanded_selected = torch.cat([edge_tokens, aux_selected], dim=-1)
        edge_repr_selected = self.edge_proj(expanded_selected)

        current_state, next_state = _compute_lookahead_states(
            edge_repr_selected,
            edge_batch,
            selected_mask,
            state_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        edge_logits = edge_logits + self.frontier_bonus * frontier_mask.float()

        stop_input = torch.cat([current_state, state_tokens], dim=-1)
        stop_logits = self.stop_head(stop_input).squeeze(-1)
        cls_out = current_state
        return edge_logits, stop_logits, cls_out


class EdgeGATPolicy(nn.Module):
    """边级注意力版策略：用全局 query 对候选边做一层 GAT-style 打分。"""

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.1,
        frontier_bonus: float = 0.5,
        direction_vocab_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.frontier_bonus = float(frontier_bonus)
        self.edge_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.q_film = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.lookahead_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.order_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.order_embeddings.weight, 0.0)
        if int(direction_vocab_size) <= 0:
            raise ValueError(f"direction_vocab_size must be positive, got {direction_vocab_size}")
        self.direction_embeddings = nn.Embedding(int(direction_vocab_size), self.hidden_dim)
        nn.init.constant_(self.direction_embeddings.weight, 0.0)
        _zero_init_last_linear(self.lookahead_head)
        _zero_init_last_linear(self.stop_head)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        state_tokens: torch.Tensor,        # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        frontier_mask: Optional[torch.Tensor] = None,  # [E_total]
        edge_direction: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        if state_tokens.shape != question_tokens.shape:
            raise ValueError("state_tokens must have the same shape as question_tokens.")

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
        if edge_direction is not None:
            if edge_direction.shape != edge_tokens.shape[:1]:
                raise ValueError("edge_direction shape must match edge_tokens length.")
            dir_idx = edge_direction.to(device=edge_tokens.device, dtype=torch.long).clamp(
                min=0,
                max=self.direction_embeddings.num_embeddings - 1,
            )
            edge_tokens = edge_tokens + self.direction_embeddings(dir_idx)
        edge_tokens = edge_tokens + self.q_film(question_tokens[edge_batch])
        candidate_mask = ~selected_mask
        if frontier_mask is None:
            frontier_mask = candidate_mask
            if current_tail is not None and edge_heads is not None and edge_tails is not None:
                cur = current_tail.to(device)
                heads = edge_heads.to(device)
                tails = edge_tails.to(device)
                frontier_mask = candidate_mask & (
                    (heads == cur[edge_batch]) | (tails == cur[edge_batch])
                )
        else:
            frontier_mask = frontier_mask.to(device) & candidate_mask

        edge_repr = self.edge_proj(edge_tokens)

        current_state, next_state = _compute_lookahead_states(
            edge_repr,
            edge_batch,
            selected_mask,
            state_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        edge_logits = edge_logits + self.frontier_bonus * frontier_mask.float()

        stop_input = torch.cat([current_state, state_tokens], dim=-1)
        stop_logits = self.stop_head(stop_input).squeeze(-1)
        cls_out = current_state
        return edge_logits, stop_logits, cls_out


class EdgeLocalGATPolicy(nn.Module):
    """局部邻域稀疏策略：仅对合法动作边打分，并做 query-aware pooling 得到 state embedding。

    目标是避免对整个 E_total 的边做统一前向：在每一步仅处理候选集合 A(s) 对应的边。
    """

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.1,
        direction_vocab_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        self.q_film = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        if int(direction_vocab_size) <= 0:
            raise ValueError(f"direction_vocab_size must be positive, got {direction_vocab_size}")
        self.direction_embeddings = nn.Embedding(int(direction_vocab_size), self.hidden_dim)
        nn.init.constant_(self.direction_embeddings.weight, 0.0)

        self.edge_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        _zero_init_last_linear(self.edge_head)
        _zero_init_last_linear(self.stop_head)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        state_tokens: torch.Tensor,        # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        frontier_mask: Optional[torch.Tensor] = None,     # [E_total]
        valid_edges_mask: Optional[torch.Tensor] = None,  # [E_total]
        edge_direction: Optional[torch.Tensor] = None,    # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        if state_tokens.shape != question_tokens.shape:
            raise ValueError("state_tokens must have the same shape as question_tokens.")
        if edge_tokens.dim() != 2 or edge_tokens.size(-1) != self.hidden_dim:
            raise ValueError("edge_tokens must be [E,H] with hidden_dim H.")
        if edge_batch.dim() != 1 or edge_batch.numel() != edge_tokens.size(0):
            raise ValueError("edge_batch must be [E_total] and aligned with edge_tokens.")
        if selected_mask.shape != edge_batch.shape:
            raise ValueError("selected_mask must have shape [E_total] aligned with edge_batch.")

        num_graphs = int(question_tokens.size(0))
        if num_graphs <= 0:
            raise ValueError("question_tokens must have positive batch size.")

        candidate = ~selected_mask.to(device=device, dtype=torch.bool)
        if valid_edges_mask is not None:
            if valid_edges_mask.shape != candidate.shape:
                raise ValueError("valid_edges_mask must have the same shape as selected_mask.")
            candidate = candidate & valid_edges_mask.to(device=device, dtype=torch.bool)
        if frontier_mask is not None:
            if frontier_mask.shape != candidate.shape:
                raise ValueError("frontier_mask must have the same shape as selected_mask.")
            candidate = candidate & frontier_mask.to(device=device, dtype=torch.bool)

        neg_inf = float(torch.finfo(edge_tokens.dtype).min)
        edge_logits = torch.full((edge_tokens.size(0),), neg_inf, device=device, dtype=edge_tokens.dtype)

        if not bool(candidate.any().item()):
            state_out = self.state_norm(state_tokens.to(device=device, dtype=edge_tokens.dtype))
            stop_in = torch.cat([state_out, state_tokens.to(device=device, dtype=edge_tokens.dtype)], dim=-1)
            stop_logits = self.stop_head(stop_in).squeeze(-1)
            return edge_logits, stop_logits, state_out

        cand_idx = torch.nonzero(candidate, as_tuple=False).view(-1)
        seg = edge_batch[cand_idx].to(device=device, dtype=torch.long)
        if seg.numel() != cand_idx.numel():
            raise ValueError("candidate indexing mismatch for edge_batch.")

        edge_repr = edge_tokens[cand_idx]
        if edge_direction is not None:
            if edge_direction.shape != edge_batch.shape:
                raise ValueError("edge_direction must have shape [E_total] aligned with edge_batch.")
            dir_idx = edge_direction[cand_idx].to(device=device, dtype=torch.long).clamp(
                min=0, max=self.direction_embeddings.num_embeddings - 1
            )
            edge_repr = edge_repr + self.direction_embeddings(dir_idx)

        edge_repr = edge_repr + self.q_film(question_tokens.to(device=device, dtype=edge_repr.dtype)[seg])
        edge_repr = self.edge_proj(edge_repr)

        state_base = self.state_norm(state_tokens.to(device=device, dtype=edge_repr.dtype))
        q = self.attn_q(state_base)[seg]
        k = self.attn_k(edge_repr)
        v = self.attn_v(edge_repr)
        att_logits = (q * k).sum(dim=-1) / max(self._attn_scale, 1.0)
        att_w = _segment_softmax_1d(att_logits, seg, num_segments=num_graphs).to(dtype=edge_repr.dtype)
        context = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=edge_repr.dtype)
        context.index_add_(0, seg, att_w.unsqueeze(-1) * v)
        state_out = self.state_norm(state_tokens.to(device=device, dtype=edge_repr.dtype) + context)

        edge_in = torch.cat([state_out[seg], edge_repr], dim=-1)
        edge_logits[cand_idx] = self.edge_head(edge_in).squeeze(-1)

        stop_in = torch.cat([state_out, state_tokens.to(device=device, dtype=edge_repr.dtype)], dim=-1)
        stop_logits = self.stop_head(stop_in).squeeze(-1)
        return edge_logits, stop_logits, state_out


__all__ = [
    "EdgeMLPMixerPolicy",
    "EdgeFrontierPolicy",
    "EdgeGATPolicy",
    "EdgeLocalGATPolicy",
]
