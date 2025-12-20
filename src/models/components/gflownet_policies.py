from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

TYPE_SELECTED = 2


def _zero_init_last_linear(module: nn.Module) -> None:
    """Zero-init the last Linear layer so the policy starts as a pure residual (all-zeros)."""
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
    question_tokens: torch.Tensor,
    *,
    graph_norm: nn.LayerNorm,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute phi(s) / phi(s') with a vectorized look-ahead aggregation."""
    device = edge_repr_selected.device
    hidden_dim = edge_repr_selected.size(-1)
    num_graphs = int(question_tokens.size(0))

    selected_mask_f = selected_mask.float().unsqueeze(-1)
    selected_sum = torch.zeros(num_graphs, hidden_dim, device=device)
    selected_sum.index_add_(0, edge_batch, edge_repr_selected * selected_mask_f)
    selected_count_raw = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)
    # Clamp only for division stability; keep the raw count for next_state to avoid double-counting at step0.
    selected_count = selected_count_raw.clamp(min=1.0).unsqueeze(-1)
    current_state = selected_sum / selected_count
    current_state = graph_norm(current_state + question_tokens)

    next_sum = selected_sum[edge_batch] + edge_repr_selected
    next_count = selected_count_raw[edge_batch].unsqueeze(-1) + 1.0
    next_state = next_sum / next_count
    next_state = graph_norm(next_state + question_tokens[edge_batch])
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


class EdgeMLPMixerPolicy(nn.Module):
    """轻量版策略：边级 MLP + 简单 token mixing，O(E)。"""

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.type_embeddings = nn.Embedding(5, self.hidden_dim)
        self.order_embeddings = nn.Embedding(self.max_steps + 2, self.hidden_dim)
        nn.init.constant_(self.order_embeddings.weight, 0.0)
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
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
        edge_tokens = edge_tokens + self.q_film(question_tokens[edge_batch])
        selected_types = torch.full_like(selected_mask, TYPE_SELECTED, dtype=torch.long, device=device)
        type_embed_selected = self.type_embeddings(selected_types)
        edge_selected = self.edge_mlp(edge_tokens + type_embed_selected)

        current_state, next_state = _compute_lookahead_states(
            edge_selected,
            edge_batch,
            selected_mask,
            question_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        stop_input = torch.cat([current_state, question_tokens], dim=-1)
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
        _zero_init_last_linear(self.lookahead_head)
        _zero_init_last_linear(self.stop_head)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        frontier_mask: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
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
            question_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        edge_logits = edge_logits + self.frontier_bonus * frontier_mask.float()

        stop_input = torch.cat([current_state, question_tokens], dim=-1)
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
        _zero_init_last_linear(self.lookahead_head)
        _zero_init_last_linear(self.stop_head)

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        selection_order: Optional[torch.Tensor] = None,  # [E_total]
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        frontier_mask: Optional[torch.Tensor] = None,  # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device

        edge_tokens = _apply_order_embedding(
            edge_tokens,
            selection_order,
            max_steps=self.max_steps,
            order_embeddings=self.order_embeddings,
        )
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
            question_tokens,
            graph_norm=self.graph_norm,
        )

        edge_logits = self.lookahead_head(torch.cat([current_state[edge_batch], next_state], dim=-1)).squeeze(-1)
        edge_logits = edge_logits + self.frontier_bonus * frontier_mask.float()

        stop_input = torch.cat([current_state, question_tokens], dim=-1)
        stop_logits = self.stop_head(stop_input).squeeze(-1)
        cls_out = current_state
        return edge_logits, stop_logits, cls_out


__all__ = [
    "EdgeMLPMixerPolicy",
    "EdgeFrontierPolicy",
    "EdgeGATPolicy",
]
