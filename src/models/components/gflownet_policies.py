from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

TYPE_SELECTED = 2
TYPE_CANDIDATE = 3


class EdgeMLPMixerPolicy(nn.Module):
    """轻量版策略：边级 MLP + 简单 token mixing，O(E)。"""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.type_embeddings = nn.Embedding(5, self.hidden_dim)
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
        self.selection_head = nn.Linear(self.hidden_dim, 1)
        self.stop_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        num_graphs = int(question_tokens.size(0))

        base_types = torch.full_like(selected_mask, TYPE_CANDIDATE, dtype=torch.long, device=device)
        base_types = torch.where(selected_mask, torch.full_like(base_types, TYPE_SELECTED), base_types)
        type_embed = self.type_embeddings(base_types)
        mixed_edges = self.edge_mlp(edge_tokens + type_embed)

        edge_logits = self.selection_head(mixed_edges).squeeze(-1)

        denom = torch.bincount(edge_batch, minlength=num_graphs).clamp(min=1).to(device).unsqueeze(-1)
        pooled_edges = torch.zeros(num_graphs, mixed_edges.size(-1), device=device)
        pooled_edges.index_add_(0, edge_batch, mixed_edges)
        pooled_edges = pooled_edges / denom
        stop_input = torch.cat([pooled_edges, question_tokens], dim=-1)
        stop_logits = self.stop_proj(stop_input).squeeze(-1)
        cls_out = pooled_edges
        return edge_logits, stop_logits, cls_out


class EdgeFrontierPolicy(nn.Module):
    """局部前沿感知策略：利用当前节点和邻接边标记，避免全局 self-attention。"""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        frontier_bonus: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.frontier_bonus = float(frontier_bonus)
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 2),
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.selection_head = nn.Linear(self.hidden_dim, 1)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        num_graphs = int(question_tokens.size(0))

        candidate_mask = ~selected_mask
        frontier_mask = candidate_mask
        if current_tail is not None and edge_heads is not None and edge_tails is not None:
            cur = current_tail.to(device)
            heads = edge_heads.to(device)
            tails = edge_tails.to(device)
            frontier_mask = candidate_mask & (
                (heads == cur[edge_batch]) | (tails == cur[edge_batch])
            )

        aux_features = torch.stack(
            [
                candidate_mask.float(),
                frontier_mask.float(),
            ],
            dim=-1,
        )
        expanded_aux = torch.cat([edge_tokens, aux_features], dim=-1)
        edge_repr = self.edge_proj(expanded_aux)
        edge_logits = self.selection_head(edge_repr).squeeze(-1)
        edge_logits = edge_logits + self.frontier_bonus * frontier_mask.float()

        denom = torch.bincount(edge_batch, minlength=num_graphs).clamp(min=1).to(device).unsqueeze(-1)
        pooled_edges = torch.zeros(num_graphs, edge_repr.size(-1), device=device)
        pooled_edges.index_add_(0, edge_batch, edge_repr)
        pooled_edges = pooled_edges / denom
        stop_input = torch.cat([pooled_edges, question_tokens], dim=-1)
        stop_logits = self.stop_head(stop_input).squeeze(-1)
        cls_out = pooled_edges
        return edge_logits, stop_logits, cls_out


class EdgeGATPolicy(nn.Module):
    """边级注意力版策略：用全局 query 对候选边做一层 GAT-style 打分。"""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        frontier_bonus: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.frontier_bonus = float(frontier_bonus)
        self.edge_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.att_vec = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.xavier_uniform_(self.att_vec.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        edge_tokens: torch.Tensor,         # [E_total, H]
        question_tokens: torch.Tensor,     # [B, H]
        edge_batch: torch.Tensor,          # [E_total]
        selected_mask: torch.Tensor,       # [E_total]
        *,
        edge_heads: Optional[torch.Tensor] = None,  # [E_total]
        edge_tails: Optional[torch.Tensor] = None,  # [E_total]
        current_tail: Optional[torch.Tensor] = None,  # [B]
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_tokens.device
        num_graphs = int(question_tokens.size(0))

        candidate_mask = ~selected_mask
        frontier_mask = candidate_mask
        if current_tail is not None and edge_heads is not None and edge_tails is not None:
            cur = current_tail.to(device)
            heads = edge_heads.to(device)
            tails = edge_tails.to(device)
            frontier_mask = candidate_mask & (
                (heads == cur[edge_batch]) | (tails == cur[edge_batch])
            )

        edge_h = self.edge_proj(edge_tokens)
        q_h = self.query_proj(question_tokens)
        att_inp = edge_h + q_h[edge_batch]
        att_raw = (att_inp * self.att_vec.view(1, -1)).sum(dim=-1)
        att_raw = self.leaky_relu(att_raw)
        att_raw = att_raw + self.frontier_bonus * frontier_mask.float()

        max_per_graph = torch.full((num_graphs,), float("-inf"), device=device)
        max_per_graph.scatter_reduce_(0, edge_batch, att_raw, reduce="amax", include_self=True)
        norm_logits = att_raw - max_per_graph[edge_batch]
        exp = torch.exp(norm_logits) * candidate_mask.float()
        sum_per_graph = torch.zeros(num_graphs, device=device)
        sum_per_graph.scatter_add_(0, edge_batch, exp)
        edge_probs = exp / sum_per_graph.clamp(min=torch.finfo(exp.dtype).eps)[edge_batch]
        edge_probs = self.dropout(edge_probs)
        edge_logits = torch.log(edge_probs.clamp(min=torch.finfo(edge_probs.dtype).eps))

        denom = torch.bincount(edge_batch, minlength=num_graphs).clamp(min=1).to(device).unsqueeze(-1)
        pooled_edges = torch.zeros(num_graphs, edge_h.size(-1), device=device)
        pooled_edges.index_add_(0, edge_batch, edge_h)
        pooled_edges = pooled_edges / denom
        stop_input = torch.cat([pooled_edges, question_tokens], dim=-1)
        stop_logits = self.stop_head(stop_input).squeeze(-1)
        cls_out = pooled_edges
        return edge_logits, stop_logits, cls_out


__all__ = [
    "EdgeMLPMixerPolicy",
    "EdgeFrontierPolicy",
    "EdgeGATPolicy",
]
