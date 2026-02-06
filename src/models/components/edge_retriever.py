from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.data.schema.constants import _NON_TEXT_EMBEDDING_ID

_ZERO = 0
_ONE = 1


class _PEConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="mean")

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x_j


class DDE(nn.Module):
    """Deterministic diffusion-based encoding used by the edge-iid retriever."""

    def __init__(self, *, num_rounds: int, num_reverse_rounds: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PEConv() for _ in range(int(num_rounds))])
        self.reverse_layers = nn.ModuleList([_PEConv() for _ in range(int(num_reverse_rounds))])

    def forward(
        self,
        topic_entity_one_hot: torch.Tensor,
        edge_index: torch.Tensor,
        reverse_edge_index: torch.Tensor,
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        h = topic_entity_one_hot
        for layer in self.layers:
            h = layer(edge_index, h)
            out.append(h)
        h_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_rev = layer(reverse_edge_index, h_rev)
            out.append(h_rev)
        return out


class EdgeRetriever(nn.Module):
    """Edge-independent triple scorer (query-conditioned).

    This is an i.i.d. edge retriever: each edge (u, r, v) is scored independently
    conditioned on the question embedding and local structural features (DDE).
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        topic_pe: bool,
        dde_num_rounds: int,
        dde_num_reverse_rounds: int,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.topic_pe = bool(topic_pe)
        self.non_text_entity_emb = nn.Embedding(_ONE, self.emb_dim)
        self.dde = DDE(num_rounds=int(dde_num_rounds), num_reverse_rounds=int(dde_num_reverse_rounds))

        pred_in_size = 4 * self.emb_dim
        if self.topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (int(dde_num_rounds) + int(dde_num_reverse_rounds))

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, _ONE),
        )

    def forward(
        self,
        *,
        edge_index: torch.Tensor,  # Long[2, E]
        edge_rel_emb: torch.Tensor,  # Float[E, D]
        node_emb: torch.Tensor,  # Float[N, D]
        node_embedding_ids: torch.Tensor,  # Long[N]
        question_emb: torch.Tensor,  # Float[1, D] or Float[D]
        q_local_indices: torch.Tensor,  # Long[Kq]
    ) -> torch.Tensor:
        if edge_index.numel() == _ZERO:
            return edge_rel_emb.new_empty((_ZERO,))
        if edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        num_nodes = int(node_emb.size(0))
        if num_nodes == _ZERO:
            return edge_rel_emb.new_empty((_ZERO,))

        device = edge_rel_emb.device
        edge_index = edge_index.to(device=device, dtype=torch.long)
        node_emb = node_emb.to(device=device)
        node_embedding_ids = node_embedding_ids.to(device=device, dtype=torch.long).view(-1)
        q_local_indices = q_local_indices.to(device=device, dtype=torch.long).view(-1)

        # Replace non-text entity embeddings with a learnable vector (reference behavior).
        non_text_mask = node_embedding_ids == int(_NON_TEXT_EMBEDDING_ID)
        if bool(non_text_mask.any().detach().tolist()):
            node_emb = node_emb.clone()
            node_emb[non_text_mask] = self.non_text_entity_emb.weight[0].to(dtype=node_emb.dtype, device=device)

        topic_mask = torch.zeros((num_nodes,), dtype=torch.long, device=device)
        if q_local_indices.numel() > _ZERO:
            topic_mask[q_local_indices.clamp(min=_ZERO, max=num_nodes - _ONE)] = 1
        topic_one_hot = F.one_hot(topic_mask, num_classes=2).to(dtype=node_emb.dtype)

        reverse_edge_index = torch.stack([edge_index[_ONE], edge_index[_ZERO]], dim=0)
        dde_list = self.dde(topic_one_hot, edge_index, reverse_edge_index)

        node_feat_parts: list[torch.Tensor] = [node_emb]
        if self.topic_pe:
            node_feat_parts.append(topic_one_hot)
        node_feat_parts.extend(dde_list)
        node_feat = torch.cat(node_feat_parts, dim=1)

        q_emb = question_emb
        if q_emb.dim() == 2 and int(q_emb.size(0)) == 1:
            q_emb = q_emb.squeeze(0)
        q_emb = q_emb.to(device=device, dtype=node_emb.dtype).view(1, -1)

        src = edge_index[_ZERO]
        dst = edge_index[_ONE]
        h_triple = torch.cat(
            [
                q_emb.expand(int(src.numel()), -1),
                node_feat.index_select(0, src),
                edge_rel_emb,
                node_feat.index_select(0, dst),
            ],
            dim=1,
        )
        return self.pred(h_triple).view(-1)


__all__ = [
    "DDE",
    "EdgeRetriever",
]
