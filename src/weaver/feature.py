from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class FeaturePack:
    query_sem_h: torch.Tensor  # [G, D]
    node_sem_h: torch.Tensor  # [N, D]
    rel_sem_h: torch.Tensor  # [E, D]
    query_h: torch.Tensor  # [G, H]
    node_h: torch.Tensor  # [N, H]
    node_has_text: torch.Tensor  # [N]
    node_graph_ids: torch.Tensor  # [N]
    anchor_node_ids: torch.Tensor  # [A]
    anchor_graph_ids: torch.Tensor  # [A]
    edge_h: torch.Tensor  # [E, H]
    edge_src: torch.Tensor  # [E]
    edge_dst: torch.Tensor  # [E]
    edge_graph_ids: torch.Tensor  # [E]
    device: torch.device


class FeatureEncoder(nn.Module):
    entity_text_semantic_table: torch.Tensor
    text_row_by_entity_id: torch.Tensor
    relation_semantic_table: torch.Tensor

    def __init__(
        self,
        *,
        entity_text_semantic_table: torch.Tensor,
        text_row_by_entity_id: torch.Tensor,
        relation_semantic_table: torch.Tensor,
        hidden_dim: int | None = None,
        non_text_node_init_std: float = 0.02,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()

        dim = int(entity_text_semantic_table.size(-1))
        hidden_dim = dim if hidden_dim is None else int(hidden_dim)

        if normalize_inputs:
            entity_text_semantic_table = F.normalize(
                entity_text_semantic_table.float(),
                dim=-1,
            )
            relation_semantic_table = F.normalize(
                relation_semantic_table.float(),
                dim=-1,
            )
        else:
            entity_text_semantic_table = entity_text_semantic_table.float()
            relation_semantic_table = relation_semantic_table.float()

        self.register_buffer(
            "entity_text_semantic_table",
            entity_text_semantic_table,
            persistent=False,
        )
        self.register_buffer(
            "text_row_by_entity_id",
            text_row_by_entity_id.long(),
            persistent=False,
        )
        self.register_buffer(
            "relation_semantic_table",
            relation_semantic_table,
            persistent=False,
        )

        self.non_text_node_h = nn.Parameter(
            torch.randn(dim) * float(non_text_node_init_std)
        )
        self.query_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.node_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.src_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.rel_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    @property
    def dim(self) -> int:
        return int(self.entity_text_semantic_table.size(-1))

    @property
    def hidden_dim(self) -> int:
        return int(self.query_proj.out_features)

    def forward(self, batch: RetrievalBatch) -> FeaturePack:
        edge_src = batch.edge_index[0].long()
        edge_dst = batch.edge_index[1].long()

        node_graph_ids = batch.batch.long()
        edge_graph_ids = node_graph_ids.index_select(0, edge_src)
        anchor_node_ids = batch.anchor_node_ids.to(device=edge_src.device, dtype=torch.long)
        anchor_graph_ids = node_graph_ids.index_select(0, anchor_node_ids) if int(anchor_node_ids.numel()) > 0 else torch.empty(0, dtype=torch.long, device=edge_src.device)

        query_sem_h = batch.question_emb.float()
        if int(query_sem_h.size(-1)) != self.dim:
            raise ValueError(
                "batch.question_emb must have the same last dimension as the "
                f"feature tables, got {int(query_sem_h.size(-1))} and {self.dim}."
            )

        node_entity_ids = batch.node_entity_catalog_ids.long()
        text_rows = self.text_row_by_entity_id.index_select(0, node_entity_ids)
        node_has_text = text_rows.ge(0)

        node_sem_h = self.non_text_node_h.view(1, -1).expand(
            node_entity_ids.size(0),
            -1,
        ).clone()

        if bool(node_has_text.any()):
            node_sem_h[node_has_text] = self.entity_text_semantic_table.index_select(
                0,
                text_rows[node_has_text],
            )

        rel_ids = batch.edge_relation_catalog_ids.long()
        rel_sem_h = self.relation_semantic_table.index_select(0, rel_ids)
        query_h = self.query_proj(query_sem_h)
        node_h = self.node_proj(node_sem_h)
        edge_h = self.edge_norm(
            query_h.index_select(0, edge_graph_ids)
            + self.src_proj(node_sem_h).index_select(0, edge_src)
            + self.rel_proj(rel_sem_h)
            + self.dst_proj(node_sem_h).index_select(0, edge_dst)
        )

        return FeaturePack(
            query_sem_h=query_sem_h,
            node_sem_h=node_sem_h,
            rel_sem_h=rel_sem_h,
            query_h=query_h,
            node_h=node_h,
            node_has_text=node_has_text,
            node_graph_ids=node_graph_ids,
            anchor_node_ids=anchor_node_ids,
            anchor_graph_ids=anchor_graph_ids,
            edge_h=edge_h,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_graph_ids=edge_graph_ids,
            device=edge_src.device,
        )


FeatureBank = FeaturePack


__all__ = [
    "FeatureBank",
    "FeatureEncoder",
    "FeaturePack",
]
