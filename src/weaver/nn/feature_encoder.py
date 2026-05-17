from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import init_xavier

from .edge_encoder import EdgeEncoder


@dataclass(frozen=True, slots=True)
class FeatureBank:
    """
    Batch-static encoded features.

    Naming convention:
        *_h: model-space hidden states, H-dimensional and task-specific.
        *_sem_h: PLM-derived semantic embeddings, D-dimensional and L2-normalized.

    rel_sem_h is edge-aligned [E, D]: rel_sem_h[i] is the raw relation
    semantic embedding for batch edge i, not a relation-catalog table.
    """

    node_h: torch.Tensor
    edge_h: torch.Tensor
    query_h: torch.Tensor
    node_is_non_text: torch.Tensor

    node_sem_h: torch.Tensor
    rel_sem_h: torch.Tensor
    query_sem_h: torch.Tensor
    rel_h: torch.Tensor


class EntityEmbeddingLayer(nn.Module):
    entity_text_embeddings: torch.Tensor
    entity_embedding_map: torch.Tensor

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        non_text_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        text = _l2_normalize(entity_text_embeddings.float().contiguous())
        mapping = entity_embedding_map.long().contiguous()

        self.register_buffer("entity_text_embeddings", text)
        self.register_buffer("entity_embedding_map", mapping)

        self.non_text_embedding = nn.Parameter(
            torch.randn(text.size(-1), dtype=torch.float32) * non_text_init_std
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.entity_text_embeddings.size(-1))

    def forward(self, entity_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        entity_ids = entity_ids.to(
            device=self.entity_embedding_map.device,
            dtype=torch.long,
        ).view(-1)

        text_rows = self.entity_embedding_map.index_select(0, entity_ids)
        has_text = text_rows.ge(0)

        text_h = self.entity_text_embeddings.index_select(
            0,
            text_rows.clamp_min(0),
        )

        non_text_h = _l2_normalize(self.non_text_embedding).view(1, -1)

        entity_h = torch.where(
            has_text.unsqueeze(-1),
            text_h,
            non_text_h.expand_as(text_h),
        )

        return _l2_normalize(entity_h), has_text.logical_not()


class RoleProjection(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

        if input_dim == hidden_dim:
            nn.init.eye_(self.proj.weight)
        else:
            init_xavier(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class FeatureEncoder(nn.Module):
    relation_embeddings: torch.Tensor

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        relation_embeddings: torch.Tensor,
        hidden_dim: int = 1024,
        non_text_init_std: float = 0.02,
        learn_edge_role_weights: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.entity_embedding = EntityEmbeddingLayer(
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            non_text_init_std=non_text_init_std,
        )

        self.embedding_dim = self.entity_embedding.embedding_dim

        self.register_buffer(
            "relation_embeddings",
            _l2_normalize(relation_embeddings.float().contiguous()),
        )

        self.node_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )
        self.rel_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )
        self.query_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

        self.edge_encoder = EdgeEncoder(
            hidden_dim=self.hidden_dim,
            learn_role_weights=learn_edge_role_weights,
        )

    def forward(self, batch: RetrievalBatch) -> FeatureBank:
        device = self.relation_embeddings.device

        edge_index = batch.edge_index.to(device=device, dtype=torch.long)

        node_entity_ids = batch.node_entity_catalog_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)

        rel_ids = batch.edge_relation_catalog_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)

        node_sem_h, node_is_non_text = self.entity_embedding(node_entity_ids)
        rel_sem_h = self.relation_embeddings.index_select(0, rel_ids)
        query_sem_h = self._query_semantic_embedding(batch=batch, device=device)

        node_h = self.node_projection(node_sem_h)
        rel_h = self.rel_projection(rel_sem_h)
        query_h = self.query_projection(query_sem_h)

        edge_h = self.edge_encoder(
            src_h=node_h.index_select(0, edge_index[0]),
            rel_h=rel_h,
            dst_h=node_h.index_select(0, edge_index[1]),
        )

        return FeatureBank(
            node_h=node_h,
            edge_h=edge_h,
            query_h=query_h,
            node_is_non_text=node_is_non_text,
            node_sem_h=node_sem_h,
            rel_sem_h=rel_sem_h,
            query_sem_h=query_sem_h,
            rel_h=rel_h,
        )

    def _query_semantic_embedding(
        self,
        *,
        batch: RetrievalBatch,
        device: torch.device,
    ) -> torch.Tensor:
        query = batch.question_emb.to(device=device, dtype=torch.float32)

        if query.ndim == 1:
            query = query.view(1, -1)

        return _l2_normalize(query)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-12)


__all__ = [
    "EntityEmbeddingLayer",
    "FeatureBank",
    "FeatureEncoder",
    "RoleProjection",
]
