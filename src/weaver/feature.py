from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class FeatureBank:
    node_embedding: torch.Tensor  # [num_nodes_total, feature_dim]
    relation_embedding: torch.Tensor  # [num_edges_total, feature_dim]
    query_embedding: torch.Tensor  # [num_graphs_total, feature_dim]
    node_text_mask: torch.Tensor  # [num_nodes_total]


class FeatureEncoder(nn.Module):
    """
    Projection-free batch-local feature reader.

    Input contract:
    - entity_text_semantic_table is already a float tensor in BGE/PLM L2 space.
    - relation_semantic_table is already a float tensor in the same BGE/PLM L2 space.
    - batch.question_emb is already a float tensor in the same BGE/PLM L2 space.
    - text_row_by_entity_id is a long tensor.
    - text_row_by_entity_id[entity_id] < 0 means the entity has no text.
    - batch.node_entity_catalog_ids is a 1D long tensor.
    - batch.edge_relation_catalog_ids is a 1D long tensor.

    No learned projection is applied. Downstream hidden_dim must equal feature_dim.
    Nodes without text use one shared learned missing-node embedding for neural
    state encoding, while node_text_mask keeps semantic-prior scoring explicit.
    """

    entity_text_semantic_table: torch.Tensor
    text_row_by_entity_id: torch.Tensor
    relation_semantic_table: torch.Tensor

    def __init__(
        self,
        *,
        entity_text_semantic_table: torch.Tensor,
        text_row_by_entity_id: torch.Tensor,
        relation_semantic_table: torch.Tensor,
        non_text_node_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        feature_dim = int(entity_text_semantic_table.size(-1))
        self.register_buffer(
            "entity_text_semantic_table",
            entity_text_semantic_table,
        )
        self.register_buffer(
            "text_row_by_entity_id",
            text_row_by_entity_id,
        )
        self.register_buffer(
            "relation_semantic_table",
            relation_semantic_table,
        )
        self.feature_dim = feature_dim
        self.missing_node_embedding = nn.Parameter(torch.empty(self.feature_dim))
        self.reset_parameters(
            non_text_node_init_std=non_text_node_init_std,
        )

    def reset_parameters(
        self,
        *,
        non_text_node_init_std: float,
    ) -> None:
        nn.init.normal_(
            self.missing_node_embedding,
            mean=0.0,
            std=float(non_text_node_init_std),
        )

    def forward(self, batch: RetrievalBatch) -> FeatureBank:
        node_embedding, node_text_mask = self.read_node_embeddings(
            batch.node_entity_catalog_ids
        )
        relation_embedding = self.read_relation_embeddings(
            batch.edge_relation_catalog_ids
        )
        query_embedding = batch.question_emb
        if int(query_embedding.size(-1)) != self.feature_dim:
            raise ValueError(
                "batch.question_emb must have the same last dimension as the "
                f"feature tables, got {int(query_embedding.size(-1))} and "
                f"{self.feature_dim}."
            )

        return FeatureBank(
            node_embedding=node_embedding,
            relation_embedding=relation_embedding,
            query_embedding=query_embedding,
            node_text_mask=node_text_mask,
        )

    def read_node_embeddings(
        self,
        entity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_rows = self.text_row_by_entity_id.index_select(0, entity_ids)
        node_text_mask = text_rows.ge(0)
        node_embedding = (
            self.missing_node_embedding.unsqueeze(0)
            .expand(entity_ids.size(0), self.feature_dim)
            .clone()
        )

        text_node_ids = node_text_mask.nonzero(as_tuple=True)[0]
        if text_node_ids.numel() == 0:
            return node_embedding, node_text_mask

        valid_text_rows = text_rows.index_select(0, text_node_ids)
        valid_text_embedding = self.entity_text_semantic_table.index_select(
            0,
            valid_text_rows,
        )
        node_embedding[text_node_ids] = valid_text_embedding
        return node_embedding, node_text_mask

    def read_relation_embeddings(
        self,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.relation_semantic_table.index_select(0, relation_ids)


def select_node_embedding(
    features: FeatureBank,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    return features.node_embedding.index_select(0, node_ids)


def select_node_text_mask(
    features: FeatureBank,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    return features.node_text_mask.index_select(0, node_ids)


def select_relation_embedding(
    features: FeatureBank,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    return features.relation_embedding.index_select(0, edge_ids)


def select_query_embedding(
    features: FeatureBank,
    graph_ids: torch.Tensor,
) -> torch.Tensor:
    return features.query_embedding.index_select(0, graph_ids)


__all__ = [
    "FeatureBank",
    "FeatureEncoder",
    "select_node_embedding",
    "select_node_text_mask",
    "select_query_embedding",
    "select_relation_embedding",
]
