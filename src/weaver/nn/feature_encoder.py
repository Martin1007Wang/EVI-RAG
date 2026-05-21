from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import init_xavier


@dataclass(frozen=True, slots=True)
class EncodedFeatures:
    """
    Batch-local features produced by FeatureEncoder.

    Contract:
    - *_semantic tensors live in the upstream PLM L2 semantic space.
    - *_model tensors live in the learned Weaver model space.
    - Non-text nodes have zero node_text_semantic and node_has_text=False.
    - Non-text nodes get a learnable model-space token in node_model.
    """

    node_text_semantic: torch.Tensor  # [num_nodes_total, semantic_dim]
    node_has_text: torch.Tensor  # [num_nodes_total]

    edge_relation_semantic: torch.Tensor  # [num_edges_total, semantic_dim]
    query_semantic: torch.Tensor  # [num_graphs_total, semantic_dim]

    node_model: torch.Tensor  # [num_nodes_total, model_dim]
    edge_relation_model: torch.Tensor  # [num_edges_total, model_dim]
    query_model: torch.Tensor  # [num_graphs_total, model_dim]


class FeatureEncoder(nn.Module):
    """
    Build batch-local semantic and model-space features.

    Input contract:
    - entity_text_semantic_table is already float tensor in PLM L2 space.
    - relation_semantic_table is already float tensor in PLM L2 space.
    - batch.question_emb is already float tensor in the same PLM L2 space.
    - text_row_by_entity_id is a long tensor.
    - text_row_by_entity_id[entity_id] < 0 means the entity has no text.
    - batch.node_entity_catalog_ids is a 1D long tensor.
    - batch.edge_relation_catalog_ids is a 1D long tensor.
    - all tensors are already on the intended device.
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
        model_dim: int = 1024,
        non_text_node_init_std: float = 0.02,
    ) -> None:
        super().__init__()

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

        self.semantic_dim = int(entity_text_semantic_table.size(-1))
        self.model_dim = int(model_dim)

        self.project_to_model = nn.Linear(
            self.semantic_dim,
            self.model_dim,
            bias=False,
        )
        self.model_norm = nn.LayerNorm(self.model_dim)

        self.non_text_node_model = nn.Parameter(torch.empty(self.model_dim))

        self.reset_parameters(
            non_text_node_init_std=non_text_node_init_std,
        )

    def reset_parameters(
        self,
        *,
        non_text_node_init_std: float,
    ) -> None:
        init_xavier(self.project_to_model)
        nn.init.normal_(
            self.non_text_node_model,
            mean=0.0,
            std=float(non_text_node_init_std),
        )

    def forward(self, batch: RetrievalBatch) -> EncodedFeatures:
        node_text_semantic, node_has_text = self.encode_node_text_semantic(batch.node_entity_catalog_ids)

        edge_relation_semantic = self.encode_edge_relation_semantic(batch.edge_relation_catalog_ids)

        query_semantic = batch.question_emb

        return EncodedFeatures(
            node_text_semantic=node_text_semantic,
            node_has_text=node_has_text,
            edge_relation_semantic=edge_relation_semantic,
            query_semantic=query_semantic,
            node_model=self.encode_node_model(
                node_text_semantic=node_text_semantic,
                node_has_text=node_has_text,
            ),
            edge_relation_model=self.to_model_space(edge_relation_semantic),
            query_model=self.to_model_space(query_semantic),
        )

    def encode_node_text_semantic(
        self,
        entity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_rows = self.text_row_by_entity_id.index_select(0, entity_ids)
        node_has_text = text_rows.ge(0)

        node_text_semantic = self.entity_text_semantic_table.new_zeros(
            entity_ids.size(0),
            self.semantic_dim,
        )

        text_node_ids = node_has_text.nonzero(as_tuple=True)[0]
        if text_node_ids.numel() == 0:
            return node_text_semantic, node_has_text

        valid_text_rows = text_rows.index_select(0, text_node_ids)
        valid_text_semantic = self.entity_text_semantic_table.index_select(
            0,
            valid_text_rows,
        )

        node_text_semantic[text_node_ids] = valid_text_semantic
        return node_text_semantic, node_has_text

    def encode_edge_relation_semantic(
        self,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.relation_semantic_table.index_select(0, relation_ids)

    def encode_node_model(
        self,
        *,
        node_text_semantic: torch.Tensor,
        node_has_text: torch.Tensor,
    ) -> torch.Tensor:
        text_node_model = self.to_model_space(node_text_semantic)

        non_text_node_model = self.non_text_node_model.unsqueeze(0).expand(
            node_text_semantic.size(0),
            self.model_dim,
        )

        return torch.where(
            node_has_text.unsqueeze(-1),
            text_node_model,
            non_text_node_model,
        )

    def to_model_space(
        self,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        return self.model_norm(self.project_to_model(semantic))


def select_node_text_semantic(
    features: EncodedFeatures,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    return features.node_text_semantic.index_select(0, node_ids)


def select_node_has_text(
    features: EncodedFeatures,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    return features.node_has_text.index_select(0, node_ids)


def select_edge_relation_semantic(
    features: EncodedFeatures,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    return features.edge_relation_semantic.index_select(0, edge_ids)


def select_query_semantic(
    features: EncodedFeatures,
    graph_ids: torch.Tensor,
) -> torch.Tensor:
    return features.query_semantic.index_select(0, graph_ids)


def select_node_model(
    features: EncodedFeatures,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    return features.node_model.index_select(0, node_ids)


def select_edge_relation_model(
    features: EncodedFeatures,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    return features.edge_relation_model.index_select(0, edge_ids)


def select_query_model(
    features: EncodedFeatures,
    graph_ids: torch.Tensor,
) -> torch.Tensor:
    return features.query_model.index_select(0, graph_ids)


__all__ = [
    "EncodedFeatures",
    "FeatureEncoder",
    "select_edge_relation_model",
    "select_edge_relation_semantic",
    "select_node_has_text",
    "select_node_model",
    "select_node_text_semantic",
    "select_query_model",
    "select_query_semantic",
]
