from __future__ import annotations

import torch

from src.models.components import EmbeddingBackbone
from src.models.components.embedding import BackboneInput
from src.models.configs import BackboneConfig
from src.graph import build_graph_batch

from .conftest import make_toy_batch


def test_embedding_backbone_encode_matches_manual_zero_gnn_pipeline() -> None:
    topology, observation = build_graph_batch(make_toy_batch())
    backbone = EmbeddingBackbone(
        BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=0,
            gnn_dropout=0.0,
            use_adapter=False,
            adapter_dim=4,
            adapter_dropout=0.0,
        )
    )
    manual_node_tokens = backbone.project_node_embeddings(observation.node_features)
    manual_relation_tokens = backbone.project_relation_embeddings(
        observation.relation_features
    )
    manual_question_tokens = backbone.project_question_embeddings(
        observation.question_embedding
    )

    encoded = backbone.encode(
        BackboneInput(
            node_features=observation.node_features,
            relation_features=observation.relation_features,
            question_embedding=observation.question_embedding,
            edge_index=topology.edge_index,
            edge_relations=topology.edge_type,
            num_nodes=topology.num_nodes,
        )
    )

    assert torch.allclose(encoded.node_tokens, manual_node_tokens)
    assert torch.allclose(encoded.relation_tokens, manual_relation_tokens)
    assert torch.allclose(encoded.question_tokens, manual_question_tokens)


def test_embedding_backbone_forward_accepts_structured_inputs() -> None:
    topology, observation = build_graph_batch(make_toy_batch())
    backbone = EmbeddingBackbone(
        BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=1,
            gnn_dropout=0.0,
            use_adapter=False,
            adapter_dim=4,
            adapter_dropout=0.0,
        )
    )

    encoded = backbone(
        BackboneInput(
            node_features=observation.node_features,
            relation_features=observation.relation_features,
            question_embedding=observation.question_embedding,
            edge_index=topology.edge_index,
            edge_relations=topology.edge_type,
            num_nodes=topology.num_nodes,
        )
    )

    assert tuple(encoded.node_tokens.shape) == (topology.num_nodes, 8)
    assert tuple(encoded.relation_tokens.shape) == (2, 8)
    assert tuple(encoded.question_tokens.shape) == (topology.num_graphs, 8)
