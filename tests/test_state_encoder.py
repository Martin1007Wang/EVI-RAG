from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.state import State


def test_state_encoder_ignores_active_nodes_without_selected_edges() -> None:
    hidden_dim = 2
    encoder = StateEncoder(hidden_dim=hidden_dim)

    graph = GraphContext(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )
    state = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    features_a = EncodedFeatures(
        node_text_semantic=torch.zeros((2, hidden_dim)),
        node_has_text=torch.tensor([True, True]),
        edge_relation_semantic=torch.zeros((1, hidden_dim)),
        query_semantic=torch.zeros((1, hidden_dim)),
        node_model=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_relation_model=torch.tensor([[5.0, 6.0]]),
        query_model=torch.tensor([[7.0, 8.0]]),
        edge_token_model=torch.tensor([[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]]),
    )
    features_b = EncodedFeatures(
        node_text_semantic=features_a.node_text_semantic,
        node_has_text=features_a.node_has_text,
        edge_relation_semantic=features_a.edge_relation_semantic,
        query_semantic=features_a.query_semantic,
        node_model=torch.tensor([[101.0, 202.0], [303.0, 404.0]]),
        edge_relation_model=features_a.edge_relation_model,
        query_model=features_a.query_model,
        edge_token_model=features_a.edge_token_model,
    )

    encoding_a = encoder(
        features=features_a,
        state=state,
        context=graph,
    )
    encoding_b = encoder(
        features=features_b,
        state=state,
        context=graph,
    )

    assert encoder.fuse[0].in_features == hidden_dim * 2
    assert torch.equal(encoding_a.edge_state_h, torch.zeros((1, hidden_dim)))
    assert torch.equal(encoding_b.edge_state_h, torch.zeros((1, hidden_dim)))
    assert torch.allclose(encoding_a.row_state_h, encoding_b.row_state_h)


def test_state_encoder_selected_edges_still_depend_on_endpoint_features() -> None:
    hidden_dim = 2
    encoder = StateEncoder(hidden_dim=hidden_dim)

    graph = GraphContext(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )
    state = State.from_selected_edges(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        selected_edge_mask=torch.tensor([[True]]),
    )

    features_a = EncodedFeatures(
        node_text_semantic=torch.zeros((2, hidden_dim)),
        node_has_text=torch.tensor([True, True]),
        edge_relation_semantic=torch.zeros((1, hidden_dim)),
        query_semantic=torch.zeros((1, hidden_dim)),
        node_model=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_relation_model=torch.tensor([[5.0, 6.0]]),
        query_model=torch.tensor([[7.0, 8.0]]),
        edge_token_model=torch.tensor([[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]]),
    )
    features_b = EncodedFeatures(
        node_text_semantic=features_a.node_text_semantic,
        node_has_text=features_a.node_has_text,
        edge_relation_semantic=features_a.edge_relation_semantic,
        query_semantic=features_a.query_semantic,
        node_model=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        edge_relation_model=features_a.edge_relation_model,
        query_model=features_a.query_model,
        edge_token_model=torch.tensor([[10.0, 20.0, 5.0, 6.0, 30.0, 40.0]]),
    )

    encoding_a = encoder(
        features=features_a,
        state=state,
        context=graph,
    )
    encoding_b = encoder(
        features=features_b,
        state=state,
        context=graph,
    )

    assert not torch.allclose(encoding_a.edge_state_h, encoding_b.edge_state_h)
    assert not torch.allclose(encoding_a.row_state_h, encoding_b.row_state_h)
