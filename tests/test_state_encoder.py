from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.state import State


def tiny_graph() -> GraphContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=2,
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )


def make_features(
    *,
    query_model: torch.Tensor,
    node_model: torch.Tensor,
    edge_relation_model: torch.Tensor,
) -> EncodedFeatures:
    hidden_dim = int(query_model.size(-1))
    return EncodedFeatures(
        node_text_semantic=torch.zeros((node_model.size(0), hidden_dim)),
        node_has_text=torch.tensor([True] * node_model.size(0)),
        edge_relation_semantic=torch.zeros((edge_relation_model.size(0), hidden_dim)),
        query_semantic=torch.zeros((query_model.size(0), hidden_dim)),
        node_model=node_model,
        edge_relation_model=edge_relation_model,
        query_model=query_model,
        edge_token_model=torch.empty((edge_relation_model.size(0), hidden_dim * 3)),
    )


def test_state_encoder_root_state_depends_on_anchor_tokens() -> None:
    hidden_dim = 4
    encoder = StateEncoder(hidden_dim=hidden_dim)
    graph = tiny_graph()
    state = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    features_a = make_features(
        query_model=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        node_model=torch.tensor(
            [
                [2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        ),
        edge_relation_model=torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
    )
    features_b = make_features(
        query_model=features_a.query_model,
        node_model=torch.tensor(
            [
                [20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        ),
        edge_relation_model=features_a.edge_relation_model,
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

    assert not torch.allclose(encoding_a.row_state_h, encoding_b.row_state_h)
    assert torch.allclose(encoding_a.edge_state_h, encoding_a.row_state_h)


def test_state_encoder_edge_tokens_change_with_query() -> None:
    hidden_dim = 4
    encoder = StateEncoder(hidden_dim=hidden_dim)
    graph = tiny_graph()
    query_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    query_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    features = make_features(
        query_model=query_a,
        node_model=torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=torch.float32,
        ),
        edge_relation_model=torch.tensor([[0.5, 1.5, 2.5, 3.5]], dtype=torch.float32),
    )
    edge_ids = torch.tensor([0], dtype=torch.long)
    src_node_ids = graph.edge_index[0].index_select(0, edge_ids)
    dst_node_ids = graph.edge_index[1].index_select(0, edge_ids)

    edge_h_a = encoder.encode_edge_tokens(
        features=features,
        src_node_ids=src_node_ids,
        edge_ids=edge_ids,
        dst_node_ids=dst_node_ids,
        query_h=query_a,
    )
    edge_h_b = encoder.encode_edge_tokens(
        features=features,
        src_node_ids=src_node_ids,
        edge_ids=edge_ids,
        dst_node_ids=dst_node_ids,
        query_h=query_b,
    )

    assert not torch.allclose(edge_h_a, edge_h_b)


def test_state_encoder_edge_tokens_preserve_src_dst_roles() -> None:
    hidden_dim = 4
    encoder = StateEncoder(hidden_dim=hidden_dim)
    query = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    features = make_features(
        query_model=query,
        node_model=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_relation_model=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    edge_ids = torch.tensor([0], dtype=torch.long)

    edge_h_src_dst = encoder.encode_edge_tokens(
        features=features,
        src_node_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=edge_ids,
        dst_node_ids=torch.tensor([1], dtype=torch.long),
        query_h=query,
    )
    edge_h_dst_src = encoder.encode_edge_tokens(
        features=features,
        src_node_ids=torch.tensor([1], dtype=torch.long),
        edge_ids=edge_ids,
        dst_node_ids=torch.tensor([0], dtype=torch.long),
        query_h=query,
    )

    assert not torch.allclose(edge_h_src_dst, edge_h_dst_src)
