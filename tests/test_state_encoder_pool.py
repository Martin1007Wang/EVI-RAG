from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.nn.state_encoder import SegmentTokenPool, StateEncoder
from src.weaver.state import State


def test_segment_token_pool_mean_pools_by_row() -> None:
    pool = SegmentTokenPool(input_dim=1, output_dim=1)
    pool.token_proj = torch.nn.Identity()

    out = pool(
        tokens=torch.tensor([[2.0], [4.0], [10.0]]),
        row_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        num_rows=3,
    )

    assert torch.allclose(
        out,
        torch.tensor([[3.0], [10.0], [0.0]]),
    )


def test_state_encoder_uses_state_num_rows() -> None:
    hidden_dim = 4
    context = _graph_context()
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
    )
    features = EncodedFeatures(
        node_text_semantic=torch.zeros((3, hidden_dim), dtype=torch.float32),
        node_has_text=torch.zeros(3, dtype=torch.bool),
        edge_relation_semantic=torch.zeros((2, hidden_dim), dtype=torch.float32),
        query_semantic=torch.zeros((1, hidden_dim), dtype=torch.float32),
        node_model=torch.randn((3, hidden_dim), dtype=torch.float32),
        edge_relation_model=torch.randn((2, hidden_dim), dtype=torch.float32),
        query_model=torch.randn((1, hidden_dim), dtype=torch.float32),
    )

    encoding = StateEncoder(hidden_dim=hidden_dim)(
        features=features,
        state=state,
        context=context,
    )

    assert encoding.query_h.shape == (state.num_rows, hidden_dim)
    assert encoding.node_state_h.shape == (state.num_rows, hidden_dim)
    assert encoding.edge_state_h.shape == (state.num_rows, hidden_dim)
    assert encoding.row_state_h.shape == (state.num_rows, hidden_dim)


def _graph_context() -> GraphContext:
    edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.zeros(3, dtype=torch.long),
        edge_to_graph=torch.zeros(2, dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False], dtype=torch.bool),
        adjacency=build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=3,
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )
