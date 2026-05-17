from __future__ import annotations

import pytest
import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.nn.frontier_encoder import FrontierEncoder
from src.weaver.state import (
    Frontier,
    FrontierBuilder,
    State,
    derive_node_mask,
)


def test_frontier_encoder_empty_frontier_returns_empty_feature_rows() -> None:
    features = _features(hidden_dim=3)
    state = _state()
    frontier = Frontier(
        row_ids=torch.empty(0, dtype=torch.long),
        edge_ids=torch.empty(0, dtype=torch.long),
    )
    batch = _batch()

    encoding = FrontierEncoder(hidden_dim=3)(
        context=GraphContext.from_batch(batch),
        features=features,
        state=state,
        frontier=frontier,
    )

    assert encoding.row_ids.shape == (0,)
    assert encoding.edge_ids.shape == (0,)
    assert encoding.edge_h.shape == (0, 3)
    assert encoding.query_h.shape == (0, 3)
    assert encoding.src_sem_h.shape == (0, 3)
    assert encoding.rel_sem_h.shape == (0, 3)
    assert encoding.dst_sem_h.shape == (0, 3)
    assert encoding.query_sem_h.shape == (0, 3)


def test_frontier_encoder_rejects_edge_hidden_dim_mismatch() -> None:
    features = _features(hidden_dim=3, edge_hidden_dim=4)

    with pytest.raises(ValueError, match="features.edge_h hidden dimension"):
        FrontierEncoder(hidden_dim=3)(
            context=GraphContext.from_batch(_batch()),
            features=features,
            state=_state(),
            frontier=_empty_frontier(),
        )


def test_frontier_encoder_rejects_query_hidden_dim_mismatch() -> None:
    features = _features(hidden_dim=3, query_hidden_dim=4)

    with pytest.raises(ValueError, match="features.query_h hidden dimension"):
        FrontierEncoder(hidden_dim=3)(
            context=GraphContext.from_batch(_batch()),
            features=features,
            state=_state(),
            frontier=_empty_frontier(),
        )


def test_frontier_encoder_materializes_semantic_tensors_from_batch_static_features() -> None:
    features = _features(hidden_dim=3)
    state = _state()
    frontier = Frontier(
        row_ids=torch.tensor([1, 0], dtype=torch.long),
        edge_ids=torch.tensor([1, 0], dtype=torch.long),
    )
    batch = _batch()

    encoding = FrontierEncoder(hidden_dim=3)(
        context=GraphContext.from_batch(batch),
        features=features,
        state=state,
        frontier=frontier,
    )

    assert torch.equal(encoding.src_sem_h, features.node_sem_h[[1, 0]])
    assert torch.equal(encoding.rel_sem_h, features.rel_sem_h[[1, 0]])
    assert torch.equal(encoding.dst_sem_h, features.node_sem_h[[2, 1]])
    assert torch.equal(encoding.query_sem_h, features.query_sem_h[[1, 0]])


def test_frontier_encoder_uses_compact_state_rows_for_query_lookup() -> None:
    features = _features(hidden_dim=3)
    full_state = State(
        node_mask=torch.zeros((3, 3), dtype=torch.bool),
        edge_mask=torch.zeros((3, 2), dtype=torch.bool),
        max_budget_by_row=torch.ones(3, dtype=torch.long),
        row_to_graph=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    compact_state = full_state.select_rows(torch.tensor([2, 1], dtype=torch.long))
    frontier = Frontier(
        row_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    batch = _batch()

    encoding = FrontierEncoder(hidden_dim=3)(
        context=GraphContext.from_batch(batch),
        features=features,
        state=compact_state,
        frontier=frontier,
    )

    assert torch.equal(compact_state.row_to_graph, torch.tensor([0, 1]))
    assert torch.equal(encoding.query_sem_h, features.query_sem_h[[0, 1]])


def test_frontier_builder_keeps_original_directed_edge_ids() -> None:
    batch = _direction_batch()
    builder = FrontierBuilder.from_batch(batch)
    state = State.initial(batch, budget=2)

    root_frontier = builder.build(state)
    assert set(root_frontier.edge_ids.tolist()) == {0, 2}

    state.apply_edges_(
        edge_index=batch.edge_index,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
    )
    frontier = builder.build(state)
    edge_ids = set(frontier.edge_ids.tolist())
    assert 1 in edge_ids
    assert 2 not in edge_ids
    assert not builder.contains(state=state, row=0, edge_id=2)


def test_frontier_builder_from_graph_context_matches_batch_path() -> None:
    batch = _direction_batch()
    graph_context = GraphContext.from_batch(batch)
    from_batch = FrontierBuilder.from_batch(batch)
    from_context = FrontierBuilder.from_graph_context(graph_context)
    state = State.initial_from_graph_context(graph_context, budget=2)

    frontier_from_batch = from_batch.build(state)
    frontier_from_context = from_context.build(state)

    assert torch.equal(frontier_from_batch.row_ids, frontier_from_context.row_ids)
    assert torch.equal(frontier_from_batch.edge_ids, frontier_from_context.edge_ids)


def test_derive_node_mask_uses_graph_context() -> None:
    batch = _direction_batch()
    graph_context = GraphContext.from_batch(batch)
    state = State.initial_from_graph_context(graph_context, budget=3)
    state.apply_edges_(
        edge_index=graph_context.edge_index,
        rows=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    derived = derive_node_mask(
        state=state,
        graph_context=graph_context,
    )

    assert torch.equal(derived, state.node_mask)


def test_frontier_builder_from_graph_context_shares_context() -> None:
    graph_context = GraphContext.from_batch(_direction_batch())
    builder = FrontierBuilder.from_graph_context(graph_context)

    assert builder.graph_context is graph_context


def test_frontier_encoding_num_edges_warns_and_matches_num_actions() -> None:
    encoding = FrontierEncoder(hidden_dim=3)(
        context=GraphContext.from_batch(_batch()),
        features=_features(hidden_dim=3),
        state=_state(),
        frontier=Frontier(
            row_ids=torch.tensor([0, 1], dtype=torch.long),
            edge_ids=torch.tensor([0, 1], dtype=torch.long),
        ),
    )

    with pytest.deprecated_call(
        match="FrontierEncoding.num_edges is deprecated; use num_actions."
    ):
        assert encoding.num_edges == encoding.num_actions


def _features(
    *,
    hidden_dim: int,
    edge_hidden_dim: int | None = None,
    query_hidden_dim: int | None = None,
) -> FeatureBank:
    edge_hidden_dim = hidden_dim if edge_hidden_dim is None else edge_hidden_dim
    query_hidden_dim = hidden_dim if query_hidden_dim is None else query_hidden_dim

    return FeatureBank(
        node_h=torch.zeros((3, hidden_dim), dtype=torch.float32),
        edge_h=torch.zeros((2, edge_hidden_dim), dtype=torch.float32),
        query_h=torch.zeros((2, query_hidden_dim), dtype=torch.float32),
        node_is_non_text=torch.zeros(3, dtype=torch.bool),
        node_sem_h=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        rel_sem_h=torch.tensor(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
            ],
            dtype=torch.float32,
        ),
        query_sem_h=torch.tensor(
            [
                [0.8, 0.1, 0.2],
                [0.3, 0.9, 0.4],
            ],
            dtype=torch.float32,
        ),
        rel_h=torch.zeros((2, hidden_dim), dtype=torch.float32),
    )


def _state() -> State:
    return State(
        node_mask=torch.zeros((2, 3), dtype=torch.bool),
        edge_mask=torch.zeros((2, 2), dtype=torch.bool),
        max_budget_by_row=torch.ones(2, dtype=torch.long),
        row_to_graph=torch.tensor([0, 1], dtype=torch.long),
    )


def _empty_frontier() -> Frontier:
    return Frontier(
        row_ids=torch.empty(0, dtype=torch.long),
        edge_ids=torch.empty(0, dtype=torch.long),
    )


def _batch() -> object:
    data = _build_retrieval_data(
        raw={
            SampleFields.EDGE_INDEX: torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 1], dtype=torch.long),
            SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.tensor(2, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
            SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.ones(3, dtype=torch.float32),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(2, dtype=torch.float32),
        },
        sample_id="frontier",
        question_emb=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _direction_batch() -> object:
    data = _build_retrieval_data(
        raw={
            SampleFields.EDGE_INDEX: torch.tensor(
                [[0, 2, 1], [1, 1, 0]],
                dtype=torch.long,
            ),
            SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 1, 1], dtype=torch.long),
            SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.tensor(3, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
            SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, -1], dtype=torch.long),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, 1, -1], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.ones(3, dtype=torch.float32),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(1, dtype=torch.float32),
        },
        sample_id="frontier-direction",
        question_emb=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])
