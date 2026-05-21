from __future__ import annotations

import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.weaver.context import GraphContext
from src.weaver.state import State


def test_frontier_keeps_original_directed_edge_ids() -> None:
    batch = _direction_batch()
    context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    root_frontier = state.frontier(context)
    assert set(root_frontier.edge_ids.tolist()) == {0}

    state = state.expand(
        graph=context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    )
    frontier = state.frontier(context)
    edge_ids = set(frontier.edge_ids.tolist())
    assert 2 in edge_ids
    assert 1 in edge_ids
    assert not _frontier_contains(frontier, row=0, edge_id=0)
    assert _frontier_contains(frontier, row=0, edge_id=2)


def test_frontier_only_includes_outgoing_edges_from_active_source_nodes() -> None:
    batch = _direction_batch()
    context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    root_frontier = state.frontier(context)
    assert set(root_frontier.edge_ids.tolist()) == {0}
    assert not _frontier_contains(root_frontier, row=0, edge_id=1)

    state = state.expand(
        graph=context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    )
    frontier = state.frontier(context)
    assert set(frontier.edge_ids.tolist()) == {1, 2}
    assert not _frontier_contains(frontier, row=0, edge_id=0)


def test_frontier_from_graph_context_is_deterministic() -> None:
    batch = _direction_batch()
    graph_context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=graph_context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    first = state.frontier(graph_context)
    second = state.frontier(graph_context)

    assert torch.equal(first.row_ids, second.row_ids)
    assert torch.equal(first.edge_ids, second.edge_ids)


def test_apply_edges_updates_node_mask_with_graph_context() -> None:
    batch = _direction_batch()
    graph_context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=graph_context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )
    state = state.expand(
        graph=graph_context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=3,
    )
    state = state.expand(
        graph=graph_context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([2], dtype=torch.long),
        expand_budget=3,
    )

    assert torch.equal(
        state.active_node_mask,
        torch.tensor([[True, True, True]], dtype=torch.bool),
    )


def _frontier_contains(frontier, *, row: int, edge_id: int) -> bool:
    return bool(
        (frontier.row_ids.eq(int(row)) & frontier.edge_ids.eq(int(edge_id))).any()
    )


def _direction_batch() -> object:
    data = _build_retrieval_data(
        raw={
            SampleFields.EDGE_INDEX: torch.tensor(
                [[0, 1, 1], [1, 0, 2]],
                dtype=torch.long,
            ),
            SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.tensor(3, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
            SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.ones(3, dtype=torch.float32),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 2], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(2, dtype=torch.float32),
        },
        sample_id="direction",
        question_emb=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])
