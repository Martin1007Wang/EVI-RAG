from __future__ import annotations

import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.preprocess.materialize import storage_record_from_sample
from src.data.preprocess.samples import PreparedSample
from src.data.schema.fields import SampleFields
from src.weaver.context import GraphContext
from src.weaver.rollout.replay import build_replay_target_views
from src.weaver.state import State


def _minimal_record() -> dict[str, torch.Tensor]:
    return {
        SampleFields.EDGE_INDEX: torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([10, 11, 12], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([20, 21], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.tensor(2, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
        SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
        SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 2], dtype=torch.long),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.tensor([1.0, 1.0], dtype=torch.float32),
    }


def _data(sample_id: str, question_emb: torch.Tensor):
    return _build_retrieval_data(
        raw=_minimal_record(),
        sample_id=sample_id,
        question_emb=question_emb,
    )


def _prepared_sample() -> PreparedSample:
    record = _minimal_record()
    return PreparedSample(
        dataset="unit",
        split="train",
        question_id="sample-0",
        question="question",
        edge_index=record[SampleFields.EDGE_INDEX],
        num_nodes=3,
        num_edges=2,
        question_entities=("q",),
        answer_entities=("a",),
        anchor_node_ids=record[SampleFields.ANCHOR_NODE_IDS],
        target_node_ids=record[SampleFields.TARGET_NODE_IDS],
        reachable_target_node_ids=record[SampleFields.REACHABLE_TARGET_NODE_IDS],
        node_entity_catalog_ids=record[SampleFields.NODE_ENTITY_CATALOG_IDS],
        edge_relation_catalog_ids=record[SampleFields.EDGE_RELATION_CATALOG_IDS],
        anchor_node_forward_distances_flat=record[
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT
        ],
        anchor_node_backward_distances_flat=record[
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT
        ],
        node_target_distance=record[SampleFields.NODE_TARGET_DISTANCE],
        node_target_distances_flat=record[SampleFields.NODE_TARGET_DISTANCES_FLAT],
        node_target_shortest_path_count_flat=torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float32
        ),
        node_target_shortest_path_edge_mask_flat=torch.tensor(
            [True, True], dtype=torch.bool
        ),
        node_target_shortest_path_edge_count_flat=torch.tensor(
            [1.0, 1.0], dtype=torch.float32
        ),
    )


def test_materialization_storage_record_writes_only_new_prior_names() -> None:
    record = storage_record_from_sample(_prepared_sample())

    assert SampleFields.NODE_TARGET_DISTANCES_FLAT in record
    assert SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT in record
    assert SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES in record
    assert SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES in record
    assert "target_node_distances_flat" not in record
    assert "target_shortest_path_edge_count_indices" not in record
    assert "target_shortest_path_edge_count_values" not in record


def test_dataset_restores_sparse_edge_counts_to_dense_runtime_fields() -> None:
    data = _data("sample-0", torch.tensor([0.25, 0.75], dtype=torch.float32))

    assert not hasattr(data, "target_node_distances_flat")
    assert not hasattr(data, "target_shortest_path_edge_mask_flat")
    assert torch.equal(
        data.node_target_shortest_path_edge_count_flat,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    assert torch.equal(
        data.node_target_shortest_path_edge_mask_flat,
        torch.tensor([True, True], dtype=torch.bool),
    )


def test_collate_stacks_questions_offsets_node_ids_and_attaches_edge_batch() -> None:
    batch = RetrievalCollator()(
        [
            _data("sample-0", torch.tensor([0.0, 1.0], dtype=torch.float32)),
            _data("sample-1", torch.tensor([2.0, 3.0], dtype=torch.float32)),
        ]
    )

    assert tuple(batch.question_emb.shape) == (2, 2)
    assert torch.equal(
        batch.question_emb,
        torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
    )
    assert torch.equal(batch.edge_batch, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    assert torch.equal(batch.anchor_node_ids, torch.tensor([0, 3], dtype=torch.long))
    assert torch.equal(batch.target_node_ids, torch.tensor([2, 5], dtype=torch.long))
    assert torch.equal(
        batch.reachable_target_node_ids,
        torch.tensor([2, 5], dtype=torch.long),
    )
    assert torch.equal(
        batch.reachable_target_node_ids_batch,
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        batch.reachable_target_node_ids_ptr,
        torch.tensor([0, 1, 2], dtype=torch.long),
    )


def test_state_initial_uses_core_graph_and_anchor_fields_only() -> None:
    data = _data("sample-0", torch.tensor([0.0, 1.0], dtype=torch.float32))
    del data.node_target_distances_flat
    del data.node_target_shortest_path_count_flat
    del data.node_target_shortest_path_edge_mask_flat
    del data.node_target_shortest_path_edge_count_flat

    batch = RetrievalCollator()([data])
    state = State.initial(
        graph=GraphContext.from_batch(batch),
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    assert tuple(state.active_node_mask.shape) == (1, 3)
    assert tuple(state.selected_edge_mask.shape) == (1, 2)
    assert torch.equal(state.active_node_mask, torch.tensor([[True, False, False]]))
    assert torch.equal(state.selected_edge_mask, torch.tensor([[False, False]]))


def test_state_initial_maps_multiple_same_graph_anchors_to_all_rollout_rows() -> None:
    data = _data("sample-0", torch.tensor([0.0, 1.0], dtype=torch.float32))
    data.anchor_node_ids = torch.tensor([0, 1], dtype=torch.long)
    batch = RetrievalCollator()([data])

    state = State.initial(
        graph=GraphContext.from_batch(batch),
        graph_ids=torch.zeros(3, dtype=torch.long),
    )

    expected = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.active_node_mask, expected)


def test_replay_target_views_derive_edge_ranges_from_edge_batch() -> None:
    batch = RetrievalCollator()([_data("sample-0", torch.tensor([0.0, 1.0], dtype=torch.float32))])
    assert not hasattr(batch, "edge_ptr")

    context = GraphContext.from_batch(batch)
    targets = batch.reachable_target_node_ids.to(dtype=torch.long)
    target_graph = context.node_to_graph.index_select(0, targets)

    views = build_replay_target_views(
        batch=batch,
        context=context,
        targets=targets,
        target_graph=target_graph,
    )

    assert len(views) == 1
    assert views[0].graph_id == 0
    assert views[0].edge_start == 0
    assert torch.equal(views[0].node_distances, torch.tensor([2, 1, 0], dtype=torch.long))
    assert torch.equal(views[0].edge_counts, torch.tensor([1.0, 1.0], dtype=torch.float32))


def test_replay_target_views_support_multi_graph_batches() -> None:
    batch = RetrievalCollator()(
        [
            _data("sample-0", torch.tensor([0.0, 1.0], dtype=torch.float32)),
            _data("sample-1", torch.tensor([2.0, 3.0], dtype=torch.float32)),
        ]
    )
    context = GraphContext.from_batch(batch)
    targets = batch.reachable_target_node_ids.to(dtype=torch.long)
    target_graph = context.node_to_graph.index_select(0, targets)

    views = build_replay_target_views(
        batch=batch,
        context=context,
        targets=targets,
        target_graph=target_graph,
    )

    assert len(views) == 2
    assert [view.graph_id for view in views] == [0, 1]
    assert [view.node_start for view in views] == [0, 3]
    assert all(
        torch.equal(view.node_distances, torch.tensor([2, 1, 0], dtype=torch.long))
        for view in views
    )
    assert all(
        torch.equal(view.edge_counts, torch.tensor([1.0, 1.0], dtype=torch.float32))
        for view in views
    )
