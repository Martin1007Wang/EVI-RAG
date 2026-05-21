from __future__ import annotations

import pytest
import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.weaver.context import GraphContext
from src.weaver.rollout.replay import (
    ReplayBatch,
    ReplayBuilder,
    ReplaySource,
    ReplayTrajectory,
    build_replay_target_views,
    replay_trajectories,
)
from src.weaver.rollout.result import RolloutResult


def test_replay_source_marks_final_answer_transition_terminal() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)

    replay = ReplaySource(expand_budget=2).sample_from_rollouts(
        batch=batch,
        context=context,
        rollouts=(),
        num_trajectories=1,
    )

    assert replay is not None
    assert replay.num_trajectories == 1
    assert len(replay.trajectories) == 1
    assert replay.trajectories[0].graph_id == 0
    assert replay.trajectories[0].edge_ids == (0, 1)


def test_replay_focuses_graphs_not_hit_by_policy_rollouts() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)

    missed = replay_trajectories(
        batch=batch,
        context=context,
        rollouts=(_rollout(edge_ids=(0,), stop_step=1, budget=2),),
        budget=2,
        max_trajectories=1,
    )
    hit = replay_trajectories(
        batch=batch,
        context=context,
        rollouts=(_rollout(edge_ids=(0, 1), stop_step=2, budget=2),),
        budget=2,
        max_trajectories=1,
    )

    assert [trajectory.edge_ids for trajectory in missed] == [(0, 1)]
    assert hit == []


def test_replay_uses_precomputed_shortest_path_labels_without_bfs() -> None:
    batch = _batch_with_branching_shortest_paths()
    context = GraphContext.from_batch(batch)

    trajectories = replay_trajectories(
        batch=batch,
        context=context,
        rollouts=(),
        budget=2,
        max_trajectories=1,
    )

    assert [trajectory.edge_ids for trajectory in trajectories] == [(0, 2)]


def test_replay_prefers_nearest_anchor_under_precomputed_labels() -> None:
    batch = _batch_with_multiple_anchors()
    context = GraphContext.from_batch(batch)

    trajectories = replay_trajectories(
        batch=batch,
        context=context,
        rollouts=(),
        budget=2,
        max_trajectories=1,
    )

    assert [trajectory.edge_ids for trajectory in trajectories] == [(2,)]


def test_replay_target_views_select_distinct_rows_for_multiple_targets_in_one_graph() -> None:
    batch = _batch_with_multiple_targets()
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
    assert views[0].target_node_id == 2
    assert views[1].target_node_id == 3
    assert torch.equal(views[0].node_distances, torch.tensor([2, 1, 0, -1], dtype=torch.long))
    assert torch.equal(views[1].node_distances, torch.tensor([1, -1, -1, 0], dtype=torch.long))
    assert torch.equal(views[0].edge_counts, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32))
    assert torch.equal(views[1].edge_counts, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))


def test_replay_builder_rejects_trajectory_edges_outside_frontier() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)
    replay = ReplayBatch(
        trajectories=(
            ReplayTrajectory(
                graph_id=0,
                edge_ids=(1,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="current frontier"):
        ReplayBuilder(expand_budget=2).build(
            graph=context,
            trajectories=replay,
        )


def _batch():
    raw = {
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
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    data = _build_retrieval_data(
        raw=raw,
        sample_id="sample-0",
        question_emb=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _rollout(
    *,
    edge_ids: tuple[int, ...],
    stop_step: int,
    budget: int,
) -> RolloutResult:
    max_steps = int(budget) + 1
    selected_edge_ids = torch.full((1, max_steps), -1, dtype=torch.long)
    policy_action_log_prob = torch.zeros((1, max_steps), dtype=torch.float32)

    for step, edge_id in enumerate(edge_ids):
        selected_edge_ids[0, step] = int(edge_id)

    return RolloutResult(
        source_graph_id=torch.tensor([0], dtype=torch.long),
        selected_edge_ids=selected_edge_ids,
        policy_action_log_prob=policy_action_log_prob,
        behavior_action_log_prob=policy_action_log_prob.clone(),
        stop_step=torch.tensor([stop_step], dtype=torch.long),
        forced_stop=torch.tensor([False], dtype=torch.bool),
        expand_budget=budget,
    )


def _batch_with_branching_shortest_paths():
    raw = {
        SampleFields.EDGE_INDEX: torch.tensor(
            [[0, 0, 1, 2], [1, 2, 3, 3]],
            dtype=torch.long,
        ),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([10, 11, 12, 13], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([20, 21, 22, 23], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(4, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.tensor(4, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
        SampleFields.TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 1, 2], dtype=torch.long),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1, -1], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor(
            [2.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.long,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.tensor(
            [2.0, 1.0, 2.0, 1.0],
            dtype=torch.float32,
        ),
    }
    data = _build_retrieval_data(
        raw=raw,
        sample_id="sample-branching",
        question_emb=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _batch_with_multiple_anchors():
    raw = {
        SampleFields.EDGE_INDEX: torch.tensor(
            [[0, 1, 2], [1, 3, 3]],
            dtype=torch.long,
        ),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([10, 11, 12, 13], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([20, 21, 22], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(4, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.tensor(3, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: torch.tensor([0, 2], dtype=torch.long),
        SampleFields.TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 0, 1], dtype=torch.long),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, 0, -1], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor(
            [1.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
    }
    data = _build_retrieval_data(
        raw=raw,
        sample_id="sample-multi-anchor",
        question_emb=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _batch_with_multiple_targets():
    raw = {
        SampleFields.EDGE_INDEX: torch.tensor(
            [[0, 1, 0], [1, 2, 3]],
            dtype=torch.long,
        ),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([10, 11, 12, 13], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([20, 21, 22], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(4, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.tensor(3, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
        SampleFields.TARGET_NODE_IDS: torch.tensor([2, 3], dtype=torch.long),
        SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2, 3], dtype=torch.long),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 2, 1], dtype=torch.long),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1, -1], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCE: torch.tensor([1, 1, 0, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0, -1, 1, -1, -1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor(
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor(
            [0, 1, 4],
            dtype=torch.long,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
    }
    data = _build_retrieval_data(
        raw=raw,
        sample_id="sample-multi-target",
        question_emb=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])
