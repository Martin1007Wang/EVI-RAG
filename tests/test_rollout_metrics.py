from __future__ import annotations

import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.eval.rollout import evaluate_rollout_samples
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor


def _record_two_paths() -> dict[str, torch.Tensor]:
    return {
        SampleFields.EDGE_INDEX: torch.tensor(
            [[0, 1, 0, 2], [1, 3, 2, 3]],
            dtype=torch.long,
        ),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2, 3], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 0, 0, 0], dtype=torch.long),
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
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1, 2, 3], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(4, dtype=torch.float32),
    }


def _batch() -> object:
    data = _build_retrieval_data(
        raw=_record_two_paths(),
        sample_id="two-paths",
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _rollout_result(
    *,
    source_graph_id: list[int],
    expand_steps: dict[int, dict[int, int]],
    max_steps: int,
    stop_steps: dict[int, int] | None = None,
    forced_stop_steps: dict[int, int] | None = None,
    policy_log_probs: dict[int, dict[int, float]] | None = None,
) -> RolloutResult:
    num_rows = len(source_graph_id)
    stop_steps = stop_steps or {}
    forced_stop_steps = forced_stop_steps or {}
    policy_log_probs = policy_log_probs or {}

    selected_edge_ids = torch.full((num_rows, max_steps), -1, dtype=torch.long)
    policy_action_log_prob = torch.zeros((num_rows, max_steps), dtype=torch.float32)
    stop_step = torch.zeros(num_rows, dtype=torch.long)
    forced_stop = torch.zeros(num_rows, dtype=torch.bool)

    for row in range(num_rows):
        last_step = -1
        for step, edge_id in expand_steps.get(row, {}).items():
            selected_edge_ids[row, step] = int(edge_id)
            policy_action_log_prob[row, step] = float(policy_log_probs.get(row, {}).get(step, 0.0))
            last_step = max(last_step, int(step))
        if row in stop_steps:
            step = int(stop_steps[row])
            policy_action_log_prob[row, step] = float(policy_log_probs.get(row, {}).get(step, 0.0))
            last_step = max(last_step, step)
            stop_step[row] = step
        elif row in forced_stop_steps:
            step = int(forced_stop_steps[row])
            policy_action_log_prob[row, step] = float(policy_log_probs.get(row, {}).get(step, 0.0))
            last_step = max(last_step, step)
            stop_step[row] = step
            forced_stop[row] = True
        else:
            stop_step[row] = max(last_step, 0)

    return RolloutResult(
        source_graph_id=torch.tensor(source_graph_id, dtype=torch.long),
        selected_edge_ids=selected_edge_ids,
        policy_action_log_prob=policy_action_log_prob,
        behavior_action_log_prob=policy_action_log_prob.clone(),
        stop_step=stop_step,
        forced_stop=forced_stop,
        expand_budget=max_steps - 1,
    )


def test_evaluate_rollout_samples_emits_minimal_validation_dashboard() -> None:
    batch = _batch()
    rollouts = (
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0, 1: 1}},
            stop_steps={0: 2},
            max_steps=3,
        ),
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 2}},
            forced_stop_steps={0: 1},
            max_steps=3,
        ),
    )

    metrics = evaluate_rollout_samples(
        rollout_samples=rollouts,
        batch=batch,
        best_of_k=8,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )

    assert metrics["best_of_k_target_recall@1"] == metrics["oracle_best@1/target_recall"]
    assert metrics["best_of_k_target_recall@2"] == metrics["oracle_best@2/target_recall"]
    assert metrics["best_of_k_target_recall@2"] >= metrics["sample@1/target_recall"]
    assert metrics["best@2/target_recall"] == metrics["oracle_best@2/target_recall"]
    assert "best@2/effective_reward" in metrics
    assert torch.isclose(
        torch.tensor(metrics["reward/mean_log_reward_of_stopped"]),
        torch.tensor(-0.1),
    )
    assert "mean_edges" in metrics
    assert "expected_target_recall" in metrics


def test_best_of_k_target_recall_uses_available_rollouts_when_k_is_larger() -> None:
    batch = _batch()
    rollouts = (
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0}},
            forced_stop_steps={0: 1},
            max_steps=3,
        ),
    )

    metrics = evaluate_rollout_samples(
        rollout_samples=rollouts,
        batch=batch,
        best_of_k=8,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )

    assert metrics["best_of_k_target_recall@1"] == 0.0


def test_oracle_model_and_union_best_of_k_have_distinct_semantics() -> None:
    batch = _batch()
    rollouts = (
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0}},
            stop_steps={0: 1},
            max_steps=3,
            policy_log_probs={0: {0: 0.0, 1: 0.0}},
        ),
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0, 1: 1}},
            stop_steps={0: 2},
            max_steps=3,
            policy_log_probs={0: {0: -5.0, 1: -5.0, 2: -5.0}},
        ),
    )

    metrics = evaluate_rollout_samples(
        rollout_samples=rollouts,
        batch=batch,
        best_of_k=2,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )

    assert metrics["sample@1/target_recall"] == 0.0
    assert metrics["oracle_best@2/target_recall"] == 1.0
    assert metrics["best@2/target_recall"] == 0.0
    assert metrics["best@2/score_gap_to_oracle"] == 1.0
    assert metrics["union@2/target_recall"] == 1.0
    assert "model_best@2/target_recall" not in metrics


def test_subgraph_reconstructor_matches_expected_masks() -> None:
    batch = _batch()
    rollout = _rollout_result(
        source_graph_id=[0],
        expand_steps={0: {0: 0, 1: 1}},
        stop_steps={0: 2},
        max_steps=3,
    )
    reconstructor = SubgraphReconstructor(batch, device=torch.device("cpu"))

    direct_nodes, direct_edges = reconstructor.reconstruct(rollout)
    assert direct_nodes.tolist() == [True, True, False, True]
    assert direct_edges.tolist() == [True, True, False, False]


def test_subgraph_reconstructor_stack_and_union() -> None:
    batch = _batch()
    rollouts = (
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0}},
            forced_stop_steps={0: 1},
            max_steps=3,
        ),
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 2}},
            forced_stop_steps={0: 1},
            max_steps=3,
        ),
    )
    reconstructor = SubgraphReconstructor(batch, device=torch.device("cpu"))

    node_masks, edge_masks = reconstructor.stack(rollouts)
    union = reconstructor.union(rollouts)

    assert node_masks.shape == (2, 4)
    assert edge_masks.shape == (2, 4)
    assert union.edge_mask.tolist() == [True, False, True, False]
    assert union.node_mask.tolist() == [True, True, True, False]
