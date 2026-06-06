from __future__ import annotations

import torch

from src.data.schema.batch import ReplayBankSample, RetrievalBatch, RetrievalData
from src.eval.retrieval import compute_node_retrieval_matrix
from src.eval.rollout import hit_terminal_stats, retrieval_from_masks
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch


def _batch(*, anchor_is_target: bool) -> RetrievalBatch:
    sample = RetrievalData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        num_edges=2,
        node_entity_catalog_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([20, 21], dtype=torch.long),
        question_emb=torch.zeros(4, dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([0] if anchor_is_target else [2], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([0] if anchor_is_target else [2], dtype=torch.long),
        node_target_distance=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([False, False], dtype=torch.bool),
        reachable_target_max_distance=torch.tensor([0 if anchor_is_target else 2], dtype=torch.long),
        replay_bank=ReplayBankSample(
            edge_ids_local=torch.full((0, 0), -1, dtype=torch.long),
            edge_count=torch.empty(0, dtype=torch.long),
            priority=torch.empty(0, dtype=torch.float32),
        ),
    )
    batch = RetrievalBatch.from_data_list(
        [sample],
        follow_batch=["reachable_target_node_ids"],
        exclude_keys=["question_emb", "replay_bank_edge_ids", "replay_bank_edge_count", "replay_bank"],
    )
    batch.question_emb = sample.question_emb.unsqueeze(0)
    batch.edge_batch = torch.tensor([0, 0], dtype=torch.long)
    batch.replay_bank = sample.replay_bank
    return batch


def _trajectories(*, edge_ids: list[list[int]], edge_count: list[int]) -> TrajectoryBatch:
    budget = len(edge_ids[0]) if edge_ids else 0
    return TrajectoryBatch(
        graph_ids=torch.zeros(len(edge_ids), dtype=torch.long),
        edge_ids=torch.tensor(edge_ids, dtype=torch.long),
        edge_logp=torch.zeros((len(edge_ids), budget), dtype=torch.float32),
        edge_count=torch.tensor(edge_count, dtype=torch.long),
        stop_reason=torch.full((len(edge_ids),), POLICY_STOP, dtype=torch.uint8),
        stop_logp=torch.zeros(len(edge_ids), dtype=torch.float32),
        source=torch.zeros(len(edge_ids), dtype=torch.bool),
    )


def test_compute_node_retrieval_matrix_excludes_anchor_targets_from_hits_and_gold() -> None:
    batch = _batch(anchor_is_target=True)
    trajectories = _trajectories(edge_ids=[[-1, -1]], edge_count=[0])

    precision, recall, f1, valid = compute_node_retrieval_matrix(
        trajectories,
        batch,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )

    assert torch.equal(valid, torch.tensor([False]))
    assert torch.equal(precision, torch.zeros((1, 1)))
    assert torch.equal(recall, torch.zeros((1, 1)))
    assert torch.equal(f1, torch.zeros((1, 1)))


def test_retrieval_from_masks_uses_consistent_anchor_excluded_semantics() -> None:
    batch = _batch(anchor_is_target=True)
    node_masks = torch.tensor([[True, False, False]], dtype=torch.bool)

    precision, recall, f1, valid = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )

    assert torch.equal(valid, torch.tensor([False]))
    assert torch.equal(precision, torch.zeros((1, 1)))
    assert torch.equal(recall, torch.zeros((1, 1)))
    assert torch.equal(f1, torch.zeros((1, 1)))


def test_hit_terminal_stats_ignores_anchor_only_hits() -> None:
    batch = _batch(anchor_is_target=True)
    trajectories = _trajectories(edge_ids=[[-1, -1], [0, -1]], edge_count=[0, 1])

    stats = hit_terminal_stats(
        trajectories,
        batch=batch,
    )

    assert torch.equal(stats["hit"], torch.tensor([[False], [False]]))
    assert torch.equal(stats["continued"], torch.tensor([[False], [False]]))
    assert torch.equal(stats["wasted_edges"], torch.zeros((2, 1)))
