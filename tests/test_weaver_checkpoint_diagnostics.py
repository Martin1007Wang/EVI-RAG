from __future__ import annotations

import math

import pytest
import torch

from scripts.diagnose_weaver_checkpoint import (
    oracle_truncate_after_hit,
    rank_of_any_gold,
    stop_residual_stats,
    weak_gold_frontier_edges,
)
from src.data.schema.batch import ReplayBankSample, RetrievalBatch, RetrievalData
from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.rollout.trajectory import BUDGET_TRUNCATED, POLICY_STOP, TrajectoryBatch


def _batch() -> RetrievalBatch:
    sample = RetrievalData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        num_edges=2,
        node_entity_catalog_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([20, 21], dtype=torch.long),
        question_emb=torch.zeros(4, dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
        node_target_distance=torch.tensor([2, 1, 0], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([True, True], dtype=torch.bool),
        reachable_target_max_distance=torch.tensor([2], dtype=torch.long),
        replay_bank=ReplayBankSample(
            edge_ids_local=torch.full((0, 0), -1, dtype=torch.long),
            edge_count=torch.empty(0, dtype=torch.long),
            priority=torch.empty(0, dtype=torch.float32),
        ),
    )
    batch = RetrievalBatch.from_data_list(
        [sample],
        follow_batch=["reachable_target_node_ids"],
        exclude_keys=[
            "question_emb",
            "replay_bank_edge_ids",
            "replay_bank_edge_count",
            "replay_bank",
        ],
    )
    batch.sample_id = ["sample-0"]
    batch.question_emb = sample.question_emb.unsqueeze(0)
    batch.edge_batch = torch.tensor([0, 0], dtype=torch.long)
    batch.replay_bank = sample.replay_bank
    return batch


def _graph_and_target() -> tuple[GraphContext, TargetContext]:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    return graph, target


def test_weak_gold_frontier_edges_selects_shortest_path_anchor_edge() -> None:
    graph, target = _graph_and_target()
    gold = weak_gold_frontier_edges(
        graph=graph,
        target=target,
        graph_id=0,
        row_id=0,
        frontier_edge_ids=torch.tensor([0, 1], dtype=torch.long),
        frontier_row_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    assert gold == {0}


def test_rank_of_any_gold_uses_descending_scores() -> None:
    rank = rank_of_any_gold(
        edge_ids=[-1, 0, 1],
        scores=[0.9, 0.1, 0.8],
        gold_edges={0, 1},
    )

    assert rank == 2


def test_oracle_truncate_after_hit_preserves_prefix_through_first_hit() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[0, 1, -1]], dtype=torch.long),
        edge_logp=torch.tensor([[-0.2, -0.3, 0.0]], dtype=torch.float32),
        edge_count=torch.tensor([2], dtype=torch.long),
        stop_reason=torch.tensor([BUDGET_TRUNCATED], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )

    truncated = oracle_truncate_after_hit(
        trajectories=trajectories,
        batch=batch,
        context=graph,
        budget=3,
    )

    assert torch.equal(truncated.edge_ids, torch.tensor([[0, 1, -1]], dtype=torch.long))
    assert torch.equal(truncated.edge_count, torch.tensor([2], dtype=torch.long))
    assert int(truncated.stop_reason[0].item()) == int(POLICY_STOP)


def test_stop_residual_stats_handles_empty_and_signed_values() -> None:
    empty = stop_residual_stats(torch.empty(0))
    assert empty.count == 0
    assert empty.mean == 0.0
    assert math.isnan(empty.min)

    stats = stop_residual_stats(torch.tensor([-2.0, 1.0, 4.0]))
    assert stats.count == 3
    assert stats.mean == 1.0
    assert stats.abs_mean == pytest.approx(7.0 / 3.0)
    assert stats.min == -2.0
    assert stats.max == 4.0
