from __future__ import annotations

import math

import torch

from src.data.collate import RetrievalCollator
from src.data.schema.batch import ReplayBankSample, RetrievalData
from src.eval.frontier_relation_similarity import (
    DatasetScoreCollection,
    FrontierStateRecord,
    collect_frontier_state_records,
    recommend_thresholds,
    replay_prefix_trajectories,
    summarize_collection,
    sweep_thresholds,
)
from src.weaver.context import GraphContext, ReplayContext, TargetContext


def _sample() -> RetrievalData:
    return RetrievalData(
        sample_id="toy/validation/q0",
        edge_index=torch.tensor([[0, 0, 1, 1], [1, 2, 2, 3]], dtype=torch.long),
        num_nodes=4,
        num_edges=4,
        node_entity_catalog_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([3], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([3], dtype=torch.long),
        node_target_distance=torch.tensor([2, 1, 1000000000, 0], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([True, False, True, False], dtype=torch.bool),
        reachable_target_max_distance=torch.tensor([2], dtype=torch.long),
        replay_bank=ReplayBankSample(
            edge_ids_local=torch.tensor([[[0, 2], [0, 2]]], dtype=torch.long),
            edge_count=torch.tensor([[2, 2]], dtype=torch.long),
            priority=torch.tensor([[1.0, 0.5]], dtype=torch.float32),
        ),
    )


def test_replay_prefix_trajectories_flattens_variants_and_slots() -> None:
    batch = RetrievalCollator()([_sample()])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay = ReplayContext.from_batch(
        batch=batch,
        graph_context=graph,
        target_context=target,
    )

    trajectories = replay_prefix_trajectories(
        replay_context=replay,
        device=graph.device,
    )

    assert trajectories.num_trajectories == 2
    assert torch.equal(trajectories.edge_count, torch.tensor([2, 2], dtype=torch.long))
    assert bool(trajectories.source.all())


def test_collect_frontier_state_records_uses_replay_prefix_frontiers() -> None:
    batch = RetrievalCollator()([_sample()])
    relation_semantic_table = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=1,
    )

    collection = collect_frontier_state_records(
        batch=batch,
        relation_semantic_table=relation_semantic_table,
    )

    assert collection.replay_trajectory_count == 2
    assert collection.skipped_empty_frontier >= 1
    assert len(collection.state_records) == 2
    assert collection.state_records[0].positive_scores == (1.0,)
    assert collection.state_records[0].negative_scores == (0.0,)
    assert collection.state_records[1].positive_scores == (0.9701424837112427,)
    assert collection.state_records[1].negative_scores == (0.0, 0.11043152958154678)


def test_sweep_thresholds_and_recommendations_use_state_level_recall() -> None:
    records = (
        FrontierStateRecord(
            positive_scores=(0.9, 0.4),
            negative_scores=(0.8, 0.1),
            frontier_size=4,
        ),
        FrontierStateRecord(
            positive_scores=(0.7,),
            negative_scores=(0.6, 0.2),
            frontier_size=3,
        ),
    )

    rows = sweep_thresholds(
        dataset="toy",
        state_records=records,
        step=0.5,
    )
    by_threshold = {row.threshold: row for row in rows}

    assert math.isclose(by_threshold[0.5].eligible_state_recall, 1.0)
    assert math.isclose(by_threshold[0.5].all_positive_state_recall, 0.5)
    assert math.isclose(by_threshold[0.5].states_with_some_gold_dropped_rate, 0.5)
    assert math.isclose(by_threshold[0.5].states_with_no_edges_left_rate, 0.0)
    assert math.isclose(by_threshold[0.5].positive_edge_recall, 2.0 / 3.0)
    assert math.isclose(by_threshold[0.5].frontier_edge_prune_rate, 3.0 / 7.0)
    assert math.isclose(by_threshold[1.0].eligible_state_recall, 0.0)

    thresholds = recommend_thresholds(
        sweep_rows=rows,
        recall_targets=(1.0, 0.5),
    )
    assert thresholds["1.00"] == 0.5
    assert thresholds["0.50"] == 0.5


def test_summarize_collection_aggregates_counts_and_thresholds() -> None:
    collection = DatasetScoreCollection(
        dataset="toy",
        state_records=(
            FrontierStateRecord(
                positive_scores=(0.8,),
                negative_scores=(0.2,),
                frontier_size=2,
            ),
        ),
        skipped_empty_frontier=3,
        skipped_no_gold_frontier=1,
        replay_trajectory_count=2,
        sample_count=1,
    )

    summary = summarize_collection(collection)

    assert summary.dataset == "toy"
    assert summary.eligible_state_count == 1
    assert summary.positive_edge_count == 1
    assert summary.negative_edge_count == 1
    assert summary.skipped_empty_frontier == 3
    assert summary.skipped_no_gold_frontier == 1
    assert summary.recommended_thresholds["0.95"] == 0.8
