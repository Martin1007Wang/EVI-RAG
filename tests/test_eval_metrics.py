from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch_scatter")

from src.data.schema import RetrievalBatch
from src.eval.metrics import (
    compute_distribution_expectations,
    compute_exploration_diversity,
    compute_high_reward_discovery,
)
from src.models.rollout import RolloutBatch


def _build_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.ptr = torch.tensor([0, 2, 3], dtype=torch.long)
    batch.batch = torch.tensor([0, 0, 1], dtype=torch.long)
    batch.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    batch.edge_batch = torch.tensor([0], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True, False, True])
    batch.is_target_mask = torch.tensor([False, True, False])
    batch.num_nodes = 3
    return batch


def _build_rollouts() -> list[RolloutBatch]:
    return [
        RolloutBatch(
            root_log_z=torch.zeros(2),
            termination_action_steps=torch.ones(2, dtype=torch.long),
            trajectory_log_pf=torch.zeros(2),
            trajectory_log_pb=torch.zeros(2),
            terminal_log_rewards=torch.zeros(2),
            terminal_active_nodes=torch.tensor([True, False, True]),
            terminal_active_edges=torch.tensor([False]),
        ),
        RolloutBatch(
            root_log_z=torch.zeros(2),
            termination_action_steps=torch.ones(2, dtype=torch.long),
            trajectory_log_pf=torch.zeros(2),
            trajectory_log_pb=torch.zeros(2),
            terminal_log_rewards=torch.zeros(2),
            terminal_active_nodes=torch.tensor([True, True, True]),
            terminal_active_edges=torch.tensor([True]),
        ),
    ]


def test_distribution_expectations_match_mc_semantics() -> None:
    metrics = compute_distribution_expectations(_build_rollouts(), _build_batch())

    assert metrics["expected_recall"] == pytest.approx(0.5)
    assert metrics["expected_nodes"] == pytest.approx(1.25)
    assert metrics["expected_dangling_ratio"] == pytest.approx(0.0)


def test_high_reward_discovery_reports_budgeted_best_of_k() -> None:
    metrics = compute_high_reward_discovery(
        _build_rollouts(), _build_batch(), ks=[1, 2]
    )

    assert metrics["oracle_max_recall@1"] == pytest.approx(0.0)
    assert metrics["success@1"] == pytest.approx(0.0)
    assert metrics["oracle_max_recall@2"] == pytest.approx(1.0)
    assert metrics["success@2"] == pytest.approx(1.0)


def test_edge_diversity_counts_no_edge_graphs_as_zero() -> None:
    metrics = compute_exploration_diversity(_build_rollouts(), _build_batch())

    # Graph 0 has pairwise edge distance 1.0, graph 1 has no edges and should
    # contribute 0.0 instead of being dropped from the batch average.
    assert metrics["edge_jaccard_diversity"] == pytest.approx(0.5)
