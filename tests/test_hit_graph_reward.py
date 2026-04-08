from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch_scatter")

from src.data.schema import RetrievalBatch
from src.eval.hit_graph_reward import (
    evaluate_hit_graph_reward,
    summarize_hit_graph_rewards,
)
from src.models.reward import RewardModel


def _build_chain_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    batch.batch = torch.tensor([0, 0, 0], dtype=torch.long)
    batch.edge_batch = torch.tensor([0, 0], dtype=torch.long)
    batch.ptr = torch.tensor([0, 3], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True, False, False])
    batch.is_target_mask = torch.tensor([False, False, True])
    batch.num_nodes = 3
    return batch


def _build_root_hit_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.batch = torch.tensor([0], dtype=torch.long)
    batch.edge_batch = torch.empty((0,), dtype=torch.long)
    batch.ptr = torch.tensor([0, 1], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True])
    batch.is_target_mask = torch.tensor([True])
    batch.num_nodes = 1
    return batch


def test_evaluate_hit_graph_reward_matches_teacher_chain_reward() -> None:
    reward_model = RewardModel()

    result = evaluate_hit_graph_reward(
        _build_chain_batch(),
        reward_model=reward_model,
        path_mode="qa_directed",
        stop_on_first_hit=True,
    )

    assert result.status == "ok"
    assert result.recall == pytest.approx(1.0)
    assert result.added_edges == 2
    assert result.log_reward == pytest.approx(math.log(100.0) + 2.0 * math.log(0.95))


def test_evaluate_hit_graph_reward_handles_root_hit() -> None:
    reward_model = RewardModel()

    result = evaluate_hit_graph_reward(
        _build_root_hit_batch(),
        reward_model=reward_model,
        path_mode="undirected",
        stop_on_first_hit=True,
    )

    assert result.status == "root_hit"
    assert result.recall == pytest.approx(1.0)
    assert result.added_edges == 0
    assert result.log_reward == pytest.approx(math.log(100.0))


def test_summarize_hit_graph_rewards_averages_successes_only() -> None:
    summary = summarize_hit_graph_rewards(
        [
            evaluate_hit_graph_reward(
                _build_chain_batch(),
                reward_model=RewardModel(),
                path_mode="qa_directed",
                stop_on_first_hit=True,
            ),
            evaluate_hit_graph_reward(
                _build_root_hit_batch(),
                reward_model=RewardModel(),
                path_mode="undirected",
                stop_on_first_hit=True,
            ),
        ]
    )

    assert summary["num_graphs"] == 2
    assert summary["graphs_with_hit_graph"] == 2
    assert summary["hit_graph_rate"] == pytest.approx(1.0)
    assert summary["avg_hit_graph_recall"] == pytest.approx(1.0)
