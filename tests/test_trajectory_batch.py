from __future__ import annotations

import pytest
import torch

from src.weaver.rollout.trajectory import BUDGET, EXTERNAL_TERMINAL, POLICY_STOP, TrajectoryBatch


def test_trajectory_batch_accepts_zero_and_full_length_rows() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0]),
        edge_ids=torch.tensor([[-1, -1], [0, 1]]),
        edge_logp=torch.tensor([[0.0, 0.0], [-0.2, -0.3]]),
        edge_count=torch.tensor([0, 2]),
        stop_reason=torch.tensor([POLICY_STOP, BUDGET], dtype=torch.uint8),
        stop_logp=torch.tensor([-0.1, 0.0]),
        source=torch.tensor([False, False]),
    )

    trajectories.validate()
    assert trajectories.valid_edge_mask().tolist() == [[False, False], [True, True]]


def test_trajectory_batch_rejects_negative_edge_count() -> None:
    with pytest.raises(ValueError, match=r"edge_count must be in \[0, budget\]"):
        _trajectory(edge_count=torch.tensor([-1]))


def test_trajectory_batch_rejects_edge_count_past_budget() -> None:
    with pytest.raises(ValueError, match=r"edge_count must be in \[0, budget\]"):
        _trajectory(edge_count=torch.tensor([2]))


def test_trajectory_batch_rejects_mismatched_edge_logp_shape() -> None:
    with pytest.raises(ValueError, match="edge_logp must match edge_ids shape"):
        TrajectoryBatch(
            graph_ids=torch.tensor([0]),
            edge_ids=torch.tensor([[0]]),
            edge_logp=torch.zeros(1, 2),
            edge_count=torch.tensor([1]),
            stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
            stop_logp=torch.zeros(1),
            source=torch.tensor([False]),
        )


def test_trajectory_batch_rejects_nonnegative_padding_edge_id() -> None:
    with pytest.raises(ValueError, match="padding edge_ids must be -1"):
        _trajectory(edge_ids=torch.tensor([[0]]), edge_count=torch.tensor([0]))


def test_trajectory_batch_rejects_nonzero_padding_edge_logp() -> None:
    with pytest.raises(ValueError, match="padding edge_logp values must be 0.0"):
        _trajectory(edge_ids=torch.tensor([[-1]]), edge_logp=torch.tensor([[0.5]]), edge_count=torch.tensor([0]))


def _trajectory(
    *,
    edge_ids: torch.Tensor | None = None,
    edge_logp: torch.Tensor | None = None,
    edge_count: torch.Tensor | None = None,
) -> TrajectoryBatch:
    return TrajectoryBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0]]) if edge_ids is None else edge_ids,
        edge_logp=torch.tensor([[-0.2]]) if edge_logp is None else edge_logp,
        edge_count=torch.tensor([1]) if edge_count is None else edge_count,
        stop_reason=torch.tensor([EXTERNAL_TERMINAL], dtype=torch.uint8),
        stop_logp=torch.zeros(1),
        source=torch.tensor([True]),
    )
