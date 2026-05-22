from __future__ import annotations

import pytest
import torch

from src.weaver.module import policy_diagnostic_metrics
from src.weaver.policy.output import ForwardPolicyOutput


def make_policy_output(
    *,
    stop_log_flow: torch.Tensor,
    continue_log_gain: torch.Tensor,
    frontier_row_ids: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    edge_log_reference: torch.Tensor,
    edge_log_advantage: torch.Tensor,
    num_rows: int,
    num_edges: int,
) -> ForwardPolicyOutput:
    continue_log_flow = stop_log_flow + continue_log_gain
    edge_log_flow = stop_log_flow.index_select(0, frontier_row_ids) + edge_log_reference + edge_log_advantage
    state_log_flow = stop_log_flow + torch.nn.functional.softplus(continue_log_gain)
    return ForwardPolicyOutput(
        frontier_row_ids=frontier_row_ids,
        frontier_edge_ids=frontier_edge_ids,
        stop_log_flow=stop_log_flow,
        continue_log_flow=continue_log_flow,
        continue_log_gain=continue_log_gain,
        edge_log_flow=edge_log_flow,
        edge_log_reference=edge_log_reference,
        edge_log_advantage=edge_log_advantage,
        state_log_flow=state_log_flow,
        stop_log_prob=-torch.nn.functional.softplus(continue_log_gain),
        edge_log_prob=edge_log_flow - state_log_flow.index_select(0, frontier_row_ids),
        num_rows=num_rows,
        num_edges=num_edges,
    )


def test_policy_diagnostic_metrics_accepts_stop_continue_outputs() -> None:
    expansion_out = make_policy_output(
        stop_log_flow=torch.tensor([0.2, 0.7], dtype=torch.float32),
        continue_log_gain=torch.tensor([-0.4, 0.3], dtype=torch.float32),
        frontier_row_ids=torch.tensor([0, 1], dtype=torch.long),
        frontier_edge_ids=torch.tensor([2, 3], dtype=torch.long),
        edge_log_reference=torch.tensor([-0.7, -0.8], dtype=torch.float32),
        edge_log_advantage=torch.tensor([0.0, 0.1], dtype=torch.float32),
        num_rows=2,
        num_edges=5,
    )
    terminal_out = make_policy_output(
        stop_log_flow=torch.tensor([0.5], dtype=torch.float32),
        continue_log_gain=torch.tensor([float("-inf")], dtype=torch.float32),
        frontier_row_ids=torch.empty(0, dtype=torch.long),
        frontier_edge_ids=torch.empty(0, dtype=torch.long),
        edge_log_reference=torch.empty(0, dtype=torch.float32),
        edge_log_advantage=torch.empty(0, dtype=torch.float32),
        num_rows=1,
        num_edges=5,
    )

    metrics = policy_diagnostic_metrics(
        expansion_out=expansion_out,
        expansion_depth=torch.tensor([0, 1], dtype=torch.long),
        terminal_out=terminal_out,
        terminal_depth=torch.tensor([2], dtype=torch.long),
    )

    assert metrics["policy_stop_vs_continue_log_ratio_depth0_mean"].item() == pytest.approx(0.4)
    assert metrics["policy_stop_vs_continue_log_ratio_depth1_mean"].item() == pytest.approx(-0.3)
    assert metrics["policy_stop_vs_continue_log_ratio_depth2_mean"].item() == float("inf")
    assert torch.isfinite(metrics["policy_frontier_size_mean"])
    assert metrics["policy_stop_prob_depth0_mean"].item() == pytest.approx(torch.sigmoid(torch.tensor(0.4)).item())
