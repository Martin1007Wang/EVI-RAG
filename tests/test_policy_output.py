from __future__ import annotations

import pytest
import torch

from src.weaver.policy.output import ForwardPolicyOutput, TERMINAL_EDGE_ID


def policy_output() -> ForwardPolicyOutput:
    frontier_row_ids = torch.tensor([1, 1], dtype=torch.long)
    edge_log_policy = torch.log_softmax(torch.tensor([1.0, 0.0], dtype=torch.float32), dim=0)
    terminal_log_flow = torch.tensor([-3.0, 0.7], dtype=torch.float32)
    continuation_log_flow = torch.tensor([float("-inf"), 1.5], dtype=torch.float32)
    edge_log_flow = continuation_log_flow.index_select(0, frontier_row_ids) + edge_log_policy
    state_log_flow = torch.logaddexp(terminal_log_flow, continuation_log_flow)
    return ForwardPolicyOutput(
        frontier_row_ids=frontier_row_ids,
        frontier_edge_ids=torch.tensor([4, 5], dtype=torch.long),
        terminal_log_flow=terminal_log_flow,
        continuation_log_flow=continuation_log_flow,
        edge_log_flow=edge_log_flow,
        edge_log_policy=edge_log_policy,
        state_log_flow=state_log_flow,
        terminal_log_prob=terminal_log_flow - state_log_flow,
        edge_log_prob=edge_log_flow - state_log_flow.index_select(0, frontier_row_ids),
        num_rows=2,
        num_edges=8,
    )


def test_action_flow_partition_and_empty_frontier_semantics() -> None:
    out = policy_output()

    expected_state_log_flow = torch.logaddexp(out.terminal_log_flow, out.continuation_log_flow)
    assert torch.allclose(out.state_log_flow, expected_state_log_flow)
    assert torch.allclose(
        out.edge_log_flow,
        out.continuation_log_flow.index_select(0, out.frontier_row_ids) + out.edge_log_policy,
    )
    assert out.terminal_log_prob[0].item() == 0.0
    assert out.continuation_log_flow[0].item() == float("-inf")
    assert out.gather_log_prob(
        row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([TERMINAL_EDGE_ID], dtype=torch.long),
    ).item() == 0.0


def test_flat_action_probability_mass_sums_to_one() -> None:
    out = policy_output()
    action_mass = out.terminal_prob() + out.edge_prob_mass()
    assert torch.allclose(action_mass, torch.ones_like(action_mass))


def test_gather_action_log_flow_uses_terminal_and_edge_flows() -> None:
    out = policy_output()
    values = out.gather_action_log_flow(
        row_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=torch.tensor([TERMINAL_EDGE_ID, 4], dtype=torch.long),
    )
    expected_edge_flow = out.continuation_log_flow[1] + out.edge_log_policy[0]
    assert torch.allclose(values, torch.stack([torch.tensor(-3.0), expected_edge_flow]))


def test_empty_frontier_row_samples_terminal_action() -> None:
    out = policy_output()
    sampled = out.sample(rows=torch.tensor([0], dtype=torch.long))
    assert sampled[0].item() == TERMINAL_EDGE_ID


def test_forward_policy_output_rejects_non_float32_values() -> None:
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = torch.empty(0, dtype=torch.float32)
    with pytest.raises(TypeError, match="torch.float32"):
        ForwardPolicyOutput(
            frontier_row_ids=empty_long,
            frontier_edge_ids=empty_long,
            terminal_log_flow=torch.empty(0, dtype=torch.bfloat16),
            continuation_log_flow=empty_float,
            edge_log_flow=empty_float,
            edge_log_policy=empty_float,
            state_log_flow=empty_float,
            terminal_log_prob=empty_float,
            edge_log_prob=empty_float,
            num_rows=0,
            num_edges=8,
        )
