from __future__ import annotations

import torch

from src.weaver.policy import PolicyOutput
from src.weaver.rollout.action import StepAction, sample_step
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.tape import RolloutTape


def test_sample_step_normalizes_bfloat16_policy_outputs_to_float32() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.tensor([-0.1, 0.0], dtype=torch.bfloat16),
        edge_logit=torch.tensor([-2.0], dtype=torch.bfloat16),
        state_log_flow=torch.zeros(2, dtype=torch.bfloat16),
        edge_row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        num_rows=2,
        num_edges=1,
    )

    action = sample_step(
        policy_out=policy_out,
        rows=torch.tensor([0, 1], dtype=torch.long),
    )

    assert action.policy_log_prob.dtype == torch.float32
    assert action.behavior_log_prob.dtype == torch.float32


def test_forced_stop_action_marks_rows_and_uses_float32_log_prob() -> None:
    action = StepAction.forced_stop(
        rows=torch.tensor([0], dtype=torch.long),
        dtype=torch.float32,
    )

    assert action.stop_rows.tolist() == [0]
    assert action.forced.tolist() == [True]
    assert action.policy_log_prob.dtype == torch.float32
    assert action.behavior_log_prob.dtype == torch.float32


def test_sample_step_records_policy_and_behavior_log_prob() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.zeros(1, dtype=torch.float32),
        edge_logit=torch.zeros(1, dtype=torch.float32),
        state_log_flow=torch.zeros(1, dtype=torch.float32),
        edge_row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([7], dtype=torch.long),
        num_rows=1,
        num_edges=8,
    )

    action = sample_step(
        policy_out=policy_out,
        rows=torch.tensor([0], dtype=torch.long),
        temperature=0.5,
    )

    if int(action.edge_ids[0].item()) == -1:
        expected_policy = -torch.log(torch.tensor(2.0))
        expected_behavior = -torch.log(torch.tensor(2.0))
    else:
        expected_policy = -torch.log(torch.tensor(2.0))
        expected_behavior = -torch.log(torch.tensor(2.0))
    assert torch.allclose(action.policy_log_prob, expected_policy.view(1))
    assert torch.allclose(action.behavior_log_prob, expected_behavior.view(1))


def test_rollout_tape_records_bfloat16_log_probs_as_float32() -> None:
    tape = RolloutTape(
        R=3,
        T=1,
        device=torch.device("cpu"),
    )
    action = StepAction(
        row_ids=torch.tensor([2, 0], dtype=torch.long),
        edge_ids=torch.tensor([-1, -1], dtype=torch.long),
        policy_log_prob=torch.tensor([-0.5, -1.5], dtype=torch.bfloat16),
        behavior_log_prob=torch.tensor([-0.25, -1.25], dtype=torch.bfloat16),
        forced=torch.tensor([False, True], dtype=torch.bool),
    )

    tape.write(0, action)

    assert tape.policy_action_log_prob.dtype == torch.float32
    assert tape.behavior_action_log_prob.dtype == torch.float32
    assert torch.allclose(
        tape.policy_action_log_prob[:, 0],
        torch.tensor([-1.5, 0.0, -0.5], dtype=torch.float32),
    )
    assert torch.allclose(
        tape.behavior_action_log_prob[:, 0],
        torch.tensor([-1.25, 0.0, -0.25], dtype=torch.float32),
    )
    assert tape.stop_step.tolist() == [0, -1, 0]
    assert tape.forced_stop.tolist() == [True, False, False]


def test_rollout_result_uses_tape_fields_for_valid_and_stop_masks() -> None:
    tape = RolloutTape(
        R=2,
        T=2,
        device=torch.device("cpu"),
    )
    tape.write(
        0,
        StepAction(
            row_ids=torch.tensor([0, 1], dtype=torch.long),
            edge_ids=torch.tensor([3, -1], dtype=torch.long),
            policy_log_prob=torch.tensor([-0.2, -0.5], dtype=torch.float32),
            behavior_log_prob=torch.tensor([-0.1, -0.4], dtype=torch.float32),
            forced=torch.tensor([False, False], dtype=torch.bool),
        ),
    )
    tape.write(
        1,
        StepAction(
            row_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([-1], dtype=torch.long),
            policy_log_prob=torch.tensor([-0.3], dtype=torch.float32),
            behavior_log_prob=torch.tensor([-0.2], dtype=torch.float32),
            forced=torch.tensor([False], dtype=torch.bool),
        ),
    )

    result = RolloutResult(
        source_graph_id=torch.tensor([0, 0], dtype=torch.long),
        selected_edge_ids=tape.selected_edge_ids,
        policy_action_log_prob=tape.policy_action_log_prob,
        behavior_action_log_prob=tape.behavior_action_log_prob,
        stop_step=tape.stop_step,
        forced_stop=tape.forced_stop,
        expand_budget=1,
    )

    assert result.valid_mask.tolist() == [[True, True], [True, False]]
    assert result.stop_mask.tolist() == [[False, True], [True, False]]
    assert result.expand_mask.tolist() == [[True, False], [False, False]]
    assert result.selected_edge_ids.tolist() == [[3, -1], [-1, -1]]
