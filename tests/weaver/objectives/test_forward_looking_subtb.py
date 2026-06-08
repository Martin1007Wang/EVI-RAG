from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.objectives.subtb.batch import SubTBBatch, SubTBTermTable, prepare_subtb_batch
from src.weaver.objectives.subtb.loss import ForwardLookingSubTBObjective
from src.weaver.objectives.subtb.scoring import SubTBPolicyScores
from src.weaver.policy import FlowEstimator
from src.weaver.rollout.trajectory import BUDGET_TRUNCATED, EXTERNAL_TERMINAL, NO_FRONTIER, POLICY_STOP, TrajectoryBatch
from src.weaver.state import StateBatch
from src.weaver.reward import EvidenceStateScoreOutput


class _ActionSpace:
    frontier = type("Frontier", (), {"row_ids": torch.empty(0, dtype=torch.long)})()


def _squared_residual(value: float) -> float:
    return value * value


def _tensor(values, *, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def _long(values):
    return torch.tensor(values, dtype=torch.long)


def _bool(values):
    return torch.tensor(values, dtype=torch.bool)


def _reward(*, log_reward, terminal_valid_mask, state_potential=None) -> EvidenceStateScoreOutput:
    log_reward_tensor = _tensor(log_reward)
    state_potential_tensor = _tensor(log_reward if state_potential is None else state_potential)
    n = int(log_reward_tensor.numel())
    zeros = torch.zeros(n, dtype=torch.float32)
    terminal_valid = _bool(terminal_valid_mask)
    return EvidenceStateScoreOutput(
        state_potential=state_potential_tensor,
        remaining_log_reward=log_reward_tensor - state_potential_tensor,
        log_reward=log_reward_tensor,
        raw_log_reward=log_reward_tensor,
        answer_count=zeros,
        candidate_count=zeros,
        target_count=zeros,
        target_recall=state_potential_tensor,
        target_precision=zeros,
        terminal_quality=log_reward_tensor,
        edge_count=zeros,
        valid_target_mask=terminal_valid,
        nonempty_mask=terminal_valid,
        success_mask=terminal_valid,
        terminal_valid_mask=terminal_valid,
        metrics={},
    )


def _scores(
    *,
    log_flow,
    terminal_stop_logp_by_traj,
    forward_prefix_by_traj,
    backward_prefix_by_traj,
    stop_log_prob_by_state=None,
) -> SubTBPolicyScores:
    log_flow_tensor = _tensor(log_flow)
    if stop_log_prob_by_state is None:
        stop_log_prob_tensor = torch.zeros_like(log_flow_tensor)
        stop_log_prob_tensor[: min(len(log_flow), len(terminal_stop_logp_by_traj))] = _tensor(terminal_stop_logp_by_traj)
    else:
        stop_log_prob_tensor = _tensor(stop_log_prob_by_state)
    forward_prefix = _tensor(forward_prefix_by_traj)
    backward_prefix = _tensor(backward_prefix_by_traj)
    return SubTBPolicyScores(
        frontier_count=torch.zeros_like(log_flow_tensor),
        log_flow=log_flow_tensor,
        stop_log_prob_by_state=stop_log_prob_tensor,
        step_log_prob=forward_prefix[:, 1:] - forward_prefix[:, :-1],
        backward_step_log_prob=backward_prefix[:, 1:] - backward_prefix[:, :-1],
        terminal_stop_logp_by_traj=_tensor(terminal_stop_logp_by_traj),
    )


def _trajectory_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        graph_ids=_long([0, 0]),
        edge_ids=_long([[0, 1], [0, -1]]),
        edge_logp=torch.zeros((2, 2), dtype=torch.float32),
        edge_count=_long([2, 1]),
        stop_reason=torch.tensor([POLICY_STOP, BUDGET_TRUNCATED], dtype=torch.uint8),
        stop_logp=torch.zeros(2, dtype=torch.float32),
        source=_bool([False, True]),
    )


def _state_batch() -> StateBatch:
    return StateBatch(
        graph_ids=_long([0, 0, 0]),
        edge_ids=_long([[-1, -1], [0, -1], [0, 1]]),
        edge_count=_long([0, 1, 2]),
    )


def _graph_context() -> GraphContext:
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 3], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 3, 3], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 2, 1], dtype=torch.long),
        ),
        num_nodes=3,
        num_edges=3,
        num_graphs=1,
    )


def test_forward_looking_subtb_uses_combined_log_flow_on_transition_terms() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 0.0], terminal_valid_mask=[False, False, False])
    scores = _scores(
        log_flow=[0.2, 0.7, 1.3],
        terminal_stop_logp_by_traj=[0.0, 0.0],
        forward_prefix_by_traj=[[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.1], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    expected_residual = 0.2 + 0.5 - 0.1 - 0.7
    assert torch.isclose(output.loss, torch.tensor(_squared_residual(expected_residual)))
    assert output.metrics["objective/subtb_transition_abs_residual_mean"] == pytest.approx(abs(expected_residual))
    assert output.metrics["objective/subtb_transition_abs_residual_p95"] == pytest.approx(abs(expected_residual))
    assert output.metrics["objective/subtb_transition_abs_residual_max"] == pytest.approx(abs(expected_residual))


def test_forward_looking_subtb_uses_stable_quadratic_penalty_for_large_residuals() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 0.0], terminal_valid_mask=[False, False, False])
    scores = _scores(
        log_flow=[0.0, 0.0, 0.0],
        terminal_stop_logp_by_traj=[0.0, 0.0],
        forward_prefix_by_traj=[[0.0, -60.0, -60.0], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    assert torch.isfinite(output.loss)
    assert torch.isclose(output.loss, torch.tensor(3600.0))
    assert output.metrics["objective/subtb_transition_abs_residual_max"] == pytest.approx(60.0)


def test_forward_looking_subtb_terminal_term_uses_log_reward_and_end_state_stop() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, False]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0, 1]),
            start_steps=_long([0, 0]),
            end_steps=_long([2, 1]),
            start_state_ids=_long([0, 0]),
            end_state_ids=_long([2, 1]),
            lambda_exponent=_long([1, 0]),
        ),
    )
    reward = _reward(log_reward=[-1.0, -0.5, 1.4], terminal_valid_mask=[True, True, True])
    scores = _scores(
        log_flow=[0.2, 0.6, 1.0],
        terminal_stop_logp_by_traj=[0.4, 0.9],
        stop_log_prob_by_state=[-1.2, -1.1, 0.4],
        forward_prefix_by_traj=[[0.0, 0.3, 0.7], [0.0, 0.25, 0.25]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.15, 0.15]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    residual_stop = 0.2 + 0.7 + 0.4 - 0.2 - 1.4
    residual_forced = 0.2 + 0.25 - 1.1 - 0.15 - (-0.5)
    expected_loss = 2.0 * (_squared_residual(residual_stop) + _squared_residual(residual_forced)) / (2.0 * 2.0)
    assert torch.isclose(output.loss, torch.tensor(expected_loss))
    assert output.metrics["objective/subtb_terminal_abs_residual_mean"] == pytest.approx((abs(residual_stop) + abs(residual_forced)) / 2.0)
    assert output.metrics["objective/subtb_terminal_abs_residual_p95"] == pytest.approx(0.3)
    assert output.metrics["objective/subtb_terminal_abs_residual_max"] == pytest.approx(0.3)


def test_forward_looking_subtb_terminal_stop_metric_ignores_invalid_terminal_states() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 0.0], terminal_valid_mask=[False, False, True])
    scores = _scores(
        log_flow=[0.0, 0.0, 0.0],
        terminal_stop_logp_by_traj=[-0.7, -1.0e9],
        stop_log_prob_by_state=[0.0, -1.0e9, -0.7],
        forward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    assert output.metrics["objective/terminal_stop_log_prob_mean"] == pytest.approx(-0.7)


def test_forward_looking_subtb_filters_invalid_terminal_rewards() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([2]),
            start_state_ids=_long([0]),
            end_state_ids=_long([2]),
            lambda_exponent=_long([1]),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 1.4], terminal_valid_mask=[True, True, False])
    scores = _scores(
        log_flow=[0.2, 0.6, 1.0],
        terminal_stop_logp_by_traj=[0.4, 0.0],
        stop_log_prob_by_state=[-0.2, -0.6, 0.0],
        forward_prefix_by_traj=[[0.0, 0.3, 0.7], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    assert torch.equal(output.loss, torch.tensor(0.0))
    assert output.metrics["objective/subtb_terminal_abs_residual_mean"] == 0.0


def test_prepare_subtb_batch_merges_policy_and_replay_terms() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=_long([0, 0, 0, 0]),
        edge_ids=_long([[0, 1], [0, -1], [2, -1], [-1, -1]]),
        edge_logp=torch.zeros((4, 2), dtype=torch.float32),
        edge_count=_long([2, 1, 1, 0]),
        stop_reason=torch.tensor([POLICY_STOP, NO_FRONTIER, EXTERNAL_TERMINAL, BUDGET_TRUNCATED], dtype=torch.uint8),
        stop_logp=torch.zeros(4, dtype=torch.float32),
        source=_bool([False, True, False, True]),
    )
    graph_context = _graph_context()

    batch = prepare_subtb_batch(trajectories=trajectories, graph_context=graph_context)

    assert batch.transition_terms.num_terms == 5
    assert batch.terminal_terms.num_terms == 9
    assert torch.equal(batch.terminal_trainable_stop_mask, _bool([True, True, True, False]))
    assert torch.equal(batch.terminal_terms.traj_ids, _long([0, 0, 0, 0, 0, 1, 1, 2, 2]))
    assert torch.equal(batch.terminal_terms.start_steps, _long([0, 0, 1, 1, 2, 0, 1, 0, 1]))
    assert torch.equal(batch.terminal_terms.end_steps, _long([1, 2, 1, 2, 2, 1, 1, 1, 1]))


def test_forward_looking_subtb_trains_stop_on_nonfinal_prefix_state() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([1]),
            end_steps=_long([1]),
            start_state_ids=_long([1]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, -0.5, 1.4], terminal_valid_mask=[False, True, True])
    scores = _scores(
        log_flow=[0.2, 0.6, 1.0],
        terminal_stop_logp_by_traj=[0.0, 0.0],
        stop_log_prob_by_state=[-1.2, -0.7, 0.4],
        forward_prefix_by_traj=[[0.0, 0.3, 0.7], [0.0, 0.25, 0.25]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.15, 0.15]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    residual = 0.6 - 0.7 - (-0.5)
    assert torch.isclose(output.loss, torch.tensor(_squared_residual(residual)))


def test_scoring_like_contract_requires_reward_state_potential_for_log_flow() -> None:
    reward = _reward(log_reward=[0.1, 0.2], terminal_valid_mask=[True, True], state_potential=[0.4, 0.6])
    assert torch.equal(reward.state_potential, torch.tensor([0.4, 0.6]))
    assert torch.allclose(reward.remaining_log_reward, torch.tensor([-0.3, -0.4]))


def test_budget_truncated_stop_mask_includes_stop_log_prob() -> None:
    batch = SubTBBatch(
        trajectories=replace(_trajectory_batch(), stop_reason=torch.tensor([POLICY_STOP, BUDGET_TRUNCATED], dtype=torch.uint8)),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([1]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, -0.5, 0.0], terminal_valid_mask=[True, True, True])
    scores = _scores(
        log_flow=[0.3, 0.8, 1.2],
        terminal_stop_logp_by_traj=[0.5, 1.3],
        stop_log_prob_by_state=[-0.3, -1.3, -1.2],
        forward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.25, 0.25]],
        backward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.15, 0.15]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    residual = 0.3 + 0.25 - 1.3 - 0.15 - (-0.5)
    expected_loss = 2.0 * _squared_residual(residual) / (2.0 * 1.0)
    assert torch.isclose(output.loss, torch.tensor(expected_loss))


def test_external_terminal_stop_mask_includes_stop_log_prob() -> None:
    batch = SubTBBatch(
        trajectories=replace(_trajectory_batch(), stop_reason=torch.tensor([POLICY_STOP, EXTERNAL_TERMINAL], dtype=torch.uint8)),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([1]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, -0.5, 0.0], terminal_valid_mask=[True, True, True])
    scores = _scores(
        log_flow=[0.3, 0.8, 1.2],
        terminal_stop_logp_by_traj=[0.5, 1.3],
        stop_log_prob_by_state=[-0.3, -1.3, -1.2],
        forward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.25, 0.25]],
        backward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.15, 0.15]],
    )

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)

    residual = 0.3 + 0.25 - 1.3 - 0.15 - (-0.5)
    expected_loss = 2.0 * _squared_residual(residual) / (2.0 * 1.0)
    assert torch.isclose(output.loss, torch.tensor(expected_loss))


def test_forward_looking_subtb_terminal_loss_weight_reweights_base_loss() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, False]),
        transition_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([2]),
            start_state_ids=_long([0]),
            end_state_ids=_long([2]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 1.0], terminal_valid_mask=[True, False, True])
    scores = _scores(
        log_flow=[0.2, 0.7, 1.3],
        terminal_stop_logp_by_traj=[0.4, 0.0],
        stop_log_prob_by_state=[-0.5, -0.4, 0.1],
        forward_prefix_by_traj=[[0.0, 0.5, 0.7], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective(terminal_loss_weight=2.0)(batch=batch, scores=scores, reward=reward)

    transition_residual = 0.2 + 0.5 - 0.1 - 0.7
    terminal_residual = 0.2 + 0.7 + 0.1 - 0.2 - 1.0
    expected_numerator = _squared_residual(transition_residual) + 2.0 * _squared_residual(terminal_residual)
    expected_denominator = 1.0 + 2.0 * 1.0
    expected_loss = expected_numerator / expected_denominator
    assert torch.isclose(output.loss, torch.tensor(expected_loss))
    assert output.metrics["objective/terminal_loss_weight"] == pytest.approx(2.0)


def test_forward_looking_subtb_terminal_loss_weight_zero_drops_terminal_aggregation() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, False]),
        transition_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([2]),
            start_state_ids=_long([0]),
            end_state_ids=_long([2]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 1.0], terminal_valid_mask=[True, False, True])
    scores = _scores(
        log_flow=[0.2, 0.7, 1.3],
        terminal_stop_logp_by_traj=[0.4, 0.0],
        stop_log_prob_by_state=[-0.5, -0.4, 0.1],
        forward_prefix_by_traj=[[0.0, 0.5, 0.7], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
    )

    output = ForwardLookingSubTBObjective(terminal_loss_weight=0.0)(batch=batch, scores=scores, reward=reward)

    transition_residual = 0.2 + 0.5 - 0.1 - 0.7
    expected_loss = _squared_residual(transition_residual)
    assert torch.isclose(output.loss, torch.tensor(expected_loss))


def test_forward_looking_subtb_adds_path_nce_loss() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([False, False]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
    )
    scores = _scores(
        log_flow=[0.0, 0.0, 0.0],
        terminal_stop_logp_by_traj=[0.0, 0.0],
        stop_log_prob_by_state=[0.0, 0.0, 0.0],
        forward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    frontier_log_prob = torch.tensor([0.1, 0.9], dtype=torch.float32, requires_grad=True)
    scores = replace(
        scores,
        frontier_count=_tensor([2.0, 0.0, 0.0]),
        frontier_row_ids=_long([0, 0]),
        frontier_edge_ids=_long([0, 1]),
        frontier_log_prob=frontier_log_prob,
    )
    reward = _reward(log_reward=[0.0, 0.0, 0.0], terminal_valid_mask=[False, False, False])

    output = ForwardLookingSubTBObjective(
        path_nce_weight=0.3,
        path_nce_temperature=1.0,
    )(
        batch=batch,
        scores=scores,
        reward=reward,
        global_step=20,
        path_gold_mask=_bool([True, False]),
    )
    output.loss.backward()

    expected_nce = torch.logsumexp(torch.tensor([0.1, 0.9]), dim=0) - torch.tensor(0.1)
    assert output.metrics["objective/path_nce_state_count"] == 1.0
    assert output.metrics["objective/path_nce_positive_count"] == 1.0
    assert output.metrics["objective/path_nce_weight"] == pytest.approx(0.3)
    assert torch.isclose(output.loss, 0.3 * expected_nce)
    assert frontier_log_prob.grad is not None
    assert frontier_log_prob.grad[0].item() < 0.0
    assert frontier_log_prob.grad[1].item() > 0.0


def test_flow_estimator_stop_logits_do_not_depend_on_frontier_edge_features() -> None:
    torch.manual_seed(0)
    estimator = FlowEstimator(hidden_dim=4)
    estimator.eval()
    state_h = torch.randn(1, 4)
    frontier_edge_h = torch.randn(2, 4)
    frontier_row_ids = torch.tensor([0, 0], dtype=torch.long)

    edge_logits, stop_logits = estimator(
        state_h=state_h,
        frontier_row_ids=frontier_row_ids,
        frontier_edge_h=frontier_edge_h,
    )
    shifted_edge_logits, shifted_stop_logits = estimator(
        state_h=state_h,
        frontier_row_ids=frontier_row_ids,
        frontier_edge_h=frontier_edge_h + 7.0,
    )

    assert torch.allclose(stop_logits, shifted_stop_logits, atol=1.0e-6)
    assert shifted_edge_logits.shape == edge_logits.shape


def test_flow_estimator_initializes_stop_head_bias() -> None:
    estimator = FlowEstimator(hidden_dim=4, stop_initial_bias=1.5)

    assert estimator.stop_head[-1].bias.detach().item() == pytest.approx(1.5)


def test_flow_estimator_edge_energy_backpropagates_through_frontier_edge_features() -> None:
    torch.manual_seed(0)
    estimator = FlowEstimator(hidden_dim=4)
    state_h = torch.randn(1, 4)
    frontier_edge_h = torch.randn(2, 4, requires_grad=True)
    frontier_row_ids = torch.tensor([0, 0], dtype=torch.long)

    edge_logits, _ = estimator(
        state_h=state_h,
        frontier_row_ids=frontier_row_ids,
        frontier_edge_h=frontier_edge_h,
    )
    edge_logits.logsumexp(dim=0).backward()

    assert frontier_edge_h.grad is not None
    assert torch.count_nonzero(frontier_edge_h.grad).item() > 0


def test_forward_looking_subtb_detaches_terminal_log_reward_from_training_graph() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, True]),
        transition_terms=SubTBTermTable(
            traj_ids=torch.empty(0, dtype=torch.long),
            start_steps=torch.empty(0, dtype=torch.long),
            end_steps=torch.empty(0, dtype=torch.long),
            start_state_ids=torch.empty(0, dtype=torch.long),
            end_state_ids=torch.empty(0, dtype=torch.long),
            lambda_exponent=torch.empty(0, dtype=torch.long),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([2]),
            start_state_ids=_long([0]),
            end_state_ids=_long([2]),
            lambda_exponent=_long([0]),
        ),
    )
    log_reward = torch.tensor([0.0, -0.5, 1.4], dtype=torch.float32, requires_grad=True)
    zeros = torch.zeros(3, dtype=torch.float32)
    terminal_valid = _bool([True, True, True])
    reward = EvidenceStateScoreOutput(
        state_potential=torch.zeros(3, dtype=torch.float32),
        remaining_log_reward=log_reward,
        log_reward=log_reward,
        raw_log_reward=log_reward,
        answer_count=zeros,
        candidate_count=zeros,
        target_count=zeros,
        target_recall=zeros,
        target_precision=zeros,
        terminal_quality=log_reward,
        edge_count=zeros,
        valid_target_mask=terminal_valid,
        nonempty_mask=terminal_valid,
        success_mask=terminal_valid,
        terminal_valid_mask=terminal_valid,
        metrics={},
    )
    scores = _scores(
        log_flow=[0.2, 0.6, 1.0],
        terminal_stop_logp_by_traj=[0.4, 0.0],
        forward_prefix_by_traj=[[0.0, 0.3, 0.7], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
    )
    scores = replace(scores, log_flow=scores.log_flow.detach().requires_grad_(True))

    output = ForwardLookingSubTBObjective()(batch=batch, scores=scores, reward=reward)
    output.loss.backward()

    assert log_reward.grad is None


def test_forward_looking_subtb_terminal_loss_weight_scales_terminal_gradients_only() -> None:
    batch = SubTBBatch(
        trajectories=_trajectory_batch(),
        states=_state_batch(),
        prefix_state_ids=_long([[0, 1, 2], [0, 1, -1]]),
        valid_steps=_bool([[True, True], [True, False]]),
        step_traj_ids=_long([0, 0, 1]),
        step_ids=_long([0, 1, 0]),
        step_parent_state_ids=_long([0, 1, 0]),
        step_edge_ids=_long([0, 1, 0]),
        terminal_state_ids=_long([2, 1]),
        terminal_trainable_stop_mask=_bool([True, False]),
        transition_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([1]),
            start_state_ids=_long([0]),
            end_state_ids=_long([1]),
            lambda_exponent=_long([0]),
        ),
        terminal_terms=SubTBTermTable(
            traj_ids=_long([0]),
            start_steps=_long([0]),
            end_steps=_long([2]),
            start_state_ids=_long([0]),
            end_state_ids=_long([2]),
            lambda_exponent=_long([0]),
        ),
    )
    reward = _reward(log_reward=[0.0, 0.0, 1.0], terminal_valid_mask=[True, False, True])
    scores_lo = _scores(
        log_flow=[0.2, 0.7, 1.3],
        terminal_stop_logp_by_traj=[0.4, 0.0],
        stop_log_prob_by_state=[-0.5, -0.4, 0.1],
        forward_prefix_by_traj=[[0.0, 0.5, 0.7], [0.0, 0.0, 0.0]],
        backward_prefix_by_traj=[[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
    )
    scores_hi = replace(scores_lo, log_flow=scores_lo.log_flow.detach().clone())
    scores_lo = replace(scores_lo, log_flow=scores_lo.log_flow.detach().requires_grad_(True))
    scores_hi = replace(scores_hi, log_flow=scores_hi.log_flow.detach().requires_grad_(True))

    output_lo = ForwardLookingSubTBObjective(terminal_loss_weight=1.0)(batch=batch, scores=scores_lo, reward=reward)
    output_hi = ForwardLookingSubTBObjective(terminal_loss_weight=4.0)(batch=batch, scores=scores_hi, reward=reward)
    output_lo.loss.backward()
    output_hi.loss.backward()

    assert scores_lo.log_flow.grad is not None
    assert scores_hi.log_flow.grad is not None
    assert scores_hi.log_flow.grad[0].abs().item() > scores_lo.log_flow.grad[0].abs().item()
    assert scores_hi.log_flow.grad[1].abs().item() < scores_lo.log_flow.grad[1].abs().item()
