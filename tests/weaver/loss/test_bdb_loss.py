from __future__ import annotations

import pytest
import torch

from src.weaver.loss.bdb import BudgetedDAGDetailedBalanceLoss
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


def _rollout_with_bdb_traces() -> RolloutBatch:
    b, t = 2, 2
    zeros = torch.zeros((b, t), dtype=torch.float32)
    bools = torch.zeros((b, t), dtype=torch.bool)
    longs = torch.full((b, t), -1, dtype=torch.long)
    return RolloutBatch(
        stats=RolloutStats(
            trajectory_length=torch.ones(b),
            terminal_log_reward=torch.zeros(b),
            terminal_answer_f1=torch.zeros(b),
            edge_action_entropy=torch.zeros(b),
            edge_action_count=torch.zeros(b),
        ),
        traces=RolloutTraces(
            log_pf=zeros.clone(),
            log_pb=zeros.clone(),
            state_log_flow=zeros.clone(),
            db_parent_log_reward=zeros.clone(),
            db_child_log_reward=zeros.clone(),
            db_parent_shortest_path_potential=zeros.clone(),
            db_child_shortest_path_potential=zeros.clone(),
            db_parent_process_log_bonus=zeros.clone(),
            db_child_process_log_bonus=zeros.clone(),
            db_log_p_stop_parent=zeros.clone(),
            db_log_p_stop_child=zeros.clone(),
            db_log_pf_expand=zeros.clone(),
            db_log_pb=zeros.clone(),
            db_valid_mask=bools.clone(),
            action_type=longs.clone(),
            continue_mask=bools.clone(),
            stop_mask=bools.clone(),
            selected_edge_ids=longs.clone(),
            target_stop_prob=zeros.clone(),
            target_continue_prob=zeros.clone(),
            policy_action_valid_mask=bools.clone(),
            edge_action_entropy=zeros.clone(),
            edge_action_entropy_valid_mask=bools.clone(),
            bdb_stop_loss=torch.tensor([[1.0, 9.0], [0.0, 0.0]]),
            bdb_edge_loss=torch.tensor([[4.0, 0.0], [16.0, 0.0]]),
            bdb_base_loss=torch.tensor([[0.0, 25.0], [0.0, 0.0]]),
            bdb_stop_valid_mask=torch.tensor(
                [[True, False], [False, False]],
            ),
            bdb_edge_valid_mask=torch.tensor(
                [[True, False], [True, False]],
            ),
            bdb_base_valid_mask=torch.tensor(
                [[False, True], [False, False]],
            ),
            bdb_delta_stop=torch.tensor([[1.0, 3.0], [0.0, 0.0]]),
            bdb_delta_edge=torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
            bdb_delta_base=torch.tensor([[0.0, 5.0], [0.0, 0.0]]),
            bdb_frontier_size=torch.tensor([[3.0, 0.0], [2.0, 0.0]]),
            bdb_parent_count=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            bdb_log_reward=torch.tensor([[0.5, 0.7], [0.9, 0.0]]),
            bdb_log_flow=torch.tensor([[1.5, 1.7], [1.9, 0.0]]),
        ),
    )


def test_bdb_loss_aggregates_weighted_components_and_metrics() -> None:
    rollout = _rollout_with_bdb_traces()
    loss_fn = BudgetedDAGDetailedBalanceLoss()

    output = loss_fn(rollout)

    assert output.loss.item() == pytest.approx((5.0 + 25.0 + 16.0) / 3.0)
    assert output.metrics["loss/total"].item() == pytest.approx(output.loss.item())
    assert output.metrics["loss/bdb"].item() == pytest.approx(output.loss.item())
    assert output.metrics["bdb/loss_delta0_mean"].item() == pytest.approx(1.0)
    assert output.metrics["bdb/loss_edge_residual_mean"].item() == pytest.approx(10.0)
    assert output.metrics["bdb/loss_forced_terminal_mean"].item() == pytest.approx(25.0)
    assert output.metrics["bdb/delta_edge_mean"].item() == pytest.approx(3.0)
    assert output.metrics["bdb/base_state_rate"].item() == pytest.approx(0.25)
    assert output.metrics["bdb/mean_frontier_size"].item() == pytest.approx(
        (3.0 + 0.0 + 2.0) / 3.0
    )
    assert output.metrics["bdb/mean_parent_count"].item() == pytest.approx(1.5)
    assert output.metrics["reward/log_reward_mean"].item() == pytest.approx(
        (0.5 + 0.7 + 0.9) / 3.0
    )
    assert output.metrics["flow/log_flow_mean"].item() == pytest.approx(
        (1.5 + 1.7 + 1.9) / 3.0
    )


def test_bdb_loss_rejects_reserved_modes() -> None:
    with pytest.raises(ValueError, match="edge_mode='full'"):
        BudgetedDAGDetailedBalanceLoss(edge_mode="sample")
    with pytest.raises(ValueError, match="child_flow_target='detach_current'"):
        BudgetedDAGDetailedBalanceLoss(child_flow_target="target_network")
    with pytest.raises(ValueError, match="backward_kernel='uniform_boundary'"):
        BudgetedDAGDetailedBalanceLoss(backward_kernel="indegree")
