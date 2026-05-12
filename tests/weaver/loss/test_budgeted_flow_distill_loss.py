from __future__ import annotations

import torch

from src.weaver.loss.budgeted_flow_distill import BudgetedFlowDistillLoss
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


def test_budgeted_flow_distill_loss_aggregates_components() -> None:
    traces = _base_traces()
    rollout = RolloutBatch(
        stats=RolloutStats(
            trajectory_length=torch.ones(2, dtype=torch.long),
            terminal_log_reward=torch.zeros(2),
            terminal_answer_f1=torch.zeros(2),
            edge_action_entropy=torch.zeros(2),
            edge_action_count=torch.zeros(2),
        ),
        traces=traces,
    )
    loss = BudgetedFlowDistillLoss(
        policy_kl_weight=1.0,
        terminal_weight=1.0,
        value_weight=0.5,
    )(rollout)

    assert torch.isclose(loss.metrics["loss/policy_kl"], torch.tensor(1.5))
    assert torch.isclose(loss.metrics["loss/terminal_huber"], torch.tensor(0.75))
    assert torch.isclose(loss.metrics["loss/value_huber"], torch.tensor(0.25))
    assert torch.isclose(loss.loss, torch.tensor(2.375))


def _base_traces() -> RolloutTraces:
    bt = (2, 1)
    zeros = torch.zeros(bt)
    bools = torch.zeros(bt, dtype=torch.bool)
    return RolloutTraces(
        log_pf=zeros,
        log_pb=zeros,
        state_log_flow=zeros,
        db_parent_log_reward=zeros,
        db_child_log_reward=zeros,
        db_parent_shortest_path_potential=zeros,
        db_child_shortest_path_potential=zeros,
        db_parent_process_log_bonus=zeros,
        db_child_process_log_bonus=zeros,
        db_log_p_stop_parent=zeros,
        db_log_p_stop_child=zeros,
        db_log_pf_expand=zeros,
        db_log_pb=zeros,
        db_valid_mask=bools,
        action_type=torch.zeros(bt, dtype=torch.long),
        continue_mask=bools,
        stop_mask=bools,
        selected_edge_ids=torch.full(bt, -1, dtype=torch.long),
        target_stop_prob=zeros,
        target_continue_prob=zeros,
        policy_action_valid_mask=torch.ones(bt, dtype=torch.bool),
        edge_action_entropy=zeros,
        edge_action_entropy_valid_mask=bools,
        budgeted_policy_kl=torch.tensor([[1.0], [2.0]]),
        budgeted_terminal_loss=torch.tensor([[0.5], [1.0]]),
        budgeted_value_loss=torch.tensor([[0.0], [0.5]]),
        budgeted_valid_mask=torch.ones(bt, dtype=torch.bool),
        oracle_v_star=zeros,
        oracle_terminal_j=zeros,
        oracle_stop_prob=zeros,
        oracle_edge_entropy=zeros,
        model_stop_prob=zeros,
        budgeted_oracle_good_edge_policy_mass=zeros,
        sampled_oracle_good_edge_rate=zeros,
    )
