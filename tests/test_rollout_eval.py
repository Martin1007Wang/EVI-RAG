from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        if dim == 0:
            out_shape = (size,) + tuple(src.shape[1:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row, dest in enumerate(index.tolist()):
                out[dest] += src[row]
            return out
        if dim == 1:
            out_shape = (src.shape[0], size) + tuple(src.shape[2:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row in range(src.shape[0]):
                for col, dest in enumerate(index[row].tolist()):
                    out[row, dest] += src[row, col]
            return out
        raise NotImplementedError("test stub only supports dim=0 or dim=1")

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -float("inf"), dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.data.collate import RetrievalCollator
from src.data.schema import RetrievalData
from src.training.diagnostics import TrainingDiagnosticsCollector
from src.training.rollout_diagnostics import (
    compute_after_hit_diagnostics,
    compute_stop_and_teacher_diagnostics,
    compute_stop_behavior_diagnostics,
)
from src.training.rollout_eval import evaluate_rollouts
from src.weaver.loss import LossOutput
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


@dataclass
class FakeLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _sample() -> RetrievalData:
    return RetrievalData(
        num_nodes=4,
        edge_index=torch.tensor([[0, 1, 0, 3], [1, 2, 3, 1]], dtype=torch.long),
        node_entity_catalog_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
        anchor_node_forward_distances_flat=torch.tensor([0, 1, 2, 1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor(
            [0, -1, -1, -1], dtype=torch.long
        ),
        node_target_distance=torch.tensor([2, 1, 0, 2], dtype=torch.long),
        target_node_distances_flat=torch.tensor([2, 1, 0, 2], dtype=torch.long),
        target_shortest_path_count_flat=torch.tensor(
            [1.0, 1.0, 1.0, 1.0], dtype=torch.float32
        ),
        target_shortest_path_edge_mask_flat=torch.tensor(
            [True, True, False, False],
            dtype=torch.bool,
        ),
        non_text_node_mask=torch.tensor([False, False, True, True], dtype=torch.bool),
    )


def _reverse_frontier_sample() -> RetrievalData:
    return RetrievalData(
        num_nodes=2,
        edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
        anchor_node_forward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, 1], dtype=torch.long),
        node_target_distance=torch.tensor([1, 0], dtype=torch.long),
        target_node_distances_flat=torch.tensor([1, 0], dtype=torch.long),
        target_shortest_path_count_flat=torch.tensor(
            [1.0, 1.0],
            dtype=torch.float32,
        ),
        target_shortest_path_edge_mask_flat=torch.tensor([True], dtype=torch.bool),
        non_text_node_mask=torch.tensor([False, False], dtype=torch.bool),
    )


def _batch():
    return RetrievalCollator()([_sample()])


def _rollout(
    *,
    expand_edges: list[int],
    terminal_answer_f1: float,
    edge_entropy_sum: float,
    edge_entropy_count: float,
    terminal_log_reward: float = 0.0,
    horizon: int = 3,
    terminal_complexity_penalty: float | None = None,
    terminal_base_log_reward: float | None = None,
    terminal_utility: float | None = None,
    terminal_expanded_edge_count: float | None = None,
    terminal_answer_degree_excess: float | None = None,
    stop_now_answer_f1: torch.Tensor | None = None,
    stop_now_valid_mask: torch.Tensor | None = None,
    target_stop_prob: torch.Tensor | None = None,
    policy_action_valid_mask: torch.Tensor | None = None,
    budget_exhausted_mask: torch.Tensor | None = None,
) -> RolloutBatch:
    trajectory_length = len(expand_edges) + 1
    selected_edge_ids = torch.full((1, horizon), -1, dtype=torch.long)
    continue_mask = torch.zeros((1, horizon), dtype=torch.bool)
    stop_mask = torch.zeros((1, horizon), dtype=torch.bool)

    for step_id, edge_id in enumerate(expand_edges):
        selected_edge_ids[0, step_id] = int(edge_id)
        continue_mask[0, step_id] = True

    stop_mask[0, trajectory_length - 1] = True

    zeros_bt = torch.zeros((1, horizon), dtype=torch.float32)
    bool_bt = torch.zeros((1, horizon), dtype=torch.bool)
    entropy = zeros_bt.clone()
    entropy_valid = bool_bt.clone()
    if edge_entropy_count > 0.0:
        count = min(int(edge_entropy_count), horizon)
        entropy[0, 0:count] = float(edge_entropy_sum) / float(edge_entropy_count)
        entropy_valid[0, 0:count] = True
    if stop_now_answer_f1 is None:
        stop_now_answer_f1 = zeros_bt.clone()
    if stop_now_valid_mask is None:
        stop_now_valid_mask = bool_bt.clone()
    if target_stop_prob is None:
        target_stop_prob = zeros_bt.clone()
    if policy_action_valid_mask is None:
        policy_action_valid_mask = bool_bt.clone()
    if budget_exhausted_mask is None:
        budget_exhausted_mask = bool_bt.clone()

    return RolloutBatch(
        stats=RolloutStats(
            root_log_z=torch.zeros(1, dtype=torch.float32),
            trajectory_length=torch.tensor([trajectory_length], dtype=torch.long),
            terminal_log_reward=torch.tensor(
                [terminal_log_reward], dtype=torch.float32
            ),
            terminal_answer_f1=torch.tensor([terminal_answer_f1], dtype=torch.float32),
            edge_action_entropy=torch.tensor([edge_entropy_sum], dtype=torch.float32),
            edge_action_count=torch.tensor([edge_entropy_count], dtype=torch.float32),
            terminal_complexity_penalty=(
                None
                if terminal_complexity_penalty is None
                else torch.tensor([terminal_complexity_penalty], dtype=torch.float32)
            ),
            terminal_base_log_reward=(
                None
                if terminal_base_log_reward is None
                else torch.tensor([terminal_base_log_reward], dtype=torch.float32)
            ),
            terminal_utility=(
                None
                if terminal_utility is None
                else torch.tensor([terminal_utility], dtype=torch.float32)
            ),
            terminal_expanded_edge_count=(
                None
                if terminal_expanded_edge_count is None
                else torch.tensor([terminal_expanded_edge_count], dtype=torch.float32)
            ),
            terminal_answer_degree_excess=(
                None
                if terminal_answer_degree_excess is None
                else torch.tensor([terminal_answer_degree_excess], dtype=torch.float32)
            ),
        ),
        traces=RolloutTraces(
            state_log_flows=zeros_bt.clone(),
            log_pf=zeros_bt.clone(),
            log_pb=zeros_bt.clone(),
            action_type=torch.full((1, horizon), -1, dtype=torch.long),
            continue_mask=continue_mask,
            stop_mask=stop_mask,
            selected_edge_ids=selected_edge_ids,
            stop_now_log_reward=zeros_bt.clone(),
            stop_now_answer_f1=stop_now_answer_f1,
            stop_now_valid_mask=stop_now_valid_mask,
            stop_log_pf=zeros_bt.clone(),
            stop_tb_valid_mask=bool_bt.clone(),
            target_stop_prob=target_stop_prob,
            target_continue_prob=zeros_bt.clone(),
            policy_action_valid_mask=policy_action_valid_mask,
            edge_action_entropy=entropy,
            edge_action_entropy_valid_mask=entropy_valid,
            budget_exhausted_mask=budget_exhausted_mask,
        ),
    )


def _batched_rollout(
    *,
    expand_edges: list[list[int]],
    horizon: int = 3,
) -> RolloutBatch:
    batch_size = len(expand_edges)
    selected_edge_ids = torch.full((batch_size, horizon), -1, dtype=torch.long)
    continue_mask = torch.zeros((batch_size, horizon), dtype=torch.bool)
    stop_mask = torch.zeros((batch_size, horizon), dtype=torch.bool)

    for row, row_edges in enumerate(expand_edges):
        trajectory_length = len(row_edges) + 1
        for step_id, edge_id in enumerate(row_edges):
            selected_edge_ids[row, step_id] = int(edge_id)
            continue_mask[row, step_id] = True
        stop_mask[row, trajectory_length - 1] = True

    zeros_bt = torch.zeros((batch_size, horizon), dtype=torch.float32)
    bool_bt = torch.zeros((batch_size, horizon), dtype=torch.bool)

    return RolloutBatch(
        stats=RolloutStats(
            root_log_z=torch.zeros(batch_size, dtype=torch.float32),
            trajectory_length=torch.tensor(
                [len(row_edges) + 1 for row_edges in expand_edges],
                dtype=torch.long,
            ),
            terminal_log_reward=torch.zeros(batch_size, dtype=torch.float32),
            terminal_answer_f1=torch.zeros(batch_size, dtype=torch.float32),
            edge_action_entropy=torch.zeros(batch_size, dtype=torch.float32),
            edge_action_count=torch.zeros(batch_size, dtype=torch.float32),
        ),
        traces=RolloutTraces(
            state_log_flows=zeros_bt.clone(),
            log_pf=zeros_bt.clone(),
            log_pb=zeros_bt.clone(),
            action_type=torch.full((batch_size, horizon), -1, dtype=torch.long),
            continue_mask=continue_mask,
            stop_mask=stop_mask,
            selected_edge_ids=selected_edge_ids,
            stop_now_log_reward=zeros_bt.clone(),
            stop_now_answer_f1=zeros_bt.clone(),
            stop_now_valid_mask=bool_bt.clone(),
            stop_log_pf=zeros_bt.clone(),
            stop_tb_valid_mask=bool_bt.clone(),
            target_stop_prob=zeros_bt.clone(),
            target_continue_prob=zeros_bt.clone(),
            policy_action_valid_mask=bool_bt.clone(),
            edge_action_entropy=zeros_bt.clone(),
            edge_action_entropy_valid_mask=bool_bt.clone(),
        ),
    )


def test_evaluate_rollouts_reports_nonzero_f1_and_rollout_diagnostics() -> None:
    batch = _batch()
    bad_rollout = _rollout(
        expand_edges=[2],
        terminal_answer_f1=0.0,
        edge_entropy_sum=1.5,
        edge_entropy_count=1.0,
    )
    good_rollout = _rollout(
        expand_edges=[0, 1],
        terminal_answer_f1=1.0,
        edge_entropy_sum=0.5,
        edge_entropy_count=1.0,
    )

    metrics = evaluate_rollouts(
        rollouts=[bad_rollout, good_rollout],
        batch=batch,
        eval_budgets=[1, 2],
        debug_metrics=False,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
        stage="val",
    )

    assert metrics["sample"]["nonzero_f1_rate"] == pytest.approx(0.5)
    assert metrics["best_of_k"]["nonzero_f1_rate_at_1"] == pytest.approx(0.0)
    assert metrics["best_of_k"]["nonzero_f1_rate_at_2"] == pytest.approx(1.0)
    assert metrics["best_of_k"]["f1_gain_at_2"] == pytest.approx(1.0 / 3.0)
    assert metrics["sample_all_answers"]["expected_target_recall"] == pytest.approx(
        metrics["sample"]["expected_target_recall"]
    )
    assert metrics["best_of_k_all_answers"]["max_target_recall_at_2"] == pytest.approx(
        metrics["best_of_k"]["max_target_recall_at_2"]
    )
    assert metrics["policy"]["edge_action_entropy_mean"] == pytest.approx(1.0)
    assert metrics["diversity"]["unique_selected_edge_set_rate_at_2"] == pytest.approx(
        1.0
    )


def test_training_diagnostics_collects_teacher_edge_metrics() -> None:
    batch = _batch()
    bad_rollout = _rollout(
        expand_edges=[2],
        terminal_answer_f1=0.0,
        edge_entropy_sum=1.5,
        edge_entropy_count=1.0,
    )
    good_rollout = _rollout(
        expand_edges=[0, 1],
        terminal_answer_f1=1.0,
        edge_entropy_sum=0.5,
        edge_entropy_count=1.0,
    )
    collector = TrainingDiagnosticsCollector(debug=False)

    metrics = collector.collect(
        loss_output=FakeLossOutput(
            loss=torch.tensor(1.0),
            metrics={"loss/total": torch.tensor(1.0)},
        ),
        batch=batch,
        online_rollouts=(bad_rollout, good_rollout),
    )

    assert metrics["train/reward/nonzero_f1_rate"] == pytest.approx(0.5)
    assert metrics["train/policy/edge_action_entropy_mean"] == pytest.approx(1.0)
    assert metrics["train/diversity/unique_terminal_subgraph_rate"] == pytest.approx(
        1.0
    )
    assert metrics["train/diversity/unique_selected_edge_set_rate"] == pytest.approx(
        1.0
    )
    assert metrics[
        "train/teacher_edge/selected_edge_on_target_shortest_path_rate"
    ] == pytest.approx(2.0 / 3.0)
    assert metrics[
        "train/teacher_edge/selected_edge_reduces_target_distance_rate"
    ] == pytest.approx(2.0 / 3.0)
    assert metrics[
        "train/teacher_edge/trajectory_any_shortest_path_hit_rate"
    ] == pytest.approx(0.5)
    assert metrics["train/teacher_edge/trajectory_any_progress_rate"] == pytest.approx(
        0.5
    )


def test_training_diagnostics_always_collects_core_reward_health_metrics() -> None:
    batch = _batch()
    bad_rollout = _rollout(
        expand_edges=[2],
        terminal_answer_f1=0.0,
        terminal_log_reward=-10.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
    )
    good_rollout = _rollout(
        expand_edges=[0, 1],
        terminal_answer_f1=1.0,
        terminal_log_reward=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
    )
    collector = TrainingDiagnosticsCollector(
        debug=False,
        rollout_diagnostics=False,
    )

    metrics = collector.collect(
        loss_output=FakeLossOutput(
            loss=torch.tensor(1.0),
            metrics={"loss/total": torch.tensor(1.0)},
        ),
        batch=batch,
        online_rollouts=(bad_rollout, good_rollout),
    )

    assert metrics["train/reward/log_reward_mean"] == pytest.approx(-5.0)
    assert metrics["train/reward/log_reward_std"] == pytest.approx(5.0)
    assert metrics["train/reward/log_reward_p90"] == pytest.approx(-1.0)
    assert metrics["train/reward/log_reward_max"] == pytest.approx(0.0)
    assert metrics["train/reward/nonzero_f1_rate"] == pytest.approx(0.5)
    assert metrics["train/reward/terminal_answer_f1_mean"] == pytest.approx(0.5)
    assert "train/policy/edge_action_entropy_mean" not in metrics


def test_training_diagnostics_collects_reward_breakdown_metrics() -> None:
    batch = _batch()
    rollout_a = _rollout(
        expand_edges=[2],
        terminal_answer_f1=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        terminal_complexity_penalty=0.1,
        terminal_base_log_reward=-9.6,
        terminal_utility=0.2,
        terminal_expanded_edge_count=3.0,
        terminal_answer_degree_excess=0.15,
    )
    rollout_b = _rollout(
        expand_edges=[0],
        terminal_answer_f1=1.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        terminal_complexity_penalty=0.3,
        terminal_base_log_reward=0.0,
        terminal_utility=1.0,
        terminal_expanded_edge_count=1.0,
        terminal_answer_degree_excess=0.0,
    )
    collector = TrainingDiagnosticsCollector(debug=False)

    metrics = collector.collect(
        loss_output=FakeLossOutput(
            loss=torch.tensor(1.0),
            metrics={"loss/total": torch.tensor(1.0)},
        ),
        batch=batch,
        online_rollouts=(rollout_a, rollout_b),
    )

    assert "train/reward/semantic_score_mean" not in metrics
    assert "train/reward/semantic_bonus_mean" not in metrics
    assert metrics["train/reward/positive_answer_f1_mean"] == pytest.approx(1.0)
    assert metrics["train/reward/base_log_reward_mean"] == pytest.approx(-4.8)
    assert metrics["train/reward/complexity_penalty_mean"] == pytest.approx(0.2)
    assert metrics["train/reward/utility_mean"] == pytest.approx(0.6)
    assert metrics["train/reward/supported_answer_recall_mean"] == pytest.approx(0.6)
    assert metrics["train/reward/expanded_edge_count_mean"] == pytest.approx(2.0)
    assert metrics["train/reward/answer_degree_excess_mean"] == pytest.approx(0.075)


def test_after_hit_diagnostics_split_continue_and_stop_probability() -> None:
    rollout = _rollout(
        expand_edges=[0, 1],
        terminal_answer_f1=1.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        stop_now_answer_f1=torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True, True, True]], dtype=torch.bool),
        target_stop_prob=torch.tensor([[0.1, 0.7, 0.9]], dtype=torch.float32),
        policy_action_valid_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
    )

    metrics = compute_after_hit_diagnostics((rollout,))

    assert metrics["rollout/continue_after_first_hit_rate"] == pytest.approx(0.5)
    assert metrics["rollout/extra_edges_after_first_hit"] == pytest.approx(1.0)
    assert metrics["policy/stop_prob_before_hit"] == pytest.approx(0.1)
    assert metrics["policy/stop_prob_after_hit"] == pytest.approx(0.7)


def test_stop_behavior_diagnostics_split_model_and_forced_stop_reasons() -> None:
    model_stop = _rollout(
        expand_edges=[],
        terminal_answer_f1=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        horizon=1,
        policy_action_valid_mask=torch.tensor([[True]], dtype=torch.bool),
        budget_exhausted_mask=torch.tensor([[False]], dtype=torch.bool),
    )
    budget_forced_stop = _rollout(
        expand_edges=[],
        terminal_answer_f1=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        horizon=1,
        policy_action_valid_mask=torch.tensor([[False]], dtype=torch.bool),
        budget_exhausted_mask=torch.tensor([[True]], dtype=torch.bool),
    )
    no_frontier_forced_stop = _rollout(
        expand_edges=[],
        terminal_answer_f1=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        horizon=1,
        policy_action_valid_mask=torch.tensor([[False]], dtype=torch.bool),
        budget_exhausted_mask=torch.tensor([[False]], dtype=torch.bool),
    )

    metrics = compute_stop_behavior_diagnostics(
        (model_stop, budget_forced_stop, no_frontier_forced_stop)
    )

    assert metrics["rollout/model_stop_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["rollout/forced_stop_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout/budget_exhausted_stop_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["rollout/no_frontier_stop_rate"] == pytest.approx(1.0 / 3.0)


def test_teacher_edge_diagnostics_uses_local_nodes_for_batched_graphs() -> None:
    batch = RetrievalCollator()([_sample(), _sample()])
    rollout = _batched_rollout(expand_edges=[[0], [4]])

    metrics = compute_stop_and_teacher_diagnostics((rollout,), batch=batch)

    assert metrics[
        "teacher_edge/selected_edge_on_target_shortest_path_rate"
    ] == pytest.approx(1.0)
    assert metrics[
        "teacher_edge/selected_edge_reduces_target_distance_rate"
    ] == pytest.approx(1.0)


def test_teacher_edge_diagnostics_accepts_reverse_frontier_expansion() -> None:
    batch = RetrievalCollator()([_reverse_frontier_sample()])
    rollout = _rollout(
        expand_edges=[0],
        terminal_answer_f1=1.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
    )

    metrics = compute_stop_and_teacher_diagnostics((rollout,), batch=batch)

    assert metrics[
        "teacher_edge/selected_edge_on_target_shortest_path_rate"
    ] == pytest.approx(1.0)
    assert metrics[
        "teacher_edge/selected_edge_reduces_target_distance_rate"
    ] == pytest.approx(1.0)
