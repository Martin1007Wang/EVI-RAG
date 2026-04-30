from __future__ import annotations

import sys
import types
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
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

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
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
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

    def _scatter_softmax(
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
        out = torch.zeros_like(src)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[mask] = torch.softmax(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    torch_scatter_stub.scatter_softmax = _scatter_softmax
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.weaver.losses import LossOutput, SubTrajectoryBalanceLoss
from src.weaver.module import (
    GFlowNetModule,
    TrainingRolloutStats,
    _policy_step_probability_metrics,
    _policy_value_trace_metrics,
    _stop_now_trace_metrics,
)
from src.weaver.nn.action_head import ContinueUtilityAggregator


def _build_module(**kwargs: object) -> GFlowNetModule:
    defaults: dict[str, object] = {
        "entity_text_embeddings": torch.randn(3, 4),
        "entity_embedding_map": torch.tensor([0, 1, 2], dtype=torch.long),
        "relation_embeddings": torch.randn(2, 4),
        "policy_hidden_dim": 4,
        "feature_encoder_cfg": {"embedding_dim": 4},
    }
    defaults.update(kwargs)
    return GFlowNetModule(**defaults)


def _fake_rollout_batch(
    *,
    traj_len: int,
    intervention_count: int,
    reward: float = 0.0,
) -> types.SimpleNamespace:
    proposal = types.SimpleNamespace(
        intervention_count=torch.tensor([intervention_count], dtype=torch.long),
        expand_count=torch.tensor([0], dtype=torch.long),
        stop_count=torch.tensor([0], dtype=torch.long),
        stop_all_targets_covered_count=torch.tensor([0], dtype=torch.long),
        stop_no_budget_feasible_count=torch.tensor([0], dtype=torch.long),
        stop_no_expand_budget_count=torch.tensor([0], dtype=torch.long),
    )
    zeros_bt = torch.zeros((1, 1), dtype=torch.float32)
    stats = types.SimpleNamespace(
        root_log_z=torch.tensor([0.0], dtype=torch.float32),
        trajectory_log_pf=torch.tensor([0.0], dtype=torch.float32),
        trajectory_log_pb=torch.tensor([0.0], dtype=torch.float32),
        terminal_log_reward=torch.tensor([reward], dtype=torch.float32),
        terminal_answer_f1=torch.tensor([0.0], dtype=torch.float32),
        traj_len=torch.tensor([traj_len], dtype=torch.long),
        proposal=proposal,
        policy=types.SimpleNamespace(
            target_stop_prob_sum=torch.tensor([0.0], dtype=torch.float32),
            target_expand_prob_sum=torch.tensor([0.0], dtype=torch.float32),
            target_action_count=torch.tensor([0.0], dtype=torch.float32),
            edge_entropy_sum=torch.tensor([0.0], dtype=torch.float32),
            edge_entropy_count=torch.tensor([0.0], dtype=torch.float32),
        ),
        auxiliary=types.SimpleNamespace(
            edge_auxiliary_loss=torch.tensor(0.0, dtype=torch.float32),
            edge_auxiliary_count=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    traces = types.SimpleNamespace(
        state_log_flows=zeros_bt.clone(),
        step_log_pf=zeros_bt.clone(),
        step_log_pb=zeros_bt.clone(),
        action_type=torch.zeros((1, 1), dtype=torch.long),
        continue_mask=torch.zeros((1, 1), dtype=torch.bool),
        stop_mask=torch.ones((1, 1), dtype=torch.bool),
        selected_edge_ids=torch.full((1, 1), -1, dtype=torch.long),
        proposal_intervention_mask=torch.zeros((1, 1), dtype=torch.bool),
        proposal_expand_mask=torch.zeros((1, 1), dtype=torch.bool),
        proposal_stop_mask=torch.zeros((1, 1), dtype=torch.bool),
        proposal_stop_all_targets_covered_mask=torch.zeros((1, 1), dtype=torch.bool),
        proposal_stop_no_budget_feasible_mask=torch.zeros((1, 1), dtype=torch.bool),
        proposal_stop_no_expand_budget_mask=torch.zeros((1, 1), dtype=torch.bool),
        stop_now_log_reward=zeros_bt.clone(),
        stop_now_answer_f1=zeros_bt.clone(),
        stop_now_answer_precision=zeros_bt.clone(),
        stop_now_answer_recall=zeros_bt.clone(),
        stop_now_expanded_edge_count=zeros_bt.clone(),
        stop_now_edge_penalty=zeros_bt.clone(),
        stop_now_valid_mask=torch.zeros((1, 1), dtype=torch.bool),
        predicted_stop_log_utility=zeros_bt.clone(),
        continue_log_utility=zeros_bt.clone(),
        continue_minus_stop=zeros_bt.clone(),
        type_residual_logit_delta=zeros_bt.clone(),
        type_value_bias_logit_delta=zeros_bt.clone(),
        policy_value_valid_mask=torch.zeros((1, 1), dtype=torch.bool),
        target_stop_prob=zeros_bt.clone(),
        target_continue_prob=zeros_bt.clone(),
        policy_action_valid_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    return types.SimpleNamespace(stats=stats, traces=traces)


def _fake_policy_rollout(
    *,
    target_stop_prob_sum: float,
    target_expand_prob_sum: float,
    target_action_count: float,
    edge_entropy_sum: float,
    edge_entropy_count: float,
    terminal_log_reward: float = 0.0,
    stop_now_log_reward: torch.Tensor | None = None,
    stop_now_valid_mask: torch.Tensor | None = None,
    stop_now_answer_f1: torch.Tensor | None = None,
    stop_now_answer_precision: torch.Tensor | None = None,
    stop_now_answer_recall: torch.Tensor | None = None,
    stop_now_expanded_edge_count: torch.Tensor | None = None,
    stop_now_edge_penalty: torch.Tensor | None = None,
    predicted_stop_log_utility: torch.Tensor | None = None,
    continue_log_utility: torch.Tensor | None = None,
    continue_minus_stop: torch.Tensor | None = None,
    type_residual_logit_delta: torch.Tensor | None = None,
    type_value_bias_logit_delta: torch.Tensor | None = None,
    policy_value_valid_mask: torch.Tensor | None = None,
    target_stop_prob: torch.Tensor | None = None,
    target_continue_prob: torch.Tensor | None = None,
    policy_action_valid_mask: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    if stop_now_log_reward is None:
        stop_now_log_reward = torch.zeros((1, 1), dtype=torch.float32)
    if stop_now_valid_mask is None:
        stop_now_valid_mask = torch.zeros((1, 1), dtype=torch.bool)
    if stop_now_answer_f1 is None:
        stop_now_answer_f1 = torch.zeros_like(stop_now_log_reward)
    if stop_now_answer_precision is None:
        stop_now_answer_precision = torch.zeros_like(stop_now_log_reward)
    if stop_now_answer_recall is None:
        stop_now_answer_recall = torch.zeros_like(stop_now_log_reward)
    if stop_now_expanded_edge_count is None:
        stop_now_expanded_edge_count = torch.zeros_like(stop_now_log_reward)
    if stop_now_edge_penalty is None:
        stop_now_edge_penalty = torch.zeros_like(stop_now_log_reward)
    if predicted_stop_log_utility is None:
        predicted_stop_log_utility = torch.zeros_like(stop_now_log_reward)
    if continue_log_utility is None:
        continue_log_utility = torch.zeros_like(stop_now_log_reward)
    if continue_minus_stop is None:
        continue_minus_stop = torch.zeros_like(stop_now_log_reward)
    if type_residual_logit_delta is None:
        type_residual_logit_delta = torch.zeros_like(stop_now_log_reward)
    if type_value_bias_logit_delta is None:
        type_value_bias_logit_delta = torch.zeros_like(stop_now_log_reward)
    if policy_value_valid_mask is None:
        policy_value_valid_mask = torch.zeros_like(stop_now_valid_mask)
    if target_stop_prob is None:
        target_stop_prob = torch.zeros_like(stop_now_log_reward)
    if target_continue_prob is None:
        target_continue_prob = torch.zeros_like(stop_now_log_reward)
    if policy_action_valid_mask is None:
        policy_action_valid_mask = torch.zeros_like(stop_now_valid_mask)

    return types.SimpleNamespace(
        stats=types.SimpleNamespace(
            terminal_log_reward=torch.tensor([terminal_log_reward], dtype=torch.float32),
            policy=types.SimpleNamespace(
                target_stop_prob_sum=torch.tensor([target_stop_prob_sum], dtype=torch.float32),
                target_expand_prob_sum=torch.tensor([target_expand_prob_sum], dtype=torch.float32),
                target_action_count=torch.tensor([target_action_count], dtype=torch.float32),
                edge_entropy_sum=torch.tensor([edge_entropy_sum], dtype=torch.float32),
                edge_entropy_count=torch.tensor([edge_entropy_count], dtype=torch.float32),
            ),
        ),
        traces=types.SimpleNamespace(
            stop_now_log_reward=stop_now_log_reward,
            stop_now_answer_f1=stop_now_answer_f1,
            stop_now_answer_precision=stop_now_answer_precision,
            stop_now_answer_recall=stop_now_answer_recall,
            stop_now_expanded_edge_count=stop_now_expanded_edge_count,
            stop_now_edge_penalty=stop_now_edge_penalty,
            stop_now_valid_mask=stop_now_valid_mask,
            predicted_stop_log_utility=predicted_stop_log_utility,
            continue_log_utility=continue_log_utility,
            continue_minus_stop=continue_minus_stop,
            type_residual_logit_delta=type_residual_logit_delta,
            type_value_bias_logit_delta=type_value_bias_logit_delta,
            policy_value_valid_mask=policy_value_valid_mask,
            target_stop_prob=target_stop_prob,
            target_continue_prob=target_continue_prob,
            policy_action_valid_mask=policy_action_valid_mask,
        ),
    )


def _loss_rollout_batch(
    *,
    stop_now_log_reward: torch.Tensor,
    stop_now_valid_mask: torch.Tensor,
    predicted_stop_log_utility: torch.Tensor,
    state_log_flows: torch.Tensor | None = None,
    step_log_pf: torch.Tensor | None = None,
    step_log_pb: torch.Tensor | None = None,
    terminal_log_reward: torch.Tensor | None = None,
    traj_len: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    batch_size, horizon = stop_now_log_reward.shape
    zeros_bt = torch.zeros((batch_size, horizon), dtype=torch.float32)
    if state_log_flows is None:
        state_log_flows = zeros_bt.clone()
    if step_log_pf is None:
        step_log_pf = zeros_bt.clone()
    if step_log_pb is None:
        step_log_pb = zeros_bt.clone()
    if terminal_log_reward is None:
        terminal_log_reward = torch.zeros(batch_size, dtype=torch.float32)
    if traj_len is None:
        traj_len = torch.full((batch_size,), horizon, dtype=torch.long)

    return types.SimpleNamespace(
        stats=types.SimpleNamespace(
            traj_len=traj_len,
            terminal_log_reward=terminal_log_reward,
            trajectory_log_pf=torch.zeros(batch_size, dtype=torch.float32),
            trajectory_log_pb=torch.zeros(batch_size, dtype=torch.float32),
            root_log_z=torch.zeros(batch_size, dtype=torch.float32),
            proposal=types.SimpleNamespace(
                intervention_count=torch.zeros(batch_size, dtype=torch.long),
                expand_count=torch.zeros(batch_size, dtype=torch.long),
                stop_count=torch.zeros(batch_size, dtype=torch.long),
                stop_all_targets_covered_count=torch.zeros(batch_size, dtype=torch.long),
                stop_no_budget_feasible_count=torch.zeros(batch_size, dtype=torch.long),
                stop_no_expand_budget_count=torch.zeros(batch_size, dtype=torch.long),
            ),
            policy=types.SimpleNamespace(
                target_stop_prob_sum=torch.zeros(batch_size, dtype=torch.float32),
                target_expand_prob_sum=torch.zeros(batch_size, dtype=torch.float32),
                target_action_count=torch.zeros(batch_size, dtype=torch.float32),
                edge_entropy_sum=torch.zeros(batch_size, dtype=torch.float32),
                edge_entropy_count=torch.zeros(batch_size, dtype=torch.float32),
            ),
            auxiliary=types.SimpleNamespace(
                edge_auxiliary_loss=torch.tensor(0.0, dtype=torch.float32),
                edge_auxiliary_count=torch.tensor(0.0, dtype=torch.float32),
            ),
        ),
        traces=types.SimpleNamespace(
            state_log_flows=state_log_flows,
            step_log_pf=step_log_pf,
            step_log_pb=step_log_pb,
            action_type=torch.zeros((batch_size, horizon), dtype=torch.long),
            continue_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            stop_mask=torch.ones((batch_size, horizon), dtype=torch.bool),
            selected_edge_ids=torch.full((batch_size, horizon), -1, dtype=torch.long),
            proposal_intervention_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            proposal_expand_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            proposal_stop_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            proposal_stop_all_targets_covered_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            proposal_stop_no_budget_feasible_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            proposal_stop_no_expand_budget_mask=torch.zeros((batch_size, horizon), dtype=torch.bool),
            stop_now_log_reward=stop_now_log_reward,
            stop_now_answer_f1=zeros_bt.clone(),
            stop_now_answer_precision=zeros_bt.clone(),
            stop_now_answer_recall=zeros_bt.clone(),
            stop_now_expanded_edge_count=zeros_bt.clone(),
            stop_now_edge_penalty=zeros_bt.clone(),
            stop_now_valid_mask=stop_now_valid_mask,
            predicted_stop_log_utility=predicted_stop_log_utility,
            continue_log_utility=zeros_bt.clone(),
            continue_minus_stop=zeros_bt.clone(),
            type_residual_logit_delta=zeros_bt.clone(),
            type_value_bias_logit_delta=zeros_bt.clone(),
            policy_value_valid_mask=torch.zeros_like(stop_now_valid_mask),
            target_stop_prob=zeros_bt.clone(),
            target_continue_prob=zeros_bt.clone(),
            policy_action_valid_mask=torch.zeros_like(stop_now_valid_mask),
        ),
    )


def test_training_rollouts_split_coverage_and_online_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _build_module(
        num_rollout=8,
        coverage_num_rollout=2,
        rollout_chunk_size=4,
    )
    batch = types.SimpleNamespace(num_graphs=1)

    calls: list[dict[str, object]] = []
    backward_values: list[float] = []

    def _run_exploration(**kwargs: object) -> list[types.SimpleNamespace]:
        calls.append(
            {
                "num_rollouts": kwargs["num_rollouts"],
                "coverage_guide": kwargs["coverage_guide"],
                "proposal_intervention_prob": kwargs["proposal_intervention_prob"],
                "coverage_auxiliary_enabled": kwargs["coverage_auxiliary_enabled"],
            }
        )
        num_rollouts = int(kwargs["num_rollouts"])
        intervention_count = 1 if kwargs["proposal_intervention_prob"] else 0
        return [_fake_rollout_batch(traj_len=1, intervention_count=intervention_count) for _ in range(num_rollouts)]

    def _run_online_vectorized(**kwargs: object) -> list[types.SimpleNamespace]:
        calls.append(
            {
                "num_rollouts": kwargs["num_rollouts"],
                "coverage_guide": None,
                "proposal_intervention_prob": 0.0,
                "coverage_auxiliary_enabled": False,
            }
        )
        num_rollouts = int(kwargs["num_rollouts"])
        return [_fake_rollout_batch(traj_len=1, intervention_count=0) for _ in range(num_rollouts)]

    class _FakeLoss(torch.nn.Module):
        def forward(self, _: object) -> LossOutput:
            loss = torch.tensor(1.0, requires_grad=True)
            return LossOutput(loss=loss, metrics={"loss/total": loss.detach()})

    monkeypatch.setattr(
        module,
        "rollout_engine",
        types.SimpleNamespace(
            expand_budget=module.expand_budget,
            run_exploration=_run_exploration,
            run_online_vectorized=_run_online_vectorized,
        ),
    )
    monkeypatch.setattr(module, "loss_fn", _FakeLoss())
    monkeypatch.setattr(
        module,
        "manual_backward",
        lambda loss: backward_values.append(float(loss.detach().item())),
    )

    _, rollout_stats = module._run_training_rollouts_and_backward(
        batch=batch,
        rollout_temperature=0.8,
        proposal_intervention_prob=0.6,
        accumulation_batches=2,
    )

    assert calls == [
        {
            "num_rollouts": 2,
            "coverage_guide": module.coverage_guide,
            "proposal_intervention_prob": 0.6,
            "coverage_auxiliary_enabled": False,
        },
        {
            "num_rollouts": 4,
            "coverage_guide": None,
            "proposal_intervention_prob": 0.0,
            "coverage_auxiliary_enabled": False,
        },
        {
            "num_rollouts": 2,
            "coverage_guide": None,
            "proposal_intervention_prob": 0.0,
            "coverage_auxiliary_enabled": False,
        },
    ]
    assert backward_values == [0.125, 0.25, 0.125]
    assert float(rollout_stats.pure_proposal_trajectory_count.item()) == 2.0
    assert float(rollout_stats.pure_online_trajectory_count.item()) == 6.0
    assert float(rollout_stats.mixed_trajectory_count.item()) == 0.0
    assert len(rollout_stats.coverage_rollouts) == 2
    assert len(rollout_stats.online_rollouts) == 6


def test_coverage_num_rollout_must_not_exceed_num_rollout() -> None:
    with pytest.raises(
        ValueError,
        match="coverage_num_rollout must be <= num_rollout",
    ):
        _build_module(num_rollout=4, coverage_num_rollout=5)


def test_log_training_metrics_writes_split_rollout_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _build_module(num_rollout=2, coverage_num_rollout=1, expand_budget=3)
    batch = types.SimpleNamespace(num_graphs=1)

    coverage_rollout = _fake_policy_rollout(
        target_stop_prob_sum=2.0,
        target_expand_prob_sum=1.0,
        target_action_count=4.0,
        edge_entropy_sum=0.5,
        edge_entropy_count=1.0,
        terminal_log_reward=0.6,
        stop_now_log_reward=torch.tensor([[0.7, 0.6]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    online_rollout = _fake_policy_rollout(
        target_stop_prob_sum=3.0,
        target_expand_prob_sum=5.0,
        target_action_count=8.0,
        edge_entropy_sum=1.5,
        edge_entropy_count=3.0,
        terminal_log_reward=0.3,
        stop_now_log_reward=torch.tensor([[0.1, 0.3]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )

    rollout_stats = TrainingRolloutStats(
        log_rewards=torch.tensor([0.1, -0.2], dtype=torch.float32),
        trajectory_lengths=torch.tensor([1.0, 4.0], dtype=torch.float32),
        proposal_intervention_counts=torch.tensor([0.0, 2.0], dtype=torch.float32),
        proposal_expand_counts=torch.tensor([0.0, 1.0], dtype=torch.float32),
        proposal_stop_counts=torch.tensor([0.0, 1.0], dtype=torch.float32),
        pure_proposal_trajectory_count=torch.tensor(1.0),
        pure_online_trajectory_count=torch.tensor(1.0),
        mixed_trajectory_count=torch.tensor(0.0),
        coverage_rollouts=(coverage_rollout,),
        online_rollouts=(online_rollout,),
    )

    def _fake_quality(
        rollouts: list[types.SimpleNamespace],
        batch: object | None = None,
        *,
        success_k: int,
        include_full_recall: bool = False,
    ) -> dict[str, float]:
        assert batch is None
        assert not include_full_recall
        if len(rollouts) == 2:
            assert success_k == 2
            return {
                "nonzero_reward_ratio": 0.5,
                "terminal_f1_mean": 0.45,
                "trajectory_count": 2.0,
                "full_recall_rate_at_2": 0.5,
            }
        if rollouts and rollouts[0] is coverage_rollout:
            return {
                "nonzero_reward_ratio": 0.25,
                "terminal_f1_mean": 0.5,
                "trajectory_count": 1.0,
                "full_recall_rate_at_1": 0.75,
            }
        if rollouts and rollouts[0] is online_rollout:
            assert success_k == 8
            return {
                "nonzero_reward_ratio": 0.6,
                "terminal_f1_mean": 0.4,
                "trajectory_count": 1.0,
                "full_recall_rate_at_1": 0.2,
            }
        raise AssertionError("Unexpected rollout group passed to compute_training_rollout_quality")

    logged: dict[str, object] = {}

    monkeypatch.setattr("src.weaver.module.compute_training_rollout_quality", _fake_quality)
    monkeypatch.setattr(module, "log_dict", lambda data, **_: logged.update(data))

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    loss = torch.tensor(1.0)
    loss_output = LossOutput(loss=loss, metrics={"loss/total": loss.detach()})

    module._log_training_metrics(
        batch=batch,
        optimizer=optimizer,
        loss_output=loss_output,
        rollout_stats=rollout_stats,
        rollout_temperature=0.8,
        proposal_intervention_prob=0.3,
    )

    assert logged["train/loss/total"] == 1.0
    assert logged["train/rollout/nonzero_reward_ratio"] == 0.5
    assert logged["train/rollout/terminal_f1_mean"] == 0.45
    assert logged["train/rollout/trajectory_length_mean"] == pytest.approx(2.5)
    assert logged["train/rollout/max_length_ratio"] == pytest.approx(0.5)
    assert logged["train/rollout/coverage/nonzero_reward_ratio"] == 0.25
    assert logged["train/rollout/online/nonzero_reward_ratio"] == 0.6
    assert logged["train/rollout/coverage/terminal_f1_mean"] == 0.5
    assert logged["train/rollout/online/terminal_f1_mean"] == 0.4
    assert logged["train/rollout/coverage/trajectory_count"] == 1.0
    assert logged["train/rollout/online/trajectory_count"] == 1.0
    assert logged["train/proposal/intervention_step_ratio"] == pytest.approx(2.0 / 5.0)
    assert logged["train/proposal/forced_expand_step_ratio"] == pytest.approx(1.0 / 5.0)
    assert logged["train/proposal/forced_stop_step_ratio"] == pytest.approx(1.0 / 5.0)
    assert logged["train/policy/target_stop_prob_mean"] == pytest.approx(5.0 / 12.0)
    assert logged["train/policy/target_expand_prob_mean"] == pytest.approx(0.5)
    assert logged["train/policy/edge_entropy_mean"] == pytest.approx(0.5)
    assert logged["train/sampled_final_vs_stop_now/final_worse_ratio"] == pytest.approx(0.25)
    assert logged["train/sampled_final_vs_stop_now/continue_better_ratio"] == pytest.approx(0.25)
    assert logged["train/sampled_final_vs_stop_now/equal_reward_ratio"] == pytest.approx(0.5)
    assert logged["train/sampled_final_vs_stop_now/mean_continue_minus_stop_log_reward"] == pytest.approx(0.025)
    assert "train/rollout/coverage/full_recall_rate_at_1" not in logged


def test_shared_eval_step_logs_only_default_core_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _build_module()
    batch = types.SimpleNamespace(num_graphs=1)

    results = {
        "sample": {
            "expected_recall": 0.7,
            "expected_nodes": 3.0,
            "dangling_edge_ratio": 0.2,
        },
        "best_of_k": {
            "max_recall_at_1": 0.1,
            "max_recall_at_2": 0.2,
            "max_recall_at_4": 0.4,
            "max_recall_at_8": 0.8,
            "full_recall_rate_at_1": 0.05,
            "full_recall_rate_at_2": 0.15,
            "full_recall_rate_at_4": 0.25,
            "full_recall_rate_at_8": 0.5,
        },
        "diversity": {"edge_jaccard": 0.3},
    }

    val_logged: dict[str, object] = {}
    monkeypatch.setattr(module, "evaluate_subgraph_retrieval", lambda _, **__: results)
    monkeypatch.setattr(module, "log_dict", lambda data, **_: val_logged.update(data))

    module._shared_eval_step(batch=batch, prefix="val")

    assert set(val_logged) == {
        "val/sample/expected_recall",
        "val/sample/expected_nodes",
        "val/best_of_k/max_recall_at_1",
        "val/best_of_k/max_recall_at_4",
        "val/best_of_k/max_recall_at_8",
        "val/best_of_k/full_recall_rate_at_1",
        "val/best_of_k/full_recall_rate_at_4",
        "val/best_of_k/full_recall_rate_at_8",
    }

    test_logged: dict[str, object] = {}
    monkeypatch.setattr(module, "log_dict", lambda data, **_: test_logged.update(data))

    module._shared_eval_step(batch=batch, prefix="test")

    assert set(test_logged) == {
        "test/sample/expected_recall",
        "test/best_of_k/max_recall_at_4",
        "test/best_of_k/full_recall_rate_at_4",
    }


def test_subtrajectory_balance_loss_reports_stop_value_without_adding_to_total() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        max_trajectory_len=1,
    )
    rollout = _loss_rollout_batch(
        stop_now_log_reward=torch.tensor([[-30.0]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True]], dtype=torch.bool),
        predicted_stop_log_utility=torch.tensor([[0.0]], dtype=torch.float32),
    )

    loss_output = loss_fn(rollout)

    assert loss_output.metric("loss/subtb") == pytest.approx(0.0)
    assert loss_output.metric("diagnostic/stop_value") == pytest.approx(19.5)
    assert loss_output.metric("diagnostic/stop_value_count") == pytest.approx(1.0)
    assert loss_output.metric("loss/total") == pytest.approx(0.0)
    assert loss_output.loss.item() == pytest.approx(0.0)


def test_subtrajectory_balance_loss_uses_span_bucket_weights() -> None:
    loss_fn = SubTrajectoryBalanceLoss(max_trajectory_len=2, subtb_lambda=0.5)
    rollout = _loss_rollout_batch(
        stop_now_log_reward=torch.zeros((1, 2), dtype=torch.float32),
        stop_now_valid_mask=torch.zeros((1, 2), dtype=torch.bool),
        predicted_stop_log_utility=torch.zeros((1, 2), dtype=torch.float32),
        step_log_pf=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
    )

    loss_output = loss_fn(rollout)

    assert loss_output.loss.item() == pytest.approx(13.0 / 1.5)


def test_continue_utility_aggregator_uses_logmeanexp_and_negative_empty_default() -> None:
    aggregator = ContinueUtilityAggregator(
        aggregation="logmeanexp",
        no_candidate_value=-10.0,
    )

    output = aggregator(
        candidate_logits=torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32),
        candidate_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        num_graphs=3,
    )

    expected_graph0 = torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0) - torch.log(torch.tensor(2.0))
    expected = torch.tensor([expected_graph0, 2.0, -10.0], dtype=torch.float32)

    assert torch.allclose(output.continue_log_utility, expected)
    assert torch.equal(output.candidate_count, torch.tensor([2.0, 1.0, 0.0]))
    assert torch.equal(output.has_candidate_edge, torch.tensor([True, True, False]))


def test_module_logs_stop_value_diagnostic_and_applies_policy_passthroughs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _build_module(
        continue_utility_cfg={
            "aggregation": "logmeanexp",
            "default_expand_cost": 0.03,
            "no_candidate_value": -10.0,
        },
        stop_utility_head_cfg={"output_init": -1.0},
        action_head_cfg={
            "value_scale_init": 0.2,
            "value_scale_trainable": True,
            "zero_init_residual": True,
        },
    )
    batch = types.SimpleNamespace(num_graphs=1)

    coverage_rollout = _fake_policy_rollout(
        target_stop_prob_sum=1.0,
        target_expand_prob_sum=0.0,
        target_action_count=1.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
    )
    online_rollout = _fake_policy_rollout(
        target_stop_prob_sum=0.0,
        target_expand_prob_sum=1.0,
        target_action_count=1.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
    )
    rollout_stats = TrainingRolloutStats(
        log_rewards=torch.zeros(2, dtype=torch.float32),
        trajectory_lengths=torch.tensor([1, 1], dtype=torch.long),
        proposal_intervention_counts=torch.zeros(2, dtype=torch.long),
        proposal_expand_counts=torch.zeros(2, dtype=torch.long),
        proposal_stop_counts=torch.zeros(2, dtype=torch.long),
        pure_proposal_trajectory_count=torch.tensor(0.0),
        pure_online_trajectory_count=torch.tensor(0.0),
        mixed_trajectory_count=torch.tensor(0.0),
        coverage_rollouts=(coverage_rollout,),
        online_rollouts=(online_rollout,),
    )

    def _fake_quality(*_: object, **__: object) -> dict[str, float]:
        return {
            "nonzero_reward_ratio": 0.0,
            "terminal_f1_mean": 0.0,
            "trajectory_count": 1.0,
        }

    logged: dict[str, object] = {}
    monkeypatch.setattr("src.weaver.module.compute_training_rollout_quality", _fake_quality)
    monkeypatch.setattr(module, "log_dict", lambda data, **_: logged.update(data))

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    loss = torch.tensor(1.0)
    loss_output = LossOutput(
        loss=loss,
        metrics={
            "loss/total": loss.detach(),
            "diagnostic/stop_value": torch.tensor(0.25),
        },
    )

    module._log_training_metrics(
        batch=batch,
        optimizer=optimizer,
        loss_output=loss_output,
        rollout_stats=rollout_stats,
        rollout_temperature=1.0,
        proposal_intervention_prob=0.0,
    )

    assert logged["train/diagnostic/stop_value"] == pytest.approx(0.25)
    assert module.policy.continue_utility.aggregation == "logmeanexp"
    assert module.policy.continue_utility.default_expand_cost == pytest.approx(0.03)
    assert module.policy.continue_utility.no_candidate_value == pytest.approx(-10.0)
    assert module.policy.stop_utility_head.net[-1].bias.detach().item() == pytest.approx(-1.0)
    assert module.policy.stop_continue_head.value_scale.detach().item() == pytest.approx(0.2)


def test_stop_now_trace_metrics_capture_reward_breakdown_means() -> None:
    rollout = _fake_policy_rollout(
        target_stop_prob_sum=0.0,
        target_expand_prob_sum=0.0,
        target_action_count=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        stop_now_log_reward=torch.tensor([[0.4, 0.1]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True, False]], dtype=torch.bool),
        stop_now_answer_f1=torch.tensor([[0.6, 0.0]], dtype=torch.float32),
        stop_now_answer_precision=torch.tensor([[0.75, 0.0]], dtype=torch.float32),
        stop_now_answer_recall=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        stop_now_expanded_edge_count=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        stop_now_edge_penalty=torch.tensor([[0.15, 0.0]], dtype=torch.float32),
    )

    metrics = _stop_now_trace_metrics((rollout,), prefix="train")

    assert metrics["train/sampled_final_vs_stop_now/stop_now_answer_f1_mean"] == pytest.approx(0.6)
    assert metrics["train/sampled_final_vs_stop_now/stop_now_answer_precision_mean"] == pytest.approx(0.75)
    assert metrics["train/sampled_final_vs_stop_now/stop_now_answer_recall_mean"] == pytest.approx(0.5)
    assert metrics["train/sampled_final_vs_stop_now/stop_now_expanded_edge_count_mean"] == pytest.approx(2.0)
    assert metrics["train/sampled_final_vs_stop_now/stop_now_edge_penalty_mean"] == pytest.approx(0.15)


def test_policy_value_trace_metrics_capture_utility_and_residual_diagnostics() -> None:
    rollout = _fake_policy_rollout(
        target_stop_prob_sum=0.0,
        target_expand_prob_sum=0.0,
        target_action_count=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        stop_now_log_reward=torch.tensor([[0.3, 0.2]], dtype=torch.float32),
        stop_now_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
        predicted_stop_log_utility=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        continue_log_utility=torch.tensor([[0.9, 0.0]], dtype=torch.float32),
        continue_minus_stop=torch.tensor([[0.4, 0.0]], dtype=torch.float32),
        type_residual_logit_delta=torch.tensor([[0.1, 0.0]], dtype=torch.float32),
        type_value_bias_logit_delta=torch.tensor([[0.6, 0.0]], dtype=torch.float32),
        policy_value_valid_mask=torch.tensor([[True, False]], dtype=torch.bool),
    )

    metrics = _policy_value_trace_metrics((rollout,), prefix="train")

    assert metrics["train/policy/predicted_stop_log_utility_mean"] == pytest.approx(0.25)
    assert metrics["train/policy/continue_log_utility_mean"] == pytest.approx(0.9)
    assert metrics["train/policy/continue_minus_stop_mean"] == pytest.approx(0.4)
    assert metrics["train/policy/type_residual_logit_delta_mean"] == pytest.approx(0.1)
    assert metrics["train/policy/type_value_bias_logit_delta_mean"] == pytest.approx(0.6)
    assert metrics["train/policy/stop_utility_minus_stop_now_reward_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["train/policy/stop_utility_stop_now_mse"] == pytest.approx(0.04)


def test_policy_step_probability_metrics_aggregate_by_step() -> None:
    rollout_a = _fake_policy_rollout(
        target_stop_prob_sum=0.0,
        target_expand_prob_sum=0.0,
        target_action_count=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        target_stop_prob=torch.tensor([[0.2, 1.0]], dtype=torch.float32),
        target_continue_prob=torch.tensor([[0.8, 0.0]], dtype=torch.float32),
        policy_action_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    rollout_b = _fake_policy_rollout(
        target_stop_prob_sum=0.0,
        target_expand_prob_sum=0.0,
        target_action_count=0.0,
        edge_entropy_sum=0.0,
        edge_entropy_count=0.0,
        target_stop_prob=torch.tensor([[0.6, 0.0]], dtype=torch.float32),
        target_continue_prob=torch.tensor([[0.4, 0.0]], dtype=torch.float32),
        policy_action_valid_mask=torch.tensor([[True, False]], dtype=torch.bool),
    )

    metrics = _policy_step_probability_metrics((rollout_a, rollout_b), prefix="train")

    assert metrics["train/policy/target_stop_prob_t0"] == pytest.approx(0.4)
    assert metrics["train/policy/target_expand_prob_t0"] == pytest.approx(0.6)
    assert metrics["train/policy/target_stop_prob_t1"] == pytest.approx(1.0)
    assert metrics["train/policy/target_expand_prob_t1"] == pytest.approx(0.0)
