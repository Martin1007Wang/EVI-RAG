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
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        if dim == 0:
            out_shape = (size,) + tuple(src.shape[1:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row, dest in enumerate(index.tolist()):
                out[dest] += src[row]
            return out
        raise NotImplementedError("test stub only supports dim=0")

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

from src.weaver.config import build_rollout_runtime_config
from src.weaver.loss import LossOutput
from src.weaver.rollout.engine import RewardMode
from src.weaver.rollout.runner import (
    RolloutRunner,
    backward_rollouts,
    concat_rollout_batches,
    rollout_chunk_sizes,
)
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


def _rollout(batch_size: int = 1, horizon: int = 1) -> RolloutBatch:
    zeros_b = torch.zeros(batch_size, dtype=torch.float32)
    zeros_bt = torch.zeros((batch_size, horizon), dtype=torch.float32)
    bool_bt = torch.zeros((batch_size, horizon), dtype=torch.bool)
    return RolloutBatch(
        stats=RolloutStats(
            root_log_z=zeros_b.clone(),
            trajectory_length=torch.ones(batch_size, dtype=torch.long),
            terminal_log_reward=zeros_b.clone(),
            terminal_answer_f1=zeros_b.clone(),
            edge_action_entropy=zeros_b.clone(),
            edge_action_count=zeros_b.clone(),
        ),
        traces=RolloutTraces(
            state_log_flows=zeros_bt.clone(),
            log_pf=zeros_bt.clone(),
            log_pb=zeros_bt.clone(),
            action_type=torch.zeros((batch_size, horizon), dtype=torch.long),
            continue_mask=bool_bt.clone(),
            stop_mask=bool_bt.clone(),
            selected_edge_ids=torch.full((batch_size, horizon), -1, dtype=torch.long),
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
            stop_adv_loss=zeros_bt.clone(),
            stop_adv_target=zeros_bt.clone(),
            stop_adv_valid_mask=bool_bt.clone(),
            stop_adv_continue_log_reward=zeros_bt.clone(),
            local_improvement_loss=zeros_bt.clone(),
            local_improvement_valid_mask=bool_bt.clone(),
        ),
    )


class _ConstantLoss(torch.nn.Module):
    def __init__(self, value: float, requires_stop_now_reward: bool = False) -> None:
        super().__init__()
        self.value = float(value)
        self.requires_stop_now_reward = bool(requires_stop_now_reward)
        self.seen_batch_sizes: list[int] = []

    def forward(self, rollout: RolloutBatch) -> LossOutput:
        self.seen_batch_sizes.append(int(rollout.stats.trajectory_length.numel()))
        loss = torch.tensor(self.value, dtype=torch.float32, requires_grad=True)
        return LossOutput(loss=loss, metrics={"loss/total": loss.detach()})


def test_rollout_chunk_sizes_cover_total_without_changing_order() -> None:
    assert list(rollout_chunk_sizes(total=8, chunk_size=3)) == [3, 3, 2]
    assert list(rollout_chunk_sizes(total=2, chunk_size=8)) == [2]
    assert list(rollout_chunk_sizes(total=0, chunk_size=8)) == []


def test_backward_rollouts_normalizes_by_full_training_step() -> None:
    loss_fn = _ConstantLoss(2.0)
    backward_values: list[float] = []

    output = backward_rollouts(
        rollouts=[_rollout(), _rollout()],
        loss_fn=loss_fn,
        backward_fn=lambda loss: backward_values.append(float(loss.detach().item())),
        normalize_by=8,
    )

    assert output.metrics["loss/total"].item() == pytest.approx(2.0)
    assert backward_values == [pytest.approx(0.5)]
    assert loss_fn.seen_batch_sizes == [2]


def test_concat_rollout_batches_concatenates_policy_traces() -> None:
    first = _rollout(batch_size=1, horizon=2)
    second = _rollout(batch_size=2, horizon=2)
    second.traces.target_stop_prob[:, 0] = torch.tensor([0.5, 0.7])
    second.traces.policy_action_valid_mask[:, 0] = True

    merged = concat_rollout_batches([first, second])

    assert merged.stats.trajectory_length.shape == (3,)
    assert merged.traces.target_stop_prob.shape == (3, 2)
    assert torch.allclose(
        merged.traces.target_stop_prob[1:, 0],
        torch.tensor([0.5, 0.7]),
    )
    assert bool(merged.traces.policy_action_valid_mask[1:, 0].all())
    assert merged.traces.stop_adv_loss is not None
    assert merged.traces.stop_adv_loss.shape == (3, 2)


def test_rollout_config_rejects_removed_rollout_mode_flags() -> None:
    with pytest.raises(ValueError, match="Unused rollout_cfg keys"):
        build_rollout_runtime_config({"use_static_batch_rollouts": True})

    with pytest.raises(ValueError, match="Unused rollout_cfg keys"):
        build_rollout_runtime_config({"use_fused_static_batch_rollouts": True})


def test_rollout_config_rejects_stop_advantage_until_fused_support_exists() -> None:
    with pytest.raises(ValueError, match="fused-only rollouts"):
        build_rollout_runtime_config({"stop_adv": {"enabled": True}})


def test_rollout_config_allows_local_improvement_auxiliary() -> None:
    cfg = build_rollout_runtime_config(
        {"local_improvement": {"enabled": True, "temperature": 0.7}}
    )

    assert cfg.local_improvement_cfg.enabled
    assert cfg.local_improvement_cfg.temperature == pytest.approx(0.7)


def test_runner_passes_step_auxiliary_only_for_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=3,
        eval_num_rollout=2,
        train_chunk_size=2,
        eval_chunk_size=1,
    )
    loss_fn = _ConstantLoss(1.0)
    backward_values: list[float] = []
    seen_auxiliary: list[object | None] = []

    def _run_vectorized(**kwargs: object) -> list[RolloutBatch]:
        seen_auxiliary.append(kwargs["auxiliary"])
        return [_rollout() for _ in range(int(kwargs["num_rollouts"]))]

    monkeypatch.setattr(runner.engine, "run_vectorized", _run_vectorized)

    auxiliary = object()
    runner.run_training_rollouts_and_backward(
        policy=object(),
        reward_model=object(),
        loss_fn=loss_fn,
        backward_fn=lambda loss: backward_values.append(float(loss.detach().item())),
        batch=object(),
        rollout_temperature=1.0,
        accumulation_batches=1,
        auxiliary=auxiliary,
    )
    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
    )

    assert seen_auxiliary == [auxiliary, auxiliary, None, None]
    assert backward_values == [pytest.approx(2.0 / 3.0), pytest.approx(1.0 / 3.0)]
    assert loss_fn.seen_batch_sizes == [2, 1]


def test_runner_derives_reward_mode_from_declared_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=2,
        eval_num_rollout=2,
        train_chunk_size=1,
        eval_chunk_size=1,
    )
    seen_modes: list[RewardMode] = []

    def _run_vectorized(**kwargs: object) -> list[RolloutBatch]:
        seen_modes.append(kwargs["reward_mode"])
        return [_rollout() for _ in range(int(kwargs["num_rollouts"]))]

    monkeypatch.setattr(runner.engine, "run_vectorized", _run_vectorized)

    runner.run_training_rollouts_and_backward(
        policy=object(),
        reward_model=object(),
        loss_fn=_ConstantLoss(1.0, requires_stop_now_reward=True),
        backward_fn=lambda loss: None,
        batch=object(),
        rollout_temperature=1.0,
        accumulation_batches=1,
        collect_stop_counterfactual=False,
        collect_policy_diagnostics=False,
    )
    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
        collect_stop_counterfactual=False,
        collect_policy_diagnostics=True,
    )
    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
        collect_stop_counterfactual=True,
        collect_policy_diagnostics=False,
    )

    assert seen_modes == [
        RewardMode.EAGER_STOP_NOW,
        RewardMode.EAGER_STOP_NOW,
        RewardMode.LAZY_TERMINAL,
        RewardMode.LAZY_TERMINAL,
        RewardMode.EAGER_STOP_NOW,
        RewardMode.EAGER_STOP_NOW,
    ]


def test_runner_rejects_lazy_reward_mode_when_stop_now_is_required() -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=1,
        eval_num_rollout=1,
        train_chunk_size=1,
        eval_chunk_size=1,
    )

    with pytest.raises(ValueError, match="eager_stop_now"):
        runner.generate_rollouts(
            policy=object(),
            reward_model=object(),
            batch=object(),
            num_rollouts=1,
            temperature=1.0,
            collect_stop_counterfactual=True,
            reward_mode=RewardMode.LAZY_TERMINAL,
        )


def test_runner_uses_fused_only_rollout_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=1,
        eval_num_rollout=2,
        train_chunk_size=1,
        eval_chunk_size=1,
    )
    seen_mode_kwargs: list[set[str]] = []

    def _run_vectorized(**kwargs: object) -> list[RolloutBatch]:
        seen_mode_kwargs.append(
            {
                key
                for key in kwargs
                if key
                in {"use_static_batch_rollouts", "use_fused_static_batch_rollouts"}
            }
        )
        return [_rollout() for _ in range(int(kwargs["num_rollouts"]))]

    monkeypatch.setattr(runner.engine, "run_vectorized", _run_vectorized)

    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
    )

    assert seen_mode_kwargs == [set(), set()]
