from __future__ import annotations

import sys
import types
import dataclasses
from pathlib import Path

import pytest
import torch

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

from src.weaver.config import (
    build_eval_runtime_config,
    build_loss_config,
    build_policy_runtime_config,
    build_reward_config,
    build_rollout_runtime_config,
    validate_algorithm_coupling,
)
from src.weaver.loss import BudgetedDAGDetailedBalanceLoss, LossOutput
from src.weaver.module import WeaverModule, build_loss
from src.weaver.rollout.runner import (
    RolloutRunner,
    TrainingRolloutChunk,
    concat_rollout_batches,
    detach_rollout_for_metrics,
    rollout_chunk_sizes,
)
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


def _rollout(batch_size: int = 1, horizon: int = 1) -> RolloutBatch:
    zeros_b = torch.zeros(batch_size, dtype=torch.float32)
    zeros_bt = torch.zeros((batch_size, horizon), dtype=torch.float32)
    bool_bt = torch.zeros((batch_size, horizon), dtype=torch.bool)
    return RolloutBatch(
        stats=RolloutStats(
            trajectory_length=torch.ones(batch_size, dtype=torch.long),
            terminal_log_reward=zeros_b.clone(),
            terminal_answer_f1=zeros_b.clone(),
            edge_action_entropy=zeros_b.clone(),
            edge_action_count=zeros_b.clone(),
        ),
        traces=RolloutTraces(
            log_pf=zeros_bt.clone(),
            log_pb=zeros_bt.clone(),
            state_log_flow=zeros_bt.clone(),
            db_parent_log_reward=zeros_bt.clone(),
            db_child_log_reward=zeros_bt.clone(),
            db_parent_shortest_path_potential=zeros_bt.clone(),
            db_child_shortest_path_potential=zeros_bt.clone(),
            db_parent_process_log_bonus=zeros_bt.clone(),
            db_child_process_log_bonus=zeros_bt.clone(),
            db_log_p_stop_parent=zeros_bt.clone(),
            db_log_p_stop_child=zeros_bt.clone(),
            db_log_pf_expand=zeros_bt.clone(),
            db_log_pb=zeros_bt.clone(),
            db_valid_mask=bool_bt.clone(),
            action_type=torch.zeros((batch_size, horizon), dtype=torch.long),
            continue_mask=bool_bt.clone(),
            stop_mask=bool_bt.clone(),
            selected_edge_ids=torch.full((batch_size, horizon), -1, dtype=torch.long),
            stop_now_log_reward=zeros_bt.clone(),
            stop_now_answer_f1=zeros_bt.clone(),
            stop_now_valid_mask=bool_bt.clone(),
            target_stop_prob=zeros_bt.clone(),
            target_continue_prob=zeros_bt.clone(),
            policy_action_valid_mask=bool_bt.clone(),
            edge_action_entropy=zeros_bt.clone(),
            edge_action_entropy_valid_mask=bool_bt.clone(),
        ),
    )


def _rollout_with_autograd_traces(
    batch_size: int = 1,
    horizon: int = 1,
) -> RolloutBatch:
    def grad_tensor(*shape: int) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.float32, requires_grad=True) * 2.0

    bool_bt = torch.zeros((batch_size, horizon), dtype=torch.bool)
    return RolloutBatch(
        stats=RolloutStats(
            trajectory_length=torch.ones(batch_size, dtype=torch.long),
            terminal_log_reward=grad_tensor(batch_size),
            terminal_answer_f1=grad_tensor(batch_size),
            edge_action_entropy=grad_tensor(batch_size),
            edge_action_count=grad_tensor(batch_size),
            terminal_complexity_penalty=grad_tensor(batch_size),
            terminal_base_log_reward=grad_tensor(batch_size),
            terminal_utility=grad_tensor(batch_size),
            terminal_shortest_path_potential=grad_tensor(batch_size),
            terminal_expanded_edge_count=grad_tensor(batch_size),
            terminal_answer_degree_excess=grad_tensor(batch_size),
        ),
        traces=RolloutTraces(
            log_pf=grad_tensor(batch_size, horizon),
            log_pb=grad_tensor(batch_size, horizon),
            state_log_flow=grad_tensor(batch_size, horizon),
            db_parent_log_reward=grad_tensor(batch_size, horizon),
            db_child_log_reward=grad_tensor(batch_size, horizon),
            db_parent_shortest_path_potential=grad_tensor(batch_size, horizon),
            db_child_shortest_path_potential=grad_tensor(batch_size, horizon),
            db_parent_process_log_bonus=grad_tensor(batch_size, horizon),
            db_child_process_log_bonus=grad_tensor(batch_size, horizon),
            db_log_p_stop_parent=grad_tensor(batch_size, horizon),
            db_log_p_stop_child=grad_tensor(batch_size, horizon),
            db_log_pf_expand=grad_tensor(batch_size, horizon),
            db_log_pb=grad_tensor(batch_size, horizon),
            db_valid_mask=bool_bt.clone(),
            action_type=torch.zeros((batch_size, horizon), dtype=torch.long),
            continue_mask=bool_bt.clone(),
            stop_mask=bool_bt.clone(),
            selected_edge_ids=torch.full((batch_size, horizon), -1, dtype=torch.long),
            stop_now_log_reward=grad_tensor(batch_size, horizon),
            stop_now_answer_f1=grad_tensor(batch_size, horizon),
            stop_now_valid_mask=bool_bt.clone(),
            target_stop_prob=grad_tensor(batch_size, horizon),
            target_continue_prob=grad_tensor(batch_size, horizon),
            policy_action_valid_mask=bool_bt.clone(),
            edge_action_entropy=grad_tensor(batch_size, horizon),
            edge_action_entropy_valid_mask=bool_bt.clone(),
            budget_exhausted_mask=bool_bt.clone(),
        ),
    )


def _rollout_has_autograd_tensor(rollout: RolloutBatch) -> bool:
    for value in (rollout.stats, rollout.traces):
        for field in dataclasses.fields(value):
            tensor = getattr(value, field.name)
            if isinstance(tensor, torch.Tensor) and (
                tensor.requires_grad or tensor.grad_fn is not None
            ):
                return True
    return False


def _tiny_module() -> WeaverModule:
    return WeaverModule(
        hidden_dim=2,
        entity_text_embeddings=torch.eye(3, 2, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1, 2], dtype=torch.long),
        relation_embeddings=torch.eye(3, 2, dtype=torch.float32),
        rollout={
            "expand_budget": 1,
            "train_num_rollout": 1,
            "eval_num_rollout": 1,
        },
        runtime={
            "train_chunk_size": 1,
            "eval_chunk_size": 1,
        },
        eval={
            "budgets": [1],
        },
        optimizer_cfg={"type": "adamw", "lr": 1.0e-3},
        scheduler_cfg={"type": "none"},
    )


class _ConstantLoss(torch.nn.Module):
    def __init__(
        self,
        value: float,
        requires_stop_now_reward: bool = False,
    ) -> None:
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


def test_module_normalizes_chunk_losses_by_full_training_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tiny_module()
    loss_fn = _ConstantLoss(2.0)
    backward_values: list[float] = []

    module.loss_fn = loss_fn
    module.rollout_runner.train_num_rollout = 4

    def _iter_training_rollout_chunks(**kwargs: object):
        del kwargs
        yield TrainingRolloutChunk(rollouts=(_rollout(), _rollout()), num_rollouts=2)
        yield TrainingRolloutChunk(rollouts=(_rollout(), _rollout()), num_rollouts=2)

    monkeypatch.setattr(
        module.rollout_runner,
        "iter_training_rollout_chunks",
        _iter_training_rollout_chunks,
    )
    monkeypatch.setattr(
        module,
        "manual_backward",
        lambda loss: backward_values.append(float(loss.detach().item())),
    )

    result = module.run_training_rollouts_and_backward(
        batch=object(),
        rollout_temperature=1.0,
        accumulation_batches=2,
    )

    assert result.loss_output.metrics["loss/total"].item() == pytest.approx(2.0)
    assert backward_values == [pytest.approx(0.5), pytest.approx(0.5)]
    assert loss_fn.seen_batch_sizes == [2, 2]


def test_detach_rollout_for_metrics_drops_autograd_from_all_tensors() -> None:
    rollout = _rollout_with_autograd_traces(batch_size=2, horizon=2)
    detached = detach_rollout_for_metrics(rollout)

    assert _rollout_has_autograd_tensor(rollout)
    assert not _rollout_has_autograd_tensor(detached)


def test_training_chunks_keep_loss_rollouts_with_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=2,
        eval_num_rollout=1,
        train_chunk_size=1,
        eval_chunk_size=1,
    )
    def _run_vectorized(**kwargs: object) -> list[RolloutBatch]:
        return [
            _rollout_with_autograd_traces()
            for _ in range(int(kwargs["num_rollouts"]))
        ]

    monkeypatch.setattr(runner.engine, "run_vectorized", _run_vectorized)

    chunks = list(runner.iter_training_rollout_chunks(
        policy=object(),
        reward_model=object(),
        batch=object(),
        rollout_temperature=1.0,
        collect_policy_diagnostics=False,
    ))

    assert len(chunks) == 2
    assert all(
        _rollout_has_autograd_tensor(rollout)
        for chunk in chunks
        for rollout in chunk.rollouts
    )


def test_module_training_result_keeps_only_detached_metric_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tiny_module()
    loss_fn = _ConstantLoss(1.0)
    loss_seen_autograd: list[bool] = []

    module.loss_fn = loss_fn

    def _iter_training_rollout_chunks(**kwargs: object):
        del kwargs
        yield TrainingRolloutChunk(
            rollouts=(_rollout_with_autograd_traces(),),
            num_rollouts=1,
        )
        yield TrainingRolloutChunk(
            rollouts=(_rollout_with_autograd_traces(),),
            num_rollouts=1,
        )

    def _loss(rollout: RolloutBatch) -> LossOutput:
        loss_seen_autograd.append(_rollout_has_autograd_tensor(rollout))
        return _ConstantLoss.forward(loss_fn, rollout)

    monkeypatch.setattr(
        module.rollout_runner,
        "iter_training_rollout_chunks",
        _iter_training_rollout_chunks,
    )
    monkeypatch.setattr(loss_fn, "forward", _loss)
    monkeypatch.setattr(module, "manual_backward", lambda loss: None)

    result = module.run_training_rollouts_and_backward(
        batch=object(),
        rollout_temperature=1.0,
        accumulation_batches=1,
    )

    assert loss_seen_autograd == [True, True]
    assert len(result.rollouts) == 2
    assert not any(_rollout_has_autograd_tensor(rollout) for rollout in result.rollouts)


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


def test_concat_rollout_batches_preserves_missing_optional_auxiliary_traces() -> None:
    first = _rollout(batch_size=1, horizon=2)
    second = _rollout(batch_size=2, horizon=2)
    first = dataclasses.replace(
        first,
        traces=dataclasses.replace(
            first.traces,
            stop_now_log_reward=None,
            stop_now_answer_f1=None,
            stop_now_valid_mask=None,
        ),
    )
    second = dataclasses.replace(
        second,
        traces=dataclasses.replace(
            second.traces,
            stop_now_log_reward=None,
            stop_now_answer_f1=None,
            stop_now_valid_mask=None,
        ),
    )

    merged = concat_rollout_batches([first, second])

    assert merged.traces.stop_now_log_reward is None
    assert merged.traces.stop_now_answer_f1 is None
    assert merged.traces.stop_now_valid_mask is None


def test_rollout_config_rejects_removed_rollout_mode_flags() -> None:
    with pytest.raises(ValueError, match="Unused rollout keys"):
        build_rollout_runtime_config(
            rollout={"use_static_batch_rollouts": True},
            runtime=None,
        )

    with pytest.raises(ValueError, match="Unused rollout keys"):
        build_rollout_runtime_config(
            rollout={"use_fused_static_batch_rollouts": True},
            runtime=None,
        )


def test_rollout_config_rejects_removed_auxiliary_configs() -> None:
    with pytest.raises(ValueError, match="Unused rollout keys"):
        build_rollout_runtime_config(
            rollout={"stop_adv": {"enabled": True}},
            runtime=None,
        )

    with pytest.raises(ValueError, match="Unused rollout keys"):
        build_rollout_runtime_config(
            rollout={"local_improvement": {"enabled": True}},
            runtime=None,
        )


def test_loss_config_rejects_removed_dag_db_selector() -> None:
    with pytest.raises(
        ValueError,
        match="Only loss.type='bdb'",
    ):
        build_loss_config(
            {"type": "dag_db"},
            max_trajectory_len=1,
        )


def test_build_loss_supports_bdb_only() -> None:
    loss = build_loss({"type": "bdb"})

    assert isinstance(loss, BudgetedDAGDetailedBalanceLoss)


def test_loss_config_parses_bdb_defaults() -> None:
    cfg = build_loss_config(
        {},
        max_trajectory_len=1,
    )

    assert cfg["type"] == "bdb"
    assert cfg["child_chunk_size"] == 2048
    assert cfg["child_flow_target"] == "detach_current"


def test_loss_config_rejects_removed_auxiliary_loss_keys() -> None:
    with pytest.raises(ValueError, match="Unused loss keys"):
        build_loss_config({"stop_oracle_weight": 0.0}, max_trajectory_len=1)

    with pytest.raises(ValueError, match="Unused loss keys"):
        build_loss_config({"counterfactual_edge_weight": 0.0}, max_trajectory_len=1)

    with pytest.raises(ValueError, match="Unused loss keys"):
        build_loss_config({"center_log_reward_by_question": True}, max_trajectory_len=1)

    with pytest.raises(ValueError, match="Removed loss keys"):
        build_loss_config(
            {"stop_boundary": {"enabled": False, "unused": True}},
            max_trajectory_len=1,
        )


def test_reward_config_parses_set_reward_defaults() -> None:
    cfg = build_reward_config({})

    assert cfg["reward_floor"] == pytest.approx(1.0e-6)
    assert cfg["edge_cost"] == pytest.approx(0.1)
    assert cfg["beta"] == pytest.approx(2.0)
    assert cfg["debug_checks"] is False


def test_reward_config_parses_set_reward_overrides() -> None:
    cfg = build_reward_config(
        {
            "reward_floor": 1.0e-3,
            "edge_cost": 0.2,
            "beta": 1.0,
            "debug_checks": True,
        }
    )

    assert cfg["reward_floor"] == pytest.approx(1.0e-3)
    assert cfg["edge_cost"] == pytest.approx(0.2)
    assert cfg["beta"] == pytest.approx(1.0)
    assert cfg["debug_checks"] is True


def test_reward_config_rejects_unknown_process_key() -> None:
    with pytest.raises(ValueError, match="Unused reward keys"):
        build_reward_config({"process": {"relation_weight": 1.0}})


def test_reward_config_rejects_removed_reward_keys() -> None:
    for key in (
        "utility_epsilon",
        "edge_cost_base",
        "edge_cost_answer",
        "score_mode",
        "length_discount",
        "path_weight",
        "prefix_answer_bonus",
        "wrong_branch_penalty",
        "path_prefix_weight",
    ):
        with pytest.raises(ValueError):
            build_reward_config({key: 1.0})


def test_reward_config_rejects_invalid_set_reward_scale() -> None:
    with pytest.raises(ValueError, match="reward_floor"):
        build_reward_config({"reward_floor": 0.0})

    with pytest.raises(ValueError, match="edge_cost"):
        build_reward_config({"edge_cost": -0.1})

    with pytest.raises(ValueError, match="beta"):
        build_reward_config({"beta": 0.0})


def test_bdb_algorithm_coupling_rejects_mismatched_loss() -> None:
    policy = build_policy_runtime_config(
        hidden_dim=2,
        entity_text_embeddings=torch.eye(2, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
        relation_embeddings=torch.eye(2, dtype=torch.float32),
    )
    rollout = build_rollout_runtime_config(
        rollout={"expand_budget": 1, "train_num_rollout": 1, "eval_num_rollout": 1},
        runtime=None,
    )
    reward = build_reward_config({})

    with pytest.raises(ValueError, match="requires loss.type='bdb'"):
        validate_algorithm_coupling(
            policy=policy,
            loss={"type": "subtb"},
            rollout=rollout,
            reward=reward,
        )

    with pytest.raises(ValueError, match="policy.mode"):
        build_policy_runtime_config(
            hidden_dim=2,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
            policy={"mode": "subtb", "flow_budget_conditioning": "none"},
        )


def test_bdb_algorithm_coupling_accepts_reward_floor() -> None:
    policy = build_policy_runtime_config(
        hidden_dim=2,
        entity_text_embeddings=torch.eye(2, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
        relation_embeddings=torch.eye(2, dtype=torch.float32),
    )
    rollout = build_rollout_runtime_config(
        rollout={"expand_budget": 1, "train_num_rollout": 1, "eval_num_rollout": 1},
        runtime=None,
    )

    validate_algorithm_coupling(
        policy=policy,
        loss={"type": "bdb"},
        rollout=rollout,
        reward={"reward_floor": 1.0e-2},
    )


def test_policy_runtime_config_does_not_carry_reward_edge_cost() -> None:
    runtime = build_policy_runtime_config(
        hidden_dim=2,
        entity_text_embeddings=torch.eye(2, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
        relation_embeddings=torch.eye(2, dtype=torch.float32),
    )

    assert runtime.hidden_dim == 2
    assert runtime.mode == "bdb"
    assert runtime.flow_budget_conditioning == "additive"
    assert not hasattr(runtime, "edge_cost")


def test_policy_runtime_config_rejects_removed_path_memory_key() -> None:
    with pytest.raises(ValueError, match="Removed evidence_state_encoder keys"):
        build_policy_runtime_config(
            hidden_dim=2,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
            policy={"evidence_state_encoder": {"use_path_memory": False}},
        )


def test_weaver_module_has_no_target_policy_parameters() -> None:
    from src.training.optimization import AdamWConfig, build_param_groups

    module = _tiny_module()

    assert not hasattr(module, "target_policy")
    param_groups = build_param_groups(module, cfg=AdamWConfig())
    names = [
        name
        for group in param_groups
        for name in group.get("param_names", ())
    ]
    assert names
    assert not any(name.startswith("target_policy.") for name in names)


def test_weaver_module_checkpoint_load_drops_legacy_target_policy_state() -> None:
    module = _tiny_module()
    state_dict = module.state_dict()
    checkpoint = {
        "state_dict": {
            **state_dict,
            "target_policy.fake_weight": torch.ones(1),
        },
    }

    module.on_load_checkpoint(checkpoint)

    assert "target_policy.fake_weight" not in checkpoint["state_dict"]


def test_weaver_module_checkpoint_save_drops_legacy_target_sync_counter() -> None:
    module = _tiny_module()
    checkpoint: dict[str, object] = {"target_policy_optimizer_steps": 1}

    module.on_save_checkpoint(checkpoint)

    assert "target_policy_optimizer_steps" not in checkpoint


def test_eval_runtime_config_defaults_to_skipping_loss() -> None:
    runtime = build_eval_runtime_config(eval_cfg={"budgets": [1]}, eval_num_rollout=1)

    assert runtime.compute_loss is False


def test_eval_runtime_config_can_enable_loss() -> None:
    runtime = build_eval_runtime_config(
        eval_cfg={"budgets": [1], "compute_loss": True},
        eval_num_rollout=1,
    )

    assert runtime.compute_loss is True


def test_runner_stores_stop_now_trace_when_loss_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RolloutRunner(
        expand_budget=1,
        train_num_rollout=2,
        eval_num_rollout=2,
        train_chunk_size=1,
        eval_chunk_size=1,
    )
    seen_store_stop_now: list[bool] = []

    def _run_vectorized(**kwargs: object) -> list[RolloutBatch]:
        seen_store_stop_now.append(bool(kwargs["store_stop_now_reward"]))
        return [_rollout() for _ in range(int(kwargs["num_rollouts"]))]

    monkeypatch.setattr(runner.engine, "run_vectorized", _run_vectorized)

    list(runner.iter_training_rollout_chunks(
        policy=object(),
        reward_model=object(),
        loss_fn=_ConstantLoss(1.0, requires_stop_now_reward=True),
        batch=object(),
        rollout_temperature=1.0,
        collect_policy_diagnostics=False,
    ))
    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
        collect_policy_diagnostics=True,
    )
    runner.generate_eval_rollouts(
        policy=object(),
        reward_model=object(),
        batch=object(),
        temperature=1.0,
        collect_policy_diagnostics=False,
        loss_fn=_ConstantLoss(1.0, requires_stop_now_reward=True),
    )

    assert seen_store_stop_now == [True, True, False, False, True, True]


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
