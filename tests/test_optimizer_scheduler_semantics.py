from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

import pytest
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.subgraph_gflownet.core.config_utils import (
    normalize_training_cfg,
)
from src.subgraph_gflownet.core.optimization import build_optimizer_and_scheduler
from src.subgraph_gflownet.core.schedules import (
    ReplayMixScheduler,
    SamplingTemperatureScheduler,
    TrainingScheduleContext,
)


def _build_optimizer_config(
    *,
    model_parameters: Iterator[tuple[str, Parameter]],
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    estimated_stepping_batches: int | None,
    trainer_max_steps: int | None = None,
    trainer_max_epochs: int | None = None,
) -> dict[str, Any]:
    return build_optimizer_and_scheduler(
        model_parameters=model_parameters,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        schedule_context=TrainingScheduleContext(
            estimated_stepping_batches=estimated_stepping_batches,
            trainer_max_steps=trainer_max_steps,
            trainer_max_epochs=trainer_max_epochs,
        ),
    )


def _build_linear_named_parameters() -> tuple[
    torch.nn.Module, Iterator[tuple[str, Parameter]]
]:
    model = torch.nn.Linear(8, 4)
    return model, model.named_parameters()


def _build_linear_with_norm_named_parameters() -> tuple[
    torch.nn.Module, Iterator[tuple[str, Parameter]]
]:
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.LayerNorm(4))
    return model, model.named_parameters()


class _ConditionedLogZToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(8, 4)
        self.root_flow_input_norm = torch.nn.LayerNorm(4)
        self.root_flow_hidden = torch.nn.Linear(4, 4)
        self.root_flow_head = torch.nn.Linear(4, 1)


def _build_conditioned_log_z_named_parameters() -> tuple[
    torch.nn.Module, Iterator[tuple[str, Parameter]]
]:
    model = _ConditionedLogZToyModel()
    return model, model.named_parameters()


def _default_optimizer_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "type": "adamw",
        "lr": 1.0e-4,
        "log_z_head_lr_multiplier": 5.0,
        "weight_decay": 1.0e-4,
        "betas": (0.9, 0.999),
        "no_decay_on_bias_and_norm": True,
    }
    cfg.update(overrides)
    return cfg


def _default_scheduler_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "type": "cosine",
        "interval": "step",
        "t_max": None,
        "t_mult": 1,
        "eta_min": 1.0e-6,
        "pct_start": 0.3,
        "anneal": "cos",
        "lr": None,
    }
    cfg.update(overrides)
    return cfg


def test_onecycle_scheduler_uses_t_max_total_steps() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "onecycle", "t_max": 120, "lr": 2e-4},
        estimated_stepping_batches=100,
    )
    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, OneCycleLR)
    assert int(scheduler.total_steps) == 120


def test_onecycle_scheduler_rejects_non_positive_t_max() -> None:
    model, named_parameters = _build_linear_named_parameters()
    with pytest.raises(ValueError, match="t_max > 0"):
        _build_optimizer_config(
            model_parameters=named_parameters,
            optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            scheduler_cfg={"type": "onecycle", "t_max": 0, "lr": 2e-4},
            estimated_stepping_batches=10,
        )


def test_onecycle_scheduler_rejects_t_max_smaller_than_estimated_steps() -> None:
    model, named_parameters = _build_linear_named_parameters()
    with pytest.raises(ValueError, match="would exhaust before training ends"):
        _build_optimizer_config(
            model_parameters=named_parameters,
            optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            scheduler_cfg={"type": "onecycle", "t_max": 40, "lr": 2e-4},
            estimated_stepping_batches=50,
        )


def test_cosine_step_interval_prefers_trainer_max_steps() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "interval": "step", "t_max": None},
        estimated_stepping_batches=1_000_000,
        trainer_max_steps=240,
    )

    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, CosineAnnealingLR)
    assert int(scheduler.T_max) == 240
    assert config["lr_scheduler"]["interval"] == "step"


def test_onecycle_scheduler_uses_trainer_max_steps_when_t_max_missing() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "onecycle", "lr": 2e-4},
        estimated_stepping_batches=1_000_000,
        trainer_max_steps=120,
    )

    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, OneCycleLR)
    assert int(scheduler.total_steps) == 120


def test_cosine_epoch_interval_uses_trainer_max_epochs() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "interval": "epoch", "t_max": None},
        estimated_stepping_batches=120,
        trainer_max_epochs=12,
    )

    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, CosineAnnealingLR)
    assert int(scheduler.T_max) == 12
    assert config["lr_scheduler"]["interval"] == "epoch"


def test_onecycle_scheduler_rejects_epoch_interval() -> None:
    model, named_parameters = _build_linear_named_parameters()
    with pytest.raises(ValueError, match="interval='step'"):
        _build_optimizer_config(
            model_parameters=named_parameters,
            optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            scheduler_cfg={
                "type": "onecycle",
                "interval": "epoch",
                "t_max": 120,
                "lr": 2e-4,
            },
            estimated_stepping_batches=100,
            trainer_max_epochs=10,
        )


def test_optimizer_excludes_bias_and_norm_from_weight_decay() -> None:
    model, named_parameters = _build_linear_with_norm_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.1},
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    optimizer = config["optimizer"]
    named_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    decay_names: set[str] = set()
    no_decay_names: set[str] = set()
    for group in optimizer.param_groups:
        param_names = {named_by_id[id(parameter)] for parameter in group["params"]}
        if float(group["weight_decay"]) == 0.0:
            no_decay_names |= param_names
        else:
            decay_names |= param_names

    assert decay_names == {"0.weight"}
    assert no_decay_names == {"0.bias", "1.weight", "1.bias"}


def test_optimizer_can_apply_weight_decay_to_all_params_when_requested() -> None:
    model, named_parameters = _build_linear_with_norm_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.1,
            "no_decay_on_bias_and_norm": False,
        },
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    optimizer = config["optimizer"]

    assert len(optimizer.param_groups) == 1
    assert float(optimizer.param_groups[0]["weight_decay"]) == pytest.approx(0.1)


def test_optimizer_can_scale_conditioned_log_z_head_learning_rate() -> None:
    model, named_parameters = _build_conditioned_log_z_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={
            "type": "adamw",
            "lr": 1e-4,
            "log_z_head_lr_multiplier": 5.0,
            "weight_decay": 0.1,
        },
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    optimizer = config["optimizer"]
    named_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    grouped_names: dict[str, set[str]] = {}
    grouped_lrs: dict[str, float] = {}
    grouped_weight_decay: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_name = str(group["group_name"])
        grouped_names[group_name] = {
            named_by_id[id(parameter)] for parameter in group["params"]
        }
        grouped_lrs[group_name] = float(group["lr"])
        grouped_weight_decay[group_name] = float(group["weight_decay"])

    assert grouped_names == {
        "decay": {"encoder.weight"},
        "no_decay": {"encoder.bias"},
        "log_z_head_decay": {"root_flow_hidden.weight", "root_flow_head.weight"},
        "log_z_head_no_decay": {
            "root_flow_input_norm.weight",
            "root_flow_input_norm.bias",
            "root_flow_hidden.bias",
            "root_flow_head.bias",
        },
    }
    assert grouped_lrs["decay"] == pytest.approx(1.0e-4)
    assert grouped_lrs["no_decay"] == pytest.approx(1.0e-4)
    assert grouped_lrs["log_z_head_decay"] == pytest.approx(5.0e-4)
    assert grouped_lrs["log_z_head_no_decay"] == pytest.approx(5.0e-4)
    assert grouped_weight_decay["decay"] == pytest.approx(0.1)
    assert grouped_weight_decay["log_z_head_decay"] == pytest.approx(0.1)
    assert grouped_weight_decay["no_decay"] == pytest.approx(0.0)
    assert grouped_weight_decay["log_z_head_no_decay"] == pytest.approx(0.0)


def test_linear_sampling_temperature_scheduler_uses_training_horizon() -> None:
    scheduler = SamplingTemperatureScheduler(
        base_temperature=2.0,
        type="linear",
        final_temperature=0.5,
    )
    schedule_context = TrainingScheduleContext(
        estimated_stepping_batches=1_000_000,
        trainer_max_steps=4,
    )

    assert scheduler.value(
        global_step=0, schedule_context=schedule_context
    ) == pytest.approx(2.0)
    assert scheduler.value(
        global_step=3, schedule_context=schedule_context
    ) == pytest.approx(0.5)


def test_annealed_sampling_temperature_scheduler_requires_known_horizon() -> None:
    scheduler = SamplingTemperatureScheduler(
        base_temperature=2.0,
        type="cosine",
        final_temperature=0.5,
    )
    schedule_context = TrainingScheduleContext(estimated_stepping_batches=None)

    with pytest.raises(RuntimeError, match="known step horizon"):
        scheduler.value(global_step=0, schedule_context=schedule_context)


def test_replay_mix_scheduler_uses_base_alpha_and_hold_steps() -> None:
    scheduler = ReplayMixScheduler(
        base_alpha=0.5,
        type="cosine",
        final_alpha=0.0,
        total_steps=4,
        hold_steps=1,
    )
    schedule_context = TrainingScheduleContext(estimated_stepping_batches=4)

    assert scheduler.value(
        global_step=0, schedule_context=schedule_context
    ) == pytest.approx(0.5)
    assert scheduler.value(
        global_step=3, schedule_context=schedule_context
    ) == pytest.approx(0.0)
    mid_value = scheduler.value(global_step=1, schedule_context=schedule_context)
    assert 0.0 < mid_value < 0.5


def test_normalize_training_cfg_default_frontier_bonus_is_stable() -> None:
    assert normalize_training_cfg({})["answer_reward"][
        "frontier_bonus"
    ] == pytest.approx(1.0)


def test_normalize_training_cfg_populates_train_action_pruning_defaults() -> None:
    training_cfg = normalize_training_cfg({})

    assert training_cfg["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }


def test_normalize_training_cfg_rejects_removed_proposal_bias_switches() -> None:
    with pytest.raises(ValueError, match="proposal.enabled was removed"):
        normalize_training_cfg({"auxiliary": {"proposal": {"enabled": False}}})

    with pytest.raises(ValueError, match="proposal.schedule was removed"):
        normalize_training_cfg(
            {
                "auxiliary": {
                    "proposal": {
                        "schedule": {"type": "constant"},
                    }
                }
            }
        )


def test_normalize_training_cfg_allows_disabling_train_action_pruning() -> None:
    training_cfg = normalize_training_cfg(
        {"action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0}}
    )

    assert training_cfg["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }
    with pytest.raises(ValueError, match="training.action_pruning.per_node_top_k"):
        normalize_training_cfg({"action_pruning": {"per_node_top_k": -1}})
    with pytest.raises(ValueError, match="training.action_pruning.per_state_top_k"):
        normalize_training_cfg({"action_pruning": {"per_state_top_k": -1}})


def test_normalize_training_cfg_validates_replay_expand_imitation_weight() -> None:
    training_cfg = normalize_training_cfg(
        {
            "auxiliary": {
                "replay": {
                    "guidance": {
                        "expand_imitation_weight": 0.75,
                        "expand_imitation_from_anchor_bonus": 1.25,
                        "expand_imitation_answer_finish_bonus": 2.5,
                    }
                }
            }
        }
    )

    assert training_cfg["auxiliary"]["replay"]["guidance"][
        "expand_imitation_weight"
    ] == pytest.approx(0.75)
    assert training_cfg["auxiliary"]["replay"]["guidance"][
        "expand_imitation_from_anchor_bonus"
    ] == pytest.approx(1.25)
    assert training_cfg["auxiliary"]["replay"]["guidance"][
        "expand_imitation_answer_finish_bonus"
    ] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="expand_imitation_weight"):
        normalize_training_cfg(
            {"auxiliary": {"replay": {"guidance": {"expand_imitation_weight": -0.1}}}}
        )
    with pytest.raises(ValueError, match="expand_imitation_from_anchor_bonus"):
        normalize_training_cfg(
            {
                "auxiliary": {
                    "replay": {"guidance": {"expand_imitation_from_anchor_bonus": -0.1}}
                }
            }
        )
    with pytest.raises(ValueError, match="expand_imitation_answer_finish_bonus"):
        normalize_training_cfg(
            {
                "auxiliary": {
                    "replay": {
                        "guidance": {"expand_imitation_answer_finish_bonus": -0.1}
                    }
                }
            }
        )


def test_optimizer_defaults_weight_decay_to_model_config_value() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4},
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    optimizer = config["optimizer"]

    assert any(
        float(group["weight_decay"]) == pytest.approx(1.0e-4)
        for group in optimizer.param_groups
    )


def test_optimizer_defaults_conditioned_log_z_lr_multiplier_to_model_config_value() -> (
    None
):
    model, named_parameters = _build_conditioned_log_z_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4},
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    optimizer = config["optimizer"]
    grouped_lrs = {
        str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups
    }

    assert grouped_lrs["log_z_head_decay"] == pytest.approx(5.0e-4)
    assert grouped_lrs["log_z_head_no_decay"] == pytest.approx(5.0e-4)


def test_optimizer_validates_conditioned_log_z_lr_multiplier() -> None:
    model, named_parameters = _build_conditioned_log_z_named_parameters()

    with pytest.raises(ValueError, match="log_z_head_lr_multiplier"):
        _build_optimizer_config(
            model_parameters=named_parameters,
            optimizer_cfg={
                "type": "adamw",
                "lr": 1e-4,
                "log_z_head_lr_multiplier": 0.0,
            },
            scheduler_cfg={"type": "cosine", "t_max": 8},
            estimated_stepping_batches=8,
        )


def test_scheduler_defaults_eta_min_to_model_config_value() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = _build_optimizer_config(
        model_parameters=named_parameters,
        optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "t_max": 8},
        estimated_stepping_batches=8,
    )

    scheduler = config["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, CosineAnnealingLR)
    assert float(scheduler.eta_min) == pytest.approx(1.0e-6)


def test_normalize_training_cfg_rejects_removed_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Removed training config detected"):
        normalize_training_cfg({"success_replay": {"mix_alpha": 0.2}})


def test_normalize_training_cfg_rejects_removed_legacy_reward_shaping() -> None:
    with pytest.raises(ValueError, match="answer-quotient configs were removed"):
        normalize_training_cfg({"answer_quotient": {"enabled": True}})
    with pytest.raises(ValueError, match="potential-reward shaping was removed"):
        normalize_training_cfg({"potential_reward": {"answer_distance_weight": 0.5}})


def test_normalize_training_cfg_rejects_removed_subtb_knobs() -> None:
    with pytest.raises(ValueError, match="SubTB was removed"):
        normalize_training_cfg({"subtb": {"root_loss_weight": 0.5}})


def test_normalize_training_cfg_rejects_removed_answer_reward_keys() -> None:
    with pytest.raises(ValueError, match="gold_answer_bonus"):
        normalize_training_cfg({"answer_reward": {"gold_answer_bonus": 3.0}})
    with pytest.raises(ValueError, match="wrong-answer penalties were removed"):
        normalize_training_cfg({"answer_reward": {"wrong_answer_penalty": 0.5}})


def test_normalize_training_cfg_rejects_enabled_proposal_auxiliary() -> None:
    with pytest.raises(ValueError, match="proposal.enabled was removed"):
        normalize_training_cfg({"auxiliary": {"proposal": {"enabled": True}}})


def test_normalize_training_cfg_validates_answer_reward() -> None:
    training_cfg = normalize_training_cfg(
        {
            "answer_reward": {
                "hit_bonus": 3.5,
                "frontier_bonus": 1.5,
                "coverage_bonus": 0.25,
            }
        }
    )

    assert training_cfg["answer_reward"]["hit_bonus"] == pytest.approx(3.5)
    assert training_cfg["answer_reward"]["frontier_bonus"] == pytest.approx(1.5)
    assert training_cfg["answer_reward"]["coverage_bonus"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="frontier_bonus"):
        normalize_training_cfg({"answer_reward": {"frontier_bonus": -0.1}})
