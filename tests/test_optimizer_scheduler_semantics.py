from __future__ import annotations

from collections.abc import Iterator
from math import log
from typing import Any

import pytest
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.models.configs import (
    ActionPriorConfig,
    ActionPriorScheduleConfig,
    AnswerQuotientConfig,
    GFlowNetTrainingConfig,
    OptimizerConfig,
    PotentialRewardConfig,
    SamplingTemperatureScheduleConfig,
    SchedulerConfig,
)
from src.models.gflownet import (
    ActionPriorScheduler,
    SamplingTemperatureScheduler,
    TrainingScheduleContext,
)
from src.models.gflownet_module import GFlowNetModule


def _build_optimizer_config(
    *,
    model_parameters: Iterator[tuple[str, Parameter]],
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    estimated_stepping_batches: int | None,
    trainer_max_steps: int | None = None,
    trainer_max_epochs: int | None = None,
) -> dict[str, Any]:
    return GFlowNetModule._build_optimizer_and_scheduler(
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


def test_linear_sampling_temperature_scheduler_uses_training_horizon() -> None:
    scheduler = SamplingTemperatureScheduler(
        base_temperature=2.0,
        config=SamplingTemperatureScheduleConfig(type="linear", final_temperature=0.5),
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
        config=SamplingTemperatureScheduleConfig(type="cosine", final_temperature=0.5),
    )
    schedule_context = TrainingScheduleContext(estimated_stepping_batches=None)

    with pytest.raises(RuntimeError, match="known step horizon"):
        scheduler.value(global_step=0, schedule_context=schedule_context)


def test_linear_action_prior_scheduler_uses_training_horizon() -> None:
    scheduler = ActionPriorScheduler(
        base_scale=1.0,
        config=ActionPriorScheduleConfig(type="linear", final_scale=0.25),
    )
    schedule_context = TrainingScheduleContext(
        estimated_stepping_batches=1_000_000,
        trainer_max_steps=4,
    )

    assert scheduler.value(
        global_step=0, schedule_context=schedule_context
    ) == pytest.approx(1.0)
    assert scheduler.value(
        global_step=3, schedule_context=schedule_context
    ) == pytest.approx(0.25)


def test_annealed_action_prior_scheduler_requires_known_horizon() -> None:
    scheduler = ActionPriorScheduler(
        base_scale=1.0,
        config=ActionPriorScheduleConfig(type="cosine", final_scale=0.25),
    )
    schedule_context = TrainingScheduleContext(estimated_stepping_batches=None)

    with pytest.raises(RuntimeError, match="known step horizon"):
        scheduler.value(global_step=0, schedule_context=schedule_context)


def test_intent_alignment_prior_requires_nonzero_action_features() -> None:
    with pytest.raises(ValueError, match="intent_alignment_weight requires"):
        ActionPriorConfig(
            root_beta=0.5,
            edge_beta=0.5,
            intent_alignment_weight=1.0,
            intent_relation_weight=0.0,
            intent_target_weight=0.0,
        )


def test_gflownet_training_config_exposes_direct_step_log_penalty() -> None:
    training_cfg = GFlowNetTrainingConfig(step_log_penalty=log(0.5))

    assert training_cfg.step_log_penalty == pytest.approx(log(0.5))


def test_gflownet_training_config_default_step_log_penalty_is_neutral() -> None:
    assert GFlowNetTrainingConfig().step_log_penalty == pytest.approx(0.0)


def test_optimizer_config_default_weight_decay_matches_model_config() -> None:
    assert OptimizerConfig().weight_decay == pytest.approx(1.0e-4)


def test_scheduler_config_default_eta_min_matches_model_config() -> None:
    assert SchedulerConfig().eta_min == pytest.approx(1.0e-6)


def test_gflownet_training_config_validates_answer_stop_bonus() -> None:
    training_cfg = GFlowNetTrainingConfig(answer_stop_log_reward_bonus=0.75)

    assert training_cfg.answer_stop_log_reward_bonus == pytest.approx(0.75)
    with pytest.raises(ValueError, match="answer_stop_log_reward_bonus"):
        GFlowNetTrainingConfig(answer_stop_log_reward_bonus=-0.1)


def test_answer_quotient_config_validates_terminal_replacement_weight() -> None:
    cfg = AnswerQuotientConfig(enabled=True, weight=0.25, replace_terminal_loss=False)

    assert cfg.active is True
    assert cfg.stop_allocation_active is False
    with pytest.raises(ValueError, match="replace_terminal_loss requires weight > 0"):
        AnswerQuotientConfig(enabled=True, weight=0.0, replace_terminal_loss=True)


def test_answer_quotient_config_exposes_stop_allocation_flag() -> None:
    cfg = AnswerQuotientConfig(enabled=True, allocate_stop_mass=True)

    assert cfg.active is False
    assert cfg.stop_allocation_active is True


def test_answer_quotient_config_exposes_direct_entity_ranking_flag() -> None:
    cfg = AnswerQuotientConfig(enabled=True, direct_entity_ranking_weight=0.25)

    assert cfg.active is False
    assert cfg.direct_entity_ranking_active is True


def test_answer_quotient_config_validates_direct_entity_ranking_weight() -> None:
    with pytest.raises(ValueError, match="direct_entity_ranking_weight must be >= 0"):
        AnswerQuotientConfig(enabled=True, direct_entity_ranking_weight=-0.1)
    with pytest.raises(
        ValueError,
        match="direct_entity_ranking_weight requires enabled=True",
    ):
        AnswerQuotientConfig(direct_entity_ranking_weight=0.1)


def test_potential_reward_config_exposes_answer_distance_flag() -> None:
    cfg = PotentialRewardConfig(answer_distance_weight=0.5)

    assert cfg.active is True


def test_potential_reward_config_validates_weight_and_unreachable_distance() -> None:
    with pytest.raises(ValueError, match="answer_distance_weight must be >= 0"):
        PotentialRewardConfig(answer_distance_weight=-0.1)
    with pytest.raises(ValueError, match="unreachable_distance must be >= 0"):
        PotentialRewardConfig(
            answer_distance_weight=0.5,
            unreachable_distance=-1,
        )
