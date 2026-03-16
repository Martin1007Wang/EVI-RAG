from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.models.gflownet_module import GFlowNetModule


def _build_optimizer_config(
    *,
    model_parameters: Iterator[tuple[str, Parameter]],
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    estimated_stepping_batches: int | None,
    trainer_max_epochs: int | None = None,
) -> dict[str, Any]:
    return GFlowNetModule._build_optimizer_and_scheduler(
        model_parameters=model_parameters,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        estimated_stepping_batches=estimated_stepping_batches,
        trainer_max_epochs=trainer_max_epochs,
    )


def _build_linear_named_parameters() -> tuple[
    torch.nn.Module, Iterator[tuple[str, Parameter]]
]:
    model = torch.nn.Linear(8, 4)
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
