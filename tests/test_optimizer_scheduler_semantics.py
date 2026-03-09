from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import OneCycleLR

from src.models.algorithms.base import build_optimizer_and_scheduler


def _build_linear_named_parameters() -> tuple[
    torch.nn.Module, Iterator[tuple[str, Parameter]]
]:
    model = torch.nn.Linear(8, 4)
    return model, model.named_parameters()


def test_onecycle_scheduler_uses_t_max_total_steps() -> None:
    model, named_parameters = _build_linear_named_parameters()
    config = build_optimizer_and_scheduler(
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
        build_optimizer_and_scheduler(
            model_parameters=named_parameters,
            optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            scheduler_cfg={"type": "onecycle", "t_max": 0, "lr": 2e-4},
            estimated_stepping_batches=10,
        )


def test_onecycle_scheduler_rejects_t_max_smaller_than_estimated_steps() -> None:
    model, named_parameters = _build_linear_named_parameters()
    with pytest.raises(ValueError, match="would exhaust before training ends"):
        build_optimizer_and_scheduler(
            model_parameters=named_parameters,
            optimizer_cfg={"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            scheduler_cfg={"type": "onecycle", "t_max": 40, "lr": 2e-4},
            estimated_stepping_batches=50,
        )
