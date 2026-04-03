from __future__ import annotations

import pytest

from omegaconf import OmegaConf

from src.runs.fit_schedule import resolve_pass_fit_schedule


def test_resolve_pass_fit_schedule_scales_training_by_dataset_passes() -> None:
    resolved = resolve_pass_fit_schedule(
        fit_schedule_cfg={
            "mode": "pass_based",
            "max_passes": 120.0,
            "val_every_passes": 2.0,
            "early_stopping_patience_passes": 24.0,
        },
        trainer_cfg={"devices": 1, "accumulate_grad_batches": 1},
        train_size=21_148,
        per_device_batch_size=32,
    )

    assert resolved.global_batch_size == 32
    assert resolved.examples_per_optimizer_step == 32
    assert resolved.max_steps == 79_305
    assert resolved.val_check_interval_batches == 1_322
    assert resolved.early_stopping_patience_checks == 12
    assert resolved.effective_pass(global_step=661) == pytest.approx(661 * 32 / 21_148)


def test_resolve_pass_fit_schedule_accounts_for_accumulation_and_device_count() -> None:
    resolved = resolve_pass_fit_schedule(
        fit_schedule_cfg=OmegaConf.create(
            {
                "mode": "pass_based",
                "max_passes": 10.0,
                "val_every_passes": 1.5,
                "early_stopping_patience_passes": 6.0,
            }
        ),
        trainer_cfg=OmegaConf.create({"devices": [0, 1], "accumulate_grad_batches": 4}),
        train_size=6_400,
        per_device_batch_size=16,
    )

    assert resolved.data_parallel_size == 2
    assert resolved.global_batch_size == 32
    assert resolved.examples_per_optimizer_step == 128
    assert resolved.optimizer_steps_per_pass == pytest.approx(50.0)
    assert resolved.train_batches_per_pass == pytest.approx(200.0)
    assert resolved.max_steps == 500
    assert resolved.val_check_interval_batches == 300
    assert resolved.early_stopping_patience_checks == 4


def test_resolve_pass_fit_schedule_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="max_passes"):
        resolve_pass_fit_schedule(
            fit_schedule_cfg={
                "mode": "pass_based",
                "max_passes": 0.0,
                "val_every_passes": 1.0,
                "early_stopping_patience_passes": 1.0,
            },
            trainer_cfg={"devices": 1, "accumulate_grad_batches": 1},
            train_size=128,
            per_device_batch_size=32,
        )
