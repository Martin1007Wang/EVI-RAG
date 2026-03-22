from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from omegaconf import OmegaConf

from src.utils.fit_schedule import resolve_pass_fit_schedule


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


def test_train_rankflow_experiment_uses_canonical_long_pass_schedule() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=train_rankflow",
                "dataset=webqsp-sub",
                "extras.enforce_tags=false",
                "extras.print_config=false",
            ],
        )

    assert cfg.fit_schedule.max_passes == pytest.approx(240.0)
    assert cfg.fit_schedule.val_every_passes == pytest.approx(8.0)
    assert cfg.fit_schedule.early_stopping_patience_passes == pytest.approx(96.0)
