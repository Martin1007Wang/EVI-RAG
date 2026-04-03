from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import pytest

from src.runs.rankflow import run_eval, validate_train_config


def test_rankflow_runner_sets_split_specific_allow_empty_answer(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {
                "name": "rankflow",
                "split": "test",
                "run_all_splits": True,
                "splits": ["train", "validation"],
                "execution_mode": "predict",
            },
        }
    )
    observed: list[tuple[str, bool]] = []

    def _evaluate_model(current_cfg):  # type: ignore[no-untyped-def]
        observed.append(
            (
                str(current_cfg.run.split),
                bool(current_cfg.run.allow_empty_answer),
            )
        )
        return {"answer/hit@1": 1.0}, {"model": SimpleNamespace()}

    run_eval(cfg, evaluate_model=_evaluate_model, allow_default_dataset_variant=True)

    assert observed == [("train", False), ("validation", True)]
    assert (tmp_path / "metrics_train.json").exists()
    assert (tmp_path / "metrics_validation.json").exists()


def test_validate_train_config_requires_fit_schedule_for_training() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "src.subgraph_gflownet.adapters.lightning.module.GFlowNetModule"
            },
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {"train": True},
            "fit_schedule": None,
        }
    )

    with pytest.raises(ValueError, match="fit_schedule"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_full_dataset() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "src.subgraph_gflownet.adapters.lightning.module.GFlowNetModule"
            },
            "dataset": {"name": "webqsp", "dataset_scope": "full"},
        }
    )

    with pytest.raises(ValueError, match="sub datasets only"):
        validate_train_config(cfg)
