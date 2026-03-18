from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import pytest

from src.runs.answer_reachability import (
    AnswerReachabilityEvalRunner,
    validate_train_config,
)


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
    runner = AnswerReachabilityEvalRunner(
        run_all_splits=True, splits=("train", "validation")
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

    runner.run(cfg=cfg, evaluate_model=_evaluate_model)

    assert observed == [("train", False), ("validation", True)]
    assert (tmp_path / "metrics_train.json").exists()
    assert (tmp_path / "metrics_validation.json").exists()


def test_rankflow_runner_applies_variant_run_overrides(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path), "data_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {
                "name": "rankflow",
                "split": "test",
                "dataset_variants": [
                    {
                        "label": "full_eval",
                        "dataset": "webqsp",
                        "run_overrides": {
                            "split": "validation",
                            "artifact_subdir": "full_eval_outputs",
                        },
                    },
                    {
                        "label": "sub_eval",
                        "dataset": "webqsp-sub",
                    },
                ],
                "execution_mode": "predict",
            },
        }
    )
    runner = AnswerReachabilityEvalRunner(
        dataset_variants=tuple(cfg.run.dataset_variants),
    )
    observed: list[tuple[str, str, str | None]] = []

    def _evaluate_model(current_cfg):  # type: ignore[no-untyped-def]
        observed.append(
            (
                str(current_cfg.run.dataset_variant),
                str(current_cfg.run.split),
                current_cfg.run.get("artifact_subdir"),
            )
        )
        return {}, {"model": SimpleNamespace()}

    runner.run(cfg=cfg, evaluate_model=_evaluate_model)

    assert observed == [
        ("full_eval", "validation", "full_eval_outputs"),
        ("sub_eval", "test", None),
    ]


def test_validate_train_config_requires_fit_schedule_for_training() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "src.models.gflownet_module.GFlowNetModule"},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {"train": True},
            "fit_schedule": None,
        }
    )

    with pytest.raises(ValueError, match="fit_schedule"):
        validate_train_config(cfg)
