from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.runs.answer_reachability import (
    ANSWER_REACHABILITY_MODEL_TARGET,
    collect_model_metrics,
    resolve_metrics_filename,
    validate_train_config,
)


def test_validate_train_config_rejects_full_dataset() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"_target_": ANSWER_REACHABILITY_MODEL_TARGET},
            "dataset": {"name": "webqsp", "dataset_scope": "full"},
        }
    )

    with pytest.raises(ValueError, match="sub datasets only"):
        validate_train_config(cfg)


def test_resolve_metrics_filename_includes_scope_and_split() -> None:
    filename = resolve_metrics_filename(
        run_cfg={"dataset_variant": "webqsp", "run_all_splits": True, "split": "test"},
        dataset_cfg={"name": "webqsp-sub", "dataset_scope": "sub"},
    )

    assert filename == "metrics_sub_test.json"


def test_collect_model_metrics_prefers_model_snapshot_when_callbacks_empty() -> None:
    model = SimpleNamespace(get_predict_metrics=lambda: {"answer/hit@1": 0.5})

    metric_dict = collect_model_metrics(callback_metrics={}, model=model)

    assert metric_dict == {"answer/hit@1": 0.5}
