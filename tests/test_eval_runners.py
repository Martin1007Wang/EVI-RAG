from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.runs.common import DatasetVariantSpec
from src.runs.llm import validate_eval_config as validate_llm_eval_config
from src.runs.rankflow import run_eval


def test_rankflow_eval_replays_requested_splits_and_restores_cfg() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {
                "split": "test",
                "run_all_splits": True,
                "splits": ["validation", "test"],
            },
            "paths": {"output_dir": "/tmp/out"},
        }
    )
    recorded: list[tuple[str | None, str, str]] = []

    def _evaluate_model(current_cfg):  # type: ignore[no-untyped-def]
        recorded.append(
            (
                current_cfg.run.get("dataset_variant"),
                str(current_cfg.run.split),
                str(current_cfg.dataset.name),
            )
        )
        return {}, {"model": SimpleNamespace()}

    run_eval(cfg, evaluate_model=_evaluate_model, allow_default_dataset_variant=True)

    assert recorded == [
        (None, "validation", "webqsp-sub"),
        (None, "test", "webqsp-sub"),
    ]
    assert cfg.run.split == "test"


def test_rankflow_eval_applies_dataset_variant_overrides(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {"split": "test", "dataset_variant": None},
            "paths": {"output_dir": "/tmp/out"},
        }
    )
    recorded: list[tuple[str | None, str, str]] = []

    def _evaluate_model(current_cfg):  # type: ignore[no-untyped-def]
        recorded.append(
            (
                current_cfg.run.get("dataset_variant"),
                str(current_cfg.run.split),
                str(current_cfg.dataset.name),
            )
        )
        return {}, {"model": SimpleNamespace()}

    monkeypatch.setattr(
        "src.runs.rankflow.resolve_dataset_variants",
        lambda _cfg: [
            DatasetVariantSpec(
                label="full_eval",
                dataset_cfg=OmegaConf.create(
                    {"name": "webqsp", "dataset_scope": "full"}
                ),
                run_overrides={"split": "validation"},
            ),
            DatasetVariantSpec(
                label="sub_eval",
                dataset_cfg=OmegaConf.create(
                    {"name": "webqsp-sub", "dataset_scope": "sub"}
                ),
                run_overrides={},
            ),
        ],
    )

    run_eval(cfg, evaluate_model=_evaluate_model)

    assert recorded == [
        ("full_eval", "validation", "webqsp"),
        ("sub_eval", "test", "webqsp-sub"),
    ]
    assert cfg.dataset.name == "webqsp-sub"
    assert cfg.run.dataset_variant is None


def test_validate_llm_eval_config_rejects_dataset_variants() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp"},
            "run": {"name": "eval_llm", "dataset_variants": ["webqsp", "webqsp-sub"]},
            "llm": {"providers": ["openai"]},
        }
    )

    with pytest.raises(ValueError, match="does not support run.dataset_variants"):
        validate_llm_eval_config(cfg)
