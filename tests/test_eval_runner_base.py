from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.runs.common import DatasetVariantSpec
from src.runs.eval_runner_base import BaseEvalRunner


@dataclass
class _DummyEvalRunner(BaseEvalRunner):
    recorded: list[tuple[str | None, str, str]] = field(default_factory=list)
    variants: list[DatasetVariantSpec] = field(default_factory=list)

    def _run_once(self, *, cfg, evaluate_model) -> None:  # type: ignore[no-untyped-def]
        del evaluate_model
        self.recorded.append(
            (
                cfg.run.get("dataset_variant"),
                str(cfg.run.split),
                str(cfg.dataset.name),
            )
        )

    def _supports_dataset_variants(self) -> bool:
        return True

    def _resolve_dataset_variants(self, cfg):  # type: ignore[no-untyped-def]
        del cfg
        return list(self.variants)

    def _logger(self):
        return SimpleNamespace(info=lambda *args, **kwargs: None)


def test_base_eval_runner_replays_requested_splits_and_restores_cfg() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp-sub"},
            "run": {"split": "test"},
            "paths": {"output_dir": "/tmp/out"},
        }
    )
    runner = _DummyEvalRunner(run_all_splits=True, splits=("validation", "test"))

    runner.run(cfg=cfg, evaluate_model=lambda _cfg: ({}, {}))

    assert runner.recorded == [
        (None, "validation", "webqsp-sub"),
        (None, "test", "webqsp-sub"),
    ]
    assert cfg.run.split == "test"


def test_base_eval_runner_applies_dataset_variant_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp-sub"},
            "run": {"split": "test", "dataset_variant": None},
            "paths": {"output_dir": "/tmp/out"},
        }
    )
    runner = _DummyEvalRunner(
        dataset_variants=["enabled"],
        variants=[
            DatasetVariantSpec(
                label="full_eval",
                dataset_name="webqsp",
                dataset_cfg=OmegaConf.create({"name": "webqsp"}),
                run_overrides={"split": "validation"},
            ),
            DatasetVariantSpec(
                label="sub_eval",
                dataset_name="webqsp-sub",
                dataset_cfg=OmegaConf.create({"name": "webqsp-sub"}),
                run_overrides={},
            ),
        ],
    )

    runner.run(cfg=cfg, evaluate_model=lambda _cfg: ({}, {}))

    assert runner.recorded == [
        ("full_eval", "validation", "webqsp"),
        ("sub_eval", "test", "webqsp-sub"),
    ]
    assert cfg.dataset.name == "webqsp-sub"
    assert cfg.run.dataset_variant is None
