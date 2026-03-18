from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

import pytest

from src.runs.common import (
    load_dataset_config_by_name,
    resolve_dataset_variants,
    resolve_execution_mode,
    temporary_cfg_overrides,
)


def test_load_dataset_config_by_name_merges_base_and_resolves_paths(
    tmp_path: Path,
) -> None:
    dataset_cfg = load_dataset_config_by_name(
        "webqsp-sub",
        OmegaConf.create({"data_dir": str(tmp_path)}),
    )

    assert dataset_cfg.name == "webqsp-sub"
    assert dataset_cfg.dataset_scope == "sub"
    assert dataset_cfg.dataset_family == "webqsp"
    assert dataset_cfg.out_dir == str(tmp_path / "webqsp" / "normalized")
    assert dataset_cfg.paths.entity_vocab == str(
        tmp_path / "webqsp" / "normalized" / "entity_vocab.parquet"
    )


def test_resolve_dataset_variants_loads_each_requested_dataset(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(tmp_path)},
            "run": {
                "dataset_variants": [
                    {"label": "full", "dataset": "webqsp"},
                    {"dataset": "webqsp-sub"},
                ]
            },
        }
    )

    variants = resolve_dataset_variants(cfg)

    assert [variant.label for variant in variants] == ["full", "webqsp-sub"]
    assert [variant.dataset_cfg.dataset_scope for variant in variants] == [
        "full",
        "sub",
    ]


def test_resolve_dataset_variants_applies_dataset_and_run_overrides(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "custom-artifacts"
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(tmp_path)},
            "run": {
                "dataset_variants": [
                    {
                        "label": "full_custom",
                        "dataset": "webqsp",
                        "dataset_overrides": {"artifact_dir": str(artifact_dir)},
                        "run_overrides": {
                            "split": "validation",
                            "artifact_subdir": "custom_eval",
                        },
                    }
                ]
            },
        }
    )

    variants = resolve_dataset_variants(cfg)

    assert len(variants) == 1
    assert variants[0].label == "full_custom"
    assert variants[0].dataset_cfg.artifact_dir == str(artifact_dir)
    assert variants[0].run_overrides == {
        "split": "validation",
        "artifact_subdir": "custom_eval",
    }


def test_resolve_dataset_variants_rejects_removed_name_key(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(tmp_path)},
            "run": {"dataset_variants": [{"name": "webqsp"}]},
        }
    )

    with pytest.raises(ValueError, match="use `dataset`, not the removed `name` key"):
        resolve_dataset_variants(cfg)


def test_resolve_dataset_variants_rejects_removed_compose_overrides_key(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(tmp_path)},
            "run": {
                "dataset_variants": [
                    {
                        "dataset": "webqsp",
                        "compose_overrides": ["dataset.out_dir=/tmp/normalized"],
                    }
                ]
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="use `overrides`, not the removed `compose_overrides` key",
    ):
        resolve_dataset_variants(cfg)


def test_train_defaults_target_rankflow_run() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs" / "train.yaml")
    defaults = OmegaConf.to_container(cfg.defaults, resolve=False)

    assert {"run": "train_rankflow"} in defaults


def test_default_callbacks_monitor_current_rank_metric() -> None:
    callbacks_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "configs" / "callbacks" / "default.yaml"
    )
    callbacks_dict = OmegaConf.to_container(callbacks_cfg, resolve=False)

    assert (
        callbacks_dict["model_checkpoint"]["monitor"]
        == "val/${dataset.dataset_scope}/answer/recall@10"
    )
    assert (
        callbacks_dict["early_stopping"]["monitor"]
        == "val/${dataset.dataset_scope}/answer/recall@10"
    )


def test_resolve_execution_mode_accepts_execution_mode_key() -> None:
    assert resolve_execution_mode({"execution_mode": "test"}) == "test"


def test_resolve_execution_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="run.execution_mode"):
        resolve_execution_mode({"execution_mode": "invalid"})


def test_temporary_cfg_overrides_restores_original_nodes() -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": "/tmp/out"},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {"split": "test", "artifact_subdir": "base_eval"},
        }
    )

    with temporary_cfg_overrides(
        cfg,
        dataset_cfg={"name": "webqsp", "dataset_scope": "full"},
        run_overrides={"split": "validation"},
        paths_overrides={"output_dir": "/tmp/override"},
    ):
        assert cfg.dataset.name == "webqsp"
        assert cfg.run.split == "validation"
        assert cfg.paths.output_dir == "/tmp/override"

    assert cfg.dataset.name == "webqsp-sub"
    assert cfg.run.split == "test"
    assert cfg.paths.output_dir == "/tmp/out"
