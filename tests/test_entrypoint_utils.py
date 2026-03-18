from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import src.utils.entrypoint_utils as entrypoint_utils


def test_require_run_target_config_rejects_missing_run_group() -> None:
    cfg = OmegaConf.create({})

    with pytest.raises(ValueError, match="Missing required config group"):
        entrypoint_utils.require_run_target_config(
            cfg,
            missing_run_message="Missing required config group: `run`.",
            missing_target_message="Missing required run target: `run._target_`.",
        )


def test_require_run_target_config_rejects_missing_target() -> None:
    cfg = OmegaConf.create({"run": {}})

    with pytest.raises(ValueError, match="Missing required run target"):
        entrypoint_utils.require_run_target_config(
            cfg,
            missing_run_message="Missing required config group: `run`.",
            missing_target_message="Missing required run target: `run._target_`.",
        )


def test_instantiate_task_runner_requires_validate_and_run(monkeypatch) -> None:
    cfg = OmegaConf.create({"run": {"_target_": "tests.Runner"}})

    monkeypatch.setattr(
        entrypoint_utils.hydra.utils,
        "instantiate",
        lambda run_cfg, **kwargs: SimpleNamespace(validate=lambda current_cfg: None),
    )

    with pytest.raises(TypeError, match="run\(cfg=..., train_model=...\)"):
        entrypoint_utils.instantiate_task_runner(
            cfg.run,
            run_signature="run(cfg=..., train_model=...)",
        )


def test_instantiate_lightning_task_objects_wires_components(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "_target_": "tests.DataModule",
                "contract": {"requires_dataset_cfg": True},
            },
            "model": {
                "_target_": "tests.Model",
                "contract": {"task_family": "answer_ranking"},
            },
            "trainer": {"_target_": "tests.Trainer"},
            "callbacks": {"writer": {"_target_": "tests.Callback"}},
            "logger": {"csv": {"_target_": "tests.Logger"}},
        }
    )
    datamodule = SimpleNamespace(name="data")
    model = SimpleNamespace(name="model")
    trainer = SimpleNamespace(name="trainer")
    callbacks = [SimpleNamespace(name="callback")]
    loggers = [SimpleNamespace(name="logger")]
    seen: dict[str, object] = {}

    def _fake_instantiate(config, **kwargs):  # type: ignore[no-untyped-def]
        target = str(config.get("_target_"))
        assert "contract" not in config
        if target == "tests.DataModule":
            return datamodule
        if target == "tests.Model":
            return model
        if target == "tests.Trainer":
            seen["trainer_kwargs"] = kwargs
            return trainer
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(entrypoint_utils.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(entrypoint_utils, "instantiate_callbacks", lambda _: callbacks)
    monkeypatch.setattr(entrypoint_utils, "instantiate_loggers", lambda _: loggers)
    monkeypatch.setattr(
        entrypoint_utils,
        "log_hyperparameters",
        lambda object_dict: seen.setdefault("object_dict", object_dict),
    )

    observed_models: list[object] = []
    objects = entrypoint_utils.instantiate_lightning_task_objects(
        cfg,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
        on_model_instantiated=lambda instantiated_model: observed_models.append(
            instantiated_model
        ),
    )

    assert objects.datamodule is datamodule
    assert objects.model is model
    assert objects.trainer is trainer
    assert objects.callbacks == callbacks
    assert objects.logger == loggers
    assert observed_models == [model]
    assert seen["trainer_kwargs"] == {"callbacks": callbacks, "logger": loggers}
    assert seen["object_dict"] == objects.as_dict()


def test_instantiate_lightning_task_objects_resolves_root_interpolations(
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp-sub", "max_steps": 3},
            "run": {"split": "validation"},
            "data": {
                "_target_": "tests.DataModule",
                "contract": {"requires_dataset_cfg": True},
                "dataset_cfg": "${dataset}",
                "eval_split": "${oc.select:run.split,test}",
            },
            "model": {
                "_target_": "tests.Model",
                "contract": {"task_family": "answer_ranking"},
                "horizon_cfg": {"max_steps": "${dataset.max_steps}"},
            },
            "trainer": {"_target_": "tests.Trainer"},
            "callbacks": None,
            "logger": None,
        }
    )
    seen: dict[str, object] = {}

    def _fake_instantiate(config, **kwargs):  # type: ignore[no-untyped-def]
        target = str(config.get("_target_"))
        if target == "tests.DataModule":
            seen["data_config"] = OmegaConf.to_container(config, resolve=True)
            return SimpleNamespace(name="data")
        if target == "tests.Model":
            seen["model_config"] = OmegaConf.to_container(config, resolve=True)
            return SimpleNamespace(name="model")
        if target == "tests.Trainer":
            return SimpleNamespace(name="trainer", kwargs=kwargs)
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(entrypoint_utils.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(entrypoint_utils, "instantiate_callbacks", lambda _: [])
    monkeypatch.setattr(entrypoint_utils, "instantiate_loggers", lambda _: [])

    entrypoint_utils.instantiate_lightning_task_objects(
        cfg,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert seen["data_config"] == {
        "_target_": "tests.DataModule",
        "dataset_cfg": {"name": "webqsp-sub", "max_steps": 3},
        "eval_split": "validation",
    }
    assert seen["model_config"] == {
        "_target_": "tests.Model",
        "horizon_cfg": {"max_steps": 3},
    }


def test_instantiate_lightning_task_objects_allows_datamodule_hook_to_mutate_cfg(
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "_target_": "tests.DataModule",
            },
            "model": {
                "_target_": "tests.Model",
                "horizon_cfg": {"max_steps": "${trainer.max_steps}"},
            },
            "trainer": {"_target_": "tests.Trainer", "max_steps": 100},
            "callbacks": None,
            "logger": None,
        }
    )
    seen: dict[str, object] = {}

    def _fake_instantiate(config, **kwargs):  # type: ignore[no-untyped-def]
        target = str(config.get("_target_"))
        if target == "tests.DataModule":
            seen["data_config"] = OmegaConf.to_container(config, resolve=True)
            return SimpleNamespace(name="data")
        if target == "tests.Model":
            seen["model_config"] = OmegaConf.to_container(config, resolve=True)
            return SimpleNamespace(name="model")
        if target == "tests.Trainer":
            return SimpleNamespace(name="trainer", kwargs=kwargs)
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(entrypoint_utils.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(entrypoint_utils, "instantiate_callbacks", lambda _: [])
    monkeypatch.setattr(entrypoint_utils, "instantiate_loggers", lambda _: [])

    def _mutate_max_steps(_datamodule: object) -> None:
        cfg.trainer.max_steps = 7

    entrypoint_utils.instantiate_lightning_task_objects(
        cfg,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
        on_datamodule_instantiated=_mutate_max_steps,
    )

    assert seen["data_config"] == {"_target_": "tests.DataModule"}
    assert seen["model_config"] == {
        "_target_": "tests.Model",
        "horizon_cfg": {"max_steps": 7},
    }
