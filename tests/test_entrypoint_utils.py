from __future__ import annotations

from types import SimpleNamespace

import src.runs.lightning as entrypoint_utils
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf


def test_instantiate_lightning_task_objects_wires_components(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "model": {"_target_": "tests.Model"},
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
                "dataset_cfg": "${dataset}",
                "eval_split": "${oc.select:run.split,test}",
            },
            "model": {
                "_target_": "tests.Model",
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


def test_instantiate_lightning_task_objects_normalizes_trainer_max_steps(
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "model": {"_target_": "tests.Model"},
            "trainer": {"_target_": "tests.Trainer", "max_steps": None},
            "callbacks": None,
            "logger": None,
        }
    )
    seen: dict[str, object] = {}

    def _fake_instantiate(config, **kwargs):  # type: ignore[no-untyped-def]
        target = str(config.get("_target_"))
        if target == "tests.DataModule":
            return SimpleNamespace(name="data")
        if target == "tests.Model":
            return SimpleNamespace(name="model")
        if target == "tests.Trainer":
            seen["trainer_config"] = OmegaConf.to_container(config, resolve=True)
            return SimpleNamespace(name="trainer", kwargs=kwargs)
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(entrypoint_utils.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(entrypoint_utils, "instantiate_callbacks", lambda _: [])
    monkeypatch.setattr(entrypoint_utils, "instantiate_loggers", lambda _: [])

    entrypoint_utils.instantiate_lightning_task_objects(
        cfg,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert seen["trainer_config"] == {"_target_": "tests.Trainer", "max_steps": -1}
    assert cfg.trainer.max_steps is None


def test_instantiate_lightning_task_objects_drops_model_checkpoint_when_disabled(
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "model": {"_target_": "tests.Model"},
            "trainer": {
                "_target_": "tests.Trainer",
                "enable_checkpointing": False,
            },
            "callbacks": None,
            "logger": None,
        }
    )
    seen: dict[str, object] = {}
    checkpoint_callback = ModelCheckpoint(dirpath="/tmp")
    callbacks = [SimpleNamespace(name="plain"), checkpoint_callback]

    def _fake_instantiate(config, **kwargs):  # type: ignore[no-untyped-def]
        target = str(config.get("_target_"))
        if target == "tests.DataModule":
            return SimpleNamespace(name="data")
        if target == "tests.Model":
            return SimpleNamespace(name="model")
        if target == "tests.Trainer":
            seen["trainer_callbacks"] = kwargs["callbacks"]
            return SimpleNamespace(name="trainer", kwargs=kwargs)
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(entrypoint_utils.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(entrypoint_utils, "instantiate_callbacks", lambda _: callbacks)
    monkeypatch.setattr(entrypoint_utils, "instantiate_loggers", lambda _: [])

    entrypoint_utils.instantiate_lightning_task_objects(
        cfg,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert seen["trainer_callbacks"] == [callbacks[0]]
