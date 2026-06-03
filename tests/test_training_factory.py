from __future__ import annotations

from types import SimpleNamespace

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from omegaconf import OmegaConf
from pathlib import Path

from src.training.factory import build_model, build_trainer
from src.weaver.module import WeaverModule
from src.weaver.objectives import ObjectiveOutput
from src.weaver.policy import ForwardPolicy

CONFIG_DIR = Path("configs").resolve()


def compose_config(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_build_model_instantiates_parent_local_subtb_policy() -> None:
    cfg = OmegaConf.create(
        {
            "model": OmegaConf.load("configs/model/weaver.yaml"),
        }
    )
    resources = SimpleNamespace(
        entity_text_semantic_table=torch.zeros(3, cfg.model.sem_dim),
        text_row_by_entity_id=torch.zeros(5, dtype=torch.long),
        entity_relation_neighborhood_semantic_table=torch.zeros(0, cfg.model.sem_dim),
        relation_neighborhood_row_by_entity_id=torch.full((5,), -1, dtype=torch.long),
        relation_semantic_table=torch.zeros(4, cfg.model.sem_dim),
    )

    module = build_model(cfg, resources)

    assert isinstance(module.policy, ForwardPolicy)
    assert module.policy.state_encoder is not None
    assert module.policy.flow_estimator is not None
    assert not hasattr(module, "state_encoder")
    assert not hasattr(module.policy, "edge_scorer")
    assert not hasattr(module.policy, "stop_policy_head")
    assert not hasattr(module.policy, "partial_reward_model")
    assert module.policy.state_flow_head is not None
    assert module.terminal_reward_model is not None


def test_build_model_accepts_simplified_replay_source_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": OmegaConf.load("configs/model/weaver.yaml"),
        }
    )
    resources = SimpleNamespace(
        entity_text_semantic_table=torch.zeros(3, cfg.model.sem_dim),
        text_row_by_entity_id=torch.zeros(5, dtype=torch.long),
        entity_relation_neighborhood_semantic_table=torch.zeros(0, cfg.model.sem_dim),
        relation_neighborhood_row_by_entity_id=torch.full((5,), -1, dtype=torch.long),
        relation_semantic_table=torch.zeros(4, cfg.model.sem_dim),
    )

    module = build_model(cfg, resources)

    assert module.runner.replay_source is not None
    assert module.runner.replay_source.anneal_steps is None
    assert not hasattr(module.runner.replay_source, "max_depth")
    assert not hasattr(module.runner.replay_source, "mode")
    assert not hasattr(module.runner.replay_source, "max_states_per_graph")


def test_train_residual_metrics_are_logged_per_step() -> None:
    log_calls: list[tuple[str, object, dict[str, object]]] = []

    module = SimpleNamespace(
        log=lambda name, value, **kwargs: log_calls.append(("log", (name, value), kwargs)),
        log_dict=lambda values, **kwargs: log_calls.append(("log_dict", values, kwargs)),
    )
    output = ObjectiveOutput(
        loss=torch.tensor(1.0),
        metrics={
            "objective/loss": 1.0,
            "objective/subtb_transition_abs_residual_mean": 0.5,
            "objective/subtb_terminal_abs_residual_mean": 0.25,
        },
        num_states=1,
    )

    WeaverModule._log_train(
        module,
        batch=SimpleNamespace(num_graphs_total=2),
        output=output,
        rollout=SimpleNamespace(metrics={}),
    )

    step_values = log_calls[1][1]
    assert log_calls[1][2]["on_step"] is True
    assert log_calls[1][2]["on_epoch"] is False
    assert set(step_values) == {
        "train/objective/subtb_terminal_abs_residual_mean",
        "train/objective/subtb_transition_abs_residual_mean",
    }
    epoch_values = log_calls[2][1]
    assert log_calls[2][2]["on_step"] is False
    assert log_calls[2][2]["on_epoch"] is True
    assert "train/objective/loss" in epoch_values
    assert not set(step_values) & set(epoch_values)


def test_train_config_declares_profiler_key() -> None:
    cfg = compose_config("train")

    assert "profiler" in cfg
    assert cfg.profiler is None


def test_evaluate_config_declares_profiler_key() -> None:
    cfg = compose_config("evaluate")

    assert "profiler" in cfg
    assert cfg.profiler is None


def test_train_profiler_override_keeps_instantiable_config() -> None:
    cfg = compose_config("train", overrides=["profiler=simple"])

    assert cfg.profiler is not None
    assert cfg.profiler._target_ == "lightning.pytorch.profilers.SimpleProfiler"


def test_train_profiler_none_override_resolves_to_null() -> None:
    cfg = compose_config("train", overrides=["profiler=none"])

    assert "profiler" in cfg
    assert cfg.profiler is None


def test_build_trainer_ignores_missing_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {"_target_": "tests.fake.Trainer"},
            "callbacks": None,
            "logger": None,
        }
    )
    instantiate_calls: list[object] = []

    def fake_instantiate(config, **kwargs):
        instantiate_calls.append(config)
        if config == cfg.trainer:
            return Trainer(**kwargs)
        raise AssertionError(f"Unexpected instantiate call: {config!r}")

    monkeypatch.setattr(hydra.utils, "instantiate", fake_instantiate)

    trainer = build_trainer(cfg)

    assert isinstance(trainer, Trainer)
    assert instantiate_calls == [cfg.trainer]


def test_build_trainer_instantiates_configured_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler_cfg = OmegaConf.create({"_target_": "tests.fake.Profiler"})
    cfg = OmegaConf.create(
        {
            "trainer": {"_target_": "tests.fake.Trainer"},
            "callbacks": None,
            "logger": None,
            "profiler": profiler_cfg,
        }
    )
    profiler_obj = object()

    def fake_instantiate(config, **kwargs):
        if config == profiler_cfg:
            return profiler_obj
        if config == cfg.trainer:
            assert kwargs["profiler"] is profiler_obj
            return Trainer(**kwargs)
        raise AssertionError(f"Unexpected instantiate call: {config!r}")

    monkeypatch.setattr(hydra.utils, "instantiate", fake_instantiate)

    trainer = build_trainer(cfg)

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize("profiler_value", [{}, 123])
def test_build_trainer_rejects_invalid_profiler(
    profiler_value: object,
) -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {"_target_": "tests.fake.Trainer"},
            "callbacks": None,
            "logger": None,
            "profiler": profiler_value,
        }
    )

    with pytest.raises((TypeError, ValueError), match=r"cfg\.profiler"):
        build_trainer(cfg)
