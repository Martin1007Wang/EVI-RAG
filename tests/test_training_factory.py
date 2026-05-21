from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from lightning import LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from src.training.factory import build_datamodule, build_model, instantiate_list, trainer_logger
from src.weaver.module import WeaverModule, stop_branch_gradient_metrics
from src.weaver.nn.feature_encoder import FeatureEncoder
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.replay import ReplayBuilder
from src.weaver.utility import TrueTerminalReward


def test_build_model_instantiates_probability_db_graph() -> None:
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/weaver.yaml")})
    resources = SimpleNamespace(
        entity_text_semantic_table=torch.eye(4, 2, dtype=torch.float32),
        text_row_by_entity_id=torch.arange(4, dtype=torch.long),
        relation_semantic_table=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    model = build_model(cfg, resources)

    assert isinstance(model, WeaverModule)
    assert isinstance(model.policy_feature_encoder, FeatureEncoder)
    assert isinstance(model.policy, ForwardPolicy)
    assert isinstance(model.reward_model, TrueTerminalReward)
    assert torch.equal(
        model.policy_feature_encoder.text_row_by_entity_id,
        resources.text_row_by_entity_id,
    )
    assert model.runner.replay_source is not None
    assert isinstance(model.runner.replay_builder, ReplayBuilder)
    assert model.runner.replay_schedule is not None
    budget = model.runner.sample_budget(10)
    assert budget.policy_rollout == 5
    assert budget.replay_expand == 5
    assert model.runner.eval_num_rollouts == 16
    assert model.metric_suite.k_windows == (1, 2, 4, 8, 16)
    assert model.train_temperature == 1.0
    assert model.eval_temperature == 1.0
    assert hasattr(model.policy, "stop_head")
    assert hasattr(model.policy, "edge_head")
    assert hasattr(model.policy, "action_logits")
    assert hasattr(model.policy, "state_encoder")


def compose_train_config(*overrides: str):
    config_dir = str((Path.cwd() / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="train", overrides=list(overrides))


def test_instantiate_list_accepts_hydra_mapping_groups() -> None:
    cfg = compose_train_config("experiment=train/webqsp", "trainer=cpu")

    callbacks = instantiate_list(cfg.callbacks, "cfg.callbacks")

    assert len(callbacks) == 4
    assert any(isinstance(callback, ModelCheckpoint) for callback in callbacks)
    assert any(isinstance(callback, LearningRateMonitor) for callback in callbacks)
    early_stopping = next(callback for callback in callbacks if isinstance(callback, EarlyStopping))
    checkpoint = next(callback for callback in callbacks if isinstance(callback, ModelCheckpoint))
    assert early_stopping.patience == 20
    assert early_stopping.monitor == "val/best@8/target_recall"
    assert checkpoint.monitor == "val/best@8/target_recall"


def test_trainer_logger_accepts_single_hydra_target_mapping() -> None:
    cfg = compose_train_config("experiment=train/webqsp", "trainer=cpu")
    logger = trainer_logger(cfg.logger)
    assert isinstance(logger, WandbLogger)


def test_manual_optimization_keeps_gradient_clipping_out_of_trainer() -> None:
    cfg = compose_train_config("experiment=train/webqsp", "trainer=cpu")

    assert cfg.trainer.gradient_clip_val is None
    assert cfg.model.gradient_clip_val == 1.0
    assert cfg.model.gradient_clip_algorithm == "norm"


def test_stop_branch_gradient_metrics_reports_conflict_cosine() -> None:
    stop_head = torch.nn.Linear(2, 1, bias=False)
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    terminal_loss = stop_head(x).sum()
    expansion_loss = -stop_head(x).sum()

    metrics = stop_branch_gradient_metrics(
        stop_head=stop_head,
        terminal_loss=terminal_loss,
        expansion_loss=expansion_loss,
    )

    assert metrics["grad/stop_head/from_terminal_loss"].gt(0)
    assert metrics["grad/stop_head/from_expansion_loss"].gt(0)
    assert torch.allclose(
        metrics["grad/stop_head/terminal_expansion_cosine"],
        torch.tensor(-1.0),
    )


@dataclass(frozen=True)
class _MaterializationStub:
    split_paths: dict[str, str]

    def require_split(self, split: str) -> str:
        return self.split_paths[split]


@dataclass(frozen=True)
class _DataConfigStub:
    materialization: _MaterializationStub
    model_resources: object
    batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 0
    eval_num_workers: int = 0
    pin_memory: bool = False
    train_shuffle: bool = False
    drop_last: bool = False
    eval_drop_last: bool = False
    lmdb_readahead: bool = False
    max_readers: int = 1
    metadata_dir: Path = Path(".")
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"


class _DatamoduleProbe(LightningDataModule):
    def __init__(self, data_config: _DataConfigStub) -> None:
        super().__init__()
        self.data_config = data_config
        self.materialization = data_config.materialization


def test_build_datamodule_preserves_runtime_dataclass_objects(monkeypatch) -> None:
    materialization = _MaterializationStub(
        split_paths={
            "train": "train-path",
            "validation": "validation-path",
            "test": "test-path",
        }
    )
    data_config = _DataConfigStub(
        materialization=materialization,
        model_resources=SimpleNamespace(),
    )
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "tests.test_training_factory._DatamoduleProbe",
            }
        }
    )

    monkeypatch.setattr("src.training.factory.build_training_data_config", lambda _: data_config)

    datamodule = build_datamodule(cfg)

    assert isinstance(datamodule, LightningDataModule)
    assert type(datamodule).__name__ == "_DatamoduleProbe"
    assert datamodule.data_config == data_config
    assert datamodule.materialization == materialization
    assert datamodule.materialization.require_split("train") == "train-path"
