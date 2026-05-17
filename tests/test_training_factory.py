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
from src.weaver.module import WeaverModule
from src.weaver.nn.feature_encoder import FeatureEncoder
from src.weaver.reward import EvidenceLogReward


def test_build_model_instantiates_probability_db_graph() -> None:
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/weaver.yaml")})
    resources = SimpleNamespace(
        entity_text_embeddings=torch.eye(4, 2, dtype=torch.float32),
        entity_embedding_map=torch.arange(4, dtype=torch.long),
        relation_embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    model = build_model(cfg, resources)

    assert isinstance(model, WeaverModule)
    assert isinstance(model.feature_encoder, FeatureEncoder)
    assert isinstance(model.reward_model, EvidenceLogReward)
    assert model.policy.max_budget == 3
    assert model.policy.state_encoder.max_budget == 3
    assert torch.equal(
        model.feature_encoder.entity_embedding.entity_embedding_map,
        resources.entity_embedding_map,
    )
    assert model.runner.replay_schedule.enabled
    assert model.runner.sample_budget(10).policy_rollout == 2
    assert model.runner.sample_budget(10).replay_expand == 8
    assert model.policy.edge_scorer.adapter_final_init_scale == 0.0
    assert model.policy.edge_scorer.semantic_prior_scale == 10.0
    assert model.policy.edge_scorer.edge_logit_shift is None
    assert not model.feature_encoder.edge_encoder.role_logits.requires_grad
    assert hasattr(model.policy, "stop_head")
    assert not hasattr(model.policy, "terminal_utility_estimator")


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
    assert early_stopping.monitor == "val/main/utility_at_8"
    assert checkpoint.monitor == "val/main/utility_at_8"


def test_trainer_logger_accepts_single_hydra_target_mapping() -> None:
    cfg = compose_train_config("experiment=train/webqsp", "trainer=cpu")
    logger = trainer_logger(cfg.logger)
    assert isinstance(logger, WandbLogger)


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
