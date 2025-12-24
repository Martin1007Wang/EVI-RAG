import pytest

hydra = pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_reasoner_oracle_experiment_wiring() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=reasoner_oracle",
                "dataset=webqsp",
                "trainer=cpu",
                "logger=none",
                "callbacks=none",
            ],
        )

    assert cfg.data._target_ == "src.data.reasoner_oracle_datamodule.ReasonerOracleDataModule"
    assert cfg.model._target_ == "src.models.reasoner_module.ReasonerModule"
