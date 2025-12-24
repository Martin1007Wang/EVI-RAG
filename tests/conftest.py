import os
from pathlib import Path

import torch  # noqa: F401  # ensure torch loads before pyarrow to avoid SIGABRT (OpenMP runtime clash)
import pytest

_MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))


@pytest.fixture(scope="package")
def cfg_train():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=train_retriever",
                "dataset=webqsp",
                "trainer=cpu",
                "logger=none",
                "callbacks=none",
            ],
        )


@pytest.fixture(scope="package")
def cfg_eval():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=eval_retriever",
                "dataset=webqsp",
                "trainer=cpu",
                "logger=none",
                "callbacks=none",
            ],
        )
