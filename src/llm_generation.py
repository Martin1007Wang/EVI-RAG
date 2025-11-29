from __future__ import annotations

from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import rootutils
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.llm_generation.datamodule import LLMGenerationDataModule  # noqa: E402
from src.llm_generation.module import LLMGenerationModule  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.run_context import apply_run_name  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    resolved_run_name = apply_run_name(cfg)
    log.info(f"Resolved run name: {resolved_run_name}")

    datamodule: LLMGenerationDataModule = hydra.utils.instantiate(cfg.data)
    model: LLMGenerationModule = hydra.utils.instantiate(cfg.model)

    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }
    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    datamodule.setup(stage="predict")
    trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())
    return trainer.callback_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="llm_generation.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    predict(cfg)
    return None


if __name__ == "__main__":
    main()
