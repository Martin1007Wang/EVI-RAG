from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logging_utils import get_logger
from src.utils.run_name import set_run_name_in_config

log = get_logger(__name__)


def _has_model_checkpoint(trainer: L.Trainer) -> bool:
    callbacks = getattr(trainer, "callbacks", [])
    return any(isinstance(cb, ModelCheckpoint) for cb in callbacks)


def _best_checkpoint_available(trainer: L.Trainer) -> bool:
    callbacks = getattr(trainer, "callbacks", [])
    for cb in callbacks:
        if not isinstance(cb, ModelCheckpoint):
            continue
        if cb.monitor is None or cb.save_top_k == 0:
            continue
        if getattr(cb, "best_model_path", ""):
            return True
    return False


def _to_float(value: Any) -> float:
    """Safely convert a metric value (tensor or scalar) to Python float."""
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    # ------------------------------------------------------------------ #
    # 0. Global PyTorch settings                                           #
    # ------------------------------------------------------------------ #
    torch.set_float32_matmul_precision("high")
    detect_anomaly = bool(cfg.get("detect_anomaly", False))
    torch.autograd.set_detect_anomaly(detect_anomaly)
    if detect_anomaly:
        log.warning("Anomaly detection is ON; expect significant slowdown.")

    # ------------------------------------------------------------------ #
    # 1. Reproducibility & run identity                                   #
    # ------------------------------------------------------------------ #
    L.seed_everything(cfg.get("seed", 42), workers=True)
    run_name = set_run_name_in_config(cfg)
    log.info("Starting training run: %s", run_name)

    # ------------------------------------------------------------------ #
    # 2. DataModule & Model                                               #
    # ------------------------------------------------------------------ #
    log.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    # ------------------------------------------------------------------ #
    # 3. Callbacks                                                        #
    # ------------------------------------------------------------------ #
    # _convert_="all" ensures nested _target_ dicts become a plain list,
    # avoiding the dict-vs-list ambiguity at runtime.
    callbacks: list[L.Callback] = []
    if cfg.get("callbacks"):
        instantiated = hydra.utils.instantiate(cfg.callbacks, _convert_="all")
        if isinstance(instantiated, dict):
            callbacks = list(instantiated.values())
        elif isinstance(instantiated, list):
            callbacks = instantiated

    # ------------------------------------------------------------------ #
    # 4. Logger & Trainer                                                 #
    # ------------------------------------------------------------------ #
    experiment_logger = hydra.utils.instantiate(cfg.get("logger", None))
    if experiment_logger is None:
        log.warning("No logger configured; metrics will not be persisted.")

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=experiment_logger,
    )

    # ------------------------------------------------------------------ #
    # 5. Training                                                         #
    # ------------------------------------------------------------------ #
    # Resume / fine-tune via cfg.ckpt_path.
    # Weight-only loading (e.g. for transfer) belongs in the model __init__.
    log.info("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
    )

    # ------------------------------------------------------------------ #
    # 6. Testing                                                          #
    # ------------------------------------------------------------------ #
    if cfg.get("test", False):
        log.info("Starting testing...")
        # Fall back to in-memory weights when no usable best checkpoint exists,
        # which is common in short smoke/debug runs.
        ckpt_path = None
        if _has_model_checkpoint(trainer):
            if _best_checkpoint_available(trainer):
                ckpt_path = "best"
            else:
                log.warning("ModelCheckpoint is configured but no best checkpoint was saved; " "testing with current in-memory weights.")
        else:
            log.warning("No ModelCheckpoint callback found; testing with current in-memory weights.")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # ------------------------------------------------------------------ #
    # 7. Return optimised metric for HPO sweeps                          #
    # ------------------------------------------------------------------ #
    # callback_metrics already contains test metrics after trainer.test(),
    # so no second merge is needed.
    metrics = dict(trainer.callback_metrics)
    optimized_metric = cfg.get("optimized_metric")

    if optimized_metric and optimized_metric in metrics:
        return _to_float(metrics[optimized_metric])

    return None


if __name__ == "__main__":
    main()
