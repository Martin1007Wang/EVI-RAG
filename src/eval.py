from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.run_context import apply_run_name

log = RankedLogger(__name__, rank_zero_only=True)


def _normalize_ckpt_value(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if set(text) <= {"?"}:
        return None
    return text


def _discover_latest_checkpoint(run_name: str, log_dir: Path) -> Path:
    run_root = (log_dir / run_name / "runs").expanduser()
    if not run_root.exists():
        raise FileNotFoundError(f"No run directory found at {run_root}")

    run_dirs = sorted(
        (p for p in run_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in run_dirs:
        ckpt_dir = candidate / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        best_ckpts = sorted(
            (p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if best_ckpts:
            return best_ckpts[0]
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.is_file():
            return last_ckpt
    raise FileNotFoundError(f"No checkpoints found under {run_root}")


def _ensure_checkpoint_path(cfg: DictConfig, run_name: str) -> str:
    normalized = _normalize_ckpt_value(cfg.get("ckpt_path"))
    if normalized:
        return normalized
    log_dir = Path(cfg.paths.log_dir)
    try:
        ckpt = _discover_latest_checkpoint(run_name, log_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Unable to auto-discover checkpoint for run '{run_name}'. "
            "Pass ckpt_path explicitly (e.g. ckpt_path=/path/to/epoch_XXX.ckpt)."
        ) from exc
    log.info(f"Auto-selected checkpoint for run '{run_name}': {ckpt}")
    with open_dict(cfg):
        cfg.ckpt_path = str(ckpt)
    return str(ckpt)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    resolved_run_name = apply_run_name(cfg)
    ckpt_path = _ensure_checkpoint_path(cfg, resolved_run_name)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if hasattr(model, "set_run_context"):
        model.set_run_context(
            run_name=resolved_run_name,
            output_dir=cfg.paths.output_dir,
            metrics_root=cfg.paths.log_dir,
        )

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
