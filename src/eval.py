from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, instantiate_callbacks, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _enforce_single_gpu_eval(trainer_cfg: DictConfig) -> None:
    accelerator = str(trainer_cfg.get("accelerator", "")).lower()
    if accelerator not in {"gpu", "cuda"}:
        raise ValueError(
            "Eval 禁止使用非 GPU accelerator。"
            f"Got trainer.accelerator={trainer_cfg.get('accelerator')!r}. "
            "Fix: set `trainer.accelerator=gpu` (and keep `trainer.devices=1`)."
        )

    devices = trainer_cfg.get("devices", None)
    num_devices: Optional[int]
    if devices is None:
        num_devices = None
    elif isinstance(devices, int):
        num_devices = int(devices)
    elif isinstance(devices, (list, tuple)):
        num_devices = len(devices)
    elif isinstance(devices, str):
        raw = devices.strip().lower()
        if raw == "auto":
            num_devices = None
        elif raw.isdigit():
            num_devices = int(raw)
        elif "," in raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            num_devices = len(parts)
        else:
            num_devices = None
    else:
        num_devices = None

    if num_devices != 1:
        raise ValueError(
            "Eval 严禁多卡/自动多卡（DDP）以保证样本数与指标聚合不被分片。"
            f"Got trainer.devices={devices!r} (parsed_num_devices={num_devices!r}). "
            "Fix: set `trainer.devices=1` (optionally select GPU via CUDA_VISIBLE_DEVICES)."
        )

    strategy = trainer_cfg.get("strategy", "auto")
    strategy_name = str(strategy).lower()
    if any(tag in strategy_name for tag in ("ddp", "fsdp", "deepspeed")):
        raise ValueError(
            "Eval 禁止分布式 strategy。"
            f"Got trainer.strategy={strategy!r}. "
            "Fix: remove the override or set `trainer.strategy=auto`."
        )


def _load_checkpoint_strict(model: LightningModule, ckpt_path: Optional[str]) -> None:
    if ckpt_path in (None, ""):
        return
    load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(ckpt_path), **load_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {ckpt_path} must be a state_dict mapping, got {type(state_dict)!r}")
    model.load_state_dict(state_dict, strict=True)


def _maybe_initialize_model_for_strict_load(
    *,
    model: LightningModule,
    datamodule: LightningDataModule,
    loop: str,
) -> None:
    """Ensure lazily constructed submodules exist before strict state_dict loading.

    Some modules (e.g. GFlowNet) materialize parts of the network inside `setup()` after accessing
    shared resources from the DataModule. When we do a strict manual state_dict load, we must
    instantiate those parts first, otherwise keys will be reported as "unexpected".
    """

    embedder = getattr(model, "embedder", None)
    embedder_setup = getattr(embedder, "setup", None) if embedder is not None else None
    if not callable(embedder_setup):
        return

    # DataModules typically populate shared resources inside `setup()`.
    stage = "test" if loop == "test" else "predict"
    datamodule_setup = getattr(datamodule, "setup", None)
    if callable(datamodule_setup):
        datamodule_setup(stage=stage)  # type: ignore[call-arg]

    resources = getattr(datamodule, "shared_resources", None)
    if resources is None:
        raise ValueError(
            "Model requires datamodule.shared_resources for initialization before checkpoint loading, "
            "but `shared_resources` is missing. Ensure the DataModule constructs SharedDataResources in `setup()`."
        )

    embedder_setup(resources)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    stage = cfg.get("stage")
    if stage is None:
        raise ValueError("Missing required config group: `stage`. Example: `python src/eval.py stage=retriever_eval dataset=webqsp`.")
    # Leakage guard: GT injection is an oracle and must never be enabled for non-train splits.
    # We validate here (before instantiating callbacks) to fail-fast and keep the rule global.
    stage_split = stage.get("split")
    if bool(stage.get("force_include_gt", False)) and str(stage_split).lower() != "train":
        raise ValueError(
            "Data leakage detected: `stage.force_include_gt=true` is only allowed when `stage.split=train`. "
            f"Got stage.split={stage_split!r}. "
            "Fix: set `stage.force_include_gt=false` for validation/test (and only enable it for train materialization)."
        )
    loop = str(stage.get("run", {}).get("loop", "")).lower()
    return_predictions = bool(stage.get("run", {}).get("return_predictions", False))

    log.info(f"Stage: {stage.get('name')} (loop={loop}, return_predictions={return_predictions})")

    _enforce_single_gpu_eval(cfg.trainer)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Some models (notably GFlowNet) lazily build submodules in `setup()` using datamodule resources.
    # Since we do a strict manual checkpoint load, we must materialize those modules first.
    _maybe_initialize_model_for_strict_load(model=model, datamodule=datamodule, loop=loop)

    _load_checkpoint_strict(model, cfg.get("ckpt_path"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if loop == "test":
        log.info("Running trainer.test()...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    elif loop == "predict":
        log.info("Running trainer.predict()...")
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=None, return_predictions=return_predictions)
    else:
        raise ValueError(f"Unsupported stage.run.loop={loop!r}. Supported: 'test', 'predict'.")

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
