from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, instantiate_callbacks, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

_STAGE_REQUIRES_CKPT_KIND = {
    "cache_g_agent": "retriever",
    "materialize_g_agent": "retriever",
    "retriever_eval": "retriever",
    "gflownet_eval": "gflownet",
    "gflownet_export": "gflownet",
}


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
    # NOTE: Lightning `.ckpt` is not guaranteed to be `weights_only`-safe under torch>=2.6.
    # We explicitly load with `weights_only=False` for compatibility (assumes checkpoint is trusted).
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(ckpt_path), **load_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {ckpt_path} must be a state_dict mapping, got {type(state_dict)!r}")
    model.load_state_dict(state_dict, strict=True)


def _save_metrics(cfg: DictConfig, metrics: Dict[str, Any], *, filename: str = "metrics.json") -> Path:
    """Persist metrics to ${paths.output_dir}/<filename> (rank0-only logic handled upstream)."""

    def _to_python(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return value

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / str(filename)
    payload = {k: _to_python(v) for k, v in metrics.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def _preflight_validate(cfg: DictConfig) -> None:
    """Fail-fast on missing Hydra groups to avoid confusing OmegaConf interpolation errors."""

    if cfg.get("dataset") is None:
        raise ValueError(
            "Missing required config group: `dataset`.\n"
            "Fix:\n"
            "  python src/eval.py stage=cache_g_agent dataset=webqsp ckpt.retriever=/path/to/retriever.ckpt\n"
            "Optional (recommended): set a default dataset in `configs/local/default.yaml` (gitignored), e.g.\n"
            "  defaults:\n"
            "    - override /dataset: webqsp"
        )

    stage = cfg.get("stage") or {}
    stage_name = str(stage.get("name", "")).strip()
    required_kind = _STAGE_REQUIRES_CKPT_KIND.get(stage_name)
    if required_kind and cfg.get("ckpt_path") in (None, ""):
        raise ValueError(
            f"Stage `{stage_name}` requires `{required_kind}` checkpoint, but `ckpt_path` is empty.\n"
            f"Fix: pass `ckpt.{required_kind}=/path/to/{required_kind}.ckpt`."
        )


def _run_cache_g_agent_all_splits(cfg: DictConfig) -> None:
    stage_cfg = cfg.get("stage") or {}
    splits = stage_cfg.get("splits") or ["train", "validation", "test"]
    split_list = [str(s) for s in splits]
    if not split_list:
        raise ValueError("stage.splits must be a non-empty list when stage.run_all_splits=true.")

    for split in split_list:
        force_gt = split == "train"
        log.info("cache_g_agent: split=%s force_include_gt=%s", split, force_gt)

        with open_dict(cfg):
            cfg.stage.split = split
            cfg.stage.force_include_gt = force_gt
        evaluate(cfg)


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
    return_predictions = bool(stage.get("run", {}).get("return_predictions", False))

    log.info(f"Stage: {stage.get('name')} (return_predictions={return_predictions})")

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

    log.info("Running trainer.predict()...")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=None, return_predictions=return_predictions)

    metric_dict = trainer.callback_metrics
    if not metric_dict and hasattr(model, "predict_metrics"):
        try:
            metrics_from_model = getattr(model, "predict_metrics")
            if isinstance(metrics_from_model, dict):
                metric_dict = metrics_from_model
        except Exception:
            pass
    try:
        metrics_filename = "metrics.json"
        stage_cfg = cfg.get("stage") or {}
        split = stage_cfg.get("split")
        if bool(stage_cfg.get("run_all_splits", False)) and split not in (None, ""):
            metrics_filename = f"metrics_{split}.json"
        metrics_path = _save_metrics(cfg, metric_dict, filename=metrics_filename)
        log.info("Metrics saved to %s", metrics_path)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to save metrics.json: %s", exc)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    _preflight_validate(cfg)
    extras(cfg)
    stage_cfg = cfg.get("stage") or {}
    if str(stage_cfg.get("name", "")).strip() == "cache_g_agent" and bool(stage_cfg.get("run_all_splits", False)):
        _run_cache_g_agent_all_splits(cfg)
        return
    evaluate(cfg)


if __name__ == "__main__":
    main()
