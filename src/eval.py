from __future__ import annotations

import inspect
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, instantiate_callbacks, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

_RUN_REQUIRES_CKPT_KIND = {
    "eval_gflownet": "gflownet",
}

_DATASET_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "dataset"
_ALLOW_CPU_EVAL_ENV = "GFLOWNET_ALLOW_CPU_EVAL"
_ALLOW_CPU_EVAL_ON = "1"
_ALLOW_CPU_EVAL_OFF = "0"


def _enforce_single_gpu_eval(trainer_cfg: DictConfig) -> None:
    if os.getenv(_ALLOW_CPU_EVAL_ENV, _ALLOW_CPU_EVAL_OFF) == _ALLOW_CPU_EVAL_ON:
        return
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
    def _strip_torch_compile_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state:
            return state
        if not any("._orig_mod." in key or key.startswith("_orig_mod.") for key in state):
            return state
        stripped: Dict[str, Any] = {}
        for key, value in state.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "", 1)
            else:
                new_key = key.replace("._orig_mod.", ".", 1)
            stripped[new_key] = value
        return stripped
    load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
    # NOTE: Lightning `.ckpt` is not guaranteed to be `weights_only`-safe under torch>=2.6.
    # We explicitly load with `weights_only=False` for compatibility (assumes checkpoint is trusted).
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(ckpt_path), **load_kwargs)
    if isinstance(checkpoint, dict):
        try:
            model.on_load_checkpoint(checkpoint)
        except Exception as exc:
            raise RuntimeError(f"Failed to apply checkpoint metadata before strict load: {exc}") from exc
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {ckpt_path} must be a state_dict mapping, got {type(state_dict)!r}")
    state_dict = _strip_torch_compile_prefix(state_dict)
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


def _normalize_dataset_scope(dataset_cfg: DictConfig | Dict[str, Any]) -> str:
    scope_raw = dataset_cfg.get("dataset_scope") if hasattr(dataset_cfg, "get") else None
    scope = str(scope_raw or "").strip().lower()
    if scope in {"full", "sub"}:
        return scope
    name_raw = dataset_cfg.get("name") if hasattr(dataset_cfg, "get") else ""
    name = str(name_raw or "")
    return "sub" if name.endswith("-sub") else "full"


def _load_dataset_config_by_name(name: str, paths_cfg: DictConfig) -> DictConfig:
    path = _DATASET_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    raw_cfg = OmegaConf.load(path)
    container = OmegaConf.create({"paths": paths_cfg, "dataset": raw_cfg})
    OmegaConf.resolve(container)
    return container["dataset"]


def _resolve_dataset_variants(cfg: DictConfig) -> List[Tuple[str, DictConfig]]:
    run_cfg = cfg.get("run") or {}
    raw_variants = run_cfg.get("dataset_variants")
    if not raw_variants:
        return []
    variants: List[Tuple[str, DictConfig]] = []
    if OmegaConf.is_list(raw_variants) or isinstance(raw_variants, (list, tuple)):
        items = list(raw_variants)
    else:
        items = [raw_variants]
    for item in items:
        label: str
        dataset_name: str
        if isinstance(item, dict):
            dataset_name = str(item.get("dataset") or item.get("name") or "").strip()
            label = str(item.get("label") or dataset_name).strip()
        else:
            dataset_name = str(item).strip()
            label = dataset_name
        if not dataset_name:
            raise ValueError("dataset_variants entries must define a dataset name.")
        dataset_cfg = _load_dataset_config_by_name(dataset_name, cfg.paths)
        variants.append((label, dataset_cfg))
    return variants


def _resolve_eval_mode(run_cfg: DictConfig | Dict[str, Any]) -> str:
    raw = run_cfg.get("eval_mode") if hasattr(run_cfg, "get") else None
    mode = str(raw or "predict").strip().lower()
    if mode in {"predict", "test"}:
        return mode
    raise ValueError("run.eval_mode must be one of {'predict', 'test'}.")



def _preflight_validate(cfg: DictConfig) -> None:
    """Fail-fast on missing Hydra groups to avoid confusing OmegaConf interpolation errors."""

    if cfg.get("dataset") is None:
        raise ValueError(
            "Missing required config group: `dataset`.\n"
            "Fix:\n"
            "  python src/eval.py experiment=eval_gflownet dataset=webqsp-sub ckpt.gflownet=/path/to/gflownet.ckpt\n"
            "Optional (recommended): set a default dataset in `configs/local/default.yaml` (gitignored), e.g.\n"
            "  defaults:\n"
            "    - override /dataset: webqsp"
        )

    run_cfg = cfg.get("run") or {}
    run_name = str(run_cfg.get("name", "")).strip()
    if run_name in ("", "null", "None"):
        raise ValueError(
            "Missing required config group: `run`.\n"
            "Fix:\n"
            "  python src/eval.py experiment=eval_gflownet dataset=webqsp-sub ckpt.gflownet=/path/to/gflownet.ckpt\n"
        )
    required_kind = _RUN_REQUIRES_CKPT_KIND.get(run_name)
    if required_kind and cfg.get("ckpt_path") in (None, ""):
        raise ValueError(
            f"Run `{run_name}` requires `{required_kind}` checkpoint, but `ckpt_path` is empty.\n"
            f"Fix: pass `ckpt.{required_kind}=/path/to/{required_kind}.ckpt`."
        )
    variants = _resolve_dataset_variants(cfg)
    require_dual = bool(run_cfg.get("require_dual_datasets", False))
    if not variants:
        if require_dual:
            raise ValueError(
                "Dual-dataset evaluation is required, but run.dataset_variants is empty. "
                "Provide both full and sub dataset names."
            )
        return
    if require_dual:
        scopes = {_normalize_dataset_scope(ds_cfg) for _, ds_cfg in variants}
        if scopes != {"full", "sub"}:
            names = [label for label, _ in variants]
            raise ValueError(
                "Dual-dataset evaluation requires both full and sub scopes. "
                f"Got scopes={sorted(scopes)} for variants={names}."
            )


def _run_eval_all_splits(cfg: DictConfig) -> None:
    run_cfg = cfg.get("run") or {}
    splits = run_cfg.get("splits") or ["train", "validation", "test"]
    split_list = [str(s) for s in splits]
    if not split_list:
        raise ValueError("run.splits must be a non-empty list when run.run_all_splits=true.")

    for split in split_list:
        log.info("eval: split=%s", split)

        with open_dict(cfg):
            cfg.run.split = split
            if cfg.run.get("allow_empty_answer") is None:
                cfg.run.allow_empty_answer = split != "train"
        evaluate(cfg)


def _run_eval_all_datasets(cfg: DictConfig) -> None:
    run_cfg = cfg.get("run") or {}
    variants = _resolve_dataset_variants(cfg)
    if not variants:
        raise ValueError("run.dataset_variants must be a non-empty list when evaluating multiple datasets.")

    if bool(run_cfg.get("require_dual_datasets", False)):
        scopes = { _normalize_dataset_scope(ds_cfg) for _, ds_cfg in variants }
        if scopes != {"full", "sub"}:
            names = [label for label, _ in variants]
            raise ValueError(
                "Dual-dataset evaluation requires both full and sub scopes. "
                f"Got scopes={sorted(scopes)} for variants={names}."
            )

    base_output_dir = cfg.paths.output_dir
    for label, dataset_cfg in variants:
        log.info("eval: dataset_variant=%s", label)
        with open_dict(cfg):
            cfg.dataset = dataset_cfg
            cfg.run.dataset_variant = label
            cfg.paths.output_dir = base_output_dir
        if bool(run_cfg.get("run_all_splits", False)):
            _run_eval_all_splits(cfg)
        else:
            evaluate(cfg)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(
            "Missing required config group: `run`. Example: "
            "`python src/eval.py experiment=eval_gflownet dataset=webqsp-sub`."
        )
    split = str(run_cfg.get("split", "test"))
    if run_cfg.get("allow_empty_answer") is None:
        with open_dict(cfg):
            cfg.run.allow_empty_answer = split != "train"
    if bool(run_cfg.get("filter_missing_start", False)):
        scope = _normalize_dataset_scope(cfg.dataset)
        if scope == "full":
            with open_dict(cfg.dataset):
                cfg.dataset.filter_missing_start = True
            log.info("eval: filter_missing_start enabled for scope=%s split=%s", scope, split)
    log.info("Run: %s", run_cfg.get("name"))

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

    eval_mode = _resolve_eval_mode(cfg.get("run") or {})
    if eval_mode == "test":
        log.info("Running trainer.test()...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None, verbose=False)
    else:
        log.info("Running trainer.predict()...")
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=None, return_predictions=True)

    metric_dict = trainer.callback_metrics
    if not metric_dict and hasattr(model, "predict_metrics"):
        try:
            metrics_from_model = getattr(model, "predict_metrics")
            if isinstance(metrics_from_model, dict):
                metric_dict = metrics_from_model
        except Exception:
            pass
    run_cfg = cfg.get("run") or {}
    if not metric_dict:
        log.warning("No metrics were produced; skipping metrics.json.")
    else:
        metrics_filename = "metrics.json"
        split = run_cfg.get("split")
        dataset_variant = run_cfg.get("dataset_variant")
        if dataset_variant:
            scope = _normalize_dataset_scope(cfg.dataset)
            metrics_filename = f"metrics_{scope}.json"
        if bool(run_cfg.get("run_all_splits", False)) and split not in (None, ""):
            prefix = f"metrics_{scope}_" if dataset_variant else "metrics_"
            metrics_filename = f"{prefix}{split}.json"
        metrics_path = _save_metrics(cfg, metric_dict, filename=metrics_filename)
        log.info("Metrics saved to %s", metrics_path)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    _preflight_validate(cfg)
    extras(cfg)
    run_cfg = cfg.get("run") or {}
    if run_cfg.get("dataset_variants"):
        _run_eval_all_datasets(cfg)
        return
    if bool(run_cfg.get("run_all_splits", False)):
        _run_eval_all_splits(cfg)
        return
    evaluate(cfg)


if __name__ == "__main__":
    main()
