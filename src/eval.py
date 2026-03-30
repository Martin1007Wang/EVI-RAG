from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, cast

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.runs.common import resolve_execution_mode
from src.utils.entrypoint_utils import (
    instantiate_lightning_task_objects,
    instantiate_task_runner,
    require_run_target_config,
)
from src.utils.entrypoint_contracts import validate_eval_entry_contract
from src.utils.hydra_utils import extras
from src.utils.logging_utils import RankedLogger
from src.utils.precision_utils import normalize_precision
from src.utils.task_utils import task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

_ALLOW_CPU_EVAL_ENV = "DUAL_FLOW_ALLOW_CPU_EVAL"
_ALLOW_CPU_EVAL_ON = "1"
_ALLOW_CPU_EVAL_OFF = "0"


EvaluateModelFn = Callable[[DictConfig], Tuple[Dict[str, Any], Dict[str, Any]]]


class EvalRunnerProtocol(Protocol):
    def validate(self, cfg: DictConfig) -> None: ...

    def run(self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn) -> None: ...


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


def _configure_eval_split(datamodule: Any, run_cfg: DictConfig) -> str:
    split = str(run_cfg.get("split") or "test").strip() or "test"
    setter = getattr(datamodule, "set_eval_split", None)
    if callable(setter):
        setter(split)
    return split


def _coerce_eval_cfg(eval_cfg: Any) -> dict[str, Any]:
    return normalize_search_eval_cfg(eval_cfg)


def _load_checkpoint_into_model_if_needed(model: Any, *, ckpt_path: str | None) -> None:
    if ckpt_path in (None, ""):
        return
    resolved_ckpt = str(ckpt_path)
    if getattr(model, "_rankflow_loaded_eval_ckpt_path", None) == resolved_ckpt:
        return
    checkpoint = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "ckpt_path must point to a checkpoint containing a `state_dict`."
        )
    loader = getattr(model, "load_state_dict", None)
    if not callable(loader):
        raise TypeError("Existing evaluation model does not support `load_state_dict`.")
    incompatible = loader(state_dict, strict=False)
    missing = sorted(getattr(incompatible, "missing_keys", []))
    unexpected = sorted(getattr(incompatible, "unexpected_keys", []))
    log.info(
        "Loaded evaluation checkpoint into existing model: %s (missing=%d, unexpected=%d)",
        resolved_ckpt,
        len(missing),
        len(unexpected),
    )
    if missing:
        log.warning("Missing keys when loading ckpt_path for eval: %s", missing)
    if unexpected:
        log.warning("Unexpected keys when loading ckpt_path for eval: %s", unexpected)
    setattr(model, "_rankflow_loaded_eval_ckpt_path", resolved_ckpt)


def _trainer_supports_inprocess_eval(trainer: Any) -> bool:
    num_devices = getattr(trainer, "num_devices", None)
    if num_devices is not None and int(num_devices) != 1:
        return False
    strategy = getattr(trainer, "strategy", None)
    strategy_name = str(getattr(strategy, "strategy_name", "") or "").lower()
    if any(tag in strategy_name for tag in ("ddp", "fsdp", "deepspeed")):
        return False
    root_device = getattr(strategy, "root_device", None)
    if isinstance(root_device, torch.device):
        device = cast(torch.device, root_device)
        return device.type == "cuda"
    return True


def _enforce_inprocess_eval_precision(cfg: DictConfig, *, trainer: Any) -> None:
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        return
    requested_precision = normalize_precision(trainer_cfg.get("precision"))
    active_precision = normalize_precision(getattr(trainer, "precision", None))
    if (
        requested_precision is None
        or active_precision is None
        or requested_precision == active_precision
    ):
        return
    raise ValueError(
        "In-process final eval precision mismatch: "
        f"active trainer.precision={active_precision!r} requested eval precision={requested_precision!r}. "
        "Use a fresh eval stack so Lightning can instantiate the requested precision plugin."
    )


def _select_evaluation_metrics(trainer: Any, *, execution_mode: str) -> dict[str, Any]:
    callback_metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
    if execution_mode == "predict":
        return {}
    return {
        key: value
        for key, value in callback_metrics.items()
        if str(key).startswith("test/")
    }


def evaluate_model_inprocess(
    cfg: DictConfig,
    *,
    trainer: Any,
    datamodule: Any,
    model: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not _trainer_supports_inprocess_eval(trainer):
        raise ValueError(
            "In-process final eval requires a single-device non-distributed trainer."
        )
    _enforce_inprocess_eval_precision(cfg, trainer=trainer)
    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError("Missing required config group: `run`.")
    replace_dataset_cfg = getattr(datamodule, "replace_dataset_cfg", None)
    if not callable(replace_dataset_cfg):
        raise TypeError(
            "Existing datamodule does not support `replace_dataset_cfg()` for in-process eval."
        )
    split = str(run_cfg.get("split") or "test").strip() or "test"
    replace_dataset_cfg(cfg.dataset, eval_split=split)
    _load_checkpoint_into_model_if_needed(model, ckpt_path=cfg.get("ckpt_path"))
    reconfigure_evaluation = getattr(model, "reconfigure_evaluation", None)
    if not callable(reconfigure_evaluation):
        raise TypeError(
            "Existing model does not support `reconfigure_evaluation()` for in-process eval."
        )
    reconfigure_evaluation(eval_cfg=_coerce_eval_cfg(cfg.model.eval_cfg))
    execution_mode = resolve_execution_mode(run_cfg)
    if execution_mode == "test":
        log.info("Running in-process trainer.test() on split=%s...", split)
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=None,
            verbose=False,
        )
    else:
        log.info("Running in-process trainer.predict() on split=%s...", split)
        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=None,
            return_predictions=False,
        )
    return _select_evaluation_metrics(trainer, execution_mode=execution_mode), {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }


@task_wrapper
def evaluate_model(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(
            "Missing required config group: `run`. Example: "
            "`python src/eval.py experiment=rankflow ckpt.gflownet=/path/to/model.ckpt`."
        )
    log.info("Run: %s", run_cfg.get("name"))

    _enforce_single_gpu_eval(cfg.trainer)

    objects = instantiate_lightning_task_objects(cfg, log=log)
    datamodule = objects.datamodule
    model = objects.model
    trainer = objects.trainer
    object_dict = objects.as_dict()

    execution_mode = resolve_execution_mode(run_cfg)
    split = _configure_eval_split(datamodule, run_cfg)
    ckpt_path = cfg.get("ckpt_path")
    if execution_mode == "test":
        log.info("Running trainer.test() on split=%s...", split)
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=False,
        )
    else:
        log.info("Running trainer.predict() on split=%s...", split)
        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )

    return _select_evaluation_metrics(
        trainer, execution_mode=execution_mode
    ), object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    require_run_target_config(
        cfg,
        missing_run_message=(
            "Missing required config group: `run`. "
            "Fix: pass `run=<group>` or use an eval experiment that overrides `/run`."
        ),
        missing_target_message=(
            "Missing required run target: `run._target_`. "
            "Fix: use a concrete run config such as `run=rankflow` or `run=eval_llm`."
        ),
    )
    validate_eval_entry_contract(cfg)
    extras(cfg)
    runner = cast(
        EvalRunnerProtocol,
        instantiate_task_runner(
            cfg.run, run_signature="run(cfg=..., evaluate_model=...)"
        ),
    )
    runner.validate(cfg)
    runner.run(cfg=cfg, evaluate_model=evaluate_model)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
