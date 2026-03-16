from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, cast

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.runs.common import resolve_execution_mode
from src.utils.entrypoint_utils import (
    instantiate_lightning_task_objects,
    instantiate_task_runner,
    require_run_target_config,
)
from src.utils.entrypoint_contracts import validate_eval_entry_contract
from src.utils.hydra_utils import extras
from src.utils.logging_utils import RankedLogger
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


@task_wrapper
def evaluate_model(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(
            "Missing required config group: `run`. Example: "
            "`python src/eval.py experiment=eval_answer_reachability ckpt.answer_reachability=/path/to/model.ckpt`."
        )
    log.info("Run: %s", run_cfg.get("name"))

    _enforce_single_gpu_eval(cfg.trainer)

    objects = instantiate_lightning_task_objects(cfg, log=log)
    datamodule = objects.datamodule
    model = objects.model
    trainer = objects.trainer
    object_dict = objects.as_dict()

    execution_mode = resolve_execution_mode(run_cfg)
    ckpt_path = cfg.get("ckpt_path")
    if execution_mode == "test":
        log.info("Running trainer.test()...")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=False,
        )
    else:
        log.info("Running trainer.predict()...")
        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )

    return dict(trainer.callback_metrics), object_dict


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
            "Fix: use a concrete run config such as `run=eval_answer_reachability` or `run=eval_llm`."
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
