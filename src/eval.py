from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.search_eval_utils import (
    format_search_eval_answer_posterior,
    normalize_search_eval_cfg,
)
from src.runs.common import resolve_execution_mode
from src.runs.entrypoints import validate_eval_entrypoint
from src.runs.hydra import extras
from src.runs.llm import (
    LLM_EVAL_RUN,
    run_eval as run_llm_eval,
    validate_eval_config as validate_llm_eval_config,
)
from src.runs.lightning import (
    finalize_task,
    instantiate_lightning_task_objects,
    seed_everything_if_needed,
    select_logged_metrics,
)
from src.runs.rankflow import (
    RANKFLOW_EVAL_RUN,
    run_eval as run_rankflow_eval,
    validate_eval_config as validate_rankflow_eval_config,
)
from src.utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_ALLOW_CPU_EVAL_ENV = "DUAL_FLOW_ALLOW_CPU_EVAL"
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


def _coerce_eval_cfg(eval_cfg: Any) -> dict[str, Any]:
    return normalize_search_eval_cfg(eval_cfg)


def evaluate_model(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        seed_everything_if_needed(cfg)

        run_cfg = cfg.get("run")
        if run_cfg is None:
            raise ValueError(
                "Missing required config group: `run`. Example: "
                "`python src/eval.py experiment=eval_rankflow ckpt.gflownet=/path/to/model.ckpt`."
            )
        log.info("Run: %s", run_cfg.get("name"))

        _enforce_single_gpu_eval(cfg.trainer)

        objects = instantiate_lightning_task_objects(cfg, log=log)
        datamodule = objects.datamodule
        model = objects.model
        trainer = objects.trainer
        object_dict = objects.as_dict()

        execution_mode = resolve_execution_mode(run_cfg)
        split = str(run_cfg.get("split") or "test").strip() or "test"
        ckpt_path = cfg.get("ckpt_path")
        eval_cfg = _coerce_eval_cfg(cfg.model.eval_cfg)
        log.info(
            "Eval config: report_profile=%s answer_posterior_surrogate=%s",
            eval_cfg.get("report_profile"),
            format_search_eval_answer_posterior(eval_cfg),
        )
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

        metrics = (
            {}
            if execution_mode == "predict"
            else select_logged_metrics(trainer, prefix="test/")
        )
        return metrics, object_dict
    except Exception:
        log.exception("")
        raise
    finally:
        finalize_task(cfg=cfg, log=log)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(
            "Missing required config group: `run`. "
            "Fix: pass `run=<group>` or use an eval experiment that overrides `/run`."
        )
    validate_eval_entrypoint(cfg)
    extras(cfg)
    run_name = str(run_cfg.get("name") or "").strip()
    if run_name == RANKFLOW_EVAL_RUN:
        validate_rankflow_eval_config(cfg)
        run_rankflow_eval(cfg, evaluate_model=evaluate_model)
        return
    if run_name == LLM_EVAL_RUN:
        validate_llm_eval_config(cfg)
        run_llm_eval(cfg)
        return
    raise ValueError(
        f"Unsupported eval run.name={run_name!r}. Supported runs: {RANKFLOW_EVAL_RUN}, {LLM_EVAL_RUN}."
    )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
