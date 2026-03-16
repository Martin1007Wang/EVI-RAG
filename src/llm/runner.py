from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from src.runs.eval_runner_base import BaseEvalRunner, EvaluateModelFn
from src.utils.logging_utils import RankedLogger

from .eval_llm import run_llm_eval

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class LlmEvalRunner(BaseEvalRunner):
    name: str = "eval_llm"
    task_name: str = "eval/llm"
    tags: tuple[str, ...] = ()
    split: str = "test"
    run_all_splits: bool = False
    splits: tuple[str, ...] = ("train", "validation", "test")
    ckpt_path: str | None = None
    dataset_variants: Any = None
    dataset_variant: str | None = None
    execution_mode: str = "predict"
    eval_mode: str | None = None

    def validate(self, cfg: DictConfig) -> None:
        if cfg.get("llm") is None:
            raise ValueError(
                "Missing required config group: `llm`. "
                "Fix: use `run=eval_llm` together with an experiment or config that sets `/llm`."
            )
        if self.dataset_variants:
            raise ValueError("eval_llm does not support run.dataset_variants.")

    def _run_once(self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn) -> None:
        del evaluate_model
        run_llm_eval(cfg)

    def _split_log_label(self) -> str:
        return "llm_eval"

    def _logger(self) -> Any:
        return log


__all__ = ["LlmEvalRunner"]
