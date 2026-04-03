from __future__ import annotations

from omegaconf import DictConfig

from src.llm.eval_llm import run_llm_eval


LLM_EVAL_RUN = "eval_llm"


def validate_eval_config(cfg: DictConfig) -> None:
    if cfg.get("llm") is None:
        raise ValueError(
            "Missing required config group: `llm`. "
            "Fix: use `run=eval_llm` together with an experiment or config that sets `/llm`."
        )
    if (cfg.get("run") or {}).get("dataset_variants"):
        raise ValueError("eval_llm does not support run.dataset_variants.")


def run_eval(cfg: DictConfig) -> None:
    run_llm_eval(cfg)


__all__ = ["LLM_EVAL_RUN", "run_eval", "validate_eval_config"]
