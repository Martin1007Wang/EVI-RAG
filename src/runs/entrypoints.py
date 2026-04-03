from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from src.runs.llm import LLM_EVAL_RUN
from src.runs.rankflow import RANKFLOW_EVAL_RUN, is_rankflow_train_run


def _normalize_choice(value: Any) -> str:
    return str(value or "").strip()


def _resolve_hydra_choice(choice_name: str) -> str | None:
    try:
        from hydra.core.hydra_config import HydraConfig
    except ModuleNotFoundError:
        return None
    try:
        hydra_cfg = HydraConfig.get()
        runtime = getattr(hydra_cfg, "runtime", None)
        if runtime is None:
            return None
        choice = runtime.choices.get(choice_name)  # type: ignore[attr-defined]
    except Exception:
        return None
    normalized = _normalize_choice(choice)
    return normalized or None


def _require_group(
    cfg: DictConfig,
    *,
    group_name: str,
    run_name: str,
    entry_label: str,
    recommended_experiment: str | None,
) -> None:
    if cfg.get(group_name) is not None:
        return
    if recommended_experiment:
        fix = f"use `experiment={recommended_experiment}`"
    else:
        fix = f"pass `{group_name}=<group>`"
    raise ValueError(
        f"{entry_label} requires `/{group_name}` for the configured run. "
        f"Got run.name={run_name!r} with {group_name}=None. "
        f"Fix: pass `{group_name}=<name>` or {fix}."
    )


def validate_train_entrypoint(
    cfg: DictConfig,
    *,
    experiment_choice: str | None = None,
) -> None:
    run_cfg = cfg.get("run") or {}
    run_name = _normalize_choice(run_cfg.get("name"))
    experiment = _normalize_choice(
        experiment_choice or _resolve_hydra_choice("experiment")
    )

    if experiment.startswith("eval_"):
        raise ValueError(
            "Train entrypoint received an eval experiment. "
            f"Got experiment={experiment!r}. Use `python src/eval.py ...` instead."
        )
    if not run_name or not run_name.startswith("train"):
        raise ValueError(
            "Train entrypoint requires a train run config. "
            f"Got run.name={run_name!r}. Use `python src/eval.py ...` or switch to `run=train_*`."
        )
    _require_group(
        cfg,
        group_name="dataset",
        run_name=run_name,
        entry_label="Train entrypoint",
        recommended_experiment="train_rankflow"
        if is_rankflow_train_run(run_name)
        else None,
    )


def validate_eval_entrypoint(
    cfg: DictConfig,
    *,
    experiment_choice: str | None = None,
) -> None:
    run_cfg = cfg.get("run") or {}
    run_name = _normalize_choice(run_cfg.get("name"))
    experiment = _normalize_choice(
        experiment_choice or _resolve_hydra_choice("experiment")
    )

    if experiment.startswith("train_"):
        raise ValueError(
            "Eval entrypoint received a train experiment. "
            f"Got experiment={experiment!r}. Use `python src/train.py ...` instead."
        )
    if not run_name or run_name.startswith("train"):
        raise ValueError(
            "Eval entrypoint requires an eval run config. "
            f"Got run.name={run_name!r}. Use `python src/train.py ...` or switch to `run=eval_*`."
        )
    if run_name == RANKFLOW_EVAL_RUN:
        _require_group(
            cfg,
            group_name="dataset",
            run_name=run_name,
            entry_label="Eval entrypoint",
            recommended_experiment="eval_rankflow",
        )
    elif run_name == LLM_EVAL_RUN:
        _require_group(
            cfg,
            group_name="llm",
            run_name=run_name,
            entry_label="Eval entrypoint",
            recommended_experiment="eval_llm",
        )


__all__ = [
    "validate_eval_entrypoint",
    "validate_train_entrypoint",
]
