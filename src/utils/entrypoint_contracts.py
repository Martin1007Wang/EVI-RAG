from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _normalize_choice(value: Any) -> str:
    return str(value or "").strip()


def _normalize_contract_value(value: Any) -> str:
    return _normalize_choice(value).lower()


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


def _resolve_run_contract(run_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    contract = run_cfg.get("contract") if hasattr(run_cfg, "get") else None
    if isinstance(contract, DictConfig):
        resolved = OmegaConf.to_container(contract, resolve=True)
        return resolved if isinstance(resolved, dict) else {}
    if isinstance(contract, dict):
        return dict(contract)

    run_name = _normalize_choice(
        run_cfg.get("name") if hasattr(run_cfg, "get") else None
    )
    if run_name == "train_rankflow":
        return {
            "entrypoint": "train",
            "required_groups": ["dataset"],
            "recommended_experiment": "train_rankflow",
        }
    if run_name == "rankflow":
        return {
            "entrypoint": "eval",
            "required_groups": ["dataset"],
            "recommended_experiment": "rankflow",
        }
    if run_name == "eval_llm":
        return {
            "entrypoint": "eval",
            "required_groups": ["llm"],
            "optional_groups": ["dataset"],
            "recommended_experiment": "eval_llm",
        }
    return {}


def _format_required_group_fix(group_name: str, contract: dict[str, Any]) -> str:
    recommended_experiment = _normalize_choice(contract.get("recommended_experiment"))
    if recommended_experiment:
        return f"use `experiment={recommended_experiment}`"
    return f"pass `{group_name}=<group>`"


def _validate_required_groups(
    cfg: DictConfig, contract: dict[str, Any], *, run_name: str, entry_label: str
) -> None:
    raw_groups = contract.get("required_groups") or []
    groups = [str(group).strip() for group in list(raw_groups) if str(group).strip()]
    for group_name in groups:
        if cfg.get(group_name) is not None:
            continue
        raise ValueError(
            f"{entry_label} requires `/{group_name}` for the configured run. "
            f"Got run.name={run_name!r} with {group_name}=None. "
            f"Fix: pass `{group_name}=<name>` or {_format_required_group_fix(group_name, contract)}."
        )


def validate_train_entry_contract(
    cfg: DictConfig,
    *,
    experiment_choice: str | None = None,
) -> None:
    run_cfg = cfg.get("run") or {}
    run_name = _normalize_choice(run_cfg.get("name"))
    contract = _resolve_run_contract(run_cfg)
    experiment = _normalize_choice(
        experiment_choice or _resolve_hydra_choice("experiment")
    )

    if experiment.startswith("eval_"):
        raise ValueError(
            "Train entrypoint received an eval experiment. "
            f"Got experiment={experiment!r}. Use `python src/eval.py ...` instead."
        )
    expected_entrypoint = _normalize_contract_value(contract.get("entrypoint"))
    if expected_entrypoint == "eval" or run_name.startswith("eval"):
        raise ValueError(
            "Train entrypoint requires a train run config. "
            f"Got run.name={run_name!r}. Use `python src/eval.py ...` or switch to `run=train_*`."
        )
    _validate_required_groups(
        cfg, contract, run_name=run_name, entry_label="Train entrypoint"
    )


def validate_eval_entry_contract(
    cfg: DictConfig,
    *,
    experiment_choice: str | None = None,
) -> None:
    run_cfg = cfg.get("run") or {}
    run_name = _normalize_choice(run_cfg.get("name"))
    contract = _resolve_run_contract(run_cfg)
    experiment = _normalize_choice(
        experiment_choice or _resolve_hydra_choice("experiment")
    )

    if experiment.startswith("train_"):
        raise ValueError(
            "Eval entrypoint received a train experiment. "
            f"Got experiment={experiment!r}. Use `python src/train.py ...` instead."
        )
    expected_entrypoint = _normalize_contract_value(contract.get("entrypoint"))
    if expected_entrypoint == "train" or run_name.startswith("train"):
        raise ValueError(
            "Eval entrypoint requires an eval run config. "
            f"Got run.name={run_name!r}. Use `python src/train.py ...` or switch to `run=eval_*`."
        )
    _validate_required_groups(
        cfg, contract, run_name=run_name, entry_label="Eval entrypoint"
    )


__all__ = [
    "validate_eval_entry_contract",
    "validate_train_entry_contract",
]
