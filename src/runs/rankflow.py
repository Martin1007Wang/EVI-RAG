from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig

from src.runs.common import (
    normalize_dataset_scope,
    resolve_execution_mode,
    resolve_splits,
    temporary_cfg_overrides,
)
from src.runs.dataset_variants import DatasetVariantSpec, resolve_dataset_variants
from src.runs.rankflow_outputs import persist_rankflow_outputs
from src.utils.logging_utils import RankedLogger


RANKFLOW_MODEL_TARGET = "src.subgraph_gflownet.adapters.lightning.module.GFlowNetModule"
RANKFLOW_EVAL_RUN = "rankflow"
RANKFLOW_TRAIN_RUN_PREFIX = "train_rankflow"
SUPPORTED_TRAINING_MODEL_TARGETS = {RANKFLOW_MODEL_TARGET}
RUN_REQUIRES_CKPT_KIND = {RANKFLOW_EVAL_RUN: "gflownet"}

log = RankedLogger(__name__, rank_zero_only=True)

EvaluateModelFn = Callable[[DictConfig], Tuple[Dict[str, Any], Dict[str, Any]]]


def is_rankflow_train_run(run_name: str) -> bool:
    return str(run_name).strip().startswith(RANKFLOW_TRAIN_RUN_PREFIX)


def validate_train_config(cfg: DictConfig) -> None:
    model_cfg = cfg.get("model") or {}
    model_target = str(model_cfg.get("_target_", "") or "")
    if model_target not in SUPPORTED_TRAINING_MODEL_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TRAINING_MODEL_TARGETS))
        raise ValueError(
            "Unsupported model target for training. "
            f"Got model._target_={model_target!r}. "
            f"Supported targets: {supported}."
        )

    if cfg.get("dataset") is None:
        raise ValueError(
            "Missing required training inputs: dataset. Please specify `dataset=<name>` for training. "
            "Example: python src/train.py experiment=train_rankflow dataset=webqsp-sub"
        )

    dataset_cfg = cfg.get("dataset") or {}
    scope = normalize_dataset_scope(dataset_cfg)
    if scope != "sub":
        dataset_name = str(dataset_cfg.get("name", "") or "")
        raise ValueError(
            "Training scope violation: supported training targets must use sub datasets only. "
            f"Got dataset={dataset_name!r} (dataset_scope={scope})."
        )

    run_cfg = cfg.get("run") or {}
    if bool(run_cfg.get("train", True)) and cfg.get("fit_schedule") is None:
        raise ValueError(
            "Training requires `fit_schedule` so progress is defined in train-set passes. "
            "Fix: use the default train config or pass `fit_schedule=pass`."
        )


def validate_eval_config(cfg: DictConfig) -> None:
    if cfg.get("dataset") is None:
        raise ValueError(
            "Missing required config group: `dataset`.\n"
            "Fix:\n"
            "  python src/eval.py experiment=eval_rankflow ckpt.gflownet=/path/to/model.ckpt\n"
            "Optional (recommended): set a default dataset in `configs/local/default.yaml` (gitignored), e.g.\n"
            "  defaults:\n"
            "    - override /dataset: webqsp"
        )

    run_cfg = cfg.get("run") or {}
    run_name = str(run_cfg.get("name", "") or "").strip()
    if run_name in {"", "null", "None"}:
        raise ValueError(
            "Missing required config group: `run`.\n"
            "Fix:\n"
            "  python src/eval.py experiment=eval_rankflow ckpt.gflownet=/path/to/model.ckpt"
        )
    required_kind = RUN_REQUIRES_CKPT_KIND.get(run_name)
    if required_kind and cfg.get("ckpt_path") in (None, ""):
        raise ValueError(
            f"Run `{run_name}` requires `{required_kind}` checkpoint, but `ckpt_path` is empty.\n"
            f"Fix: pass `ckpt.{required_kind}=/path/to/{required_kind}.ckpt`."
        )

    variants = resolve_dataset_variants(cfg)
    if not variants:
        raise ValueError(
            "Evaluation requires run.dataset_variants with both full and sub datasets."
        )
    scopes = {normalize_dataset_scope(variant.dataset_cfg) for variant in variants}
    if scopes != {"full", "sub"}:
        names = [variant.label for variant in variants]
        raise ValueError(
            "Evaluation requires both full and sub scopes. "
            f"Got scopes={sorted(scopes)} for variants={names}."
        )


def _default_dataset_variant(cfg: DictConfig) -> DatasetVariantSpec:
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is None:
        raise ValueError(
            "RankFlow evaluation requires `dataset` to resolve a default variant."
        )
    label = str(dataset_cfg.get("name") or normalize_dataset_scope(dataset_cfg))
    return DatasetVariantSpec(
        label=label,
        dataset_cfg=dataset_cfg,
        run_overrides={},
        set_dataset_variant=False,
    )


def persist_outputs(
    *,
    cfg: DictConfig,
    callback_metrics: dict[str, Any],
    model: Any,
    log: Any,
) -> dict[str, Any]:
    return persist_rankflow_outputs(
        cfg=cfg,
        callback_metrics=callback_metrics,
        model=model,
        log=log,
        default_name=RANKFLOW_EVAL_RUN,
    )


def _resolve_eval_targets(
    cfg: DictConfig,
    *,
    allow_default_dataset_variant: bool,
) -> list[DatasetVariantSpec]:
    variants = resolve_dataset_variants(cfg)
    if variants:
        return variants
    if allow_default_dataset_variant:
        return [_default_dataset_variant(cfg)]
    raise ValueError(
        "run.dataset_variants must be a non-empty list when evaluating multiple datasets."
    )


def _resolve_requested_splits(cfg: DictConfig) -> list[str]:
    run_cfg = cfg.get("run") or {}
    if not bool(run_cfg.get("run_all_splits", False)):
        split = str(run_cfg.get("split") or "test").strip() or "test"
        return [split]
    return resolve_splits(run_cfg.get("splits") or ())


def _build_split_run_overrides(cfg: DictConfig, *, split: str) -> dict[str, Any]:
    explicit_allow_empty = (cfg.get("run") or {}).get("allow_empty_answer")
    return {
        "split": split,
        "allow_empty_answer": (
            split != "train"
            if explicit_allow_empty is None
            else bool(explicit_allow_empty)
        ),
    }


def _namespace_metrics(
    *,
    metrics: dict[str, Any],
    prefix: str,
    dataset_variant: str,
    split: str,
) -> dict[str, Any]:
    root = f"{prefix}/{dataset_variant}/{split}"
    return {f"{root}/{name}": value for name, value in metrics.items()}


def run_eval(
    cfg: DictConfig,
    *,
    evaluate_model: EvaluateModelFn,
    allow_default_dataset_variant: bool = False,
    metric_namespace_prefix: str | None = None,
) -> dict[str, Any]:
    variants = _resolve_eval_targets(
        cfg,
        allow_default_dataset_variant=allow_default_dataset_variant,
    )
    aggregated_metrics: dict[str, Any] = {}

    for variant in variants:
        log.info("rankflow_eval: dataset_variant=%s", variant.label)
        with temporary_cfg_overrides(
            cfg,
            dataset_cfg=variant.dataset_cfg,
            run_overrides=(
                {
                    **variant.run_overrides,
                    "dataset_variant": variant.label,
                }
                if variant.set_dataset_variant
                else dict(variant.run_overrides)
            ),
        ):
            for split in _resolve_requested_splits(cfg):
                log.info("rankflow_eval: split=%s", split)
                with temporary_cfg_overrides(
                    cfg,
                    run_overrides=_build_split_run_overrides(cfg, split=split),
                ):
                    metric_dict, object_dict = evaluate_model(cfg)
                    metrics = persist_outputs(
                        cfg=cfg,
                        callback_metrics=metric_dict,
                        model=object_dict["model"],
                        log=log,
                    )
                    if metric_namespace_prefix:
                        aggregated_metrics.update(
                            _namespace_metrics(
                                metrics=metrics,
                                prefix=metric_namespace_prefix,
                                dataset_variant=variant.label,
                                split=split,
                            )
                        )
                    else:
                        aggregated_metrics.update(metrics)

    return aggregated_metrics


__all__ = [
    "RUN_REQUIRES_CKPT_KIND",
    "RANKFLOW_EVAL_RUN",
    "RANKFLOW_MODEL_TARGET",
    "RANKFLOW_TRAIN_RUN_PREFIX",
    "SUPPORTED_TRAINING_MODEL_TARGETS",
    "is_rankflow_train_run",
    "persist_outputs",
    "run_eval",
    "validate_eval_config",
    "validate_train_config",
]
