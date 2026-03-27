from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig

from src.runs.eval_runner_base import BaseEvalRunner, EvaluateModelFn
from src.runs.common import (
    collect_model_metrics as _collect_model_metrics,
    normalize_dataset_scope,
    resolve_dataset_variants,
    resolve_execution_mode,
    save_metrics_snapshot,
)
from src.runs.output_orchestrator import RunOutputOrchestrator
from src.utils.logging_utils import RankedLogger
from src.utils.output_sinks import (
    PredictionArtifactSettings,
)


RANKFLOW_MODEL_TARGET = "src.models.gflownet_module.GFlowNetModule"
ANSWER_REACHABILITY_MODEL_TARGET = RANKFLOW_MODEL_TARGET
RANKFLOW_EVAL_RUN = "rankflow"
RANKFLOW_TRAIN_RUN = "train_rankflow"
SUPPORTED_TRAINING_MODEL_TARGETS = {RANKFLOW_MODEL_TARGET}
RUN_REQUIRES_CKPT_KIND = {RANKFLOW_EVAL_RUN: "gflownet"}

log = RankedLogger(__name__, rank_zero_only=True)

TrainModelFn = Callable[[DictConfig], Tuple[Dict[str, Any], Dict[str, Any]]]


@dataclass(frozen=True)
class AnswerReachabilityEvalReporter:
    @staticmethod
    def collect_model_metrics(
        *, callback_metrics: dict[str, Any], model: Any
    ) -> dict[str, Any]:
        return _collect_model_metrics(callback_metrics=callback_metrics, model=model)

    @staticmethod
    def resolve_metrics_filename(
        *,
        run_cfg: DictConfig | dict[str, Any],
        dataset_cfg: DictConfig | dict[str, Any],
    ) -> str:
        metrics_filename = "metrics.json"
        split = run_cfg.get("split") if hasattr(run_cfg, "get") else None
        dataset_variant = (
            run_cfg.get("dataset_variant") if hasattr(run_cfg, "get") else None
        )
        scope = None
        if dataset_variant:
            scope = normalize_dataset_scope(dataset_cfg)
            metrics_filename = f"metrics_{scope}.json"
        if bool(run_cfg.get("run_all_splits", False)) and split not in (None, ""):
            prefix = f"metrics_{scope}_" if scope else "metrics_"
            metrics_filename = f"{prefix}{split}.json"
        return metrics_filename

    @staticmethod
    def save_metrics(
        cfg: DictConfig,
        metrics: dict[str, Any],
        *,
        filename: str = "metrics.json",
    ) -> Path:
        return save_metrics_snapshot(
            output_dir=cfg.paths.output_dir,
            metrics=metrics,
            filename=filename,
        )

    @staticmethod
    def build_artifact_settings(cfg: DictConfig) -> PredictionArtifactSettings:
        run_cfg = cfg.get("run") or {}
        dataset_cfg = cfg.get("dataset") or {}
        dataset_paths = dataset_cfg.get("paths") or {}
        return PredictionArtifactSettings(
            enabled=bool(run_cfg.get("write_artifacts", False)),
            execution_mode=resolve_execution_mode(run_cfg),
            output_root=dataset_cfg.get("artifact_dir"),
            artifact_subdir=str(run_cfg.get("artifact_subdir") or RANKFLOW_EVAL_RUN),
            artifact_name=str(run_cfg.get("artifact_name") or RANKFLOW_EVAL_RUN),
            schema_version=int(run_cfg.get("artifact_schema_version", 1) or 1),
            split=str(run_cfg.get("split") or "test"),
            dataset_scope=normalize_dataset_scope(dataset_cfg),
            dataset_variant=run_cfg.get("dataset_variant"),
            entity_vocab_path=dataset_paths.get("entity_vocab"),
            relation_vocab_path=dataset_paths.get("relation_vocab"),
            questions_path=run_cfg.get("questions_path"),
            dataset_out_dir=dataset_cfg.get("out_dir"),
            overwrite=bool(run_cfg.get("artifact_overwrite", True)),
        )

    def build_orchestrator(self) -> RunOutputOrchestrator:
        return RunOutputOrchestrator(
            collect_metrics=lambda callback_metrics, model: self.collect_model_metrics(
                callback_metrics=callback_metrics,
                model=model,
            ),
            resolve_metrics_filename=lambda cfg: self.resolve_metrics_filename(
                run_cfg=cfg.get("run") or {},
                dataset_cfg=cfg.dataset,
            ),
            save_metrics=lambda cfg, metrics, filename: self.save_metrics(
                cfg,
                metrics,
                filename=filename,
            ),
            build_artifact_settings=self.build_artifact_settings,
        )

    def persist_outputs(
        self,
        *,
        cfg: DictConfig,
        callback_metrics: dict[str, Any],
        model: Any,
        log: Any,
    ) -> dict[str, Any]:
        return (
            self.build_orchestrator()
            .persist(
                cfg=cfg,
                callback_metrics=callback_metrics,
                model=model,
                log=log,
            )
            .metrics
        )


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
            "  python src/eval.py experiment=rankflow ckpt.gflownet=/path/to/model.ckpt\n"
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
            "  python src/eval.py experiment=rankflow ckpt.gflownet=/path/to/model.ckpt"
        )
    required_kind = RUN_REQUIRES_CKPT_KIND.get(run_name)
    if required_kind and cfg.get("ckpt_path") in (None, ""):
        raise ValueError(
            f"Run `{run_name}` requires `{required_kind}` checkpoint, but `ckpt_path` is empty.\n"
            f"Fix: pass `ckpt.{required_kind}=/path/to/{required_kind}.ckpt`."
        )
    if run_name != RANKFLOW_EVAL_RUN:
        return

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


@dataclass
class AnswerReachabilityEvalRunner(BaseEvalRunner):
    name: str = RANKFLOW_EVAL_RUN
    task_name: str = "eval/rankflow"
    tags: tuple[str, ...] = ()
    split: str = "test"
    run_all_splits: bool = False
    splits: tuple[str, ...] = ("train", "validation", "test")
    ckpt_path: str | None = None
    dataset_variants: Any = None
    dataset_variant: str | None = None
    execution_mode: str = "predict"
    write_artifacts: bool = True
    artifact_subdir: str = RANKFLOW_EVAL_RUN
    artifact_name: str = RANKFLOW_EVAL_RUN
    artifact_schema_version: int = 1
    artifact_overwrite: bool = True
    questions_path: str | None = None
    reporter: AnswerReachabilityEvalReporter = field(
        default_factory=AnswerReachabilityEvalReporter
    )

    def validate(self, cfg: DictConfig) -> None:
        validate_eval_config(cfg)

    def _run_once(self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn) -> None:
        metric_dict, object_dict = evaluate_model(cfg)
        self.reporter.persist_outputs(
            cfg=cfg,
            callback_metrics=metric_dict,
            model=object_dict["model"],
            log=log,
        )

    def _build_split_run_overrides(
        self, *, cfg: DictConfig, split: str
    ) -> dict[str, Any]:
        explicit_allow_empty = (cfg.get("run") or {}).get("allow_empty_answer")
        return {
            "split": split,
            "allow_empty_answer": (
                split != "train"
                if explicit_allow_empty is None
                else bool(explicit_allow_empty)
            ),
        }

    def _supports_dataset_variants(self) -> bool:
        return True

    def _logger(self) -> Any:
        return log


@dataclass
class AnswerReachabilityTrainRunner:
    name: str = RANKFLOW_TRAIN_RUN
    task_name: str = "train/rankflow"
    tags: tuple[str, ...] = ()
    contract: dict[str, Any] | None = None
    train: bool = True
    test: bool = False
    test_ckpt_path: str | None = None
    allow_test_without_checkpoint: bool = False
    final_eval_experiment: str | None = None
    final_eval_split: str = "test"
    final_eval_output_subdir: str = "final_eval"
    ckpt_path: str | None = None
    init_ckpt_path: str | None = None

    def validate(self, cfg: DictConfig) -> None:
        validate_train_config(cfg)

    def run(
        self,
        *,
        cfg: DictConfig,
        train_model: TrainModelFn,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return train_model(cfg)


_DEFAULT_REPORTER = AnswerReachabilityEvalReporter()


def collect_model_metrics(
    *, callback_metrics: dict[str, Any], model: Any
) -> dict[str, Any]:
    return _DEFAULT_REPORTER.collect_model_metrics(
        callback_metrics=callback_metrics,
        model=model,
    )


def resolve_metrics_filename(
    *,
    run_cfg: DictConfig | dict[str, Any],
    dataset_cfg: DictConfig | dict[str, Any],
) -> str:
    return _DEFAULT_REPORTER.resolve_metrics_filename(
        run_cfg=run_cfg,
        dataset_cfg=dataset_cfg,
    )


def save_metrics(
    cfg: DictConfig,
    metrics: dict[str, Any],
    *,
    filename: str = "metrics.json",
) -> Path:
    return _DEFAULT_REPORTER.save_metrics(cfg, metrics, filename=filename)


def persist_eval_outputs(
    *,
    cfg: DictConfig,
    callback_metrics: dict[str, Any],
    model: Any,
    log: Any,
) -> dict[str, Any]:
    return _DEFAULT_REPORTER.persist_outputs(
        cfg=cfg,
        callback_metrics=callback_metrics,
        model=model,
        log=log,
    )


__all__ = [
    "ANSWER_REACHABILITY_MODEL_TARGET",
    "RANKFLOW_EVAL_RUN",
    "RANKFLOW_MODEL_TARGET",
    "RANKFLOW_TRAIN_RUN",
    "RUN_REQUIRES_CKPT_KIND",
    "SUPPORTED_TRAINING_MODEL_TARGETS",
    "AnswerReachabilityEvalReporter",
    "AnswerReachabilityEvalRunner",
    "AnswerReachabilityTrainRunner",
    "collect_model_metrics",
    "persist_eval_outputs",
    "resolve_metrics_filename",
    "save_metrics",
    "validate_eval_config",
    "validate_train_config",
]
