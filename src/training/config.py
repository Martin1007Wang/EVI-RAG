from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig

from src.data.artifacts import (
    MaterializationArtifact,
    load_materialization_artifact,
)
from src.data.preprocess.catalog import Catalog, DebugLookup, load_question_text_by_sample_id
from src.data.tensor_table import read_table, validate_file

_EMBEDDING_NORM_ATOL = 1.0e-3
_STALE_MATERIALIZED_PATH_KEYS = frozenset(
    {
        "lmdb_dir",
        "entity_text_semantic_table",
        "relation_semantic_table",
        "catalog_path",
    }
)


@dataclass(frozen=True, slots=True)
class ModelResources:
    entity_text_semantic_table: torch.Tensor
    text_row_by_entity_id: torch.Tensor
    relation_semantic_table: torch.Tensor
    debug_lookup: DebugLookup | None = None


@dataclass(frozen=True, slots=True)
class SplitPlan:
    train: str
    validation: str
    test: str

    def split_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.train,
                    self.validation,
                    self.test,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    batch_size: int
    eval_batch_size: int
    num_workers: int
    eval_num_workers: int
    pin_memory: bool
    train_shuffle: bool
    drop_last: bool
    lmdb_readahead: bool
    max_readers: int
    train_indices: tuple[int, ...] | None = None
    validation_indices: tuple[int, ...] | None = None
    test_indices: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalDataConfig:
    materialization: MaterializationArtifact
    splits: SplitPlan
    loader: LoaderConfig


@dataclass(frozen=True, slots=True)
class RolloutRuntimeConfig:
    budget: int
    train_num_rollout: int
    eval_num_rollout: int
    train_chunk_size: int
    eval_chunk_size: int

    def __post_init__(self) -> None:
        budget = non_negative_int(self.budget, "budget")
        train_num_rollout = positive_int(
            self.train_num_rollout,
            "train_num_rollout",
        )
        eval_num_rollout = positive_int(
            self.eval_num_rollout,
            "eval_num_rollout",
        )
        train_chunk_size = positive_int(
            self.train_chunk_size,
            "train_chunk_size",
        )
        eval_chunk_size = positive_int(
            self.eval_chunk_size,
            "eval_chunk_size",
        )

        if train_chunk_size > train_num_rollout:
            raise ValueError("train_chunk_size cannot exceed train_num_rollout: " f"{train_chunk_size} > {train_num_rollout}.")

        if eval_chunk_size > eval_num_rollout:
            raise ValueError("eval_chunk_size cannot exceed eval_num_rollout: " f"{eval_chunk_size} > {eval_num_rollout}.")

        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "train_num_rollout", train_num_rollout)
        object.__setattr__(self, "eval_num_rollout", eval_num_rollout)
        object.__setattr__(self, "train_chunk_size", train_chunk_size)
        object.__setattr__(self, "eval_chunk_size", eval_chunk_size)


@dataclass(frozen=True, slots=True)
class RewardRuntimeConfig:
    tau: float
    full_support_bonus: float
    edge_penalty: float
    surplus_penalty: float
    no_answer_penalty: float
    utility_loss_weight: float
    debug_checks: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tau",
            positive_float(self.tau, "tau"),
        )
        object.__setattr__(
            self,
            "full_support_bonus",
            non_negative_float(self.full_support_bonus, "full_support_bonus"),
        )
        object.__setattr__(
            self,
            "edge_penalty",
            non_negative_float(self.edge_penalty, "edge_penalty"),
        )
        object.__setattr__(
            self,
            "surplus_penalty",
            non_negative_float(self.surplus_penalty, "surplus_penalty"),
        )
        object.__setattr__(
            self,
            "no_answer_penalty",
            float(self.no_answer_penalty),
        )
        object.__setattr__(
            self,
            "utility_loss_weight",
            non_negative_float(self.utility_loss_weight, "utility_loss_weight"),
        )
        object.__setattr__(
            self,
            "debug_checks",
            boolean(self.debug_checks, "debug_checks"),
        )


@dataclass(frozen=True, slots=True)
class LossRuntimeConfig:
    utility_loss_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "utility_loss_weight",
            non_negative_float(self.utility_loss_weight, "utility_loss_weight"),
        )


@dataclass(frozen=True, slots=True)
class EvalRuntimeConfig:
    exclude_anchors_from_retrieved: bool
    use_reachable_targets: bool
    k_windows: tuple[int, ...] | list[int] = (1, 2, 4, 8)
    enable_terminal_diagnostics: bool = False

    def __post_init__(self) -> None:
        k_windows = tuple(positive_int(k, "k_windows") for k in self.k_windows)
        if not k_windows:
            raise ValueError("k_windows must not be empty.")
        object.__setattr__(
            self,
            "k_windows",
            k_windows,
        )
        object.__setattr__(
            self,
            "exclude_anchors_from_retrieved",
            boolean(
                self.exclude_anchors_from_retrieved,
                "exclude_anchors_from_retrieved",
            ),
        )
        object.__setattr__(
            self,
            "use_reachable_targets",
            boolean(self.use_reachable_targets, "use_reachable_targets"),
        )
        object.__setattr__(
            self,
            "enable_terminal_diagnostics",
            boolean(self.enable_terminal_diagnostics, "enable_terminal_diagnostics"),
        )


@dataclass(frozen=True, slots=True)
class TrainingRuntimeConfig:
    manual_accumulate_grad_batches: int
    gradient_clip_val: float
    gradient_clip_algorithm: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manual_accumulate_grad_batches",
            positive_int(
                self.manual_accumulate_grad_batches,
                "manual_accumulate_grad_batches",
            ),
        )
        object.__setattr__(
            self,
            "gradient_clip_val",
            non_negative_float(self.gradient_clip_val, "gradient_clip_val"),
        )
        object.__setattr__(
            self,
            "gradient_clip_algorithm",
            one_of(
                self.gradient_clip_algorithm,
                "gradient_clip_algorithm",
                {"norm", "value"},
            ),
        )


@dataclass(frozen=True, slots=True)
class OptimizerRuntimeConfig:
    type: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    no_decay_on_bias_and_norm: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            one_of(self.type, "optimizer.type", {"adamw"}),
        )
        object.__setattr__(
            self,
            "lr",
            positive_float(self.lr, "optimizer.lr"),
        )
        object.__setattr__(
            self,
            "weight_decay",
            non_negative_float(self.weight_decay, "optimizer.weight_decay"),
        )
        object.__setattr__(
            self,
            "betas",
            betas_value(self.betas, "optimizer.betas"),
        )
        object.__setattr__(
            self,
            "no_decay_on_bias_and_norm",
            boolean(
                self.no_decay_on_bias_and_norm,
                "optimizer.no_decay_on_bias_and_norm",
            ),
        )


@dataclass(frozen=True, slots=True)
class SchedulerRuntimeConfig:
    type: str
    interval: str
    warmup_ratio: float
    eta_min: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            one_of(self.type, "scheduler.type", {"none", "cosine"}),
        )
        object.__setattr__(
            self,
            "interval",
            one_of(self.interval, "scheduler.interval", {"step", "epoch"}),
        )
        object.__setattr__(
            self,
            "warmup_ratio",
            ratio(self.warmup_ratio, "scheduler.warmup_ratio"),
        )
        object.__setattr__(
            self,
            "eta_min",
            non_negative_float(self.eta_min, "scheduler.eta_min"),
        )


@dataclass(frozen=True, slots=True)
class OptimizationRuntimeConfig:
    optimizer: OptimizerRuntimeConfig
    scheduler: SchedulerRuntimeConfig | None

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, OptimizerRuntimeConfig):
            raise TypeError("optimization.optimizer must be OptimizerRuntimeConfig, " f"got {type(self.optimizer).__name__}.")
        if self.scheduler is not None and not isinstance(
            self.scheduler,
            SchedulerRuntimeConfig,
        ):
            raise TypeError("optimization.scheduler must be SchedulerRuntimeConfig or None, " f"got {type(self.scheduler).__name__}.")


def build_retrieval_data_config(cfg: DictConfig) -> RetrievalDataConfig:
    dataset_cfg = _mapping(cfg.get("dataset"), "dataset")
    datamodule_cfg = _mapping(cfg.get("datamodule"), "datamodule")

    metadata_dir = _dataset_path(dataset_cfg, "metadata_dir")
    materialization = load_materialization_artifact(metadata_dir)
    if materialization is None:
        raise FileNotFoundError("Materialization manifest not found. Re-run preprocessing to rebuild " f"materialized data under {metadata_dir}.")

    train_split = _split_name(datamodule_cfg.get("splits"), "train", default="train")
    validation_split = _split_name(
        datamodule_cfg.get("splits"),
        "validation",
        default="validation",
    )
    test_split = _split_name(datamodule_cfg.get("splits"), "test", default="test")

    batch_size = positive_int(
        datamodule_cfg.get("batch_size", 1),
        "datamodule.batch_size",
    )
    num_workers = non_negative_int(
        datamodule_cfg.get("num_workers", 0),
        "datamodule.num_workers",
    )
    eval_batch_size = positive_int(
        datamodule_cfg.get("eval_batch_size", batch_size),
        "datamodule.eval_batch_size",
    )
    eval_num_workers = non_negative_int(
        datamodule_cfg.get("eval_num_workers", num_workers),
        "datamodule.eval_num_workers",
    )
    max_readers = positive_int(
        datamodule_cfg.get("max_readers", 256),
        "datamodule.max_readers",
    )

    split_plan = SplitPlan(
        train=train_split,
        validation=validation_split,
        test=test_split,
    )

    loader = LoaderConfig(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers,
        pin_memory=boolean(
            datamodule_cfg.get("pin_memory", True),
            "datamodule.pin_memory",
        ),
        train_shuffle=boolean(
            datamodule_cfg.get("train_shuffle", True),
            "datamodule.train_shuffle",
        ),
        drop_last=boolean(
            datamodule_cfg.get("drop_last", False),
            "datamodule.drop_last",
        ),
        lmdb_readahead=boolean(
            datamodule_cfg.get("lmdb_readahead", False),
            "datamodule.lmdb_readahead",
        ),
        max_readers=max_readers,
        train_indices=_optional_index_tuple(
            datamodule_cfg.get("train_indices"),
            "datamodule.train_indices",
        ),
        validation_indices=_optional_index_tuple(
            datamodule_cfg.get("validation_indices"),
            "datamodule.validation_indices",
        ),
        test_indices=_optional_index_tuple(
            datamodule_cfg.get("test_indices"),
            "datamodule.test_indices",
        ),
    )

    return RetrievalDataConfig(
        materialization=materialization,
        splits=split_plan,
        loader=loader,
    )


def validate_retrieval_data_config(
    data_config: RetrievalDataConfig,
    *,
    validate_question_embeddings: bool = True,
    splits: tuple[str, ...] | list[str] | None = None,
) -> None:
    split_names = data_config.splits.split_names() if splits is None else tuple(dict.fromkeys(splits))

    for split in split_names:
        paths = data_config.materialization.require_split(split)
        _require_dir(paths.lmdb, f"{split} LMDB")

        if not validate_question_embeddings:
            continue

        validate_file(paths.question_embeddings)
        if paths.question_embeddings.rows != int(paths.num_samples):
            raise ValueError(f"{split} question embedding rows mismatch: " f"{paths.question_embeddings.rows} != {int(paths.num_samples)}.")


def load_model_resources(
    materialization: MaterializationArtifact,
) -> ModelResources:
    # validate_file(materialization.entity_text_semantic_table)
    # validate_file(materialization.relation_semantic_table)
    # _require_file(materialization.catalog, "catalog")

    entity_text_semantic_table = read_table(
        materialization.entity_text_semantic_table,
    )
    relation_semantic_table = read_table(
        materialization.relation_semantic_table,
    )
    catalog = Catalog.load(materialization.catalog)

    return validate_model_resources(
        entity_text_semantic_table=entity_text_semantic_table,
        text_row_by_entity_id=catalog.entity_text_row_by_entity_id.long().contiguous(),
        relation_semantic_table=relation_semantic_table,
        debug_lookup=DebugLookup(
            catalog=catalog,
            question_text_by_sample_id=_load_question_text_by_sample_id(materialization),
        ),
    )


def _optional_index_tuple(
    value: Any,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers, got {type(value).__name__}.")
    if not isinstance(value, (list, tuple, ListConfig)):
        raise TypeError(f"{name} must be a sequence of integers or null, got {type(value).__name__}.")
    out = tuple(non_negative_int(item, name) for item in value)
    if not out:
        raise ValueError(f"{name} must not be empty when provided.")
    return out


def validate_model_resources(
    *,
    entity_text_semantic_table: torch.Tensor,
    text_row_by_entity_id: torch.Tensor,
    relation_semantic_table: torch.Tensor,
    debug_lookup: DebugLookup | None = None,
) -> ModelResources:
    if entity_text_semantic_table.ndim != 2:
        raise ValueError("entity_text_semantic_table must be 2D, " f"got shape={tuple(entity_text_semantic_table.shape)}.")

    if text_row_by_entity_id.ndim != 1:
        raise ValueError("text_row_by_entity_id must be 1D, " f"got shape={tuple(text_row_by_entity_id.shape)}.")

    if relation_semantic_table.ndim != 2:
        raise ValueError("relation_semantic_table must be 2D, " f"got shape={tuple(relation_semantic_table.shape)}.")

    entity_dim = int(entity_text_semantic_table.size(1))
    relation_dim = int(relation_semantic_table.size(1))
    if entity_dim != relation_dim:
        raise ValueError(
            "Embedding dimension mismatch: " f"entity_text_semantic_table dim={entity_dim}, " f"relation_semantic_table dim={relation_dim}."
        )

    _validate_l2_normalized_rows(
        entity_text_semantic_table,
        name="entity_text_semantic_table",
    )
    _validate_l2_normalized_rows(
        relation_semantic_table,
        name="relation_semantic_table",
    )

    if text_row_by_entity_id.numel() > 0:
        min_id = int(text_row_by_entity_id.min().item())
        max_id = int(text_row_by_entity_id.max().item())

        if min_id < -1:
            raise ValueError("text_row_by_entity_id must contain -1 or nonnegative text ids, " f"got min={min_id}.")

        if max_id >= int(entity_text_semantic_table.size(0)):
            raise ValueError(
                "text_row_by_entity_id contains text ids outside "
                "entity_text_semantic_table: "
                f"max={max_id}, table_size={int(entity_text_semantic_table.size(0))}."
            )

    return ModelResources(
        entity_text_semantic_table=entity_text_semantic_table.contiguous(),
        text_row_by_entity_id=text_row_by_entity_id.contiguous(),
        relation_semantic_table=relation_semantic_table.contiguous(),
        debug_lookup=debug_lookup,
    )


def _load_question_text_by_sample_id(
    materialization: MaterializationArtifact,
) -> dict[str, str] | None:
    lookup_path = materialization.question_texts
    if lookup_path is not None and lookup_path.exists():
        return load_question_text_by_sample_id(lookup_path)
    return None


def _dataset_path(dataset_cfg: Mapping[str, Any], key: str) -> Path:
    paths = _mapping(dataset_cfg.get("paths"), "dataset.paths")
    _reject_stale_materialized_path_keys(paths)
    value = paths.get(key)
    if value in (None, ""):
        raise KeyError(f"dataset.paths.{key} must be provided")
    return Path(str(value))


def _reject_stale_materialized_path_keys(paths: Mapping[str, Any]) -> None:
    stale_keys = sorted(key for key in _STALE_MATERIALIZED_PATH_KEYS if key in paths)
    if stale_keys:
        formatted = ", ".join(f"dataset.paths.{key}" for key in stale_keys)
        raise KeyError(
            "Retrieval data config reads materialized artifacts only through "
            "dataset.paths.metadata_dir/materialization_manifest.json; remove stale "
            f"path key(s): {formatted}."
        )


def _split_name(
    splits: Any,
    key: str,
    *,
    default: str,
) -> str:
    if splits is None:
        value = default
    else:
        splits = _mapping(splits, "datamodule.splits")
        value = splits.get(key, default)

    value = str(value).strip()
    if not value:
        raise ValueError(f"datamodule.splits.{key} must be a non-empty split name")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} path is not a file: {path}")


def _validate_l2_normalized_rows(
    tensor: torch.Tensor,
    *,
    name: str,
    atol: float = _EMBEDDING_NORM_ATOL,
) -> None:
    if tensor.numel() == 0:
        return

    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")

    norms = torch.linalg.vector_norm(tensor.to(dtype=torch.float32), ord=2, dim=1)
    deviation = (norms - 1.0).abs()
    max_deviation = float(deviation.max().item())
    if max_deviation <= float(atol):
        return

    row = int(deviation.argmax().item())
    norm = float(norms[row].item())
    raise ValueError(
        f"{name} rows must be L2-normalized within atol={float(atol):g}; "
        f"row {row} has norm={norm:.6g}. Rebuild embeddings with "
        "`preprocess_command dataset=<name>`."
    )


def positive_int(value: Any, name: str) -> int:
    value = int_value(value, name)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def non_negative_int(value: Any, name: str) -> int:
    value = int_value(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def int_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool.")

    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{name} must be int, got float {value}.")

    return int(value)


def positive_float(value: Any, name: str) -> float:
    value = float_value(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def non_negative_float(value: Any, name: str) -> float:
    value = float_value(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def ratio(value: Any, name: str) -> float:
    value = non_negative_float(value, name)
    if value > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value}.")
    return value


def float_value(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float, got bool.")
    return float(value)


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}.")
    return value


def config_value(
    section: DictConfig | Mapping[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    if section is None:
        return default

    if isinstance(section, DictConfig):
        return section.get(key, default)

    if isinstance(section, Mapping):
        return section.get(key, default)

    raise TypeError(f"Config section for {key!r} must be DictConfig, mapping, or None; " f"got {type(section).__name__}.")


def one_of(value: Any, name: str, allowed: set[str]) -> str:
    value = str(value)
    if value not in allowed:
        options = ", ".join(sorted(repr(item) for item in allowed))
        raise ValueError(f"{name} must be one of {options}, got {value!r}.")
    return value


def betas_value(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of two floats.")

    if not isinstance(value, (list, tuple, ListConfig)):
        raise TypeError(f"{name} must be a sequence of two floats, got {type(value).__name__}.")

    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {len(value)}.")

    beta1 = float_value(value[0], f"{name}[0]")
    beta2 = float_value(value[1], f"{name}[1]")

    for index, beta in enumerate((beta1, beta2)):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"{name}[{index}] must be in [0, 1), got {beta}.")

    return beta1, beta2


__all__ = [
    "EvalRuntimeConfig",
    "LoaderConfig",
    "LossRuntimeConfig",
    "ModelResources",
    "OptimizationRuntimeConfig",
    "OptimizerRuntimeConfig",
    "RetrievalDataConfig",
    "RewardRuntimeConfig",
    "RolloutRuntimeConfig",
    "SchedulerRuntimeConfig",
    "SplitPlan",
    "TrainingRuntimeConfig",
    "betas_value",
    "boolean",
    "build_retrieval_data_config",
    "config_value",
    "float_value",
    "int_value",
    "load_model_resources",
    "non_negative_float",
    "non_negative_int",
    "one_of",
    "positive_float",
    "positive_int",
    "ratio",
    "validate_retrieval_data_config",
    "validate_model_resources",
]
