from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate import RetrievalCollator
from .dataset import RetrievalDataset


_EMBEDDING_NORM_ATOL = 1.0e-3


@dataclass(frozen=True)
class ModelResources:
    """
    Static tensor resources required to initialize WeaverModule / Policy.

    entity_text_embeddings:
        [num_text_entities, hidden_dim] PLM embeddings for text entities.

    entity_embedding_map:
        [num_entities] long tensor mapping entity catalog id to text embedding id.
        Non-text entities should have value -1.

    relation_embeddings:
        [num_relations, hidden_dim] PLM embeddings for relations.
    """

    entity_text_embeddings: torch.Tensor
    entity_embedding_map: torch.Tensor
    relation_embeddings: torch.Tensor


class RetrievalDataModule(LightningDataModule):
    """
    LightningDataModule for materialized retrieval datasets.

    Responsibilities:
    - read materialized dataset paths from config;
    - load model initialization resources;
    - instantiate RetrievalDataset for train/validation/test;
    - build DataLoader objects.

    Non-responsibilities:
    - no model construction;
    - no runtime sample filtering;
    - no sample deserialization logic;
    - no graph/path recomputation;
    - no legacy schema handling.
    """

    def __init__(
        self,
        dataset_cfg: Mapping[str, Any],
        batch_size: int,
        num_workers: int,
        *,
        eval_batch_size: int | None = None,
        eval_num_workers: int | None = None,
        pin_memory: bool = True,
        train_shuffle: bool = True,
        drop_last: bool = False,
        eval_drop_last: bool = False,
        lmdb_readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        super().__init__()

        self.dataset_cfg = dataset_cfg

        self.batch_size = int(batch_size)
        self.eval_batch_size = int(
            eval_batch_size if eval_batch_size is not None else batch_size
        )

        self.num_workers = int(num_workers)
        self.eval_num_workers = int(
            eval_num_workers if eval_num_workers is not None else num_workers
        )

        self.pin_memory = bool(pin_memory)
        self.train_shuffle = bool(train_shuffle)
        self.drop_last = bool(drop_last)
        self.eval_drop_last = bool(eval_drop_last)

        self.lmdb_readahead = bool(lmdb_readahead)
        self.max_readers = int(max_readers)
        if self.max_readers <= 0:
            raise ValueError(f"max_readers must be positive, got {self.max_readers}.")

        self.lmdb_dir = _path_from_dataset_cfg(dataset_cfg, "lmdb_dir")
        self.metadata_dir = _path_from_dataset_cfg(dataset_cfg, "metadata_dir")

        self.entity_text_embeddings_path = _path_from_dataset_cfg(
            dataset_cfg,
            "entity_text_embeddings",
        )
        self.entity_metadata_path = _path_from_dataset_cfg(
            dataset_cfg,
            "entity_metadata_path",
        )
        self.relation_embeddings_path = _path_from_dataset_cfg(
            dataset_cfg,
            "relation_embeddings",
        )

        self.train_split = _split_name(dataset_cfg, "train", default="train")
        self.val_split = _split_name(
            dataset_cfg,
            "validation",
            default="validation",
        )
        self.test_split = _split_name(dataset_cfg, "test", default="test")

        self.collator = RetrievalCollator()

        self.train_dataset: RetrievalDataset | None = None
        self.val_dataset: RetrievalDataset | None = None
        self.test_dataset: RetrievalDataset | None = None

        self._model_resources: ModelResources | None = None

    @property
    def model_resources(self) -> ModelResources:
        """
        Resources required by WeaverModule initialization.

        Call prepare_data() and setup(...) before reading this property.
        """
        if self._model_resources is None:
            raise RuntimeError(
                "model_resources requested before setup(). "
                "Call datamodule.prepare_data(); datamodule.setup('fit') first."
            )
        return self._model_resources

    def prepare_data(self) -> None:
        """
        Lightning hook for single-process data checks.

        The dataset is assumed to be materialized already.
        """
        _require_dir(self.lmdb_dir, "LMDB directory")
        _require_dir(self.metadata_dir, "Metadata directory")

        _require_file(self.entity_text_embeddings_path, "entity_text_embeddings")
        _require_file(self.entity_metadata_path, "entity_metadata")
        _require_file(self.relation_embeddings_path, "relation_embeddings")

    def setup(self, stage: str | None = None) -> None:
        self._ensure_model_resources_loaded()

        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(self.train_split)
            self.val_dataset = self._build_dataset(self.val_split)

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = self._build_dataset(self.val_split)

        if stage in (None, "test"):
            self.test_dataset = self._build_dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self.train_dataset,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self.val_dataset,
            training=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self.test_dataset,
            training=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()

    def _ensure_model_resources_loaded(self) -> None:
        if self._model_resources is not None:
            return

        entity_text_embeddings = _load_tensor_artifact(
            path=self.entity_text_embeddings_path,
            name="entity_text_embeddings",
            keys=("entity_text_embeddings", "embeddings", "tensor"),
        ).to(dtype=torch.float32)

        entity_metadata = _load_artifact(
            path=self.entity_metadata_path,
            name="entity_metadata",
        )
        entity_embedding_map = _extract_entity_embedding_map(
            artifact=entity_metadata,
            name="entity_metadata",
        ).to(dtype=torch.long)

        relation_embeddings = _load_tensor_artifact(
            path=self.relation_embeddings_path,
            name="relation_embeddings",
            keys=("relation_embeddings", "embeddings", "tensor"),
        ).to(dtype=torch.float32)

        _validate_model_resources(
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            relation_embeddings=relation_embeddings,
        )

        self._model_resources = ModelResources(
            entity_text_embeddings=entity_text_embeddings.contiguous(),
            entity_embedding_map=entity_embedding_map.contiguous(),
            relation_embeddings=relation_embeddings.contiguous(),
        )

    def _build_dataset(self, split: str) -> RetrievalDataset:
        return RetrievalDataset(
            lmdb_dir=self.lmdb_dir,
            metadata_dir=self.metadata_dir,
            split=split,
            lmdb_readahead=self.lmdb_readahead,
            max_readers=self.max_readers,
        )

    def _build_loader(
        self,
        *,
        dataset: RetrievalDataset | None,
        training: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        batch_size = self.batch_size if training else self.eval_batch_size
        num_workers = self.num_workers if training else self.eval_num_workers
        drop_last = self.drop_last if training else self.eval_drop_last

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=training and self.train_shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
        )


def _load_artifact(
    *,
    path: Path,
    name: str,
) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load {name} artifact from {path}") from exc


def _load_tensor_artifact(
    *,
    path: Path,
    name: str,
    keys: tuple[str, ...],
) -> torch.Tensor:
    artifact = _load_artifact(path=path, name=name)

    if isinstance(artifact, torch.Tensor):
        return artifact

    if isinstance(artifact, Mapping):
        for key in keys:
            value = artifact.get(key)
            if isinstance(value, torch.Tensor):
                return value

        raise KeyError(
            f"{name} artifact at {path} is a mapping but contains none of "
            f"the tensor keys {list(keys)}. Available keys: {sorted(artifact.keys())}."
        )

    raise TypeError(
        f"{name} artifact at {path} must be a Tensor or mapping containing "
        f"a tensor, got {type(artifact)!r}."
    )


def _extract_tensor(
    *,
    artifact: Any,
    name: str,
    key: str,
) -> torch.Tensor:
    if isinstance(artifact, Mapping):
        value = artifact.get(key)
        if isinstance(value, torch.Tensor):
            return value
        raise KeyError(
            f"{name} mapping does not contain tensor key {key!r}. "
            f"Available keys: {sorted(artifact.keys())}."
        )

    value = getattr(artifact, key, None)
    if isinstance(value, torch.Tensor):
        return value

    raise TypeError(f"{name} must expose tensor field {key!r}; got {type(artifact)!r}.")


def _extract_entity_embedding_map(*, artifact: Any, name: str) -> torch.Tensor:
    if isinstance(artifact, Mapping):
        entity_embedding_map = artifact.get("entity_embedding_map")
        if isinstance(entity_embedding_map, torch.Tensor):
            return entity_embedding_map

        entity_text_embedding_ids = artifact.get("entity_text_embedding_ids")
        if isinstance(entity_text_embedding_ids, torch.Tensor):
            return _entity_text_embedding_ids_to_map(entity_text_embedding_ids)

        raise KeyError(
            f"{name} mapping does not contain tensor key 'entity_embedding_map' or "
            "'entity_text_embedding_ids'. Available keys: "
            f"{sorted(artifact.keys())}."
        )

    entity_embedding_map = getattr(artifact, "entity_embedding_map", None)
    if isinstance(entity_embedding_map, torch.Tensor):
        return entity_embedding_map

    entity_text_embedding_ids = getattr(artifact, "entity_text_embedding_ids", None)
    if isinstance(entity_text_embedding_ids, torch.Tensor):
        return _entity_text_embedding_ids_to_map(entity_text_embedding_ids)

    raise TypeError(
        f"{name} must expose tensor field 'entity_embedding_map' or "
        f"'entity_text_embedding_ids'; got {type(artifact)!r}."
    )


def _entity_text_embedding_ids_to_map(
    entity_text_embedding_ids: torch.Tensor,
) -> torch.Tensor:
    if entity_text_embedding_ids.ndim != 1:
        raise ValueError(
            "entity_text_embedding_ids must be 1D, "
            f"got shape={tuple(entity_text_embedding_ids.shape)}."
        )

    return entity_text_embedding_ids.to(dtype=torch.long) - 1


def _validate_model_resources(
    *,
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
) -> None:
    if entity_text_embeddings.ndim != 2:
        raise ValueError(
            "entity_text_embeddings must be 2D, "
            f"got shape={tuple(entity_text_embeddings.shape)}."
        )

    if entity_embedding_map.ndim != 1:
        raise ValueError(
            "entity_embedding_map must be 1D, "
            f"got shape={tuple(entity_embedding_map.shape)}."
        )

    if relation_embeddings.ndim != 2:
        raise ValueError(
            "relation_embeddings must be 2D, "
            f"got shape={tuple(relation_embeddings.shape)}."
        )

    entity_dim = int(entity_text_embeddings.size(1))
    relation_dim = int(relation_embeddings.size(1))
    if entity_dim != relation_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"entity_text_embeddings dim={entity_dim}, "
            f"relation_embeddings dim={relation_dim}."
        )

    _validate_l2_normalized_rows(
        entity_text_embeddings,
        name="entity_text_embeddings",
    )
    _validate_l2_normalized_rows(
        relation_embeddings,
        name="relation_embeddings",
    )

    if entity_embedding_map.numel() > 0:
        min_id = int(entity_embedding_map.min().item())
        max_id = int(entity_embedding_map.max().item())

        if min_id < -1:
            raise ValueError(
                "entity_embedding_map must contain -1 or nonnegative text ids, "
                f"got min={min_id}."
            )

        if max_id >= int(entity_text_embeddings.size(0)):
            raise ValueError(
                "entity_embedding_map contains text ids outside "
                "entity_text_embeddings: "
                f"max={max_id}, table_size={int(entity_text_embeddings.size(0))}."
            )


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
        "`python src/preprocess.py dataset=<name>`."
    )


def _path_from_dataset_cfg(cfg: Mapping[str, Any], key: str) -> Path:
    paths = cfg.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("dataset.paths must be a mapping")

    value = paths.get(key)
    if value in (None, ""):
        raise KeyError(f"dataset.paths.{key} must be provided")

    return Path(str(value))


def _split_name(
    cfg: Mapping[str, Any],
    key: str,
    *,
    default: str,
) -> str:
    splits = cfg.get("splits", {})
    if splits is None:
        splits = {}

    if not isinstance(splits, Mapping):
        raise TypeError("dataset.splits must be a mapping if provided")

    value = str(splits.get(key, default)).strip()
    if not value:
        raise ValueError(f"dataset.splits.{key} must be a non-empty split name")

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


__all__ = [
    "ModelResources",
    "RetrievalDataModule",
]
