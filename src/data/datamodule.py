from __future__ import annotations

from typing import Any, Dict, Optional
from collections.abc import Mapping

from lightning import LightningDataModule
from torch.utils.data import Dataset

from .components import SharedDataResources
from .components.loader import UnifiedDataLoader
from .dataset import (
    RetrievalDataset,
    create_dataset,
    resolve_dataset_paths,
    DatasetPaths,
)


def _resolve_split(overrides: Dict[str, Any], name: str, default: str) -> str:
    raw = overrides.get(name, default)
    if raw in (None, ""):
        return default
    return str(raw)


try:  # Optional Hydra dependency
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover - keep import optional
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


def _infer_batch_size(total_batch: int, world_size: int) -> int:
    if world_size <= 1:
        return total_batch
    if total_batch % world_size != 0:
        raise RuntimeError(
            f"Global batch size {total_batch} must be divisible by world size {world_size}"
        )
    return total_batch // world_size


class RetrievalDataModule(LightningDataModule):
    """LightningDataModule wrapping RetrievalDataset + UnifiedDataLoader."""

    def __init__(
        self,
        *,
        dataset_cfg: Dict[str, Any],
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_cfg = self._to_plain_dict(dataset_cfg)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.persistent_workers = bool(persistent_workers)
        self.train_dataset: Optional[RetrievalDataset] = None
        self.val_dataset: Optional[RetrievalDataset] = None
        self.test_dataset: Optional[RetrievalDataset] = None
        self.batch_size_per_device = self.batch_size
        self._train_steps = 1
        self._dataset_paths: Optional[DatasetPaths] = None
        self._shared_resources: Optional[SharedDataResources] = None

    @property
    def train_steps_per_epoch(self) -> int:
        return self._train_steps

    def prepare_data(self) -> None:
        dataset_cfg = dict(self.dataset_cfg)
        split_overrides = dataset_cfg.get("splits") or {}
        if not isinstance(split_overrides, dict):
            split_overrides = {}
        train_split = _resolve_split(split_overrides, "train", "train")
        dataset_cfg["split"] = train_split
        self._dataset_paths = resolve_dataset_paths(dataset_cfg, train_split)

        missing = []
        if not self._dataset_paths.vocabulary_path.exists():
            missing.append(str(self._dataset_paths.vocabulary_path))
        if not self._dataset_paths.embeddings_dir.exists():
            missing.append(str(self._dataset_paths.embeddings_dir))
        if missing:
            raise FileNotFoundError(
                "Required retrieval assets not found: " + ", ".join(missing)
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            self.batch_size_per_device = _infer_batch_size(self.batch_size, self.trainer.world_size)

        dataset_cfg = dict(self.dataset_cfg)
        split_overrides = dataset_cfg.get("splits") or {}
        if not isinstance(split_overrides, dict):
            split_overrides = {}

        if self._dataset_paths is None:
            default_split = _resolve_split(split_overrides, "train", "train")
            dataset_cfg["split"] = default_split
            self._dataset_paths = resolve_dataset_paths(dataset_cfg, default_split)

        if self._shared_resources is None:
            self._shared_resources = SharedDataResources(
                vocabulary_path=self._dataset_paths.vocabulary_path,
                embeddings_dir=self._dataset_paths.embeddings_dir,
            )

        if stage in (None, "fit"):
            train_split = _resolve_split(split_overrides, "train", "train")
            val_split = _resolve_split(split_overrides, "validation", "validation")
            self.train_dataset = create_dataset(
                dataset_cfg,
                train_split,
                resources=self._shared_resources,
            )
            self.val_dataset = create_dataset(
                dataset_cfg,
                val_split,
                resources=self._shared_resources,
            )
            self._train_steps = self._compute_steps(self.train_dataset)
        if stage in (None, "test"):
            test_split = _resolve_split(split_overrides, "test", "test")
            self.test_dataset = create_dataset(
                dataset_cfg,
                test_split,
                resources=self._shared_resources,
            )

    def train_dataloader(self):
        assert self.train_dataset is not None
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        assert self.val_dataset is not None
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        assert self.test_dataset is not None
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def _build_loader(
        self,
        dataset: RetrievalDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> UnifiedDataLoader:
        sampler_cfg = self._get_sampler_cfg()
        samples_per_epoch = self._coerce_positive_int(sampler_cfg.get("train_samples_per_epoch"))
        batches_per_epoch = self._coerce_positive_int(sampler_cfg.get("train_batches_per_epoch"))
        hard_negative_k = self._coerce_non_negative_int(self.dataset_cfg.get("hard_negative_k")) or 0
        return UnifiedDataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            samples_per_epoch=samples_per_epoch if shuffle else None,
            batches_per_epoch=batches_per_epoch if shuffle else None,
            random_seed=self.dataset_cfg.get("random_seed"),
            hard_negative_k=hard_negative_k,
            hard_negative_similarity=self.dataset_cfg.get("hard_negative_similarity", "cosine"),
        )

    def _compute_steps(self, dataset: Dataset) -> int:
        sampler_cfg = self._get_sampler_cfg()
        batches_per_epoch = self._coerce_positive_int(sampler_cfg.get("train_batches_per_epoch"))
        if batches_per_epoch:
            return batches_per_epoch
        samples_per_epoch = self._coerce_positive_int(sampler_cfg.get("train_samples_per_epoch"))
        if samples_per_epoch:
            steps = samples_per_epoch // self.batch_size_per_device
            return max(1, steps)
        size = max(1, len(dataset))
        if self.drop_last:
            steps = size // self.batch_size_per_device
        else:
            steps = -(-size // self.batch_size_per_device)
        return max(1, steps)

    @staticmethod
    def _to_plain_dict(config: Any) -> Dict[str, Any]:
        if DictConfig is not None and isinstance(config, DictConfig):
            container = OmegaConf.to_container(config, resolve=True)  # type: ignore[arg-type]
            if isinstance(container, Mapping):
                return dict(container)
            raise TypeError("dataset_cfg DictConfig was not converted to a mapping.")
        if isinstance(config, Mapping):
            return dict(config)
        return dict(config)

    def _get_sampler_cfg(self) -> Dict[str, Any]:
        sampler = self.dataset_cfg.get("sampler")
        if isinstance(sampler, Mapping):
            return dict(sampler)
        return {}

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None
        if int_value <= 0:
            return None
        return int_value

    @staticmethod
    def _coerce_non_negative_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None
        if int_value < 0:
            return None
        return int_value
