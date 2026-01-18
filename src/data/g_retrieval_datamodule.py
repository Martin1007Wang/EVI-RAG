from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from lightning import LightningDataModule
import torch

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
from torch.utils.data import DataLoader

from .components import SharedDataResources
from .components.loader import build_retrieval_dataloader
from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset
from .io.lmdb_utils import _resolve_core_lmdb_paths
_EMBEDDINGS_DEVICE_CPU = "cpu"
_EMBEDDINGS_DEVICE_CUDA = "cuda"


def _canonicalize_dataset_cfg(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset_cfg to the SSOT representation: `paths.entity_vocab` + `paths.embeddings`."""

    cfg = dict(dataset_cfg)
    paths = cfg.get("paths")
    if isinstance(paths, dict) and paths.get("entity_vocab") and paths.get("embeddings"):
        return cfg
    raise ValueError("dataset_cfg must define `paths.entity_vocab` and `paths.embeddings`.")


class GRetrievalDataModule(LightningDataModule):
    """
    Refactored GRetrievalDataModule following System Engineering principles.

    Principles:
    1. Dependency Injection: Receives a full `dataset_cfg` object.
    2. Zero Logic Config: Paths are resolved in YAML, not Python.
    3. Separation of Concerns: DataModule handles Logistics (batching), Dataset handles Logic.
    """

    def __init__(
        self,
        *,
        dataset_cfg: DictConfig | Dict[str, Any],
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
        train_shuffle: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        precompute_edge_batch: bool = False,
        validate_edge_batch: bool = False,
        embeddings_device: str | None = None,
        splits: Optional[Dict[str, str]] = None,
        expand_multi_start: bool = True,
        expand_multi_answer: bool = True,
    ) -> None:
        super().__init__()
        normalized_embeddings_device = None if embeddings_device is None else str(embeddings_device).lower()
        if normalized_embeddings_device not in (None, _EMBEDDINGS_DEVICE_CPU, _EMBEDDINGS_DEVICE_CUDA):
            raise ValueError(f"embeddings_device must be cpu or cuda, got {normalized_embeddings_device!r}.")
        if normalized_embeddings_device == _EMBEDDINGS_DEVICE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("embeddings_device=cuda requested but CUDA is not available.")
        embeddings_device = normalized_embeddings_device

        # dataset_cfg 可能包含 OmegaConf 对象；避免写入 checkpoint 元数据。
        self.save_hyperparameters(logger=False, ignore=["dataset_cfg"])

        # 1. Store Configuration (The Source of Truth)
        # Convert DictConfig to primitive dict if necessary, or keep it wrapper
        # Using OmegaConf.to_container allows downstream code to be simple dict-based
        if OmegaConf is not None and isinstance(dataset_cfg, DictConfig):
            cfg = OmegaConf.to_container(dataset_cfg, resolve=True)  # type: ignore[arg-type]
        else:
            cfg = dataset_cfg

        if not isinstance(cfg, dict):
            raise TypeError(f"dataset_cfg must be a mapping, got {type(cfg)!r}")
        self.dataset_cfg = _canonicalize_dataset_cfg(cfg)

        # 2. Store Logistics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_shuffle = bool(train_shuffle)
        self.persistent_workers = persistent_workers
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.precompute_edge_batch = bool(precompute_edge_batch)
        self.validate_edge_batch = bool(validate_edge_batch)
        self.embeddings_device = None if embeddings_device is None else str(embeddings_device)
        self.expand_multi_start = bool(expand_multi_start)
        self.expand_multi_answer = bool(expand_multi_answer)

        # Default splits mapping if not provided in `data/g_retrieval.yaml`
        self.splits = splits or {"train": "train", "validation": "validation", "test": "test"}
        # 3. Runtime State
        self.train_dataset: Optional[GRetrievalDataset] = None
        self.val_dataset: Optional[GRetrievalDataset] = None
        self.test_dataset: Optional[GRetrievalDataset] = None
        self.batch_size_per_device = self.batch_size
        self._shared_resources: Optional[SharedDataResources] = None

    @property
    def shared_resources(self) -> Optional[SharedDataResources]:
        return self._shared_resources

    def prepare_data(self) -> None:
        """
        Verify data existence.
        Since paths are injected via YAML, we just check them.
        """
        # Defensive check: ensure the injected config has what we need
        # This replaces the complex `resolve_dataset_paths` logic
        paths = self.dataset_cfg.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("Invalid dataset_cfg: expected mapping at `paths`.")
        if "entity_vocab" not in paths or "embeddings" not in paths:
            raise ValueError("Invalid dataset_cfg: expected `paths.entity_vocab` and `paths.embeddings`.")

        entity_vocab_path = Path(paths["entity_vocab"])
        emb_dir = Path(paths["embeddings"])

        missing = []
        if not entity_vocab_path.exists():
            missing.append(f"Entity vocab: {entity_vocab_path}")
        if not emb_dir.exists():
            missing.append(f"Embeddings Dir: {emb_dir}")

        if missing:
            raise FileNotFoundError(
                f"Critical Data Error: The following injected paths do not exist:\n"
                + "\n".join(missing)
                + "\nPlease check 'configs/dataset/YOUR_DATASET.yaml'."
            )
        for split_name in sorted(set(self.splits.values())):
            _resolve_core_lmdb_paths(emb_dir, split_name)

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Batch size is defined per device; keep as-is for DDP.
        self.batch_size_per_device = self.batch_size

        # 2. Initialize Shared Resources (One-time load)
        if self._shared_resources is None:
            paths = self.dataset_cfg["paths"]
            self._shared_resources = SharedDataResources(
                entity_vocab_path=Path(paths["entity_vocab"]),
                embeddings_dir=Path(paths["embeddings"]),
                embeddings_device=self.embeddings_device,
            )

        # 3. Instantiate Datasets
        # We pass the WHOLE config + the specific split name
        # The factory `create_g_retrieval_dataset` should handle the rest
        if stage in (None, "fit"):
            self.train_dataset = create_g_retrieval_dataset(
                cfg=self.dataset_cfg,
                split_name=self.splits["train"],
                resources=self._shared_resources,
            )
            self.val_dataset = create_g_retrieval_dataset(
                cfg=self.dataset_cfg,
                split_name=self.splits["validation"],
                resources=self._shared_resources,
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = create_g_retrieval_dataset(
                cfg=self.dataset_cfg,
                split_name=self.splits["test"],
                resources=self._shared_resources,
            )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=self.train_shuffle, drop_last=self.drop_last)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        # Predict reuses the test split.
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def train_eval_dataloader(self) -> DataLoader:
        """
        Deterministic loader for train split during evaluation/export stages.
        """
        return self._build_loader(self.train_dataset, shuffle=False, drop_last=False)

    def get_split_dataloader(self, split: str) -> DataLoader:
        if split == "train":
            return self.train_eval_dataloader()
        if split in ("val", "validation"):
            return self.val_dataloader()
        if split == "test":
            return self.test_dataloader()
        raise ValueError(f"Unsupported split: {split}")

    def teardown(self, stage: Optional[str] = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if self._shared_resources is not None:
            self._shared_resources.clear()
        self._shared_resources = None

    def _build_loader(
        self,
        dataset: GRetrievalDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """
        Constructs the retrieval DataLoader using params injected via dataset_cfg.
        """
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Did you run setup()?")

        return build_retrieval_dataloader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            precompute_edge_batch=self.precompute_edge_batch,
            validate_edge_batch=self.validate_edge_batch,
            random_seed=self.dataset_cfg.get("random_seed"),
            expand_multi_start=self.expand_multi_start,
            expand_multi_answer=self.expand_multi_answer,
        )
