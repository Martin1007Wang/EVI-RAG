from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from lightning import LightningDataModule

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
from torch.utils.data import Dataset

from .components import SharedDataResources
from .components.loader import UnifiedDataLoader
from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)
_EMBEDDINGS_DEVICE_CPU = "cpu"
_EMBEDDINGS_DEVICE_CUDA = "cuda"


def _canonicalize_dataset_cfg(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset_cfg to the SSOT representation: `paths.vocabulary` + `paths.embeddings`."""

    cfg = dict(dataset_cfg)
    paths = cfg.get("paths")
    if isinstance(paths, dict) and paths.get("vocabulary") and paths.get("embeddings"):
        return cfg
    raise ValueError("dataset_cfg must define `paths.vocabulary` and `paths.embeddings`.")


def _infer_batch_size(total_batch: int, world_size: int) -> int:
    """Helper to split batch size across GPUs."""
    if world_size <= 1:
        return total_batch
    if total_batch % world_size != 0:
        raise ValueError(
            f"batch_size={total_batch} must be divisible by world_size={world_size}. "
            "Please adjust `data.batch_size` or trainer devices to keep per-rank batch sizes equal."
        )
    return total_batch // world_size


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
        worker_embed_lookup: bool = False,
        precompute_edge_batch: bool = False,
        precompute_node_in_degree: bool = False,
        validate_edge_batch: bool = False,
        embeddings_device: str | None = None,
        splits: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        normalized_embeddings_device = None if embeddings_device is None else str(embeddings_device).lower()
        if num_workers > 0 and normalized_embeddings_device == _EMBEDDINGS_DEVICE_CUDA:
            log_event(
                LOGGER,
                "datamodule_force_cpu_embeddings",
                reason="avoid_cuda_init_in_workers",
                num_workers=int(num_workers),
            )
            normalized_embeddings_device = _EMBEDDINGS_DEVICE_CPU
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
        self.worker_embed_lookup = bool(worker_embed_lookup)
        self.precompute_edge_batch = bool(precompute_edge_batch)
        self.precompute_node_in_degree = bool(precompute_node_in_degree)
        self.validate_edge_batch = bool(validate_edge_batch)
        self.embeddings_device = None if embeddings_device is None else str(embeddings_device)

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
        if "vocabulary" not in paths or "embeddings" not in paths:
            raise ValueError("Invalid dataset_cfg: expected `paths.vocabulary` and `paths.embeddings`.")

        vocab_path = Path(paths["vocabulary"])
        emb_dir = Path(paths["embeddings"])

        missing = []
        if not vocab_path.exists():
            missing.append(f"Vocabulary: {vocab_path}")
        if not emb_dir.exists():
            missing.append(f"Embeddings Dir: {emb_dir}")

        if missing:
            raise FileNotFoundError(
                f"Critical Data Error: The following injected paths do not exist:\n"
                + "\n".join(missing)
                + "\nPlease check 'configs/dataset/YOUR_DATASET.yaml'."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Adjust Batch Size for DDP
        if self.trainer is not None:
            self.batch_size_per_device = _infer_batch_size(self.batch_size, self.trainer.world_size)

        # 2. Initialize Shared Resources (One-time load)
        if self._shared_resources is None:
            paths = self.dataset_cfg["paths"]
            self._shared_resources = SharedDataResources(
                vocabulary_path=Path(paths["vocabulary"]),
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

    def predict_dataloader(self) -> UnifiedDataLoader:
        # Predict reuses the test split.
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def train_eval_dataloader(self) -> UnifiedDataLoader:
        """
        Deterministic loader for train split during evaluation/export stages.
        """
        return self._build_loader(self.train_dataset, shuffle=False, drop_last=False)

    def get_split_dataloader(self, split: str) -> UnifiedDataLoader:
        if split == "train":
            return self.train_eval_dataloader()
        if split in ("val", "validation"):
            return self.val_dataloader()
        if split == "test":
            return self.test_dataloader()
        raise ValueError(f"Unsupported split: {split}")

    def _build_loader(
        self,
        dataset: GRetrievalDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> UnifiedDataLoader:
        """
        Constructs the UnifiedDataLoader using params injected via dataset_cfg.
        """
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Did you run setup()?")

        return UnifiedDataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            worker_embed_lookup=self.worker_embed_lookup,
            precompute_edge_batch=self.precompute_edge_batch,
            precompute_node_in_degree=self.precompute_node_in_degree,
            validate_edge_batch=self.validate_edge_batch,
            embeddings_device=self.embeddings_device,
            random_seed=self.dataset_cfg.get("random_seed"),
        )
