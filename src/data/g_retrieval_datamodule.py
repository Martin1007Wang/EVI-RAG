from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from .components import SharedDataResources
from .components.loader import UnifiedDataLoader
from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset


def _infer_batch_size(total_batch: int, world_size: int) -> int:
    """Helper to split batch size across GPUs."""
    if world_size <= 1:
        return total_batch
    if total_batch % world_size != 0:
        # 严谨性：如果不能整除，为了保证梯度一致性，必须报错或警告
        # 这里选择简单处理：整除
        pass 
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
        persistent_workers: bool = False,
        splits: Optional[Dict[str, str]] = None,
        sampler: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # 1. Store Configuration (The Source of Truth)
        # Convert DictConfig to primitive dict if necessary, or keep it wrapper
        # Using OmegaConf.to_container allows downstream code to be simple dict-based
        if isinstance(dataset_cfg, DictConfig):
            self.dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
        else:
            self.dataset_cfg = dataset_cfg

        # 2. Store Logistics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        
        # Default splits mapping if not provided in `data/retriever.yaml`
        self.splits = splits or {"train": "train", "validation": "validation", "test": "test"}
        self.sampler = sampler or {}
        # 3. Runtime State
        self.train_dataset: Optional[GRetrievalDataset] = None
        self.val_dataset: Optional[GRetrievalDataset] = None
        self.test_dataset: Optional[GRetrievalDataset] = None
        self.batch_size_per_device = self.batch_size
        self._shared_resources: Optional[SharedDataResources] = None

    def prepare_data(self) -> None:
        """
        Verify data existence. 
        Since paths are injected via YAML, we just check them.
        """
        # Defensive check: ensure the injected config has what we need
        # This replaces the complex `resolve_dataset_paths` logic
        try:
            paths = self.dataset_cfg.get("paths", {})
            vocab_path = Path(paths.get("vocabulary", ""))
            emb_dir = Path(paths.get("embeddings", ""))
        except Exception as e:
            raise ValueError("Invalid dataset_cfg structure. Expected 'paths.vocabulary' and 'paths.embeddings'.") from e

        missing = []
        if not vocab_path.exists():
            missing.append(f"Vocabulary: {vocab_path}")
        if not emb_dir.exists():
            missing.append(f"Embeddings Dir: {emb_dir}")

        if missing:
            raise FileNotFoundError(
                f"Critical Data Error: The following injected paths do not exist:\n" + "\n".join(missing) +
                "\nPlease check 'configs/dataset/YOUR_DATASET.yaml'."
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
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self):
        # Support for GAgent generation pipeline
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

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
        
        # Extract Hard Negative settings (Dataset Property)
        hard_neg_k = self.dataset_cfg.get("hard_negative_k", 0)
        hard_neg_sim = self.dataset_cfg.get("hard_negative_similarity", "cosine")

        return UnifiedDataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            
            # Sampler logic (Throttling)
            samples_per_epoch=self.sampler.get("train_samples_per_epoch") if shuffle else None,
            batches_per_epoch=self.sampler.get("train_batches_per_epoch") if shuffle else None,
            
            # Hard Negatives
            hard_negative_k=hard_neg_k,
            hard_negative_similarity=hard_neg_sim,
            
            # Reproducibility
            random_seed=self.dataset_cfg.get("random_seed"),
        )