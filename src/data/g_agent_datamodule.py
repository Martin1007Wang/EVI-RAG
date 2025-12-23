from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from .g_agent_dataset import (
    GAgentPyGDataset,
)
from .components import SharedDataResources


class GAgentDataModule(LightningDataModule):
    """LightningDataModule wrapping g_agent caches for GFlowNet (no split-specific filtering)."""

    def __init__(
        self,
        *,
        cache_paths: Dict[str, str],
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        shuffle_train: bool = True,
        drop_unreachable: bool = False,
        resources: Optional[Dict[str, str]] = None,
        prefer_jsonl: bool = True,
        convert_pt_to_jsonl: bool = True,
    ) -> None:
        super().__init__()
        self.cache_paths = {split: Path(path).expanduser().resolve() for split, path in cache_paths.items()}
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.shuffle_train = bool(shuffle_train)
        self.drop_unreachable = bool(drop_unreachable)
        self.resources_cfg = resources
        self.prefer_jsonl = bool(prefer_jsonl)
        self.convert_pt_to_jsonl = bool(convert_pt_to_jsonl)

        self.train_dataset: Optional[GAgentPyGDataset] = None
        self.val_dataset: Optional[GAgentPyGDataset] = None
        self.test_dataset: Optional[GAgentPyGDataset] = None
        self.shared_resources: Optional[SharedDataResources] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.shared_resources is None and self.resources_cfg is not None:
            vocab = Path(self.resources_cfg["vocabulary_path"]).expanduser().resolve()
            emb = Path(self.resources_cfg["embeddings_dir"]).expanduser().resolve()
            self.shared_resources = SharedDataResources(vocabulary_path=vocab, embeddings_dir=emb)

        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("validation")
        if stage in (None, "test", "predict"):
            self.test_dataset = self._build_dataset("test")

    def train_dataloader(self) -> PyGDataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting train dataloader.")
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> PyGDataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting val dataloader.")
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> PyGDataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting test dataloader.")
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self) -> PyGDataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting predict dataloader.")
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def _build_dataset(self, split: str) -> Optional[GAgentPyGDataset]:
        path = self.cache_paths.get(split)
        if path is None:
            raise FileNotFoundError(f"g_agent cache path not configured for split={split}")
        if not path.exists():
            raise FileNotFoundError(f"g_agent cache not found for split={split}: {path}")
        return GAgentPyGDataset(
            path,
            drop_unreachable=self.drop_unreachable,
            prefer_jsonl=self.prefer_jsonl,
            convert_pt_to_jsonl=self.convert_pt_to_jsonl,
        )


__all__ = ["GAgentDataModule"]
