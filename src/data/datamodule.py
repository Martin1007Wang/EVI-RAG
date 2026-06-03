from __future__ import annotations

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .collate import RetrievalCollator
from .dataset import RetrievalDataset
from src.training.config import RetrievalDataConfig


class RetrievalDataModule(LightningDataModule):
    """
    LightningDataModule for materialized retrieval datasets.

    Responsibilities:
    - read materialized dataset paths from config;
    - instantiate RetrievalDataset for train/validation/test;
    - build DataLoader objects.

    Non-responsibilities:
    - no model construction;
    - no runtime sample filtering;
    - no sample deserialization logic;
    - no graph/path recomputation;
    - no broad deprecated schema handling.
    """

    def __init__(
        self,
        data_config: RetrievalDataConfig,
    ) -> None:
        super().__init__()

        self.cfg = data_config

        self.collator = RetrievalCollator()

        self.train_dataset: RetrievalDataset | None = None
        self.val_dataset: RetrievalDataset | None = None
        self.test_dataset: RetrievalDataset | None = None
        self._datasets_by_split: dict[str, RetrievalDataset] = {}

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        splits = self.cfg.splits

        if stage in (None, "fit"):
            self.train_dataset = self._dataset_for_split(splits.train)
            self.val_dataset = self._dataset_for_split(splits.validation)

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = self._dataset_for_split(splits.validation)

        if stage in (None, "test"):
            self.test_dataset = self._dataset_for_split(splits.test)

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self._loader_dataset(
                self.train_dataset,
                indices=self.cfg.loader.train_indices,
            ),
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self._loader_dataset(
                self.val_dataset,
                indices=self.cfg.loader.validation_indices,
            ),
            training=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self._loader_dataset(
                self.test_dataset,
                indices=self.cfg.loader.test_indices,
            ),
            training=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        for dataset in self._datasets_by_split.values():
            dataset.close()
        self._datasets_by_split.clear()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_dataset(self, split: str) -> RetrievalDataset:
        loader = self.cfg.loader
        return RetrievalDataset(
            materialization=self.cfg.materialization,
            split=split,
            lmdb_readahead=loader.lmdb_readahead,
            max_readers=loader.max_readers,
        )

    def _dataset_for_split(self, split: str) -> RetrievalDataset:
        dataset = self._datasets_by_split.get(split)
        if dataset is None:
            dataset = self._build_dataset(split)
            self._datasets_by_split[split] = dataset
        return dataset

    def _build_loader(
        self,
        *,
        dataset: RetrievalDataset | Subset[RetrievalDataset] | None,
        training: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        loader = self.cfg.loader
        batch_size = loader.batch_size if training else loader.eval_batch_size
        num_workers = loader.num_workers if training else loader.eval_num_workers

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=training and loader.train_shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=loader.pin_memory,
            prefetch_factor=loader.prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=loader.drop_last if training else False,
        )

    def _loader_dataset(
        self,
        dataset: RetrievalDataset | None,
        *,
        indices: tuple[int, ...] | None,
    ) -> RetrievalDataset | Subset[RetrievalDataset] | None:
        if dataset is None or indices is None:
            return dataset
        max_index = len(dataset) - 1
        for index in indices:
            if index < 0 or index > max_index:
                raise IndexError(
                    f"Subset index out of range for split {dataset.split}: "
                    f"{index} not in [0, {max_index}]."
                )
        return Subset(dataset, list(indices))

__all__ = [
    "RetrievalDataModule",
]
