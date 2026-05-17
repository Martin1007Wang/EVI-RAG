from __future__ import annotations

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate import RetrievalCollator
from .dataset import RetrievalDataset
from src.training.config import ModelResources, TrainingDataConfig


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
    - no broad legacy schema handling.
    """

    def __init__(
        self,
        data_config: TrainingDataConfig,
    ) -> None:
        super().__init__()

        self.data_config = data_config
        self.batch_size = data_config.batch_size
        self.eval_batch_size = data_config.eval_batch_size
        self.num_workers = data_config.num_workers
        self.eval_num_workers = data_config.eval_num_workers
        self.pin_memory = data_config.pin_memory
        self.train_shuffle = data_config.train_shuffle
        self.drop_last = data_config.drop_last
        self.eval_drop_last = data_config.eval_drop_last
        self.lmdb_readahead = data_config.lmdb_readahead
        self.max_readers = data_config.max_readers
        self.metadata_dir = data_config.metadata_dir
        self.materialization = data_config.materialization
        self.train_split = data_config.train_split
        self.val_split = data_config.validation_split
        self.test_split = data_config.test_split

        self.collator = RetrievalCollator()

        self.train_dataset: RetrievalDataset | None = None
        self.val_dataset: RetrievalDataset | None = None
        self.test_dataset: RetrievalDataset | None = None
        self._datasets_by_split: dict[str, RetrievalDataset] = {}

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
        return None

    def setup(self, stage: str | None = None) -> None:
        self._ensure_model_resources_loaded()

        if stage in (None, "fit"):
            self.train_dataset = self._dataset_for_split(self.train_split)
            self.val_dataset = self._dataset_for_split(self.val_split)

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = self._dataset_for_split(self.val_split)

        if stage in (None, "test"):
            self.test_dataset = self._dataset_for_split(self.test_split)

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
        for dataset in self._datasets_by_split.values():
            dataset.close()
        self._datasets_by_split.clear()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _ensure_model_resources_loaded(self) -> None:
        if self._model_resources is not None:
            return
        self._model_resources = self.data_config.model_resources

    def _build_dataset(self, split: str) -> RetrievalDataset:
        return RetrievalDataset(
            materialization=self.materialization,
            split=split,
            lmdb_readahead=self.lmdb_readahead,
            max_readers=self.max_readers,
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

__all__ = [
    "ModelResources",
    "RetrievalDataModule",
]
