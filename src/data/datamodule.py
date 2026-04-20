from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, List, Sequence
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.schema.batch import RetrievalBatch
from .dataset import RetrievalDataset
from .collate import RetrievalCollator
from .retrieval import DataResource
from .preprocess_steps.manifest import load_manifest, manifest_path
from src.utils.lmdb_utils import (
    resolve_core_lmdb_paths,
    get_all_keys_from_lmdb,
    apply_filter_intersection,
)
class RetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfg: Any,
        batch_size: int,
        num_workers: int,
        eval_batch_size: Optional[int] = None,
        eval_num_workers: Optional[int] = None,
        pin_memory: bool = True,
        train_shuffle: bool = True,
        train_edge_budget: Optional[int] = None,
        train_node_budget: Optional[int] = None,
        max_graphs_per_batch: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.save_hyperparameters(logger=False, ignore=["dataset_cfg"])
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.eval_num_workers = num_workers if eval_num_workers is None else eval_num_workers
        self.pin_memory = pin_memory
        self.train_shuffle = train_shuffle
        self.train_edge_budget = train_edge_budget
        self.train_node_budget = train_node_budget
        self.max_graphs_per_batch = max_graphs_per_batch
        self._shared_resources: Optional[DataResource] = None
        self.train_dataset: Optional[RetrievalDataset] = None
        self.val_dataset: Optional[RetrievalDataset] = None
        self.test_dataset: Optional[RetrievalDataset] = None
        self.predict_dataset: Optional[RetrievalDataset] = None
        self._train_sample_id_to_index: dict[str, int] = {}
    def prepare_data(self) -> None:
        paths = self.dataset_cfg["paths"]
        for key in ("entity_metadata", "embeddings"):
            p = Path(paths[key])
            if not p.exists():
                raise FileNotFoundError(f"Critical data path missing: {p}")
    def setup(self, stage: Optional[str] = None) -> None:
        if self._shared_resources is None:
            paths = self.dataset_cfg["paths"]
            self._shared_resources = DataResource(
                entity_metadata_path=Path(paths["entity_metadata"]),
                embeddings_dir=Path(paths["embeddings"]),
            )
        cfg = self.dataset_cfg
        emb_dir = Path(cfg["paths"]["embeddings"])
        splits = {
            "train":    cfg.get("train_split",   "train"),
            "val":      cfg.get("val_split",      "validation"),
            "test":     cfg.get("eval_split",     "test"),
            "predict":  cfg.get("predict_split",  cfg.get("eval_split", "test")),
        }
        def build(split_name: str) -> RetrievalDataset:
            lmdb_paths = resolve_core_lmdb_paths(emb_dir, split_name)
            meta_file = manifest_path(emb_dir, split_name)
            if meta_file.exists():
                meta = load_manifest(meta_file)
                all_ids = meta.sample_ids
            else:
                meta = None
                all_ids = [k for p in lmdb_paths for k in get_all_keys_from_lmdb(p)]
            final_ids = apply_filter_intersection(all_ids, self._get_filter_paths(split_name))
            return RetrievalDataset(
                sample_ids=final_ids,
                lmdb_paths=lmdb_paths,
                split=split_name,
                lmdb_readahead=bool(cfg.get("lmdb_readahead", False)),
                sample_num_nodes=(
                    _align_metadata(meta.sample_ids, final_ids, meta.num_nodes)
                    if meta is not None else None
                ),
                sample_num_edges=(
                    _align_metadata(meta.sample_ids, final_ids, meta.num_edges)
                    if meta is not None else None
                ),
            )
        if stage in (None, "fit"):
            self.train_dataset = build(splits["train"])
            self.val_dataset   = build(splits["val"])
            self._train_sample_id_to_index = {
                sid: i for i, sid in enumerate(self.train_dataset.sample_ids)
            }
        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = build(splits["val"])
        if stage in (None, "test"):
            self.test_dataset = build(splits["test"])
        if stage == "predict":
            self.predict_dataset = build(splits["predict"])
    def teardown(self, stage: Optional[str] = None) -> None:
        for ds in (self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset):
            if ds is not None:
                ds.close()
        self._shared_resources = None
        self._train_sample_id_to_index = {}
    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, training=True)
    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, training=False)
    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.test_dataset, training=False)
    def predict_dataloader(self) -> DataLoader:
        return self._build_loader(self.predict_dataset, training=False)
    def build_train_batch_from_ids(self, sample_ids: Sequence[str]) -> RetrievalBatch:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty.")
        if self.train_dataset is None or self._shared_resources is None:
            raise RuntimeError("Call setup('fit') before build_train_batch_from_ids().")
        missing = [sid for sid in sample_ids if sid not in self._train_sample_id_to_index]
        if missing:
            raise KeyError(f"Unknown train sample ids: {missing[:3]}")
        samples = [
            self.train_dataset.get(self._train_sample_id_to_index[sid])
            for sid in sample_ids
        ]
        return RetrievalCollator(self._shared_resources)(samples)
    def _build_loader(self, dataset: Optional[RetrievalDataset], training: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        if self._shared_resources is None:
            raise RuntimeError("DataResource not initialized. Call setup() first.")
        n_workers = self.num_workers if training else self.eval_num_workers
        collator = RetrievalCollator(self._shared_resources)
        if training:
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
                num_workers=n_workers,
                collate_fn=collator,
                pin_memory=self.pin_memory,
                persistent_workers=n_workers > 0,
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collator,
            pin_memory=self.pin_memory,
            persistent_workers=n_workers > 0,
        )
    def _get_filter_paths(self, split_name: str) -> List[Path]:
        paths = [Path(p) for p in self.dataset_cfg.get("filter_files", {}).get(split_name, [])]
        extra = self.dataset_cfg.get("sample_filter_path")
        if extra:
            paths.append(Path(extra))
        return paths
def _align_metadata(
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    values: torch.Tensor,
) -> list[int]:
    mapping = {sid: int(v) for sid, v in zip(source_ids, values.tolist())}
    missing = [sid for sid in target_ids if sid not in mapping]
    if missing:
        raise ValueError(f"Manifest missing sample ids, examples: {missing[:3]}")
    return [max(mapping[sid], 1) for sid in target_ids]
def _coerce_positive_int(value: Optional[int], name: str) -> int | None:
    if value in (None, 0):
        return None
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1 when set, got {v}.")
    return v