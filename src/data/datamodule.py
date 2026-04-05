from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import RetrievalDataset
from .collate import RetrievalCollator
from .retrieval import DataResource
from .preprocess_steps.metadata import load_metadata, metadata_path
from src.utils.lmdb_utils import (
    resolve_core_lmdb_paths,
    get_all_keys_from_lmdb,
    apply_filter_intersection,
)


class RetrievalDataModule(LightningDataModule):  # ← 改名
    def __init__(
        self,
        dataset_cfg: Any,
        batch_size: int,
        num_workers: int,
        eval_batch_size: Optional[int] = None,
        eval_num_workers: Optional[int] = None,
        pin_memory: bool = True,
        train_shuffle: bool = True,
        train_num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg

        # 只保存标量超参数，dataset_cfg 不参与 hparam 记录
        # （DictConfig 可序列化，但内容属于数据配置而非训练超参，语义上不该在此）
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_cfg"],
        )

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.eval_num_workers = (
            eval_num_workers if eval_num_workers is not None else num_workers
        )
        self.pin_memory = pin_memory
        self.train_shuffle = train_shuffle
        self.train_num_samples = train_num_samples

        self._shared_resources: Optional[DataResource] = None
        self.train_dataset: Optional[RetrievalDataset] = None
        self.val_dataset: Optional[RetrievalDataset] = None
        self.test_dataset: Optional[RetrievalDataset] = None
        self.predict_dataset: Optional[RetrievalDataset] = None  # ← 补上

    def prepare_data(self) -> None:
        """验证磁盘文件完整性，只在主进程运行一次。"""
        paths = self.dataset_cfg["paths"]
        for key in ["entity_metadata", "embeddings"]:
            p = Path(paths[key])
            if not p.exists():
                raise FileNotFoundError(f"Critical data path missing: {p}")

    def setup(self, stage: Optional[str] = None) -> None:
        """分阶段初始化资源和 Dataset。"""
        # 资源中心在当前进程内单例化（多卡训练每个 rank 各初始化一次，符合预期）
        if self._shared_resources is None:
            paths = self.dataset_cfg["paths"]
            self._shared_resources = DataResource(
                entity_metadata_path=Path(paths["entity_metadata"]),
                embeddings_dir=Path(paths["embeddings"]),
            )

        emb_dir = Path(self.dataset_cfg["paths"]["embeddings"])

        def _build_split(split_name: str) -> RetrievalDataset:
            lmdb_paths = resolve_core_lmdb_paths(emb_dir, split_name)
            metadata_file = metadata_path(emb_dir, split_name)
            if metadata_file.exists():
                all_ids = load_metadata(metadata_file).sample_ids
            else:
                all_ids = []
                for lp in lmdb_paths:
                    all_ids.extend(get_all_keys_from_lmdb(lp))
            filter_paths = self._get_filter_paths(split_name)
            final_ids = apply_filter_intersection(all_ids, filter_paths)
            return RetrievalDataset(
                sample_ids=final_ids,
                lmdb_paths=lmdb_paths,  # 直接传递Path对象
                split=split_name,
                lmdb_readahead=bool(self.dataset_cfg.get("lmdb_readahead", False)),
            )

        if stage in (None, "fit"):
            self.train_dataset = _build_split("train")
            self.val_dataset = _build_split("validation")

        if stage in (None, "validate"):  # ← Lightning 的 validate stage
            if self.val_dataset is None:
                self.val_dataset = _build_split("validation")

        if stage in (None, "test"):
            self.test_dataset = _build_split(self.dataset_cfg.get("eval_split", "test"))

        if stage == "predict":
            self.predict_dataset = _build_split(
                self.dataset_cfg.get("predict_split", "test")
            )

    def _get_filter_paths(self, split_name: str) -> List[Path]:
        filter_paths = [
            Path(p)
            for p in self.dataset_cfg.get("filter_files", {}).get(split_name, [])
        ]
        sample_filter_path = self.dataset_cfg.get("sample_filter_path")
        if sample_filter_path not in (None, ""):
            filter_paths.append(Path(sample_filter_path))
        processed_dir = (self.dataset_cfg.get("paths") or {}).get("processed")
        if processed_dir not in (None, ""):
            runtime_filter_missing_anchor = self.dataset_cfg.get(
                "runtime_filter_missing_anchor", {}
            )
            runtime_filter_missing_answer = self.dataset_cfg.get(
                "runtime_filter_missing_answer", {}
            )
            if bool(runtime_filter_missing_anchor.get(split_name, False)):
                filter_paths.append(Path(processed_dir) / "filter_missing_anchor.json")
            if bool(runtime_filter_missing_answer.get(split_name, False)):
                filter_paths.append(Path(processed_dir) / "filter_missing_answer.json")
        return filter_paths

    # ------------------------------------------------------------------ #
    # DataLoader 工厂
    # ------------------------------------------------------------------ #

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, training=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, training=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.test_dataset, training=False)

    def predict_dataloader(self) -> DataLoader:  # ← 补上，消除 Lightning warning
        return self._build_loader(self.predict_dataset, training=False)

    def _build_loader(
        self, dataset: Optional[RetrievalDataset], training: bool
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Did you call setup()?")
        if self._shared_resources is None:
            raise RuntimeError("DataResource not initialized. Did you call setup()?")
        collator = RetrievalCollator(self._shared_resources)
        sampler = None
        if training and self.train_num_samples:
            from .sampler import StepDrivenTrainSampler

            sampler = StepDrivenTrainSampler(
                dataset_size=len(dataset),
                num_samples=self.train_num_samples,
                shuffle=self.train_shuffle,
                seed=self.dataset_cfg.get("random_seed", 0),
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if training else self.eval_batch_size,
            num_workers=self.num_workers if training else self.eval_num_workers,
            collate_fn=collator,
            sampler=sampler,
            shuffle=(self.train_shuffle and sampler is None) if training else False,
            pin_memory=self.pin_memory,
            persistent_workers=(
                (self.num_workers if training else self.eval_num_workers) > 0
            ),
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def teardown(self, stage: Optional[str] = None) -> None:
        """释放 LMDB 句柄和 embedding mmap。"""
        for ds in [
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.predict_dataset,
        ]:
            if ds is not None:
                ds.close()

        if self._shared_resources is not None:
            self._shared_resources = None  # ← 防止悬空引用，重复 teardown 安全
