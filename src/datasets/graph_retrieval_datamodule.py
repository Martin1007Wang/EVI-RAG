from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, Optional, cast

from lightning import LightningDataModule
import torch

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .components import SharedDataResources
from .components.embeddings import attach_embeddings_to_batch
from .graph_retrieval_collate import build_retrieval_dataloader
from .graph_retrieval_dataset import (
    GraphRetrievalDataset,
    create_graph_retrieval_dataset,
)
from src.data.io.lmdb_utils import resolve_core_lmdb_paths
from src.graph_runtime import TrajectoryBatch

_EMBEDDINGS_DEVICE_CPU = "cpu"
_EMBEDDINGS_DEVICE_CUDA = "cuda"
_DEFAULT_TRAIN_NUM_SAMPLES = 1_000_000_000


def _canonicalize_dataset_cfg(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset_cfg to the SSOT representation: `paths.entity_vocab` + `paths.relation_vocab` + `paths.embeddings`."""

    cfg = dict(dataset_cfg)
    paths = cfg.get("paths")
    if (
        isinstance(paths, dict)
        and paths.get("entity_vocab")
        and paths.get("relation_vocab")
        and paths.get("embeddings")
    ):
        return cfg
    raise ValueError(
        "dataset_cfg must define `paths.entity_vocab`, `paths.relation_vocab`, and `paths.embeddings`."
    )


def _normalize_embeddings_device(embeddings_device: str | None) -> str | None:
    normalized = None if embeddings_device is None else str(embeddings_device).lower()
    if normalized not in (None, _EMBEDDINGS_DEVICE_CPU, _EMBEDDINGS_DEVICE_CUDA):
        raise ValueError(f"embeddings_device must be cpu or cuda, got {normalized!r}.")
    if normalized == _EMBEDDINGS_DEVICE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "embeddings_device=cuda requested but CUDA is not available."
        )
    return normalized


def _normalize_split_key(split: str | None) -> str:
    normalized = str(split or "test").strip().lower()
    if normalized in {"train", "validation", "test"}:
        return normalized
    raise ValueError("eval_split must be one of {'train', 'validation', 'test'}.")


def _resolve_dataset_cfg(dataset_cfg: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(dataset_cfg, DictConfig):
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)  # type: ignore[arg-type]
    else:
        cfg = dataset_cfg
    if not isinstance(cfg, dict):
        raise TypeError(f"dataset_cfg must be a mapping, got {type(cfg)!r}")
    return _canonicalize_dataset_cfg(cfg)


def _resolve_distributed_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(
            torch.distributed.get_world_size()
        )
    return 0, 1


class StepDrivenTrainSampler(Sampler[int]):
    """Large finite sampler that keeps fit loops step-driven instead of dataset-driven."""

    def __init__(
        self,
        *,
        dataset_size: int,
        num_samples: int,
        shuffle: bool,
        seed: int | None = None,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("train dataset must contain at least one sample.")
        if num_samples < 1:
            raise ValueError("train_num_samples must be >= 1.")
        self.dataset_size = int(dataset_size)
        self.num_samples = int(num_samples)
        self.shuffle = bool(shuffle)
        self.seed = 0 if seed is None else int(seed)
        self.epoch = 0
        self.rank, self.world_size = _resolve_distributed_rank_and_world_size()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        return (
            self._iter_random_indices() if self.shuffle else self._iter_cycled_indices()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_random_indices(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch * max(self.world_size, 1) + self.rank
        )
        remaining = self.num_samples
        chunk_size = 4096
        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            indices = torch.randint(
                low=0,
                high=self.dataset_size,
                size=(current_chunk,),
                generator=generator,
            )
            for index in indices.tolist():
                yield int(index)
            remaining -= current_chunk

    def _iter_cycled_indices(self) -> Iterator[int]:
        stride = max(self.world_size, 1)
        start = (self.rank + self.epoch * self.num_samples * stride) % self.dataset_size
        for offset in range(self.num_samples):
            yield int((start + offset * stride) % self.dataset_size)


class GraphRetrievalDataModule(LightningDataModule):
    """
    Refactored GraphRetrievalDataModule following System Engineering principles.

    Principles:
    1. Dependency Injection: Receives a full `dataset_cfg` object.
    2. Zero Logic Config: Paths are resolved in YAML, not Python.
    3. Separation of Concerns: DataModule handles Logistics (batching), Dataset handles Logic.
    """

    def __init__(
        self,
        *,
        dataset_cfg: Any,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
        train_shuffle: bool = True,
        train_num_samples: int = _DEFAULT_TRAIN_NUM_SAMPLES,
        prefetch_factor: int | None = 2,
        persistent_workers: bool = False,
        precompute_edge_batch: bool = False,
        embeddings_device: str | None = None,
        eval_split: str = "test",
        splits: Optional[Dict[str, str]] = None,
        expand_multi_answer: bool = True,
        filter_zero_hop: bool = True,
    ) -> None:
        super().__init__()
        embeddings_device = _normalize_embeddings_device(embeddings_device)

        # dataset_cfg 可能包含 OmegaConf 对象；避免写入 checkpoint 元数据。
        self.save_hyperparameters(logger=False, ignore=["dataset_cfg"])
        self.dataset_cfg = _resolve_dataset_cfg(dataset_cfg)
        self._init_dataloader_cfg(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_shuffle=train_shuffle,
            train_num_samples=train_num_samples,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            precompute_edge_batch=precompute_edge_batch,
        )
        self._init_runtime_state(
            embeddings_device=embeddings_device,
            eval_split=eval_split,
            splits=splits,
            expand_multi_answer=expand_multi_answer,
            filter_zero_hop=filter_zero_hop,
        )

    def _init_dataloader_cfg(
        self,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        train_shuffle: bool,
        train_num_samples: int,
        prefetch_factor: int | None,
        persistent_workers: bool,
        precompute_edge_batch: bool,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_shuffle = bool(train_shuffle)
        self.train_num_samples = int(train_num_samples)
        if self.train_num_samples < 1:
            raise ValueError("train_num_samples must be >= 1.")
        self.persistent_workers = persistent_workers
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.precompute_edge_batch = bool(precompute_edge_batch)

    def _init_runtime_state(
        self,
        *,
        embeddings_device: str | None,
        eval_split: str,
        splits: Optional[Dict[str, str]],
        expand_multi_answer: bool,
        filter_zero_hop: bool,
    ) -> None:
        self.embeddings_device = (
            None if embeddings_device is None else str(embeddings_device)
        )
        self.eval_split = _normalize_split_key(eval_split)
        self.expand_multi_answer = bool(expand_multi_answer)
        self.filter_zero_hop = bool(filter_zero_hop)
        self.dataset_scope = _resolve_dataset_scope(self.dataset_cfg)
        self.splits = splits or {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }
        self.train_dataset: Optional[GraphRetrievalDataset] = None
        self.val_dataset: Optional[GraphRetrievalDataset] = None
        self.test_dataset: Optional[GraphRetrievalDataset] = None
        self.eval_dataset: Optional[GraphRetrievalDataset] = None
        self.batch_size_per_device = self.batch_size
        self._shared_resources: Optional[SharedDataResources] = None

    @property
    def shared_resources(self) -> Optional[SharedDataResources]:
        return self._shared_resources

    def set_eval_split(self, split: str) -> None:
        self.eval_split = _normalize_split_key(split)

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
        if (
            "entity_vocab" not in paths
            or "relation_vocab" not in paths
            or "embeddings" not in paths
        ):
            raise ValueError(
                "Invalid dataset_cfg: expected `paths.entity_vocab`, `paths.relation_vocab`, and `paths.embeddings`."
            )

        entity_vocab_path = Path(paths["entity_vocab"])
        relation_vocab_path = Path(paths["relation_vocab"])
        emb_dir = Path(paths["embeddings"])

        missing = []
        if not entity_vocab_path.exists():
            missing.append(f"Entity vocab: {entity_vocab_path}")
        if not relation_vocab_path.exists():
            missing.append(f"Relation vocab: {relation_vocab_path}")
        if not emb_dir.exists():
            missing.append(f"Embeddings Dir: {emb_dir}")

        if missing:
            raise FileNotFoundError(
                f"Critical Data Error: The following injected paths do not exist:\n"
                + "\n".join(missing)
                + "\nPlease check 'configs/dataset/YOUR_DATASET.yaml'."
            )
        for split_name in sorted(set(self.splits.values())):
            resolve_core_lmdb_paths(emb_dir, split_name)

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Batch size is defined per device; keep as-is for DDP.
        self.batch_size_per_device = self.batch_size

        # Eval runners can replay multiple splits by mutating `run.split` between
        # fresh datamodule instantiations. Cache canonical datasets by split key,
        # then expose the requested eval split through `eval_dataset`.
        if stage in (None, "fit"):
            self.train_dataset = self._ensure_split_dataset("train")
            self.val_dataset = self._ensure_split_dataset("validation")
        if stage in (None, "test", "predict"):
            self.eval_dataset = self._ensure_split_dataset(self.eval_split)

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            shuffle=self.train_shuffle,
            drop_last=self.drop_last,
            training=True,
        )

    def val_dataloader(self):
        return self._build_loader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            training=False,
        )

    def test_dataloader(self):
        return self._build_loader(
            self._resolve_eval_dataset(),
            shuffle=False,
            drop_last=False,
            training=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self._build_loader(
            self._resolve_eval_dataset(),
            shuffle=False,
            drop_last=False,
            training=False,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        _ = dataloader_idx
        resources = self._shared_resources
        if resources is None:
            return batch
        if not hasattr(batch, "node_embeddings") or not hasattr(
            batch, "edge_embeddings"
        ):
            attach_embeddings_to_batch(
                batch,
                global_embeddings=resources.global_embeddings,
                embeddings_device=self.embeddings_device,
            )
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        _ = dataloader_idx
        if isinstance(batch, TrajectoryBatch):
            return batch
        device = _infer_batch_device(batch)
        return TrajectoryBatch.from_pyg_batch(
            batch,
            device=device,
            dataset_scope=self.dataset_scope,
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        _ = dataloader_idx
        if isinstance(batch, TrajectoryBatch):
            return batch.to(device)
        if hasattr(batch, "to"):
            batch = batch.to(device)
        return TrajectoryBatch.from_pyg_batch(
            batch,
            device=device,
            dataset_scope=self.dataset_scope,
        )

    def train_eval_dataloader(self) -> DataLoader:
        """
        Deterministic loader for train split during evaluation/export stages.
        """
        return self._build_loader(
            self.train_dataset,
            shuffle=False,
            drop_last=False,
            training=False,
        )

    def get_split_dataloader(self, split: str) -> DataLoader:
        split_key = _normalize_split_key(split)
        dataset = self._ensure_split_dataset(split_key)
        return self._build_loader(
            dataset,
            shuffle=False,
            drop_last=False,
            training=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        del stage
        seen_dataset_ids: set[int] = set()
        for dataset in (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.eval_dataset,
        ):
            if dataset is None:
                continue
            dataset_id = id(dataset)
            if dataset_id in seen_dataset_ids:
                continue
            dataset.close()
            seen_dataset_ids.add(dataset_id)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.eval_dataset = None
        if self._shared_resources is not None:
            self._shared_resources.clear()
        self._shared_resources = None

    def _ensure_shared_resources(self) -> SharedDataResources:
        if self._shared_resources is None:
            paths = self.dataset_cfg["paths"]
            self._shared_resources = SharedDataResources(
                entity_vocab_path=Path(paths["entity_vocab"]),
                relation_vocab_path=Path(paths["relation_vocab"]),
                embeddings_dir=Path(paths["embeddings"]),
                embeddings_device=self.embeddings_device,
                heuristic_log_v_path=(
                    Path(paths["heuristic_log_v"]).expanduser().resolve()
                    if isinstance(paths, dict) and paths.get("heuristic_log_v")
                    else None
                ),
            )
        return self._shared_resources

    @staticmethod
    def _dataset_attr_name(split_key: str) -> str:
        return {
            "train": "train_dataset",
            "validation": "val_dataset",
            "test": "test_dataset",
        }[_normalize_split_key(split_key)]

    def _resolve_split_name(self, split_key: str) -> str:
        canonical_split = _normalize_split_key(split_key)
        split_name = self.splits.get(canonical_split)
        if split_name in (None, ""):
            raise ValueError(
                "splits must define mappings for train/validation/test. "
                f"Missing split={canonical_split!r}."
            )
        return str(split_name)

    def _ensure_split_dataset(self, split_key: str) -> GraphRetrievalDataset:
        canonical_split = _normalize_split_key(split_key)
        dataset_attr = self._dataset_attr_name(canonical_split)
        dataset = getattr(self, dataset_attr)
        if dataset is not None:
            return dataset
        dataset = create_graph_retrieval_dataset(
            cfg=self.dataset_cfg,
            split_name=self._resolve_split_name(canonical_split),
            resources=self._ensure_shared_resources(),
        )
        setattr(self, dataset_attr, dataset)
        return dataset

    def _resolve_eval_dataset(self) -> GraphRetrievalDataset:
        if self.eval_dataset is not None:
            return self.eval_dataset
        self.eval_dataset = self._ensure_split_dataset(self.eval_split)
        return self.eval_dataset

    def _build_loader(
        self,
        dataset: GraphRetrievalDataset,
        *,
        shuffle: bool,
        drop_last: bool,
        training: bool,
    ) -> DataLoader:
        """
        Constructs the retrieval DataLoader using params injected via dataset_cfg.
        """
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Did you run setup()?")

        sampler = self._build_sampler(
            dataset,
            training=training,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return build_retrieval_dataloader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            precompute_edge_batch=self.precompute_edge_batch,
            random_seed=self.dataset_cfg.get("random_seed"),
            expand_multi_answer=self.expand_multi_answer,
            filter_zero_hop=self.filter_zero_hop,
        )

    def _build_sampler(
        self,
        dataset: GraphRetrievalDataset,
        *,
        training: bool,
        shuffle: bool,
        drop_last: bool,
    ) -> Sampler[int] | None:
        if training:
            return StepDrivenTrainSampler(
                dataset_size=len(dataset),
                num_samples=self.train_num_samples,
                shuffle=shuffle,
                seed=self.dataset_cfg.get("random_seed"),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        return None


def _resolve_dataset_scope(dataset_cfg: Dict[str, Any]) -> str:
    scope = str(dataset_cfg.get("dataset_scope", "")).strip().lower()
    if scope in {"full", "sub"}:
        return scope
    name = str(dataset_cfg.get("name", "") or "")
    return "sub" if name.endswith("-sub") else "full"


def _infer_batch_device(batch: Any) -> torch.device:
    for attr in ("edge_index", "node_embeddings", "edge_attr", "question_emb"):
        value = getattr(batch, attr, None)
        if torch.is_tensor(value):
            return cast(torch.Tensor, value).device
    return torch.device("cpu")
