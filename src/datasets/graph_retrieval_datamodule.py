from __future__ import annotations

from collections.abc import Iterable, Iterator
import multiprocessing as mp
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
from src.graph import TrajectoryBatch

_EMBEDDINGS_DEVICE_CPU = "cpu"
_EMBEDDINGS_DEVICE_CUDA = "cuda"
_LEGACY_TRAIN_NUM_SAMPLES_SENTINEL = 1_000_000_000
_FEATURE_DTYPE_ALIASES = {
    None: None,
    "": None,
    "none": None,
    "fp32": torch.float32,
    "float32": torch.float32,
    "32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "16": torch.float16,
}
_FLOAT_BATCH_FEATURE_ATTRS = (
    "node_embeddings",
    "edge_embeddings",
    "relation_embeddings",
    "question_emb",
    "question_ctx",
    "heuristic_log_v",
)


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


def _resolve_feature_dtype(dtype_name: str | None) -> torch.dtype | None:
    normalized = None if dtype_name is None else str(dtype_name).strip().lower()
    if normalized not in _FEATURE_DTYPE_ALIASES:
        raise ValueError(
            "feature dtype must be one of {None, fp32, bf16, fp16}, "
            f"got {dtype_name!r}."
        )
    return _FEATURE_DTYPE_ALIASES[normalized]


def _normalize_split_key(split: str | None) -> str:
    normalized = str(split or "test").strip().lower()
    if normalized in {"train", "validation", "test"}:
        return normalized
    raise ValueError("eval_split must be one of {'train', 'validation', 'test'}.")


def _normalize_multiprocessing_context(context_name: str | None) -> str | None:
    normalized = None if context_name is None else str(context_name).strip().lower()
    if normalized in (None, "", "none"):
        return None
    valid_methods = {method.lower() for method in mp.get_all_start_methods()}
    if normalized not in valid_methods:
        valid_text = ", ".join(sorted(valid_methods))
        raise ValueError(
            "multiprocessing_context must be one of "
            f"{{None, {valid_text}}}, got {context_name!r}."
        )
    return normalized


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


def _resolve_optional_length(iterable: object) -> int | None:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, NotImplementedError):
        return None


def _normalize_train_num_samples(num_samples: Any) -> int | None:
    if num_samples in (None, "", 0):
        return None
    resolved = int(num_samples)
    if resolved < 1:
        raise ValueError("train_num_samples must be >= 1 when set.")
    if resolved == _LEGACY_TRAIN_NUM_SAMPLES_SENTINEL:
        return None
    return resolved


class StepDrivenTrainSampler(Sampler[int]):
    """Step-driven sampler with an unsized default for Lightning training loops."""

    def __init__(
        self,
        *,
        dataset_size: int,
        num_samples: int | None = None,
        shuffle: bool,
        seed: int | None = None,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("train dataset must contain at least one sample.")
        self.dataset_size = int(dataset_size)
        self.num_samples = None if num_samples is None else int(num_samples)
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("train_num_samples must be >= 1 when set.")
        self.shuffle = bool(shuffle)
        self.seed = 0 if seed is None else int(seed)
        self.epoch = 0
        self.rank, self.world_size = _resolve_distributed_rank_and_world_size()

    def __len__(self) -> int:
        if self.num_samples is None:
            raise TypeError(
                "StepDrivenTrainSampler is intentionally unsized. "
                "Control training lifetime with trainer.max_steps or fit_schedule."
            )
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
        while remaining is None or remaining > 0:
            current_chunk = (
                chunk_size if remaining is None else min(chunk_size, remaining)
            )
            indices = torch.randint(
                low=0,
                high=self.dataset_size,
                size=(current_chunk,),
                generator=generator,
            )
            for index in indices.tolist():
                yield int(index)
            if remaining is not None:
                remaining -= current_chunk

    def _iter_cycled_indices(self) -> Iterator[int]:
        stride = max(self.world_size, 1)
        if self.num_samples is None:
            index = (self.rank + self.epoch * stride) % self.dataset_size
            while True:
                yield int(index)
                index = (index + stride) % self.dataset_size
            return
        start = (self.rank + self.epoch * self.num_samples * stride) % self.dataset_size
        for offset in range(self.num_samples):
            yield int((start + offset * stride) % self.dataset_size)


class BudgetBatchSampler(Sampler[list[int]]):
    """Dynamic batcher that caps graphs and graph-size budgets per batch."""

    def __init__(
        self,
        *,
        sampler: Iterable[int],
        dataset: GraphRetrievalDataset,
        max_graphs_per_batch: int | None,
        max_nodes_per_batch: int | None,
        max_edges_per_batch: int | None,
        max_question_tokens_per_batch: int | None,
        drop_last: bool,
    ) -> None:
        self.sampler = sampler
        self.dataset = dataset
        self.max_graphs_per_batch = (
            None if max_graphs_per_batch is None else int(max_graphs_per_batch)
        )
        self.max_nodes_per_batch = (
            None if max_nodes_per_batch is None else int(max_nodes_per_batch)
        )
        self.max_edges_per_batch = (
            None if max_edges_per_batch is None else int(max_edges_per_batch)
        )
        self.max_question_tokens_per_batch = (
            None
            if max_question_tokens_per_batch is None
            else int(max_question_tokens_per_batch)
        )
        self.drop_last = bool(drop_last)
        if all(
            limit is None
            for limit in (
                self.max_graphs_per_batch,
                self.max_nodes_per_batch,
                self.max_edges_per_batch,
                self.max_question_tokens_per_batch,
            )
        ):
            raise ValueError("BudgetBatchSampler requires at least one active limit.")
        for name, value in (
            ("max_graphs_per_batch", self.max_graphs_per_batch),
            ("max_nodes_per_batch", self.max_nodes_per_batch),
            ("max_edges_per_batch", self.max_edges_per_batch),
            ("max_question_tokens_per_batch", self.max_question_tokens_per_batch),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when set.")
        self._cached_length: int | None = None
        self._cached_length_key: tuple[int, ...] | None = None

    def __len__(self) -> int:
        cache_key = self._length_cache_key()
        if cache_key is None:
            raise TypeError(
                "BudgetBatchSampler length is undefined when backed by an unsized sampler."
            )
        if self._cached_length_key == cache_key and self._cached_length is not None:
            return self._cached_length
        length = sum(1 for _ in self._iter_batches())
        self._cached_length_key = cache_key
        self._cached_length = length
        return length

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._iter_batches()

    def _iter_batches(self) -> Iterator[list[int]]:
        batch: list[int] = []
        nodes = 0
        edges = 0
        question_tokens = 0
        for raw_idx in self.sampler:
            idx = int(raw_idx)
            stats = self.dataset.get_sample_stats(idx)
            if self._would_exceed(
                batch_size=1,
                nodes=int(stats.num_nodes),
                edges=int(stats.num_edges),
                question_tokens=int(stats.question_tokens),
            ):
                raise ValueError(
                    "Single sample exceeds batch budget: "
                    f"idx={idx} nodes={int(stats.num_nodes)} edges={int(stats.num_edges)} "
                    f"question_tokens={int(stats.question_tokens)}. Increase the active batch limits."
                )
            would_exceed = bool(batch) and self._would_exceed(
                batch_size=len(batch) + 1,
                nodes=nodes + int(stats.num_nodes),
                edges=edges + int(stats.num_edges),
                question_tokens=question_tokens + int(stats.question_tokens),
            )
            if would_exceed:
                yield batch
                batch = []
                nodes = 0
                edges = 0
                question_tokens = 0
            batch.append(idx)
            nodes += int(stats.num_nodes)
            edges += int(stats.num_edges)
            question_tokens += int(stats.question_tokens)
        if batch and not self.drop_last:
            yield batch

    def _length_cache_key(self) -> tuple[int, ...] | None:
        sampler_length = _resolve_optional_length(self.sampler)
        if sampler_length is None:
            return None
        epoch = getattr(self.sampler, "epoch", -1)
        return (
            int(epoch),
            sampler_length,
            int(self.drop_last),
            -1 if self.max_graphs_per_batch is None else int(self.max_graphs_per_batch),
            -1 if self.max_nodes_per_batch is None else int(self.max_nodes_per_batch),
            -1 if self.max_edges_per_batch is None else int(self.max_edges_per_batch),
            -1
            if self.max_question_tokens_per_batch is None
            else int(self.max_question_tokens_per_batch),
        )

    def _would_exceed(
        self,
        *,
        batch_size: int,
        nodes: int,
        edges: int,
        question_tokens: int,
    ) -> bool:
        return bool(
            (
                self.max_graphs_per_batch is not None
                and batch_size > self.max_graphs_per_batch
            )
            or (
                self.max_nodes_per_batch is not None
                and nodes > self.max_nodes_per_batch
            )
            or (
                self.max_edges_per_batch is not None
                and edges > self.max_edges_per_batch
            )
            or (
                self.max_question_tokens_per_batch is not None
                and question_tokens > self.max_question_tokens_per_batch
            )
        )


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
        eval_batch_size: int | None = None,
        num_workers: int,
        eval_num_workers: int | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
        train_shuffle: bool = True,
        train_num_samples: int | None = None,
        prefetch_factor: int | None = 2,
        eval_prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        eval_persistent_workers: bool | None = None,
        multiprocessing_context: str | None = None,
        eval_multiprocessing_context: str | None = None,
        precompute_edge_batch: bool = False,
        embeddings_device: str | None = None,
        train_feature_dtype: str | None = None,
        eval_feature_dtype: str | None = None,
        eval_split: str = "test",
        splits: Optional[Dict[str, str]] = None,
        expand_multi_answer: bool = True,
        filter_zero_hop: bool = True,
        train_max_graphs_per_batch: int | None = None,
        train_max_nodes_per_batch: int | None = None,
        train_max_edges_per_batch: int | None = None,
        train_max_question_tokens_per_batch: int | None = None,
    ) -> None:
        super().__init__()
        embeddings_device = _normalize_embeddings_device(embeddings_device)

        # dataset_cfg 可能包含 OmegaConf 对象；避免写入 checkpoint 元数据。
        self.save_hyperparameters(logger=False, ignore=["dataset_cfg"])
        self.dataset_cfg = _resolve_dataset_cfg(dataset_cfg)
        self._init_dataloader_cfg(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            eval_num_workers=eval_num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_shuffle=train_shuffle,
            train_num_samples=train_num_samples,
            prefetch_factor=prefetch_factor,
            eval_prefetch_factor=eval_prefetch_factor,
            persistent_workers=persistent_workers,
            eval_persistent_workers=eval_persistent_workers,
            multiprocessing_context=multiprocessing_context,
            eval_multiprocessing_context=eval_multiprocessing_context,
            precompute_edge_batch=precompute_edge_batch,
            train_max_graphs_per_batch=train_max_graphs_per_batch,
            train_max_nodes_per_batch=train_max_nodes_per_batch,
            train_max_edges_per_batch=train_max_edges_per_batch,
            train_max_question_tokens_per_batch=train_max_question_tokens_per_batch,
        )
        self._init_runtime_state(
            embeddings_device=embeddings_device,
            train_feature_dtype=train_feature_dtype,
            eval_feature_dtype=eval_feature_dtype,
            eval_split=eval_split,
            splits=splits,
            expand_multi_answer=expand_multi_answer,
            filter_zero_hop=filter_zero_hop,
        )

    def _init_dataloader_cfg(
        self,
        *,
        batch_size: int,
        eval_batch_size: int | None,
        num_workers: int,
        eval_num_workers: int | None,
        pin_memory: bool,
        drop_last: bool,
        train_shuffle: bool,
        train_num_samples: int | None,
        prefetch_factor: int | None,
        eval_prefetch_factor: int | None,
        persistent_workers: bool,
        eval_persistent_workers: bool | None,
        multiprocessing_context: str | None,
        eval_multiprocessing_context: str | None,
        precompute_edge_batch: bool,
        train_max_graphs_per_batch: int | None,
        train_max_nodes_per_batch: int | None,
        train_max_edges_per_batch: int | None,
        train_max_question_tokens_per_batch: int | None,
    ) -> None:
        self.batch_size = batch_size
        self.eval_batch_size = (
            self.batch_size if eval_batch_size is None else int(eval_batch_size)
        )
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        self.num_workers = num_workers
        self.eval_num_workers = (
            self.num_workers if eval_num_workers is None else int(eval_num_workers)
        )
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_shuffle = bool(train_shuffle)
        self.train_num_samples = _normalize_train_num_samples(train_num_samples)
        self.persistent_workers = persistent_workers
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.eval_prefetch_factor = (
            self.prefetch_factor
            if eval_prefetch_factor is None
            else int(eval_prefetch_factor)
        )
        self.eval_persistent_workers = (
            self.persistent_workers
            if eval_persistent_workers is None
            else bool(eval_persistent_workers)
        )
        self.multiprocessing_context = _normalize_multiprocessing_context(
            multiprocessing_context
        )
        self.eval_multiprocessing_context = (
            self.multiprocessing_context
            if eval_multiprocessing_context is None
            else _normalize_multiprocessing_context(eval_multiprocessing_context)
        )
        if self.num_workers < 0 or self.eval_num_workers < 0:
            raise ValueError("num_workers and eval_num_workers must be >= 0.")
        if self.num_workers == 0:
            self.multiprocessing_context = None
        if self.eval_num_workers == 0:
            self.eval_prefetch_factor = None
            self.eval_persistent_workers = False
            self.eval_multiprocessing_context = None
        self.precompute_edge_batch = bool(precompute_edge_batch)
        self.train_max_graphs_per_batch = (
            None
            if train_max_graphs_per_batch is None
            else int(train_max_graphs_per_batch)
        )
        self.train_max_nodes_per_batch = (
            None
            if train_max_nodes_per_batch is None
            else int(train_max_nodes_per_batch)
        )
        self.train_max_edges_per_batch = (
            None
            if train_max_edges_per_batch is None
            else int(train_max_edges_per_batch)
        )
        self.train_max_question_tokens_per_batch = (
            None
            if train_max_question_tokens_per_batch is None
            else int(train_max_question_tokens_per_batch)
        )

    def _init_runtime_state(
        self,
        *,
        embeddings_device: str | None,
        train_feature_dtype: str | None,
        eval_feature_dtype: str | None,
        eval_split: str,
        splits: Optional[Dict[str, str]],
        expand_multi_answer: bool,
        filter_zero_hop: bool,
    ) -> None:
        self.embeddings_device = (
            None if embeddings_device is None else str(embeddings_device)
        )
        self.train_feature_dtype = _resolve_feature_dtype(train_feature_dtype)
        self.eval_feature_dtype = _resolve_feature_dtype(eval_feature_dtype)
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
        normalized = _normalize_split_key(split)
        if normalized == self.eval_split:
            return
        self.eval_split = normalized
        self.eval_dataset = None

    def replace_dataset_cfg(
        self,
        dataset_cfg: Any,
        *,
        eval_split: str | None = None,
    ) -> None:
        self.teardown()
        self.dataset_cfg = _resolve_dataset_cfg(dataset_cfg)
        self.dataset_scope = _resolve_dataset_scope(self.dataset_cfg)
        if eval_split is not None:
            self.eval_split = _normalize_split_key(eval_split)

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
        feature_dtype = self._current_feature_dtype()
        if (
            not _has_attached_tensor(batch, "node_embeddings")
            or not _has_attached_tensor(batch, "relation_embeddings")
            or not _has_attached_tensor(batch, "edge_rel_local")
        ):
            attach_embeddings_to_batch(
                batch,
                global_embeddings=resources.global_embeddings,
                embeddings_device=_resolve_embedding_attachment_device(
                    self.embeddings_device,
                    trainer=getattr(self, "trainer", None),
                ),
                feature_dtype=feature_dtype,
            )
        if feature_dtype is not None:
            _cast_batch_float_features_inplace(batch, feature_dtype=feature_dtype)
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
        feature_dtype = self._current_feature_dtype()
        if isinstance(batch, TrajectoryBatch):
            return batch.to(device, feature_dtype=feature_dtype)
        if hasattr(batch, "to"):
            batch = batch.to(device)
        runtime_batch = TrajectoryBatch.from_pyg_batch(
            batch,
            device=device,
            dataset_scope=self.dataset_scope,
        )
        if feature_dtype is None:
            return runtime_batch
        return runtime_batch.to(device, feature_dtype=feature_dtype)

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
        batch_sampler = self._build_batch_sampler(
            dataset,
            sampler=sampler,
            training=training,
            drop_last=drop_last,
        )
        (
            num_workers,
            prefetch_factor,
            persistent_workers,
            multiprocessing_context,
        ) = self._loader_worker_cfg(training=training)

        return build_retrieval_dataloader(
            dataset,
            loader_name=self._loader_name(training=training, dataset=dataset),
            batch_size=(
                self.batch_size_per_device if training else int(self.eval_batch_size)
            ),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            precompute_edge_batch=self.precompute_edge_batch,
            random_seed=self.dataset_cfg.get("random_seed"),
            expand_multi_answer=self.expand_multi_answer,
            filter_zero_hop=self.filter_zero_hop,
        )

    def _loader_worker_cfg(
        self,
        *,
        training: bool,
    ) -> tuple[int, int | None, bool, str | None]:
        if training:
            return (
                self.num_workers,
                self.prefetch_factor,
                self.persistent_workers,
                self.multiprocessing_context,
            )
        return (
            self.eval_num_workers,
            self.eval_prefetch_factor,
            self.eval_persistent_workers,
            self.eval_multiprocessing_context,
        )

    @staticmethod
    def _loader_name(*, training: bool, dataset: GraphRetrievalDataset) -> str:
        if training:
            return f"train:{dataset.split}"
        return f"eval:{dataset.split}"

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

    def _build_batch_sampler(
        self,
        dataset: GraphRetrievalDataset,
        *,
        sampler: Sampler[int] | None,
        training: bool,
        drop_last: bool,
    ) -> Sampler[list[int]] | None:
        if not training:
            return None
        if all(
            limit is None
            for limit in (
                self.train_max_graphs_per_batch,
                self.train_max_nodes_per_batch,
                self.train_max_edges_per_batch,
                self.train_max_question_tokens_per_batch,
            )
        ):
            return None
        base_sampler = sampler
        if base_sampler is None:
            raise RuntimeError(
                "training batch budget requires a base sampler to provide sample indices."
            )
        return BudgetBatchSampler(
            sampler=base_sampler,
            dataset=dataset,
            max_graphs_per_batch=(
                self.train_max_graphs_per_batch
                if self.train_max_graphs_per_batch is not None
                else self.batch_size_per_device
            ),
            max_nodes_per_batch=self.train_max_nodes_per_batch,
            max_edges_per_batch=self.train_max_edges_per_batch,
            max_question_tokens_per_batch=self.train_max_question_tokens_per_batch,
            drop_last=drop_last,
        )

    def _current_feature_dtype(self) -> torch.dtype | None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return self.eval_feature_dtype
        trainer_state = getattr(trainer, "state", None)
        trainer_fn = str(getattr(trainer_state, "fn", "") or "").lower()
        if "fit" in trainer_fn or "train" in trainer_fn:
            return self.train_feature_dtype
        return self.eval_feature_dtype


def _resolve_dataset_scope(dataset_cfg: Dict[str, Any]) -> str:
    scope = str(dataset_cfg.get("dataset_scope", "")).strip().lower()
    if scope in {"full", "sub"}:
        return scope
    name = str(dataset_cfg.get("name", "") or "")
    return "sub" if name.endswith("-sub") else "full"


def _infer_batch_device(batch: Any) -> torch.device:
    for attr in (
        "edge_index",
        "node_embeddings",
        "relation_embeddings",
        "edge_attr",
        "question_emb",
    ):
        value = getattr(batch, attr, None)
        if torch.is_tensor(value):
            return cast(torch.Tensor, value).device
    return torch.device("cpu")


def _has_attached_tensor(batch: Any, name: str) -> bool:
    return torch.is_tensor(getattr(batch, name, None))


def _cast_batch_float_features_inplace(
    batch: Any, *, feature_dtype: torch.dtype
) -> None:
    for attr in _FLOAT_BATCH_FEATURE_ATTRS:
        value = getattr(batch, attr, None)
        if torch.is_tensor(value) and torch.is_floating_point(value):
            tensor = cast(torch.Tensor, value)
            if tensor.dtype != feature_dtype:
                setattr(batch, attr, tensor.to(dtype=feature_dtype))


def _resolve_trainer_root_device(trainer: Any) -> torch.device | None:
    if trainer is None:
        return None
    strategy = getattr(trainer, "strategy", None)
    root_device = getattr(strategy, "root_device", None)
    if isinstance(root_device, torch.device):
        return root_device
    if root_device is not None:
        return torch.device(root_device)
    module = getattr(trainer, "lightning_module", None)
    module_device = getattr(module, "device", None)
    if isinstance(module_device, torch.device):
        return module_device
    if module_device is not None:
        return torch.device(module_device)
    return None


def _resolve_embedding_attachment_device(
    embeddings_device: str | None,
    *,
    trainer: Any,
) -> torch.device | None:
    root_device = _resolve_trainer_root_device(trainer)
    if embeddings_device == _EMBEDDINGS_DEVICE_CPU:
        return torch.device("cpu")
    if embeddings_device == _EMBEDDINGS_DEVICE_CUDA:
        if root_device is not None and root_device.type == "cuda":
            return root_device
        return torch.device("cuda")
    if root_device is not None and root_device.type == "cuda":
        return root_device
    return None
