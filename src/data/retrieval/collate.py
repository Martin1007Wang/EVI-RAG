from __future__ import annotations

import atexit
import faulthandler
import os
from functools import partial
from pathlib import Path
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Sampler
from torch_geometric.loader.dataloader import Collater

from .dataset import GraphRetrievalDataset
from src.graph.batch import compute_edge_batch_and_ptr
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_WORKER_DIAG_DIR_ENV = "RETRIEVAL_WORKER_DIAG_DIR"


def _elapsed_seconds(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 3)


class _LoggedDataLoaderIterator:
    def __init__(self, iterator: Any, *, loader_name: str) -> None:
        self._iterator = iterator
        self._loader_name = loader_name
        self._first_batch_pending = True

    def __iter__(self) -> "_LoggedDataLoaderIterator":
        return self

    def __next__(self) -> Any:
        if not self._first_batch_pending:
            return next(self._iterator)
        start_time = time.perf_counter()
        batch = next(self._iterator)
        self._first_batch_pending = False
        log_event(
            logger,
            "retrieval_dataloader_first_batch_ready",
            loader_name=self._loader_name,
            elapsed_s=_elapsed_seconds(start_time),
        )
        return batch


class _InstrumentedDataLoader(DataLoader):
    def __init__(
        self,
        *args: Any,
        loader_name: str,
        multiprocessing_context_name: str | None,
        **kwargs: Any,
    ) -> None:
        self._loader_name = str(loader_name)
        self._multiprocessing_context_name = multiprocessing_context_name
        super().__init__(*args, **kwargs)

    def __iter__(self) -> _LoggedDataLoaderIterator:
        log_event(
            logger,
            "retrieval_dataloader_iter_start",
            loader_name=self._loader_name,
            num_workers=int(self.num_workers),
            multiprocessing_context=self._multiprocessing_context_name,
        )
        start_time = time.perf_counter()
        iterator = super().__iter__()
        log_event(
            logger,
            "retrieval_dataloader_iter_ready",
            loader_name=self._loader_name,
            num_workers=int(self.num_workers),
            multiprocessing_context=self._multiprocessing_context_name,
            elapsed_s=_elapsed_seconds(start_time),
        )
        return _LoggedDataLoaderIterator(iterator, loader_name=self._loader_name)


def build_retrieval_dataloader(
    dataset: GraphRetrievalDataset,
    *,
    loader_name: str = "loader",
    batch_size: int,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
    batch_sampler: Sampler[list[int]] | None = None,
    drop_last: bool,
    num_workers: int,
    random_seed: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    multiprocessing_context: str | None = None,
    pin_memory: bool = True,
    precompute_edge_batch: bool = True,
    follow_batch: Optional[list[str]] = None,
    exclude_keys: Optional[list[str]] = None,
    expand_multi_answer: bool = True,
    filter_zero_hop: bool = True,
    **kwargs: Any,
) -> DataLoader:
    if dataset is None:
        raise RuntimeError("Dataset not initialized. Did you run setup()?")

    if num_workers == 0:
        persistent_workers = False
        multiprocessing_context = None

    augmenter = BatchAugmenter(
        precompute_edge_batch=precompute_edge_batch,
    )
    collate_fn = RetrievalCollater(
        dataset,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
        augmenter=augmenter,
        expand_multi_answer=expand_multi_answer,
        filter_zero_hop=filter_zero_hop,
    )

    if random_seed is not None and "generator" not in kwargs:
        generator = torch.Generator()
        generator.manual_seed(int(random_seed))
        kwargs["generator"] = generator
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if multiprocessing_context is not None and num_workers > 0:
        kwargs["multiprocessing_context"] = str(multiprocessing_context)
    if sampler is not None:
        shuffle = False
        kwargs["sampler"] = sampler
    if batch_sampler is not None:
        shuffle = False
        kwargs.pop("sampler", None)
        kwargs["batch_sampler"] = batch_sampler
    worker_init_fn = _build_worker_diagnostics_init_fn(loader_name=loader_name)
    if worker_init_fn is not None:
        kwargs["worker_init_fn"] = worker_init_fn

    loader_kwargs = dict(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **kwargs,
    )
    if batch_sampler is None:
        loader_kwargs.update(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    loader = _InstrumentedDataLoader(
        **loader_kwargs,
        loader_name=loader_name,
        multiprocessing_context_name=multiprocessing_context,
    )
    log_event(
        logger,
        "retrieval_dataloader_init",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
    )
    return loader


def _build_worker_diagnostics_init_fn(*, loader_name: str):
    diag_dir = os.environ.get(_WORKER_DIAG_DIR_ENV)
    if diag_dir in (None, ""):
        return None
    base_dir = Path(diag_dir).expanduser()
    return partial(
        _init_worker_diagnostics,
        loader_name=loader_name,
        diag_dir=str(base_dir),
    )


def _init_worker_diagnostics(
    worker_id: int,
    *,
    loader_name: str,
    diag_dir: str,
) -> None:
    base_dir = Path(diag_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    log_path = base_dir / f"{loader_name}.worker{worker_id}.pid{pid}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    faulthandler.enable(file=log_file, all_threads=True)
    log_file.write(
        f"worker_start pid={pid} ppid={os.getppid()} "
        f"loader={loader_name} worker_id={worker_id}\n"
    )
    log_file.flush()

    def _on_exit() -> None:
        log_file.write(f"worker_exit pid={pid} loader={loader_name}\n")
        log_file.flush()
        log_file.close()

    atexit.register(_on_exit)


def _as_1d_long(value: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        return tensor.view(-1)
    tensor = value
    target_device = tensor.device if device is None else device
    if tensor.device != target_device:
        tensor = tensor.to(device=target_device)
    if tensor.dtype != torch.long:
        tensor = tensor.to(dtype=torch.long)
    return tensor.view(-1)


def _require_long_tensor(data: Any, name: str) -> torch.Tensor:
    value = getattr(data, name, None)
    if value is None:
        raise AttributeError(f"Batch missing {name} required for expansion.")
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value, dtype=torch.long)


def _maybe_long_tensor(data: Any, name: str) -> Optional[torch.Tensor]:
    value = getattr(data, name, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value, dtype=torch.long)


def _resolve_answer_values(
    *,
    a_vals: torch.Tensor,
    answer_vals: torch.Tensor,
    node_entity_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    if a_vals.numel() == answer_vals.numel():
        return answer_vals
    if node_entity_ids is None:
        raise AttributeError("Batch missing node_entity_ids required to align answers.")
    if a_vals.numel() == 0:
        return a_vals.new_empty((0,))
    return node_entity_ids.view(-1).index_select(0, a_vals)


def _iter_answer_candidates(
    a_vals: torch.Tensor,
    answer_vals: torch.Tensor,
    *,
    expand_multi_answer: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, int | None]]:
    if not expand_multi_answer or a_vals.numel() <= 1:
        return [(a_vals, answer_vals, None)]
    return [
        (a_vals[idx].view(1), answer_vals[idx].view(1), idx)
        for idx in range(a_vals.numel())
    ]


def _filter_zero_hop_answers(
    anchor_vals: torch.Tensor,
    a_vals: torch.Tensor,
    answer_vals: torch.Tensor,
    *,
    filter_zero_hop: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not filter_zero_hop or anchor_vals.numel() == 0 or a_vals.numel() == 0:
        return a_vals, answer_vals
    keep_mask = ~(anchor_vals.view(-1, 1) == a_vals.view(1, -1)).any(dim=0)
    if bool(keep_mask.all().item()):
        return a_vals, answer_vals
    return a_vals[keep_mask], answer_vals[keep_mask]


def _should_skip_zero_hop(
    anchor_vals: torch.Tensor,
    a_val: torch.Tensor,
    *,
    filter_zero_hop: bool,
) -> bool:
    if not filter_zero_hop or anchor_vals.numel() == 0 or a_val.numel() == 0:
        return False
    return bool((anchor_vals.view(-1, 1) == a_val.view(1, -1)).any().item())


def _expand_answer_samples(
    batch_list: list[Any],
    *,
    expand_multi_answer: bool,
    filter_zero_hop: bool,
) -> list[Any]:
    if not expand_multi_answer and not filter_zero_hop:
        return batch_list
    expanded: list[Any] = []
    for data in batch_list:
        anchor_local = _require_long_tensor(data, "anchor_local_indices")
        a_local = _require_long_tensor(data, "a_local_indices")
        answer_ids = _require_long_tensor(data, "answer_entity_ids")
        node_entity_ids = _maybe_long_tensor(data, "node_entity_ids")
        anchor_vals = anchor_local.view(-1)
        a_vals = a_local.view(-1)
        answer_vals = _resolve_answer_values(
            a_vals=a_vals,
            answer_vals=answer_ids.view(-1),
            node_entity_ids=node_entity_ids,
        )
        if not expand_multi_answer:
            a_vals, answer_vals = _filter_zero_hop_answers(
                anchor_vals,
                a_vals,
                answer_vals,
                filter_zero_hop=filter_zero_hop,
            )
            if a_vals.numel() == 0:
                continue
        a_candidates = _iter_answer_candidates(
            a_vals,
            answer_vals,
            expand_multi_answer=expand_multi_answer,
        )
        base_id = str(getattr(data, "sample_id", ""))
        for a_val, ans_val, a_idx in a_candidates:
            if _should_skip_zero_hop(
                anchor_vals, a_val, filter_zero_hop=filter_zero_hop
            ):
                continue
            clone = data.clone()
            clone.a_local_indices = a_val
            clone.answer_entity_ids = ans_val
            if base_id and a_idx is not None:
                clone.sample_id = f"{base_id}::a{a_idx}"
            expanded.append(clone)
    if filter_zero_hop and not expanded:
        raise ValueError(
            "All samples filtered by zero-hop guard; disable filter_zero_hop to proceed."
        )
    return expanded


def _attach_answer_ids(batch: Any) -> None:
    if not hasattr(batch, "answer_entity_ids"):
        raise AttributeError("Batch missing answer_entity_ids required for metrics.")
    batch.answer_entity_ids = _as_1d_long(
        batch.answer_entity_ids, device=torch.device("cpu")
    )
    answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
    if answer_ptr is None and hasattr(batch, "_slice_dict"):
        answer_ptr = batch._slice_dict.get("answer_entity_ids")
    if answer_ptr is None:
        raise AttributeError(
            "Batch missing answer_entity_ids_ptr; PyG collate may have failed."
        )
    batch.answer_entity_ids_ptr = _as_1d_long(answer_ptr, device=torch.device("cpu"))
    batch.answer_ptr = batch.answer_entity_ids_ptr
    answer_counts = batch.answer_entity_ids_ptr[1:] - batch.answer_entity_ids_ptr[:-1]
    batch.num_valid_graphs = int((answer_counts > 0).sum().item())
    batch.dummy_mask = answer_counts <= 0


def _attach_qa_ptrs(batch: Any) -> None:
    slice_dict = getattr(batch, "_slice_dict", None)
    if not isinstance(slice_dict, dict):
        raise AttributeError("Batch missing _slice_dict required for anchor_ptr/a_ptr.")
    anchor_ptr = slice_dict.get("anchor_local_indices")
    a_ptr = slice_dict.get("a_local_indices")
    if anchor_ptr is None or a_ptr is None:
        raise AttributeError(
            "Batch _slice_dict missing anchor_local_indices/a_local_indices pointers."
        )
    batch.anchor_ptr = _as_1d_long(anchor_ptr, device=torch.device("cpu"))
    batch.a_ptr = _as_1d_long(a_ptr, device=torch.device("cpu"))


def _attach_local_indices(batch: Any) -> None:
    if not hasattr(batch, "anchor_local_indices"):
        raise AttributeError(
            "Batch missing anchor_local_indices required for graph alignment."
        )
    if not hasattr(batch, "a_local_indices"):
        raise AttributeError(
            "Batch missing a_local_indices required for graph alignment."
        )
    batch.anchor_local_indices = _as_1d_long(
        batch.anchor_local_indices, device=torch.device("cpu")
    )
    batch.a_local_indices = _as_1d_long(
        batch.a_local_indices, device=torch.device("cpu")
    )


def _attach_graph_stats(batch: Any) -> None:
    node_ptr = getattr(batch, "ptr", None)
    if node_ptr is None:
        raise AttributeError("Batch missing ptr; cannot infer graph counts.")
    node_ptr = _as_1d_long(node_ptr, device=torch.device("cpu"))
    num_graphs = int(node_ptr.numel() - 1)
    num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
    batch.num_graphs = num_graphs
    batch.num_nodes_total = num_nodes_total
    batch.node_ptr = node_ptr


def _attach_edge_batch(batch: Any) -> None:
    edge_index = getattr(batch, "edge_index", None)
    node_ptr = getattr(batch, "ptr", None)
    if edge_index is None or node_ptr is None:
        raise AttributeError(
            "Batch missing edge_index/ptr; cannot precompute edge_batch."
        )
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    elif edge_index.dtype != torch.long:
        edge_index = edge_index.to(dtype=torch.long)
    node_ptr = _as_1d_long(node_ptr, device=edge_index.device)
    num_graphs = int(node_ptr.numel() - 1)
    if num_graphs <= 0:
        raise ValueError(
            "ptr must encode at least one graph when precomputing edge_batch."
        )
    edge_batch, edge_ptr = compute_edge_batch_and_ptr(
        edge_index,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
        device=edge_index.device,
        validate=False,
    )
    batch.edge_batch = edge_batch
    batch.edge_ptr = edge_ptr


def _validate_ptrs(batch: Any) -> None:
    num_graphs = getattr(batch, "num_graphs", None)
    if not isinstance(num_graphs, int) or num_graphs <= 0:
        raise ValueError("Batch missing valid num_graphs; cannot validate ptrs.")
    anchor_ptr = getattr(batch, "anchor_ptr", None)
    a_ptr = getattr(batch, "a_ptr", None)
    answer_ptr = getattr(batch, "answer_ptr", None)
    if anchor_ptr is None or a_ptr is None or answer_ptr is None:
        raise AttributeError(
            "Batch missing anchor_ptr/a_ptr/answer_ptr required for validation."
        )
    if anchor_ptr.numel() != num_graphs + 1:
        raise ValueError("anchor_ptr length mismatch with num_graphs.")
    if a_ptr.numel() != num_graphs + 1:
        raise ValueError("a_ptr length mismatch with num_graphs.")
    if answer_ptr.numel() != num_graphs + 1:
        raise ValueError("answer_ptr length mismatch with num_graphs.")
    if int(anchor_ptr[-1].item()) != int(batch.anchor_local_indices.numel()):
        raise ValueError("anchor_ptr[-1] mismatch anchor_local_indices length.")
    if int(a_ptr[-1].item()) != int(batch.a_local_indices.numel()):
        raise ValueError("a_ptr[-1] mismatch a_local_indices length.")
    if int(answer_ptr[-1].item()) != int(batch.answer_entity_ids.numel()):
        raise ValueError("answer_ptr[-1] mismatch answer_entity_ids length.")
    if not hasattr(batch, "dummy_mask"):
        raise AttributeError("Batch missing dummy_mask derived from answer_ptr.")
    if batch.dummy_mask.numel() != num_graphs:
        raise ValueError("dummy_mask length mismatch with num_graphs.")


class BatchAugmenter:
    """Attach derived fields to a PyG batch."""

    def __init__(
        self,
        *,
        precompute_edge_batch: bool,
    ) -> None:
        self._precompute_edge_batch = bool(precompute_edge_batch)

    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, list):
            raise TypeError(
                "RetrievalCollater received a list batch; dataset must return GraphData."
            )
        _attach_graph_stats(batch)
        _attach_local_indices(batch)
        _attach_qa_ptrs(batch)
        _attach_answer_ids(batch)
        if self._precompute_edge_batch:
            _attach_edge_batch(batch)
        _validate_ptrs(batch)
        return batch


class RetrievalCollater:
    """Collate PyG graphs and apply optional batch augmentation."""

    def __init__(
        self,
        dataset: Any,
        *,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        augmenter: Optional[BatchAugmenter] = None,
        expand_multi_answer: bool = False,
        filter_zero_hop: bool = True,
    ) -> None:
        self._augmenter = augmenter
        self._expand_multi_answer = bool(expand_multi_answer)
        self._filter_zero_hop = bool(filter_zero_hop)
        self._collater = Collater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> Any:
        if self._expand_multi_answer or self._filter_zero_hop:
            batch_list = _expand_answer_samples(
                batch_list,
                expand_multi_answer=self._expand_multi_answer,
                filter_zero_hop=self._filter_zero_hop,
            )
        batch = self._collater(batch_list)
        if self._augmenter is None:
            return batch
        return self._augmenter(batch)


__all__ = ["BatchAugmenter", "RetrievalCollater", "build_retrieval_dataloader"]
