from __future__ import annotations

import functools
import random
from typing import Any, Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but optional
    np = None

import torch
from torch.utils.data import DataLoader

from .collate import BatchAugmenter, RetrievalCollater
from ..g_retrieval_dataset import GRetrievalDataset
from ...utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_NUMPY_SEED_MOD = 2**32 - 1
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_NUM_WORKERS = 0
_ZERO = 0


def _init_worker_seed(
    worker_id: int,
    *,
    base_seed: Optional[int],
    user_init_fn: Optional[Callable[[int], None]],
) -> None:
    if base_seed is not None:
        worker_seed = int(base_seed) + int(worker_id)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(worker_seed)
        random.seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed % _NUMPY_SEED_MOD)
    if user_init_fn is not None:
        user_init_fn(worker_id)


def build_retrieval_dataloader(
    dataset: GRetrievalDataset,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    random_seed: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    precompute_edge_batch: bool = True,
    validate_edge_batch: bool = False,
    follow_batch: Optional[list[str]] = None,
    exclude_keys: Optional[list[str]] = None,
    expand_multi_start: bool = True,
    expand_multi_answer: bool = True,
    **kwargs: Any,
) -> DataLoader:
    if dataset is None:
        raise RuntimeError("Dataset not initialized. Did you run setup()?")

    if num_workers == _ZERO:
        persistent_workers = False

    augmenter = BatchAugmenter(
        precompute_edge_batch=precompute_edge_batch,
        validate_edge_batch=validate_edge_batch,
    )
    collate_fn = RetrievalCollater(
        dataset,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
        augmenter=augmenter,
        expand_multi_start=expand_multi_start,
        expand_multi_answer=expand_multi_answer,
    )

    user_init_fn: Optional[Callable[[int], None]] = kwargs.pop("worker_init_fn", None)
    base_seed = int(random_seed) if random_seed is not None else None
    kwargs["worker_init_fn"] = functools.partial(
        _init_worker_seed,
        base_seed=base_seed,
        user_init_fn=user_init_fn,
    )
    if base_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(base_seed)
        kwargs.setdefault("generator", generator)
    if num_workers > _ZERO and "multiprocessing_context" not in kwargs:
        if torch.cuda.is_available():
            kwargs["multiprocessing_context"] = "spawn"
    if prefetch_factor is not None and num_workers > _ZERO:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **kwargs,
    )
    log_event(
        logger,
        "retrieval_dataloader_init",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader
