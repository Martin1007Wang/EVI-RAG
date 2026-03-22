from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency guard
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from src.data.schema.constants import (
    _DEFAULT_BATCH_SIZE,
    _DISABLE_PARALLEL_WORKERS,
    _MIN_CHUNK_SIZE,
)
from src.data.schema.types import EmbeddingConfig, SplitFilter

_AUTO_EMBEDDING_DEVICE = "auto"
_DEFAULT_GPU_BATCH_SIZE = 256


def resolve_embedding_device(raw_device: Any) -> str:
    device = str(raw_device or _AUTO_EMBEDDING_DEVICE).strip().lower()
    if device in {"", _AUTO_EMBEDDING_DEVICE}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        device = "cuda"
    if device == "cpu":
        return device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available.")
        return device
    raise ValueError(
        "device must be one of {'auto', 'cpu', 'cuda', 'cuda:N', 'gpu'}, "
        f"got {raw_device!r}."
    )


def resolve_embedding_batch_size(cfg: Any, *, device: str | None = None) -> int:
    device = resolve_embedding_device(cfg.get("device")) if device is None else device
    raw_batch_size = cfg.get("batch_size")
    if raw_batch_size in (None, ""):
        batch_size = (
            _DEFAULT_GPU_BATCH_SIZE
            if device.startswith("cuda")
            else _DEFAULT_BATCH_SIZE
        )
    else:
        batch_size = int(raw_batch_size)
    if batch_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"batch_size must be >= {_MIN_CHUNK_SIZE}, got {batch_size}.")
    return batch_size


def resolve_embedding_fp16(cfg: Any, *, device: str | None = None) -> bool:
    device = resolve_embedding_device(cfg.get("device")) if device is None else device
    raw_fp16 = cfg.get("fp16")
    if raw_fp16 in (None, ""):
        return device.startswith("cuda")
    return bool(raw_fp16)


def _resolve_parquet_chunk_size(cfg, *, fallback: int) -> int:
    chunk_cfg = cfg.get("parquet_chunk_size")
    chunk_size = fallback if chunk_cfg is None else int(chunk_cfg)
    if chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(
            f"parquet_chunk_size must be >= {_MIN_CHUNK_SIZE}, got {chunk_size}"
        )
    return chunk_size


def _resolve_parquet_num_workers(cfg) -> int:
    workers_cfg = cfg.get("parquet_num_workers", _DISABLE_PARALLEL_WORKERS)
    num_workers = int(workers_cfg)
    if num_workers < _DISABLE_PARALLEL_WORKERS:
        raise ValueError(
            f"parquet_num_workers must be >= {_DISABLE_PARALLEL_WORKERS}, got {num_workers}"
        )
    return num_workers


def build_embedding_cfg(cfg) -> Optional[EmbeddingConfig]:
    encoder = str(cfg.get("encoder", "")).strip()
    if not encoder:
        return None
    question_ctx_max_tokens = int(cfg.get("question_ctx_max_tokens", 0))
    if question_ctx_max_tokens < 0:
        raise ValueError(
            f"question_ctx_max_tokens must be >= 0, got {question_ctx_max_tokens}."
        )
    embeddings_out_dir_cfg = cfg.get("embeddings_out_dir")
    if not embeddings_out_dir_cfg:
        raise ValueError(
            "embeddings_out_dir must be set when embedding encoding is enabled."
        )
    try:
        import hydra
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "hydra-core is required to resolve embeddings_out_dir."
        ) from exc
    device = resolve_embedding_device(cfg.get("device"))
    return EmbeddingConfig(
        encoder=encoder,
        device=device,
        batch_size=resolve_embedding_batch_size(cfg, device=device),
        fp16=resolve_embedding_fp16(cfg, device=device),
        progress_bar=bool(cfg.get("progress_bar", True)),
        embeddings_out_dir=Path(hydra.utils.to_absolute_path(embeddings_out_dir_cfg)),
        question_ctx_max_tokens=question_ctx_max_tokens,
    )


def build_preprocess_filters(
    cfg,
) -> Tuple[SplitFilter, SplitFilter, dict[str, SplitFilter]]:
    default_filter = SplitFilter(
        skip_no_topic=False, skip_no_ans=False, skip_no_path=False
    )
    filter_cfg = cfg.get("preprocess_filter")
    if filter_cfg is None:
        return default_filter, default_filter, {}
    train_section = filter_cfg.get("train")
    eval_section = filter_cfg.get("eval")
    train_filter = SplitFilter(
        skip_no_topic=bool(train_section.get("skip_no_topic", False))
        if train_section is not None
        else False,
        skip_no_ans=bool(train_section.get("skip_no_ans", False))
        if train_section is not None
        else False,
        skip_no_path=bool(train_section.get("skip_no_path", False))
        if train_section is not None
        else False,
    )
    eval_filter = SplitFilter(
        skip_no_topic=bool(eval_section.get("skip_no_topic", False))
        if eval_section is not None
        else False,
        skip_no_ans=bool(eval_section.get("skip_no_ans", False))
        if eval_section is not None
        else False,
        skip_no_path=bool(eval_section.get("skip_no_path", False))
        if eval_section is not None
        else False,
    )
    overrides = {}
    for key in filter_cfg.keys():
        if key in {"train", "eval"}:
            continue
        section = filter_cfg.get(key)
        overrides[str(key)] = SplitFilter(
            skip_no_topic=bool(section.get("skip_no_topic", False)),
            skip_no_ans=bool(section.get("skip_no_ans", False)),
            skip_no_path=bool(section.get("skip_no_path", False)),
        )
    return train_filter, eval_filter, overrides


__all__ = [
    "_resolve_parquet_chunk_size",
    "_resolve_parquet_num_workers",
    "build_embedding_cfg",
    "build_preprocess_filters",
    "resolve_embedding_batch_size",
    "resolve_embedding_device",
    "resolve_embedding_fp16",
]
