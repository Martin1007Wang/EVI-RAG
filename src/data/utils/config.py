from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from src.data.schema.constants import (
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_COSINE_EPS,
    _DISABLE_PARALLEL_WORKERS,
    _MIN_CHUNK_SIZE,
)
from src.data.schema.types import EmbeddingConfig, SplitFilter


def _resolve_parquet_chunk_size(cfg, *, fallback: int) -> int:
    chunk_cfg = cfg.get("parquet_chunk_size")
    chunk_size = fallback if chunk_cfg is None else int(chunk_cfg)
    if chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"parquet_chunk_size must be >= {_MIN_CHUNK_SIZE}, got {chunk_size}")
    return chunk_size


def _resolve_parquet_num_workers(cfg) -> int:
    workers_cfg = cfg.get("parquet_num_workers", _DISABLE_PARALLEL_WORKERS)
    num_workers = int(workers_cfg)
    if num_workers < _DISABLE_PARALLEL_WORKERS:
        raise ValueError(f"parquet_num_workers must be >= {_DISABLE_PARALLEL_WORKERS}, got {num_workers}")
    return num_workers


def build_embedding_cfg(cfg) -> Optional[EmbeddingConfig]:
    embed_flags = {
        "precompute_entities": bool(cfg.get("precompute_entities", False)),
        "precompute_relations": bool(cfg.get("precompute_relations", False)),
        "precompute_questions": bool(cfg.get("precompute_questions", False)),
        "canonicalize_relations": bool(cfg.get("canonicalize_relations", False)),
    }
    if not any(embed_flags.values()):
        return None
    embeddings_out_dir_cfg = cfg.get("embeddings_out_dir")
    if not embeddings_out_dir_cfg:
        raise ValueError("embeddings_out_dir must be set when embedding precompute is enabled.")
    try:
        import hydra
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("hydra-core is required to resolve embeddings_out_dir.") from exc
    return EmbeddingConfig(
        encoder=str(cfg.get("encoder", "")),
        device=str(cfg.get("device", "cuda")),
        batch_size=int(cfg.get("batch_size", _DEFAULT_BATCH_SIZE)),
        fp16=bool(cfg.get("fp16", False)),
        progress_bar=bool(cfg.get("progress_bar", True)),
        embeddings_out_dir=Path(hydra.utils.to_absolute_path(embeddings_out_dir_cfg)),
        precompute_entities=embed_flags["precompute_entities"],
        precompute_relations=embed_flags["precompute_relations"],
        precompute_questions=embed_flags["precompute_questions"],
        canonicalize_relations=embed_flags["canonicalize_relations"],
        cosine_eps=float(cfg.get("cosine_eps", _DEFAULT_COSINE_EPS)),
    )


def build_split_filters(cfg) -> Tuple[SplitFilter, SplitFilter, dict[str, SplitFilter]]:
    default_filter = SplitFilter(skip_no_topic=False, skip_no_ans=False, skip_no_path=False)
    filter_cfg = cfg.get("filter")
    if filter_cfg is None:
        return default_filter, default_filter, {}
    train_section = filter_cfg.get("train")
    eval_section = filter_cfg.get("eval")
    train_filter = SplitFilter(
        skip_no_topic=bool(train_section.get("skip_no_topic", False)) if train_section is not None else False,
        skip_no_ans=bool(train_section.get("skip_no_ans", False)) if train_section is not None else False,
        skip_no_path=bool(train_section.get("skip_no_path", False)) if train_section is not None else False,
    )
    eval_filter = SplitFilter(
        skip_no_topic=bool(eval_section.get("skip_no_topic", False)) if eval_section is not None else False,
        skip_no_ans=bool(eval_section.get("skip_no_ans", False)) if eval_section is not None else False,
        skip_no_path=bool(eval_section.get("skip_no_path", False)) if eval_section is not None else False,
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
