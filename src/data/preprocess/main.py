from __future__ import annotations

import uuid
from typing import Iterable

from src.data.preprocess.context import PreprocessContext
from src.data.io.lmdb_utils import ensure_dir
from src.data.preprocess.stages.step2_graph import preprocess
from src.data.preprocess.stages.step3_lmdb import build_dataset
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)

_REMOVED_PREPROCESS_KEYS = (
    "skip_parquet_stage",
    "skip_lmdb_stage",
    "filter",
    "keep_start_adjacent_edges",
    "canonicalize_relations",
    "cosine_eps",
    "emit_nonzero_positive_filter",
    "nonzero_positive_filter_filename",
    "nonzero_positive_filter_splits",
)
_REMOVED_DATASET_PREPROCESS_KEYS = (
    "time_relation_mode",
    "time_relation_regex",
    "time_question_regex",
)


def _assert_no_removed_keys(
    cfg_section, *, keys: Iterable[str], section_name: str
) -> None:
    if cfg_section is None or not hasattr(cfg_section, "get"):
        return
    removed = [key for key in keys if cfg_section.get(key) not in (None, "")]
    if removed:
        raise ValueError(
            f"Removed {section_name} config keys detected: {removed}. "
            "Delete these overrides and rerun preprocess."
        )


def _validate_preprocess_cfg(ctx: PreprocessContext) -> None:
    cfg = ctx.cfg

    _ = ctx.parquet_chunk_size
    _ = ctx.parquet_num_workers

    if cfg.get("parquet_dir") not in (None, ""):
        raise ValueError("`parquet_dir` was removed; use `out_dir` instead.")
    if cfg.get("dataset_name") not in (None, ""):
        raise ValueError("`dataset_name` was removed; use `dataset.name` instead.")

    _assert_no_removed_keys(
        cfg,
        keys=_REMOVED_PREPROCESS_KEYS,
        section_name="preprocess",
    )
    _assert_no_removed_keys(
        cfg.get("dataset"),
        keys=_REMOVED_DATASET_PREPROCESS_KEYS,
        section_name="dataset preprocess",
    )


def _resolve_preprocess_stage(cfg) -> str:
    stage = str(cfg.get("pipeline_stage", "all")).strip().lower()
    if not stage:
        stage = "all"
    if stage not in ("all", "parquet", "lmdb"):
        raise ValueError(
            f"pipeline_stage must be one of: all, parquet, lmdb (got {stage!r})."
        )
    return stage


def _run_parquet_stage(ctx: PreprocessContext) -> None:
    log_event(
        ctx.logger,
        "parquet_stage_start",
        dataset=ctx.dataset_name,
        out_dir=str(ctx.out_dir),
    )
    preprocess(ctx)
    log_event(ctx.logger, "parquet_stage_done", out_dir=str(ctx.out_dir))


def _ensure_preprocess_dirs(ctx: PreprocessContext) -> None:
    out_dir = ctx.out_dir
    output_dir = ctx.output_dir
    embeddings_dir = ctx.embeddings_dir
    for path in (out_dir, output_dir, embeddings_dir):
        ensure_dir(path)
    log_event(
        ctx.logger,
        "pipeline_dirs_ready",
        out_dir=str(out_dir),
        output_dir=str(output_dir),
        embeddings_dir=str(embeddings_dir),
    )


def run_preprocess_pipeline(cfg) -> None:
    run_id = str(cfg.get("run_id") or uuid.uuid4().hex)
    ctx = PreprocessContext(cfg=cfg, logger=LOGGER, run_id=run_id)
    stage = _resolve_preprocess_stage(cfg)
    _validate_preprocess_cfg(ctx)
    _ensure_preprocess_dirs(ctx)
    if stage in {"all", "parquet"}:
        _run_parquet_stage(ctx)
    if stage in {"all", "lmdb"}:
        build_dataset(ctx)
