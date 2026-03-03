from __future__ import annotations

import uuid
from typing import Dict

from src.data.preprocess.context import PreprocessContext
from src.data.io.lmdb_utils import ensure_dir
from src.data.preprocess.stages.step2_graph import preprocess
from src.data.preprocess.stages.step3_lmdb import build_dataset
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)


def _validate_preprocess_cfg(ctx: PreprocessContext) -> None:
    cfg = ctx.cfg
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    skip_parquet_stage = bool(cfg.get("skip_parquet_stage", False))
    skip_lmdb_stage = bool(cfg.get("skip_lmdb_stage", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    _ = ctx.parquet_chunk_size
    _ = ctx.parquet_num_workers

    if skip_parquet_stage and skip_lmdb_stage:
        raise ValueError(
            "Both skip_parquet_stage and skip_lmdb_stage are true; nothing to run."
        )

    _ = use_precomputed_embeddings
    _ = use_precomputed_questions
    parquet_dir_cfg = cfg.get("parquet_dir")
    out_dir_cfg = cfg.get("out_dir")
    if parquet_dir_cfg and out_dir_cfg:
        if (
            not skip_parquet_stage
            and not skip_lmdb_stage
            and ctx.parquet_dir.resolve() != ctx.out_dir.resolve()
        ):
            raise ValueError(
                "parquet_dir must match out_dir in the unified pipeline. "
                f"Got parquet_dir={ctx.parquet_dir} vs out_dir={ctx.out_dir}."
            )


def _apply_preprocess_stage(ctx: PreprocessContext) -> str:
    cfg = ctx.cfg
    stage = str(cfg.get("pipeline_stage", "all")).strip().lower()
    if not stage:
        stage = "all"
    if stage not in ("all", "parquet", "lmdb"):
        raise ValueError(
            f"pipeline_stage must be one of: all, parquet, lmdb (got {stage!r})."
        )
    desired = {
        "skip_parquet_stage": stage == "lmdb",
        "skip_lmdb_stage": stage == "parquet",
    }
    changed: Dict[str, object] = {}
    for key, value in desired.items():
        if cfg.get(key) != value:
            cfg[key] = value
            changed[key] = value
    if changed:
        log_event(ctx.logger, "pipeline_stage_applied", stage=stage, overrides=changed)
    return stage


def _apply_preprocess_overrides(ctx: PreprocessContext) -> None:
    cfg = ctx.cfg
    if not bool(cfg.get("skip_lmdb_stage", False)):
        return
    overrides = {
        "canonicalize_relations": False,
        "use_precomputed_embeddings": False,
        "use_precomputed_questions": False,
    }
    changed: Dict[str, object] = {}
    for key, value in overrides.items():
        if cfg.get(key) != value:
            cfg[key] = value
            changed[key] = value
    if changed:
        log_event(
            ctx.logger,
            "pipeline_overrides",
            reason="skip_lmdb_stage",
            overrides=changed,
        )


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
    stage = _apply_preprocess_stage(ctx)
    _validate_preprocess_cfg(ctx)
    _apply_preprocess_overrides(ctx)
    _ensure_preprocess_dirs(ctx)
    if not bool(cfg.get("skip_parquet_stage", False)):
        _run_parquet_stage(ctx)
    if not bool(cfg.get("skip_lmdb_stage", False)):
        build_dataset(ctx)
