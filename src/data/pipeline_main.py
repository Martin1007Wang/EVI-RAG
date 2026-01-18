from __future__ import annotations

import uuid

from src.data.context import StageContext
from src.data.io.lmdb_utils import ensure_dir
from src.data.stages.step2_graph import preprocess
from src.data.stages.step3_lmdb import build_dataset
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)


def _validate_pipeline_cfg(ctx: StageContext) -> None:
    cfg = ctx.cfg
    precompute_embeddings = bool(cfg.get("precompute_entities", False)) or bool(cfg.get("precompute_relations", False))
    precompute_questions = bool(cfg.get("precompute_questions", False))
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    require_precomputed_questions = bool(cfg.get("require_precomputed_questions", False))
    skip_parquet_stage = bool(cfg.get("skip_parquet_stage", False))
    skip_lmdb_stage = bool(cfg.get("skip_lmdb_stage", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    _ = ctx.parquet_chunk_size
    _ = ctx.parquet_num_workers

    if skip_parquet_stage or skip_lmdb_stage:
        raise ValueError("skip_parquet_stage/skip_lmdb_stage are disabled; run the unified parquet+LMDB pipeline.")

    if use_precomputed_embeddings and not precompute_embeddings and not skip_parquet_stage and not reuse_embeddings_if_exists:
        raise ValueError(
            "use_precomputed_embeddings=true requires precompute_entities or precompute_relations "
            "to be enabled in the same pipeline run."
        )
    if use_precomputed_questions and not precompute_questions and not skip_parquet_stage:
        raise ValueError(
            "use_precomputed_questions=true requires precompute_questions to be enabled in the same pipeline run."
        )
    if precompute_questions and not require_precomputed_questions:
        raise ValueError(
            "precompute_questions=true requires require_precomputed_questions=true "
            "to ensure LMDB uses the freshly computed embeddings."
        )

    parquet_dir_cfg = cfg.get("parquet_dir")
    out_dir_cfg = cfg.get("out_dir")
    if parquet_dir_cfg and out_dir_cfg:
        if ctx.parquet_dir.resolve() != ctx.out_dir.resolve():
            raise ValueError(
                "parquet_dir must match out_dir in the unified pipeline. "
                f"Got parquet_dir={ctx.parquet_dir} vs out_dir={ctx.out_dir}."
            )

def _run_parquet_stage(ctx: StageContext) -> None:
    log_event(
        ctx.logger,
        "parquet_stage_start",
        dataset=ctx.dataset_name,
        out_dir=str(ctx.out_dir),
    )
    preprocess(ctx)
    log_event(ctx.logger, "parquet_stage_done", out_dir=str(ctx.out_dir))


def _ensure_pipeline_dirs(ctx: StageContext) -> None:
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


def build_pipeline(cfg) -> None:
    run_id = str(cfg.get("run_id") or uuid.uuid4().hex)
    ctx = StageContext(cfg=cfg, logger=LOGGER, run_id=run_id)
    _validate_pipeline_cfg(ctx)
    _ensure_pipeline_dirs(ctx)
    _run_parquet_stage(ctx)
    build_dataset(ctx)
