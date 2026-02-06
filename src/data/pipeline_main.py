from __future__ import annotations

import uuid
from typing import Dict

from src.data.context import StageContext
from src.data.io.lmdb_utils import ensure_dir
from src.data.stages.inverse_relations_stage import (
    build_inverse_relations_all,
    build_inverse_relations_describe,
    build_inverse_relations_detect,
    build_inverse_relations_resolve,
)
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
    skip_parquet_stage = bool(cfg.get("skip_parquet_stage", False))
    skip_lmdb_stage = bool(cfg.get("skip_lmdb_stage", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    _ = ctx.parquet_chunk_size
    _ = ctx.parquet_num_workers

    if skip_parquet_stage and skip_lmdb_stage:
        stage = str(cfg.get("pipeline_stage", "all")).strip().lower()
        if not stage.startswith("inverse"):
            raise ValueError("Both skip_parquet_stage and skip_lmdb_stage are true; nothing to run.")

    if use_precomputed_embeddings and not precompute_embeddings and not skip_parquet_stage and not reuse_embeddings_if_exists:
        raise ValueError(
            "use_precomputed_embeddings=true requires precompute_entities or precompute_relations "
            "to be enabled in the same pipeline run."
        )
    if use_precomputed_questions and not precompute_questions and not skip_parquet_stage:
        raise ValueError(
            "use_precomputed_questions=true requires precompute_questions to be enabled in the same pipeline run."
        )
    parquet_dir_cfg = cfg.get("parquet_dir")
    out_dir_cfg = cfg.get("out_dir")
    if parquet_dir_cfg and out_dir_cfg:
        if not skip_parquet_stage and not skip_lmdb_stage and ctx.parquet_dir.resolve() != ctx.out_dir.resolve():
            raise ValueError(
                "parquet_dir must match out_dir in the unified pipeline. "
                f"Got parquet_dir={ctx.parquet_dir} vs out_dir={ctx.out_dir}."
            )


def _apply_pipeline_stage(ctx: StageContext) -> str:
    cfg = ctx.cfg
    stage = str(cfg.get("pipeline_stage", "all")).strip().lower()
    if not stage:
        stage = "all"
    if stage not in (
        "all",
        "parquet",
        "lmdb",
        "inverse_detect",
        "inverse_resolve",
        "inverse_describe",
        "inverse_relations",
    ):
        raise ValueError(
            "pipeline_stage must be one of: all, parquet, lmdb, "
            "inverse_detect, inverse_resolve, inverse_describe, inverse_relations "
            f"(got {stage!r})."
        )
    if stage.startswith("inverse"):
        cfg["skip_parquet_stage"] = True
        cfg["skip_lmdb_stage"] = True
        log_event(
            ctx.logger,
            "pipeline_stage_applied",
            stage=stage,
            overrides={"skip_parquet_stage": True, "skip_lmdb_stage": True},
        )
        return stage
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
    if stage == "parquet":
        inv_cfg = cfg.get("inverse_relations") if hasattr(cfg, "get") else None
        if isinstance(inv_cfg, dict) or hasattr(inv_cfg, "get"):
            enabled = bool(inv_cfg.get("enabled", False))
            mapping_path = inv_cfg.get("mapping_path")
            if enabled and mapping_path:
                path = ctx.resolve_path(mapping_path)
                if not path.exists():
                    inv_cfg["enabled"] = False
                    log_event(
                        ctx.logger,
                        "pipeline_stage_applied",
                        stage=stage,
                        overrides={"inverse_relations.enabled": False},
                        reason="inverse_relations_missing",
                    )
    return stage


def _apply_stage_overrides(ctx: StageContext) -> None:
    cfg = ctx.cfg
    if not bool(cfg.get("skip_lmdb_stage", False)):
        return
    overrides = {
        "precompute_entities": False,
        "precompute_relations": False,
        "precompute_questions": False,
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
        log_event(ctx.logger, "pipeline_overrides", reason="skip_lmdb_stage", overrides=changed)

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
    stage = _apply_pipeline_stage(ctx)
    _validate_pipeline_cfg(ctx)
    _apply_stage_overrides(ctx)
    _ensure_pipeline_dirs(ctx)
    if stage == "inverse_detect":
        build_inverse_relations_detect(ctx)
        return
    if stage == "inverse_resolve":
        build_inverse_relations_resolve(ctx)
        return
    if stage == "inverse_describe":
        build_inverse_relations_describe(ctx)
        return
    if stage == "inverse_relations":
        build_inverse_relations_all(ctx)
        return
    if not bool(cfg.get("skip_parquet_stage", False)):
        _run_parquet_stage(ctx)
    if not bool(cfg.get("skip_lmdb_stage", False)):
        build_dataset(ctx)
