from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import hydra
import torch
from omegaconf import DictConfig

from .preprocess.graph_collect import collect_and_filter_graphs
from .preprocess.samples import PreparedSample, SplitFilter
from .preprocess.source import iter_samples
from .preprocess.vocab import (
    EntityCatalog,
    EntityTyping,
    RelationCatalog,
)

log = logging.getLogger(__name__)


def _resolve_path(value: Any, *, name: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"Expected a non-empty path value for {name}.")
    return Path(str(value))


def _build_split_filter(section: Mapping[str, Any] | None) -> SplitFilter:
    section = section or {}
    return SplitFilter(
        require_answer_in_graph=bool(section.get("require_answer_in_graph", False)),
        require_reachable_answer=bool(section.get("require_reachable_answer", False)),
    )


def _resolve_dataset_paths(dataset_cfg: Any) -> dict[str, Path]:
    paths_cfg = dataset_cfg.get("paths")
    if paths_cfg is None:
        raise KeyError("dataset.paths must be provided")
    return {
        "lmdb_dir": _resolve_path(
            paths_cfg.get("lmdb_dir"),
            name="dataset.paths.lmdb_dir",
        ),
        "metadata_dir": _resolve_path(
            paths_cfg.get("metadata_dir"),
            name="dataset.paths.metadata_dir",
        ),
        "embeddings_dir": _resolve_path(
            paths_cfg.get("embeddings_dir"),
            name="dataset.paths.embeddings_dir",
        ),
        "entity_metadata_path": _resolve_path(
            paths_cfg.get("entity_metadata_path"),
            name="dataset.paths.entity_metadata_path",
        ),
        "entity_catalog_path": _resolve_path(
            paths_cfg.get("entity_catalog_path"),
            name="dataset.paths.entity_catalog_path",
        ),
        "relation_catalog_path": _resolve_path(
            paths_cfg.get("relation_catalog_path"),
            name="dataset.paths.relation_catalog_path",
        ),
    }


def run_preprocess_pipeline(raw_cfg: DictConfig) -> None:
    from .preprocess.materialize import materialize_preprocessed_data
    from .preprocess.text_encode import encode_text_features

    dataset_cfg = raw_cfg.dataset
    preprocess_cfg = raw_cfg.preprocess
    dataset_name = str(dataset_cfg.get("name", "")).strip()
    if not dataset_name:
        raise ValueError("dataset.name must be non-empty")
    resolved_paths = _resolve_dataset_paths(dataset_cfg)
    for path in (
        resolved_paths["lmdb_dir"],
        resolved_paths["metadata_dir"],
        resolved_paths["embeddings_dir"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    split_filters = {
        "train": _build_split_filter(preprocess_cfg.preprocess_filter.get("train")),
        "validation": _build_split_filter(
            preprocess_cfg.preprocess_filter.get("validation")
            or preprocess_cfg.preprocess_filter.get("eval")
        ),
        "test": _build_split_filter(
            preprocess_cfg.preprocess_filter.get("test")
            or preprocess_cfg.preprocess_filter.get("eval")
        ),
    }
    sample_iter = iter_samples(
        dataset=dataset_name,
        splits=tuple(split_filters.keys()),
        column_map=dict(dataset_cfg.get("column_map", {})),
        dataset_source=str(dataset_cfg.get("dataset_source", "hf")).strip().lower(),
        hf_dataset=dataset_cfg.get("hf_dataset"),
        hf_cache_dir=preprocess_cfg.get("hf_env", {}).get("cache_dir"),
    )
    log.info("Stage 1/3: graph_collect")
    prepared_samples, entity_vocab, relation_vocab = collect_and_filter_graphs(
        sample_iter=sample_iter,
        split_filters=split_filters,
        dedup_edges=bool(preprocess_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(preprocess_cfg.get("remove_self_loops", True)),
    )
    log.info("Stage 1 complete: %d samples", len(prepared_samples))
    entity_catalog = EntityCatalog.build(entity_vocab, typing=EntityTyping())
    relation_catalog = RelationCatalog.build(relation_vocab)
    log.info("Stage 2/3: text encoding")
    question_texts = [s.question for s in prepared_samples]
    encoder_cache_dir = None
    if bool(preprocess_cfg.encoder.get("cache_enabled", True)):
        configured_cache_dir = preprocess_cfg.encoder.get("cache_dir")
        encoder_cache_dir = (
            _resolve_path(configured_cache_dir, name="preprocess.encoder.cache_dir")
            if configured_cache_dir not in (None, "")
            else resolved_paths["embeddings_dir"] / "text_encode_cache"
        )
    encoded = encode_text_features(
        entity_text_labels=entity_catalog.entity_text_labels,
        relation_text_labels=relation_catalog.relation_text_labels,
        question_texts=question_texts,
        encoder_name=str(preprocess_cfg.encoder.model_name),
        device=str(preprocess_cfg.encoder.get("device", "auto")),
        batch_size=int(preprocess_cfg.encoder.batch_size),
        progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
        cache_dir=encoder_cache_dir,
    )
    log.info("Stage 2 complete")
    log.info("Stage 3/3: materialize")
    materialize_preprocessed_data(
        prepared_samples=prepared_samples,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        entity_text_embeddings=encoded.entity_text_embeddings,
        relation_embeddings=encoded.relation_embeddings,
        question_embeddings=encoded.question_embeddings,
        lmdb_dir=resolved_paths["lmdb_dir"],
        metadata_dir=resolved_paths["metadata_dir"],
        embeddings_dir=resolved_paths["embeddings_dir"],
        entity_metadata_path=resolved_paths["entity_metadata_path"],
        entity_catalog_path=resolved_paths["entity_catalog_path"],
        relation_catalog_path=resolved_paths["relation_catalog_path"],
        overwrite_lmdb=bool(preprocess_cfg.get("overwrite_lmdb", False)),
        map_size_gb=float(preprocess_cfg.get("map_size_gb", 128)),
    )
    log.info("Stage 3 complete")
    log.info("Preprocess finished for dataset=%s", dataset_name)
