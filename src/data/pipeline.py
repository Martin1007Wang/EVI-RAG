from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig

from .preprocess.graph_collect import GraphCollectStats, prepare_sample
from .preprocess.materialize import MaterializationPlan, SplitPlan, StreamingMaterializer
from .preprocess.samples import SplitFilter
from .preprocess.source import iter_samples
from .preprocess.vocab import (
    EntityCatalog,
    EntityTextPolicy,
    EntityVocab,
    RelationCatalog,
    RelationVocab,
)

log = logging.getLogger(__name__)

LOGICAL_SPLITS = ("train", "validation", "test")
_LEGACY_ROOT_PREPROCESS_KEYS = frozenset(
    {
        "preprocess_filter",
        "encoder",
        "dedup_edges",
        "remove_self_loops",
        "validate_graph_alignment",
        "overwrite_lmdb",
        "map_size_gb",
        "progress_bar",
        "stream_chunk_size",
        "commit_frequency",
        "hf_env",
    }
)
_STALE_MATERIALIZED_PATH_KEYS = frozenset(
    {
        "lmdb_dir",
        "entity_text_embeddings",
        "relation_embeddings",
        "entity_metadata_path",
        "entity_catalog_path",
        "relation_catalog_path",
    }
)


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path | None
    metadata_dir: Path
    embeddings_dir: Path


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str
    revision: str
    tokenizer_name: str | None
    tokenizer_revision: str | None
    max_length: int | None
    device: str
    batch_size: int
    cache_enabled: bool
    cache_dir: Path | None
    entity_prefix: str
    relation_prefix: str
    question_prefix: str


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    dataset_source: str
    hf_dataset: str | None
    hf_revision: str | None
    hf_cache_dir: Path | None
    entity_text_policy: EntityTextPolicy
    source_splits: dict[str, str]
    column_map: dict[str, str]
    source_options: dict[str, Any]
    paths: DatasetPaths
    split_filters: dict[str, SplitFilter]
    dedup_edges: bool
    remove_self_loops: bool
    validate_graph_alignment: bool
    overwrite_lmdb: bool
    map_size_gb: float
    progress_bar: bool
    stream_chunk_size: int
    commit_frequency: int
    encoder: TextEncoderConfig


@dataclass(frozen=True)
class PreprocessResult:
    dataset_name: str
    num_samples: int
    num_entities: int
    num_relations: int
    paths: DatasetPaths
    split_counts: dict[str, int]


def _resolve_path(value: Any, *, name: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"Expected a non-empty path value for {name}.")
    return Path(str(value))


def _resolve_optional_path(value: Any, *, name: str) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value, name=name)


def _resolve_required_string(value: Any, *, name: str) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(
            f"{name} must be non-empty. Default null revisions must be "
            "overridden by CLI or experiment config before preprocessing."
        )
    return resolved


def _resolve_optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip()
    return resolved or None


def _resolve_optional_positive_int(value: Any, *, name: str) -> int | None:
    if value in (None, ""):
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive or null, got {value}.")
    return resolved


def _build_split_filter(section: Mapping[str, Any] | None) -> SplitFilter:
    section = section or {}
    return SplitFilter(
        require_answer_in_graph=bool(section.get("require_answer_in_graph", False)),
        require_reachable_answer=bool(section.get("require_reachable_answer", False)),
    )


def _resolve_split_filters(
    *,
    preprocess_filter_cfg: Mapping[str, Any],
    logical_splits: Mapping[str, str],
) -> dict[str, SplitFilter]:
    filters: dict[str, SplitFilter] = {}
    eval_filter = preprocess_filter_cfg.get("eval")
    for split in logical_splits:
        section = preprocess_filter_cfg.get(split)
        if section is None and split in {"validation", "test"}:
            section = eval_filter
        filters[split] = _build_split_filter(_as_optional_mapping(section, name=split))
    return filters


def _resolve_source_splits(dataset_cfg: Mapping[str, Any]) -> dict[str, str]:
    splits_cfg = dataset_cfg.get("splits", {})
    if splits_cfg is None:
        splits_cfg = {}
    if not isinstance(splits_cfg, Mapping):
        raise TypeError("dataset.splits must be a mapping if provided")

    split_items = (
        splits_cfg.items()
        if splits_cfg
        else ((split, split) for split in LOGICAL_SPLITS)
    )
    source_splits: dict[str, str] = {}
    for logical_split, configured_source_split in split_items:
        logical = str(logical_split).strip()
        if not logical:
            raise ValueError(
                "dataset.splits keys must be non-empty logical split names"
            )
        source_split = str(configured_source_split).strip()
        if not source_split:
            raise ValueError(
                f"dataset.splits.{logical} must be a non-empty source split name"
            )
        source_splits[logical] = source_split
    if not source_splits:
        raise ValueError("dataset.splits must contain at least one split")
    return source_splits


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _as_optional_mapping(value: Any, *, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _as_mapping(value, name=name)


def _resolve_dataset_paths(dataset_cfg: Any) -> DatasetPaths:
    paths_cfg = dataset_cfg.get("paths")
    if paths_cfg is None:
        raise KeyError("dataset.paths must be provided")
    paths_cfg = _as_mapping(paths_cfg, name="dataset.paths")
    _reject_stale_materialized_path_keys(paths_cfg)
    return DatasetPaths(
        raw_dir=_resolve_optional_path(
            paths_cfg.get("raw_dir"),
            name="dataset.paths.raw_dir",
        ),
        metadata_dir=_resolve_path(
            paths_cfg.get("metadata_dir"),
            name="dataset.paths.metadata_dir",
        ),
        embeddings_dir=_resolve_path(
            paths_cfg.get("embeddings_dir"),
            name="dataset.paths.embeddings_dir",
        ),
    )


def _reject_stale_materialized_path_keys(paths_cfg: Mapping[str, Any]) -> None:
    stale_keys = sorted(key for key in _STALE_MATERIALIZED_PATH_KEYS if key in paths_cfg)
    if stale_keys:
        formatted = ", ".join(f"dataset.paths.{key}" for key in stale_keys)
        raise KeyError(
            "Materialized training artifacts are manifest-addressed; remove stale "
            f"path key(s): {formatted}. Configure dataset.paths.metadata_dir instead."
        )


def _resolve_entity_text_policy(dataset_cfg: Mapping[str, Any]) -> EntityTextPolicy:
    policy_cfg = dataset_cfg.get("entity_typing", {})
    if policy_cfg is None:
        policy_cfg = {}
    policy_cfg = _as_mapping(policy_cfg, name="dataset.entity_typing")
    prefixes = policy_cfg.get("non_text_prefixes", ())
    if prefixes is None:
        prefixes = ()
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    elif isinstance(prefixes, Mapping) or not isinstance(prefixes, Iterable):
        raise TypeError("dataset.entity_typing.non_text_prefixes must be a sequence")
    return EntityTextPolicy(non_text_prefixes=tuple(str(prefix) for prefix in prefixes))


def _resolve_source_options(
    *,
    dataset_source: str,
    dataset_cfg: Mapping[str, Any],
    paths: DatasetPaths,
) -> dict[str, Any]:
    if dataset_source != "stark":
        return {}
    stark_options = dict(
        _as_mapping(dataset_cfg.get("stark", {}), name="dataset.stark")
    )
    if stark_options.get("root") in (None, ""):
        if paths.raw_dir is None:
            raise ValueError(
                "dataset_source=stark requires dataset.paths.raw_dir or dataset.stark.root."
            )
        stark_options["root"] = str(paths.raw_dir)
    return {"stark": stark_options}


def build_preprocess_config(raw_cfg: DictConfig) -> PreprocessConfig:
    _reject_legacy_root_preprocess_keys(raw_cfg)
    dataset_cfg = _as_mapping(raw_cfg.get("dataset"), name="dataset")
    preprocess_cfg = _as_mapping(raw_cfg.get("preprocess"), name="preprocess")

    dataset_name = str(dataset_cfg.get("name", "")).strip()
    if not dataset_name:
        raise ValueError("dataset.name must be non-empty")

    paths = _resolve_dataset_paths(dataset_cfg)
    source_splits = _resolve_source_splits(dataset_cfg)
    preprocess_filter_cfg = _as_mapping(
        preprocess_cfg.get("preprocess_filter", {}),
        name="preprocess.preprocess_filter",
    )
    split_filters = _resolve_split_filters(
        preprocess_filter_cfg=preprocess_filter_cfg,
        logical_splits=source_splits,
    )
    dataset_source = str(dataset_cfg.get("dataset_source", "hf")).strip().lower()
    hf_dataset = _resolve_optional_string(dataset_cfg.get("hf_dataset"))
    hf_revision = None
    if dataset_source == "hf":
        hf_revision = _resolve_required_string(
            dataset_cfg.get("hf_revision"),
            name="dataset.hf_revision",
        )

    encoder_cfg = _as_mapping(preprocess_cfg.get("encoder"), name="preprocess.encoder")
    encoder_cache_enabled = bool(encoder_cfg.get("cache_enabled", True))
    encoder_cache_dir = None
    if encoder_cache_enabled:
        configured_cache_dir = encoder_cfg.get("cache_dir")
        encoder_cache_dir = (
            _resolve_optional_path(
                configured_cache_dir,
                name="preprocess.encoder.cache_dir",
            )
            or paths.embeddings_dir / "text_encode_cache"
        )

    hf_env_cfg = _as_mapping(preprocess_cfg.get("hf_env", {}), name="preprocess.hf_env")
    stream_chunk_size = int(preprocess_cfg.get("stream_chunk_size", 4096))
    if stream_chunk_size <= 0:
        raise ValueError("preprocess.stream_chunk_size must be positive")
    commit_frequency = int(preprocess_cfg.get("commit_frequency", 1000))
    if commit_frequency <= 0:
        raise ValueError("preprocess.commit_frequency must be positive")

    return PreprocessConfig(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        hf_dataset=hf_dataset,
        hf_revision=hf_revision,
        hf_cache_dir=_resolve_optional_path(
            hf_env_cfg.get("cache_dir"),
            name="preprocess.hf_env.cache_dir",
        ),
        entity_text_policy=_resolve_entity_text_policy(dataset_cfg),
        source_splits=source_splits,
        column_map={
            str(k): str(v) for k, v in dataset_cfg.get("column_map", {}).items()
        },
        source_options=_resolve_source_options(
            dataset_source=dataset_source,
            dataset_cfg=dataset_cfg,
            paths=paths,
        ),
        paths=paths,
        split_filters=split_filters,
        dedup_edges=bool(preprocess_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(preprocess_cfg.get("remove_self_loops", True)),
        validate_graph_alignment=bool(
            preprocess_cfg.get("validate_graph_alignment", False)
        ),
        overwrite_lmdb=bool(preprocess_cfg.get("overwrite_lmdb", False)),
        map_size_gb=float(preprocess_cfg.get("map_size_gb", 128)),
        progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
        stream_chunk_size=stream_chunk_size,
        commit_frequency=commit_frequency,
        encoder=TextEncoderConfig(
            model_name=_resolve_required_string(
                encoder_cfg.get("model_name"),
                name="preprocess.encoder.model_name",
            ),
            revision=_resolve_required_string(
                encoder_cfg.get("revision"),
                name="preprocess.encoder.revision",
            ),
            tokenizer_name=_resolve_optional_string(encoder_cfg.get("tokenizer_name")),
            tokenizer_revision=_resolve_optional_string(
                encoder_cfg.get("tokenizer_revision")
            ),
            max_length=_resolve_optional_positive_int(
                encoder_cfg.get("max_length"),
                name="preprocess.encoder.max_length",
            ),
            device=str(encoder_cfg.get("device", "auto")),
            batch_size=int(encoder_cfg.get("batch_size", 512)),
            cache_enabled=encoder_cache_enabled,
            cache_dir=encoder_cache_dir,
            entity_prefix=str(encoder_cfg.get("entity_prefix", "")),
            relation_prefix=str(encoder_cfg.get("relation_prefix", "")),
            question_prefix=str(
                encoder_cfg.get(
                    "question_prefix",
                    "Represent this sentence for searching relevant passages: ",
                )
            ),
        ),
    )


def _reject_legacy_root_preprocess_keys(raw_cfg: Mapping[str, Any]) -> None:
    if "preprocess" in raw_cfg and raw_cfg.get("preprocess") is not None:
        legacy_keys = sorted(
            key for key in _LEGACY_ROOT_PREPROCESS_KEYS if key in raw_cfg
        )
        if legacy_keys:
            formatted = ", ".join(legacy_keys)
            raise KeyError(
                "Preprocess options must live under cfg.preprocess; "
                f"found legacy root key(s): {formatted}"
            )


def run_preprocess_pipeline(raw_cfg: DictConfig) -> PreprocessResult:
    return run_preprocess(build_preprocess_config(raw_cfg))


def run_preprocess(config: PreprocessConfig) -> PreprocessResult:
    from .preprocess.text_encode import TextEncoder, encode_text_table, text_encoder_provenance

    for path in (config.paths.metadata_dir, config.paths.embeddings_dir):
        path.mkdir(parents=True, exist_ok=True)

    def sample_iter_factory() -> Any:
        return iter_samples(
            dataset=config.dataset_name,
            split_mapping=config.source_splits,
            column_map=config.column_map,
            dataset_source=config.dataset_source,
            hf_dataset=config.hf_dataset,
            hf_revision=config.hf_revision,
            hf_cache_dir=config.hf_cache_dir,
            source_options=config.source_options,
        )

    log.info("Stage 1/4: scan graph samples")
    scan = _scan_samples(
        sample_iter=sample_iter_factory(),
        split_filters=config.split_filters,
        dedup_edges=config.dedup_edges,
        remove_self_loops=config.remove_self_loops,
        validate_graph_alignment=config.validate_graph_alignment,
    )
    if scan.stats.kept <= 0:
        raise RuntimeError("No valid samples after graph collection.")
    log.info(
        "Stage 1 complete: %d samples (drop: %s)",
        scan.stats.kept,
        scan.stats.drops(),
    )

    entity_catalog = EntityCatalog.build(
        scan.entity_vocab,
        text_policy=config.entity_text_policy,
    )
    relation_catalog = RelationCatalog.build(scan.relation_vocab)
    encoder_provenance = text_encoder_provenance(
        encoder_name=config.encoder.model_name,
        encoder_revision=config.encoder.revision,
        tokenizer_name=config.encoder.tokenizer_name,
        tokenizer_revision=config.encoder.tokenizer_revision,
        max_length=config.encoder.max_length,
    )
    encoder = TextEncoder(
        model_name=config.encoder.model_name,
        revision=config.encoder.revision,
        device=config.encoder.device,
        progress_bar=config.progress_bar,
        tokenizer_name=config.encoder.tokenizer_name,
        tokenizer_revision=config.encoder.tokenizer_revision,
        max_length=config.encoder.max_length,
    )

    log.info("Stage 2/4: encode global text tables")
    entity_text_embeddings = encode_text_table(
        texts=entity_catalog.entity_text_labels,
        encoder_name=config.encoder.model_name,
        encoder_revision=config.encoder.revision,
        device=config.encoder.device,
        batch_size=config.encoder.batch_size,
        progress_bar=config.progress_bar,
        cache_dir=config.encoder.cache_dir,
        cache_kind="entities",
        desc="Entities",
        query_prefix=config.encoder.entity_prefix,
        encoder=encoder,
        tokenizer_name=config.encoder.tokenizer_name,
        tokenizer_revision=config.encoder.tokenizer_revision,
        max_length=config.encoder.max_length,
    )
    relation_embeddings = encode_text_table(
        texts=relation_catalog.relation_text_labels,
        encoder_name=config.encoder.model_name,
        encoder_revision=config.encoder.revision,
        device=config.encoder.device,
        batch_size=config.encoder.batch_size,
        progress_bar=config.progress_bar,
        cache_dir=config.encoder.cache_dir,
        cache_kind="relations",
        desc="Relations",
        query_prefix=config.encoder.relation_prefix,
        encoder=encoder,
        tokenizer_name=config.encoder.tokenizer_name,
        tokenizer_revision=config.encoder.tokenizer_revision,
        max_length=config.encoder.max_length,
    )
    hidden_dim = int(entity_text_embeddings.size(1))
    if hidden_dim <= 0:
        raise ValueError("encoder produced empty entity embedding dimension")
    log.info("Stage 2 complete")

    split_plans = {
        split: SplitPlan(num_samples=count)
        for split, count in scan.split_counts.items()
        if count > 0
    }
    materialization_plan = MaterializationPlan(
        split_plans=split_plans,
        question_embedding_dim=hidden_dim,
    )

    log.info("Stage 3/4: stream materialize")
    with StreamingMaterializer(
        plan=materialization_plan,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        entity_text_embeddings=entity_text_embeddings,
        relation_embeddings=relation_embeddings,
        metadata_dir=config.paths.metadata_dir,
        overwrite=config.overwrite_lmdb,
        map_size_gb=config.map_size_gb,
        commit_frequency=config.commit_frequency,
        provenance=_build_preprocess_provenance(
            dataset_name=config.dataset_name,
            dataset_source=config.dataset_source,
            hf_dataset=config.hf_dataset,
            hf_revision=config.hf_revision,
            source_splits=config.source_splits,
            column_map=config.column_map,
            dedup_edges=config.dedup_edges,
            remove_self_loops=config.remove_self_loops,
            validate_graph_alignment=config.validate_graph_alignment,
            split_filters=config.split_filters,
            entity_text_policy=config.entity_text_policy,
            encoder_provenance=encoder_provenance,
            encoder_device=config.encoder.device,
            encoder_batch_size=config.encoder.batch_size,
            encoder_cache_enabled=config.encoder.cache_enabled,
            encoder_cache_dir=config.encoder.cache_dir,
            entity_prefix=config.encoder.entity_prefix,
            relation_prefix=config.encoder.relation_prefix,
            question_prefix=config.encoder.question_prefix,
            progress_bar=config.progress_bar,
            stream_chunk_size=config.stream_chunk_size,
            map_size_gb=config.map_size_gb,
            commit_frequency=config.commit_frequency,
        ),
    ) as materializer:
        _stream_samples(
            sample_iter=sample_iter_factory(),
            split_filters=config.split_filters,
            entity_vocab=scan.entity_vocab,
            relation_vocab=scan.relation_vocab,
            materializer=materializer,
            encoder=encoder,
            encoder_name=config.encoder.model_name,
            encoder_revision=config.encoder.revision,
            tokenizer_name=config.encoder.tokenizer_name,
            tokenizer_revision=config.encoder.tokenizer_revision,
            max_length=config.encoder.max_length,
            encoder_device=config.encoder.device,
            encoder_batch_size=config.encoder.batch_size,
            progress_bar=config.progress_bar,
            encoder_cache_dir=config.encoder.cache_dir,
            question_prefix=config.encoder.question_prefix,
            chunk_size=config.stream_chunk_size,
            dedup_edges=config.dedup_edges,
            remove_self_loops=config.remove_self_loops,
            validate_graph_alignment=config.validate_graph_alignment,
        )
    log.info("Stage 3 complete")
    log.info("Stage 4/4: verify stream counts")
    if materializer.split_counts != {
        split: plan.num_samples for split, plan in split_plans.items()
    }:
        raise RuntimeError(
            "materialized split counts mismatch: "
            f"got {materializer.split_counts}, expected={split_plans}"
        )
    log.info("Preprocess finished for dataset=%s", config.dataset_name)
    return PreprocessResult(
        dataset_name=config.dataset_name,
        num_samples=scan.stats.kept,
        num_entities=entity_catalog.num_entities,
        num_relations=len(relation_catalog.relation_labels),
        paths=config.paths,
        split_counts=dict(materializer.split_counts),
    )


class _ScanResult:
    def __init__(
        self,
        *,
        entity_vocab: EntityVocab,
        relation_vocab: RelationVocab,
        split_counts: dict[str, int],
        stats: GraphCollectStats,
    ) -> None:
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.split_counts = split_counts
        self.stats = stats


def _scan_samples(
    *,
    sample_iter: Any,
    split_filters: dict[str, SplitFilter],
    dedup_edges: bool,
    remove_self_loops: bool,
    validate_graph_alignment: bool,
) -> _ScanResult:
    entity_vocab = EntityVocab()
    relation_vocab = RelationVocab()
    stats = GraphCollectStats()
    split_counts: dict[str, int] = {}
    for sample in sample_iter:
        prepared = prepare_sample(
            sample=sample,
            split_filters=split_filters,
            entity_vocab=entity_vocab,
            relation_vocab=relation_vocab,
            dedup_edges=dedup_edges,
            remove_self_loops=remove_self_loops,
            validate_alignment=validate_graph_alignment,
            update_vocab=True,
            stats=stats,
        )
        if prepared is not None:
            split_counts[prepared.split] = split_counts.get(prepared.split, 0) + 1
    return _ScanResult(
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        split_counts=split_counts,
        stats=stats,
    )


def _stream_samples(
    *,
    sample_iter: Any,
    split_filters: dict[str, SplitFilter],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    materializer: StreamingMaterializer,
    encoder: Any,
    encoder_name: str,
    encoder_revision: str,
    tokenizer_name: str | None,
    tokenizer_revision: str | None,
    max_length: int | None,
    encoder_device: str,
    encoder_batch_size: int,
    progress_bar: bool,
    encoder_cache_dir: Path | None,
    question_prefix: str,
    chunk_size: int,
    dedup_edges: bool,
    remove_self_loops: bool,
    validate_graph_alignment: bool,
) -> None:
    chunk: list[Any] = []
    for sample in sample_iter:
        prepared = prepare_sample(
            sample=sample,
            split_filters=split_filters,
            entity_vocab=entity_vocab,
            relation_vocab=relation_vocab,
            dedup_edges=dedup_edges,
            remove_self_loops=remove_self_loops,
            validate_alignment=validate_graph_alignment,
            update_vocab=False,
        )
        if prepared is None:
            continue
        chunk.append(prepared)
        if len(chunk) >= chunk_size:
            _write_question_chunk(
                chunk=chunk,
                materializer=materializer,
                encoder=encoder,
                encoder_name=encoder_name,
                encoder_revision=encoder_revision,
                tokenizer_name=tokenizer_name,
                tokenizer_revision=tokenizer_revision,
                max_length=max_length,
                encoder_device=encoder_device,
                encoder_batch_size=encoder_batch_size,
                progress_bar=progress_bar,
                encoder_cache_dir=encoder_cache_dir,
                question_prefix=question_prefix,
            )
            chunk = []
    if chunk:
        _write_question_chunk(
            chunk=chunk,
            materializer=materializer,
            encoder=encoder,
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            max_length=max_length,
            encoder_device=encoder_device,
            encoder_batch_size=encoder_batch_size,
            progress_bar=progress_bar,
            encoder_cache_dir=encoder_cache_dir,
            question_prefix=question_prefix,
        )


def _write_question_chunk(
    *,
    chunk: list[Any],
    materializer: StreamingMaterializer,
    encoder: Any,
    encoder_name: str,
    encoder_revision: str,
    tokenizer_name: str | None,
    tokenizer_revision: str | None,
    max_length: int | None,
    encoder_device: str,
    encoder_batch_size: int,
    progress_bar: bool,
    encoder_cache_dir: Path | None,
    question_prefix: str,
) -> None:
    from .preprocess.text_encode import encode_text_table

    cache_key = f"questions-{_question_chunk_key(chunk)}"
    question_embeddings = encode_text_table(
        texts=[sample.question for sample in chunk],
        encoder_name=encoder_name,
        encoder_revision=encoder_revision,
        device=encoder_device,
        batch_size=encoder_batch_size,
        progress_bar=progress_bar,
        cache_dir=encoder_cache_dir,
        cache_kind=cache_key,
        desc="Questions",
        query_prefix=question_prefix,
        encoder=encoder,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        max_length=max_length,
    )
    materializer.write_chunk(
        prepared_samples=chunk,
        question_embeddings=question_embeddings,
    )


def _question_chunk_key(chunk: list[Any]) -> str:
    import hashlib
    import json

    payload = [
        {
            "sample_id": f"{sample.dataset}/{sample.split}/{sample.question_id}",
            "question": sample.question,
        }
        for sample in chunk
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_preprocess_provenance(
    *,
    dataset_name: str,
    dataset_source: str,
    hf_dataset: Any,
    hf_revision: str | None,
    source_splits: Mapping[str, str],
    column_map: Mapping[str, Any],
    dedup_edges: bool,
    remove_self_loops: bool,
    validate_graph_alignment: bool,
    split_filters: Mapping[str, SplitFilter],
    entity_text_policy: EntityTextPolicy,
    encoder_provenance: Mapping[str, object],
    encoder_device: str,
    encoder_batch_size: int,
    encoder_cache_enabled: bool,
    encoder_cache_dir: Path | None,
    entity_prefix: str,
    relation_prefix: str,
    question_prefix: str,
    progress_bar: bool,
    stream_chunk_size: int,
    map_size_gb: float,
    commit_frequency: int,
) -> dict[str, Any]:
    return {
        "dataset": {
            "name": dataset_name,
            "dataset_source": dataset_source,
            "hf_dataset": None if hf_dataset in (None, "") else str(hf_dataset),
            "hf_revision": hf_revision,
            "splits": {str(k): str(v) for k, v in source_splits.items()},
            "column_map": {str(k): str(v) for k, v in column_map.items()},
        },
        "preprocess": {
            "dedup_edges": bool(dedup_edges),
            "remove_self_loops": bool(remove_self_loops),
            "validate_graph_alignment": bool(validate_graph_alignment),
            "split_filters": {
                str(split): {
                    "require_answer_in_graph": bool(filter_cfg.require_answer_in_graph),
                    "require_reachable_answer": bool(
                        filter_cfg.require_reachable_answer
                    ),
                }
                for split, filter_cfg in split_filters.items()
            },
            "entity_typing": {
                "non_text_prefixes": list(entity_text_policy.non_text_prefixes),
            },
            "progress_bar": bool(progress_bar),
            "stream_chunk_size": int(stream_chunk_size),
            "map_size_gb": float(map_size_gb),
            "commit_frequency": int(commit_frequency),
        },
        "encoder": {
            **dict(encoder_provenance),
            "device": str(encoder_device),
            "batch_size": int(encoder_batch_size),
            "cache_enabled": bool(encoder_cache_enabled),
            "cache_dir": None if encoder_cache_dir is None else str(encoder_cache_dir),
            "entity_prefix": str(entity_prefix),
            "relation_prefix": str(relation_prefix),
            "question_prefix": str(question_prefix),
        },
    }


__all__ = [
    "DatasetPaths",
    "PreprocessConfig",
    "PreprocessResult",
    "TextEncoderConfig",
    "build_preprocess_config",
    "run_preprocess",
    "run_preprocess_pipeline",
]
