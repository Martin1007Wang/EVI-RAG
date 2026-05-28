from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .preprocess.catalog import CatalogBuilder, EntityTextPolicy
from .preprocess.graph_collect import GraphCollectStats, prepare_sample
from .preprocess.materialize import Materializer
from .preprocess.samples import SplitFilter
from .preprocess.source import iter_samples
from .preprocess.text_encode import TextEncoder, encode_text_table, text_encoder_provenance


LOGICAL_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetPaths:
    metadata_dir: Path
    embeddings_dir: Path


@dataclass(frozen=True)
class PreprocessResult:
    dataset_name: str
    num_samples: int
    num_entities: int
    num_relations: int
    paths: DatasetPaths
    split_counts: dict[str, int]


def run_preprocess_pipeline(raw_cfg: DictConfig) -> PreprocessResult:
    dataset_cfg = _mapping(raw_cfg.get("dataset"), "dataset")
    preprocess_cfg = _mapping(raw_cfg.get("preprocess"), "preprocess")

    dataset_name = str(dataset_cfg.get("name", "")).strip()
    if not dataset_name:
        raise ValueError("dataset.name must be non-empty")

    paths = _resolve_dataset_paths(dataset_cfg)
    split_filters = _resolve_split_filters(dataset_cfg, preprocess_cfg)
    source_splits = _resolve_source_splits(dataset_cfg)
    dataset_source = str(dataset_cfg.get("dataset_source", "hf")).strip().lower()
    hf_dataset = _optional_str(dataset_cfg.get("hf_dataset"))
    hf_revision = _optional_str(dataset_cfg.get("hf_revision"))
    entity_text_policy = _resolve_entity_text_policy(dataset_cfg)
    column_map = {
        str(k): str(v) for k, v in _mapping(dataset_cfg.get("column_map", {}), "dataset.column_map").items()
    }
    encoder_cfg = _mapping(preprocess_cfg.get("encoder"), "preprocess.encoder")
    sample_iter_factory = lambda: iter_samples(
        dataset=dataset_name,
        split_mapping=source_splits,
        column_map=column_map,
        dataset_source=dataset_source,
        hf_dataset=hf_dataset,
        hf_revision=hf_revision,
        hf_cache_dir=_optional_path(_mapping(preprocess_cfg.get("hf_env", {}), "preprocess.hf_env").get("cache_dir")),
    )

    scan = _scan_samples(
        sample_iter=sample_iter_factory(),
        split_filters=split_filters,
        dedup_edges=bool(preprocess_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(preprocess_cfg.get("remove_self_loops", True)),
        validate_graph_alignment=bool(preprocess_cfg.get("validate_graph_alignment", False)),
    )
    if scan.stats.kept <= 0:
        raise RuntimeError("No valid samples after graph collection.")

    catalog = scan.catalog.build(text_policy=entity_text_policy)
    encoder_provenance = text_encoder_provenance(
        encoder_name=_required_str(encoder_cfg.get("model_name"), "preprocess.encoder.model_name"),
        encoder_revision=_required_str(encoder_cfg.get("revision"), "preprocess.encoder.revision"),
        tokenizer_name=_optional_str(encoder_cfg.get("tokenizer_name")),
        tokenizer_revision=_optional_str(encoder_cfg.get("tokenizer_revision")),
        max_length=_optional_int(encoder_cfg.get("max_length")),
    )
    encoder = TextEncoder(
        model_name=encoder_provenance["encoder_name"],
        revision=encoder_provenance["encoder_revision"],
        device=str(encoder_cfg.get("device", "auto")),
        progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
        tokenizer_name=encoder_provenance["tokenizer_name"],
        tokenizer_revision=encoder_provenance["tokenizer_revision"],
        max_length=encoder_provenance["max_length"],
    )

    entity_text_semantic_table = encode_text_table(
        texts=catalog.entity_text_labels,
        encoder_name=encoder_provenance["encoder_name"],
        encoder_revision=encoder_provenance["encoder_revision"],
        device=str(encoder_cfg.get("device", "auto")),
        batch_size=int(encoder_cfg.get("batch_size", 512)),
        progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
        cache_dir=paths.embeddings_dir / "text_encode_cache",
        cache_kind="entities",
        desc="Entities",
        query_prefix=str(encoder_cfg.get("entity_prefix", "")),
        encoder=encoder,
        tokenizer_name=encoder_provenance["tokenizer_name"],
        tokenizer_revision=encoder_provenance["tokenizer_revision"],
        max_length=encoder_provenance["max_length"],
    )
    relation_semantic_table = encode_text_table(
        texts=catalog.relation_text_labels,
        encoder_name=encoder_provenance["encoder_name"],
        encoder_revision=encoder_provenance["encoder_revision"],
        device=str(encoder_cfg.get("device", "auto")),
        batch_size=int(encoder_cfg.get("batch_size", 512)),
        progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
        cache_dir=paths.embeddings_dir / "text_encode_cache",
        cache_kind="relations",
        desc="Relations",
        query_prefix=str(encoder_cfg.get("relation_prefix", "")),
        encoder=encoder,
        tokenizer_name=encoder_provenance["tokenizer_name"],
        tokenizer_revision=encoder_provenance["tokenizer_revision"],
        max_length=encoder_provenance["max_length"],
    )

    prepared_samples = scan.prepared_samples
    split_counts = dict(scan.split_counts)
    question_dim = int(entity_text_semantic_table.size(1))
    chunk_size = int(preprocess_cfg.get("stream_chunk_size", 4096))

    with Materializer(
        split_counts=split_counts,
        question_dim=question_dim,
        catalog=catalog,
        entity_text_semantic_table=entity_text_semantic_table,
        relation_semantic_table=relation_semantic_table,
        metadata_dir=paths.metadata_dir,
        overwrite=bool(preprocess_cfg.get("overwrite_lmdb", False)),
        map_size_gb=float(preprocess_cfg.get("map_size_gb", 128.0)),
        commit_frequency=int(preprocess_cfg.get("commit_frequency", 1000)),
        provenance={
            "dataset": {
                "name": dataset_name,
                "dataset_source": dataset_source,
                "hf_dataset": hf_dataset,
                "hf_revision": hf_revision,
                "splits": source_splits,
                "column_map": column_map,
            },
                "preprocess": {
                "dedup_edges": bool(preprocess_cfg.get("dedup_edges", True)),
                "remove_self_loops": bool(preprocess_cfg.get("remove_self_loops", True)),
                "validate_graph_alignment": bool(preprocess_cfg.get("validate_graph_alignment", False)),
                "progress_bar": bool(preprocess_cfg.get("progress_bar", True)),
                "stream_chunk_size": chunk_size,
                "map_size_gb": float(preprocess_cfg.get("map_size_gb", 128.0)),
                    "commit_frequency": int(preprocess_cfg.get("commit_frequency", 1000)),
                    "weak_replay_labels": {
                        "kind": "witness_path_v1",
                        "path_policy": "deterministic_single_shortest_per_target",
                    },
                    "entity_typing": {"non_text_prefixes": list(entity_text_policy.non_text_prefixes)},
                },
            "encoder": dict(encoder_provenance),
        },
    ) as materializer:
        for chunk in _chunked(prepared_samples, chunk_size):
            question_embeddings = encode_text_table(
                texts=[sample.question for sample in chunk],
                encoder_name=encoder_provenance["encoder_name"],
                encoder_revision=encoder_provenance["encoder_revision"],
                device=str(encoder_cfg.get("device", "auto")),
                batch_size=int(encoder_cfg.get("batch_size", 512)),
                progress_bar=bool(preprocess_cfg.get("progress_bar", True)),
                cache_dir=paths.embeddings_dir / "text_encode_cache",
                cache_kind=f"questions-{_question_chunk_key(chunk)}",
                desc="Questions",
                query_prefix=str(encoder_cfg.get("question_prefix", "")),
                encoder=encoder,
                tokenizer_name=encoder_provenance["tokenizer_name"],
                tokenizer_revision=encoder_provenance["tokenizer_revision"],
                max_length=encoder_provenance["max_length"],
            )
            materializer.write_chunk(
                prepared_samples=chunk,
                question_embeddings=question_embeddings,
            )

    return PreprocessResult(
        dataset_name=dataset_name,
        num_samples=len(prepared_samples),
        num_entities=len(catalog.entity_labels),
        num_relations=len(catalog.relation_labels),
        paths=paths,
        split_counts=split_counts,
    )


@dataclass
class _ScannedSamples:
    prepared_samples: list[Any]
    split_counts: dict[str, int]
    stats: GraphCollectStats
    catalog: CatalogBuilder


def _scan_samples(
    *,
    sample_iter: Iterable[Any],
    split_filters: dict[str, SplitFilter],
    dedup_edges: bool,
    remove_self_loops: bool,
    validate_graph_alignment: bool,
) -> _ScannedSamples:
    catalog = CatalogBuilder()
    prepared_samples: list[Any] = []
    stats = GraphCollectStats()
    split_counts: dict[str, int] = {}
    for sample in sample_iter:
        prepared = prepare_sample(
            sample=sample,
            split_filters=split_filters,
            catalog_builder=catalog,
            dedup_edges=dedup_edges,
            remove_self_loops=remove_self_loops,
            validate_alignment=validate_graph_alignment,
            stats=stats,
        )
        if prepared is None:
            continue
        prepared_samples.append(prepared)
        split_counts[prepared.split] = split_counts.get(prepared.split, 0) + 1
    return _ScannedSamples(
        prepared_samples=prepared_samples,
        split_counts=split_counts,
        stats=stats,
        catalog=catalog,
    )


def _resolve_dataset_paths(dataset_cfg: Mapping[str, Any]) -> DatasetPaths:
    paths_cfg = _mapping(dataset_cfg.get("paths"), "dataset.paths")
    metadata_dir = _required_path(paths_cfg.get("metadata_dir"), "dataset.paths.metadata_dir")
    embeddings_dir = _required_path(paths_cfg.get("embeddings_dir"), "dataset.paths.embeddings_dir")
    return DatasetPaths(metadata_dir=metadata_dir, embeddings_dir=embeddings_dir)


def _resolve_source_splits(dataset_cfg: Mapping[str, Any]) -> dict[str, str]:
    splits_cfg = dataset_cfg.get("splits")
    if splits_cfg is None:
        return {split: split for split in LOGICAL_SPLITS}
    mapping_cfg = _mapping(splits_cfg, "dataset.splits")
    return {str(k): str(v) for k, v in mapping_cfg.items()}


def _resolve_split_filters(dataset_cfg: Mapping[str, Any], preprocess_cfg: Mapping[str, Any]) -> dict[str, SplitFilter]:
    filters_cfg = _mapping(preprocess_cfg.get("preprocess_filter", {}), "preprocess.preprocess_filter")
    eval_cfg = filters_cfg.get("eval")
    out: dict[str, SplitFilter] = {}
    for split in LOGICAL_SPLITS:
        section = filters_cfg.get(split)
        if section is None and split in {"validation", "test"}:
            section = eval_cfg
        section_map = _mapping(section, split) if section is not None else {}
        out[split] = SplitFilter(
            require_answer_in_graph=bool(section_map.get("require_answer_in_graph", False)),
            require_reachable_answer=bool(section_map.get("require_reachable_answer", False)),
        )
    return out


def _resolve_entity_text_policy(dataset_cfg: Mapping[str, Any]) -> EntityTextPolicy:
    policy_cfg = _mapping(dataset_cfg.get("entity_typing", {}), "dataset.entity_typing")
    prefixes = policy_cfg.get("non_text_prefixes", ())
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return EntityTextPolicy(non_text_prefixes=tuple(str(prefix) for prefix in prefixes))

def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("stream_chunk_size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


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
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_str(value: Any, name: str) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved


def _required_path(value: Any, name: str) -> Path:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"{name} must be provided")
    return Path(resolved)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip()
    return resolved or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


__all__ = ["DatasetPaths", "PreprocessResult", "run_preprocess_pipeline"]
