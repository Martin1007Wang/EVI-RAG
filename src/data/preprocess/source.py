from __future__ import annotations
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import datasets

from .samples import RawSample

log = logging.getLogger(__name__)


@dataclass
class _GraphParseStats:
    samples: int = 0
    kept_edges: int = 0
    dropped_triples: int = 0
    malformed_triple: int = 0
    empty_head: int = 0
    empty_tail: int = 0
    empty_relation: int = 0

    def drops(self) -> dict[str, int]:
        return {
            "malformed_triple": self.malformed_triple,
            "empty_head": self.empty_head,
            "empty_tail": self.empty_tail,
            "empty_relation": self.empty_relation,
        }


def iter_samples(
    *,
    dataset: str,
    split_mapping: Mapping[str, str] | None = None,
    column_map: Mapping[str, str],
    splits: Sequence[str] | None = None,
    dataset_source: str = "hf",
    hf_dataset: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: Path | None = None,
    source_options: Mapping[str, object] | None = None,
) -> Iterable[RawSample]:
    resolved_split_mapping = _resolve_split_mapping(
        split_mapping=split_mapping,
        splits=splits,
    )
    source = str(dataset_source or "hf").strip().lower()
    if source == "hf":
        dataset_path = str(hf_dataset or "").strip()
        if not dataset_path:
            raise ValueError("hf dataset source requires a non-empty `hf_dataset`.")
        revision = str(hf_revision or "").strip()
        if not revision:
            raise ValueError("hf dataset source requires a non-empty `hf_revision`.")
        for logical_split, source_split in resolved_split_mapping.items():
            graph_parse_stats = _GraphParseStats()
            dataset_obj = datasets.load_dataset(
                dataset_path,
                split=source_split,
                revision=revision,
                cache_dir=None if hf_cache_dir in (None, "") else str(hf_cache_dir),
            )
            for row in dataset_obj:
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"Expected dataset row to be a mapping, got {type(row).__name__}."
                    )
                yield _row_to_sample(
                    cast(Mapping[str, object], row),
                    dataset=dataset,
                    split=logical_split,
                    column_map=column_map,
                    graph_parse_stats=graph_parse_stats,
                )
            _log_graph_parse_stats(
                dataset=dataset,
                logical_split=logical_split,
                source_split=source_split,
                stats=graph_parse_stats,
            )
        return
    if source == "stark":
        from .stark import iter_stark_samples

        yield from iter_stark_samples(
            dataset=dataset,
            split_mapping=resolved_split_mapping,
            options=source_options,
        )
        return
    raise ValueError(f"Unsupported dataset_source={dataset_source!r}.")


def _resolve_split_mapping(
    *,
    split_mapping: Mapping[str, str] | None,
    splits: Sequence[str] | None,
) -> dict[str, str]:
    if split_mapping is None:
        if splits is None:
            raise TypeError("iter_samples requires split_mapping or splits")
        split_mapping = {str(split): str(split) for split in splits}

    resolved: dict[str, str] = {}
    for logical_split, source_split in split_mapping.items():
        logical = str(logical_split).strip()
        source = str(source_split).strip()
        if not logical:
            raise ValueError("logical split names must be non-empty")
        if not source:
            raise ValueError(f"source split for {logical!r} must be non-empty")
        resolved[logical] = source
    if not resolved:
        raise ValueError("iter_samples requires at least one split")
    return resolved


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    column_map: Mapping[str, str],
    graph_parse_stats: _GraphParseStats | None = None,
) -> RawSample:
    graph_field = column_map.get("graph_field", "graph")
    question_entity_field = column_map.get("question_entity_field", "question_entities")
    answer_entity_field = column_map.get("answer_entity_field", "answer_entities")
    question_id_field = column_map.get("question_id_field", "id")
    question_field = column_map.get("question_field", "question")
    parsed_graph = _parse_graph(row.get(graph_field), stats=graph_parse_stats)
    question_entities = tuple(
        _normalize_entity(item)
        for item in _coerce_string_list(row.get(question_entity_field))
    )
    answer_entities = tuple(
        _normalize_entity(item)
        for item in _coerce_string_list(row.get(answer_entity_field))
    )
    return RawSample(
        dataset=dataset,
        split=split,
        question_id=str(row.get(question_id_field, "")).strip(),
        question=str(row.get(question_field, "")).strip(),
        graph=parsed_graph,
        question_entities=question_entities,
        answer_entities=answer_entities,
    )


def _parse_graph(
    graph_raw: object,
    *,
    stats: _GraphParseStats | None = None,
) -> tuple[tuple[str, str, str], ...]:
    if stats is not None:
        stats.samples += 1
    if graph_raw is None:
        return ()
    if not isinstance(graph_raw, (list, tuple)):
        raise TypeError(
            f"Expected graph to be a list/tuple of triples, got {type(graph_raw).__name__}."
        )
    edges: list[tuple[str, str, str]] = []
    for item in graph_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            _record_dropped_triple(stats, malformed_triple=True)
            continue
        head = _normalize_entity(item[0])
        relation = _normalize_relation(item[1])
        tail = _normalize_entity(item[2])
        if not head or not relation or not tail:
            _record_dropped_triple(
                stats,
                empty_head=not head,
                empty_relation=not relation,
                empty_tail=not tail,
            )
            continue
        edges.append((head, relation, tail))
        if stats is not None:
            stats.kept_edges += 1
    return tuple(edges)


def _record_dropped_triple(
    stats: _GraphParseStats | None,
    *,
    malformed_triple: bool = False,
    empty_head: bool = False,
    empty_relation: bool = False,
    empty_tail: bool = False,
) -> None:
    if stats is None:
        return
    stats.dropped_triples += 1
    if malformed_triple:
        stats.malformed_triple += 1
    if empty_head:
        stats.empty_head += 1
    if empty_relation:
        stats.empty_relation += 1
    if empty_tail:
        stats.empty_tail += 1


def _log_graph_parse_stats(
    *,
    dataset: str,
    logical_split: str,
    source_split: str,
    stats: _GraphParseStats,
) -> None:
    if stats.dropped_triples:
        log.warning(
            "Source graph parse dropped triples: dataset=%s logical_split=%s "
            "source_split=%s samples=%d kept_edges=%d dropped_triples=%d "
            "drop_reasons=%s",
            dataset,
            logical_split,
            source_split,
            stats.samples,
            stats.kept_edges,
            stats.dropped_triples,
            stats.drops(),
        )
        return
    log.info(
        "Source graph parse complete: dataset=%s logical_split=%s source_split=%s "
        "samples=%d kept_edges=%d dropped_triples=0",
        dataset,
        logical_split,
        source_split,
        stats.samples,
        stats.kept_edges,
    )


def _normalize_entity(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_relation(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                result.append(s)
        return result
    s = str(value).strip()
    return [s] if s else []


__all__ = ["iter_samples"]
