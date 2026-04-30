from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast
import datasets
from .samples import RawSample


def iter_samples(
    *,
    dataset: str,
    splits: Sequence[str],
    column_map: Mapping[str, str],
    dataset_source: str = "hf",
    hf_dataset: str | None = None,
    hf_cache_dir: Path | None = None,
) -> Iterable[RawSample]:
    source = str(dataset_source or "hf").strip().lower()
    if source == "hf":
        dataset_path = str(hf_dataset or "").strip()
        if not dataset_path:
            raise ValueError("hf dataset source requires a non-empty `hf_dataset`.")
        for split in splits:
            dataset_obj = datasets.load_dataset(
                dataset_path,
                split=split,
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
                    split=str(split),
                    column_map=column_map,
                )
        return
    if source == "stark":
        raise NotImplementedError("stark dataset source is temporarily disabled.")
    raise ValueError(f"Unsupported dataset_source={dataset_source!r}.")


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    column_map: Mapping[str, str],
) -> RawSample:
    graph_field = column_map.get("graph_field", "graph")
    question_entity_field = column_map.get("question_entity_field", "question_entities")
    answer_entity_field = column_map.get("answer_entity_field", "answer_entities")
    question_id_field = column_map.get("question_id_field", "id")
    question_field = column_map.get("question_field", "question")
    parsed_graph = _parse_graph(row.get(graph_field))
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


def _parse_graph(graph_raw: object) -> tuple[tuple[str, str, str], ...]:
    if graph_raw is None:
        return ()
    if not isinstance(graph_raw, (list, tuple)):
        raise TypeError(
            f"Expected graph to be a list/tuple of triples, got {type(graph_raw).__name__}."
        )
    edges: list[tuple[str, str, str]] = []
    for item in graph_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        edges.append(
            (
                _normalize_entity(item[0]),
                str(item[1] or "").strip(),
                _normalize_entity(item[2]),
            )
        )
    return tuple(edges)


def _normalize_entity(value: object) -> str:
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
            s = str(item).strip()
            if s:
                result.append(s)
        return result
    s = str(value).strip()
    return [s] if s else []


__all__ = ["iter_samples"]
