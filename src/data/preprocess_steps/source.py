from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Optional, Any

from .samples import RawSample
from .stark_adapter import iter_stark_samples

_HF_DATASETS_MODULE = None


def iter_samples(
    *,
    dataset: str,
    kb: str,
    splits: Sequence[str],
    column_map: Mapping[str, str],
    entity_normalization: str,
    dataset_source: str = "hf",
    hf_dataset: str | None = None,
    hf_cache_dir: Optional[Path] = None,
    stark_cfg: Optional[Mapping[str, object]] = None,
) -> Iterable[RawSample]:
    """统一的数据集样本迭代器"""
    source = str(dataset_source or "hf").strip().lower()

    if source == "hf":
        dataset_path = str(hf_dataset or "").strip()
        if not dataset_path:
            raise ValueError("hf dataset source requires a non-empty `hf_dataset`.")

        for split in splits:
            yield from _iter_hf_samples(
                dataset=dataset,
                kb=kb,
                dataset_path=dataset_path,
                split=str(split),
                column_map=column_map,
                entity_normalization=entity_normalization,
                cache_dir=hf_cache_dir,
            )
    elif source == "stark":
        if stark_cfg is None:
            raise ValueError("stark dataset source requires a non-empty `stark_cfg`.")
        yield from iter_stark_samples(
            dataset=dataset,
            kb=kb,
            splits=splits,
            stark_cfg=stark_cfg,
        )
    else:
        raise ValueError(f"Unsupported dataset_source={dataset_source!r}.")


def _iter_hf_samples(
    *,
    dataset: str,
    kb: str,
    dataset_path: str,
    split: str,
    column_map: Mapping[str, str],
    entity_normalization: str,
    cache_dir: Optional[Path],
) -> Iterator[RawSample]:
    datasets = _import_hf_datasets_module()
    dataset_obj = datasets.load_dataset(
        dataset_path,
        split=split,
        cache_dir=None if cache_dir in (None, "") else str(cache_dir),
    )

    for row in dataset_obj:
        yield _row_to_sample(
            row,
            dataset=dataset,
            split=split,
            kb=kb,
            column_map=column_map,
            entity_normalization=entity_normalization,
        )


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    kb: str,
    column_map: Mapping[str, str],
    entity_normalization: str,
) -> RawSample:
    # 动态获取映射列名
    f = lambda key, default: column_map.get(key, default)

    graph_raw = row.get(f("graph_field", "graph"))

    parsed_graph = tuple(_parse_graph(graph_raw, entity_normalization))

    question_entities = tuple(
        _normalize_entity(item, entity_normalization)
        for item in _coerce_string_list(row.get(f("question_entity_field", "question_entities")))
    )
    answer_entities = tuple(
        _normalize_entity(item, entity_normalization)
        for item in _coerce_string_list(row.get(f("answer_entity_field", "answer_entities")))
    )

    return RawSample(
        dataset=dataset,
        split=split,
        question_id=str(row.get(f("question_id_field", "id"), "")),
        kb=kb,
        question=str(row.get(f("question_field", "question"), "")),
        graph=parsed_graph,
        question_entities=question_entities,
        answer_entities=answer_entities,
        answer_texts=tuple(_coerce_string_list(row.get(f("answer_text_field", "answer_text")))),
    )


def _parse_graph(graph_raw: Any, norm: str) -> Iterator[tuple[str, str, str]]:
    if not isinstance(graph_raw, (list, tuple)):
        return
    for t in graph_raw:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            yield (_normalize_entity(t[0], norm), str(t[1] or "").strip(), _normalize_entity(t[2], norm))


def _normalize_entity(value: Any, norm: str) -> str:
    entity = str(value or "").strip()
    if norm == "none":
        return entity
    raise ValueError(f"Unsupported entity_normalization={norm!r}.")


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(i).strip() for i in value if str(i).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _import_hf_datasets_module():
    """防御性导入：防止本地同名文件夹遮蔽官方库"""
    global _HF_DATASETS_MODULE
    if _HF_DATASETS_MODULE is not None:
        return _HF_DATASETS_MODULE
    # 这里保留你之前的 sys.modules.pop 逻辑...
    _HF_DATASETS_MODULE = importlib.import_module("datasets")
    return _HF_DATASETS_MODULE


__all__ = ["iter_samples"]
