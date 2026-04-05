from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Mapping, Optional

from .sample_types import RawSample
from .stark_adapter import iter_stark_samples

_HF_DATASETS_MODULE = None

_LEGACY_COLUMN_MAP_KEYS = {
    "q_entity_field": "question_entity_field",
    "a_entity_field": "answer_entity_field",
}


# ==============================================================================
# 1. 顶层入口
# ==============================================================================
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


# ==============================================================================
# 2. HuggingFace 数据加载与解析逻辑
# ==============================================================================
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

    resolved_column_map = _resolve_column_map(column_map)

    for row in dataset_obj:
        if not isinstance(row, Mapping):
            raise ValueError(f"Expected each row to be a mapping, got {type(row)!r}.")
        yield _row_to_sample(
            row,
            dataset=dataset,
            split=split,
            kb=kb,
            resolved_column_map=resolved_column_map,
            entity_normalization=entity_normalization,
        )


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    kb: str,
    resolved_column_map: Mapping[str, str],
    entity_normalization: str,
) -> RawSample:
    """将 DataFrame 的单行转换为严格不可变的 RawSample 对象"""
    graph_col = resolved_column_map.get("graph_field", "graph")
    question_entity_col = resolved_column_map.get(
        "question_entity_field", "question_entities"
    )
    answer_entity_col = resolved_column_map.get(
        "answer_entity_field", "answer_entities"
    )
    answer_text_col = resolved_column_map.get("answer_text_field", "answer_text")
    question_id_col = resolved_column_map.get("question_id_field", "id")
    question_col = resolved_column_map.get("question_field", "question")

    # 彻底移除了 qid_lookup 的构建和传递，直接解析图和实体
    graph_raw = row.get(graph_col)

    parsed_graph = tuple(
        _parse_graph(graph_raw, entity_normalization=entity_normalization)
    )
    question_entities = tuple(
        _normalize_entity(item, entity_normalization=entity_normalization)
        for item in _coerce_string_list(row.get(question_entity_col))
    )
    answer_entities = tuple(
        _normalize_entity(item, entity_normalization=entity_normalization)
        for item in _coerce_string_list(row.get(answer_entity_col))
    )
    answer_texts = tuple(_coerce_string_list(row.get(answer_text_col)))

    return RawSample(
        dataset=str(dataset),
        split=str(split),
        question_id=str(row.get(question_id_col, "")),
        kb=str(kb),
        question=str(row.get(question_col, "")),
        graph=parsed_graph,
        question_entities=question_entities,
        answer_entities=answer_entities,
        answer_texts=answer_texts,
    )


def _parse_graph(
    graph_raw: object,
    *,
    entity_normalization: str,
) -> Iterator[tuple[str, str, str]]:
    """使用生成器优化内存，保留最纯粹的三元组提取逻辑"""
    if not isinstance(graph_raw, (list, tuple)):
        return

    for triple in graph_raw:
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        head = _normalize_entity(triple[0], entity_normalization=entity_normalization)
        relation = _strip_entity_text(triple[1])
        tail = _normalize_entity(triple[2], entity_normalization=entity_normalization)
        yield (head, relation, tail)


# ==============================================================================
# 3. 实体清洗与辅助工具 (极简版)
# ==============================================================================
def _normalize_entity(
    value: object,
    *,
    entity_normalization: str,
) -> str:
    """仅保留 Freebase 所需的纯净实体提取"""
    entity = _strip_entity_text(value)
    if entity_normalization == "none":
        return entity

    # 拦截旧版遗留配置，确保 Fail-Fast
    raise ValueError(
        f"Unsupported entity_normalization={entity_normalization!r}; "
        "Expected 'none'. Legacy Wikidata (QID) normalization has been removed."
    )


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [_strip_entity_text(item) for item in value if _strip_entity_text(item)]
    item = _strip_entity_text(value)
    return [item] if item else []


def _strip_entity_text(value: object) -> str:
    return str(value or "").strip()


def _resolve_column_map(column_map: Mapping[str, str]) -> dict[str, str]:
    resolved = dict(column_map)
    for legacy_key, new_key in _LEGACY_COLUMN_MAP_KEYS.items():
        if legacy_key in resolved:
            raise ValueError(
                f"Legacy column map key detected: {legacy_key}->{new_key}."
            )
    return resolved


# ==============================================================================
# 4. 解决本地包重名的防御性 Hack
# ==============================================================================
def _import_hf_datasets_module():
    """防止本地可能存在的 datasets 文件夹覆盖官方库"""
    global _HF_DATASETS_MODULE
    if _HF_DATASETS_MODULE is not None and not _is_local_datasets_shadow(
        _HF_DATASETS_MODULE
    ):
        return _HF_DATASETS_MODULE

    shadow = sys.modules.get("datasets")
    if shadow is not None and _is_local_datasets_shadow(shadow):
        sys.modules.pop("datasets", None)

    module = importlib.import_module("datasets")
    if _is_local_datasets_shadow(module):
        raise ImportError(
            "Imported local `datasets` module shadowed HuggingFace datasets package."
        )

    _HF_DATASETS_MODULE = module
    return module


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_local_datasets_shadow(module: object) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    module_path = Path(str(module_file)).resolve()
    src_root = Path(__file__).resolve().parents[2]
    return _path_is_within(module_path, src_root)


__all__ = ["iter_samples"]
