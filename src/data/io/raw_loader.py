from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import pyarrow.dataset as ds

from omegaconf import DictConfig

from src.data.schema.types import Sample, TextEntityConfig


_QID_IN_PARENS_RE = re.compile(r"(Q\d+)")
_LABEL_QID_RE = re.compile(r"(.+)\s+\((Q\d+)\)$")
_DATA_SOURCE_HF = "hf"
_DATA_SOURCE_PARQUET = "parquet"
_HF_DATASET_BY_FAMILY = {
    "cwq": "rmanluo/RoG-cwq",
    "webqsp": "rmanluo/RoG-webqsp",
}
_HF_DATASET_ALLOWED = tuple(_HF_DATASET_BY_FAMILY.values())


def build_text_entity_config(cfg: DictConfig) -> TextEntityConfig:
    mode = str(cfg.get("entity_text_mode", "regex"))
    prefixes_cfg = cfg.get("text_prefixes") or []
    prefixes = tuple(str(prefix) for prefix in prefixes_cfg)
    regex_str = cfg.get("text_regex")
    regex = re.compile(str(regex_str)) if regex_str else None
    if mode == "regex" and regex is None:
        raise ValueError("entity_text_mode=regex requires text_regex to be set.")
    if mode == "prefix_allowlist" and not prefixes:
        raise ValueError("entity_text_mode=prefix_allowlist requires non-empty text_prefixes.")
    return TextEntityConfig(mode=mode, prefixes=prefixes, regex=regex)


def normalize_entity(entity: str, mode: str) -> str:
    if mode == "qid_in_parentheses":
        match = _QID_IN_PARENS_RE.search(entity)
        if match:
            return match.group(1)
    return entity


def normalize_entity_with_lookup(entity: str, mode: str, label_to_qid: Dict[str, str]) -> str:
    normalized = normalize_entity(entity, mode)
    if mode == "qid_in_parentheses" and normalized == entity:
        qid = label_to_qid.get(entity)
        if qid:
            return qid
    return normalized


def to_list(field: object) -> List[str]:
    if field is None:
        return []
    if isinstance(field, (list, tuple)):
        return [str(x) for x in field]
    import numpy as np

    if isinstance(field, np.ndarray):
        return [str(x) for x in field.tolist()]
    return [str(field)]


def load_split(raw_root: Path, split: str) -> ds.Dataset:
    paths = sorted(raw_root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards found for split '{split}' under {raw_root}")
    return ds.dataset([str(p) for p in paths])


def _resolve_hf_dataset_id(dataset: str, dataset_family: Optional[str], hf_dataset: Optional[str]) -> str:
    dataset_key = (dataset_family or dataset or "").strip().lower()
    dataset_id = hf_dataset or _HF_DATASET_BY_FAMILY.get(dataset_key)
    if dataset_id is None:
        allowed = ", ".join(sorted(_HF_DATASET_BY_FAMILY))
        raise ValueError(
            "Unsupported dataset for HF loader. "
            f"Got dataset={dataset!r}, dataset_family={dataset_family!r}. "
            f"Allowed families: {allowed}."
        )
    if dataset_id not in _HF_DATASET_ALLOWED:
        allowed = ", ".join(sorted(_HF_DATASET_ALLOWED))
        raise ValueError(f"hf_dataset must be one of: {allowed}. Got {dataset_id!r}.")
    return dataset_id


def _load_hf_split(
    dataset_id: str,
    split: str,
    *,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
):
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("datasets is required for HF-based preprocessing.") from exc

    download_config = DownloadConfig(local_files_only=offline)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    return load_dataset(
        dataset_id,
        split=split,
        cache_dir=cache_dir_str,
        download_config=download_config,
    )


def _iter_parquet_rows(dataset_obj: ds.Dataset) -> Iterator[Mapping[str, object]]:
    for batch in dataset_obj.to_batches():
        for row in batch.to_pylist():
            yield row


def _iter_hf_rows(dataset_obj) -> Iterator[Mapping[str, object]]:
    for row in dataset_obj:
        yield row


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    kb: str,
    column_map: Dict[str, str],
    entity_normalization: str,
) -> Sample:
    graph_raw = row.get(column_map["graph_field"]) or []
    label_to_qid: Dict[str, str] = {}
    graph: List[tuple[str, str, str]] = []
    for tr in graph_raw:
        if len(tr) >= 3:
            h_raw = str(tr[0])
            t_raw = str(tr[2])
            if entity_normalization == "qid_in_parentheses":
                for node_raw in (h_raw, t_raw):
                    label_match = _LABEL_QID_RE.match(node_raw)
                    if label_match:
                        label_to_qid[label_match.group(1).strip()] = label_match.group(2)
            h = normalize_entity_with_lookup(h_raw, entity_normalization, label_to_qid)
            r = str(tr[1])
            t = normalize_entity_with_lookup(t_raw, entity_normalization, label_to_qid)
            graph.append((h, r, t))

    q_entities = [
        normalize_entity_with_lookup(ent, entity_normalization, label_to_qid)
        for ent in to_list(row.get(column_map["q_entity_field"]))
    ]
    a_entities = [
        normalize_entity_with_lookup(ent, entity_normalization, label_to_qid)
        for ent in to_list(row.get(column_map["a_entity_field"]))
    ]
    answer_texts = to_list(row.get(column_map["answer_text_field"]))
    graph_iso_type = None
    if "graph_iso_field" in column_map:
        val = row.get(column_map["graph_iso_field"])
        graph_iso_type = str(val) if val is not None else None
    redundant = None
    if "redundant_field" in column_map:
        red_val = row.get(column_map["redundant_field"])
        if isinstance(red_val, bool):
            redundant = red_val
        elif red_val is not None:
            redundant = str(red_val).lower() == "true"
    test_type: List[str] = []
    if "test_type_field" in column_map:
        test_type = to_list(row.get(column_map["test_type_field"]))

    return Sample(
        dataset=dataset,
        split=split,
        question_id=str(row[column_map["question_id_field"]]),
        kb=kb,
        question=row.get(column_map["question_field"]) or "",
        graph=graph,
        q_entity=q_entities,
        a_entity=a_entities,
        answer_texts=answer_texts,
        graph_iso_type=graph_iso_type,
        redundant=redundant,
        test_type=test_type,
    )


def iter_samples(
    dataset: str,
    kb: str,
    raw_root: Optional[Path],
    splits: Sequence[str],
    column_map: Dict[str, str],
    entity_normalization: str,
    *,
    dataset_source: str = _DATA_SOURCE_PARQUET,
    dataset_family: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_cache_dir: Optional[Path] = None,
    hf_offline: bool = False,
) -> Iterable[Sample]:
    for split in splits:
        source = str(dataset_source).strip().lower()
        if source == _DATA_SOURCE_HF:
            dataset_id = _resolve_hf_dataset_id(dataset, dataset_family, hf_dataset)
            dataset_obj = _load_hf_split(dataset_id, split, cache_dir=hf_cache_dir, offline=hf_offline)
            row_iter = _iter_hf_rows(dataset_obj)
        elif source == _DATA_SOURCE_PARQUET:
            if raw_root is None:
                raise ValueError("raw_root must be set when dataset_source='parquet'.")
            dataset_obj = load_split(raw_root, split)
            row_iter = _iter_parquet_rows(dataset_obj)
        else:
            raise ValueError(f"Unsupported dataset_source={dataset_source!r}; expected '{_DATA_SOURCE_HF}' or 'parquet'.")

        for row in row_iter:
            yield _row_to_sample(
                row,
                dataset=dataset,
                split=split,
                kb=kb,
                column_map=column_map,
                entity_normalization=entity_normalization,
            )
