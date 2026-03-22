from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from omegaconf import DictConfig

from src.data.schema.types import (
    CvtEntityConfig,
    Sample,
    TextEntityConfig,
)


_QID_IN_PARENS_RE = re.compile(r"(Q\d+)")
_LABEL_QID_RE = re.compile(r"(.+)\s+\((Q\d+)\)$")
_DATA_SOURCE_HF = "hf"
_HF_DATASET_BY_FAMILY = {
    "cwq": "rmanluo/RoG-cwq",
    "webqsp": "rmanluo/RoG-webqsp",
}
_HF_DATASET_ALLOWED = tuple(_HF_DATASET_BY_FAMILY.values())
_SRC_DIR = Path(__file__).resolve().parents[2]
_LOCAL_DATASETS_INIT = _SRC_DIR / "datasets" / "__init__.py"
_HF_DATASETS_MODULE: ModuleType | None = None


def _module_file_path(module: ModuleType) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    try:
        return Path(module_file).resolve()
    except (OSError, RuntimeError):
        return None


def _is_local_datasets_shadow(module: ModuleType) -> bool:
    module_path = _module_file_path(module)
    if module_path is None:
        return False
    return (
        module_path == _LOCAL_DATASETS_INIT
        or _LOCAL_DATASETS_INIT.parent in module_path.parents
    )


def _import_hf_datasets_module() -> ModuleType:
    global _HF_DATASETS_MODULE

    cached_module = _HF_DATASETS_MODULE
    if cached_module is not None and not _is_local_datasets_shadow(cached_module):
        return cached_module

    imported_module = sys.modules.get("datasets")
    if imported_module is not None and _is_local_datasets_shadow(imported_module):
        sys.modules.pop("datasets", None)
        imported_module = None

    if imported_module is not None:
        _HF_DATASETS_MODULE = imported_module
        return imported_module

    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in original_sys_path
            if Path(entry or ".").resolve() != _SRC_DIR
        ]
        hf_datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "datasets is required for HF-based preprocessing."
        ) from exc
    finally:
        sys.path = original_sys_path

    if _is_local_datasets_shadow(hf_datasets):
        raise RuntimeError(
            "Resolved local `src.datasets` while importing HuggingFace `datasets`; "
            "remove the repo `src` directory from `sys.path` before preprocessing."
        )

    _HF_DATASETS_MODULE = hf_datasets
    return hf_datasets


def build_text_entity_config(cfg: DictConfig) -> TextEntityConfig:
    mode = str(cfg.get("entity_text_mode", "regex"))
    prefixes_cfg = cfg.get("text_prefixes") or []
    prefixes = tuple(str(prefix) for prefix in prefixes_cfg)
    regex_str = cfg.get("text_regex")
    regex = re.compile(str(regex_str)) if regex_str else None
    if mode == "regex" and regex is None:
        raise ValueError("entity_text_mode=regex requires text_regex to be set.")
    if mode == "prefix_allowlist" and not prefixes:
        raise ValueError(
            "entity_text_mode=prefix_allowlist requires non-empty text_prefixes."
        )
    return TextEntityConfig(mode=mode, prefixes=prefixes, regex=regex)


def build_cvt_entity_config(cfg: DictConfig) -> CvtEntityConfig:
    mode = str(cfg.get("cvt_entity_mode", "regex"))
    prefixes_cfg = cfg.get("cvt_prefixes") or []
    prefixes = tuple(str(prefix) for prefix in prefixes_cfg)
    regex_str = cfg.get("cvt_regex")
    regex = re.compile(str(regex_str)) if regex_str else None
    if mode == "regex" and regex is None:
        raise ValueError("cvt_entity_mode=regex requires cvt_regex to be set.")
    if mode == "prefix_allowlist" and not prefixes:
        raise ValueError(
            "cvt_entity_mode=prefix_allowlist requires non-empty cvt_prefixes."
        )
    return CvtEntityConfig(mode=mode, prefixes=prefixes, regex=regex)


def normalize_entity(entity: str, mode: str) -> str:
    if mode == "qid_in_parentheses":
        match = _QID_IN_PARENS_RE.search(entity)
        if match:
            return match.group(1)
    return entity


def normalize_entity_with_lookup(
    entity: str, mode: str, label_to_qid: Dict[str, str]
) -> str:
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
    tolist = getattr(field, "tolist", None)
    if callable(tolist):
        values = tolist()
        if isinstance(values, list):
            return [str(x) for x in values]
    return [str(field)]


def _resolve_hf_dataset_id(
    dataset: str, dataset_family: Optional[str], hf_dataset: Optional[str]
) -> str:
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
    hf_datasets = _import_hf_datasets_module()
    DownloadConfig = hf_datasets.DownloadConfig
    load_dataset = hf_datasets.load_dataset

    download_config = DownloadConfig(local_files_only=offline)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    return load_dataset(
        dataset_id,
        split=split,
        cache_dir=cache_dir_str,
        download_config=download_config,
    )


def _iter_hf_rows(dataset_obj) -> Iterator[Mapping[str, object]]:
    for row in dataset_obj:
        yield row


def _register_label_qid(label_to_qid: Dict[str, str], raw_value: object) -> None:
    label_match = _LABEL_QID_RE.match(str(raw_value))
    if label_match is None:
        return
    label_to_qid[label_match.group(1).strip()] = label_match.group(2)


def _build_label_to_qid_lookup(
    graph_raw: Sequence[object],
    *,
    q_entities_raw: Sequence[str],
    a_entities_raw: Sequence[str],
    entity_normalization: str,
) -> Dict[str, str]:
    if entity_normalization != "qid_in_parentheses":
        return {}
    label_to_qid: Dict[str, str] = {}
    for tr in graph_raw:
        if not isinstance(tr, (list, tuple)):
            continue
        if len(tr) < 3:
            continue
        _register_label_qid(label_to_qid, tr[0])
        _register_label_qid(label_to_qid, tr[2])
    for entity in q_entities_raw:
        _register_label_qid(label_to_qid, entity)
    for entity in a_entities_raw:
        _register_label_qid(label_to_qid, entity)
    return label_to_qid


def _row_to_sample(
    row: Mapping[str, object],
    *,
    dataset: str,
    split: str,
    kb: str,
    column_map: Dict[str, str],
    entity_normalization: str,
) -> Sample:
    graph_raw_value = row.get(column_map["graph_field"])
    if isinstance(graph_raw_value, (list, tuple)):
        graph_raw: list[object] = list(graph_raw_value)
    else:
        graph_raw = []
    q_entities_raw = to_list(row.get(column_map["q_entity_field"]))
    a_entities_raw = to_list(row.get(column_map["a_entity_field"]))
    label_to_qid = _build_label_to_qid_lookup(
        graph_raw,
        q_entities_raw=q_entities_raw,
        a_entities_raw=a_entities_raw,
        entity_normalization=entity_normalization,
    )
    graph: List[tuple[str, str, str]] = []
    for tr in graph_raw:
        if not isinstance(tr, (list, tuple)):
            continue
        if len(tr) >= 3:
            h_raw = str(tr[0])
            t_raw = str(tr[2])
            h = normalize_entity_with_lookup(h_raw, entity_normalization, label_to_qid)
            r = str(tr[1])
            t = normalize_entity_with_lookup(t_raw, entity_normalization, label_to_qid)
            graph.append((h, r, t))

    q_entities = [
        normalize_entity_with_lookup(ent, entity_normalization, label_to_qid)
        for ent in q_entities_raw
    ]
    a_entities = [
        normalize_entity_with_lookup(ent, entity_normalization, label_to_qid)
        for ent in a_entities_raw
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
        question=str(row.get(column_map["question_field"]) or ""),
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
    dataset_source: str = _DATA_SOURCE_HF,
    dataset_family: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_cache_dir: Optional[Path] = None,
    hf_offline: bool = False,
) -> Iterable[Sample]:
    del raw_root
    for split in splits:
        source = str(dataset_source).strip().lower()
        if source != _DATA_SOURCE_HF:
            raise ValueError(
                f"Unsupported dataset_source={dataset_source!r}; expected '{_DATA_SOURCE_HF}'."
            )
        dataset_id = _resolve_hf_dataset_id(dataset, dataset_family, hf_dataset)
        dataset_obj = _load_hf_split(
            dataset_id, split, cache_dir=hf_cache_dir, offline=hf_offline
        )
        row_iter = _iter_hf_rows(dataset_obj)

        for row in row_iter:
            yield _row_to_sample(
                row,
                dataset=dataset,
                split=split,
                kb=kb,
                column_map=column_map,
                entity_normalization=entity_normalization,
            )
