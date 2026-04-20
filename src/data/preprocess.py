from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Callable, Mapping, Any
import hydra
from omegaconf import DictConfig
from .preprocess_steps.graph_collect import (
    collect_and_filter_graphs,
    load_question_entity_blacklist,
)
from .preprocess_steps.source import iter_samples
from .preprocess_steps.samples import SplitFilter

log = logging.getLogger(__name__)


def _resolve_path(raw_path: object) -> Path:
    if raw_path in (None, ""):
        raise ValueError("Expected a non-empty path value.")
    return Path(hydra.utils.to_absolute_path(str(raw_path)))


def _compile_entity_matcher(
    mode: str,
    prefixes: list[str],
    regex_value: str | None,
    default_result: bool,
) -> Callable[[str], bool]:
    resolved_mode = str(mode or "none").strip().lower()
    prefix_tuple = tuple(str(prefix) for prefix in (prefixes or []) if str(prefix))
    compiled_regex = re.compile(str(regex_value)) if regex_value else None

    def _matcher(entity: str) -> bool:
        value = str(entity or "").strip()
        if resolved_mode in {"", "none"}:
            return default_result if default_result and compiled_regex is None else False
        if resolved_mode in {"prefix", "prefix_allowlist"}:
            return bool(prefix_tuple) and value.startswith(prefix_tuple)
        if resolved_mode == "regex":
            return bool(compiled_regex and compiled_regex.match(value))
        raise ValueError(f"Unsupported entity matcher mode: {resolved_mode!r}.")

    return _matcher


def _build_split_filter(section: Mapping[str, Any] | None) -> SplitFilter:
    section = section or {}
    return SplitFilter(
        skip_no_question_entity=bool(section.get("skip_no_question_entity", False)),
        skip_no_ans=bool(section.get("skip_no_ans", False)),
        skip_no_path=bool(section.get("skip_no_path", False)),
    )


def _resolve_path_mode(raw_value: object) -> str:
    path_mode = str(raw_value or "qa_directed").strip().lower()
    if path_mode not in {"undirected", "qa_directed"}:
        raise ValueError(f"Unsupported path_mode={path_mode!r}; expected one of ('undirected', 'qa_directed').")
    return path_mode


def _validate_split_filters_require_anchors(
    split_filters: Mapping[str, SplitFilter],
) -> None:
    invalid_splits = [
        split_name for split_name, split_filter in split_filters.items() if not split_filter.skip_no_question_entity
    ]
    if invalid_splits:
        raise ValueError(
            "preprocess_filter.*.skip_no_question_entity must be true because materialization "
            f"requires at least one in-graph question entity; invalid splits: {sorted(invalid_splits)}."
        )


def run_preprocess_pipeline(raw_cfg: DictConfig) -> None:
    from .preprocess_steps.materialize import materialize_preprocessed_data
    from .preprocess_steps.text_encode import encode_preprocessed_features

    dataset_cfg = raw_cfg.dataset
    dataset_name = str(dataset_cfg.get("name", ""))
    out_dir = _resolve_path(dataset_cfg.get("out_dir"))
    embeddings_dir = _resolve_path(dataset_cfg.paths.get("embeddings"))
    path_mode = _resolve_path_mode(raw_cfg.get("path_mode", "qa_directed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    is_text_entity_fn = _compile_entity_matcher(
        mode=str(dataset_cfg.get("entity_text_mode", "regex")),
        prefixes=[str(v) for v in dataset_cfg.get("text_prefixes", [])],
        regex_value=dataset_cfg.get("text_regex") or None,
        default_result=True,
    )
    is_cvt_entity_fn = _compile_entity_matcher(
        mode=str(dataset_cfg.get("cvt_entity_mode", "none")),
        prefixes=[str(v) for v in dataset_cfg.get("cvt_prefixes", [])],
        regex_value=dataset_cfg.get("cvt_regex") or None,
        default_result=False,
    )
    blacklist = load_question_entity_blacklist(
        inline_list=list(raw_cfg.get("question_entity_blacklist", []))
        + list(dataset_cfg.get("question_entity_blacklist", [])),
        file_path=(
            _resolve_path(raw_cfg.question_entity_blacklist_path)
            if raw_cfg.get("question_entity_blacklist_path")
            else None
        ),
    )
    sample_iter = iter_samples(
        dataset=dataset_name,
        kb=str(dataset_cfg.get("kb", "freebase")),
        splits=("train", "validation", "test"),
        column_map=dict(dataset_cfg.get("column_map", {})),
        entity_normalization=str(dataset_cfg.get("entity_normalization", "none")),
        dataset_source=str(dataset_cfg.get("dataset_source", "hf")).strip().lower(),
        hf_dataset=dataset_cfg.get("hf_dataset"),
        hf_cache_dir=raw_cfg.get("hf_env", {}).get("cache_dir"),
        stark_cfg=dataset_cfg.get("stark"),
    )
    split_filters = {
        "train": _build_split_filter(raw_cfg.preprocess_filter.get("train")),
        "validation": _build_split_filter(
            raw_cfg.preprocess_filter.get("validation") or raw_cfg.preprocess_filter.get("eval")
        ),
        "test": _build_split_filter(raw_cfg.preprocess_filter.get("test") or raw_cfg.preprocess_filter.get("eval")),
    }
    _validate_split_filters_require_anchors(split_filters)
    prepared_samples, entity_vocab, relation_vocab = collect_and_filter_graphs(
        sample_iter=sample_iter,
        out_dir=out_dir,
        dataset_name=dataset_name,
        split_filters=split_filters,
        is_text_entity_fn=is_text_entity_fn,
        is_cvt_entity_fn=is_cvt_entity_fn,
        blacklist=blacklist,
        path_mode=path_mode,
        teacher_budget_max_steps=int(raw_cfg.get("teacher_budget_max_steps", 0)),
        dedup_edges=bool(raw_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(raw_cfg.get("remove_self_loops", True)),
        emit_sub_filter=bool(raw_cfg.get("emit_sub_filter", True)),
        sub_filter_filename=str(raw_cfg.get("sub_filter_filename", "sub_filter.json")),
    )
    encoded_payload = encode_preprocessed_features(
        prepared_samples=prepared_samples,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        embeddings_dir=embeddings_dir,
        encoder_name=str(raw_cfg.encoder.model_name),
        device=str(raw_cfg.encoder.get("device", "auto")),
        batch_size=int(raw_cfg.encoder.batch_size),
        progress_bar=bool(raw_cfg.get("progress_bar", True)),
    )
    materialize_preprocessed_data(
        prepared_samples=prepared_samples,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        payload=encoded_payload,
        embeddings_dir=embeddings_dir,
        path_mode=path_mode,
        overwrite_lmdb=bool(raw_cfg.get("overwrite_lmdb", False)),
        lmdb_shards=int(raw_cfg.get("lmdb_shards", 1)),
        map_size_gb=float(raw_cfg.get("map_size_gb", 128)),
    )
    log.info(f"Successfully preprocessed and materialized dataset: {dataset_name}")


__all__ = ["run_preprocess_pipeline"]
