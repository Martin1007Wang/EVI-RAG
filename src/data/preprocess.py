from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping

import hydra
from omegaconf import DictConfig

from .preprocess_steps.graph_collect import (
    collect_and_filter_graphs,
    load_question_entity_blacklist,
)
from .preprocess_steps.sample_loader import iter_samples
from .preprocess_steps.sample_types import SplitFilter


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
            return (
                default_result if default_result and compiled_regex is None else False
            )
        if resolved_mode in {"prefix", "prefix_allowlist"}:
            return bool(prefix_tuple) and value.startswith(prefix_tuple)
        if resolved_mode == "regex":
            return bool(compiled_regex and compiled_regex.match(value))
        raise ValueError(f"Unsupported entity matcher mode: {resolved_mode!r}.")

    return _matcher


def _build_split_filter(section: Mapping[str, object] | None) -> SplitFilter:
    section = section or {}
    return SplitFilter(
        skip_no_question_entity=bool(section.get("skip_no_question_entity", False)),
        skip_no_ans=bool(section.get("skip_no_ans", False)),
        skip_no_path=bool(section.get("skip_no_path", False)),
    )


def run_preprocess_pipeline(raw_cfg: DictConfig) -> None:
    """Run preprocessing directly from the composed Hydra config."""
    from .preprocess_steps.materialize import materialize_preprocessed_data
    from .preprocess_steps.text_encode import encode_preprocessed_features

    dataset_cfg = raw_cfg.dataset
    dataset_name = str(dataset_cfg.get("name", ""))
    dataset_scope = str(dataset_cfg.get("dataset_scope", "")).strip().lower()
    if dataset_scope == "sub" or dataset_name.endswith("-sub"):
        raise ValueError(
            "Sub datasets are runtime filters only. Build the full dataset and use sub_filter.json."
        )

    paths_cfg = dataset_cfg.get("paths", {})
    out_dir = _resolve_path(dataset_cfg.get("out_dir"))
    embeddings_dir = _resolve_path(paths_cfg.get("embeddings"))
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    encoder_cfg = raw_cfg.get("encoder", {})
    encoder_name = str(
        encoder_cfg.get("model_name", encoder_cfg.get("name", ""))
    ).strip()
    device = str(encoder_cfg.get("device", "auto")) or "auto"
    precision = str(encoder_cfg.get("precision"))
    batch_size = int(encoder_cfg.get("batch_size"))
    reuse_embeddings_if_exists = bool(
        encoder_cfg.get("reuse_embeddings_if_exists", False)
    )

    dataset_source = str(dataset_cfg.get("dataset_source", "hf")).strip().lower()
    if dataset_source not in {"hf", "stark"}:
        raise ValueError(
            f"Unsupported dataset_source={dataset_source!r}; expected 'hf' or 'stark'."
        )
    hf_dataset = str(dataset_cfg.get("hf_dataset"))
    stark_cfg = dataset_cfg.get("stark")
    if dataset_source == "hf" and hf_dataset is None:
        raise ValueError("dataset_source=hf requires a non-empty dataset.hf_dataset.")
    if dataset_source == "stark" and stark_cfg is None:
        raise ValueError(
            "dataset_source=stark requires a non-empty dataset.stark mapping."
        )

    hf_env = raw_cfg.get("hf_env", {})
    hf_cache_raw = hf_env.get("cache_dir") or raw_cfg.get("hf_cache_dir")
    hf_cache_dir = None if hf_cache_raw in (None, "") else Path(str(hf_cache_raw))

    filter_cfg = raw_cfg.get("preprocess_filter", {})
    split_filters = {
        "train": _build_split_filter(filter_cfg.get("train")),
        "validation": _build_split_filter(
            filter_cfg.get("validation") or filter_cfg.get("eval")
        ),
        "test": _build_split_filter(filter_cfg.get("test") or filter_cfg.get("eval")),
    }

    is_text_entity_fn = _compile_entity_matcher(
        mode=str(dataset_cfg.get("entity_text_mode", "regex")),
        prefixes=[str(value) for value in dataset_cfg.get("text_prefixes", [])],
        regex_value=str(dataset_cfg.get("text_regex")),
        default_result=True,
    )
    is_cvt_entity_fn = _compile_entity_matcher(
        mode=str(dataset_cfg.get("cvt_entity_mode", "none")),
        prefixes=[str(value) for value in dataset_cfg.get("cvt_prefixes", [])],
        regex_value=str(dataset_cfg.get("cvt_regex")),
        default_result=False,
    )

    blacklist_path_raw = raw_cfg.get("question_entity_blacklist_path")
    blacklist = load_question_entity_blacklist(
        inline_list=list(raw_cfg.get("question_entity_blacklist", []))
        + list(dataset_cfg.get("question_entity_blacklist", [])),
        file_path=(
            None
            if blacklist_path_raw in (None, "")
            else _resolve_path(blacklist_path_raw)
        ),
    )

    sample_iter = iter_samples(
        dataset=dataset_name,
        kb=str(dataset_cfg.get("kb", "freebase")),
        splits=("train", "validation", "test"),
        column_map={
            str(key): str(value)
            for key, value in dict(dataset_cfg.get("column_map", {})).items()
        },
        entity_normalization=str(dataset_cfg.get("entity_normalization", "none")),
        dataset_source=dataset_source,
        hf_dataset=hf_dataset,
        hf_cache_dir=hf_cache_dir,
        stark_cfg=stark_cfg,
    )

    prepared_samples, entity_vocab, relation_vocab = collect_and_filter_graphs(
        sample_iter=sample_iter,
        out_dir=out_dir,
        dataset_name=dataset_name,
        split_filters=split_filters,
        is_text_entity_fn=is_text_entity_fn,
        is_cvt_entity_fn=is_cvt_entity_fn,
        blacklist=blacklist,
        path_mode=str(raw_cfg.get("path_mode", "undirected")).strip().lower(),
        dedup_edges=bool(raw_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(raw_cfg.get("remove_self_loops", True)),
        emit_sub_filter=bool(raw_cfg.get("emit_sub_filter", True)),
        sub_filter_filename=str(raw_cfg.get("sub_filter_filename", "sub_filter.json")),
    )

    encoded = encode_preprocessed_features(
        prepared_samples=prepared_samples,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        embeddings_dir=embeddings_dir,
        encoder_name=encoder_name,
        device=device,
        precision=precision,
        batch_size=batch_size,
        progress_bar=bool(raw_cfg.get("progress_bar", True)),
        reuse_embeddings_if_exists=reuse_embeddings_if_exists,
    )

    materialize_preprocessed_data(
        prepared_samples=prepared_samples,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        encoded=encoded,
        embeddings_dir=embeddings_dir,
        overwrite_lmdb=bool(raw_cfg.get("overwrite_lmdb", False)),
        lmdb_shards=int(raw_cfg.get("lmdb_shards", 1)),
        map_size_gb=float(raw_cfg.get("map_size_gb", 128)),
    )


__all__ = ["run_preprocess_pipeline"]
