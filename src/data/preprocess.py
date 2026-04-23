from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import hydra
from omegaconf import DictConfig

from .preprocess_steps.graph_collect import collect_and_filter_graphs
from .preprocess_steps.samples import SplitFilter
from .preprocess_steps.source import iter_samples

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
            return default_result
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
        skip_no_train_target=bool(section.get("skip_no_train_target", False)),
    )

def _log_pipeline_configuration(
    dataset_name: str,
    out_dir: Path,
    embeddings_dir: Path,
    split_filters: Mapping[str, SplitFilter],
    raw_cfg: DictConfig,
) -> None:
    log.info(
        "Preprocess configuration resolved: "
        f"dataset={dataset_name}, "
        f"out_dir={out_dir}, "
        f"embeddings_dir={embeddings_dir}"
    )
    log.info(
        "Preprocess execution policy: full serial rebuild only "
        "(graph_collect -> text_encode -> materialize)."
    )
    log.info(
        "Graph preprocessing options: "
        f"dedup_edges={bool(raw_cfg.get('dedup_edges', True))}, "
        f"remove_self_loops={bool(raw_cfg.get('remove_self_loops', True))}, "
        f"emit_sub_filter={bool(raw_cfg.get('emit_sub_filter', True))}, "
        f"sub_filter_filename={str(raw_cfg.get('sub_filter_filename', 'sub_filter.json'))!r}"
    )
    log.info(
        "Materialization options: "
        f"expand_budget={int(raw_cfg.get('expand_budget', 0))}, "
        f"overwrite_lmdb={bool(raw_cfg.get('overwrite_lmdb', False))}, "
        f"lmdb_shards={int(raw_cfg.get('lmdb_shards', 1))}, "
        f"map_size_gb={float(raw_cfg.get('map_size_gb', 128))}"
    )
    log.info(
        "Encoder options: "
        f"model_name={str(raw_cfg.encoder.model_name)!r}, "
        f"device={str(raw_cfg.encoder.get('device', 'auto'))!r}, "
        f"batch_size={int(raw_cfg.encoder.batch_size)}, "
        f"progress_bar={bool(raw_cfg.get('progress_bar', True))}"
    )
    log.info(
        "Split filters: "
        + ", ".join(
            (
                f"{split}("
                f"skip_no_question_entity={flt.skip_no_question_entity}, "
                f"skip_no_ans={flt.skip_no_ans}, "
                f"skip_no_path={flt.skip_no_path}, "
                f"skip_no_train_target={flt.skip_no_train_target}"
                f")"
            )
            for split, flt in split_filters.items()
        )
    )


def run_preprocess_pipeline(raw_cfg: DictConfig) -> None:
    from .preprocess_steps.materialize import materialize_preprocessed_data
    from .preprocess_steps.text_encode import encode_preprocessed_features
    dataset_cfg = raw_cfg.dataset
    dataset_name = str(dataset_cfg.get("name", "")).strip()
    if not dataset_name:
        raise ValueError("dataset.name must be a non-empty string.")
    out_dir = _resolve_path(dataset_cfg.get("out_dir"))
    embeddings_dir = _resolve_path(dataset_cfg.paths.get("embeddings"))
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
    split_filters = {
        "train": _build_split_filter(raw_cfg.preprocess_filter.get("train")),
        "validation": _build_split_filter(
            raw_cfg.preprocess_filter.get("validation")
            or raw_cfg.preprocess_filter.get("eval")
        ),
        "test": _build_split_filter(
            raw_cfg.preprocess_filter.get("test")
            or raw_cfg.preprocess_filter.get("eval")
        ),
    }

    _log_pipeline_configuration(
        dataset_name=dataset_name,
        out_dir=out_dir,
        embeddings_dir=embeddings_dir,
        split_filters=split_filters,
        raw_cfg=raw_cfg,
    )

    sample_iter = iter_samples(
        dataset=dataset_name,
        splits=("train", "validation", "test"),
        column_map=dict(dataset_cfg.get("column_map", {})),
        dataset_source=str(dataset_cfg.get("dataset_source", "hf")).strip().lower(),
        hf_dataset=dataset_cfg.get("hf_dataset"),
        hf_cache_dir=raw_cfg.get("hf_env", {}).get("cache_dir"),
    )

    log.info("Stage 1/3: collect_and_filter_graphs")
    prepared_samples, entity_vocab, relation_vocab = collect_and_filter_graphs(
        sample_iter=sample_iter,
        out_dir=out_dir,
        dataset_name=dataset_name,
        split_filters=split_filters,
        is_text_entity_fn=is_text_entity_fn,
        is_cvt_entity_fn=is_cvt_entity_fn,
        dedup_edges=bool(raw_cfg.get("dedup_edges", True)),
        remove_self_loops=bool(raw_cfg.get("remove_self_loops", True)),
        emit_sub_filter=bool(raw_cfg.get("emit_sub_filter", True)),
        sub_filter_filename=str(raw_cfg.get("sub_filter_filename", "sub_filter.json")),
    )
    log.info(
        "Stage 1/3 complete: collected and prepared graphs for dataset=%s",
        dataset_name,
    )

    log.info("Stage 2/3: encode_preprocessed_features")
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
    log.info(
        "Stage 2/3 complete: encoded text features for dataset=%s",
        dataset_name,
    )

    log.info("Stage 3/3: materialize_preprocessed_data")
    materialize_preprocessed_data(
        prepared_samples=prepared_samples,
        split_filters=split_filters,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        payload=encoded_payload,
        embeddings_dir=embeddings_dir,
        expand_budget=int(raw_cfg.get("expand_budget", 0)),
        overwrite_lmdb=bool(raw_cfg.get("overwrite_lmdb", False)),
        lmdb_shards=int(raw_cfg.get("lmdb_shards", 1)),
        map_size_gb=float(raw_cfg.get("map_size_gb", 128)),
    )
    log.info(
        "Stage 3/3 complete: materialized dataset artifacts for dataset=%s",
        dataset_name,
    )

    log.info(
        "Successfully finished full preprocess pipeline for dataset=%s",
        dataset_name,
    )


__all__ = ["run_preprocess_pipeline"]