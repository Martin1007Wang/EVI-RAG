from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
import torch
from tqdm import tqdm

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap
from src.data.io.lmdb_utils import ensure_dir
from src.data.context import StageContext
from src.data.io.parquet_io import ParquetDatasetWriter, write_embedding_vocab, write_entity_vocab, write_relation_vocab
from src.data.io.raw_loader import build_cvt_entity_config, build_text_entity_config, build_time_relation_config, iter_samples
from src.data.relation_cleaning_rules import (
    DEFAULT_RELATION_CLEANING_RULES,
    RELATION_ACTION_DROP,
    RelationCleaningRules,
    relation_action,
)
from src.data.schema.constants import (
    _ALLOWED_SPLITS,
    _DISABLE_PARALLEL_WORKERS,
    _EDGE_INDEX_MIN,
    _EDGE_STAT_KEYS,
    _FILTER_STAT_KEYS,
    _INVERSE_RELATION_SUFFIX_DEFAULT,
    _PATH_MODE_QA_DIRECTED,
    _PATH_MODE_UNDIRECTED,
    _REL_LABEL_SAMPLE_LIMIT,
    _REMOVE_SELF_LOOPS_DEFAULT,
    _VALIDATE_GRAPH_EDGES_DEFAULT,
    _ZERO,
)
from src.data.schema.types import (
    EntityLookup,
    EntityVocab,
    GraphRecord,
    RelationLookup,
    RelationVocab,
    Sample,
    TimeRelationConfig,
)
from src.data.stages.step1_vocab import _partition_graph_edges, _resolve_split_filter, _should_keep_sample
from src.data.utils.connectivity import _validate_path_mode, has_connectivity
from src.data.utils.stats import _init_split_counters, _safe_div, _sample_labels
from src.data.utils.validation import _validate_split_names
from src.utils.logging_utils import log_event


@dataclass(frozen=True)
class _WorkerState:
    entity_lookup: EntityLookup
    relation_lookup: RelationLookup
    dedup_edges: bool
    validate_graph_edges: bool
    remove_self_loops: bool
    relation_cleaning_enabled: bool
    relation_cleaning_rules: RelationCleaningRules
    inverse_relations_key_map: Optional[Dict[str, str]]
    inverse_relations_suffix: str
    time_relation_cfg: Optional[TimeRelationConfig]


_WORKER_STATE: Optional[_WorkerState] = None

_INVERSE_RELATIONS_CFG_KEY = "inverse_relations"
_INVERSE_RELATIONS_ENABLED_KEY = "enabled"
_INVERSE_RELATIONS_MAPPING_KEY = "mapping_path"
_INVERSE_RELATIONS_SUFFIX_KEY = "kg_id_suffix"
_INVERSE_RELATIONS_STRICT_KEY = "strict"
_INVERSE_RELATIONS_FALLBACK_KEY = "fallback_template"
_INVERSE_RELATIONS_GLOBAL_VOCAB_KEY = "global_vocab"
_INVERSE_RELATIONS_LIST_KEY = "inverse_relations"
_INVERSE_RELATIONS_FORWARD_KEY = "forward"
_INVERSE_RELATIONS_FORWARD_LABEL_KEY = "forward_label"
_INVERSE_RELATIONS_FORWARD_TEXT_KEY = "forward_text"
_INVERSE_RELATIONS_INVERSE_KEY = "inverse"
_INVERSE_RELATIONS_INVERSE_TEXT_KEY = "inverse_text"


@dataclass(frozen=True)
class _RelationTextMap:
    forward_labels: Dict[str, str]
    inverse_labels: Dict[str, str]


def _init_worker_state(state: _WorkerState) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _require_worker_state() -> _WorkerState:
    if _WORKER_STATE is None:
        raise RuntimeError("Worker state is not initialized.")
    return _WORKER_STATE


def _build_graph_worker(sample: Sample) -> GraphRecord:
    state = _require_worker_state()
    graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
    return build_graph(
        sample,
        state.entity_lookup,
        state.relation_lookup,
        graph_id,
        dedup_edges=state.dedup_edges,
        validate_graph_edges=state.validate_graph_edges,
        remove_self_loops=state.remove_self_loops,
        relation_cleaning_enabled=state.relation_cleaning_enabled,
        relation_cleaning_rules=state.relation_cleaning_rules,
        inverse_relations_key_map=state.inverse_relations_key_map,
        inverse_relations_suffix=state.inverse_relations_suffix,
        time_relation_cfg=state.time_relation_cfg,
    )


def _validate_graph_record(graph: GraphRecord) -> None:
    num_nodes = len(graph.node_entity_ids)
    num_edges = len(graph.edge_src)
    if len(graph.edge_dst) != num_edges or len(graph.edge_relation_ids) != num_edges:
        raise ValueError(f"Edge length mismatch for {graph.graph_id}: edges={num_edges}.")
    if num_edges > _EDGE_INDEX_MIN:
        min_src = min(graph.edge_src)
        min_dst = min(graph.edge_dst)
        max_src = max(graph.edge_src)
        max_dst = max(graph.edge_dst)
        if min_src < _EDGE_INDEX_MIN or min_dst < _EDGE_INDEX_MIN:
            raise ValueError(f"Negative edge index detected for {graph.graph_id}.")
        if max_src >= num_nodes or max_dst >= num_nodes:
            raise ValueError(f"Edge index exceeds num_nodes for {graph.graph_id}.")


def _dedup_directed_edges(edges: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    if not edges:
        return []
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Tuple[str, str, str]] = []
    for head, rel, tail in edges:
        key = (head, rel, tail)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _build_inverse_relation_key(rel: str, suffix: str) -> str:
    if not suffix:
        raise ValueError("inverse_relations.kg_id_suffix must be non-empty.")
    return f"{rel}{suffix}"


def _resolve_forward_label(entry: Mapping[str, object], forward_key: str) -> str:
    forward_label = entry.get(_INVERSE_RELATIONS_FORWARD_LABEL_KEY)
    if forward_label is None:
        forward_label = entry.get(_INVERSE_RELATIONS_FORWARD_TEXT_KEY)
    return forward_key if forward_label is None else str(forward_label)


def _resolve_inverse_label(entry: Mapping[str, object], *, context: str) -> str:
    inverse_label = entry.get(_INVERSE_RELATIONS_INVERSE_KEY)
    if inverse_label is None:
        inverse_label = entry.get(_INVERSE_RELATIONS_INVERSE_TEXT_KEY)
    if not inverse_label:
        raise ValueError(f"{context} entry missing inverse label.")
    return str(inverse_label)


def _update_relation_texts(
    mapping: _RelationTextMap,
    *,
    forward_key: str,
    forward_label: str,
    inverse_label: str,
    context: str,
) -> None:
    existing_forward = mapping.forward_labels.get(forward_key)
    if existing_forward is not None and existing_forward != forward_label:
        raise ValueError(f"{context} duplicate mismatch for {forward_key!r} forward label.")
    existing_inverse = mapping.inverse_labels.get(forward_key)
    if existing_inverse is not None and existing_inverse != inverse_label:
        raise ValueError(f"{context} duplicate mismatch for {forward_key!r} inverse label.")
    mapping.forward_labels[forward_key] = forward_label
    mapping.inverse_labels[forward_key] = inverse_label


def _parse_inverse_relations_list(items: Sequence[object], *, context: str) -> _RelationTextMap:
    mapping = _RelationTextMap(forward_labels={}, inverse_labels={})
    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"{context} entry {idx} must be a dict with forward/inverse fields.")
        forward = raw.get(_INVERSE_RELATIONS_FORWARD_KEY)
        if not forward:
            raise ValueError(f"{context} entry {idx} missing forward field.")
        forward_key = str(forward)
        forward_label = _resolve_forward_label(raw, forward_key)
        inverse_label = _resolve_inverse_label(raw, context=f"{context} entry {idx}")
        _update_relation_texts(
            mapping,
            forward_key=forward_key,
            forward_label=forward_label,
            inverse_label=inverse_label,
            context=context,
        )
    return mapping


def _parse_inverse_relations_dict(mapping: Mapping[object, object], *, context: str) -> _RelationTextMap:
    parsed = _RelationTextMap(forward_labels={}, inverse_labels={})
    for raw_key, raw_val in mapping.items():
        forward_key = str(raw_key)
        if isinstance(raw_val, Mapping):
            raw_forward = raw_val.get(_INVERSE_RELATIONS_FORWARD_KEY)
            if raw_forward is not None and str(raw_forward) != forward_key:
                raise ValueError(f"{context} entry forward key mismatch for {forward_key!r}.")
            forward_label = _resolve_forward_label(raw_val, forward_key)
            inverse_label = _resolve_inverse_label(raw_val, context=f"{context} entry {forward_key!r}")
        else:
            forward_label = forward_key
            if raw_val is None or raw_val == "":
                raise ValueError(f"{context} entry {forward_key!r} missing inverse label.")
            inverse_label = str(raw_val)
        _update_relation_texts(
            parsed,
            forward_key=forward_key,
            forward_label=forward_label,
            inverse_label=inverse_label,
            context=context,
        )
    return parsed


def _parse_inverse_relations_payload(payload: object, *, context: str) -> _RelationTextMap:
    if isinstance(payload, dict):
        inner = payload.get(_INVERSE_RELATIONS_LIST_KEY, payload)
        if isinstance(inner, dict):
            return _parse_inverse_relations_dict(inner, context=context)
        if isinstance(inner, list):
            return _parse_inverse_relations_list(inner, context=context)
    if isinstance(payload, list):
        return _parse_inverse_relations_list(payload, context=context)
    raise ValueError(f"{context} must be a dict or list.")


def _apply_inverse_relations_fallback(
    mapping: _RelationTextMap,
    missing: Sequence[str],
    *,
    fallback: str,
) -> None:
    for rel in missing:
        mapping.forward_labels.setdefault(rel, rel)
        mapping.inverse_labels.setdefault(rel, fallback.format(relation=rel))


def _resolve_relation_vocab_labels(
    mapping: _RelationTextMap,
    dataset_rel_labels: Sequence[str],
    *,
    strict: bool,
    fallback: Optional[str],
    global_vocab: bool,
    context: str,
) -> List[str]:
    dataset_set = set(dataset_rel_labels)
    mapping_keys = set(mapping.inverse_labels)
    if not global_vocab:
        unknown = [rel for rel in mapping_keys if rel not in dataset_set]
        if unknown:
            preview = ", ".join(unknown[:5])
            raise ValueError(f"{context} includes unknown relations, examples: {preview}.")
    missing = [rel for rel in dataset_rel_labels if rel not in mapping_keys]
    if missing:
        if strict:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{context} missing {len(missing)} relations, examples: {preview}.")
        if not isinstance(fallback, str):
            raise ValueError("inverse_relations.fallback_template must be set when strict=false.")
        _apply_inverse_relations_fallback(mapping, missing, fallback=fallback)
        mapping_keys = set(mapping.inverse_labels)
    if global_vocab:
        return sorted(mapping_keys)
    return sorted(dataset_rel_labels)


def _load_inverse_relations_map(
    cfg: object,
    ctx: StageContext,
    *,
    dataset_rel_labels: Sequence[str],
) -> tuple[Optional[_RelationTextMap], str, List[str], bool]:
    inv_cfg = cfg.get(_INVERSE_RELATIONS_CFG_KEY) if hasattr(cfg, "get") else None
    if not isinstance(inv_cfg, Mapping):
        return None, _INVERSE_RELATION_SUFFIX_DEFAULT, sorted(dataset_rel_labels), False
    if not bool(inv_cfg.get(_INVERSE_RELATIONS_ENABLED_KEY, False)):
        return None, _INVERSE_RELATION_SUFFIX_DEFAULT, sorted(dataset_rel_labels), False
    mapping_path = inv_cfg.get(_INVERSE_RELATIONS_MAPPING_KEY)
    if not mapping_path:
        raise ValueError("inverse_relations.mapping_path must be set when inverse_relations.enabled=true.")
    path = ctx.resolve_path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"inverse_relations mapping not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = _parse_inverse_relations_payload(payload, context=str(path))
    strict = bool(inv_cfg.get(_INVERSE_RELATIONS_STRICT_KEY, True))
    fallback = inv_cfg.get(_INVERSE_RELATIONS_FALLBACK_KEY)
    global_vocab = bool(inv_cfg.get(_INVERSE_RELATIONS_GLOBAL_VOCAB_KEY, False))
    forward_rel_labels = _resolve_relation_vocab_labels(
        mapping,
        dataset_rel_labels,
        strict=strict,
        fallback=fallback,
        global_vocab=global_vocab,
        context="inverse_relations",
    )
    suffix = str(inv_cfg.get(_INVERSE_RELATIONS_SUFFIX_KEY, _INVERSE_RELATION_SUFFIX_DEFAULT))
    return mapping, suffix, forward_rel_labels, global_vocab


def _build_inverse_relation_key_map(
    mapping: Dict[str, str],
    *,
    suffix: str,
    forward_rel_labels: Sequence[str],
) -> Dict[str, str]:
    forward_set = set(forward_rel_labels)
    key_map: Dict[str, str] = {}
    for rel, label in mapping.items():
        inv_key = _build_inverse_relation_key(rel, suffix)
        if inv_key in forward_set:
            raise ValueError(f"inverse_relations key collision with forward relation: {inv_key!r}.")
        if inv_key in key_map and key_map[inv_key] != label:
            raise ValueError(f"inverse_relations duplicate key mismatch for {inv_key!r}.")
        key_map[inv_key] = label
    return key_map


def _expand_edges_with_inverse(
    edges: Sequence[Tuple[str, str, str]],
    inverse_relations_key_map: Dict[str, str],
    *,
    suffix: str,
) -> List[Tuple[str, str, str]]:
    if not edges:
        return []
    expanded = list(edges)
    for head, rel, tail in edges:
        inv_rel = _build_inverse_relation_key(rel, suffix)
        if inv_rel not in inverse_relations_key_map:
            raise ValueError(f"inverse_relations missing label for {rel!r}.")
        expanded.append((tail, inv_rel, head))
    return expanded


def _normalize_embeddings(embeddings: torch.Tensor, eps: float) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings
    denom = embeddings.norm(dim=-1, keepdim=True).clamp(min=eps)
    return embeddings / denom


def _canonicalize_graph_edges(
    graph: GraphRecord,
    question_embedding_norm: torch.Tensor,
    relation_embeddings_norm: torch.Tensor,
) -> None:
    raise ValueError(
        "canonicalize_relations requires offline shortest-path labels; disable canonicalize_relations for GFlowNet."
    )


def preprocess(ctx: StageContext) -> None:
    cfg = ctx.cfg
    logger = ctx.logger
    dataset = ctx.dataset_name
    kb = cfg.kb
    out_dir = ctx.out_dir
    column_map = dict(cfg.column_map)
    entity_normalization = cfg.entity_normalization
    text_cfg = build_text_entity_config(cfg)
    cvt_cfg = build_cvt_entity_config(cfg)
    time_relation_cfg = build_time_relation_config(cfg)
    dataset_family = cfg.get("dataset_family")
    dataset_source = str(cfg.get("dataset_source", "hf")).strip().lower()
    hf_dataset = cfg.get("hf_dataset")
    hf_cache_dir_cfg = cfg.get("hf_cache_dir")
    hf_cache_dir = ctx.resolve_path(hf_cache_dir_cfg) if hf_cache_dir_cfg else None
    hf_offline = bool(cfg.get("hf_offline", False))
    if dataset_source != "hf":
        raise ValueError("dataset_source must be 'hf'; raw parquet ingestion is disabled.")
    train_filter, eval_filter, override_filters = ctx.split_filters
    path_mode = _validate_path_mode(str(cfg.get("path_mode", _PATH_MODE_UNDIRECTED)))
    dedup_edges = bool(cfg.get("dedup_edges", True))
    validate_graph_edges = bool(cfg.get("validate_graph_edges", _VALIDATE_GRAPH_EDGES_DEFAULT))
    remove_self_loops = bool(cfg.get("remove_self_loops", _REMOVE_SELF_LOOPS_DEFAULT))
    relation_cleaning_enabled = bool(cfg.get("relation_cleaning", True))
    relation_cleaning_rules = DEFAULT_RELATION_CLEANING_RULES
    embedding_cfg = ctx.embedding_cfg
    if embedding_cfg is not None and embedding_cfg.canonicalize_relations:
        raise ValueError("canonicalize_relations requires offline labels; disable it for GFlowNet.")
    emit_sub_filter = bool(cfg.get("emit_sub_filter", False))
    sub_filter_filename = str(cfg.get("sub_filter_filename", "sub_filter.json"))
    emit_nonzero_positive_filter = bool(cfg.get("emit_nonzero_positive_filter", False))
    nonzero_positive_filter_filename = str(cfg.get("nonzero_positive_filter_filename", "nonzero_positive_filter.json"))
    nonzero_positive_filter_splits = cfg.get("nonzero_positive_filter_splits")
    parquet_chunk_size = ctx.parquet_chunk_size
    parquet_num_workers = ctx.parquet_num_workers
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb, text_cfg=text_cfg, cvt_cfg=cvt_cfg)

    splits = list(_ALLOWED_SPLITS)
    connectivity_cache: Dict[Tuple[str, str, str], bool] = {}
    total_by_split: Dict[str, int] = {}
    kept_by_split: Dict[str, int] = {}
    sub_by_split: Dict[str, int] = {}
    empty_graph_by_split: Dict[str, int] = {}
    empty_graph_ids: List[str] = []
    empty_graph_id_set: Set[str] = set()
    sub_sample_ids: List[str] = []
    edge_stats = _init_split_counters(splits, _EDGE_STAT_KEYS)
    filter_stats = _init_split_counters(splits, _FILTER_STAT_KEYS)
    kept_rel_labels: Set[str] = set()
    type_rel_labels: Set[str] = set()
    dropped_rel_labels: Set[str] = set()
    graphs_written_by_split = {split: _ZERO for split in splits}
    questions_written_by_split = {split: _ZERO for split in splits}
    if emit_nonzero_positive_filter:
        raise ValueError("emit_nonzero_positive_filter is disabled in GFlowNet; remove this flag.")

    log_event(
        logger,
        "preprocess_start",
        dataset=dataset,
        kb=kb,
        splits=splits,
        dataset_source=dataset_source,
        hf_dataset=hf_dataset,
        path_mode=path_mode,
        dedup_edges=dedup_edges,
        remove_self_loops=remove_self_loops,
        relation_cleaning=relation_cleaning_enabled,
        parquet_chunk_size=parquet_chunk_size,
        parquet_num_workers=parquet_num_workers,
    )
    if relation_cleaning_enabled:
        log_event(
            logger,
            "relation_cleaning_rules",
            type_exact=sorted(relation_cleaning_rules.type_exact),
            type_prefixes=sorted(relation_cleaning_rules.type_prefixes),
            type_regexes=sorted(pattern.pattern for pattern in relation_cleaning_rules.type_regexes),
            drop_exact=sorted(relation_cleaning_rules.drop_exact),
            drop_prefixes=sorted(relation_cleaning_rules.drop_prefixes),
            drop_regexes=sorted(pattern.pattern for pattern in relation_cleaning_rules.drop_regexes),
        )
    log_event(
        logger,
        "time_relation_filter",
        mode=time_relation_cfg.mode,
        relation_regex=None if time_relation_cfg.relation_regex is None else time_relation_cfg.relation_regex.pattern,
        question_regex=None if time_relation_cfg.question_regex is None else time_relation_cfg.question_regex.pattern,
    )

    log_event(logger, "vocab_start", stage="vocab")
    for sample in tqdm(
        iter_samples(
            dataset,
            kb,
            None,
            splits,
            column_map,
            entity_normalization,
            dataset_source=dataset_source,
            dataset_family=dataset_family,
            hf_dataset=hf_dataset,
            hf_cache_dir=hf_cache_dir,
            hf_offline=hf_offline,
        ),
        desc=f"Vocab from {dataset}",
    ):
        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        total_by_split[sample.split] = total_by_split.get(sample.split, 0) + 1
        kept_edges, type_edges = _partition_graph_edges(
            sample.graph,
            relation_cleaning_rules,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            time_relation_cfg=time_relation_cfg,
            question_text=sample.question,
        )
        kept_edges = _dedup_directed_edges(kept_edges)
        split_key = sample.split
        raw_edges = len(sample.graph)
        self_loop_edges = _ZERO
        if remove_self_loops:
            self_loop_edges = sum(1 for h, _, t in sample.graph if h == t)
        kept_edges_count = len(kept_edges)
        type_edges_count = len(type_edges)
        dropped_edges = raw_edges - self_loop_edges - kept_edges_count - type_edges_count
        raw_nodes = len({h for h, _, _ in sample.graph} | {t for _, _, t in sample.graph})
        kept_node_set = {h for h, _, _ in kept_edges} | {t for _, _, t in kept_edges}
        kept_nodes = len(kept_node_set)
        type_orphan_edges = sum(1 for h, _, _ in type_edges if h not in kept_node_set)
        edge_stats[split_key]["raw_edges"] += raw_edges
        edge_stats[split_key]["self_loop_edges"] += self_loop_edges
        edge_stats[split_key]["kept_edges"] += kept_edges_count
        edge_stats[split_key]["type_edges"] += type_edges_count
        edge_stats[split_key]["dropped_edges"] += dropped_edges
        edge_stats[split_key]["raw_nodes"] += raw_nodes
        edge_stats[split_key]["kept_nodes"] += kept_nodes
        edge_stats[split_key]["type_orphan_edges"] += type_orphan_edges
        for _, rel, _ in kept_edges:
            kept_rel_labels.add(rel)
        for _, rel, _ in type_edges:
            type_rel_labels.add(rel)
        for h, rel, t in sample.graph:
            if remove_self_loops and h == t:
                continue
            if relation_action(rel, relation_cleaning_rules, enabled=relation_cleaning_enabled) == RELATION_ACTION_DROP:
                dropped_rel_labels.add(rel)
        if not kept_edges:
            empty_graph_by_split[sample.split] = empty_graph_by_split.get(sample.split, 0) + 1
            empty_graph_id_set.add(graph_id)
            if len(empty_graph_ids) < 20:
                empty_graph_ids.append(graph_id)
            continue
        for h, r, t in kept_edges:
            entity_vocab.add_entity(h)
            entity_vocab.add_entity(t)
        for h, _, t in type_edges:
            entity_vocab.add_entity(h)
            entity_vocab.add_entity(t)
        for ent in sample.q_entity + sample.a_entity:
            entity_vocab.add_entity(ent)

        split_filter = _resolve_split_filter(sample.split, train_filter, eval_filter, override_filters)
        outcome = _should_keep_sample(
            sample,
            split_filter,
            connectivity_cache,
            path_mode=path_mode,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            relation_cleaning_rules=relation_cleaning_rules,
            kept_edges=kept_edges,
        )
        if outcome.keep:
            kept_by_split[sample.split] = kept_by_split.get(sample.split, 0) + 1
        else:
            if split_filter.skip_no_topic and not outcome.has_topic:
                filter_stats[sample.split]["dropped_no_topic"] += 1
            if split_filter.skip_no_ans and not outcome.has_answer:
                filter_stats[sample.split]["dropped_no_answer"] += 1
            if split_filter.skip_no_path and not outcome.has_path:
                filter_stats[sample.split]["dropped_no_path"] += 1

    entity_vocab.finalize()
    relation_vocab = RelationVocab(kb=kb)
    dataset_rel_labels = sorted(kept_rel_labels)
    relation_texts, inverse_relations_suffix, forward_rel_labels, inverse_global_vocab = _load_inverse_relations_map(
        cfg,
        ctx,
        dataset_rel_labels=dataset_rel_labels,
    )
    inverse_relations_key_map = None
    forward_label_map: Optional[Dict[str, str]] = None
    if relation_texts is not None:
        inverse_relations_key_map = _build_inverse_relation_key_map(
            relation_texts.inverse_labels,
            suffix=inverse_relations_suffix,
            forward_rel_labels=forward_rel_labels,
        )
        forward_label_map = relation_texts.forward_labels
    for rel in forward_rel_labels:
        label = rel if forward_label_map is None else forward_label_map.get(rel)
        if label is None:
            raise ValueError(f"inverse_relations missing forward label for {rel!r}.")
        relation_vocab.relation_id(rel, label=label)
    if inverse_relations_key_map is not None:
        for rel in forward_rel_labels:
            inv_key = _build_inverse_relation_key(rel, inverse_relations_suffix)
            inv_label = inverse_relations_key_map[inv_key]
            relation_vocab.relation_id(inv_key, label=inv_label)

    relation_lookup = relation_vocab.to_lookup()

    entity_count = len(entity_vocab.struct_records)
    text_entity_count = len(entity_vocab.embedding_records)
    relation_count = len(relation_vocab.records)
    log_event(
        logger,
        "vocab_summary",
        entity_count=entity_count,
        text_entity_count=text_entity_count,
        non_text_entity_count=entity_count - text_entity_count,
        relation_count=relation_count,
    )
    if inverse_relations_key_map is not None:
        log_event(
            logger,
            "inverse_relations_loaded",
            count=len(inverse_relations_key_map),
            suffix=inverse_relations_suffix,
            global_vocab=inverse_global_vocab,
        )
    total_edges_raw = sum(edge_stats[split]["raw_edges"] for split in splits)
    total_edges_kept = sum(edge_stats[split]["kept_edges"] for split in splits)
    total_edges_type = sum(edge_stats[split]["type_edges"] for split in splits)
    total_edges_drop = sum(edge_stats[split]["dropped_edges"] for split in splits)
    total_edges_self = sum(edge_stats[split]["self_loop_edges"] for split in splits)
    total_type_orphan = sum(edge_stats[split]["type_orphan_edges"] for split in splits)
    log_event(
        logger,
        "edge_summary_total",
        raw_edges=total_edges_raw,
        kept_edges=total_edges_kept,
        type_edges=total_edges_type,
        dropped_edges=total_edges_drop,
        self_loop_edges=total_edges_self,
        type_orphan_edges=total_type_orphan,
    )
    for split in splits:
        split_total = total_by_split.get(split, _ZERO)
        log_event(
            logger,
            "split_summary",
            split=split,
            samples_total=split_total,
            samples_kept=kept_by_split.get(split, _ZERO),
            samples_empty_graph=empty_graph_by_split.get(split, _ZERO),
            dropped_no_topic=filter_stats[split]["dropped_no_topic"],
            dropped_no_answer=filter_stats[split]["dropped_no_answer"],
            dropped_no_path=filter_stats[split]["dropped_no_path"],
            raw_edges=edge_stats[split]["raw_edges"],
            kept_edges=edge_stats[split]["kept_edges"],
            type_edges=edge_stats[split]["type_edges"],
            dropped_edges=edge_stats[split]["dropped_edges"],
            self_loop_edges=edge_stats[split]["self_loop_edges"],
            type_orphan_edges=edge_stats[split]["type_orphan_edges"],
            avg_raw_edges=_safe_div(edge_stats[split]["raw_edges"], split_total),
            avg_kept_edges=_safe_div(edge_stats[split]["kept_edges"], split_total),
            avg_type_edges=_safe_div(edge_stats[split]["type_edges"], split_total),
            avg_raw_nodes=_safe_div(edge_stats[split]["raw_nodes"], split_total),
            avg_kept_nodes=_safe_div(edge_stats[split]["kept_nodes"], split_total),
        )
    log_event(
        logger,
        "relation_label_stats",
        kept_relation_types=len(kept_rel_labels),
        type_relation_types=len(type_rel_labels),
        dropped_relation_types=len(dropped_rel_labels),
        kept_relation_examples=_sample_labels(kept_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT),
        type_relation_examples=_sample_labels(type_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT),
        dropped_relation_examples=_sample_labels(dropped_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT),
    )

    def _format_counts(counts: Dict[str, int]) -> str:
        return ", ".join(f"{s}={counts.get(s, 0)}" for s in splits)

    log_event(logger, "samples_total", counts=_format_counts(total_by_split))
    log_event(logger, "samples_kept", counts=_format_counts(kept_by_split))
    if empty_graph_by_split:
        log_event(logger, "samples_empty_graph", counts=_format_counts(empty_graph_by_split))
        if empty_graph_ids:
            log_event(logger, "empty_graph_examples", examples=empty_graph_ids)

    encoder: Optional[TextEncoder] = None
    relation_embeddings_norm: Optional[torch.Tensor] = None
    if embedding_cfg is not None:
        embeddings_out_dir = embedding_cfg.embeddings_out_dir
        entity_emb_path = embeddings_out_dir / "entity_embeddings.pt"
        relation_emb_path = embeddings_out_dir / "relation_embeddings.pt"
        need_entity_encode = embedding_cfg.precompute_entities and not (
            reuse_embeddings_if_exists and entity_emb_path.exists()
        )
        need_relation_encode = (embedding_cfg.precompute_relations or embedding_cfg.canonicalize_relations) and not (
            reuse_embeddings_if_exists and relation_emb_path.exists()
        )
        need_question_encode = embedding_cfg.precompute_questions or embedding_cfg.canonicalize_relations
        need_encoder = need_entity_encode or need_relation_encode or need_question_encode
        if need_encoder:
            encoder = TextEncoder(
                embedding_cfg.encoder,
                embedding_cfg.device,
                embedding_cfg.fp16,
                embedding_cfg.progress_bar,
            )
        if embedding_cfg.precompute_entities or embedding_cfg.precompute_relations:
            ensure_dir(embeddings_out_dir)
        if embedding_cfg.precompute_entities:
            if not need_entity_encode:
                log_event(logger, "preprocess_reuse_entity_embeddings", path=str(entity_emb_path))
            else:
                emb_rows = sorted(
                    ((rec["embedding_id"], rec.get("label", "")) for rec in entity_vocab.embedding_records),
                    key=lambda x: x[0],
                )
                text_labels = [str(label) for _, label in emb_rows]
                text_ids = [int(eid) for eid, _ in emb_rows]
                struct_records = entity_vocab.struct_records
                max_embedding_id = max((int(rec["embedding_id"]) for rec in struct_records), default=0)
                encode_to_memmap(
                    encoder=encoder,
                    texts=text_labels,
                    emb_ids=text_ids,
                    batch_size=embedding_cfg.batch_size,
                    max_embedding_id=max_embedding_id,
                    out_path=entity_emb_path,
                    desc="Entities",
                    show_progress=embedding_cfg.progress_bar,
                )
        if embedding_cfg.precompute_relations or embedding_cfg.canonicalize_relations:
            if need_relation_encode:
                relation_rows = sorted(
                    ((rec["relation_id"], rec.get("label", "")) for rec in relation_vocab.records),
                    key=lambda x: x[0],
                )
                relation_labels = [str(label) for _, label in relation_rows]
                relation_emb = encoder.encode(
                    relation_labels,
                    embedding_cfg.batch_size,
                    show_progress=embedding_cfg.progress_bar,
                    desc="Relations",
                )
                if embedding_cfg.precompute_relations:
                    torch.save(relation_emb, relation_emb_path)
            else:
                log_event(logger, "preprocess_reuse_relation_embeddings", path=str(relation_emb_path))
                relation_emb = torch.load(relation_emb_path, map_location="cpu")
            if embedding_cfg.canonicalize_relations:
                relation_embeddings_norm = _normalize_embeddings(relation_emb, embedding_cfg.cosine_eps)
                if relation_embeddings_norm.numel() == 0:
                    raise ValueError("relation_embeddings are empty; cannot canonicalize positives.")

    log_event(logger, "graphs_questions_start", stage="graphs_questions")
    chunk_size = parquet_chunk_size
    include_question_emb = bool(embedding_cfg and embedding_cfg.precompute_questions)
    base_writer = ParquetDatasetWriter(out_dir=out_dir, include_question_emb=include_question_emb)
    need_question_emb = bool(embedding_cfg and (embedding_cfg.precompute_questions or embedding_cfg.canonicalize_relations))

    def _process_sample_batch(samples: List[Sample], executor: Optional[ProcessPoolExecutor]) -> None:
        if not samples:
            return
        question_emb_batch = None
        question_emb_norm_batch = None
        if need_question_emb:
            if encoder is None:
                raise RuntimeError("Question embeddings requested but encoder is not configured.")
            question_texts = [sample.question for sample in samples]
            question_emb_batch = encoder.encode(
                question_texts,
                embedding_cfg.batch_size,
                show_progress=False,
                desc="Questions",
            )
            if embedding_cfg and embedding_cfg.canonicalize_relations:
                question_emb_norm_batch = _normalize_embeddings(question_emb_batch, embedding_cfg.cosine_eps)
        if executor is None:
            graphs = [
                build_graph(
                    sample,
                    entity_vocab,
                    relation_lookup,
                    f"{sample.dataset}/{sample.split}/{sample.question_id}",
                    dedup_edges=dedup_edges,
                    validate_graph_edges=validate_graph_edges,
                    remove_self_loops=remove_self_loops,
                    relation_cleaning_enabled=relation_cleaning_enabled,
                    relation_cleaning_rules=relation_cleaning_rules,
                    inverse_relations_key_map=inverse_relations_key_map,
                    inverse_relations_suffix=inverse_relations_suffix,
                    time_relation_cfg=time_relation_cfg,
                )
                for sample in samples
            ]
        else:
            graphs = list(executor.map(_build_graph_worker, samples))
        for idx, (sample, graph) in enumerate(zip(samples, graphs)):
            if embedding_cfg and embedding_cfg.canonicalize_relations:
                if relation_embeddings_norm is None or question_emb_norm_batch is None:
                    raise RuntimeError("Canonicalization requested but embeddings are missing.")
                _canonicalize_graph_edges(graph, question_emb_norm_batch[idx], relation_embeddings_norm)
            question_emb = None
            if embedding_cfg and embedding_cfg.precompute_questions:
                if question_emb_batch is None:
                    raise RuntimeError("question_emb batch missing while precompute_questions is enabled.")
                question_emb = question_emb_batch[idx].tolist()
            question = build_question_record(sample, entity_vocab, graph.graph_id, question_emb=question_emb)
            base_writer.append(graph, question)
            graphs_written_by_split[sample.split] += 1
            questions_written_by_split[sample.split] += 1
            if emit_sub_filter:
                label_to_idx = {label: idx for idx, label in enumerate(graph.node_labels)}
                q_local = {label_to_idx[ent] for ent in sample.q_entity if ent in label_to_idx}
                a_local = {label_to_idx[ent] for ent in sample.a_entity if ent in label_to_idx}
                has_topic = bool(q_local)
                has_answer = bool(a_local)
                if has_topic and has_answer:
                    cleaned_edges, _ = _partition_graph_edges(
                        sample.graph,
                        relation_cleaning_rules,
                        remove_self_loops=remove_self_loops,
                        relation_cleaning_enabled=relation_cleaning_enabled,
                        time_relation_cfg=time_relation_cfg,
                        question_text=sample.question,
                    )
                    cleaned_edges = _dedup_directed_edges(cleaned_edges)
                    if inverse_relations_key_map is not None:
                        cleaned_edges = _expand_edges_with_inverse(
                            cleaned_edges,
                            inverse_relations_key_map,
                            suffix=inverse_relations_suffix,
                        )
                    has_path = has_connectivity(cleaned_edges, sample.q_entity, sample.a_entity, path_mode=path_mode)
                else:
                    has_path = False
                if has_topic and has_answer and has_path:
                    sub_sample_ids.append(graph.graph_id)
                    sub_by_split[sample.split] = sub_by_split.get(sample.split, 0) + 1
            if len(base_writer.graphs) >= chunk_size or len(base_writer.questions) >= chunk_size:
                base_writer.flush()

    def _run_pass2(executor: Optional[ProcessPoolExecutor]) -> None:
        pending_samples: List[Sample] = []
        for sample in tqdm(
            iter_samples(
                dataset,
                kb,
                None,
                splits,
                column_map,
                entity_normalization,
                dataset_source=dataset_source,
                dataset_family=dataset_family,
                hf_dataset=hf_dataset,
                hf_cache_dir=hf_cache_dir,
                hf_offline=hf_offline,
            ),
            desc=f"Graphs from {dataset}",
        ):
            graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
            if graph_id in empty_graph_id_set:
                continue
            split_filter = _resolve_split_filter(sample.split, train_filter, eval_filter, override_filters)
            outcome = _should_keep_sample(
                sample,
                split_filter,
                connectivity_cache,
                path_mode=path_mode,
                remove_self_loops=remove_self_loops,
                relation_cleaning_enabled=relation_cleaning_enabled,
                relation_cleaning_rules=relation_cleaning_rules,
                time_relation_cfg=time_relation_cfg,
            )
            if not outcome.keep:
                continue
            pending_samples.append(sample)
            if len(pending_samples) >= chunk_size:
                _process_sample_batch(pending_samples, executor)
                pending_samples = []

        if pending_samples:
            _process_sample_batch(pending_samples, executor)

    if parquet_num_workers > _DISABLE_PARALLEL_WORKERS:
        worker_state = _WorkerState(
            entity_lookup=entity_vocab.to_lookup(),
            relation_lookup=relation_lookup,
            dedup_edges=dedup_edges,
            validate_graph_edges=validate_graph_edges,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            relation_cleaning_rules=relation_cleaning_rules,
            inverse_relations_key_map=inverse_relations_key_map,
            inverse_relations_suffix=inverse_relations_suffix,
            time_relation_cfg=time_relation_cfg,
        )
        with ProcessPoolExecutor(
            max_workers=parquet_num_workers,
            initializer=_init_worker_state,
            initargs=(worker_state,),
        ) as executor:
            _run_pass2(executor)
    else:
        _run_pass2(None)

    base_writer.close()
    log_event(
        logger,
        "graphs_questions_written",
        graphs_by_split=graphs_written_by_split,
        questions_by_split=questions_written_by_split,
    )

    write_entity_vocab(entity_vocab.struct_records, out_dir / "entity_vocab.parquet")
    write_embedding_vocab(entity_vocab.embedding_records, out_dir / "embedding_vocab.parquet")
    write_relation_vocab(relation_vocab.records, out_dir / "relation_vocab.parquet")

    if emit_sub_filter:
        sub_payload = {
            "dataset": dataset,
            "sample_ids": sorted(sub_sample_ids),
        }
        (out_dir / sub_filter_filename).write_text(json.dumps(sub_payload, indent=2))
        log_event(logger, "sub_filter_saved", counts=_format_counts(sub_by_split), path=str(out_dir / sub_filter_filename))


def build_graph(
    sample: Sample,
    entity_vocab: EntityVocab | EntityLookup,
    relation_vocab: RelationVocab | RelationLookup,
    graph_id: str,
    *,
    dedup_edges: bool = True,
    validate_graph_edges: bool = _VALIDATE_GRAPH_EDGES_DEFAULT,
    remove_self_loops: bool = _REMOVE_SELF_LOOPS_DEFAULT,
    relation_cleaning_enabled: bool = True,
    relation_cleaning_rules: RelationCleaningRules = DEFAULT_RELATION_CLEANING_RULES,
    inverse_relations_key_map: Optional[Dict[str, str]] = None,
    inverse_relations_suffix: str = _INVERSE_RELATION_SUFFIX_DEFAULT,
    time_relation_cfg: Optional[TimeRelationConfig] = None,
) -> GraphRecord:
    dedup_edges = bool(dedup_edges)
    validate_graph_edges = bool(validate_graph_edges)
    remove_self_loops = bool(remove_self_loops)
    relation_cleaning_enabled = bool(relation_cleaning_enabled)
    inverse_enabled = inverse_relations_key_map is not None
    node_index: Dict[str, int] = {}
    node_entity_ids: List[int] = []
    node_embedding_ids: List[int] = []
    node_labels: List[str] = []

    def local_index(ent: str) -> int:
        if ent not in node_index:
            node_index[ent] = len(node_entity_ids)
            node_entity_ids.append(entity_vocab.entity_id(ent))
            node_embedding_ids.append(entity_vocab.embedding_id(ent))
            node_labels.append(ent)
        return node_index[ent]

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_relation_ids: List[int] = []
    edge_key_to_indices: Dict[Tuple[str, str, str], int] = {}

    # sample.graph must be derived only from q_entity (e.g., PPR on the full graph) with no answer-conditioned steps,
    # per prior work by rmanluo.
    kept_edges, _ = _partition_graph_edges(
        sample.graph,
        relation_cleaning_rules,
        remove_self_loops=remove_self_loops,
        relation_cleaning_enabled=relation_cleaning_enabled,
        time_relation_cfg=time_relation_cfg,
        question_text=sample.question,
    )
    kept_edges = _dedup_directed_edges(kept_edges)
    if inverse_enabled:
        kept_edges = _expand_edges_with_inverse(
            kept_edges,
            inverse_relations_key_map,
            suffix=inverse_relations_suffix,
        )
    for h, r, t in kept_edges:
        edge_key = (h, r, t)
        if dedup_edges and edge_key in edge_key_to_indices:
            continue
        src_idx = local_index(h)
        dst_idx = local_index(t)
        label = None
        if inverse_enabled and inverse_relations_key_map is not None:
            label = inverse_relations_key_map.get(r)
        if isinstance(relation_vocab, RelationLookup):
            rel_idx = relation_vocab.relation_id(r)
        else:
            rel_idx = relation_vocab.relation_id(r, label=label)
        edge_src.append(src_idx)
        edge_dst.append(dst_idx)
        edge_relation_ids.append(rel_idx)
        if dedup_edges:
            edge_key_to_indices[edge_key] = len(edge_src) - 1

    graph = GraphRecord(
        graph_id=graph_id,
        node_entity_ids=node_entity_ids,
        node_embedding_ids=node_embedding_ids,
        node_labels=node_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_relation_ids=edge_relation_ids,
    )
    if validate_graph_edges:
        _validate_graph_record(graph)
    return graph


def build_question_record(
    sample: Sample,
    entity_vocab: EntityVocab,
    graph_id: str,
    *,
    question_emb: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    seed_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.q_entity]
    answer_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.a_entity]
    record = {
        "question_uid": graph_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "kb": sample.kb,
        "question": sample.question,
        "seed_entity_ids": seed_entity_ids,
        "answer_entity_ids": answer_entity_ids,
        "answer_texts": sample.answer_texts,
        "graph_id": graph_id,
    }
    if question_emb is not None:
        record["question_emb"] = list(question_emb)
    return record
