from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import deque
import torch
from tqdm import tqdm

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap
from src.data.io.lmdb_utils import ensure_dir
from src.data.preprocess.context import PreprocessContext
from src.data.io.parquet_io import (
    ParquetDatasetWriter,
    write_embedding_vocab,
    write_entity_vocab,
    write_relation_vocab,
)
from src.data.io.raw_loader import (
    build_cvt_entity_config,
    build_text_entity_config,
    iter_samples,
)
from src.data.preprocess.cleaning.relation_rules import (
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
    _PATH_MODE_QA_DIRECTED,
    _PATH_MODE_UNDIRECTED,
    _REL_LABEL_SAMPLE_LIMIT,
    _REMOVE_SELF_LOOPS_DEFAULT,
    _VALIDATE_GRAPH_EDGES_DEFAULT,
)
from src.data.schema.types import (
    EntityLookup,
    EntityVocab,
    GraphRecord,
    RelationLookup,
    RelationVocab,
    Sample,
    SplitFilter,
)
from src.data.preprocess.stages.step1_vocab import (
    _partition_graph_edges,
    _resolve_split_filter,
    _should_keep_sample,
)
from src.data.utils.connectivity import _validate_path_mode, reachable_targets_by_index
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
    keep_start_adjacent_edges: bool
    relation_cleaning_rules: RelationCleaningRules
    target_reachable_pruning: bool
    train_filter: SplitFilter
    eval_filter: SplitFilter
    override_filters: Dict[str, SplitFilter]


_WORKER_STATE: Optional[_WorkerState] = None


def _init_worker_state(state: _WorkerState) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _build_graph_worker(sample: Sample) -> Optional[GraphRecord]:
    state = _WORKER_STATE
    graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
    split_filter = _resolve_split_filter(
        sample.split, state.train_filter, state.eval_filter, state.override_filters
    )
    apply_target_reachable_pruning = (
        state.target_reachable_pruning and split_filter.skip_no_path
    )
    return build_graph(
        sample,
        state.entity_lookup,
        state.relation_lookup,
        graph_id,
        dedup_edges=state.dedup_edges,
        validate_graph_edges=state.validate_graph_edges,
        remove_self_loops=state.remove_self_loops,
        relation_cleaning_enabled=state.relation_cleaning_enabled,
        keep_start_adjacent_edges=state.keep_start_adjacent_edges,
        relation_cleaning_rules=state.relation_cleaning_rules,
        target_reachable_pruning=apply_target_reachable_pruning,
    )


def _dedup_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_q_entity_blacklist(cfg) -> set[str]:
    if cfg is None:
        return set()
    raw_list = cfg.get("q_entity_blacklist") or []
    path = cfg.get("q_entity_blacklist_path")
    entries: List[str] = []
    if isinstance(raw_list, (list, tuple, set)):
        entries.extend(str(x).strip() for x in raw_list if str(x).strip())
    if path:
        path = Path(str(path))
        if not path.exists():
            raise FileNotFoundError(f"q_entity_blacklist_path not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                entries.extend(str(x).strip() for x in payload if str(x).strip())
            else:
                raise ValueError(
                    "q_entity_blacklist_path JSON must contain a list of entity ids."
                )
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(line)
    return {str(x) for x in entries if str(x)}


def _apply_q_entity_blacklist(
    sample: Sample,
    blacklist: set[str],
    stats: Optional[Dict[str, int]] = None,
) -> Sample:
    if not blacklist:
        return sample
    q_entities = list(sample.q_entity or [])
    filtered = [ent for ent in q_entities if ent not in blacklist]
    if stats is not None:
        removed = len(q_entities) - len(filtered)
        if removed > 0:
            stats["blacklist_removed_entities"] += removed
            stats["blacklist_reduced_samples"] += 1
            if not filtered:
                stats["blacklist_empty_samples"] += 1
    if filtered == q_entities:
        return sample
    return Sample(
        dataset=sample.dataset,
        split=sample.split,
        question_id=sample.question_id,
        kb=sample.kb,
        question=sample.question,
        graph=sample.graph,
        q_entity=filtered,
        a_entity=sample.a_entity,
        answer_texts=sample.answer_texts,
        graph_iso_type=sample.graph_iso_type,
        redundant=sample.redundant,
        test_type=list(sample.test_type),
    )


def _validate_graph_record(graph: GraphRecord) -> None:
    num_nodes = len(graph.node_entity_ids)
    num_edges = len(graph.edge_src)
    if len(graph.edge_dst) != num_edges or len(graph.edge_relation_ids) != num_edges:
        raise ValueError(
            f"Edge length mismatch for {graph.graph_id}: edges={num_edges}."
        )
    if num_edges > _EDGE_INDEX_MIN:
        min_src = min(graph.edge_src)
        min_dst = min(graph.edge_dst)
        max_src = max(graph.edge_src)
        max_dst = max(graph.edge_dst)
        if min_src < _EDGE_INDEX_MIN or min_dst < _EDGE_INDEX_MIN:
            raise ValueError(f"Negative edge index detected for {graph.graph_id}.")
        if max_src >= num_nodes or max_dst >= num_nodes:
            raise ValueError(f"Edge index exceeds num_nodes for {graph.graph_id}.")


def _dedup_directed_edges(
    edges: Sequence[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
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


def _compute_target_reachable_nodes(
    edges: Sequence[Tuple[str, str, str]],
    targets: Sequence[str],
) -> Set[str]:
    if not edges or not targets:
        return set()
    node_index: Dict[str, int] = {}
    reverse_adj: List[List[int]] = []

    def _add_node(label: str) -> int:
        idx = node_index.get(label)
        if idx is not None:
            return idx
        idx = len(node_index)
        node_index[label] = idx
        reverse_adj.append([])
        return idx

    for head, _, tail in edges:
        head_idx = _add_node(head)
        tail_idx = _add_node(tail)
        reverse_adj[tail_idx].append(head_idx)

    target_ids = [node_index[tgt] for tgt in targets if tgt in node_index]
    if not target_ids:
        return set()
    visited = [False] * len(node_index)
    q: deque[int] = deque()
    for tid in target_ids:
        visited[tid] = True
        q.append(tid)
    while q:
        cur = q.popleft()
        for nbr in reverse_adj[cur]:
            if visited[nbr]:
                continue
            visited[nbr] = True
            q.append(nbr)
    return {label for label, idx in node_index.items() if visited[idx]}


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
        "canonicalize_relations requires offline shortest-path labels; disable canonicalize_relations for DualFlow."
    )


def preprocess(ctx: PreprocessContext) -> None:
    cfg = ctx.cfg
    logger = ctx.logger
    dataset = ctx.dataset_name
    kb = cfg.kb
    out_dir = ctx.out_dir
    column_map = dict(cfg.column_map)
    entity_normalization = cfg.entity_normalization
    text_cfg = build_text_entity_config(cfg)
    cvt_cfg = build_cvt_entity_config(cfg)
    dataset_family = cfg.get("dataset_family")
    dataset_source = str(cfg.get("dataset_source", "hf")).strip().lower()
    hf_dataset = cfg.get("hf_dataset")
    hf_cache_dir_cfg = cfg.get("hf_cache_dir")
    hf_cache_dir = ctx.resolve_path(hf_cache_dir_cfg) if hf_cache_dir_cfg else None
    hf_offline = bool(cfg.get("hf_offline", False))
    if dataset_source != "hf":
        raise ValueError(
            "dataset_source must be 'hf'; raw parquet ingestion is disabled."
        )
    train_filter, eval_filter, override_filters = ctx.split_filters
    path_mode = _validate_path_mode(str(cfg.get("path_mode", _PATH_MODE_UNDIRECTED)))
    target_reachable_pruning = bool(cfg.get("target_reachable_pruning", False))
    dedup_edges = bool(cfg.get("dedup_edges", True))
    validate_graph_edges = bool(
        cfg.get("validate_graph_edges", _VALIDATE_GRAPH_EDGES_DEFAULT)
    )
    remove_self_loops = bool(cfg.get("remove_self_loops", _REMOVE_SELF_LOOPS_DEFAULT))
    relation_cleaning_enabled = bool(cfg.get("relation_cleaning", True))
    relation_cleaning_rules = DEFAULT_RELATION_CLEANING_RULES
    keep_start_adjacent_edges = bool(cfg.get("keep_start_adjacent_edges", False))
    embedding_cfg = ctx.embedding_cfg
    if embedding_cfg is not None and embedding_cfg.canonicalize_relations:
        raise ValueError(
            "canonicalize_relations requires offline labels; disable it for DualFlow."
        )
    emit_sub_filter = bool(cfg.get("emit_sub_filter", False))
    sub_filter_filename = str(cfg.get("sub_filter_filename", "sub_filter.json"))
    emit_nonzero_positive_filter = bool(cfg.get("emit_nonzero_positive_filter", False))
    nonzero_positive_filter_filename = str(
        cfg.get("nonzero_positive_filter_filename", "nonzero_positive_filter.json")
    )
    nonzero_positive_filter_splits = cfg.get("nonzero_positive_filter_splits")
    parquet_chunk_size = ctx.parquet_chunk_size
    parquet_num_workers = ctx.parquet_num_workers
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))
    q_entity_blacklist = _load_q_entity_blacklist(cfg)

    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb, text_cfg=text_cfg, cvt_cfg=cvt_cfg)

    splits = list(_ALLOWED_SPLITS)
    connectivity_cache: Dict[Tuple[str, str, str], bool] = {}
    total_by_split: Dict[str, int] = {}
    kept_by_split: Dict[str, int] = {}
    sub_by_split: Dict[str, int] = {}
    sub_filter_stats: Dict[str, Dict[str, object]] = {
        split: {
            "total_samples": 0,
            "kept_samples": 0,
            "missing_q_any_samples": 0,
            "missing_q_all_samples": 0,
            "missing_q_partial_samples": 0,
            "missing_a_any_samples": 0,
            "missing_a_all_samples": 0,
            "missing_a_partial_samples": 0,
            "unreachable_a_any_samples": 0,
            "unreachable_a_all_samples": 0,
            "unreachable_a_partial_samples": 0,
            "no_path_samples": 0,
            "missing_q_entities": 0,
            "missing_a_entities": 0,
            "reachable_a_entities": 0,
            "unreachable_a_entities": 0,
            "overlap_samples": {},
        }
        for split in splits
    }
    qa_clean_stats: Dict[str, Dict[str, int]] = {
        split: {
            "samples_total": 0,
            "q_total_before": 0,
            "q_total_after": 0,
            "q_samples_reduced": 0,
            "q_samples_empty": 0,
            "q_samples_all_present": 0,
            "q_samples_partial_missing": 0,
            "q_samples_all_missing": 0,
            "a_total_before": 0,
            "a_total_after": 0,
            "a_samples_reduced": 0,
            "a_samples_all_present": 0,
            "a_samples_partial_missing": 0,
            "a_samples_all_missing": 0,
            "reachable_all_samples": 0,
            "reachable_partial_samples": 0,
            "reachable_none_samples": 0,
            "reachable_na_samples": 0,
            "blacklist_removed_entities": 0,
            "blacklist_reduced_samples": 0,
            "blacklist_empty_samples": 0,
        }
        for split in splits
    }
    empty_graph_by_split: Dict[str, int] = {}
    empty_graph_ids: List[str] = []
    empty_graph_id_set: Set[str] = set()
    pruned_drop_by_split: Dict[str, int] = {}
    sub_sample_ids: List[str] = []
    edge_stats = _init_split_counters(splits, _EDGE_STAT_KEYS)
    filter_stats = _init_split_counters(splits, _FILTER_STAT_KEYS)
    kept_rel_labels: Set[str] = set()
    type_rel_labels: Set[str] = set()
    dropped_rel_labels: Set[str] = set()
    graphs_written_by_split = {split: 0 for split in splits}
    questions_written_by_split = {split: 0 for split in splits}
    if emit_nonzero_positive_filter:
        raise ValueError(
            "emit_nonzero_positive_filter is disabled in DualFlow; remove this flag."
        )

    if target_reachable_pruning and path_mode != _PATH_MODE_QA_DIRECTED:
        raise ValueError("target_reachable_pruning requires path_mode=qa_directed.")
    log_event(
        logger,
        "preprocess_start",
        dataset=dataset,
        kb=kb,
        splits=splits,
        dataset_source=dataset_source,
        hf_dataset=hf_dataset,
        path_mode=path_mode,
        target_reachable_pruning=target_reachable_pruning,
        dedup_edges=dedup_edges,
        remove_self_loops=remove_self_loops,
        relation_cleaning=relation_cleaning_enabled,
        parquet_chunk_size=parquet_chunk_size,
        parquet_num_workers=parquet_num_workers,
    )
    if q_entity_blacklist:
        log_event(
            logger,
            "q_entity_blacklist_loaded",
            count=len(q_entity_blacklist),
            examples=sorted(list(q_entity_blacklist))[:20],
        )
    if relation_cleaning_enabled:
        log_event(
            logger,
            "relation_cleaning_rules",
            type_exact=sorted(relation_cleaning_rules.type_exact),
            type_prefixes=sorted(relation_cleaning_rules.type_prefixes),
            type_regexes=sorted(
                pattern.pattern for pattern in relation_cleaning_rules.type_regexes
            ),
            drop_exact=sorted(relation_cleaning_rules.drop_exact),
            drop_prefixes=sorted(relation_cleaning_rules.drop_prefixes),
            drop_regexes=sorted(
                pattern.pattern for pattern in relation_cleaning_rules.drop_regexes
            ),
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
        sample = _apply_q_entity_blacklist(sample, q_entity_blacklist)
        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        total_by_split[sample.split] = total_by_split.get(sample.split, 0) + 1
        kept_edges, type_edges = _partition_graph_edges(
            sample.graph,
            relation_cleaning_rules,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            anchor_entities=sample.q_entity,
            keep_anchor_edges=keep_start_adjacent_edges,
        )
        kept_edges = _dedup_directed_edges(kept_edges)
        split_key = sample.split
        raw_edges = len(sample.graph)
        self_loop_edges = 0
        if remove_self_loops:
            self_loop_edges = sum(1 for h, _, t in sample.graph if h == t)
        kept_edges_count = len(kept_edges)
        type_edges_count = len(type_edges)
        dropped_edges = (
            raw_edges - self_loop_edges - kept_edges_count - type_edges_count
        )
        raw_nodes = len(
            {h for h, _, _ in sample.graph} | {t for _, _, t in sample.graph}
        )
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
            if (
                relation_action(
                    rel, relation_cleaning_rules, enabled=relation_cleaning_enabled
                )
                == RELATION_ACTION_DROP
            ):
                dropped_rel_labels.add(rel)
        if not kept_edges:
            empty_graph_by_split[sample.split] = (
                empty_graph_by_split.get(sample.split, 0) + 1
            )
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

        split_filter = _resolve_split_filter(
            sample.split, train_filter, eval_filter, override_filters
        )
        outcome = _should_keep_sample(
            sample,
            split_filter,
            connectivity_cache,
            path_mode=path_mode,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            relation_cleaning_rules=relation_cleaning_rules,
            kept_edges=kept_edges,
            anchor_entities=sample.q_entity,
            keep_anchor_edges=keep_start_adjacent_edges,
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
    for rel in dataset_rel_labels:
        relation_vocab.relation_id(rel, label=rel)

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
        split_total = total_by_split.get(split, 0)
        log_event(
            logger,
            "split_summary",
            split=split,
            samples_total=split_total,
            samples_kept=kept_by_split.get(split, 0),
            samples_empty_graph=empty_graph_by_split.get(split, 0),
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
        kept_relation_examples=_sample_labels(
            kept_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT
        ),
        type_relation_examples=_sample_labels(
            type_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT
        ),
        dropped_relation_examples=_sample_labels(
            dropped_rel_labels, limit=_REL_LABEL_SAMPLE_LIMIT
        ),
    )

    def _format_counts(counts: Dict[str, int]) -> str:
        return ", ".join(f"{s}={counts.get(s, 0)}" for s in splits)

    log_event(logger, "samples_total", counts=_format_counts(total_by_split))
    log_event(logger, "samples_kept", counts=_format_counts(kept_by_split))
    if empty_graph_by_split:
        log_event(
            logger, "samples_empty_graph", counts=_format_counts(empty_graph_by_split)
        )
        if empty_graph_ids:
            log_event(logger, "empty_graph_examples", examples=empty_graph_ids)

    encoder: Optional[TextEncoder] = None
    relation_embeddings_norm: Optional[torch.Tensor] = None
    if embedding_cfg is not None:
        embeddings_out_dir = embedding_cfg.embeddings_out_dir
        entity_emb_path = embeddings_out_dir / "entity_embeddings.pt"
        relation_emb_path = embeddings_out_dir / "relation_embeddings.pt"
        need_entity_encode = not (
            reuse_embeddings_if_exists and entity_emb_path.exists()
        )
        need_relation_encode = not (
            reuse_embeddings_if_exists and relation_emb_path.exists()
        )
        encoder = TextEncoder(
            embedding_cfg.encoder,
            embedding_cfg.device,
            embedding_cfg.fp16,
            embedding_cfg.progress_bar,
        )
        ensure_dir(embeddings_out_dir)
        if not need_entity_encode:
            log_event(
                logger, "preprocess_reuse_entity_embeddings", path=str(entity_emb_path)
            )
        else:
            emb_rows = sorted(
                (
                    (rec["embedding_id"], rec.get("label", ""))
                    for rec in entity_vocab.embedding_records
                ),
                key=lambda x: x[0],
            )
            text_labels = [str(label) for _, label in emb_rows]
            text_ids = [int(eid) for eid, _ in emb_rows]
            struct_records = entity_vocab.struct_records
            max_embedding_id = max(
                (int(rec["embedding_id"]) for rec in struct_records), default=0
            )
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
        if need_relation_encode:
            relation_rows = sorted(
                (
                    (rec["relation_id"], rec.get("label", ""))
                    for rec in relation_vocab.records
                ),
                key=lambda x: x[0],
            )
            relation_labels = [str(label) for _, label in relation_rows]
            relation_emb = encoder.encode(
                relation_labels,
                embedding_cfg.batch_size,
                show_progress=embedding_cfg.progress_bar,
                desc="Relations",
            )
            torch.save(relation_emb, relation_emb_path)
        else:
            log_event(
                logger,
                "preprocess_reuse_relation_embeddings",
                path=str(relation_emb_path),
            )
            relation_emb = torch.load(relation_emb_path, map_location="cpu")
        if embedding_cfg.canonicalize_relations:
            relation_embeddings_norm = _normalize_embeddings(
                relation_emb, embedding_cfg.cosine_eps
            )
            if relation_embeddings_norm.numel() == 0:
                raise ValueError(
                    "relation_embeddings are empty; cannot canonicalize positives."
                )

    log_event(logger, "graphs_questions_start", stage="graphs_questions")
    chunk_size = parquet_chunk_size
    include_question_emb = bool(embedding_cfg)
    include_question_ctx = (
        bool(embedding_cfg) and int(embedding_cfg.question_ctx_max_tokens) > 0
    )
    base_writer = ParquetDatasetWriter(
        out_dir=out_dir,
        include_question_emb=include_question_emb,
        include_question_ctx=include_question_ctx,
    )
    need_question_emb = bool(embedding_cfg)
    need_question_ctx = (
        bool(embedding_cfg) and int(embedding_cfg.question_ctx_max_tokens) > 0
    )

    def _process_sample_batch(
        samples: List[Sample], executor: Optional[ProcessPoolExecutor]
    ) -> None:
        if not samples:
            return
        question_emb_batch = None
        question_emb_norm_batch = None
        question_ctx_batch = None
        question_ctx_mask_batch = None
        if need_question_emb:
            if encoder is None:
                raise RuntimeError(
                    "Question embeddings requested but encoder is not configured."
                )
            question_texts = [sample.question for sample in samples]
            if need_question_ctx:
                question_emb_batch, question_ctx_batch, question_ctx_mask_batch = (
                    encoder.encode_with_context(
                        question_texts,
                        embedding_cfg.batch_size,
                        max_tokens=int(embedding_cfg.question_ctx_max_tokens),
                        show_progress=False,
                        desc="QuestionsWithContext",
                    )
                )
            else:
                question_emb_batch = encoder.encode(
                    question_texts,
                    embedding_cfg.batch_size,
                    show_progress=False,
                    desc="Questions",
                )
            if embedding_cfg and embedding_cfg.canonicalize_relations:
                question_emb_norm_batch = _normalize_embeddings(
                    question_emb_batch, embedding_cfg.cosine_eps
                )
        if executor is None:
            graphs: List[Optional[GraphRecord]] = []
            for sample in samples:
                split_filter = _resolve_split_filter(
                    sample.split, train_filter, eval_filter, override_filters
                )
                apply_target_reachable_pruning = (
                    target_reachable_pruning and split_filter.skip_no_path
                )
                graph = build_graph(
                    sample,
                    entity_vocab,
                    relation_lookup,
                    f"{sample.dataset}/{sample.split}/{sample.question_id}",
                    dedup_edges=dedup_edges,
                    validate_graph_edges=validate_graph_edges,
                    remove_self_loops=remove_self_loops,
                    relation_cleaning_enabled=relation_cleaning_enabled,
                    keep_start_adjacent_edges=keep_start_adjacent_edges,
                    relation_cleaning_rules=relation_cleaning_rules,
                    target_reachable_pruning=apply_target_reachable_pruning,
                )
                graphs.append(graph)
        else:
            graphs = list(executor.map(_build_graph_worker, samples))
        for idx, (sample, graph) in enumerate(zip(samples, graphs)):
            if graph is None:
                pruned_drop_by_split[sample.split] = (
                    pruned_drop_by_split.get(sample.split, 0) + 1
                )
                continue
            if embedding_cfg and embedding_cfg.canonicalize_relations:
                if relation_embeddings_norm is None or question_emb_norm_batch is None:
                    raise RuntimeError(
                        "Canonicalization requested but embeddings are missing."
                    )
                _canonicalize_graph_edges(
                    graph, question_emb_norm_batch[idx], relation_embeddings_norm
                )
            question_emb = None
            question_ctx = None
            question_ctx_mask = None
            if embedding_cfg:
                if question_emb_batch is None:
                    raise RuntimeError(
                        "question_emb batch missing during question embedding encode."
                    )
                question_emb = question_emb_batch[idx].tolist()
                if need_question_ctx:
                    if question_ctx_batch is None or question_ctx_mask_batch is None:
                        raise RuntimeError(
                            "question_ctx batch missing during question context encode."
                        )
                    question_ctx = question_ctx_batch[idx].tolist()
                    question_ctx_mask = question_ctx_mask_batch[idx].tolist()
            split_key = sample.split
            stats_clean = qa_clean_stats[split_key]
            label_to_idx = {label: idx for idx, label in enumerate(graph.node_labels)}
            q_entities_raw = _dedup_preserve_order(sample.q_entity or [])
            a_entities_raw = _dedup_preserve_order(sample.a_entity or [])
            q_in_graph = [ent for ent in q_entities_raw if ent in label_to_idx]
            a_in_graph = [ent for ent in a_entities_raw if ent in label_to_idx]
            q_total = len(q_entities_raw)
            a_total = len(a_entities_raw)
            q_in_graph_count = len(q_in_graph)
            a_in_graph_count = len(a_in_graph)
            stats_clean["samples_total"] += 1
            stats_clean["q_total_before"] += q_total
            stats_clean["q_total_after"] += q_in_graph_count
            stats_clean["a_total_before"] += a_total
            stats_clean["a_total_after"] += a_in_graph_count
            if q_in_graph_count < q_total:
                stats_clean["q_samples_reduced"] += 1
            if q_in_graph_count == 0:
                stats_clean["q_samples_empty"] += 1
            if q_in_graph_count == q_total and q_total > 0:
                stats_clean["q_samples_all_present"] += 1
            elif q_in_graph_count == 0:
                stats_clean["q_samples_all_missing"] += 1
            else:
                stats_clean["q_samples_partial_missing"] += 1
            if a_in_graph_count < a_total:
                stats_clean["a_samples_reduced"] += 1
            if a_in_graph_count == 0:
                stats_clean["a_samples_all_missing"] += 1
            elif a_in_graph_count == a_total and a_total > 0:
                stats_clean["a_samples_all_present"] += 1
            else:
                stats_clean["a_samples_partial_missing"] += 1
            q_local = [label_to_idx[ent] for ent in q_in_graph]
            a_local = [label_to_idx[ent] for ent in a_in_graph]
            reachable_count = 0
            if q_in_graph_count > 0 and a_in_graph_count > 0:
                reachable_targets = reachable_targets_by_index(
                    num_nodes=len(graph.node_labels),
                    edge_src=graph.edge_src,
                    edge_dst=graph.edge_dst,
                    seeds=q_local,
                    targets=a_local,
                    path_mode=path_mode,
                )
                reachable_count = len(reachable_targets)
                if reachable_count == a_in_graph_count:
                    stats_clean["reachable_all_samples"] += 1
                elif reachable_count == 0:
                    stats_clean["reachable_none_samples"] += 1
                else:
                    stats_clean["reachable_partial_samples"] += 1
            else:
                stats_clean["reachable_na_samples"] += 1
            if emit_sub_filter:
                stats = sub_filter_stats[split_key]
                stats["total_samples"] += 1
                missing_q_entities = max(q_total - q_in_graph_count, 0)
                missing_a_entities = max(a_total - a_in_graph_count, 0)
                stats["missing_q_entities"] += missing_q_entities
                stats["missing_a_entities"] += missing_a_entities
                missing_q_any = (q_total == 0) or (q_in_graph_count < q_total)
                missing_a_any = (a_total == 0) or (a_in_graph_count < a_total)
                missing_q_all = q_in_graph_count == 0
                missing_a_all = a_in_graph_count == 0
                missing_q_partial = missing_q_any and not missing_q_all
                missing_a_partial = missing_a_any and not missing_a_all
                stats["missing_q_any_samples"] += 1 if missing_q_any else 0
                stats["missing_q_all_samples"] += 1 if missing_q_all else 0
                stats["missing_q_partial_samples"] += 1 if missing_q_partial else 0
                stats["missing_a_any_samples"] += 1 if missing_a_any else 0
                stats["missing_a_all_samples"] += 1 if missing_a_all else 0
                stats["missing_a_partial_samples"] += 1 if missing_a_partial else 0
                unreachable_count = a_in_graph_count - reachable_count
                if unreachable_count < 0:
                    unreachable_count = 0
                stats["reachable_a_entities"] += reachable_count
                stats["unreachable_a_entities"] += unreachable_count
                unreachable_a_any = (
                    a_in_graph_count > 0 and reachable_count < a_in_graph_count
                )
                unreachable_a_all = a_in_graph_count > 0 and reachable_count == 0
                unreachable_a_partial = (
                    a_in_graph_count > 0 and 0 < reachable_count < a_in_graph_count
                )
                no_path = (
                    q_in_graph_count > 0
                    and a_in_graph_count > 0
                    and reachable_count == 0
                )
                stats["unreachable_a_any_samples"] += 1 if unreachable_a_any else 0
                stats["unreachable_a_all_samples"] += 1 if unreachable_a_all else 0
                stats["unreachable_a_partial_samples"] += (
                    1 if unreachable_a_partial else 0
                )
                stats["no_path_samples"] += 1 if no_path else 0
                overlap_flags: List[str] = []
                if missing_q_any:
                    overlap_flags.append("missing_q")
                if missing_a_any:
                    overlap_flags.append("missing_a")
                if unreachable_a_any:
                    overlap_flags.append("unreachable_a")
                overlap_key = "+".join(overlap_flags) if overlap_flags else "ok"
                overlap = stats["overlap_samples"]
                overlap[overlap_key] = overlap.get(overlap_key, 0) + 1
                if not (missing_q_any or missing_a_any or unreachable_a_any):
                    sub_sample_ids.append(graph.graph_id)
                    sub_by_split[split_key] = sub_by_split.get(split_key, 0) + 1
                    stats["kept_samples"] += 1
            if q_in_graph_count == 0:
                continue
            question = build_question_record(
                sample,
                entity_vocab,
                graph.graph_id,
                question_emb=question_emb,
                question_ctx=question_ctx,
                question_ctx_mask=question_ctx_mask,
                q_entities=q_in_graph,
                a_entities=a_in_graph,
            )
            base_writer.append(graph, question)
            graphs_written_by_split[split_key] += 1
            questions_written_by_split[split_key] += 1
            if (
                len(base_writer.graphs) >= chunk_size
                or len(base_writer.questions) >= chunk_size
            ):
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
            q_entities_raw = list(sample.q_entity or [])
            sample = _apply_q_entity_blacklist(sample, q_entity_blacklist)
            graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
            if graph_id in empty_graph_id_set:
                continue
            split_filter = _resolve_split_filter(
                sample.split, train_filter, eval_filter, override_filters
            )
            outcome = _should_keep_sample(
                sample,
                split_filter,
                connectivity_cache,
                path_mode=path_mode,
                remove_self_loops=remove_self_loops,
                relation_cleaning_enabled=relation_cleaning_enabled,
                relation_cleaning_rules=relation_cleaning_rules,
                anchor_entities=sample.q_entity,
                keep_anchor_edges=keep_start_adjacent_edges,
            )
            if not outcome.keep:
                continue
            if q_entity_blacklist:
                removed = len(q_entities_raw) - len(sample.q_entity or [])
                if removed > 0:
                    stats = qa_clean_stats[sample.split]
                    stats["blacklist_removed_entities"] += int(removed)
                    stats["blacklist_reduced_samples"] += 1
                    if not sample.q_entity:
                        stats["blacklist_empty_samples"] += 1
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
            keep_start_adjacent_edges=keep_start_adjacent_edges,
            relation_cleaning_rules=relation_cleaning_rules,
            target_reachable_pruning=target_reachable_pruning,
            train_filter=train_filter,
            eval_filter=eval_filter,
            override_filters=override_filters,
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
    if pruned_drop_by_split:
        log_event(
            logger,
            "samples_dropped_target_pruning",
            counts=_format_counts(pruned_drop_by_split),
        )
    log_event(
        logger,
        "graphs_questions_written",
        graphs_by_split=graphs_written_by_split,
        questions_by_split=questions_written_by_split,
    )

    write_entity_vocab(entity_vocab.struct_records, out_dir / "entity_vocab.parquet")
    write_embedding_vocab(
        entity_vocab.embedding_records, out_dir / "embedding_vocab.parquet"
    )
    write_relation_vocab(relation_vocab.records, out_dir / "relation_vocab.parquet")

    if emit_sub_filter:
        sub_payload = {
            "dataset": dataset,
            "sample_ids": sorted(sub_sample_ids),
            "criteria": {
                "require_all_questions_present": True,
                "require_all_answers_present": True,
                "require_all_answers_reachable": True,
                "path_mode": path_mode,
            },
            "stats": sub_filter_stats,
        }
        (out_dir / sub_filter_filename).write_text(json.dumps(sub_payload, indent=2))
        log_event(
            logger,
            "sub_filter_saved",
            counts=_format_counts(sub_by_split),
            path=str(out_dir / sub_filter_filename),
        )
        for split in splits:
            stats = sub_filter_stats[split]
            total = int(stats["total_samples"])
            kept = int(stats["kept_samples"])
            log_event(
                logger,
                "sub_filter_stats",
                split=split,
                total=total,
                kept=kept,
                filtered=total - kept,
                missing_q_any=int(stats["missing_q_any_samples"]),
                missing_q_all=int(stats["missing_q_all_samples"]),
                missing_q_partial=int(stats["missing_q_partial_samples"]),
                missing_a_any=int(stats["missing_a_any_samples"]),
                missing_a_all=int(stats["missing_a_all_samples"]),
                missing_a_partial=int(stats["missing_a_partial_samples"]),
                unreachable_a_any=int(stats["unreachable_a_any_samples"]),
                unreachable_a_all=int(stats["unreachable_a_all_samples"]),
                unreachable_a_partial=int(stats["unreachable_a_partial_samples"]),
                no_path=int(stats["no_path_samples"]),
                missing_q_entities=int(stats["missing_q_entities"]),
                missing_a_entities=int(stats["missing_a_entities"]),
                reachable_a_entities=int(stats["reachable_a_entities"]),
                unreachable_a_entities=int(stats["unreachable_a_entities"]),
                overlap=stats["overlap_samples"],
            )
        for split in splits:
            stats = qa_clean_stats[split]
            total = int(stats["samples_total"])
            log_event(
                logger,
                "qa_entity_cleaning_stats",
                split=split,
                samples=total,
                q_total_before=int(stats["q_total_before"]),
                q_total_after=int(stats["q_total_after"]),
                q_avg_before=_safe_div(int(stats["q_total_before"]), total),
                q_avg_after=_safe_div(int(stats["q_total_after"]), total),
                q_samples_reduced=int(stats["q_samples_reduced"]),
                q_samples_empty=int(stats["q_samples_empty"]),
                q_samples_all_present=int(stats["q_samples_all_present"]),
                q_samples_partial_missing=int(stats["q_samples_partial_missing"]),
                q_samples_all_missing=int(stats["q_samples_all_missing"]),
                a_total_before=int(stats["a_total_before"]),
                a_total_after=int(stats["a_total_after"]),
                a_avg_before=_safe_div(int(stats["a_total_before"]), total),
                a_avg_after=_safe_div(int(stats["a_total_after"]), total),
                a_samples_reduced=int(stats["a_samples_reduced"]),
                a_samples_all_present=int(stats["a_samples_all_present"]),
                a_samples_partial_missing=int(stats["a_samples_partial_missing"]),
                a_samples_all_missing=int(stats["a_samples_all_missing"]),
                reachable_all=int(stats["reachable_all_samples"]),
                reachable_partial=int(stats["reachable_partial_samples"]),
                reachable_none=int(stats["reachable_none_samples"]),
                reachable_na=int(stats["reachable_na_samples"]),
                blacklist_removed_entities=int(stats["blacklist_removed_entities"]),
                blacklist_reduced_samples=int(stats["blacklist_reduced_samples"]),
                blacklist_empty_samples=int(stats["blacklist_empty_samples"]),
            )


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
    keep_start_adjacent_edges: bool = False,
    relation_cleaning_rules: RelationCleaningRules = DEFAULT_RELATION_CLEANING_RULES,
    target_reachable_pruning: bool = False,
) -> Optional[GraphRecord]:
    dedup_edges = bool(dedup_edges)
    validate_graph_edges = bool(validate_graph_edges)
    remove_self_loops = bool(remove_self_loops)
    relation_cleaning_enabled = bool(relation_cleaning_enabled)
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
        anchor_entities=sample.q_entity,
        keep_anchor_edges=keep_start_adjacent_edges,
    )
    kept_edges = _dedup_directed_edges(kept_edges)
    if target_reachable_pruning:
        reachable_nodes = _compute_target_reachable_nodes(kept_edges, sample.a_entity)
        if not reachable_nodes:
            return None
        start_reachable = any(ent in reachable_nodes for ent in sample.q_entity)
        if not start_reachable:
            return None
        kept_edges = [
            edge
            for edge in kept_edges
            if edge[0] in reachable_nodes and edge[2] in reachable_nodes
        ]
        if not kept_edges:
            return None
    for h, r, t in kept_edges:
        edge_key = (h, r, t)
        if dedup_edges and edge_key in edge_key_to_indices:
            continue
        src_idx = local_index(h)
        dst_idx = local_index(t)
        if isinstance(relation_vocab, RelationLookup):
            rel_idx = relation_vocab.relation_id(r)
        else:
            rel_idx = relation_vocab.relation_id(r)
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
    question_ctx: Optional[Sequence[Sequence[float]]] = None,
    question_ctx_mask: Optional[Sequence[bool]] = None,
    q_entities: Optional[Sequence[str]] = None,
    a_entities: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    q_entities = list(sample.q_entity) if q_entities is None else list(q_entities)
    a_entities = list(sample.a_entity) if a_entities is None else list(a_entities)
    seed_entity_ids = [entity_vocab.entity_id(ent) for ent in q_entities]
    answer_entity_ids = [entity_vocab.entity_id(ent) for ent in a_entities]
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
    if question_ctx is not None or question_ctx_mask is not None:
        if question_ctx is None or question_ctx_mask is None:
            raise ValueError(
                "question_ctx and question_ctx_mask must be provided together."
            )
        if len(question_ctx) != len(question_ctx_mask):
            raise ValueError(
                "question_ctx length must equal question_ctx_mask length: "
                f"ctx={len(question_ctx)} mask={len(question_ctx_mask)}."
            )
        record["question_ctx"] = [list(token) for token in question_ctx]
        record["question_ctx_mask"] = [bool(flag) for flag in question_ctx_mask]
    return record
