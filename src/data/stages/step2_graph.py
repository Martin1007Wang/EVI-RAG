from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Sequence, Set, Tuple
import torch
from tqdm import tqdm

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap
from src.data.io.lmdb_utils import ensure_dir
from src.data.context import StageContext
from src.data.io.parquet_io import ParquetDatasetWriter, write_embedding_vocab, write_entity_vocab, write_relation_vocab
from src.data.io.raw_loader import build_text_entity_config, iter_samples
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
    _PATH_MODE_QA_DIRECTED,
    _PATH_MODE_UNDIRECTED,
    _REL_LABEL_SAMPLE_LIMIT,
    _REMOVE_SELF_LOOPS_DEFAULT,
    _VALIDATE_GRAPH_EDGES_DEFAULT,
    _ZERO,
)
from src.data.schema.types import EntityLookup, EntityVocab, GraphRecord, RelationLookup, RelationVocab, Sample
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


_WORKER_STATE: Optional[_WorkerState] = None


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
    )


def _validate_node_type_fields(graph: GraphRecord) -> None:
    num_nodes = len(graph.node_entity_ids)
    counts = graph.node_type_counts
    ids = graph.node_type_ids
    if len(counts) != num_nodes:
        raise ValueError(
            f"node_type_counts length {len(counts)} != num_nodes {num_nodes} for {graph.graph_id}."
        )
    if any(count < _ZERO for count in counts):
        raise ValueError(f"node_type_counts contains negatives for {graph.graph_id}.")
    total = int(sum(counts))
    if total != len(ids):
        raise ValueError(
            f"node_type_counts sum {total} != node_type_ids length {len(ids)} for {graph.graph_id}."
        )


def _build_node_type_fields(
    node_labels: Sequence[str],
    node_type_ids_by_entity: Dict[str, Set[int]],
) -> Tuple[List[int], List[int]]:
    node_type_counts: List[int] = []
    node_type_ids: List[int] = []
    for ent in node_labels:
        type_ids = node_type_ids_by_entity.get(ent)
        if not type_ids:
            node_type_counts.append(_ZERO)
            continue
        sorted_ids = sorted(type_ids)
        node_type_counts.append(len(sorted_ids))
        node_type_ids.extend(sorted_ids)
    return node_type_counts, node_type_ids


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
    _validate_node_type_fields(graph)


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
        "canonicalize_relations requires offline shortest-path labels; disable canonicalize_relations for MPM-RAG."
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
    if path_mode != _PATH_MODE_QA_DIRECTED:
        raise ValueError(
            "path_mode must be qa_directed to match directed distance cache semantics."
        )
    dedup_edges = bool(cfg.get("dedup_edges", True))
    validate_graph_edges = bool(cfg.get("validate_graph_edges", _VALIDATE_GRAPH_EDGES_DEFAULT))
    remove_self_loops = bool(cfg.get("remove_self_loops", _REMOVE_SELF_LOOPS_DEFAULT))
    relation_cleaning_enabled = bool(cfg.get("relation_cleaning", True))
    relation_cleaning_rules = DEFAULT_RELATION_CLEANING_RULES
    embedding_cfg = ctx.embedding_cfg
    if embedding_cfg is not None and embedding_cfg.canonicalize_relations:
        raise ValueError("canonicalize_relations requires offline labels; disable it for MPM-RAG.")
    emit_sub_filter = bool(cfg.get("emit_sub_filter", False))
    sub_filter_filename = str(cfg.get("sub_filter_filename", "sub_filter.json"))
    emit_nonzero_positive_filter = bool(cfg.get("emit_nonzero_positive_filter", False))
    nonzero_positive_filter_filename = str(cfg.get("nonzero_positive_filter_filename", "nonzero_positive_filter.json"))
    nonzero_positive_filter_splits = cfg.get("nonzero_positive_filter_splits")
    parquet_chunk_size = ctx.parquet_chunk_size
    parquet_num_workers = ctx.parquet_num_workers
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb, text_cfg=text_cfg)

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
        raise ValueError("emit_nonzero_positive_filter is disabled in MPM-RAG; remove this flag.")

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
    forward_rel_labels = sorted(kept_rel_labels)
    relation_vocab.add_relations(forward_rel_labels)

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
                    relation_vocab,
                    f"{sample.dataset}/{sample.split}/{sample.question_id}",
                    dedup_edges=dedup_edges,
                    validate_graph_edges=validate_graph_edges,
                    remove_self_loops=remove_self_loops,
                    relation_cleaning_enabled=relation_cleaning_enabled,
                    relation_cleaning_rules=relation_cleaning_rules,
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
                    )
                    cleaned_edges = _dedup_directed_edges(cleaned_edges)
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
            relation_lookup=relation_vocab.to_lookup(),
            dedup_edges=dedup_edges,
            validate_graph_edges=validate_graph_edges,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            relation_cleaning_rules=relation_cleaning_rules,
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
) -> GraphRecord:
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
    kept_edges, type_edges = _partition_graph_edges(
        sample.graph,
        relation_cleaning_rules,
        remove_self_loops=remove_self_loops,
        relation_cleaning_enabled=relation_cleaning_enabled,
    )
    kept_edges = _dedup_directed_edges(kept_edges)
    node_type_ids_by_entity: Dict[str, Set[int]] = {}
    for h, _, t in type_edges:
        type_id = entity_vocab.entity_id(t)
        type_set = node_type_ids_by_entity.get(h)
        if type_set is None:
            type_set = set()
            node_type_ids_by_entity[h] = type_set
        type_set.add(type_id)

    for h, r, t in kept_edges:
        edge_key = (h, r, t)
        if dedup_edges and edge_key in edge_key_to_indices:
            continue
        src_idx = local_index(h)
        dst_idx = local_index(t)
        rel_idx = relation_vocab.relation_id(r)
        edge_src.append(src_idx)
        edge_dst.append(dst_idx)
        edge_relation_ids.append(rel_idx)
        if dedup_edges:
            edge_key_to_indices[edge_key] = len(edge_src) - 1

    node_type_counts, node_type_ids = _build_node_type_fields(node_labels, node_type_ids_by_entity)

    graph = GraphRecord(
        graph_id=graph_id,
        node_entity_ids=node_entity_ids,
        node_embedding_ids=node_embedding_ids,
        node_labels=node_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_relation_ids=edge_relation_ids,
        node_type_counts=node_type_counts,
        node_type_ids=node_type_ids,
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
