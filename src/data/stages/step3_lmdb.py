from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from collections import deque

import json

import lmdb
import torch

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap

from src.data.context import StageContext
from src.data.io.lmdb_utils import (
    _assign_lmdb_shard,
    _commit_pending_with_growth,
    _finalize_lmdb_dir,
    _format_lmdb_path,
    _local_indices,
    _prepare_lmdb_dir,
    _replay_pending_with_growth,
    _resolve_lmdb_map_config,
    _resolve_lmdb_shards,
    _serialize_sample,
    _write_sample,
    ensure_dir,
)
from src.data.io.parquet_io import _load_parquet
from src.data.edge_retriever_labels import compute_shortest_path_labels
from src.data.schema.constants import (
    _BYTES_PER_GB,
    _FILTER_MISSING_ANSWER_FILENAME,
    _FILTER_MISSING_START_FILENAME,
    _MIN_CHUNK_SIZE,
    _ONE,
    _ZERO,
)
from src.config.data_config import _resolve_parquet_chunk_size
from src.data.utils.validation import _validate_split_names
from src.utils.logging_utils import log_event


def _format_core_path(base_dir: Path, split: str, shard_id: int, num_shards: int) -> Path:
    return _format_lmdb_path(base_dir, split, shard_id, num_shards, suffix=".lmdb")


def _write_sample_filter(path: Path, *, dataset: str, sample_ids: List[str]) -> None:
    payload = {
        "dataset": dataset,
        "sample_ids": sorted(sample_ids),
    }
    path.write_text(json.dumps(payload, indent=2))


def _bfs_multi(num_nodes: int, adjacency: List[List[int]], sources: List[int]) -> List[int]:
    dist = [-1] * num_nodes
    if num_nodes <= 0 or not sources:
        return dist
    q: deque[int] = deque()
    for s_raw in sources:
        s = int(s_raw)
        if 0 <= s < num_nodes and dist[s] < 0:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        du = dist[u] + _ONE
        for v in adjacency[u]:
            if dist[v] >= 0:
                continue
            dist[v] = du
            q.append(v)
    return dist


def _count_reachable_any_direction(
    num_nodes: int,
    edge_src: List[int],
    edge_dst: List[int],
    q_nodes: List[int],
    a_nodes: List[int],
) -> int:
    if num_nodes <= 0 or not q_nodes or not a_nodes:
        return _ZERO
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    rev_adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(edge_src, edge_dst):
        u = int(u_raw)
        v = int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append(v)
            rev_adjacency[v].append(u)
    for nbrs in adjacency:
        nbrs.sort()
    for nbrs in rev_adjacency:
        nbrs.sort()
    dist_fwd = _bfs_multi(num_nodes, adjacency, q_nodes)
    dist_rev = _bfs_multi(num_nodes, rev_adjacency, q_nodes)
    reachable = _ZERO
    for a_raw in a_nodes:
        a = int(a_raw)
        if 0 <= a < num_nodes and (dist_fwd[a] >= 0 or dist_rev[a] >= 0):
            reachable += _ONE
    return reachable


def build_dataset(ctx: StageContext) -> None:
    cfg = ctx.cfg
    logger = ctx.logger
    dataset_cfg = cfg.get("dataset") if hasattr(cfg, "get") else {}
    dataset_name = str(dataset_cfg.get("name", "") or "")
    dataset_scope = str(dataset_cfg.get("dataset_scope", "") or "").strip().lower()
    emit_edge_retriever_labels = bool(cfg.get("emit_edge_retriever_labels", False))
    labels_dir_cfg = cfg.get("edge_retriever_labels_dir")
    log_event(
        logger,
        "lmdb_start",
        dataset=dataset_name,
        output_dir=str(ctx.output_dir),
        parquet_dir=str(ctx.parquet_dir),
    )
    if dataset_scope == "sub" or dataset_name.endswith("-sub"):
        raise ValueError(
            "Sub datasets are mask-only and must not be materialized into LMDB. "
            "Build the full dataset once, then use sample_filter_path at runtime."
        )
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)

    entity_vocab = _load_parquet(ctx.parquet_dir / "entity_vocab.parquet").to_pydict()
    embedding_vocab = _load_parquet(ctx.parquet_dir / "embedding_vocab.parquet").to_pydict()
    relation_vocab = _load_parquet(ctx.parquet_dir / "relation_vocab.parquet").to_pydict()

    relation_rows = sorted(zip(relation_vocab["relation_id"], relation_vocab["label"]), key=lambda x: x[0])
    relation_labels: List[str] = [str(label) for _, label in relation_rows]
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    emb_dir = ctx.embeddings_dir
    ensure_dir(emb_dir)
    entity_emb_path = emb_dir / "entity_embeddings.pt"
    relation_emb_path = emb_dir / "relation_embeddings.pt"

    encoder: TextEncoder | None = None

    def _get_encoder() -> TextEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = TextEncoder(cfg.encoder, cfg.device, cfg.fp16, cfg.progress_bar)
        return encoder

    if not use_precomputed_embeddings and reuse_embeddings_if_exists:
        if entity_emb_path.exists() and relation_emb_path.exists():
            log_event(logger, "lmdb_reuse_embeddings", path=str(emb_dir))
            use_precomputed_embeddings = True

    if use_precomputed_embeddings:
        missing_paths = [str(p) for p in (entity_emb_path, relation_emb_path) if not p.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Precomputed embeddings missing: {missing_paths}")
        relation_emb = torch.load(relation_emb_path, map_location="cpu")
        expected_relations = len(relation_labels)
        actual_relations = int(relation_emb.size(0))
        if actual_relations != expected_relations:
            log_event(
                logger,
                "relation_embeddings_mismatch",
                expected=expected_relations,
                actual=actual_relations,
                path=str(relation_emb_path),
            )
            encoder = _get_encoder()
            log_event(logger, "lmdb_encode_relation_embeddings", count=len(relation_labels))
            relation_emb = encoder.encode(
                relation_labels,
                cfg.batch_size,
                show_progress=cfg.progress_bar,
                desc="Relations",
            )
            torch.save(relation_emb, relation_emb_path)
    else:
        encoder = _get_encoder()
        emb_rows = sorted(zip(embedding_vocab["embedding_id"], embedding_vocab["label"]), key=lambda x: x[0])
        text_labels: List[str] = [str(label) for _, label in emb_rows]
        text_ids: List[int] = [int(eid) for eid, _ in emb_rows]
        log_event(logger, "lmdb_encode_entity_embeddings", count=len(text_labels))
        max_embedding_id = max(entity_vocab["embedding_id"]) if entity_vocab["embedding_id"] else 0
        encode_to_memmap(
            encoder=encoder,
            texts=text_labels,
            emb_ids=text_ids,
            batch_size=cfg.batch_size,
            max_embedding_id=max_embedding_id,
            out_path=entity_emb_path,
            desc="Entities",
            show_progress=cfg.progress_bar,
        )
        log_event(logger, "lmdb_encode_relation_embeddings", count=len(relation_labels))
        relation_emb = encoder.encode(relation_labels, cfg.batch_size, show_progress=cfg.progress_bar, desc="Relations")
        torch.save(relation_emb, relation_emb_path)

    graphs_table = _load_parquet(ctx.parquet_dir / "graphs.parquet")
    questions_table = _load_parquet(ctx.parquet_dir / "questions.parquet")
    questions_have_emb = "question_emb" in questions_table.schema.names
    if dataset_name and "dataset" in questions_table.schema.names:
        dataset_values = {str(val) for val in questions_table.column("dataset").unique().to_pylist() if val is not None}
        if len(dataset_values) > 1:
            raise RuntimeError(f"questions.parquet contains multiple dataset names: {sorted(dataset_values)}")
        if dataset_values and dataset_name not in dataset_values:
            raise RuntimeError("questions.parquet dataset mismatch: " f"expected={dataset_name} found={sorted(dataset_values)}")
    if use_precomputed_questions and not questions_have_emb:
        use_precomputed_questions = False
    graph_ids_all = graphs_table.column("graph_id")
    distinct = graph_ids_all.unique()
    if len(distinct) != len(graph_ids_all):
        from collections import Counter

        counts = Counter(graph_ids_all.to_pylist())
        dupes = [gid for gid, c in counts.items() if c > 1][:5]
        raise RuntimeError(f"Duplicate graph_id detected in graphs.parquet, examples: {dupes}")

    graph_id_list = graph_ids_all.to_pylist()
    graph_id_to_row: Dict[str, int] = {gid: idx for idx, gid in enumerate(graph_id_list)}
    graph_cols: Dict[str, List] = {
        "node_entity_ids": graphs_table.column("node_entity_ids").to_pylist(),
        "node_embedding_ids": graphs_table.column("node_embedding_ids").to_pylist(),
        "edge_src": graphs_table.column("edge_src").to_pylist(),
        "edge_dst": graphs_table.column("edge_dst").to_pylist(),
        "edge_relation_ids": graphs_table.column("edge_relation_ids").to_pylist(),
    }

    questions_rows = questions_table.num_rows
    log_event(logger, "lmdb_prepare_samples", samples=questions_rows)

    overwrite_lmdb = bool(cfg.get("overwrite_lmdb", True))
    lmdb_shards = _resolve_lmdb_shards(cfg)

    envs: Dict[str, Dict[int, lmdb.Environment]] = {}
    all_splits = questions_table.column("split").unique().to_pylist()
    all_splits = _validate_split_names(all_splits, context="questions.parquet")
    lmdb_stats = {str(split): {"samples": _ZERO, "nodes": _ZERO, "edges": _ZERO} for split in all_splits}
    label_entries = None
    label_stats = None
    labels_dir: Optional[Path] = None
    if emit_edge_retriever_labels:
        if labels_dir_cfg:
            labels_dir = Path(str(labels_dir_cfg))
        else:
            artifact_dir = dataset_cfg.get("artifact_dir")
            if not artifact_dir:
                raise ValueError("edge_retriever_labels_dir or dataset.artifact_dir is required when emitting labels.")
            labels_dir = Path(str(artifact_dir)) / "edge_retriever_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_entries = {str(split): {} for split in all_splits}
        label_stats = {
            str(split): {
                "num_samples": _ZERO,
                "no_path_samples": _ZERO,
                "zero_hop_samples": _ZERO,
                "reachable_all_samples": _ZERO,
                "reachable_partial_samples": _ZERO,
                "reachable_none_samples": _ZERO,
                "q_empty_samples": _ZERO,
                "a_empty_samples": _ZERO,
            }
            for split in all_splits
        }
    map_size_bytes, map_growth_bytes, map_growth_factor, map_max_bytes = _resolve_lmdb_map_config(cfg)
    map_sizes: Dict[str, Dict[int, int]] = {}
    tmp_dirs: Dict[str, Dict[int, Path]] = {}
    for split in all_splits:
        split_key = str(split)
        envs[split_key] = {}
        map_sizes[split_key] = {}
        tmp_dirs[split_key] = {}
        for shard_id in range(lmdb_shards):
            shard_key = int(shard_id)
            core_final_dir = _format_core_path(emb_dir, split, shard_key, lmdb_shards)
            core_tmp_dir = _prepare_lmdb_dir(core_final_dir, overwrite=overwrite_lmdb)
            tmp_dirs[split_key][shard_key] = core_tmp_dir
            map_sizes[split_key][shard_key] = map_size_bytes
            envs[split_key][shard_key] = lmdb.open(
                str(core_tmp_dir),
                map_size=map_sizes[split_key][shard_key],
                subdir=True,
                lock=False,
            )

    success = False
    try:
        txn_cache: Dict[str, Dict[int, lmdb.Transaction]] = {}
        pending_payloads: Dict[str, Dict[int, List[tuple[bytes, bytes]]]] = {}
        for split_key, shard_group in envs.items():
            txn_cache[split_key] = {}
            pending_payloads[split_key] = {}
            for shard_id, env in shard_group.items():
                txn_cache[split_key][shard_id] = env.begin(write=True)
                pending_payloads[split_key][shard_id] = []

        parquet_chunk_size = _resolve_parquet_chunk_size(cfg, fallback=int(cfg.get("batch_size", _MIN_CHUNK_SIZE)))
        question_batches = questions_table.to_batches(max_chunksize=parquet_chunk_size)
        total_batches = questions_table.num_rows / parquet_chunk_size
        from tqdm import tqdm

        pbar = tqdm(question_batches, total=int(total_batches) + 1, desc="Writing LMDB")
        missing_graph_ids: List[str] = []
        keep_start_ids: List[str] = []
        keep_answer_ids: List[str] = []
        missing_start = _ZERO
        missing_answer = _ZERO

        for q_batch in pbar:
            q_batch_dict = q_batch.to_pydict()
            if use_precomputed_questions:
                q_batch_emb_list = q_batch_dict.get("question_emb")
                if q_batch_emb_list is None:
                    raise RuntimeError("questions.parquet missing required column `question_emb`.")
                if any(emb is None for emb in q_batch_emb_list):
                    raise ValueError("question_emb contains null entries; rebuild with precomputed embeddings.")
                q_batch_emb = torch.tensor(q_batch_emb_list, dtype=torch.float32)
            else:
                q_batch_texts = [str(q) for q in q_batch_dict["question"]]
                q_batch_emb = _get_encoder().encode(q_batch_texts, cfg.batch_size, show_progress=False)

            graph_ids_batch = q_batch_dict["graph_id"]
            graph_row_indices: List[int] = []
            for gid in graph_ids_batch:
                idx = graph_id_to_row.get(gid)
                if idx is None:
                    missing_graph_ids.append(gid)
                graph_row_indices.append(idx)

            if missing_graph_ids:
                sample_ids = ", ".join(list(dict.fromkeys(missing_graph_ids))[:5])
                raise RuntimeError(f"Missing graph_id(s) in graphs.parquet, examples: {sample_ids}")

            for i in range(q_batch.num_rows):
                graph_id = graph_ids_batch[i]
                split = q_batch_dict["split"][i]
                if split not in envs:
                    continue

                g_idx = graph_row_indices[i]
                node_entity_ids = graph_cols["node_entity_ids"][g_idx]
                node_embedding_ids = graph_cols["node_embedding_ids"][g_idx]
                edge_src = graph_cols["edge_src"][g_idx]
                edge_dst = graph_cols["edge_dst"][g_idx]
                edge_rel = graph_cols["edge_relation_ids"][g_idx]

                num_nodes = len(node_entity_ids)
                num_edges = len(edge_src)
                if num_edges <= 0:
                    raise ValueError(
                        f"Invalid graph with zero edges for {graph_id} (split={split}). "
                        "Fix raw parquet/filters and rebuild; empty edge_index is unsupported."
                    )
                num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.long)
                node_global_ids = torch.tensor(node_entity_ids, dtype=torch.long)
                node_emb_ids = torch.tensor(node_embedding_ids, dtype=torch.long)
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_attr = torch.tensor(edge_rel, dtype=torch.long)

                q_entities = q_batch_dict["seed_entity_ids"][i] or []
                a_entities = q_batch_dict["answer_entity_ids"][i] or []
                if q_entities is None or len(q_entities) == 0:
                    missing_start += 1
                    continue
                q_local = _local_indices(node_entity_ids, q_entities)
                if not q_local:
                    missing_start += 1
                    continue
                a_local = _local_indices(node_entity_ids, a_entities)
                if not a_local:
                    missing_answer += 1
                keep_start_ids.append(graph_id)
                if a_local:
                    keep_answer_ids.append(graph_id)
                answer_entity_ids = torch.as_tensor(a_entities, dtype=torch.long)
                if emit_edge_retriever_labels and label_entries is not None and label_stats is not None:
                    split_key = str(split)
                    entries = label_entries[split_key]
                    stats = label_stats[split_key]
                    labels = compute_shortest_path_labels(
                        edge_index=edge_index,
                        q_local_indices=torch.as_tensor(q_local, dtype=torch.long),
                        a_local_indices=torch.as_tensor(a_local, dtype=torch.long),
                        num_nodes=int(num_nodes),
                    )
                    entries[graph_id] = {
                        "num_edges": int(labels.num_edges),
                        "positive_edge_ids": labels.positive_edge_ids,
                        "max_path_length": labels.max_path_length,
                    }
                    stats["num_samples"] += _ONE
                    if labels.max_path_length is None:
                        stats["no_path_samples"] += _ONE
                    elif int(labels.max_path_length) == _ZERO:
                        stats["zero_hop_samples"] += _ONE
                    if not q_local:
                        stats["q_empty_samples"] += _ONE
                    if not a_local:
                        stats["a_empty_samples"] += _ONE
                    reachable_count = _count_reachable_any_direction(
                        num_nodes,
                        edge_src,
                        edge_dst,
                        q_local,
                        a_local,
                    )
                    if a_local:
                        if reachable_count == len(a_local):
                            stats["reachable_all_samples"] += _ONE
                        elif reachable_count == _ZERO:
                            stats["reachable_none_samples"] += _ONE
                        else:
                            stats["reachable_partial_samples"] += _ONE

                split_key = str(split)
                lmdb_stats[split_key]["samples"] += 1
                lmdb_stats[split_key]["nodes"] += num_nodes
                lmdb_stats[split_key]["edges"] += num_edges
                core_sample = {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "num_nodes": num_nodes_tensor,
                    "node_global_ids": node_global_ids,
                    "node_embedding_ids": node_emb_ids,
                    "question_emb": q_batch_emb[i].unsqueeze(0),
                    "q_local_indices": torch.as_tensor(q_local, dtype=torch.long),
                    "a_local_indices": torch.as_tensor(a_local, dtype=torch.long),
                    "answer_entity_ids": answer_entity_ids,
                }

                sample_key = graph_id.encode("utf-8")
                shard_id = _assign_lmdb_shard(sample_key, lmdb_shards)
                core_payload = _serialize_sample(core_sample)
                pending_payloads[split][shard_id].append((sample_key, core_payload))
                txn = txn_cache[split][shard_id]
                try:
                    _write_sample(txn, sample_key, core_payload)
                except lmdb.MapFullError:
                    txn.abort()
                    txn, map_sizes[split][shard_id] = _replay_pending_with_growth(
                        env=envs[split][shard_id],
                        pending_payloads=pending_payloads[split][shard_id],
                        map_size_bytes=map_sizes[split][shard_id],
                        growth_bytes=map_growth_bytes,
                        growth_factor=map_growth_factor,
                        max_size_bytes=map_max_bytes,
                    )
                    txn_cache[split][shard_id] = txn
                if len(pending_payloads[split][shard_id]) >= cfg.txn_size:
                    txn_cache[split][shard_id], map_sizes[split][shard_id] = _commit_pending_with_growth(
                        env=envs[split][shard_id],
                        txn=txn_cache[split][shard_id],
                        pending_payloads=pending_payloads[split][shard_id],
                        map_size_bytes=map_sizes[split][shard_id],
                        growth_bytes=map_growth_bytes,
                        growth_factor=map_growth_factor,
                        max_size_bytes=map_max_bytes,
                    )
                    pending_payloads[split][shard_id].clear()

        for split_key, shard_group in txn_cache.items():
            for shard_id, txn_group in shard_group.items():
                if pending_payloads[split_key][shard_id]:
                    txn_cache[split_key][shard_id], map_sizes[split_key][shard_id] = _commit_pending_with_growth(
                        env=envs[split_key][shard_id],
                        txn=txn_group,
                        pending_payloads=pending_payloads[split_key][shard_id],
                        map_size_bytes=map_sizes[split_key][shard_id],
                        growth_bytes=map_growth_bytes,
                        growth_factor=map_growth_factor,
                        max_size_bytes=map_max_bytes,
                    )
                    pending_payloads[split_key][shard_id].clear()
        success = True
    finally:
        for shard_group in envs.values():
            for env in shard_group.values():
                env.close()
        if success:
            for split in all_splits:
                split_key = str(split)
                for shard_id in range(lmdb_shards):
                    shard_key = int(shard_id)
                    core_final_dir = _format_core_path(emb_dir, split, shard_key, lmdb_shards)
                    _finalize_lmdb_dir(
                        tmp_path=tmp_dirs[split_key][shard_key],
                        final_path=core_final_dir,
                        overwrite=overwrite_lmdb,
                    )
            total_samples = sum(stats["samples"] for stats in lmdb_stats.values())
            total_nodes = sum(stats["nodes"] for stats in lmdb_stats.values())
            total_edges = sum(stats["edges"] for stats in lmdb_stats.values())
            log_event(
                logger,
                "lmdb_write_summary_total",
                samples=total_samples,
                nodes=total_nodes,
                edges=total_edges,
                avg_nodes=total_nodes / total_samples if total_samples else 0.0,
                avg_edges=total_edges / total_samples if total_samples else 0.0,
            )
            for split_key, stats in lmdb_stats.items():
                log_event(
                    logger,
                    "lmdb_write_summary_split",
                    split=split_key,
                    samples=stats["samples"],
                    nodes=stats["nodes"],
                    edges=stats["edges"],
                    avg_nodes=stats["nodes"] / stats["samples"] if stats["samples"] else 0.0,
                    avg_edges=stats["edges"] / stats["samples"] if stats["samples"] else 0.0,
                )
            processed_dir = ctx.output_dir / "processed"
            ensure_dir(processed_dir)
            _write_sample_filter(
                processed_dir / _FILTER_MISSING_START_FILENAME,
                dataset=dataset_name,
                sample_ids=keep_start_ids,
            )
            _write_sample_filter(
                processed_dir / _FILTER_MISSING_ANSWER_FILENAME,
                dataset=dataset_name,
                sample_ids=keep_answer_ids,
            )
            log_event(
                logger,
                "missing_anchor_filters_written",
                missing_start=missing_start,
                missing_answer=missing_answer,
                keep_start=len(keep_start_ids),
                keep_answer=len(keep_answer_ids),
                path=str(processed_dir),
            )
            if emit_edge_retriever_labels and labels_dir is not None and label_entries is not None and label_stats is not None:
                manifest: Dict[str, object] = {"outputs": {}, "splits": list(label_entries.keys())}
                for split_key, entries in label_entries.items():
                    out_path = labels_dir / f"{split_key}.pt"
                    stats = label_stats.get(split_key, {})
                    payload = {
                        "meta": {
                            "algo": "edge_retriever_shortest_paths_strict_v1",
                            "split": split_key,
                            "num_samples": int(stats.get("num_samples", _ZERO)),
                            "no_path_samples": int(stats.get("no_path_samples", _ZERO)),
                            "zero_hop_samples": int(stats.get("zero_hop_samples", _ZERO)),
                            "reachable_all_samples": int(stats.get("reachable_all_samples", _ZERO)),
                            "reachable_partial_samples": int(stats.get("reachable_partial_samples", _ZERO)),
                            "reachable_none_samples": int(stats.get("reachable_none_samples", _ZERO)),
                            "q_empty_samples": int(stats.get("q_empty_samples", _ZERO)),
                            "a_empty_samples": int(stats.get("a_empty_samples", _ZERO)),
                        },
                        "entries": entries,
                    }
                    torch.save(payload, out_path)
                    manifest["outputs"][split_key] = str(out_path)
                (labels_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                log_event(
                    logger,
                    "edge_retriever_labels_written",
                    path=str(labels_dir),
                    splits=list(label_entries.keys()),
                    stats=label_stats,
                )
