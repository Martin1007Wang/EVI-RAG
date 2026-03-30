from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
from collections import deque

import json

import lmdb
import pyarrow as pa
import torch

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap

from src.data.preprocess.context import PreprocessContext
from src.data.io.lmdb_utils import (
    assign_lmdb_shard,
    _commit_pending_with_growth,
    _finalize_lmdb_dir,
    _format_lmdb_path,
    _local_indices,
    _prepare_lmdb_dir,
    _retry_pending_with_growth,
    _resolve_lmdb_map_config,
    _resolve_lmdb_shards,
    _serialize_sample,
    _write_sample,
    ensure_dir,
)
from src.data.io.parquet_io import _load_parquet
from src.data.io.runtime_sample_metadata import (
    runtime_sample_metadata_path,
    save_runtime_sample_metadata,
)
from src.data.preprocess.labels.edge_retrieval import compute_shortest_path_labels
from src.data.schema.constants import (
    _FILTER_MISSING_ANSWER_FILENAME,
    _FILTER_MISSING_ANCHOR_FILENAME,
)
from src.data.preprocess.config import (
    _resolve_parquet_chunk_size,
    resolve_embedding_batch_size,
    resolve_embedding_device,
    resolve_embedding_fp16,
)
from src.data.utils.validation import _validate_split_names
from src.utils.logging_utils import log_event


def _format_core_path(
    base_dir: Path, split: str, shard_id: int, num_shards: int
) -> Path:
    return _format_lmdb_path(base_dir, split, shard_id, num_shards, suffix=".lmdb")


def _write_sample_filter(path: Path, *, dataset: str, sample_ids: List[str]) -> None:
    payload = {
        "dataset": dataset,
        "sample_ids": sorted(sample_ids),
    }
    path.write_text(json.dumps(payload, indent=2))


def _new_runtime_sample_metadata_buffers(
    splits: List[str],
) -> Dict[str, Dict[str, List[object]]]:
    return {
        str(split): {
            "sample_ids": [],
            "questions": [],
            "num_nodes": [],
            "num_edges": [],
            "question_tokens": [],
        }
        for split in splits
    }


def _bfs_multi(
    num_nodes: int, adjacency: List[List[int]], sources: List[int]
) -> List[int]:
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
        du = dist[u] + 1
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
    anchor_nodes: List[int],
    a_nodes: List[int],
) -> int:
    if num_nodes <= 0 or not anchor_nodes or not a_nodes:
        return 0
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
    dist_fwd = _bfs_multi(num_nodes, adjacency, anchor_nodes)
    dist_rev = _bfs_multi(num_nodes, rev_adjacency, anchor_nodes)
    reachable = 0
    for a_raw in a_nodes:
        a = int(a_raw)
        if 0 <= a < num_nodes and (dist_fwd[a] >= 0 or dist_rev[a] >= 0):
            reachable += 1
    return reachable


def _require_question_entity_ids_column(
    q_batch_dict: Dict[str, Sequence[object]],
) -> Sequence[object]:
    if "question_entity_ids" in q_batch_dict:
        return q_batch_dict["question_entity_ids"]
    if "seed_entity_ids" in q_batch_dict:
        raise ValueError(
            "questions.parquet still uses legacy seed_entity_ids; rebuild parquet outputs "
            "to emit question_entity_ids before LMDB materialization."
        )
    raise ValueError(
        "questions.parquet is missing question_entity_ids; rebuild parquet outputs before "
        "LMDB materialization."
    )


def _expected_entity_embedding_rows(entity_vocab: Dict[str, List[object]]) -> int:
    embedding_ids = [int(emb_id) for emb_id in entity_vocab.get("embedding_id", [])]
    if not embedding_ids:
        return 1
    return max(embedding_ids) + 1


def _build_entity_embedding_inputs(
    entity_vocab: Dict[str, List[object]],
    embedding_vocab: Dict[str, List[object]],
) -> tuple[list[str], list[int], int]:
    emb_rows = sorted(
        zip(embedding_vocab["embedding_id"], embedding_vocab["label"]),
        key=lambda item: int(item[0]),
    )
    text_labels = [str(label) for _, label in emb_rows]
    text_ids = [int(emb_id) for emb_id, _ in emb_rows]
    max_embedding_id = _expected_entity_embedding_rows(entity_vocab) - 1
    return text_labels, text_ids, max_embedding_id


def _take_graph_batch_columns(
    graphs_table, row_indices: List[int]
) -> tuple[dict[int, int], Dict[str, List[object]]]:
    unique_row_indices = list(dict.fromkeys(int(row_idx) for row_idx in row_indices))
    graph_batch = graphs_table.take(pa.array(unique_row_indices, type=pa.int64()))
    row_lookup = {
        row_idx: local_idx for local_idx, row_idx in enumerate(unique_row_indices)
    }
    return row_lookup, graph_batch.to_pydict()


def _resolve_local_entity_indices(
    node_entity_ids: Sequence[object],
    target_entity_ids: Sequence[object],
) -> List[int]:
    node_entity_id_list = [int(value) for value in list(node_entity_ids)]
    target_entity_id_list = [int(value) for value in list(target_entity_ids)]
    return _local_indices(node_entity_id_list, target_entity_id_list)


def build_dataset(ctx: PreprocessContext) -> None:
    cfg = ctx.cfg
    logger = ctx.logger
    dataset_cfg = cfg.get("dataset") if hasattr(cfg, "get") else {}
    dataset_name = str(dataset_cfg.get("name", "") or "")
    dataset_scope = str(dataset_cfg.get("dataset_scope", "") or "").strip().lower()
    emit_edge_retrieval_labels = bool(cfg.get("emit_edge_retrieval_labels", False))
    labels_dir_cfg = cfg.get("edge_retrieval_labels_dir")
    log_event(
        logger,
        "lmdb_start",
        dataset=dataset_name,
        output_dir=str(ctx.output_dir),
        out_dir=str(ctx.out_dir),
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

    entity_vocab = _load_parquet(ctx.out_dir / "entity_vocab.parquet").to_pydict()
    embedding_vocab = _load_parquet(ctx.out_dir / "embedding_vocab.parquet").to_pydict()
    relation_vocab = _load_parquet(ctx.out_dir / "relation_vocab.parquet").to_pydict()

    relation_rows = sorted(
        zip(relation_vocab["relation_id"], relation_vocab["label"]), key=lambda x: x[0]
    )
    relation_labels: List[str] = [str(label) for _, label in relation_rows]
    text_labels, text_ids, max_embedding_id = _build_entity_embedding_inputs(
        entity_vocab,
        embedding_vocab,
    )
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))
    question_ctx_max_tokens = int(cfg.get("question_ctx_max_tokens", 0))
    if question_ctx_max_tokens < 0:
        raise ValueError(
            f"question_ctx_max_tokens must be >= 0, got {question_ctx_max_tokens}."
        )
    need_question_ctx = question_ctx_max_tokens > 0
    embedding_device = resolve_embedding_device(cfg.get("device"))
    embedding_batch_size = resolve_embedding_batch_size(cfg, device=embedding_device)
    embedding_fp16 = resolve_embedding_fp16(cfg, device=embedding_device)
    log_event(
        logger,
        "embedding_runtime",
        device=embedding_device,
        batch_size=embedding_batch_size,
        fp16=embedding_fp16,
    )

    emb_dir = ctx.embeddings_dir
    ensure_dir(emb_dir)
    entity_emb_path = emb_dir / "entity_embeddings.pt"
    relation_emb_path = emb_dir / "relation_embeddings.pt"

    encoder: TextEncoder | None = None

    def _get_encoder() -> TextEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = TextEncoder(
                cfg.encoder,
                embedding_device,
                embedding_fp16,
                cfg.progress_bar,
            )
        return encoder

    def _encode_entity_embeddings() -> None:
        log_event(logger, "lmdb_encode_entity_embeddings", count=len(text_labels))
        encode_to_memmap(
            encoder=_get_encoder(),
            texts=text_labels,
            emb_ids=text_ids,
            batch_size=embedding_batch_size,
            max_embedding_id=max_embedding_id,
            out_path=entity_emb_path,
            desc="Entities",
            show_progress=cfg.progress_bar,
        )

    def _encode_relation_embeddings() -> torch.Tensor:
        log_event(logger, "lmdb_encode_relation_embeddings", count=len(relation_labels))
        relation_emb = _get_encoder().encode(
            relation_labels,
            embedding_batch_size,
            show_progress=cfg.progress_bar,
            desc="Relations",
        )
        torch.save(relation_emb, relation_emb_path)
        return relation_emb

    if not use_precomputed_embeddings and reuse_embeddings_if_exists:
        if entity_emb_path.exists() and relation_emb_path.exists():
            log_event(logger, "lmdb_reuse_embeddings", path=str(emb_dir))
            use_precomputed_embeddings = True

    if use_precomputed_embeddings:
        missing_paths = [
            str(p) for p in (entity_emb_path, relation_emb_path) if not p.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(f"Precomputed embeddings missing: {missing_paths}")
        entity_emb = torch.load(entity_emb_path, map_location="cpu")
        expected_entity_rows = max_embedding_id + 1
        actual_entity_rows = int(entity_emb.size(0))
        if actual_entity_rows != expected_entity_rows:
            log_event(
                logger,
                "entity_embeddings_mismatch",
                expected=expected_entity_rows,
                actual=actual_entity_rows,
                path=str(entity_emb_path),
            )
            _encode_entity_embeddings()
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
            relation_emb = _encode_relation_embeddings()
    else:
        _encode_entity_embeddings()
        relation_emb = _encode_relation_embeddings()

    graphs_table = _load_parquet(ctx.out_dir / "graphs.parquet")
    questions_table = _load_parquet(ctx.out_dir / "questions.parquet")
    questions_have_emb = "question_emb" in questions_table.schema.names
    questions_have_ctx = "question_ctx" in questions_table.schema.names
    questions_have_ctx_mask = "question_ctx_mask" in questions_table.schema.names
    if questions_have_ctx != questions_have_ctx_mask:
        raise RuntimeError(
            "questions.parquet must contain question_ctx and question_ctx_mask together."
        )
    if dataset_name and "dataset" in questions_table.schema.names:
        dataset_values = {
            str(val)
            for val in questions_table.column("dataset").unique().to_pylist()
            if val is not None
        }
        if len(dataset_values) > 1:
            raise RuntimeError(
                f"questions.parquet contains multiple dataset names: {sorted(dataset_values)}"
            )
        if dataset_values and dataset_name not in dataset_values:
            raise RuntimeError(
                "questions.parquet dataset mismatch: "
                f"expected={dataset_name} found={sorted(dataset_values)}"
            )
    if use_precomputed_questions and not questions_have_emb:
        use_precomputed_questions = False
    graph_ids_all = graphs_table.column("graph_id")
    distinct = graph_ids_all.unique()
    if len(distinct) != len(graph_ids_all):
        from collections import Counter

        counts = Counter(graph_ids_all.to_pylist())
        dupes = [gid for gid, c in counts.items() if c > 1][:5]
        raise RuntimeError(
            f"Duplicate graph_id detected in graphs.parquet, examples: {dupes}"
        )

    graph_id_list = graph_ids_all.to_pylist()
    graph_id_to_row: Dict[str, int] = {
        gid: idx for idx, gid in enumerate(graph_id_list)
    }

    questions_rows = questions_table.num_rows
    log_event(logger, "lmdb_prepare_samples", samples=questions_rows)

    overwrite_lmdb = bool(cfg.get("overwrite_lmdb", True))
    lmdb_shards = _resolve_lmdb_shards(cfg)

    envs: Dict[str, Dict[int, lmdb.Environment]] = {}
    all_splits = questions_table.column("split").unique().to_pylist()
    all_splits = _validate_split_names(all_splits, context="questions.parquet")
    lmdb_stats = {
        str(split): {"samples": 0, "nodes": 0, "edges": 0} for split in all_splits
    }
    runtime_sample_metadata = _new_runtime_sample_metadata_buffers(all_splits)
    label_entries = None
    label_stats = None
    labels_dir: Optional[Path] = None
    if emit_edge_retrieval_labels:
        if labels_dir_cfg:
            labels_dir = Path(str(labels_dir_cfg))
        else:
            artifact_dir = dataset_cfg.get("artifact_dir")
            if not artifact_dir:
                raise ValueError(
                    "edge_retrieval_labels_dir or dataset.artifact_dir is required when emitting labels."
                )
            labels_dir = Path(str(artifact_dir)) / "edge_retrieval_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_entries = {str(split): {} for split in all_splits}
        label_stats = {
            str(split): {
                "num_samples": 0,
                "no_path_samples": 0,
                "zero_hop_samples": 0,
                "reachable_all_samples": 0,
                "reachable_partial_samples": 0,
                "reachable_none_samples": 0,
                "anchor_empty_samples": 0,
                "a_empty_samples": 0,
            }
            for split in all_splits
        }
    map_size_bytes, map_growth_bytes, map_growth_factor, map_max_bytes = (
        _resolve_lmdb_map_config(cfg)
    )
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

        parquet_chunk_size = _resolve_parquet_chunk_size(
            cfg, fallback=embedding_batch_size
        )
        question_batches = questions_table.to_batches(max_chunksize=parquet_chunk_size)
        total_batches = questions_table.num_rows / parquet_chunk_size
        from tqdm import tqdm

        pbar = tqdm(question_batches, total=int(total_batches) + 1, desc="Writing LMDB")
        missing_graph_ids: List[str] = []
        keep_anchor_ids: List[str] = []
        keep_answer_ids: List[str] = []
        missing_anchor = 0
        missing_answer = 0

        for q_batch in pbar:
            q_batch_dict = q_batch.to_pydict()
            question_entity_ids_column = _require_question_entity_ids_column(
                q_batch_dict
            )
            q_batch_ctx = None
            q_batch_ctx_mask = None
            if use_precomputed_questions:
                q_batch_emb_list = q_batch_dict.get("question_emb")
                if q_batch_emb_list is None:
                    raise RuntimeError(
                        "questions.parquet missing required column `question_emb`."
                    )
                if any(emb is None for emb in q_batch_emb_list):
                    raise ValueError(
                        "question_emb contains null entries; rebuild with precomputed embeddings."
                    )
                q_batch_emb = torch.tensor(q_batch_emb_list, dtype=torch.float32)
                if need_question_ctx:
                    if questions_have_ctx:
                        q_batch_ctx_list = q_batch_dict.get("question_ctx")
                        q_batch_ctx_mask_list = q_batch_dict.get("question_ctx_mask")
                        if q_batch_ctx_list is None or q_batch_ctx_mask_list is None:
                            raise RuntimeError(
                                "questions.parquet missing required columns "
                                "`question_ctx`/`question_ctx_mask` for token-level question context."
                            )
                        if any(ctx is None for ctx in q_batch_ctx_list):
                            raise ValueError(
                                "question_ctx contains null entries; rebuild with complete context."
                            )
                        if any(mask is None for mask in q_batch_ctx_mask_list):
                            raise ValueError(
                                "question_ctx_mask contains null entries; rebuild with complete context."
                            )
                        q_batch_ctx = torch.tensor(
                            q_batch_ctx_list, dtype=torch.float32
                        )
                        q_batch_ctx_mask = torch.tensor(
                            q_batch_ctx_mask_list, dtype=torch.bool
                        )
                        if q_batch_ctx.dim() != 3:
                            raise ValueError(
                                "question_ctx must be 3D [B, L, d], "
                                f"got shape={tuple(q_batch_ctx.shape)}."
                            )
                        if q_batch_ctx_mask.dim() != 2:
                            raise ValueError(
                                "question_ctx_mask must be 2D [B, L], "
                                f"got shape={tuple(q_batch_ctx_mask.shape)}."
                            )
                        if tuple(q_batch_ctx_mask.shape) != tuple(
                            q_batch_ctx.shape[:2]
                        ):
                            raise ValueError(
                                "question_ctx_mask shape mismatch with question_ctx: "
                                f"mask={tuple(q_batch_ctx_mask.shape)}, ctx={tuple(q_batch_ctx.shape[:2])}."
                            )
                    else:
                        q_batch_texts = [str(q) for q in q_batch_dict["question"]]
                        _, q_batch_ctx, q_batch_ctx_mask = (
                            _get_encoder().encode_with_context(
                                q_batch_texts,
                                embedding_batch_size,
                                max_tokens=question_ctx_max_tokens,
                                show_progress=False,
                                desc="QuestionsWithContext",
                            )
                        )
            else:
                q_batch_texts = [str(q) for q in q_batch_dict["question"]]
                if need_question_ctx:
                    q_batch_emb, q_batch_ctx, q_batch_ctx_mask = (
                        _get_encoder().encode_with_context(
                            q_batch_texts,
                            embedding_batch_size,
                            max_tokens=question_ctx_max_tokens,
                            show_progress=False,
                            desc="QuestionsWithContext",
                        )
                    )
                else:
                    q_batch_emb = _get_encoder().encode(
                        q_batch_texts, embedding_batch_size, show_progress=False
                    )

            graph_ids_batch = q_batch_dict["graph_id"]
            graph_row_indices: List[int] = []
            for gid in graph_ids_batch:
                row_idx = graph_id_to_row.get(gid)
                if row_idx is None:
                    missing_graph_ids.append(gid)
                    continue
                graph_row_indices.append(int(row_idx))

            if missing_graph_ids:
                sample_ids = ", ".join(list(dict.fromkeys(missing_graph_ids))[:5])
                raise RuntimeError(
                    f"Missing graph_id(s) in graphs.parquet, examples: {sample_ids}"
                )

            graph_row_lookup, graph_batch_cols = _take_graph_batch_columns(
                graphs_table,
                graph_row_indices,
            )

            for i in range(q_batch.num_rows):
                graph_id = graph_ids_batch[i]
                split = str(q_batch_dict["split"][i])
                if split not in envs:
                    continue

                g_idx = graph_row_indices[i]
                graph_local_idx = graph_row_lookup[g_idx]
                node_entity_ids = graph_batch_cols["node_entity_ids"][graph_local_idx]
                node_embedding_ids = graph_batch_cols["node_embedding_ids"][
                    graph_local_idx
                ]
                edge_src = graph_batch_cols["edge_src"][graph_local_idx]
                edge_dst = graph_batch_cols["edge_dst"][graph_local_idx]
                edge_rel = graph_batch_cols["edge_relation_ids"][graph_local_idx]

                num_nodes = len(node_entity_ids)
                num_edges = len(edge_src)
                if num_edges <= 0:
                    raise ValueError(
                        f"Invalid graph with zero edges for {graph_id} (split={split}). "
                        "Fix raw parquet/filters and rebuild; empty edge_index is unsupported."
                    )
                num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.long)
                question_entity_ids = question_entity_ids_column[i] or []
                answer_entity_ids_raw = q_batch_dict["answer_entity_ids"][i] or []
                anchor_local_indices = _resolve_local_entity_indices(
                    node_entity_ids, question_entity_ids
                )
                a_local = _resolve_local_entity_indices(
                    node_entity_ids, answer_entity_ids_raw
                )

                node_entity_ids = torch.tensor(node_entity_ids, dtype=torch.long)
                node_emb_ids = torch.tensor(node_embedding_ids, dtype=torch.long)
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_attr = torch.tensor(edge_rel, dtype=torch.long)

                if question_entity_ids is None or len(question_entity_ids) == 0:
                    missing_anchor += 1
                    continue
                if not anchor_local_indices:
                    missing_anchor += 1
                    continue
                if not a_local:
                    missing_answer += 1
                keep_anchor_ids.append(graph_id)
                if a_local:
                    keep_answer_ids.append(graph_id)
                answer_entity_ids = torch.as_tensor(
                    answer_entity_ids_raw, dtype=torch.long
                )
                if (
                    emit_edge_retrieval_labels
                    and label_entries is not None
                    and label_stats is not None
                ):
                    split_key = split
                    entries = label_entries[split_key]
                    stats = label_stats[split_key]
                    labels = compute_shortest_path_labels(
                        edge_index=edge_index,
                        anchor_local_indices=torch.as_tensor(
                            anchor_local_indices, dtype=torch.long
                        ),
                        a_local_indices=torch.as_tensor(a_local, dtype=torch.long),
                        num_nodes=int(num_nodes),
                    )
                    entries[graph_id] = {
                        "num_edges": int(labels.num_edges),
                        "positive_edge_ids": labels.positive_edge_ids,
                        "max_path_length": labels.max_path_length,
                    }
                    stats["num_samples"] += 1
                    if labels.max_path_length is None:
                        stats["no_path_samples"] += 1
                    elif int(labels.max_path_length) == 0:
                        stats["zero_hop_samples"] += 1
                    if not anchor_local_indices:
                        stats["anchor_empty_samples"] += 1
                    if not a_local:
                        stats["a_empty_samples"] += 1
                    reachable_count = _count_reachable_any_direction(
                        num_nodes,
                        edge_src,
                        edge_dst,
                        anchor_local_indices,
                        a_local,
                    )
                    if a_local:
                        if reachable_count == len(a_local):
                            stats["reachable_all_samples"] += 1
                        elif reachable_count == 0:
                            stats["reachable_none_samples"] += 1
                        else:
                            stats["reachable_partial_samples"] += 1

                split_key = split
                lmdb_stats[split_key]["samples"] += 1
                lmdb_stats[split_key]["nodes"] += num_nodes
                lmdb_stats[split_key]["edges"] += num_edges
                runtime_sample_metadata[split_key]["sample_ids"].append(str(graph_id))
                runtime_sample_metadata[split_key]["questions"].append(
                    str(q_batch_dict["question"][i])
                )
                runtime_sample_metadata[split_key]["num_nodes"].append(int(num_nodes))
                runtime_sample_metadata[split_key]["num_edges"].append(int(num_edges))
                runtime_sample_metadata[split_key]["question_tokens"].append(
                    int(q_batch_ctx_mask[i].sum().item())
                    if q_batch_ctx_mask is not None
                    else 1
                )
                core_sample = {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "num_nodes": num_nodes_tensor,
                    "node_entity_ids": node_entity_ids,
                    "node_embedding_ids": node_emb_ids,
                    "question_emb": q_batch_emb[i].unsqueeze(0),
                    "anchor_local_indices": torch.as_tensor(
                        anchor_local_indices, dtype=torch.long
                    ),
                    "a_local_indices": torch.as_tensor(a_local, dtype=torch.long),
                    "answer_entity_ids": answer_entity_ids,
                }
                if need_question_ctx:
                    if q_batch_ctx is None or q_batch_ctx_mask is None:
                        raise RuntimeError(
                            "question_ctx generation failed while question_ctx_max_tokens > 0."
                        )
                    core_sample["question_ctx"] = q_batch_ctx[i].unsqueeze(0)
                    core_sample["question_ctx_mask"] = q_batch_ctx_mask[i].unsqueeze(0)

                sample_key = graph_id.encode("utf-8")
                shard_id = assign_lmdb_shard(sample_key, lmdb_shards)
                core_payload = _serialize_sample(core_sample)
                pending_payloads[split][shard_id].append((sample_key, core_payload))
                txn = txn_cache[split][shard_id]
                try:
                    _write_sample(txn, sample_key, core_payload)
                except lmdb.MapFullError:
                    txn.abort()
                    txn, map_sizes[split][shard_id] = _retry_pending_with_growth(
                        env=envs[split][shard_id],
                        pending_payloads=pending_payloads[split][shard_id],
                        map_size_bytes=map_sizes[split][shard_id],
                        growth_bytes=map_growth_bytes,
                        growth_factor=map_growth_factor,
                        max_size_bytes=map_max_bytes,
                    )
                    txn_cache[split][shard_id] = txn
                if len(pending_payloads[split][shard_id]) >= cfg.txn_size:
                    txn_cache[split][shard_id], map_sizes[split][shard_id] = (
                        _commit_pending_with_growth(
                            env=envs[split][shard_id],
                            txn=txn_cache[split][shard_id],
                            pending_payloads=pending_payloads[split][shard_id],
                            map_size_bytes=map_sizes[split][shard_id],
                            growth_bytes=map_growth_bytes,
                            growth_factor=map_growth_factor,
                            max_size_bytes=map_max_bytes,
                        )
                    )
                    pending_payloads[split][shard_id].clear()

        for split_key, shard_group in txn_cache.items():
            for shard_id, txn_group in shard_group.items():
                if pending_payloads[split_key][shard_id]:
                    txn_cache[split_key][shard_id], map_sizes[split_key][shard_id] = (
                        _commit_pending_with_growth(
                            env=envs[split_key][shard_id],
                            txn=txn_group,
                            pending_payloads=pending_payloads[split_key][shard_id],
                            map_size_bytes=map_sizes[split_key][shard_id],
                            growth_bytes=map_growth_bytes,
                            growth_factor=map_growth_factor,
                            max_size_bytes=map_max_bytes,
                        )
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
                    core_final_dir = _format_core_path(
                        emb_dir, split, shard_key, lmdb_shards
                    )
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
                    avg_nodes=stats["nodes"] / stats["samples"]
                    if stats["samples"]
                    else 0.0,
                    avg_edges=stats["edges"] / stats["samples"]
                    if stats["samples"]
                    else 0.0,
                )
            processed_dir = ctx.output_dir / "processed"
            ensure_dir(processed_dir)
            _write_sample_filter(
                processed_dir / _FILTER_MISSING_ANCHOR_FILENAME,
                dataset=dataset_name,
                sample_ids=keep_anchor_ids,
            )
            _write_sample_filter(
                processed_dir / _FILTER_MISSING_ANSWER_FILENAME,
                dataset=dataset_name,
                sample_ids=keep_answer_ids,
            )
            log_event(
                logger,
                "missing_anchor_filters_written",
                missing_anchor=missing_anchor,
                missing_answer=missing_answer,
                keep_anchor=len(keep_anchor_ids),
                keep_answer=len(keep_answer_ids),
                path=str(processed_dir),
            )
            for split_key, split_metadata in runtime_sample_metadata.items():
                metadata_path = runtime_sample_metadata_path(emb_dir, split_key)
                save_runtime_sample_metadata(
                    metadata_path,
                    split=split_key,
                    sample_ids=split_metadata["sample_ids"],
                    questions=split_metadata["questions"],
                    num_nodes=split_metadata["num_nodes"],
                    num_edges=split_metadata["num_edges"],
                    question_tokens=split_metadata["question_tokens"],
                )
            log_event(
                logger,
                "runtime_sample_metadata_written",
                path=str(emb_dir),
                splits=list(runtime_sample_metadata.keys()),
            )
            if (
                emit_edge_retrieval_labels
                and labels_dir is not None
                and label_entries is not None
                and label_stats is not None
            ):
                manifest: Dict[str, object] = {
                    "outputs": {},
                    "splits": list(label_entries.keys()),
                }
                for split_key, entries in label_entries.items():
                    out_path = labels_dir / f"{split_key}.pt"
                    stats = label_stats.get(split_key, {})
                    payload = {
                        "meta": {
                            "algo": "edge_retrieval_shortest_paths_strict_v1",
                            "split": split_key,
                            "num_samples": int(stats.get("num_samples", 0)),
                            "no_path_samples": int(stats.get("no_path_samples", 0)),
                            "zero_hop_samples": int(stats.get("zero_hop_samples", 0)),
                            "reachable_all_samples": int(
                                stats.get("reachable_all_samples", 0)
                            ),
                            "reachable_partial_samples": int(
                                stats.get("reachable_partial_samples", 0)
                            ),
                            "reachable_none_samples": int(
                                stats.get("reachable_none_samples", 0)
                            ),
                            "anchor_empty_samples": int(
                                stats.get("anchor_empty_samples", 0)
                            ),
                            "a_empty_samples": int(stats.get("a_empty_samples", 0)),
                        },
                        "entries": entries,
                    }
                    torch.save(payload, out_path)
                    manifest["outputs"][split_key] = str(out_path)
                (labels_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                log_event(
                    logger,
                    "edge_retrieval_labels_written",
                    path=str(labels_dir),
                    splits=list(label_entries.keys()),
                    stats=label_stats,
                )
