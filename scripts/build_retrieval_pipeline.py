#!/usr/bin/env python3
"""End-to-end retrieval preprocessing: normalize parquet + materialize LMDB.

Targets Hydra data configs (e.g., WebQSP/CWQ/GTSQA/KGQAGen) under
/mnt/data/retrieval_dataset/<dataset>/raw/*.

Stage 1 (normalize):
  - Build normalized parquet (graphs/questions/vocabs).
  - Compute undirected shortest-path supervision.
  - Optionally precompute entity/relation/question embeddings.

Stage 2 (materialize):
  - Build LMDB caches for G_retrieval from normalized parquet.
  - Reuse precomputed embeddings/questions when configured.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import lmdb
import torch
import torch.nn.functional as F
try:
    from omegaconf import DictConfig
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = object  # type: ignore[assignment]
from tqdm import tqdm

try:
    from scripts.text_encode_utils import TextEncoder, encode_to_memmap
except ModuleNotFoundError:
    from text_encode_utils import TextEncoder, encode_to_memmap


@dataclass(frozen=True)
class SplitFilter:
    skip_no_topic: bool
    skip_no_ans: bool
    skip_no_path: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    encoder: str
    device: str
    batch_size: int
    fp16: bool
    progress_bar: bool
    embeddings_out_dir: Path
    precompute_entities: bool
    precompute_relations: bool
    precompute_questions: bool
    canonicalize_relations: bool
    cosine_eps: float


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    question_id: str
    kb: str
    question: str
    graph: List[Tuple[str, str, str]]
    q_entity: List[str]
    a_entity: List[str]
    answer_texts: List[str]
    answer_subgraph: List[Tuple[str, str, str]] = field(default_factory=list)
    graph_iso_type: Optional[str] = None
    redundant: Optional[bool] = None
    test_type: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextEntityConfig:
    mode: str
    prefixes: Tuple[str, ...]
    regex: Optional[re.Pattern]

    def is_text(self, entity: str) -> bool:
        if self.mode == "prefix_allowlist":
            return any(entity.startswith(prefix) for prefix in self.prefixes)
        if self.mode == "regex":
            return bool(self.regex.match(entity)) if self.regex is not None else False
        raise ValueError(f"Unsupported entity_text_mode: {self.mode}")


@dataclass
class GraphRecord:
    graph_id: str
    node_entity_ids: List[int]
    node_embedding_ids: List[int]
    node_labels: List[str]
    edge_src: List[int]
    edge_dst: List[int]
    edge_relation_ids: List[int]
    # Triple-level positives: union of shortest-path edges per (seed, answer) pair.
    positive_triple_mask: List[bool]
    # Pair-level shortest-path supervision (CSR-style).
    pair_start_node_locals: List[int]
    pair_answer_node_locals: List[int]
    pair_edge_local_ids: List[int]
    pair_edge_counts: List[int]
    pair_shortest_lengths: List[int]


@dataclass(frozen=True)
class EntityLookup:
    entity_to_struct: Dict[str, int]
    text_kg_id_to_embed_id: Dict[str, int]

    def entity_id(self, ent: str) -> int:
        idx = self.entity_to_struct.get(ent)
        if idx is None:
            raise KeyError(f"Unknown entity id: {ent}")
        return idx

    def embedding_id(self, ent: str) -> int:
        return self.text_kg_id_to_embed_id.get(ent, _NON_TEXT_EMBEDDING_ID)


@dataclass(frozen=True)
class RelationLookup:
    rel_to_id: Dict[str, int]

    def relation_id(self, rel: str) -> int:
        idx = self.rel_to_id.get(rel)
        if idx is None:
            raise KeyError(f"Unknown relation id: {rel}")
        return idx


class EntityVocab:
    """Assign structural IDs and embedding IDs; separate text vs non-text."""

    def __init__(self, kb: str, text_cfg: TextEntityConfig) -> None:
        self.kb = kb
        self._text_cfg = text_cfg
        self._entity_to_struct: Dict[str, int] = {}
        self._struct_records: List[Dict[str, object]] = []
        self._embedding_records: List[Dict[str, object]] = []
        self._text_entities: List[str] = []
        self._finalized = False
        self._text_kg_id_to_embed_id: Dict[str, int] = {}  # O(1) lookup

    def add_entity(self, ent: str) -> int:
        if ent in self._entity_to_struct:
            return self._entity_to_struct[ent]
        if self._finalized:
            raise RuntimeError("Cannot add entities after finalize")
        idx = len(self._entity_to_struct)
        self._entity_to_struct[ent] = idx
        if self._text_cfg.is_text(ent):
            self._text_entities.append(ent)
        else:
            # non-text handled via embedding_id=0; no separate list needed
            pass
        return idx

    def finalize(self) -> None:
        if self._finalized:
            return
        self._text_entities = sorted(self._text_entities)
        text_embedding_offset = 1
        text_embedding_ids: Dict[str, int] = {
            ent: text_embedding_offset + idx for idx, ent in enumerate(self._text_entities)
        }
        for ent in sorted(self._entity_to_struct, key=self._entity_to_struct.get):
            struct_id = self._entity_to_struct[ent]
            is_text = self._text_cfg.is_text(ent)
            embedding_id = text_embedding_ids.get(ent, 0)
            record = {
                "entity_id": struct_id,
                "kb": self.kb,
                "kg_id": ent,
                "label": ent,
                "is_text": is_text,
                "embedding_id": embedding_id,
            }
            self._struct_records.append(record)
            if is_text:
                self._embedding_records.append(
                    {
                        "embedding_id": embedding_id,
                        "kb": self.kb,
                        "kg_id": ent,
                        "label": ent,
                    }
                )
                self._text_kg_id_to_embed_id[ent] = embedding_id
        self._finalized = True

    @property
    def struct_records(self) -> List[Dict[str, object]]:
        if not self._finalized:
            self.finalize()
        return self._struct_records

    @property
    def embedding_records(self) -> List[Dict[str, object]]:
        if not self._finalized:
            self.finalize()
        return self._embedding_records

    def entity_id(self, ent: str) -> int:
        return self._entity_to_struct[ent]

    def embedding_id(self, ent: str) -> int:
        return _NON_TEXT_EMBEDDING_ID if not self._text_cfg.is_text(ent) else self._embedding_id_for_text(ent)

    def _embedding_id_for_text(self, ent: str) -> int:
        if not self._finalized:
            self.finalize()
        return self._text_kg_id_to_embed_id.get(ent, _NON_TEXT_EMBEDDING_ID)

    def to_lookup(self) -> EntityLookup:
        if not self._finalized:
            self.finalize()
        return EntityLookup(
            entity_to_struct=dict(self._entity_to_struct),
            text_kg_id_to_embed_id=dict(self._text_kg_id_to_embed_id),
        )


class RelationVocab:
    def __init__(self, kb: str) -> None:
        self.kb = kb
        self._rel_to_id: Dict[str, int] = {}
        self._records: List[Dict[str, object]] = []

    def relation_id(self, rel: str) -> int:
        idx = self._rel_to_id.get(rel)
        if idx is None:
            idx = len(self._rel_to_id)
            self._rel_to_id[rel] = idx
            self._records.append({"relation_id": idx, "kb": self.kb, "kg_id": rel, "label": rel})
        return idx

    @property
    def records(self) -> List[Dict[str, object]]:
        return self._records

    def to_lookup(self) -> RelationLookup:
        return RelationLookup(rel_to_id=dict(self._rel_to_id))


@dataclass
class ParquetDatasetWriter:
    out_dir: Path
    graphs: List[GraphRecord] = field(default_factory=list)
    questions: List[Dict[str, object]] = field(default_factory=list)
    graph_writer: pq.ParquetWriter | None = None
    question_writer: pq.ParquetWriter | None = None
    include_question_emb: bool = False

    def append(self, graph: GraphRecord, question: Dict[str, object]) -> None:
        self.graphs.append(graph)
        self.questions.append(question)

    def flush(self) -> None:
        if self.graphs:
            table = pa.table(
                {
                    "graph_id": pa.array([g.graph_id for g in self.graphs], type=pa.string()),
                    "node_entity_ids": pa.array([g.node_entity_ids for g in self.graphs], type=pa.list_(pa.int64())),
                    "node_embedding_ids": pa.array(
                        [g.node_embedding_ids for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "node_labels": pa.array([g.node_labels for g in self.graphs], type=pa.list_(pa.string())),
                    "edge_src": pa.array([g.edge_src for g in self.graphs], type=pa.list_(pa.int64())),
                    "edge_dst": pa.array([g.edge_dst for g in self.graphs], type=pa.list_(pa.int64())),
                    "edge_relation_ids": pa.array(
                        [g.edge_relation_ids for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "positive_triple_mask": pa.array(
                        [g.positive_triple_mask for g in self.graphs], type=pa.list_(pa.bool_())
                    ),
                    "pair_start_node_locals": pa.array(
                        [g.pair_start_node_locals for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "pair_answer_node_locals": pa.array(
                        [g.pair_answer_node_locals for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "pair_edge_local_ids": pa.array(
                        [g.pair_edge_local_ids for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "pair_edge_counts": pa.array([g.pair_edge_counts for g in self.graphs], type=pa.list_(pa.int64())),
                    "pair_shortest_lengths": pa.array(
                        [g.pair_shortest_lengths for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                }
            )
            if self.graph_writer is None:
                self.graph_writer = pq.ParquetWriter(self.out_dir / "graphs.parquet", table.schema, compression="zstd")
            self.graph_writer.write_table(table)
            self.graphs = []

        if self.questions:
            table_q_data = {
                "question_uid": pa.array([row["question_uid"] for row in self.questions], type=pa.string()),
                "dataset": pa.array([row["dataset"] for row in self.questions], type=pa.string()),
                "split": pa.array([row["split"] for row in self.questions], type=pa.string()),
                "kb": pa.array([row["kb"] for row in self.questions], type=pa.string()),
                "question": pa.array([row["question"] for row in self.questions], type=pa.string()),
                "seed_entity_ids": pa.array([row["seed_entity_ids"] for row in self.questions], type=pa.list_(pa.int64())),
                "answer_entity_ids": pa.array(
                    [row["answer_entity_ids"] for row in self.questions], type=pa.list_(pa.int64())
                ),
                "answer_texts": pa.array([row["answer_texts"] for row in self.questions], type=pa.list_(pa.string())),
                "graph_id": pa.array([row["graph_id"] for row in self.questions], type=pa.string()),
            }
            if self.include_question_emb:
                question_embs: List[object] = []
                for row in self.questions:
                    if "question_emb" not in row:
                        raise ValueError("question_emb missing while include_question_emb is enabled.")
                    question_embs.append(row["question_emb"])
                table_q_data["question_emb"] = pa.array(question_embs, type=pa.list_(pa.float32()))
            table_q = pa.table(table_q_data)
            if self.question_writer is None:
                self.question_writer = pq.ParquetWriter(
                    self.out_dir / "questions.parquet", table_q.schema, compression="zstd"
                )
            self.question_writer.write_table(table_q)
            self.questions = []

    def close(self) -> None:
        self.flush()
        if self.graph_writer is not None:
            self.graph_writer.close()
        if self.question_writer is not None:
            self.question_writer.close()


# Regex compiled once for reuse.
_QID_IN_PARENS_RE = re.compile(r"(Q\d+)")
_LABEL_QID_RE = re.compile(r"(.+)\s+\((Q\d+)\)$")
_PATH_MODE_UNDIRECTED = "undirected"
_PATH_MODE_QA_DIRECTED = "qa_directed"
_PATH_MODES = (_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)
_ALLOWED_SPLITS = ("train", "validation", "test")
_NON_TEXT_EMBEDDING_ID = 0
_NUM_TOPICS_BINARY = 2
_DIST_UNREACHABLE = -1
_DIST_REACHABLE_MIN = 0
_EDGE_STEP = 1
_MIN_CHUNK_SIZE = 1
_DISABLE_PARALLEL_WORKERS = 0
_EDGE_INDEX_MIN = 0
_BYTES_PER_GB = 1 << 30
_LMDB_MAP_SIZE_GB_MIN = 1
_LMDB_GROWTH_GB_MIN = 0
_LMDB_GROWTH_FACTOR_MIN = 1.0
_DEFAULT_LMDB_MAP_GROWTH_GB = 32
_DEFAULT_LMDB_MAP_GROWTH_FACTOR = 1.5
_VALIDATE_GRAPH_EDGES_DEFAULT = True
_REMOVE_SELF_LOOPS_DEFAULT = True
_AUX_LMDB_SUFFIX = ".aux.lmdb"


# Helpers


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_path_mode(path_mode: str) -> str:
    mode = str(path_mode)
    if mode not in _PATH_MODES:
        raise ValueError(f"Unsupported path_mode: {mode}. Expected one of {_PATH_MODES}.")
    return mode


def _validate_split_names(splits: Sequence[str], *, context: str) -> List[str]:
    normalized = [str(split) for split in splits]
    unknown = sorted(set(normalized) - set(_ALLOWED_SPLITS))
    if unknown:
        raise ValueError(
            f"{context} contains unsupported split names: {unknown}. Expected one of {_ALLOWED_SPLITS}."
        )
    return normalized


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


@dataclass(frozen=True)
class _WorkerState:
    entity_lookup: EntityLookup
    relation_lookup: RelationLookup
    path_mode: str
    dedup_edges: bool
    validate_graph_edges: bool
    remove_self_loops: bool


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
        path_mode=state.path_mode,
        dedup_edges=state.dedup_edges,
        validate_graph_edges=state.validate_graph_edges,
        remove_self_loops=state.remove_self_loops,
    )


def _shortest_path_single(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[int], List[int]]:
    if not sources or not targets or num_nodes <= 0:
        return [], []

    from collections import deque

    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
        u = int(u_raw)
        v = int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append((v, idx))
            if u != v:
                adjacency[v].append((u, idx))

    for nbrs in adjacency:
        nbrs.sort(key=lambda item: (item[0], item[1]))

    sources_unique = sorted({int(s) for s in sources if 0 <= int(s) < num_nodes})
    targets_unique = sorted({int(t) for t in targets if 0 <= int(t) < num_nodes})
    if not sources_unique or not targets_unique:
        return [], []

    dist = [-1] * num_nodes
    parent = [-1] * num_nodes
    parent_edge = [-1] * num_nodes
    q: deque[int] = deque()
    for s in sources_unique:
        dist[s] = 0
        q.append(s)

    while q:
        cur = q.popleft()
        next_dist = dist[cur] + 1
        for nb, e_idx in adjacency[cur]:
            if dist[nb] != -1:
                continue
            dist[nb] = next_dist
            parent[nb] = cur
            parent_edge[nb] = int(e_idx)
            q.append(nb)

    best_target = None
    best_dist = None
    for tgt in targets_unique:
        if dist[tgt] < 0:
            continue
        if best_dist is None or dist[tgt] < best_dist or (dist[tgt] == best_dist and tgt < best_target):
            best_target = tgt
            best_dist = dist[tgt]

    if best_target is None:
        return [], []

    nodes_rev: List[int] = [int(best_target)]
    edges_rev: List[int] = []
    cur = int(best_target)
    sources_set = set(sources_unique)
    while cur not in sources_set:
        prev = int(parent[cur])
        edge = int(parent_edge[cur])
        if prev < 0 or edge < 0:
            return [], []
        edges_rev.append(edge)
        nodes_rev.append(prev)
        cur = prev

    edges = list(reversed(edges_rev))
    nodes = list(reversed(nodes_rev))
    if not edges:
        return [], nodes
    return edges, nodes


def _validate_graph_record(graph: "GraphRecord") -> None:
    num_nodes = len(graph.node_entity_ids)
    num_edges = len(graph.edge_src)
    if len(graph.edge_dst) != num_edges or len(graph.edge_relation_ids) != num_edges:
        raise ValueError(f"Edge length mismatch for {graph.graph_id}: edges={num_edges}.")
    if len(graph.positive_triple_mask) != num_edges:
        raise ValueError(f"positive_triple_mask length mismatch for {graph.graph_id}: edges={num_edges}.")
    if num_edges > _EDGE_INDEX_MIN:
        min_src = min(graph.edge_src)
        min_dst = min(graph.edge_dst)
        max_src = max(graph.edge_src)
        max_dst = max(graph.edge_dst)
        if min_src < _EDGE_INDEX_MIN or min_dst < _EDGE_INDEX_MIN:
            raise ValueError(f"Negative edge index detected for {graph.graph_id}.")
        if max_src >= num_nodes or max_dst >= num_nodes:
            raise ValueError(f"Edge index exceeds num_nodes for {graph.graph_id}.")
    if graph.pair_edge_counts:
        total_pairs = sum(graph.pair_edge_counts)
        if total_pairs != len(graph.pair_edge_local_ids):
            raise ValueError(f"pair_edge_counts sum mismatch for {graph.graph_id}.")
    if graph.pair_edge_local_ids:
        min_pair = min(graph.pair_edge_local_ids)
        max_pair = max(graph.pair_edge_local_ids)
        if min_pair < _EDGE_INDEX_MIN or max_pair >= num_edges:
            raise ValueError(f"pair_edge_local_ids out of range for {graph.graph_id}.")
    if graph.pair_start_node_locals:
        min_start = min(graph.pair_start_node_locals)
        max_start = max(graph.pair_start_node_locals)
        if min_start < _EDGE_INDEX_MIN or max_start >= num_nodes:
            raise ValueError(f"pair_start_node_locals out of range for {graph.graph_id}.")
    if graph.pair_answer_node_locals:
        min_ans = min(graph.pair_answer_node_locals)
        max_ans = max(graph.pair_answer_node_locals)
        if min_ans < _EDGE_INDEX_MIN or max_ans >= num_nodes:
            raise ValueError(f"pair_answer_node_locals out of range for {graph.graph_id}.")


def _build_undirected_adjacency(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(edge_src, edge_dst):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adjacency[u].append(v)
        if u != v:
            adjacency[v].append(u)
    for nbrs in adjacency:
        nbrs.sort()
    return adjacency


def _build_directed_adjacency(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(edge_src, edge_dst):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adjacency[u].append(v)
    for nbrs in adjacency:
        nbrs.sort()
    return adjacency


def _normalize_local_nodes(num_nodes: int, nodes: Sequence[int]) -> List[int]:
    return sorted({int(n) for n in nodes if 0 <= int(n) < num_nodes})


def _bfs_dist(num_nodes: int, adjacency: Sequence[Sequence[int]], sources: Sequence[int]) -> List[int]:
    dist = [_DIST_UNREACHABLE] * num_nodes
    if num_nodes <= 0:
        return dist
    from collections import deque

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


def _as_numpy_int(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _valid_edge_indices(edge_src: np.ndarray, edge_dst: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes <= _DIST_REACHABLE_MIN:
        return np.empty((0,), dtype=np.int64)
    valid = (
        (edge_src >= _DIST_REACHABLE_MIN)
        & (edge_dst >= _DIST_REACHABLE_MIN)
        & (edge_src < num_nodes)
        & (edge_dst < num_nodes)
    )
    return np.nonzero(valid)[0]


def _select_shortest_edges_undirected(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    dist_s: np.ndarray,
    dist_a: np.ndarray,
    dist_sa: int,
) -> np.ndarray:
    dist_s_u = dist_s[edge_src]
    dist_s_v = dist_s[edge_dst]
    dist_a_u = dist_a[edge_src]
    dist_a_v = dist_a[edge_dst]
    mask_uv = (
        (dist_s_u >= _DIST_REACHABLE_MIN)
        & (dist_a_v >= _DIST_REACHABLE_MIN)
        & (dist_s_u + _EDGE_STEP + dist_a_v == dist_sa)
    )
    mask_vu = (
        (dist_s_v >= _DIST_REACHABLE_MIN)
        & (dist_a_u >= _DIST_REACHABLE_MIN)
        & (dist_s_v + _EDGE_STEP + dist_a_u == dist_sa)
    )
    return np.nonzero(mask_uv | mask_vu)[0]


def _select_shortest_edges_directed(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    dist_s: np.ndarray,
    dist_a: np.ndarray,
    dist_sa: int,
) -> np.ndarray:
    dist_s_u = dist_s[edge_src]
    dist_a_v = dist_a[edge_dst]
    mask = (
        (dist_s_u >= _DIST_REACHABLE_MIN)
        & (dist_a_v >= _DIST_REACHABLE_MIN)
        & (dist_s_u + _EDGE_STEP + dist_a_v == dist_sa)
    )
    return np.nonzero(mask)[0]


def _shortest_path_union_mask_by_pair(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[bool], List[int], List[int], List[int], List[int], List[int]]:
    num_edges = len(edge_src)
    if num_nodes <= 0 or num_edges == 0 or not sources or not targets:
        return [False] * num_edges, [], [], [], [], []

    edge_src_arr = _as_numpy_int(edge_src)
    edge_dst_arr = _as_numpy_int(edge_dst)
    valid_edge_indices = _valid_edge_indices(edge_src_arr, edge_dst_arr, num_nodes)
    if valid_edge_indices.size == 0:
        return [False] * num_edges, [], [], [], [], []
    edge_src_valid = edge_src_arr[valid_edge_indices]
    edge_dst_valid = edge_dst_arr[valid_edge_indices]

    adjacency = _build_undirected_adjacency(num_nodes, edge_src, edge_dst)
    starts = sorted({int(s) for s in sources if 0 <= int(s) < num_nodes})
    answers = sorted({int(t) for t in targets if 0 <= int(t) < num_nodes})
    if not starts or not answers:
        return [False] * num_edges, [], [], [], [], []

    dist_from_start: Dict[int, np.ndarray] = {
        s: _as_numpy_int(_bfs_dist(num_nodes, adjacency, [s])) for s in starts
    }
    dist_to_answer: Dict[int, np.ndarray] = {a: _as_numpy_int(_bfs_dist(num_nodes, adjacency, [a])) for a in answers}

    mask = np.zeros(num_edges, dtype=bool)
    pair_start_nodes: List[int] = []
    pair_answer_nodes: List[int] = []
    pair_edge_local_ids: List[int] = []
    pair_edge_counts: List[int] = []
    pair_shortest_lengths: List[int] = []

    for s in starts:
        dist_s = dist_from_start[s]
        for a in answers:
            dist_a = dist_to_answer[a]
            dist_sa = int(dist_s[a])
            if dist_sa < _DIST_REACHABLE_MIN:
                continue
            pair_start_nodes.append(s)
            pair_answer_nodes.append(a)
            pair_shortest_lengths.append(dist_sa)
            edge_keep = _select_shortest_edges_undirected(edge_src_valid, edge_dst_valid, dist_s, dist_a, dist_sa)
            edge_ids = valid_edge_indices[edge_keep]
            if edge_ids.size > 0:
                mask[edge_ids] = True
                pair_edge_local_ids.extend(edge_ids.tolist())
            pair_edge_counts.append(int(edge_ids.size))

    return (
        mask.tolist(),
        pair_start_nodes,
        pair_answer_nodes,
        pair_edge_local_ids,
        pair_edge_counts,
        pair_shortest_lengths,
    )


def _shortest_path_union_mask_by_pair_directed(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[bool], List[int], List[int], List[int], List[int], List[int]]:
    num_edges = len(edge_src)
    if num_nodes <= 0 or num_edges == 0 or not sources or not targets:
        return [False] * num_edges, [], [], [], [], []

    starts = _normalize_local_nodes(num_nodes, sources)
    answers = _normalize_local_nodes(num_nodes, targets)
    if not starts or not answers:
        return [False] * num_edges, [], [], [], [], []

    edge_src_arr = _as_numpy_int(edge_src)
    edge_dst_arr = _as_numpy_int(edge_dst)
    valid_edge_indices = _valid_edge_indices(edge_src_arr, edge_dst_arr, num_nodes)
    if valid_edge_indices.size == 0:
        return [False] * num_edges, [], [], [], [], []
    edge_src_valid = edge_src_arr[valid_edge_indices]
    edge_dst_valid = edge_dst_arr[valid_edge_indices]

    adjacency = _build_directed_adjacency(num_nodes, edge_src, edge_dst)
    reverse_adjacency = _build_directed_adjacency(num_nodes, edge_dst, edge_src)
    dist_from_start = {s: _as_numpy_int(_bfs_dist(num_nodes, adjacency, [s])) for s in starts}
    dist_to_answer = {a: _as_numpy_int(_bfs_dist(num_nodes, reverse_adjacency, [a])) for a in answers}

    mask = np.zeros(num_edges, dtype=bool)
    pair_start_nodes: List[int] = []
    pair_answer_nodes: List[int] = []
    pair_edge_local_ids: List[int] = []
    pair_edge_counts: List[int] = []
    pair_shortest_lengths: List[int] = []

    for s in starts:
        dist_s = dist_from_start[s]
        for a in answers:
            dist_a = dist_to_answer[a]
            dist_sa = int(dist_s[a])
            if dist_sa < _DIST_REACHABLE_MIN:
                continue
            pair_start_nodes.append(s)
            pair_answer_nodes.append(a)
            pair_shortest_lengths.append(dist_sa)
            edge_keep = _select_shortest_edges_directed(edge_src_valid, edge_dst_valid, dist_s, dist_a, dist_sa)
            edge_ids = valid_edge_indices[edge_keep]
            if edge_ids.size > 0:
                mask[edge_ids] = True
                pair_edge_local_ids.extend(edge_ids.tolist())
            pair_edge_counts.append(int(edge_ids.size))

    return (
        mask.tolist(),
        pair_start_nodes,
        pair_answer_nodes,
        pair_edge_local_ids,
        pair_edge_counts,
        pair_shortest_lengths,
    )


def _shortest_path_union_mask_by_pair_mode(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    path_mode: str,
) -> Tuple[List[bool], List[int], List[int], List[int], List[int], List[int]]:
    mode = _validate_path_mode(path_mode)
    if mode == _PATH_MODE_QA_DIRECTED:
        return _shortest_path_union_mask_by_pair_directed(num_nodes, edge_src, edge_dst, sources, targets)
    return _shortest_path_union_mask_by_pair(num_nodes, edge_src, edge_dst, sources, targets)


def _normalize_embeddings(embeddings: torch.Tensor, eps: float) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings
    denom = embeddings.norm(dim=-1, keepdim=True).clamp(min=eps)
    return embeddings / denom


def _group_positive_edges_by_pair(
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    positive_mask: Sequence[bool],
) -> Dict[Tuple[int, int], List[int]]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, keep in enumerate(positive_mask):
        if not keep:
            continue
        u = int(edge_src[idx])
        v = int(edge_dst[idx])
        key = (u, v) if u <= v else (v, u)
        groups.setdefault(key, []).append(idx)
    return groups


def _select_canonical_edge_indices(
    groups: Dict[Tuple[int, int], List[int]],
    edge_relation_ids: Sequence[int],
    relation_embeddings_norm: torch.Tensor,
    question_embedding_norm: torch.Tensor,
) -> List[int]:
    question_vec = question_embedding_norm.view(-1)
    keep_indices: List[int] = []
    for edge_indices in groups.values():
        if len(edge_indices) == 1:
            keep_indices.append(edge_indices[0])
            continue
        ordered = sorted(edge_indices, key=lambda idx: (int(edge_relation_ids[idx]), idx))
        rel_ids = torch.tensor([int(edge_relation_ids[idx]) for idx in ordered], dtype=torch.long)
        rel_vecs = relation_embeddings_norm.index_select(0, rel_ids)
        scores = torch.mv(rel_vecs, question_vec)
        best = int(torch.argmax(scores).item())
        keep_indices.append(ordered[best])
    return keep_indices


def _filter_pair_edges(
    pair_edge_local_ids: Sequence[int],
    pair_edge_counts: Sequence[int],
    keep_mask: Sequence[bool],
) -> Tuple[List[int], List[int]]:
    if not pair_edge_local_ids or not pair_edge_counts:
        return list(pair_edge_local_ids), list(pair_edge_counts)
    new_ids: List[int] = []
    new_counts: List[int] = []
    offset = 0
    for count in pair_edge_counts:
        span = pair_edge_local_ids[offset : offset + count]
        kept = [idx for idx in span if keep_mask[int(idx)]]
        new_ids.extend(kept)
        new_counts.append(len(kept))
        offset += count
    if offset != len(pair_edge_local_ids):
        raise ValueError("pair_edge_counts do not sum to len(pair_edge_local_ids)")
    return new_ids, new_counts


def _canonicalize_graph_edges(
    graph: GraphRecord,
    question_embedding_norm: torch.Tensor,
    relation_embeddings_norm: torch.Tensor,
) -> None:
    if not graph.positive_triple_mask:
        return
    if question_embedding_norm.numel() == 0:
        raise ValueError(f"question_embedding is empty for {graph.graph_id}")
    if relation_embeddings_norm.numel() == 0:
        raise ValueError("relation_embeddings are empty; cannot canonicalize positives.")
    if relation_embeddings_norm.dim() != 2 or question_embedding_norm.dim() != 1:
        raise ValueError("Embeddings must be 2D (relations) and 1D (question) for canonicalization.")
    if int(relation_embeddings_norm.size(1)) != int(question_embedding_norm.numel()):
        raise ValueError("Question embedding dim does not match relation embedding dim.")
    groups = _group_positive_edges_by_pair(graph.edge_src, graph.edge_dst, graph.positive_triple_mask)
    if not groups:
        return
    keep_indices = _select_canonical_edge_indices(
        groups,
        graph.edge_relation_ids,
        relation_embeddings_norm,
        question_embedding_norm,
    )
    keep_mask = [False] * len(graph.edge_src)
    for idx in keep_indices:
        keep_mask[idx] = True
    new_pair_edge_local_ids, new_pair_edge_counts = _filter_pair_edges(
        graph.pair_edge_local_ids,
        graph.pair_edge_counts,
        keep_mask,
    )
    graph.positive_triple_mask = keep_mask
    graph.pair_edge_local_ids = new_pair_edge_local_ids
    graph.pair_edge_counts = new_pair_edge_counts


def shortest_path_edge_indices_undirected(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    seeds: Sequence[int],
    answers: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Deterministic single shortest path between seeds and answers (undirected traversal)."""
    return _shortest_path_single(num_nodes, edge_src, edge_dst, seeds, answers)


def has_connectivity(
    graph: Sequence[Tuple[str, str, str]],
    seeds: Sequence[str],
    answers: Sequence[str],
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
) -> bool:
    """Check existence of path seed->answer using local indexing."""
    if not graph or not seeds or not answers:
        return False
    node_index: Dict[str, int] = {}
    edge_src: List[int] = []
    edge_dst: List[int] = []

    def local_idx(node: str) -> int:
        if node not in node_index:
            node_index[node] = len(node_index)
        return node_index[node]

    for h, _, t in graph:
        edge_src.append(local_idx(h))
        edge_dst.append(local_idx(t))

    seed_ids = [node_index[s] for s in seeds if s in node_index]
    answer_ids = [node_index[a] for a in answers if a in node_index]
    if not seed_ids or not answer_ids:
        return False
    mode = _validate_path_mode(path_mode)
    if mode == _PATH_MODE_QA_DIRECTED:
        adjacency = _build_directed_adjacency(len(node_index), edge_src, edge_dst)
    else:
        adjacency = _build_undirected_adjacency(len(node_index), edge_src, edge_dst)
    dist = _bfs_dist(len(node_index), adjacency, seed_ids)
    return any(dist[a] >= 0 for a in answer_ids)


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


def _resolve_split_filter(
    split: str, train_filter: SplitFilter, eval_filter: SplitFilter, override_filters: Dict[str, SplitFilter]
) -> SplitFilter:
    override = override_filters.get(split)
    if override is not None:
        return override
    return train_filter if split == "train" else eval_filter


def _should_keep_sample(
    sample: Sample,
    split_filter: SplitFilter,
    connectivity_cache: Dict[Tuple[str, str, str], bool],
    *,
    path_mode: str,
) -> bool:
    node_strings = {h for h, _, t in sample.graph} | {t for _, _, t in sample.graph}
    has_topic = any(ent in node_strings for ent in sample.q_entity)
    has_answer = any(ent in node_strings for ent in sample.a_entity)

    cache_key = (sample.dataset, sample.split, sample.question_id)
    has_path = connectivity_cache.get(cache_key)
    if has_path is None:
        if sample.answer_subgraph:
            has_path = True
        elif split_filter.skip_no_path:
            has_path = has_connectivity(sample.graph, sample.q_entity, sample.a_entity, path_mode=path_mode)
        else:
            has_path = True
        connectivity_cache[cache_key] = has_path

    if split_filter.skip_no_topic and not has_topic:
        return False
    if split_filter.skip_no_ans and not has_answer:
        return False
    if split_filter.skip_no_path and not has_path:
        return False
    return True


def iter_samples(
    dataset: str,
    kb: str,
    raw_root: Path,
    splits: Sequence[str],
    column_map: Dict[str, str],
    entity_normalization: str,
) -> Iterable[Sample]:
    for split in splits:
        dataset_obj = load_split(raw_root, split)
        for batch in dataset_obj.to_batches():
            for row in batch.to_pylist():
                graph_raw = row.get(column_map["graph_field"]) or []
                label_to_qid: Dict[str, str] = {}
                graph: List[Tuple[str, str, str]] = []
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
                # Optional fields (dataset-specific)
                answer_subgraph_raw = []
                if "answer_subgraph_field" in column_map:
                    answer_subgraph_raw = row.get(column_map["answer_subgraph_field"]) or []
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

                yield Sample(
                    dataset=dataset,
                    split=split,
                    question_id=str(row[column_map["question_id_field"]]),
                    kb=kb,
                    question=row.get(column_map["question_field"]) or "",
                    graph=graph,
                    q_entity=q_entities,
                    a_entity=a_entities,
                    answer_texts=answer_texts,
                    answer_subgraph=[
                        (
                            normalize_entity_with_lookup(str(tr[0]), entity_normalization, label_to_qid),
                            str(tr[1]),
                            normalize_entity_with_lookup(str(tr[2]), entity_normalization, label_to_qid),
                        )
                        for tr in answer_subgraph_raw
                        if isinstance(tr, (list, tuple)) and len(tr) >= 3
                    ],
                    graph_iso_type=graph_iso_type,
                    redundant=redundant,
                    test_type=test_type,
                )


def preprocess(
    dataset: str,
    kb: str,
    raw_root: Path,
    out_dir: Path,
    column_map: Dict[str, str],
    entity_normalization: str,
    text_cfg: TextEntityConfig,
    train_filter: SplitFilter,
    eval_filter: SplitFilter,
    override_filters: Dict[str, SplitFilter],
    path_mode: str = _PATH_MODE_UNDIRECTED,
    dedup_edges: bool = True,
    validate_graph_edges: bool = _VALIDATE_GRAPH_EDGES_DEFAULT,
    remove_self_loops: bool = _REMOVE_SELF_LOOPS_DEFAULT,
    embedding_cfg: Optional[EmbeddingConfig] = None,
    *,
    emit_sub_filter: bool = False,
    sub_filter_filename: str = "sub_filter.json",
    emit_nonzero_positive_filter: bool = False,
    nonzero_positive_filter_filename: str = "nonzero_positive_filter.json",
    nonzero_positive_filter_splits: Optional[Sequence[str]] = None,
    parquet_chunk_size: int = _MIN_CHUNK_SIZE,
    parquet_num_workers: int = _DISABLE_PARALLEL_WORKERS,
    reuse_embeddings_if_exists: bool = False,
) -> None:
    path_mode = _validate_path_mode(path_mode)
    dedup_edges = bool(dedup_edges)
    validate_graph_edges = bool(validate_graph_edges)
    remove_self_loops = bool(remove_self_loops)
    if parquet_chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"parquet_chunk_size must be >= {_MIN_CHUNK_SIZE}, got {parquet_chunk_size}")
    if parquet_num_workers < _DISABLE_PARALLEL_WORKERS:
        raise ValueError(f"parquet_num_workers must be >= {_DISABLE_PARALLEL_WORKERS}, got {parquet_num_workers}")
    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb, text_cfg=text_cfg)
    relation_vocab = RelationVocab(kb=kb)

    available_files = {p.name for p in raw_root.glob("*.parquet")}
    splits = sorted({name.split("-")[0] for name in available_files})
    splits = _validate_split_names(splits, context="raw parquet files")
    connectivity_cache: Dict[Tuple[str, str, str], bool] = {}
    total_by_split: Dict[str, int] = {}
    kept_by_split: Dict[str, int] = {}
    sub_by_split: Dict[str, int] = {}
    empty_graph_by_split: Dict[str, int] = {}
    empty_graph_ids: List[str] = []
    empty_graph_id_set: Set[str] = set()
    sub_sample_ids: List[str] = []
    nonzero_positive_sample_ids: List[str] = []
    nonzero_positive_by_split: Dict[str, int] = {}
    nonzero_positive_splits: Optional[Set[str]] = None
    if emit_nonzero_positive_filter:
        splits_cfg = nonzero_positive_filter_splits
        if isinstance(splits_cfg, str):
            splits_cfg = [splits_cfg]
        if splits_cfg is None:
            nonzero_positive_splits = None
        else:
            split_list = _validate_split_names(splits_cfg, context="nonzero_positive_filter_splits")
            nonzero_positive_splits = set(split_list)
            if not nonzero_positive_splits:
                raise ValueError(
                    "emit_nonzero_positive_filter=true requires nonzero_positive_filter_splits to be non-empty."
                )

    # Pass 1: Build vocabularies
    print("Pass 1: Building vocabularies...")
    for sample in tqdm(
        iter_samples(dataset, kb, raw_root, splits, column_map, entity_normalization),
        desc=f"Pass 1/2: Vocab from {dataset}",
    ):
        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        total_by_split[sample.split] = total_by_split.get(sample.split, 0) + 1
        if not sample.graph:
            empty_graph_by_split[sample.split] = empty_graph_by_split.get(sample.split, 0) + 1
            empty_graph_id_set.add(graph_id)
            if len(empty_graph_ids) < 20:
                empty_graph_ids.append(graph_id)
            continue
        for h, r, t in sample.graph:
            entity_vocab.add_entity(h)
            entity_vocab.add_entity(t)
            relation_vocab.relation_id(r)
        for ent in sample.q_entity + sample.a_entity:
            entity_vocab.add_entity(ent)

        split_filter = _resolve_split_filter(sample.split, train_filter, eval_filter, override_filters)
        if _should_keep_sample(sample, split_filter, connectivity_cache, path_mode=path_mode):
            kept_by_split[sample.split] = kept_by_split.get(sample.split, 0) + 1

    entity_vocab.finalize()

    def _format_counts(counts: Dict[str, int]) -> str:
        return ", ".join(f"{s}={counts.get(s, 0)}" for s in splits)

    print(f"Samples total: {_format_counts(total_by_split)}")
    print(f"Samples kept : {_format_counts(kept_by_split)}")
    if empty_graph_by_split:
        print(f"Samples dropped (empty graph): {_format_counts(empty_graph_by_split)}")
        if empty_graph_ids:
            print(f"Empty-graph examples: {empty_graph_ids}")

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
                print(f"Reusing entity embeddings at {entity_emb_path}")
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
                print(f"Reusing relation embeddings at {relation_emb_path}")
                relation_emb = torch.load(relation_emb_path, map_location="cpu")
            if embedding_cfg.canonicalize_relations:
                relation_embeddings_norm = _normalize_embeddings(relation_emb, embedding_cfg.cosine_eps)
                if relation_embeddings_norm.numel() == 0:
                    raise ValueError("relation_embeddings are empty; cannot canonicalize positives.")

    # Pass 2: Build graphs and questions
    print("Pass 2: Building graphs and questions...")
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
                    path_mode=path_mode,
                    dedup_edges=dedup_edges,
                    validate_graph_edges=validate_graph_edges,
                    remove_self_loops=remove_self_loops,
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
            if emit_sub_filter:
                label_to_idx = {label: idx for idx, label in enumerate(graph.node_labels)}
                q_local = {label_to_idx[ent] for ent in sample.q_entity if ent in label_to_idx}
                a_local = {label_to_idx[ent] for ent in sample.a_entity if ent in label_to_idx}
                has_topic = bool(q_local)
                has_answer = bool(a_local)
                has_path = len(graph.pair_start_node_locals) > 0
                nonzero_min_len = False
                if graph.pair_shortest_lengths:
                    nonzero_min_len = min(graph.pair_shortest_lengths) > 0
                no_overlap = q_local.isdisjoint(a_local)
                if has_topic and has_answer and has_path and (nonzero_min_len or no_overlap):
                    sub_sample_ids.append(graph.graph_id)
                    sub_by_split[sample.split] = sub_by_split.get(sample.split, 0) + 1
            if emit_nonzero_positive_filter:
                split_ok = nonzero_positive_splits is None or sample.split in nonzero_positive_splits
                if split_ok and any(graph.positive_triple_mask):
                    nonzero_positive_sample_ids.append(graph.graph_id)
                    nonzero_positive_by_split[sample.split] = nonzero_positive_by_split.get(sample.split, 0) + 1
            if len(base_writer.graphs) >= chunk_size or len(base_writer.questions) >= chunk_size:
                base_writer.flush()

    def _run_pass2(executor: Optional[ProcessPoolExecutor]) -> None:
        pending_samples: List[Sample] = []
        for sample in tqdm(
            iter_samples(dataset, kb, raw_root, splits, column_map, entity_normalization),
            desc=f"Pass 2/2: Graphs from {dataset}",
        ):
            graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
            if graph_id in empty_graph_id_set:
                continue
            split_filter = _resolve_split_filter(sample.split, train_filter, eval_filter, override_filters)
            if not _should_keep_sample(sample, split_filter, connectivity_cache, path_mode=path_mode):
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
            path_mode=path_mode,
            dedup_edges=dedup_edges,
            validate_graph_edges=validate_graph_edges,
            remove_self_loops=remove_self_loops,
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

    write_entity_vocab(entity_vocab.struct_records, out_dir / "entity_vocab.parquet")
    write_embedding_vocab(entity_vocab.embedding_records, out_dir / "embedding_vocab.parquet")
    write_relation_vocab(relation_vocab.records, out_dir / "relation_vocab.parquet")

    if emit_sub_filter:
        sub_payload = {
            "dataset": dataset,
            "sample_ids": sorted(sub_sample_ids),
        }
        (out_dir / sub_filter_filename).write_text(json.dumps(sub_payload, indent=2))
        print(f"Sub filter samples kept: {_format_counts(sub_by_split)}")
        print(f"Sub filter written to: {out_dir / sub_filter_filename}")

    if emit_nonzero_positive_filter:
        filter_splits = sorted(nonzero_positive_splits) if nonzero_positive_splits is not None else list(splits)
        nonzero_payload = {
            "dataset": dataset,
            "splits": filter_splits,
            "sample_ids": sorted(nonzero_positive_sample_ids),
        }
        (out_dir / nonzero_positive_filter_filename).write_text(json.dumps(nonzero_payload, indent=2))
        print(f"Nonzero-positive filter samples kept: {_format_counts(nonzero_positive_by_split)}")
        print(f"Nonzero-positive filter written to: {out_dir / nonzero_positive_filter_filename}")


def build_graph(
    sample: Sample,
    entity_vocab: EntityVocab | EntityLookup,
    relation_vocab: RelationVocab | RelationLookup,
    graph_id: str,
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
    dedup_edges: bool = True,
    validate_graph_edges: bool = _VALIDATE_GRAPH_EDGES_DEFAULT,
    remove_self_loops: bool = _REMOVE_SELF_LOOPS_DEFAULT,
) -> GraphRecord:
    path_mode = _validate_path_mode(path_mode)
    dedup_edges = bool(dedup_edges)
    validate_graph_edges = bool(validate_graph_edges)
    remove_self_loops = bool(remove_self_loops)
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
    edge_keys: List[Tuple[str, str, str]] = []
    edge_key_to_indices: Dict[Tuple[str, str, str], List[int]] = {}

    for h, r, t in sample.graph:
        if remove_self_loops and h == t:
            continue
        edge_key = (h, r, t)
        if dedup_edges and edge_key in edge_key_to_indices:
            continue
        src_idx = local_index(h)
        dst_idx = local_index(t)
        rel_idx = relation_vocab.relation_id(r)
        edge_src.append(src_idx)
        edge_dst.append(dst_idx)
        edge_relation_ids.append(rel_idx)
        edge_keys.append(edge_key)
        edge_key_to_indices.setdefault(edge_key, []).append(len(edge_keys) - 1)

    q_local = [node_index[ent] for ent in sample.q_entity if ent in node_index]
    a_local = [node_index[ent] for ent in sample.a_entity if ent in node_index]
    # Priority 1: use provided answer_subgraph to mark positives.
    answer_edge_indices: List[int] = []
    if sample.answer_subgraph:
        for tr in sample.answer_subgraph:
            if not isinstance(tr, tuple) or len(tr) < 3:
                continue
            idxs = edge_key_to_indices.get(tr)
            if idxs:
                answer_edge_indices.extend(idxs)

    num_edges = len(edge_src)
    positive_triple_mask = [False] * num_edges
    pair_start_node_locals: List[int] = []
    pair_answer_node_locals: List[int] = []
    pair_edge_local_ids: List[int] = []
    pair_edge_counts: List[int] = []
    pair_shortest_lengths: List[int] = []

    if answer_edge_indices:
        # Deduplicate while preserving order for determinism.
        seen = set()
        dedup_answer_edges = []
        for idx in answer_edge_indices:
            if idx not in seen:
                seen.add(idx)
                dedup_answer_edges.append(idx)

        sub_edge_src = [edge_src[idx] for idx in dedup_answer_edges]
        sub_edge_dst = [edge_dst[idx] for idx in dedup_answer_edges]
        (
            sub_mask,
            pair_start_node_locals,
            pair_answer_node_locals,
            pair_edge_local_ids,
            pair_edge_counts,
            pair_shortest_lengths,
        ) = _shortest_path_union_mask_by_pair_mode(
            num_nodes=len(node_entity_ids),
            edge_src=sub_edge_src,
            edge_dst=sub_edge_dst,
            sources=q_local,
            targets=a_local,
            path_mode=path_mode,
        )
        has_pairs = len(pair_start_node_locals) > 0
        if has_pairs:
            positive_triple_mask = [False] * num_edges
            for sub_idx, keep in enumerate(sub_mask):
                if keep:
                    positive_triple_mask[dedup_answer_edges[sub_idx]] = True
            if pair_edge_local_ids:
                pair_edge_local_ids = [dedup_answer_edges[idx] for idx in pair_edge_local_ids]
        else:
            (
                positive_triple_mask,
                pair_start_node_locals,
                pair_answer_node_locals,
                pair_edge_local_ids,
                pair_edge_counts,
                pair_shortest_lengths,
            ) = _shortest_path_union_mask_by_pair_mode(
                num_nodes=len(node_entity_ids),
                edge_src=edge_src,
                edge_dst=edge_dst,
                sources=q_local,
                targets=a_local,
                path_mode=path_mode,
            )
    else:
        (
            positive_triple_mask,
            pair_start_node_locals,
            pair_answer_node_locals,
            pair_edge_local_ids,
            pair_edge_counts,
            pair_shortest_lengths,
        ) = _shortest_path_union_mask_by_pair_mode(
            num_nodes=len(node_entity_ids),
            edge_src=edge_src,
            edge_dst=edge_dst,
            sources=q_local,
            targets=a_local,
            path_mode=path_mode,
        )

    graph = GraphRecord(
        graph_id=graph_id,
        node_entity_ids=node_entity_ids,
        node_embedding_ids=node_embedding_ids,
        node_labels=node_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_relation_ids=edge_relation_ids,
        positive_triple_mask=positive_triple_mask,
        pair_start_node_locals=pair_start_node_locals,
        pair_answer_node_locals=pair_answer_node_locals,
        pair_edge_local_ids=pair_edge_local_ids,
        pair_edge_counts=pair_edge_counts,
        pair_shortest_lengths=pair_shortest_lengths,
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


# Writers


def write_graphs(graphs: List[GraphRecord], output_path: Path) -> None:
    table = pa.table(
        {
            "graph_id": pa.array([g.graph_id for g in graphs], type=pa.string()),
            "node_entity_ids": pa.array([g.node_entity_ids for g in graphs], type=pa.list_(pa.int64())),
            "node_embedding_ids": pa.array([g.node_embedding_ids for g in graphs], type=pa.list_(pa.int64())),
            "node_labels": pa.array([g.node_labels for g in graphs], type=pa.list_(pa.string())),
            "edge_src": pa.array([g.edge_src for g in graphs], type=pa.list_(pa.int64())),
            "edge_dst": pa.array([g.edge_dst for g in graphs], type=pa.list_(pa.int64())),
            "edge_relation_ids": pa.array([g.edge_relation_ids for g in graphs], type=pa.list_(pa.int64())),
            "positive_triple_mask": pa.array([g.positive_triple_mask for g in graphs], type=pa.list_(pa.bool_())),
            "pair_start_node_locals": pa.array([g.pair_start_node_locals for g in graphs], type=pa.list_(pa.int64())),
            "pair_answer_node_locals": pa.array([g.pair_answer_node_locals for g in graphs], type=pa.list_(pa.int64())),
            "pair_edge_local_ids": pa.array([g.pair_edge_local_ids for g in graphs], type=pa.list_(pa.int64())),
            "pair_edge_counts": pa.array([g.pair_edge_counts for g in graphs], type=pa.list_(pa.int64())),
            "pair_shortest_lengths": pa.array([g.pair_shortest_lengths for g in graphs], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_questions(rows: List[Dict[str, object]], output_path: Path) -> None:
    table_data = {
        "question_uid": pa.array([row["question_uid"] for row in rows], type=pa.string()),
        "dataset": pa.array([row["dataset"] for row in rows], type=pa.string()),
        "split": pa.array([row["split"] for row in rows], type=pa.string()),
        "kb": pa.array([row["kb"] for row in rows], type=pa.string()),
        "question": pa.array([row["question"] for row in rows], type=pa.string()),
        "seed_entity_ids": pa.array([row["seed_entity_ids"] for row in rows], type=pa.list_(pa.int64())),
        "answer_entity_ids": pa.array([row["answer_entity_ids"] for row in rows], type=pa.list_(pa.int64())),
        "answer_texts": pa.array([row["answer_texts"] for row in rows], type=pa.list_(pa.string())),
        "graph_id": pa.array([row["graph_id"] for row in rows], type=pa.string()),
    }
    if any("question_emb" in row for row in rows):
        table_data["question_emb"] = pa.array(
            [row.get("question_emb") for row in rows], type=pa.list_(pa.float32())
        )
    table = pa.table(table_data)
    pq.write_table(table, output_path, compression="zstd")


def write_entity_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    table = pa.table(
        {
            "entity_id": pa.array([rec["entity_id"] for rec in vocab_records], type=pa.int64()),
            "kb": pa.array([rec["kb"] for rec in vocab_records], type=pa.string()),
            "kg_id": pa.array([rec["kg_id"] for rec in vocab_records], type=pa.string()),
            "label": pa.array([rec.get("label", "") for rec in vocab_records], type=pa.string()),
            "is_text": pa.array([rec["is_text"] for rec in vocab_records], type=pa.bool_()),
            "embedding_id": pa.array([rec["embedding_id"] for rec in vocab_records], type=pa.int64()),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_embedding_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    table = pa.table(
        {
            "embedding_id": pa.array([rec["embedding_id"] for rec in vocab_records], type=pa.int64()),
            "kb": pa.array([rec["kb"] for rec in vocab_records], type=pa.string()),
            "kg_id": pa.array([rec["kg_id"] for rec in vocab_records], type=pa.string()),
            "label": pa.array([rec.get("label", "") for rec in vocab_records], type=pa.string()),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_relation_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    table = pa.table(
        {
            "relation_id": pa.array([rec["relation_id"] for rec in vocab_records], type=pa.int64()),
            "kb": pa.array([rec["kb"] for rec in vocab_records], type=pa.string()),
            "kg_id": pa.array([rec["kg_id"] for rec in vocab_records], type=pa.string()),
            "label": pa.array([rec.get("label", "") for rec in vocab_records], type=pa.string()),
        }
    )
    pq.write_table(table, output_path, compression="zstd")



# === LMDB materialization ===
def _load_parquet(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pq.read_table(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_vocab_lmdb(entity_labels: List[str], relation_labels: List[str], vocab_dir: Path, map_size_bytes: int) -> None:
    _ensure_dir(vocab_dir)
    env = lmdb.open(str(vocab_dir), map_size=map_size_bytes)
    try:
        with env.begin(write=True) as txn:
            entity_to_id = {label: idx for idx, label in enumerate(entity_labels)}
            id_to_entity = {idx: label for idx, label in enumerate(entity_labels)}
            relation_to_id = {label: idx for idx, label in enumerate(relation_labels)}
            id_to_relation = {idx: label for idx, label in enumerate(relation_labels)}
            txn.put(b"entity_to_id", pickle.dumps(entity_to_id))
            txn.put(b"id_to_entity", pickle.dumps(id_to_entity))
            txn.put(b"relation_to_id", pickle.dumps(relation_to_id))
            txn.put(b"id_to_relation", pickle.dumps(id_to_relation))
    finally:
        env.close()


def _local_indices(node_ids: Sequence[int], targets: Sequence[int]) -> List[int]:
    position = {nid: idx for idx, nid in enumerate(node_ids)}
    return [position[t] for t in targets if t in position]


def _serialize_sample(sample: Dict) -> bytes:
    return pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)


def _write_sample(txn: lmdb.Transaction, sample_key: bytes, payload: bytes) -> None:
    txn.put(sample_key, payload)


def _resolve_lmdb_map_config(cfg: DictConfig) -> Tuple[int, int, float, Optional[int]]:
    map_size_gb_cfg = cfg.get("map_size_gb")
    if map_size_gb_cfg is None:
        raise ValueError("map_size_gb must be set for LMDB materialization.")
    map_size_gb = int(map_size_gb_cfg)
    growth_gb_cfg = cfg.get("map_growth_gb")
    growth_gb = _DEFAULT_LMDB_MAP_GROWTH_GB if growth_gb_cfg is None else int(growth_gb_cfg)
    growth_factor_cfg = cfg.get("map_growth_factor")
    growth_factor = _DEFAULT_LMDB_MAP_GROWTH_FACTOR if growth_factor_cfg is None else float(growth_factor_cfg)
    max_gb_cfg = cfg.get("map_max_gb")
    max_map_size_bytes = None if max_gb_cfg is None else int(max_gb_cfg) * _BYTES_PER_GB
    if map_size_gb < _LMDB_MAP_SIZE_GB_MIN:
        raise ValueError(f"map_size_gb must be >= {_LMDB_MAP_SIZE_GB_MIN}, got {map_size_gb}")
    if growth_gb <= _LMDB_GROWTH_GB_MIN and growth_factor <= _LMDB_GROWTH_FACTOR_MIN:
        raise ValueError(
            "LMDB map growth disabled; set map_growth_gb > 0 or map_growth_factor > 1.0."
        )
    map_size_bytes = map_size_gb * _BYTES_PER_GB
    growth_bytes = growth_gb * _BYTES_PER_GB
    if max_map_size_bytes is not None and max_map_size_bytes < map_size_bytes:
        raise ValueError("map_max_gb must be >= map_size_gb when set.")
    return map_size_bytes, growth_bytes, growth_factor, max_map_size_bytes


def _next_lmdb_map_size_bytes(
    current_size: int,
    growth_bytes: int,
    growth_factor: float,
    max_size: Optional[int],
) -> int:
    next_size = max(current_size + growth_bytes, int(current_size * growth_factor))
    if max_size is not None:
        next_size = min(next_size, max_size)
    if next_size <= current_size:
        raise RuntimeError("LMDB map size limit reached; increase map_max_gb to continue.")
    return next_size


def _replay_pending_with_growth(
    *,
    env: lmdb.Environment,
    pending_payloads: List[Tuple[bytes, bytes]],
    map_size_bytes: int,
    growth_bytes: int,
    growth_factor: float,
    max_size_bytes: Optional[int],
) -> Tuple[lmdb.Transaction, int]:
    while True:
        txn = env.begin(write=True)
        try:
            for key, payload in pending_payloads:
                txn.put(key, payload)
            return txn, map_size_bytes
        except lmdb.MapFullError:
            txn.abort()
            map_size_bytes = _next_lmdb_map_size_bytes(
                current_size=map_size_bytes,
                growth_bytes=growth_bytes,
                growth_factor=growth_factor,
                max_size=max_size_bytes,
            )
            env.set_mapsize(map_size_bytes)


def _commit_pending_with_growth(
    *,
    env: lmdb.Environment,
    txn: lmdb.Transaction,
    pending_payloads: List[Tuple[bytes, bytes]],
    map_size_bytes: int,
    growth_bytes: int,
    growth_factor: float,
    max_size_bytes: Optional[int],
) -> Tuple[lmdb.Transaction, int]:
    while True:
        try:
            txn.commit()
            return env.begin(write=True), map_size_bytes
        except lmdb.MapFullError:
            txn.abort()
            map_size_bytes = _next_lmdb_map_size_bytes(
                current_size=map_size_bytes,
                growth_bytes=growth_bytes,
                growth_factor=growth_factor,
                max_size=max_size_bytes,
            )
            env.set_mapsize(map_size_bytes)
            txn = env.begin(write=True)
            for key, payload in pending_payloads:
                txn.put(key, payload)


def _prepare_lmdb_dir(path: Path, *, overwrite: bool) -> Path:
    """Prepare a clean temporary LMDB directory and return its path.

    We always write into a sibling ``*.tmp`` directory and atomically swap on success.
    This prevents stale keys from older builds from leaking into the current dataset.
    """
    tmp_path = Path(str(path) + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"LMDB already exists at {path}; set overwrite_lmdb=true to rebuild deterministically.")
    _ensure_dir(tmp_path)
    return tmp_path


def _finalize_lmdb_dir(*, tmp_path: Path, final_path: Path, overwrite: bool) -> None:
    if not tmp_path.exists():
        raise FileNotFoundError(f"Temporary LMDB dir missing: {tmp_path}")
    if final_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing LMDB at {final_path}")
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)


def build_dataset(cfg: DictConfig) -> None:
    dataset_cfg = cfg.get("dataset") if hasattr(cfg, "get") else {}
    dataset_name = str(dataset_cfg.get("name", "") or "")
    dataset_scope = str(dataset_cfg.get("dataset_scope", "") or "").strip().lower()
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

    num_topics = int(cfg.get("num_topics", 0))
    if num_topics != _NUM_TOPICS_BINARY:
        raise ValueError(
            f"num_topics must be {_NUM_TOPICS_BINARY} (seed vs non-seed), got {num_topics}"
        )

    # Load vocabulary tables (small) and convert to Python dicts
    entity_vocab = _load_parquet(Path(cfg.parquet_dir) / "entity_vocab.parquet").to_pydict()
    embedding_vocab = _load_parquet(Path(cfg.parquet_dir) / "embedding_vocab.parquet").to_pydict()
    relation_vocab = _load_parquet(Path(cfg.parquet_dir) / "relation_vocab.parquet").to_pydict()

    # Sort vocab by explicit ids to ensure determinism
    entity_rows = sorted(zip(entity_vocab["entity_id"], entity_vocab["label"]), key=lambda x: x[0])
    entity_labels: List[str] = [str(label) for _, label in entity_rows]
    relation_rows = sorted(zip(relation_vocab["relation_id"], relation_vocab["label"]), key=lambda x: x[0])
    relation_labels: List[str] = [str(label) for _, label in relation_rows]

    vocab_map_size = cfg.get("vocab_map_size_gb", _LMDB_MAP_SIZE_GB_MIN) * _BYTES_PER_GB
    _write_vocab_lmdb(
        entity_labels,
        relation_labels,
        Path(cfg.output_dir) / "vocabulary" / "vocabulary.lmdb",
        vocab_map_size,
    )

    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    require_precomputed_questions = bool(cfg.get("require_precomputed_questions", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    emb_dir = Path(cfg.output_dir) / "embeddings"
    _ensure_dir(emb_dir)
    entity_emb_path = emb_dir / "entity_embeddings.pt"
    relation_emb_path = emb_dir / "relation_embeddings.pt"

    encoder: Optional[TextEncoder] = None

    def _get_encoder() -> TextEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = TextEncoder(cfg.encoder, cfg.device, cfg.fp16, cfg.progress_bar)
        return encoder

    if not use_precomputed_embeddings and reuse_embeddings_if_exists:
        if entity_emb_path.exists() and relation_emb_path.exists():
            print(f"Reusing embeddings at {emb_dir}")
            use_precomputed_embeddings = True

    if use_precomputed_embeddings:
        missing_paths = [str(p) for p in (entity_emb_path, relation_emb_path) if not p.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Precomputed embeddings missing: {missing_paths}")
    else:
        encoder = _get_encoder()
        # Build entity embedding tensor with slot 0 reserved for non-text placeholders.
        emb_rows = sorted(zip(embedding_vocab["embedding_id"], embedding_vocab["label"]), key=lambda x: x[0])
        text_labels: List[str] = [str(label) for _, label in emb_rows]
        text_ids: List[int] = [int(eid) for eid, _ in emb_rows]
        print("Encoding textual entity labels...")
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
        print("Encoding relation labels...")
        relation_emb = encoder.encode(relation_labels, cfg.batch_size, show_progress=cfg.progress_bar, desc="Relations")
        torch.save(relation_emb, relation_emb_path)

    # Sub-datasets are no longer materialized; use split filters in parquet preprocessing instead.

    # Load main tables as pyarrow objects without converting to dicts
    graphs_table = _load_parquet(Path(cfg.parquet_dir) / "graphs.parquet")
    questions_table = _load_parquet(Path(cfg.parquet_dir) / "questions.parquet")
    questions_have_emb = "question_emb" in questions_table.schema.names
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
        if require_precomputed_questions:
            raise RuntimeError(
                "questions.parquet is missing required column `question_emb`. "
            "Re-run scripts/build_retrieval_pipeline.py with precomputed question embeddings."
            )
        use_precomputed_questions = False
    graph_ids_all = graphs_table.column("graph_id")
    # Integrity: graph_id must be unique; detect duplicates early.
    distinct = graph_ids_all.unique()
    if len(distinct) != len(graph_ids_all):
        # Find a few offending ids for debugging.
        from collections import Counter

        counts = Counter(graph_ids_all.to_pylist())
        dupes = [gid for gid, c in counts.items() if c > 1][:5]
        raise RuntimeError(f"Duplicate graph_id detected in graphs.parquet, examples: {dupes}")

    graph_id_list = graph_ids_all.to_pylist()
    graph_id_to_row: Dict[str, int] = {gid: idx for idx, gid in enumerate(graph_id_list)}
    if "positive_triple_mask" not in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet is missing required column `positive_triple_mask`. "
            "Re-run scripts/build_retrieval_pipeline.py to regenerate normalized parquet with triple-level supervision."
        )
    if "pair_edge_offsets" in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet contains deprecated column `pair_edge_offsets`. "
            "Re-run scripts/build_retrieval_pipeline.py to regenerate with pair_edge_counts only."
        )
    deprecated_cols = [name for name in ("gt_path_edge_indices", "gt_path_node_indices", "gt_source") if name in graphs_table.schema.names]
    if deprecated_cols:
        raise RuntimeError(
            "graphs.parquet contains deprecated columns "
            f"{deprecated_cols}. Re-run scripts/build_retrieval_pipeline.py to regenerate with the new schema."
        )
    if "pair_edge_counts" not in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet is missing required column `pair_edge_counts`. "
            "Re-run scripts/build_retrieval_pipeline.py to regenerate with pair_edge_counts."
        )
    if "pair_edge_local_ids" not in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet is missing required column `pair_edge_local_ids`. "
            "Re-run scripts/build_retrieval_pipeline.py to regenerate with pair_edge_local_ids."
        )
    if "pair_edge_indices" in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet contains deprecated column `pair_edge_indices`. "
            "Re-run scripts/build_retrieval_pipeline.py to regenerate with pair_edge_local_ids."
        )

    def _require_graph_col(name: str) -> List:
        if name not in graphs_table.schema.names:
            raise RuntimeError(
                f"graphs.parquet is missing required column `{name}`. "
                "Re-run scripts/build_retrieval_pipeline.py to regenerate with the updated schema."
            )
        return graphs_table.column(name).to_pylist()

    graph_cols: Dict[str, List] = {
        "node_entity_ids": _require_graph_col("node_entity_ids"),
        "node_embedding_ids": _require_graph_col("node_embedding_ids"),
        "edge_src": _require_graph_col("edge_src"),
        "edge_dst": _require_graph_col("edge_dst"),
        "edge_relation_ids": _require_graph_col("edge_relation_ids"),
        # Fixed invariant: retriever supervision is ALWAYS triple-level.
        "positive_triple_mask": _require_graph_col("positive_triple_mask"),
        "pair_start_node_locals": _require_graph_col("pair_start_node_locals"),
        "pair_answer_node_locals": _require_graph_col("pair_answer_node_locals"),
        "pair_edge_local_ids": _require_graph_col("pair_edge_local_ids"),
        "pair_edge_counts": _require_graph_col("pair_edge_counts"),
    }
    has_pair_shortest_lengths = "pair_shortest_lengths" in graphs_table.schema.names
    if has_pair_shortest_lengths:
        graph_cols["pair_shortest_lengths"] = graphs_table.column("pair_shortest_lengths").to_pylist()

    questions_rows = questions_table.num_rows
    print(f"Preparing {questions_rows} samples...")

    overwrite_lmdb = bool(cfg.get("overwrite_lmdb", True))

    # Set up LMDB environments (core + aux)
    _LMDB_CORE_KEY = "core"
    _LMDB_AUX_KEY = "aux"
    envs: Dict[str, Dict[str, lmdb.Environment]] = {}
    all_splits = questions_table.column("split").unique().to_pylist()
    all_splits = _validate_split_names(all_splits, context="questions.parquet")
    map_size_bytes, map_growth_bytes, map_growth_factor, map_max_bytes = _resolve_lmdb_map_config(cfg)
    map_sizes: Dict[str, Dict[str, int]] = {
        split: {_LMDB_CORE_KEY: map_size_bytes, _LMDB_AUX_KEY: map_size_bytes} for split in all_splits
    }
    tmp_dirs: Dict[str, Dict[str, Path]] = {}
    for split in all_splits:
        split_key = str(split)
        tmp_dirs[split_key] = {}
        envs[split_key] = {}
        core_final_dir = emb_dir / f"{split}.lmdb"
        aux_final_dir = emb_dir / f"{split}{_AUX_LMDB_SUFFIX}"
        core_tmp_dir = _prepare_lmdb_dir(core_final_dir, overwrite=overwrite_lmdb)
        aux_tmp_dir = _prepare_lmdb_dir(aux_final_dir, overwrite=overwrite_lmdb)
        tmp_dirs[split_key][_LMDB_CORE_KEY] = core_tmp_dir
        tmp_dirs[split_key][_LMDB_AUX_KEY] = aux_tmp_dir
        envs[split_key][_LMDB_CORE_KEY] = lmdb.open(
            str(core_tmp_dir),
            map_size=map_sizes[split_key][_LMDB_CORE_KEY],
            subdir=True,
            lock=False,
        )
        envs[split_key][_LMDB_AUX_KEY] = lmdb.open(
            str(aux_tmp_dir),
            map_size=map_sizes[split_key][_LMDB_AUX_KEY],
            subdir=True,
            lock=False,
        )

    success = False
    try:
        txn_cache: Dict[str, Dict[str, lmdb.Transaction]] = {}
        pending_payloads: Dict[str, Dict[str, List[Tuple[bytes, bytes]]]] = {}
        for split_key, env_group in envs.items():
            txn_cache[split_key] = {
                _LMDB_CORE_KEY: env_group[_LMDB_CORE_KEY].begin(write=True),
                _LMDB_AUX_KEY: env_group[_LMDB_AUX_KEY].begin(write=True),
            }
            pending_payloads[split_key] = {_LMDB_CORE_KEY: [], _LMDB_AUX_KEY: []}

        parquet_chunk_size = _resolve_parquet_chunk_size(cfg, fallback=int(cfg.batch_size))
        question_batches = questions_table.to_batches(max_chunksize=parquet_chunk_size)
        total_batches = questions_table.num_rows / parquet_chunk_size
        pbar = tqdm(question_batches, total=int(total_batches) + 1, desc="Writing LMDB")
        missing_graph_ids: List[str] = []

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
                labels = graph_cols["positive_triple_mask"][g_idx]

                num_nodes = len(node_entity_ids)
                num_edges = len(edge_src)
                if num_edges <= 0:
                    raise ValueError(
                        f"Invalid graph with zero edges for {graph_id} (split={split}). "
                        "Fix raw parquet/filters and rebuild; empty edge_index is unsupported."
                    )
                node_global_ids = torch.tensor(node_entity_ids, dtype=torch.long)
                node_emb_ids = torch.tensor(node_embedding_ids, dtype=torch.long)
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_attr = torch.tensor(edge_rel, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.float32)
                if label_tensor.numel() != num_edges:
                    raise ValueError(
                        f"Label length mismatch for {graph_id}: labels={label_tensor.numel()} vs num_edges={num_edges}. "
                        "Rebuild normalized parquet caches to match the updated schema."
                    )

                q_entities = q_batch_dict["seed_entity_ids"][i]
                a_entities = q_batch_dict["answer_entity_ids"][i]
                if q_entities is None:
                    raise ValueError(f"seed_entity_ids is null for {graph_id}")
                if a_entities is None:
                    raise ValueError(f"answer_entity_ids is null for {graph_id}")
                q_local = _local_indices(node_entity_ids, q_entities)
                a_local = _local_indices(node_entity_ids, a_entities)
                topic_entity_mask = torch.zeros(num_nodes, dtype=torch.long)
                if q_local:
                    topic_entity_mask[torch.as_tensor(q_local, dtype=torch.long)] = 1
                topic_one_hot = F.one_hot(topic_entity_mask, num_classes=num_topics).to(dtype=torch.float32)
                pair_start_node_locals = graph_cols["pair_start_node_locals"][g_idx]
                pair_answer_node_locals = graph_cols["pair_answer_node_locals"][g_idx]
                pair_edge_local_ids = graph_cols["pair_edge_local_ids"][g_idx]
                pair_edge_counts = graph_cols["pair_edge_counts"][g_idx]
                pair_shortest_lengths = graph_cols["pair_shortest_lengths"][g_idx] if has_pair_shortest_lengths else None
                if pair_start_node_locals is None or pair_answer_node_locals is None:
                    raise ValueError(f"pair_start/answer_node_locals missing for {graph_id}")
                if pair_edge_local_ids is None or pair_edge_counts is None:
                    raise ValueError(f"pair_edge_local_ids/counts missing for {graph_id}")
                if pair_start_node_locals and len(pair_edge_counts) != len(pair_start_node_locals):
                    raise ValueError(
                        f"pair_edge_counts length {len(pair_edge_counts)} != pair_count "
                        f"{len(pair_start_node_locals)} for {graph_id}"
                    )

                core_sample = {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "labels": label_tensor,
                    "num_nodes": num_nodes,
                    "node_global_ids": node_global_ids,
                    "node_embedding_ids": node_emb_ids,
                    "question_emb": q_batch_emb[i].unsqueeze(0),
                    "topic_one_hot": topic_one_hot,
                    "q_local_indices": torch.as_tensor(q_local, dtype=torch.long),
                    "a_local_indices": torch.as_tensor(a_local, dtype=torch.long),
                    "answer_entity_ids": torch.as_tensor(a_entities, dtype=torch.long),
                    "answer_entity_ids_len": torch.tensor([len(a_entities)], dtype=torch.long),
                }
                aux_sample = {
                    "question": q_batch_dict["question"][i],
                    # Seed entities used by downstream start_entity_ids.
                    "seed_entity_ids": torch.as_tensor(q_entities, dtype=torch.long),
                    "pair_start_node_locals": torch.as_tensor(pair_start_node_locals, dtype=torch.long),
                    "pair_answer_node_locals": torch.as_tensor(pair_answer_node_locals, dtype=torch.long),
                    "pair_edge_local_ids": torch.as_tensor(pair_edge_local_ids, dtype=torch.long),
                    "pair_edge_counts": torch.as_tensor(pair_edge_counts, dtype=torch.long),
                }
                if has_pair_shortest_lengths:
                    aux_sample["pair_shortest_lengths"] = torch.as_tensor(pair_shortest_lengths, dtype=torch.long)

                sample_key = graph_id.encode("utf-8")
                core_payload = _serialize_sample(core_sample)
                aux_payload = _serialize_sample(aux_sample)

                for kind, payload in (
                    (_LMDB_CORE_KEY, core_payload),
                    (_LMDB_AUX_KEY, aux_payload),
                ):
                    pending_payloads[split][kind].append((sample_key, payload))
                    txn = txn_cache[split][kind]
                    try:
                        _write_sample(txn, sample_key, payload)
                    except lmdb.MapFullError:
                        txn.abort()
                        txn, map_sizes[split][kind] = _replay_pending_with_growth(
                            env=envs[split][kind],
                            pending_payloads=pending_payloads[split][kind],
                            map_size_bytes=map_sizes[split][kind],
                            growth_bytes=map_growth_bytes,
                            growth_factor=map_growth_factor,
                            max_size_bytes=map_max_bytes,
                        )
                        txn_cache[split][kind] = txn
                    if len(pending_payloads[split][kind]) >= cfg.txn_size:
                        txn_cache[split][kind], map_sizes[split][kind] = _commit_pending_with_growth(
                            env=envs[split][kind],
                            txn=txn_cache[split][kind],
                            pending_payloads=pending_payloads[split][kind],
                            map_size_bytes=map_sizes[split][kind],
                            growth_bytes=map_growth_bytes,
                            growth_factor=map_growth_factor,
                            max_size_bytes=map_max_bytes,
                        )
                        pending_payloads[split][kind].clear()

        for split_key, txn_group in txn_cache.items():
            for kind, txn in txn_group.items():
                if pending_payloads[split_key][kind]:
                    txn_cache[split_key][kind], map_sizes[split_key][kind] = _commit_pending_with_growth(
                        env=envs[split_key][kind],
                        txn=txn,
                        pending_payloads=pending_payloads[split_key][kind],
                        map_size_bytes=map_sizes[split_key][kind],
                        growth_bytes=map_growth_bytes,
                        growth_factor=map_growth_factor,
                        max_size_bytes=map_max_bytes,
                    )
                    pending_payloads[split_key][kind].clear()
        success = True
    finally:
        for env_group in envs.values():
            for env in env_group.values():
                env.close()
        if success:
            for split in all_splits:
                split_key = str(split)
                core_final_dir = emb_dir / f"{split}.lmdb"
                aux_final_dir = emb_dir / f"{split}{_AUX_LMDB_SUFFIX}"
                _finalize_lmdb_dir(
                    tmp_path=tmp_dirs[split_key][_LMDB_CORE_KEY],
                    final_path=core_final_dir,
                    overwrite=overwrite_lmdb,
                )
                _finalize_lmdb_dir(
                    tmp_path=tmp_dirs[split_key][_LMDB_AUX_KEY],
                    final_path=aux_final_dir,
                    overwrite=overwrite_lmdb,
                )
        else:
            # Keep tmp dirs for debugging, but avoid silently masking partial builds.
            pass



# === Unified pipeline runner ===

def _default_filter() -> SplitFilter:
    return SplitFilter(skip_no_topic=False, skip_no_ans=False, skip_no_path=False)


def _as_filter(section, *, default: SplitFilter) -> SplitFilter:
    if section is None:
        return default
    return SplitFilter(
        skip_no_topic=bool(section.get("skip_no_topic", False)),
        skip_no_ans=bool(section.get("skip_no_ans", False)),
        skip_no_path=bool(section.get("skip_no_path", False)),
    )


def _resolve_override_filters(cfg, *, default_filter: SplitFilter):
    filter_cfg = cfg.get("filter")
    if filter_cfg is None:
        return {}
    overrides = {}
    for key in filter_cfg.keys():
        if key in {"train", "eval"}:
            continue
        overrides[str(key)] = _as_filter(filter_cfg.get(key), default=default_filter)
    return overrides


def _resolve_parquet_chunk_size(cfg, *, fallback: int) -> int:
    chunk_cfg = cfg.get("parquet_chunk_size")
    chunk_size = fallback if chunk_cfg is None else int(chunk_cfg)
    if chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"parquet_chunk_size must be >= {_MIN_CHUNK_SIZE}, got {chunk_size}")
    return chunk_size


def _resolve_parquet_num_workers(cfg) -> int:
    workers_cfg = cfg.get("parquet_num_workers", _DISABLE_PARALLEL_WORKERS)
    num_workers = int(workers_cfg)
    if num_workers < _DISABLE_PARALLEL_WORKERS:
        raise ValueError(f"parquet_num_workers must be >= {_DISABLE_PARALLEL_WORKERS}, got {num_workers}")
    return num_workers


def _build_embedding_cfg(cfg):
    embed_flags = {
        "precompute_entities": bool(cfg.get("precompute_entities", False)),
        "precompute_relations": bool(cfg.get("precompute_relations", False)),
        "precompute_questions": bool(cfg.get("precompute_questions", False)),
        "canonicalize_relations": bool(cfg.get("canonicalize_relations", False)),
    }
    if not any(embed_flags.values()):
        return None
    embeddings_out_dir_cfg = cfg.get("embeddings_out_dir")
    if not embeddings_out_dir_cfg:
        raise ValueError("embeddings_out_dir must be set when embedding precompute is enabled.")
    return EmbeddingConfig(
        encoder=str(cfg.get("encoder", "")),
        device=str(cfg.get("device", "cuda")),
        batch_size=int(cfg.get("batch_size", 64)),
        fp16=bool(cfg.get("fp16", False)),
        progress_bar=bool(cfg.get("progress_bar", True)),
        embeddings_out_dir=Path(hydra.utils.to_absolute_path(embeddings_out_dir_cfg)),
        precompute_entities=embed_flags["precompute_entities"],
        precompute_relations=embed_flags["precompute_relations"],
        precompute_questions=embed_flags["precompute_questions"],
        canonicalize_relations=embed_flags["canonicalize_relations"],
        cosine_eps=float(cfg.get("cosine_eps", 1e-6)),
    )


def _validate_pipeline_cfg(cfg):
    precompute_embeddings = bool(cfg.get("precompute_entities", False)) or bool(cfg.get("precompute_relations", False))
    precompute_questions = bool(cfg.get("precompute_questions", False))
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))
    use_precomputed_questions = bool(cfg.get("use_precomputed_questions", False))
    require_precomputed_questions = bool(cfg.get("require_precomputed_questions", False))
    skip_parquet_stage = bool(cfg.get("skip_parquet_stage", False))
    skip_lmdb_stage = bool(cfg.get("skip_lmdb_stage", False))
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    _resolve_parquet_chunk_size(cfg, fallback=int(cfg.get("batch_size", _MIN_CHUNK_SIZE)))
    _resolve_parquet_num_workers(cfg)

    if skip_parquet_stage or skip_lmdb_stage:
        raise ValueError("skip_parquet_stage/skip_lmdb_stage are disabled; run the unified parquet+LMDB pipeline.")

    if use_precomputed_embeddings and not precompute_embeddings and not skip_parquet_stage and not reuse_embeddings_if_exists:
        raise ValueError(
            "use_precomputed_embeddings=true requires precompute_entities or precompute_relations "
            "to be enabled in the same pipeline run."
        )
    if use_precomputed_questions and not precompute_questions and not skip_parquet_stage:
        raise ValueError(
            "use_precomputed_questions=true requires precompute_questions "
            "to be enabled in the same pipeline run."
        )
    if precompute_questions and not require_precomputed_questions:
        raise ValueError(
            "precompute_questions=true requires require_precomputed_questions=true "
            "to ensure LMDB uses the freshly computed embeddings."
        )

    parquet_dir = cfg.get("parquet_dir")
    out_dir = cfg.get("out_dir")
    if parquet_dir and out_dir:
        parquet_path = Path(str(parquet_dir))
        out_path = Path(str(out_dir))
        if parquet_path.resolve() != out_path.resolve():
            raise ValueError(
                "parquet_dir must match out_dir in the unified pipeline. "
                f"Got parquet_dir={parquet_path} vs out_dir={out_path}."
            )


def _assert_parquet_outputs(cfg) -> None:
    parquet_dir = cfg.get("parquet_dir")
    if not parquet_dir:
        raise ValueError("skip_parquet_stage=true requires parquet_dir to be set.")
    parquet_path = Path(hydra.utils.to_absolute_path(parquet_dir))
    required = [
        parquet_path / "graphs.parquet",
        parquet_path / "questions.parquet",
        parquet_path / "entity_vocab.parquet",
        parquet_path / "embedding_vocab.parquet",
        parquet_path / "relation_vocab.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"skip_parquet_stage=true but missing parquet outputs: {missing}")


def _run_parquet_stage(cfg):
    raw_root = Path(hydra.utils.to_absolute_path(cfg.raw_root))
    out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
    dataset_name = cfg.get("dataset_name") or cfg.get("dataset") or "dataset"
    text_cfg = build_text_entity_config(cfg)
    path_mode = str(cfg.get("path_mode", _PATH_MODE_UNDIRECTED))
    parquet_chunk_size = _resolve_parquet_chunk_size(cfg, fallback=int(cfg.get("batch_size", _MIN_CHUNK_SIZE)))
    parquet_num_workers = _resolve_parquet_num_workers(cfg)
    reuse_embeddings_if_exists = bool(cfg.get("reuse_embeddings_if_exists", False))

    filter_cfg = cfg.get("filter")
    default_filter = _default_filter()
    if filter_cfg is None:
        train_filter = default_filter
        eval_filter = default_filter
        override_filters = {}
    else:
        train_filter = _as_filter(filter_cfg.train, default=default_filter)
        eval_filter = _as_filter(filter_cfg.eval, default=default_filter)
        override_filters = _resolve_override_filters(cfg, default_filter=default_filter)

    embedding_cfg = _build_embedding_cfg(cfg)

    preprocess(
        dataset=str(dataset_name),
        kb=cfg.kb,
        raw_root=raw_root,
        out_dir=out_dir,
        column_map=dict(cfg.column_map),
        entity_normalization=cfg.entity_normalization,
        text_cfg=text_cfg,
        train_filter=train_filter,
        eval_filter=eval_filter,
        override_filters=override_filters,
        path_mode=path_mode,
        dedup_edges=bool(cfg.get("dedup_edges", True)),
        validate_graph_edges=bool(cfg.get("validate_graph_edges", _VALIDATE_GRAPH_EDGES_DEFAULT)),
        remove_self_loops=bool(cfg.get("remove_self_loops", _REMOVE_SELF_LOOPS_DEFAULT)),
        embedding_cfg=embedding_cfg,
        emit_sub_filter=bool(cfg.get("emit_sub_filter", False)),
        sub_filter_filename=str(cfg.get("sub_filter_filename", "sub_filter.json")),
        emit_nonzero_positive_filter=bool(cfg.get("emit_nonzero_positive_filter", False)),
        nonzero_positive_filter_filename=str(
            cfg.get("nonzero_positive_filter_filename", "nonzero_positive_filter.json")
        ),
        nonzero_positive_filter_splits=cfg.get("nonzero_positive_filter_splits"),
        parquet_chunk_size=parquet_chunk_size,
        parquet_num_workers=parquet_num_workers,
        reuse_embeddings_if_exists=reuse_embeddings_if_exists,
    )


def build_pipeline(cfg):
    _validate_pipeline_cfg(cfg)
    skip_parquet_stage = bool(cfg.get("skip_parquet_stage", False))
    skip_lmdb_stage = bool(cfg.get("skip_lmdb_stage", False))
    if not skip_parquet_stage:
        _run_parquet_stage(cfg)
    else:
        _assert_parquet_outputs(cfg)
    if not skip_lmdb_stage:
        _ensure_dir(Path(cfg.output_dir))
        build_dataset(cfg)


if hydra is not None:

    @hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_pipeline")
    def main(cfg):
        build_pipeline(cfg)

else:  # pragma: no cover

    def main(cfg):
        raise ModuleNotFoundError("hydra-core is required to run scripts/build_retrieval_pipeline.py")


if __name__ == "__main__":
    main()
