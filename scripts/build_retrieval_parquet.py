#!/usr/bin/env python3
"""Normalize local HF parquet shards into retrieval-ready parquet files.

Targets Hydra data configs (e.g., WebQSP/CWQ/GTSQA/KGQAGen) under /mnt/data/retrieval_dataset/<dataset>/raw/*.
Implements topic/answer/path 连通性过滤 (train vs eval separately) 并分离结构 ID 与嵌入 ID：
  - Structural entity_id: unique per entity string.
  - Embedding space: embedding_id=0 reserved for non-text entities; textual entities get
    stable ids starting at 1 (text-first, lexicographic order).
Outputs five parquet files:
  graphs.parquet, questions.parquet, entity_vocab.parquet, embedding_vocab.parquet, relation_vocab.parquet
compatible with build_retrieval_dataset.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
try:
    from omegaconf import DictConfig
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = object  # type: ignore[assignment]
from tqdm import tqdm


@dataclass(frozen=True)
class SplitFilter:
    skip_no_topic: bool
    skip_no_ans: bool
    skip_no_path: bool


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
        return 0 if not self._text_cfg.is_text(ent) else self._embedding_id_for_text(ent)

    def _embedding_id_for_text(self, ent: str) -> int:
        if not self._finalized:
            self.finalize()
        return self._text_kg_id_to_embed_id.get(ent, 0)


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


@dataclass
class ParquetDatasetWriter:
    out_dir: Path
    graphs: List[GraphRecord] = field(default_factory=list)
    questions: List[Dict[str, object]] = field(default_factory=list)
    graph_writer: pq.ParquetWriter | None = None
    question_writer: pq.ParquetWriter | None = None

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
            table_q = pa.table(
                {
                    "question_uid": pa.array([row["question_uid"] for row in self.questions], type=pa.string()),
                    "dataset": pa.array([row["dataset"] for row in self.questions], type=pa.string()),
                    "split": pa.array([row["split"] for row in self.questions], type=pa.string()),
                    "kb": pa.array([row["kb"] for row in self.questions], type=pa.string()),
                    "question": pa.array([row["question"] for row in self.questions], type=pa.string()),
                    "paraphrased_question": pa.array(
                        [row.get("paraphrased_question") for row in self.questions], type=pa.string()
                    ),
                    "seed_entity_ids": pa.array(
                        [row["seed_entity_ids"] for row in self.questions], type=pa.list_(pa.int64())
                    ),
                    "answer_entity_ids": pa.array(
                        [row["answer_entity_ids"] for row in self.questions], type=pa.list_(pa.int64())
                    ),
                    "seed_embedding_ids": pa.array(
                        [row["seed_embedding_ids"] for row in self.questions], type=pa.list_(pa.int64())
                    ),
                    "answer_embedding_ids": pa.array(
                        [row["answer_embedding_ids"] for row in self.questions], type=pa.list_(pa.int64())
                    ),
                    "answer_texts": pa.array([row["answer_texts"] for row in self.questions], type=pa.list_(pa.string())),
                    "graph_id": pa.array([row["graph_id"] for row in self.questions], type=pa.string()),
                    "metadata": pa.array([row["metadata"] for row in self.questions], type=pa.string()),
                }
            )
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


# Helpers


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_path_mode(path_mode: str) -> str:
    mode = str(path_mode)
    if mode not in _PATH_MODES:
        raise ValueError(f"Unsupported path_mode: {mode}. Expected one of {_PATH_MODES}.")
    return mode


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
    dist = [-1] * num_nodes
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

    adjacency = _build_undirected_adjacency(num_nodes, edge_src, edge_dst)
    starts = sorted({int(s) for s in sources if 0 <= int(s) < num_nodes})
    answers = sorted({int(t) for t in targets if 0 <= int(t) < num_nodes})
    if not starts or not answers:
        return [False] * num_edges, [], [], [], [], []

    dist_from_start: Dict[int, List[int]] = {s: _bfs_dist(num_nodes, adjacency, [s]) for s in starts}
    dist_to_answer: Dict[int, List[int]] = {a: _bfs_dist(num_nodes, adjacency, [a]) for a in answers}

    mask = [False] * num_edges
    pair_start_nodes: List[int] = []
    pair_answer_nodes: List[int] = []
    pair_edge_local_ids: List[int] = []
    pair_edge_counts: List[int] = []
    pair_shortest_lengths: List[int] = []

    for s in starts:
        dist_s = dist_from_start[s]
        for a in answers:
            dist_a = dist_to_answer[a]
            dist_sa = dist_s[a]
            if dist_sa < 0:
                continue
            pair_start_nodes.append(s)
            pair_answer_nodes.append(a)
            pair_shortest_lengths.append(int(dist_sa))
            before = len(pair_edge_local_ids)
            for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
                u = int(u_raw)
                v = int(v_raw)
                if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                    continue
                if dist_s[u] >= 0 and dist_a[v] >= 0 and dist_s[u] + 1 + dist_a[v] == dist_sa:
                    pair_edge_local_ids.append(idx)
                    mask[idx] = True
                    continue
                if dist_s[v] >= 0 and dist_a[u] >= 0 and dist_s[v] + 1 + dist_a[u] == dist_sa:
                    pair_edge_local_ids.append(idx)
                    mask[idx] = True
            pair_edge_counts.append(len(pair_edge_local_ids) - before)

    return (
        mask,
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

    adjacency = _build_directed_adjacency(num_nodes, edge_src, edge_dst)
    reverse_adjacency = _build_directed_adjacency(num_nodes, edge_dst, edge_src)
    dist_from_start = {s: _bfs_dist(num_nodes, adjacency, [s]) for s in starts}
    dist_to_answer = {a: _bfs_dist(num_nodes, reverse_adjacency, [a]) for a in answers}

    mask = [False] * num_edges
    pair_start_nodes: List[int] = []
    pair_answer_nodes: List[int] = []
    pair_edge_local_ids: List[int] = []
    pair_edge_counts: List[int] = []
    pair_shortest_lengths: List[int] = []

    for s in starts:
        dist_s = dist_from_start[s]
        for a in answers:
            dist_a = dist_to_answer[a]
            dist_sa = dist_s[a]
            if dist_sa < 0:
                continue
            pair_start_nodes.append(s)
            pair_answer_nodes.append(a)
            pair_shortest_lengths.append(int(dist_sa))
            before = len(pair_edge_local_ids)
            for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
                u = int(u_raw)
                v = int(v_raw)
                if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                    continue
                if dist_s[u] >= 0 and dist_a[v] >= 0 and dist_s[u] + 1 + dist_a[v] == dist_sa:
                    pair_edge_local_ids.append(idx)
                    mask[idx] = True
            pair_edge_counts.append(len(pair_edge_local_ids) - before)

    return (
        mask,
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
    *,
    emit_sub_filter: bool = False,
    sub_filter_filename: str = "sub_filter.json",
) -> None:
    path_mode = _validate_path_mode(path_mode)
    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb, text_cfg=text_cfg)
    relation_vocab = RelationVocab(kb=kb)

    available_files = {p.name for p in raw_root.glob("*.parquet")}
    splits = sorted({name.split("-")[0] for name in available_files})
    connectivity_cache: Dict[Tuple[str, str, str], bool] = {}
    total_by_split: Dict[str, int] = {}
    kept_by_split: Dict[str, int] = {}
    sub_by_split: Dict[str, int] = {}
    empty_graph_by_split: Dict[str, int] = {}
    empty_graph_ids: List[str] = []
    empty_graph_id_set: Set[str] = set()
    sub_sample_ids: List[str] = []

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

    # Pass 2: Build graphs and questions
    print("Pass 2: Building graphs and questions...")
    chunk_size = 2000
    base_writer = ParquetDatasetWriter(out_dir=out_dir)

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

        graph = build_graph(sample, entity_vocab, relation_vocab, graph_id, path_mode=path_mode)
        question = build_question_record(sample, entity_vocab, graph_id)
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
                sub_sample_ids.append(graph_id)
                sub_by_split[sample.split] = sub_by_split.get(sample.split, 0) + 1

        if len(base_writer.graphs) >= chunk_size or len(base_writer.questions) >= chunk_size:
            base_writer.flush()

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


def build_graph(
    sample: Sample,
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    graph_id: str,
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
) -> GraphRecord:
    path_mode = _validate_path_mode(path_mode)
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
        src_idx = local_index(h)
        dst_idx = local_index(t)
        rel_idx = relation_vocab.relation_id(r)
        edge_src.append(src_idx)
        edge_dst.append(dst_idx)
        edge_relation_ids.append(rel_idx)
        edge_keys.append((h, r, t))
        edge_key_to_indices.setdefault((h, r, t), []).append(len(edge_keys) - 1)

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

    return GraphRecord(
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


def build_question_record(
    sample: Sample,
    entity_vocab: EntityVocab,
    graph_id: str,
) -> Dict[str, object]:
    seed_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.q_entity]
    answer_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.a_entity]
    seed_embedding_ids = [entity_vocab.embedding_id(ent) for ent in sample.q_entity]
    answer_embedding_ids = [entity_vocab.embedding_id(ent) for ent in sample.a_entity]
    metadata: Dict[str, Any] = {}
    if sample.graph_iso_type is not None:
        metadata["graph_isomorphism"] = sample.graph_iso_type
    if sample.redundant is not None:
        metadata["redundant"] = sample.redundant
    if sample.test_type:
        metadata["test_type"] = sample.test_type
    if sample.answer_subgraph:
        metadata["answer_subgraph_len"] = len(sample.answer_subgraph)
    return {
        "question_uid": graph_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "kb": sample.kb,
        "question": sample.question,
        "paraphrased_question": None,
        "seed_entity_ids": seed_entity_ids,
        "answer_entity_ids": answer_entity_ids,
        "seed_embedding_ids": seed_embedding_ids,
        "answer_embedding_ids": answer_embedding_ids,
        "answer_texts": sample.answer_texts,
        "graph_id": graph_id,
        "metadata": json.dumps(metadata),
    }


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
    table = pa.table(
        {
            "question_uid": pa.array([row["question_uid"] for row in rows], type=pa.string()),
            "dataset": pa.array([row["dataset"] for row in rows], type=pa.string()),
            "split": pa.array([row["split"] for row in rows], type=pa.string()),
            "kb": pa.array([row["kb"] for row in rows], type=pa.string()),
            "question": pa.array([row["question"] for row in rows], type=pa.string()),
            "paraphrased_question": pa.array([row.get("paraphrased_question") for row in rows], type=pa.string()),
            "seed_entity_ids": pa.array([row["seed_entity_ids"] for row in rows], type=pa.list_(pa.int64())),
            "answer_entity_ids": pa.array([row["answer_entity_ids"] for row in rows], type=pa.list_(pa.int64())),
            "seed_embedding_ids": pa.array([row["seed_embedding_ids"] for row in rows], type=pa.list_(pa.int64())),
            "answer_embedding_ids": pa.array([row["answer_embedding_ids"] for row in rows], type=pa.list_(pa.int64())),
            "answer_texts": pa.array([row["answer_texts"] for row in rows], type=pa.list_(pa.string())),
            "graph_id": pa.array([row["graph_id"] for row in rows], type=pa.string()),
            "metadata": pa.array([row["metadata"] for row in rows], type=pa.string()),
        }
    )
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


if hydra is not None:

    @hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_parquet")
    def main(cfg: DictConfig) -> None:
        raw_root = Path(hydra.utils.to_absolute_path(cfg.raw_root))
        out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
        dataset_name = cfg.get("dataset_name") or cfg.get("dataset") or "dataset"
        text_cfg = build_text_entity_config(cfg)
        path_mode = str(cfg.get("path_mode", _PATH_MODE_UNDIRECTED))

        def _as_filter(section: DictConfig) -> SplitFilter:
            return SplitFilter(
                skip_no_topic=bool(section.get("skip_no_topic", False)),
                skip_no_ans=bool(section.get("skip_no_ans", False)),
                skip_no_path=bool(section.get("skip_no_path", False)),
            )

        train_filter = _as_filter(cfg.filter.train)
        eval_filter = _as_filter(cfg.filter.eval)
        override_filters: Dict[str, SplitFilter] = {}
        for key in cfg.filter.keys():
            if key in {"train", "eval"}:
                continue
            override_filters[str(key)] = _as_filter(cfg.filter[key])

        preprocess(
            dataset=dataset_name,
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
            emit_sub_filter=bool(cfg.get("emit_sub_filter", False)),
            sub_filter_filename=str(cfg.get("sub_filter_filename", "sub_filter.json")),
        )

else:  # pragma: no cover

    def main(cfg: DictConfig) -> None:
        raise ModuleNotFoundError("hydra-core is required to run scripts/build_retrieval_parquet.py")


if __name__ == "__main__":
    main()
