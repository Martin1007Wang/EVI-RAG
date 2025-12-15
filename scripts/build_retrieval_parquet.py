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
    use_shortest_path_positive: bool


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


@dataclass
class GraphRecord:
    graph_id: str
    node_entity_ids: List[int]
    node_embedding_ids: List[int]
    node_labels: List[str]
    edge_src: List[int]
    edge_dst: List[int]
    edge_relation_ids: List[int]
    # Transition-level positives: supervision defined on node transitions (u->v),
    # then lifted to all parallel edges between the same endpoints.
    positive_edge_mask: List[bool]
    # Triple-level positives: supervision defined on a specific edge id in the
    # preserved multi-edge list (h,r,t). For shortest-path supervision this is
    # the canonical edge id induced by SubgraphRAG-parity collapsing.
    positive_triple_mask: List[bool]
    gt_path_edge_indices: List[int]
    gt_path_node_indices: List[int]
    # Provenance of gt_path_* (used for downstream label selection).
    gt_source: str


class EntityVocab:
    """Assign structural IDs and embedding IDs; separate text vs non-text."""

    def __init__(self, kb: str) -> None:
        self.kb = kb
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
        if is_text_entity(ent):
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
            is_text = is_text_entity(ent)
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
        return 0 if not is_text_entity(ent) else self._embedding_id_for_text(ent)

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
                    "positive_edge_mask": pa.array(
                        [g.positive_edge_mask for g in self.graphs], type=pa.list_(pa.bool_())
                    ),
                    "positive_triple_mask": pa.array(
                        [g.positive_triple_mask for g in self.graphs], type=pa.list_(pa.bool_())
                    ),
                    "gt_path_edge_indices": pa.array(
                        [g.gt_path_edge_indices for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "gt_path_node_indices": pa.array(
                        [g.gt_path_node_indices for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    "gt_source": pa.array([g.gt_source for g in self.graphs], type=pa.string()),
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


# Helpers


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_text_entity(entity: str) -> bool:
    return not (entity.startswith("m.") or entity.startswith("g."))


def shortest_path_edge_indices_directed(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    seeds: Sequence[int],
    answers: Sequence[int],
    undirected: bool,
) -> Tuple[List[int], List[int]]:
    """Bidirectional shortest-path union (SubgraphRAG parity), preserving multi-edges.

    - Compute all shortest paths seed->answer (forward) and answer->seed (backward).
    - If only one direction exists, use that direction.
    - If both exist, keep only the direction(s) with minimal hop length (could be one side or both).
    - Return the union of edge indices that appear in these shortest paths, plus incident nodes.
    """
    if not seeds or not answers or num_nodes <= 0:
        return [], []

    from collections import deque

    # SubgraphRAG uses `nx.DiGraph()` which collapses parallel edges by (u,v) with last-write-wins
    # on edge attributes. We mirror that behavior by:
    #   - storing a canonical `pair_to_edge_idx[(u,v)] = last_edge_idx`
    #   - building adjacency over unique (u,v) pairs (in first-seen order)
    pair_to_edge_idx: Dict[Tuple[int, int], int] = {}
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
        u = int(u_raw)
        v = int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            if (u, v) not in pair_to_edge_idx:
                adjacency[u].append(v)
            pair_to_edge_idx[(u, v)] = idx
            if undirected:
                if (v, u) not in pair_to_edge_idx:
                    adjacency[v].append(u)
                pair_to_edge_idx[(v, u)] = idx

    def _all_shortest_paths(src: int, dst: int) -> List[List[int]]:
        """Return all shortest node paths (inclusive) from src->dst on the collapsed adjacency."""
        if src == dst:
            return []

        dist = [-1] * num_nodes
        parents: List[List[int]] = [[] for _ in range(num_nodes)]
        dist[src] = 0
        q: deque[int] = deque([src])
        while q:
            cur = q.popleft()
            next_dist = dist[cur] + 1
            for nb in adjacency[cur]:
                if dist[nb] == -1:
                    dist[nb] = next_dist
                    q.append(nb)
                if dist[nb] == next_dist:
                    parents[nb].append(cur)

        if dist[dst] == -1:
            return []

        paths: List[List[int]] = []

        def _dfs(node: int, suffix: List[int]) -> None:
            if node == src:
                paths.append([src] + list(reversed(suffix)))
                return
            for p in parents[node]:
                _dfs(p, suffix + [node])

        _dfs(dst, [])
        return paths

    seeds_unique = [int(s) for s in seeds if 0 <= int(s) < num_nodes]
    answers_unique = [int(a) for a in answers if 0 <= int(a) < num_nodes]
    if not seeds_unique or not answers_unique:
        return [], []

    # SubgraphRAG parity: refine per (seed, answer) pair.
    selected_paths: List[List[int]] = []
    for s in seeds_unique:
        for a in answers_unique:
            forward_paths = _all_shortest_paths(s, a)
            backward_paths = _all_shortest_paths(a, s)
            full_paths = forward_paths + backward_paths
            if not full_paths:
                continue

            # If either direction is missing, keep the existing one(s).
            if (len(forward_paths) == 0) or (len(backward_paths) == 0):
                selected_paths.extend(full_paths)
                continue

            # Both directions exist: keep only shortest-length paths.
            min_len = min(len(p) for p in full_paths)
            selected_paths.extend([p for p in full_paths if len(p) == min_len])

    if not selected_paths:
        return [], []

    edge_set: Set[int] = set()
    node_set: Set[int] = set()
    for path in selected_paths:
        for u, v in zip(path[:-1], path[1:]):
            node_set.add(u)
            node_set.add(v)
            e_idx = pair_to_edge_idx.get((u, v))
            if e_idx is not None:
                edge_set.add(int(e_idx))

    return sorted(edge_set), sorted(node_set)


def has_connectivity(
    graph: Sequence[Tuple[str, str, str]], seeds: Sequence[str], answers: Sequence[str], undirected: bool
) -> bool:
    """Check existence of path seed->answer using local indexing (optionally undirected traversal)."""
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
    path_edges, _ = shortest_path_edge_indices_directed(
        len(node_index), edge_src, edge_dst, seed_ids, answer_ids, undirected=undirected
    )
    return len(path_edges) > 0


def normalize_entity(entity: str, mode: str) -> str:
    if mode == "qid_in_parentheses":
        match = re.search(r"(Q\d+)", entity)
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
    undirected_traversal: bool,
    connectivity_cache: Dict[Tuple[str, str], bool],
) -> bool:
    node_strings = {h for h, _, t in sample.graph} | {t for _, _, t in sample.graph}
    has_topic = any(ent in node_strings for ent in sample.q_entity)
    has_answer = any(ent in node_strings for ent in sample.a_entity)

    has_path = connectivity_cache.get((sample.split, sample.question_id))
    if has_path is None:
        if sample.answer_subgraph:
            has_path = True
        elif split_filter.skip_no_path:
            has_path = has_connectivity(sample.graph, sample.q_entity, sample.a_entity, undirected=undirected_traversal)
        else:
            has_path = True
        connectivity_cache[(sample.split, sample.question_id)] = has_path

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
                                label_match = re.match(r"(.+)\s+\((Q\d+)\)$", node_raw)
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
    sub_out_dir: Optional[Path],
    column_map: Dict[str, str],
    entity_normalization: str,
    undirected_traversal: bool,
    train_filter: SplitFilter,
    eval_filter: SplitFilter,
    override_filters: Dict[str, SplitFilter],
    write_sub_if_filtered: bool,
) -> None:
    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb)
    relation_vocab = RelationVocab(kb=kb)

    available_files = {p.name for p in raw_root.glob("*.parquet")}
    splits = sorted({name.split("-")[0] for name in available_files})
    connectivity_cache: Dict[Tuple[str, str], bool] = {}
    total_by_split: Dict[str, int] = {}
    kept_by_split: Dict[str, int] = {}
    empty_graph_by_split: Dict[str, int] = {}
    empty_graph_ids: List[str] = []
    empty_graph_id_set: Set[str] = set()

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
        if _should_keep_sample(sample, split_filter, undirected_traversal, connectivity_cache):
            kept_by_split[sample.split] = kept_by_split.get(sample.split, 0) + 1

    entity_vocab.finalize()

    filtered_any = any(kept_by_split.get(s, 0) != total_by_split.get(s, 0) for s in total_by_split)
    write_sub = bool(write_sub_if_filtered) and bool(sub_out_dir) and filtered_any
    if write_sub:
        ensure_dir(sub_out_dir)

    def _format_counts(counts: Dict[str, int]) -> str:
        return ", ".join(f"{s}={counts.get(s, 0)}" for s in splits)

    print(f"Samples total: {_format_counts(total_by_split)}")
    print(f"Samples kept : {_format_counts(kept_by_split)}")
    if empty_graph_by_split:
        print(f"Samples dropped (empty graph): {_format_counts(empty_graph_by_split)}")
        if empty_graph_ids:
            print(f"Empty-graph examples: {empty_graph_ids}")
    print(f"Write sub dataset: {write_sub}")

    # Pass 2: Build graphs and questions
    print("Pass 2: Building graphs and questions...")
    chunk_size = 2000
    base_writer = ParquetDatasetWriter(out_dir=out_dir)
    sub_writer = ParquetDatasetWriter(out_dir=sub_out_dir) if write_sub and sub_out_dir is not None else None

    for sample in tqdm(
        iter_samples(dataset, kb, raw_root, splits, column_map, entity_normalization),
        desc=f"Pass 2/2: Graphs from {dataset}",
    ):
        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        if graph_id in empty_graph_id_set:
            continue
        split_filter = _resolve_split_filter(sample.split, train_filter, eval_filter, override_filters)
        keep_for_sub = _should_keep_sample(sample, split_filter, undirected_traversal, connectivity_cache) if write_sub else False

        use_sp = split_filter.use_shortest_path_positive
        graph = build_graph(sample, entity_vocab, relation_vocab, graph_id, use_sp, undirected_traversal)
        question = build_question_record(sample, entity_vocab, graph_id, use_sp)
        base_writer.append(graph, question)
        if sub_writer is not None and keep_for_sub:
            sub_writer.append(graph, question)

        if len(base_writer.graphs) >= chunk_size or len(base_writer.questions) >= chunk_size:
            base_writer.flush()
        if sub_writer is not None and (len(sub_writer.graphs) >= chunk_size or len(sub_writer.questions) >= chunk_size):
            sub_writer.flush()

    base_writer.close()
    if sub_writer is not None:
        sub_writer.close()

    write_entity_vocab(entity_vocab.struct_records, out_dir / "entity_vocab.parquet")
    write_embedding_vocab(entity_vocab.embedding_records, out_dir / "embedding_vocab.parquet")
    write_relation_vocab(relation_vocab.records, out_dir / "relation_vocab.parquet")
    if write_sub and sub_out_dir is not None:
        write_entity_vocab(entity_vocab.struct_records, sub_out_dir / "entity_vocab.parquet")
        write_embedding_vocab(entity_vocab.embedding_records, sub_out_dir / "embedding_vocab.parquet")
        write_relation_vocab(relation_vocab.records, sub_out_dir / "relation_vocab.parquet")


def build_graph(
    sample: Sample,
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    graph_id: str,
    use_shortest_path_positive: bool,
    undirected_traversal: bool,
) -> GraphRecord:
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

    if answer_edge_indices:
        # Deduplicate while preserving order for determinism
        seen = set()
        dedup_answer_edges = []
        for idx in answer_edge_indices:
            if idx not in seen:
                seen.add(idx)
                dedup_answer_edges.append(idx)
        path_edge_indices = dedup_answer_edges
        gt_source = "answer_subgraph"
        node_set = set()
        for idx in path_edge_indices:
            node_set.add(edge_src[idx])
            node_set.add(edge_dst[idx])
        path_node_indices = sorted(node_set)
    else:
        # Fallback: shortest-path labelling or dense positives.
        path_edge_indices, path_node_indices = shortest_path_edge_indices_directed(
            num_nodes=len(node_entity_ids),
            edge_src=edge_src,
            edge_dst=edge_dst,
            seeds=q_local,
            answers=a_local,
            undirected=undirected_traversal,
        )
        gt_source = "shortest_path" if path_edge_indices else "none"

    num_edges = len(edge_src)
    positive_triple_set = {int(idx) for idx in path_edge_indices if 0 <= int(idx) < num_edges}
    positive_triple_mask = [i in positive_triple_set for i in range(num_edges)]

    if use_shortest_path_positive:
        # Transition-level supervision: lift GT transitions (u->v) to all parallel edges.
        positive_pairs = {(edge_src[idx], edge_dst[idx]) for idx in positive_triple_set}
        positive_edge_mask = [(u, v) in positive_pairs for u, v in zip(edge_src, edge_dst)]
    else:
        positive_edge_mask = [True] * num_edges

    return GraphRecord(
        graph_id=graph_id,
        node_entity_ids=node_entity_ids,
        node_embedding_ids=node_embedding_ids,
        node_labels=node_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_relation_ids=edge_relation_ids,
        positive_edge_mask=positive_edge_mask,
        positive_triple_mask=positive_triple_mask,
        gt_path_edge_indices=path_edge_indices,
        gt_path_node_indices=path_node_indices,
        gt_source=gt_source,
    )


def build_question_record(
    sample: Sample,
    entity_vocab: EntityVocab,
    graph_id: str,
    use_shortest_path_positive: bool,
) -> Dict[str, object]:
    seed_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.q_entity]
    answer_entity_ids = [entity_vocab.entity_id(ent) for ent in sample.a_entity]
    seed_embedding_ids = [entity_vocab.embedding_id(ent) for ent in sample.q_entity]
    answer_embedding_ids = [entity_vocab.embedding_id(ent) for ent in sample.a_entity]
    metadata: Dict[str, Any] = {"use_shortest_path_positive": use_shortest_path_positive}
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
            "positive_edge_mask": pa.array([g.positive_edge_mask for g in graphs], type=pa.list_(pa.bool_())),
            "positive_triple_mask": pa.array([g.positive_triple_mask for g in graphs], type=pa.list_(pa.bool_())),
            "gt_path_edge_indices": pa.array([g.gt_path_edge_indices for g in graphs], type=pa.list_(pa.int64())),
            "gt_path_node_indices": pa.array([g.gt_path_node_indices for g in graphs], type=pa.list_(pa.int64())),
            "gt_source": pa.array([g.gt_source for g in graphs], type=pa.string()),
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
        sub_out_dir = None
        if cfg.get("sub") and bool(cfg.sub.get("enabled", True)):
            sub_out_dir = Path(hydra.utils.to_absolute_path(cfg.sub.out_dir))
        dataset_name = cfg.get("dataset_name") or cfg.get("dataset") or "dataset"

        def _as_filter(section: DictConfig) -> SplitFilter:
            return SplitFilter(
                skip_no_topic=bool(section.get("skip_no_topic", False)),
                skip_no_ans=bool(section.get("skip_no_ans", False)),
                skip_no_path=bool(section.get("skip_no_path", False)),
                use_shortest_path_positive=bool(section.get("use_shortest_path_positive", False)),
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
            sub_out_dir=sub_out_dir,
            column_map=dict(cfg.column_map),
            entity_normalization=cfg.entity_normalization,
            undirected_traversal=cfg.undirected_traversal,
            train_filter=train_filter,
            eval_filter=eval_filter,
            override_filters=override_filters,
            write_sub_if_filtered=bool(cfg.get("sub", {}).get("enabled", True)),
        )

else:  # pragma: no cover

    def main(cfg: DictConfig) -> None:
        raise ModuleNotFoundError("hydra-core is required to run scripts/build_retrieval_parquet.py")


if __name__ == "__main__":
    main()
