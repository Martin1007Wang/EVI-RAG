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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import hydra
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm


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
    positive_edge_mask: List[bool]
    gt_path_edge_indices: List[int]
    gt_path_node_indices: List[int]


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
    """Directed BFS to get one shortest path (edge indices + node indices)."""
    if not seeds or not answers or num_nodes == 0:
        return [], []
    from collections import deque

    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for idx, (u, v) in enumerate(zip(edge_src, edge_dst)):
        adjacency[u].append((v, idx))
        if undirected:
            adjacency[v].append((u, idx))

    answer_set = set(answers)
    visited = [-1] * num_nodes
    parent_edge = [-1] * num_nodes
    queue: deque[int] = deque()
    for seed in seeds:
        if 0 <= seed < num_nodes:
            queue.append(seed)
            visited[seed] = seed

    while queue:
        node = queue.popleft()
        if node in answer_set:
            path_edges: List[int] = []
            path_nodes: List[int] = []
            cur = node
            while visited[cur] != cur:
                path_nodes.append(cur)
                e_idx = parent_edge[cur]
                path_edges.append(e_idx)
                cur = visited[cur]
            path_nodes.append(cur)
            path_nodes.reverse()
            path_edges.reverse()
            return path_edges, path_nodes
        for nb, e_idx in adjacency[node]:
            if visited[nb] != -1:
                continue
            visited[nb] = node
            parent_edge[nb] = e_idx
            queue.append(nb)
    return [], []


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
    column_map: Dict[str, str],
    entity_normalization: str,
    undirected_traversal: bool,
    skip_no_topic_train: bool,
    skip_no_ans_train: bool,
    skip_no_path_train: bool,
    use_shortest_path_positive_train: bool,
    skip_no_topic_eval: bool,
    skip_no_ans_eval: bool,
    skip_no_path_eval: bool,
    use_shortest_path_positive_eval: bool,
) -> None:
    ensure_dir(out_dir)
    entity_vocab = EntityVocab(kb=kb)
    relation_vocab = RelationVocab(kb=kb)

    available_files = {p.name for p in raw_root.glob("*.parquet")}
    splits = sorted({name.split("-")[0] for name in available_files})
    connectivity_cache: Dict[Tuple[str, str], bool] = {}

    # Pass 1: Build vocabularies
    print("Pass 1: Building vocabularies...")
    for sample in tqdm(
        iter_samples(dataset, kb, raw_root, splits, column_map, entity_normalization),
        desc=f"Pass 1/2: Vocab from {dataset}",
    ):
        node_strings = {h for h, _, t in sample.graph} | {t for _, _, t in sample.graph}
        has_topic = any(ent in node_strings for ent in sample.q_entity)
        has_answer = any(ent in node_strings for ent in sample.a_entity)
        is_train_split = sample.split == "train"
        use_sp = use_shortest_path_positive_train if is_train_split else use_shortest_path_positive_eval
        skip_no_path = skip_no_path_train if is_train_split else skip_no_path_eval

        has_path = True
        if sample.answer_subgraph:
            has_path = True
        elif skip_no_path or use_sp:
            has_path = has_connectivity(sample.graph, sample.q_entity, sample.a_entity, undirected=undirected_traversal)
        connectivity_cache[(sample.split, sample.question_id)] = has_path

        if is_train_split:
            if (skip_no_topic_train and not has_topic) or (skip_no_ans_train and not has_answer) or (skip_no_path_train and not has_path):
                continue
        else:
            if (skip_no_topic_eval and not has_topic) or (skip_no_ans_eval and not has_answer) or (skip_no_path_eval and not has_path):
                continue

        for h, r, t in sample.graph:
            entity_vocab.add_entity(h)
            entity_vocab.add_entity(t)
            relation_vocab.relation_id(r)
        for ent in sample.q_entity + sample.a_entity:
            entity_vocab.add_entity(ent)

    entity_vocab.finalize()

    # Pass 2: Build graphs and questions
    graphs: List[GraphRecord] = []
    questions: List[Dict[str, object]] = []
    print("Pass 2: Building graphs and questions...")
    graph_writer: pq.ParquetWriter | None = None
    question_writer: pq.ParquetWriter | None = None
    chunk_size = 2000

    def flush_buffers() -> None:
        nonlocal graph_writer, question_writer, graphs, questions
        if graphs:
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
                    "gt_path_edge_indices": pa.array([g.gt_path_edge_indices for g in graphs], type=pa.list_(pa.int64())),
                    "gt_path_node_indices": pa.array([g.gt_path_node_indices for g in graphs], type=pa.list_(pa.int64())),
                }
            )
            if graph_writer is None:
                graph_writer = pq.ParquetWriter(out_dir / "graphs.parquet", table.schema, compression="zstd")
            graph_writer.write_table(table)
            graphs = []
        if questions:
            table_q = pa.table(
                {
                    "question_uid": pa.array([row["question_uid"] for row in questions], type=pa.string()),
                    "dataset": pa.array([row["dataset"] for row in questions], type=pa.string()),
                    "split": pa.array([row["split"] for row in questions], type=pa.string()),
                    "kb": pa.array([row["kb"] for row in questions], type=pa.string()),
                    "question": pa.array([row["question"] for row in questions], type=pa.string()),
                    "paraphrased_question": pa.array([row.get("paraphrased_question") for row in questions], type=pa.string()),
                    "seed_entity_ids": pa.array([row["seed_entity_ids"] for row in questions], type=pa.list_(pa.int64())),
                    "answer_entity_ids": pa.array([row["answer_entity_ids"] for row in questions], type=pa.list_(pa.int64())),
                    "seed_embedding_ids": pa.array([row["seed_embedding_ids"] for row in questions], type=pa.list_(pa.int64())),
                    "answer_embedding_ids": pa.array([row["answer_embedding_ids"] for row in questions], type=pa.list_(pa.int64())),
                    "answer_texts": pa.array([row["answer_texts"] for row in questions], type=pa.list_(pa.string())),
                    "graph_id": pa.array([row["graph_id"] for row in questions], type=pa.string()),
                    "metadata": pa.array([row["metadata"] for row in questions], type=pa.string()),
                }
            )
            if question_writer is None:
                question_writer = pq.ParquetWriter(out_dir / "questions.parquet", table_q.schema, compression="zstd")
            question_writer.write_table(table_q)
            questions = []

    for sample in tqdm(
        iter_samples(dataset, kb, raw_root, splits, column_map, entity_normalization),
        desc=f"Pass 2/2: Graphs from {dataset}",
    ):
        node_strings = {h for h, _, t in sample.graph} | {t for _, _, t in sample.graph}
        has_topic = any(ent in node_strings for ent in sample.q_entity)
        has_answer = any(ent in node_strings for ent in sample.a_entity)
        is_train_split = sample.split == "train"
        use_sp = use_shortest_path_positive_train if is_train_split else use_shortest_path_positive_eval
        skip_no_path = skip_no_path_train if is_train_split else skip_no_path_eval
        has_path = connectivity_cache.get((sample.split, sample.question_id))
        if (skip_no_path or use_sp) and has_path is None:
            if sample.answer_subgraph:
                has_path = True
            else:
                has_path = has_connectivity(sample.graph, sample.q_entity, sample.a_entity, undirected=undirected_traversal)
            connectivity_cache[(sample.split, sample.question_id)] = has_path

        if is_train_split:
            if (skip_no_topic_train and not has_topic) or (skip_no_ans_train and not has_answer) or (skip_no_path_train and not has_path):
                continue
        else:
            if (skip_no_topic_eval and not has_topic) or (skip_no_ans_eval and not has_answer) or (skip_no_path_eval and not has_path):
                continue

        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        graphs.append(build_graph(sample, entity_vocab, relation_vocab, graph_id, use_sp, undirected_traversal))
        questions.append(build_question_record(sample, entity_vocab, graph_id, use_sp))

        if len(graphs) >= chunk_size or len(questions) >= chunk_size:
            flush_buffers()

    flush_buffers()
    if graph_writer is not None:
        graph_writer.close()
    if question_writer is not None:
        question_writer.close()

    write_entity_vocab(entity_vocab.struct_records, out_dir / "entity_vocab.parquet")
    write_embedding_vocab(entity_vocab.embedding_records, out_dir / "embedding_vocab.parquet")
    write_relation_vocab(relation_vocab.records, out_dir / "relation_vocab.parquet")


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
        node_set = set()
        for idx in path_edge_indices:
            node_set.add(edge_src[idx])
            node_set.add(edge_dst[idx])
        path_node_indices = sorted(node_set)
        positive_edge_mask = [False] * len(edge_src)
        for idx in path_edge_indices:
            if 0 <= idx < len(positive_edge_mask):
                positive_edge_mask[idx] = True
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

        if use_shortest_path_positive:
            positive_edge_mask = [False] * len(edge_src)
            for idx in path_edge_indices:
                if 0 <= idx < len(positive_edge_mask):
                    positive_edge_mask[idx] = True
        else:
            positive_edge_mask = [True] * len(edge_src)

    return GraphRecord(
        graph_id=graph_id,
        node_entity_ids=node_entity_ids,
        node_embedding_ids=node_embedding_ids,
        node_labels=node_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_relation_ids=edge_relation_ids,
        positive_edge_mask=positive_edge_mask,
        gt_path_edge_indices=path_edge_indices,
        gt_path_node_indices=path_node_indices,
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
            "gt_path_edge_indices": pa.array([g.gt_path_edge_indices for g in graphs], type=pa.list_(pa.int64())),
            "gt_path_node_indices": pa.array([g.gt_path_node_indices for g in graphs], type=pa.list_(pa.int64())),
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


@hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_parquet")
def main(cfg: DictConfig) -> None:
    raw_root = Path(hydra.utils.to_absolute_path(cfg.raw_root))
    out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
    dataset_name = cfg.get("dataset_name") or cfg.get("dataset") or "dataset"
    preprocess(
        dataset=dataset_name,
        kb=cfg.kb,
        raw_root=raw_root,
        out_dir=out_dir,
        column_map=dict(cfg.column_map),
        entity_normalization=cfg.entity_normalization,
        undirected_traversal=cfg.undirected_traversal,
        skip_no_topic_train=cfg.filter.train.skip_no_topic,
        skip_no_ans_train=cfg.filter.train.skip_no_ans,
        skip_no_path_train=cfg.filter.train.skip_no_path,
        use_shortest_path_positive_train=cfg.filter.train.use_shortest_path_positive,
        skip_no_topic_eval=cfg.filter.eval.skip_no_topic,
        skip_no_ans_eval=cfg.filter.eval.skip_no_ans,
        skip_no_path_eval=cfg.filter.eval.skip_no_path,
        use_shortest_path_positive_eval=cfg.filter.eval.use_shortest_path_positive,
    )


if __name__ == "__main__":
    main()
