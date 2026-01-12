#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ZERO = 0
_ONE = 1
_TWO = 2
_NEG_ONE = -1
_ZERO_FLOAT = 0.0

_HEAD_DEFAULT = 0
_REL_DEFAULT = 1
_TAIL_DEFAULT = 2
_TRIPLE_MIN_LEN = 3

_DIST_UNREACHABLE = _NEG_ONE

_PATH_MODE_Q_TO_A = "q_to_a"
_PATH_MODE_SUBGRAPH_RAG = "subgraph_rag"
_PATH_MODES = (_PATH_MODE_Q_TO_A, _PATH_MODE_SUBGRAPH_RAG)

_DATASETS_DIRNAME = "datasets"
_DATASET_INFO_FILENAME = "dataset_info.json"

_HF_ROOT_DEFAULT = Path("/mnt/data/huggingface") / _DATASETS_DIRNAME
_OUTPUT_DIR_DEFAULT = Path("outputs")
_OUTPUT_PREFIX = "hf_gsub_stats"

_PROGRESS_INTERVAL_DEFAULT = 5000
_PROGRESS_DISABLED = 0

_RECURSION_FACTOR = 4
_RECURSION_MIN = 10000

_CVT_PREFIXES = ("m.", "g.")

_ANS_BUCKET_EQ_1 = 1
_ANS_BUCKET_2_MIN = 2
_ANS_BUCKET_2_MAX = 4
_ANS_BUCKET_3_MIN = 5
_ANS_BUCKET_3_MAX = 9
_ANS_BUCKET_4_MIN = 10

_HOP_BUCKET_MIN = 1
_HOP_BUCKET_MAX = 5

_COUNTS_KEYS = (
    "total_samples",
    "missing_start",
    "missing_answer",
    "start_not_in_graph",
    "answer_not_in_graph",
    "bfs_attempted",
    "no_path",
    "one_hop",
    "edge_disjoint_attempted",
)

_TOTAL_KEYS = (
    "total_nodes",
    "total_edges",
    "total_text_entities",
    "total_cvt_entities",
)

_UNIQUE_KEYS = (
    "unique_entities",
    "unique_relations",
    "unique_node_types",
    "unique_text_entities",
    "unique_cvt_entities",
)

_METRIC_KEYS = (
    "node_count",
    "node_type_count",
    "edge_count",
    "edge_type_count",
    "text_entity_count",
    "cvt_entity_count",
    "start_count",
    "answer_count",
    "connected_pair_count",
    "start_out_degree_mean",
    "start_in_degree_mean",
    "answer_in_degree_mean",
    "gold_edge_ratio",
    "dead_end_rate",
    "edge_disjoint_paths",
)

_ANSWER_BUCKET_KEYS = (
    "ans_eq_0",
    "ans_eq_1",
    "ans_2_4",
    "ans_5_9",
    "ans_ge_10",
)

_QUESTION_HOP_BUCKET_KEYS = (
    "hop_0",
    "hop_1",
    "hop_2",
    "hop_3",
    "hop_4",
    "hop_5",
    "hop_gt5",
    "hop_unreachable",
)


@dataclass
class SummaryStats:
    count: int = _ZERO
    total: float = _ZERO_FLOAT
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def update(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.count += _ONE
        self.total += float(value)
        if self.min_val is None or value < self.min_val:
            self.min_val = float(value)
        if self.max_val is None or value > self.max_val:
            self.max_val = float(value)

    def to_dict(self) -> Dict[str, Optional[float]]:
        mean = None if self.count == _ZERO else self.total / self.count
        return {
            "count": self.count,
            "mean": mean,
            "min": self.min_val,
            "max": self.max_val,
        }


@dataclass
class GraphData:
    node_to_idx: Dict[Any, int]
    adjacency: List[List[int]]
    reverse_adj: List[List[int]]
    indegree: List[int]
    edges: List[Tuple[int, int]]
    edge_types: List[Any]


def _is_text_entity(entity: Any) -> bool:
    return not str(entity).startswith(_CVT_PREFIXES)


def _build_text_entity_classifier() -> callable:
    return _is_text_entity


def _count_text_cvt_entities(
    nodes: Iterable[Any],
    classifier: Optional[callable],
    entity_sets: Optional[Sequence[set]] = None,
    text_sets: Optional[Sequence[set]] = None,
    cvt_sets: Optional[Sequence[set]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if classifier is None:
        if entity_sets:
            node_iter = nodes
            if len(entity_sets) > _ONE:
                node_iter = list(nodes)
            for entity_set in entity_sets:
                entity_set.update(node_iter)
        return None, None
    text_count = _ZERO
    total = _ZERO
    for node in nodes:
        total += _ONE
        if entity_sets:
            for entity_set in entity_sets:
                entity_set.add(node)
        if classifier(node):
            text_count += _ONE
            if text_sets:
                for text_set in text_sets:
                    text_set.add(node)
        elif cvt_sets:
            for cvt_set in cvt_sets:
                cvt_set.add(node)
    return text_count, total - text_count


def _unique_entity_sets(
    unique_sets: Optional[Sequence[Dict[str, set]]],
) -> Tuple[List[set], List[set], List[set]]:
    if not unique_sets:
        return [], [], []
    return (
        [unique["unique_entities"] for unique in unique_sets],
        [unique["unique_text_entities"] for unique in unique_sets],
        [unique["unique_cvt_entities"] for unique in unique_sets],
    )


def _update_unique_relations(
    unique_sets: Optional[Sequence[Dict[str, set]]], relation_ids: Sequence[Any]
) -> None:
    if not unique_sets:
        return
    if len(unique_sets) == _ONE:
        unique_sets[_ZERO]["unique_relations"].update(relation_ids)
        return
    relations = list(relation_ids)
    for unique in unique_sets:
        unique["unique_relations"].update(relations)


def _update_unique_node_types(
    unique_sets: Optional[Sequence[Dict[str, set]]],
    node_type_values: Optional[Sequence[Any]],
) -> None:
    if not unique_sets or not node_type_values:
        return
    if len(unique_sets) == _ONE:
        unique_sets[_ZERO]["unique_node_types"].update(node_type_values)
        return
    values = list(node_type_values)
    for unique in unique_sets:
        unique["unique_node_types"].update(values)


def _update_unique_text_cvt(
    graph_data: GraphData,
    edge_type_values: Sequence[Any],
    text_classifier: Optional[callable],
    unique_sets: Optional[Sequence[Dict[str, set]]],
) -> Tuple[Optional[int], Optional[int]]:
    _update_unique_relations(unique_sets, edge_type_values)
    entity_sets, text_sets, cvt_sets = _unique_entity_sets(unique_sets)
    return _count_text_cvt_entities(
        graph_data.node_to_idx.keys(),
        text_classifier,
        entity_sets,
        text_sets,
        cvt_sets,
    )


def _import_pyarrow_ipc():
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read HF arrow files.") from exc
    return ipc


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _collapse_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def _default_output_path(dataset: str, config: Optional[str]) -> Path:
    name = _slugify(dataset)
    if config:
        name = f"{name}__{_slugify(config)}"
    filename = f"{_OUTPUT_PREFIX}_{name}.json"
    return _OUTPUT_DIR_DEFAULT / filename


def _match_dataset_info(
    info: Dict[str, Any], dataset_name: str, config_name: Optional[str]
) -> bool:
    dataset_slug = dataset_name.split("/")[-_ONE]
    collapsed = _collapse_key(dataset_slug)
    info_candidates = [
        info.get("dataset_name", ""),
        info.get("builder_name", ""),
    ]
    if config_name:
        info_candidates.append(info.get("config_name", ""))
    if not any(_collapse_key(candidate) == collapsed for candidate in info_candidates):
        return False
    if config_name is None:
        return True
    return _collapse_key(info.get("config_name", "")) == _collapse_key(config_name)


def _iter_dataset_info_files(hf_root: Path) -> Iterable[Path]:
    yield from hf_root.rglob(_DATASET_INFO_FILENAME)


def _select_dataset_root(
    candidates: List[Tuple[float, Path, Dict[str, Any]]]
) -> Tuple[Path, Dict[str, Any]]:
    if not candidates:
        raise FileNotFoundError("No matching dataset_info.json found in cache.")
    candidates.sort(key=lambda item: item[_ZERO], reverse=True)
    _, root, info = candidates[_ZERO]
    return root, info


def _find_dataset_root(
    hf_root: Path, dataset_name: str, config_name: Optional[str]
) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    for info_path in _iter_dataset_info_files(hf_root):
        info = _load_json(info_path)
        if _match_dataset_info(info, dataset_name, config_name):
            candidates.append((info_path.stat().st_mtime, info_path.parent, info))
    return _select_dataset_root(candidates)


def _split_names_from_info(info: Dict[str, Any]) -> List[str]:
    splits = info.get("splits", {})
    return list(splits.keys()) if splits else []


def _assign_split(path: Path, split_names: Sequence[str]) -> Optional[str]:
    name = path.name
    for split in split_names:
        token = f"-{split}-"
        if token in name:
            return split
    return None


def _collect_arrow_files(
    dataset_root: Path, split_names: Sequence[str]
) -> Dict[str, List[Path]]:
    files = sorted(dataset_root.glob("*.arrow"))
    by_split: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        if path.name.startswith("cache-"):
            continue
        split = _assign_split(path, split_names)
        if split:
            by_split[split].append(path)
    if not by_split:
        raise FileNotFoundError(f"No split arrow files found under {dataset_root}")
    return by_split


def _ensure_columns(schema, required: Sequence[str]) -> None:
    available = set(schema.names)
    missing = [name for name in required if name is not None and name not in available]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")


def _select_optional_columns(schema, optional: Sequence[Optional[str]]) -> List[str]:
    available = set(schema.names)
    chosen: List[str] = []
    for name in optional:
        if name and name in available:
            chosen.append(name)
    return chosen


def _iter_batches(
    files_by_split: Dict[str, List[Path]], columns: Sequence[str]
) -> Iterable[Tuple[str, Dict[str, list]]]:
    ipc = _import_pyarrow_ipc()
    filtered_columns = [name for name in columns if name is not None]
    for split, files in sorted(files_by_split.items()):
        for path in files:
            with path.open("rb") as handle:
                reader = ipc.open_stream(handle)
                _ensure_columns(reader.schema, filtered_columns)
                for batch in reader:
                    trimmed = batch.select(filtered_columns)
                    yield split, trimmed.to_pydict()


def _normalize_list(value: Optional[Sequence[Any]]) -> List[Any]:
    if value is None:
        return []
    return list(value)


def _get_node_idx(
    node: Any,
    node_to_idx: Dict[Any, int],
    adjacency: List[List[int]],
    reverse_adj: List[List[int]],
    indegree: List[int],
) -> int:
    idx = node_to_idx.get(node)
    if idx is not None:
        return idx
    idx = len(adjacency)
    node_to_idx[node] = idx
    adjacency.append([])
    reverse_adj.append([])
    indegree.append(_ZERO)
    return idx


def _build_graph(
    triples: Sequence[Sequence[Any]],
    head_idx: int,
    rel_idx: int,
    tail_idx: int,
) -> GraphData:
    node_to_idx: Dict[Any, int] = {}
    adjacency: List[List[int]] = []
    reverse_adj: List[List[int]] = []
    indegree: List[int] = []
    edges: List[Tuple[int, int]] = []
    edge_types: List[Any] = []
    for triple in triples:
        if not triple or len(triple) < _TRIPLE_MIN_LEN:
            continue
        if head_idx >= len(triple) or rel_idx >= len(triple) or tail_idx >= len(triple):
            continue
        head = triple[head_idx]
        rel = triple[rel_idx]
        tail = triple[tail_idx]
        u = _get_node_idx(head, node_to_idx, adjacency, reverse_adj, indegree)
        v = _get_node_idx(tail, node_to_idx, adjacency, reverse_adj, indegree)
        adjacency[u].append(v)
        reverse_adj[v].append(u)
        indegree[v] += _ONE
        edges.append((u, v))
        if rel is not None:
            edge_types.append(rel)
    return GraphData(
        node_to_idx=node_to_idx,
        adjacency=adjacency,
        reverse_adj=reverse_adj,
        indegree=indegree,
        edges=edges,
        edge_types=edge_types,
    )


def _unique_count(values: Optional[Sequence[Any]]) -> Optional[int]:
    if values is None:
        return None
    filtered = [value for value in values if value is not None]
    return len(set(filtered))


def _map_entities(
    entities: Sequence[Any], node_to_idx: Dict[Any, int]
) -> List[int]:
    return [node_to_idx[ent] for ent in entities if ent in node_to_idx]


def _degree_stats(
    values: Sequence[int],
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    if not values:
        return None, None, None
    total = sum(values)
    return total / len(values), min(values), max(values)


def _start_out_degrees(adjacency: List[List[int]], starts: Sequence[int]) -> List[int]:
    return [len(adjacency[idx]) for idx in starts]


def _start_in_degrees(indegree: Sequence[int], starts: Sequence[int]) -> List[int]:
    return [indegree[idx] for idx in starts]


def _answer_in_degrees(indegree: Sequence[int], answers: Sequence[int]) -> List[int]:
    return [indegree[idx] for idx in answers]


def _presence_flags(
    start_list: Sequence[Any],
    answer_list: Sequence[Any],
    start_nodes: Sequence[int],
    answer_nodes: Sequence[int],
) -> Dict[str, bool]:
    start_count = len(start_list)
    answer_count = len(answer_list)
    return {
        "missing_start": start_count == _ZERO,
        "missing_answer": answer_count == _ZERO,
        "start_not_in_graph": start_count > _ZERO and not start_nodes,
        "answer_not_in_graph": answer_count > _ZERO and not answer_nodes,
    }


def _degree_metrics(
    graph_data: GraphData, start_nodes: Sequence[int], answer_nodes: Sequence[int]
) -> Dict[str, Optional[float]]:
    start_out = _start_out_degrees(graph_data.adjacency, start_nodes)
    start_in = _start_in_degrees(graph_data.indegree, start_nodes)
    answer_in = _answer_in_degrees(graph_data.indegree, answer_nodes)
    start_out_mean, _, _ = _degree_stats(start_out)
    start_in_mean, _, _ = _degree_stats(start_in)
    answer_in_mean, _, _ = _degree_stats(answer_in)
    return {
        "start_out_mean": start_out_mean,
        "start_in_mean": start_in_mean,
        "answer_in_mean": answer_in_mean,
    }


def _bfs_reachable(adjacency: List[List[int]], sources: Sequence[int]) -> List[bool]:
    visited = [False] * len(adjacency)
    queue = deque()
    for src in sources:
        if not visited[src]:
            visited[src] = True
            queue.append(src)
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    return visited


def _bfs_distances(adjacency: List[List[int]], sources: Sequence[int]) -> List[int]:
    distances = [_DIST_UNREACHABLE] * len(adjacency)
    queue = deque()
    for src in sources:
        if distances[src] == _DIST_UNREACHABLE:
            distances[src] = _ZERO
            queue.append(src)
    while queue:
        node = queue.popleft()
        next_dist = distances[node] + _ONE
        for nxt in adjacency[node]:
            if distances[nxt] == _DIST_UNREACHABLE:
                distances[nxt] = next_dist
                queue.append(nxt)
    return distances


def _unique_ints(values: Sequence[int]) -> List[int]:
    return sorted({int(v) for v in values})


def _min_reachable_distance(forward: int, backward: int) -> Optional[int]:
    if forward == _DIST_UNREACHABLE and backward == _DIST_UNREACHABLE:
        return None
    if forward == _DIST_UNREACHABLE:
        return backward
    if backward == _DIST_UNREACHABLE:
        return forward
    return min(forward, backward)


def _pairwise_shortest_lengths(
    adjacency: List[List[int]], start_nodes: Sequence[int], answer_nodes: Sequence[int]
) -> List[int]:
    if not start_nodes or not answer_nodes:
        return []
    starts = _unique_ints(start_nodes)
    answers = _unique_ints(answer_nodes)
    start_dists = {s: _bfs_distances(adjacency, [s]) for s in starts}
    answer_dists = {a: _bfs_distances(adjacency, [a]) for a in answers}
    lengths: List[int] = []
    for s in starts:
        dist_from_s = start_dists[s]
        for a in answers:
            dist_from_a = answer_dists[a]
            dist = _min_reachable_distance(dist_from_s[a], dist_from_a[s])
            if dist is not None:
                lengths.append(dist)
    return lengths


def _subgraph_rag_distance(
    adjacency: List[List[int]], start_nodes: Sequence[int], answer_nodes: Sequence[int]
) -> Tuple[Optional[int], bool]:
    lengths = _pairwise_shortest_lengths(adjacency, start_nodes, answer_nodes)
    if not lengths:
        return _DIST_UNREACHABLE, False
    return max(lengths), _ONE in lengths


def _reachable_target_count(
    adjacency: List[List[int]], source: int, targets: Sequence[int]
) -> int:
    target_set = set(targets)
    found = _ONE if source in target_set else _ZERO
    if found == len(target_set):
        return found
    visited = [False] * len(adjacency)
    visited[source] = True
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if visited[nxt]:
                continue
            if nxt in target_set:
                found += _ONE
                if found == len(target_set):
                    return found
            visited[nxt] = True
            queue.append(nxt)
    return found


def _count_connected_pairs(
    adjacency: List[List[int]], sources: Sequence[int], targets: Sequence[int]
) -> Optional[int]:
    if not sources or not targets:
        return None
    total = _ZERO
    for src in sources:
        total += _reachable_target_count(adjacency, src, targets)
    return total


def _count_gold_edges(
    edges: Sequence[Tuple[int, int]],
    reach_from_start: Sequence[bool],
    reach_to_answer: Sequence[bool],
) -> int:
    count = _ZERO
    for src, dst in edges:
        if reach_from_start[src] and reach_to_answer[dst]:
            count += _ONE
    return count


def _dead_end_rate(
    reach_from_start: Sequence[bool], reach_to_answer: Sequence[bool]
) -> Optional[float]:
    if not reach_from_start:
        return None
    dead = _ZERO
    for flag_start, flag_answer in zip(reach_from_start, reach_to_answer):
        if not flag_start and not flag_answer:
            dead += _ONE
    return dead / len(reach_from_start)


def _answer_distance_stats(
    adjacency: List[List[int]], start_nodes: Sequence[int], answer_nodes: Sequence[int]
) -> Dict[str, Any]:
    if not start_nodes or not answer_nodes:
        return {
            "shortest_dist": None,
            "max_dist": None,
            "one_hop": False,
            "reach_from_start": None,
        }
    distances = _bfs_distances(adjacency, start_nodes)
    answer_distances = [distances[idx] for idx in answer_nodes]
    reachable = [d for d in answer_distances if d != _DIST_UNREACHABLE]
    if reachable:
        shortest_dist = min(reachable)
        max_dist = max(reachable)
        one_hop = _ONE in reachable
    else:
        shortest_dist = _DIST_UNREACHABLE
        max_dist = None
        one_hop = False
    reach_from_start = [d != _DIST_UNREACHABLE for d in distances]
    return {
        "shortest_dist": shortest_dist,
        "max_dist": max_dist,
        "one_hop": one_hop,
        "reach_from_start": reach_from_start,
    }


def _shortest_distance_mode(
    adjacency: List[List[int]],
    start_nodes: Sequence[int],
    answer_nodes: Sequence[int],
    path_mode: str,
) -> Tuple[Optional[int], bool]:
    if path_mode == _PATH_MODE_SUBGRAPH_RAG:
        return _subgraph_rag_distance(adjacency, start_nodes, answer_nodes)
    dist_stats = _answer_distance_stats(adjacency, start_nodes, answer_nodes)
    return dist_stats["shortest_dist"], dist_stats["one_hop"]


def _connectivity_metrics(
    graph_data: GraphData,
    start_nodes: Sequence[int],
    answer_nodes: Sequence[int],
    path_mode: str,
) -> Dict[str, Any]:
    connected_pairs = _count_connected_pairs(
        graph_data.adjacency, start_nodes, answer_nodes
    )
    shortest_dist = None
    question_hop = None
    one_hop = False
    gold_ratio = None
    dead_rate = None
    if start_nodes and answer_nodes:
        dist_stats = _answer_distance_stats(
            graph_data.adjacency, start_nodes, answer_nodes
        )
        shortest_dist, one_hop = _shortest_distance_mode(
            graph_data.adjacency, start_nodes, answer_nodes, path_mode
        )
        if shortest_dist not in (None, _DIST_UNREACHABLE):
            question_hop = shortest_dist
        reach_from_start = dist_stats["reach_from_start"]
        reach_to_answer = _bfs_reachable(graph_data.reverse_adj, answer_nodes)
        gold_edges = _count_gold_edges(graph_data.edges, reach_from_start, reach_to_answer)
        edge_count = len(graph_data.edges)
        gold_ratio = None if edge_count == _ZERO else gold_edges / edge_count
        dead_rate = _dead_end_rate(reach_from_start, reach_to_answer)
    return {
        "connected_pairs": connected_pairs,
        "shortest_dist": shortest_dist,
        "question_hop": question_hop,
        "one_hop": one_hop,
        "gold_ratio": gold_ratio,
        "dead_rate": dead_rate,
    }


def _edge_type_values(
    edge_types: Optional[Sequence[Any]], graph_edge_types: Sequence[Any]
) -> Sequence[Any]:
    return edge_types if edge_types is not None else graph_edge_types


def _node_type_values(node_types: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    if node_types is None:
        return None
    if isinstance(node_types, (str, bytes)):
        return [node_types]
    flattened: List[Any] = []
    for value in node_types:
        if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _type_entity_stats(
    graph_data: GraphData,
    node_types: Optional[Sequence[Any]],
    edge_types: Optional[Sequence[Any]],
    text_classifier: Optional[callable],
    unique_sets: Optional[Sequence[Dict[str, set]]],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    node_type_values = _node_type_values(node_types)
    node_type_count = _unique_count(node_type_values)
    _update_unique_node_types(unique_sets, node_type_values)
    edge_type_values = _edge_type_values(edge_types, graph_data.edge_types)
    edge_type_count = _unique_count(edge_type_values)
    text_count, cvt_count = _update_unique_text_cvt(
        graph_data, edge_type_values, text_classifier, unique_sets
    )
    return node_type_count, edge_type_count, text_count, cvt_count




def _resolve_entity_lists(
    starts: Sequence[Any], answers: Sequence[Any], node_to_idx: Dict[Any, int]
) -> Tuple[List[Any], List[Any], List[int], List[int]]:
    start_list = _normalize_list(starts)
    answer_list = _normalize_list(answers)
    start_nodes = _map_entities(start_list, node_to_idx)
    answer_nodes = _map_entities(answer_list, node_to_idx)
    return start_list, answer_list, start_nodes, answer_nodes


def _assemble_metrics(
    graph_data: GraphData,
    start_list: Sequence[Any],
    answer_list: Sequence[Any],
    node_type_count: Optional[int],
    edge_type_count: Optional[int],
    text_count: Optional[int],
    cvt_count: Optional[int],
    degrees: Dict[str, Optional[float]],
    connectivity: Dict[str, Any],
    edge_disjoint: Optional[int],
) -> Dict[str, Any]:
    return {
        "node_count": len(graph_data.adjacency),
        "node_type_count": node_type_count,
        "edge_count": len(graph_data.edges),
        "edge_type_count": edge_type_count,
        "text_entity_count": text_count,
        "cvt_entity_count": cvt_count,
        "start_count": len(start_list),
        "answer_count": len(answer_list),
        "connected_pair_count": connectivity["connected_pairs"],
        "start_out_degree_mean": degrees["start_out_mean"],
        "start_in_degree_mean": degrees["start_in_mean"],
        "answer_in_degree_mean": degrees["answer_in_mean"],
        "gold_edge_ratio": connectivity["gold_ratio"],
        "dead_end_rate": connectivity["dead_rate"],
        "edge_disjoint_paths": edge_disjoint,
    }


def _flow_add_edge(graph: List[List[List[int]]], src: int, dst: int, cap: int) -> None:
    graph[src].append([dst, cap, len(graph[dst])])
    graph[dst].append([src, _ZERO, len(graph[src]) - _ONE])


def _flow_build_graph(
    node_count: int,
    edges: Sequence[Tuple[int, int]],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[List[List[int]]], int, int]:
    total_nodes = node_count + _TWO
    source = node_count
    sink = node_count + _ONE
    graph: List[List[List[int]]] = [[] for _ in range(total_nodes)]
    cap_inf = max(len(edges), _ONE)
    for src, dst in edges:
        _flow_add_edge(graph, src, dst, _ONE)
    for src in sources:
        _flow_add_edge(graph, source, src, cap_inf)
    for dst in targets:
        _flow_add_edge(graph, dst, sink, cap_inf)
    return graph, source, sink


def _flow_bfs_level(graph: List[List[List[int]]], source: int, sink: int) -> List[int]:
    level = [_NEG_ONE] * len(graph)
    queue = deque([source])
    level[source] = _ZERO
    while queue:
        node = queue.popleft()
        for nxt, cap, _ in graph[node]:
            if cap > _ZERO and level[nxt] < _ZERO:
                level[nxt] = level[node] + _ONE
                queue.append(nxt)
    return level


def _flow_dfs(
    graph: List[List[List[int]]],
    node: int,
    sink: int,
    flow: int,
    level: Sequence[int],
    iters: List[int],
) -> int:
    if node == sink:
        return flow
    for idx in range(iters[node], len(graph[node])):
        iters[node] = idx
        nxt, cap, rev = graph[node][idx]
        if cap <= _ZERO or level[nxt] != level[node] + _ONE:
            continue
        pushed = _flow_dfs(graph, nxt, sink, min(flow, cap), level, iters)
        if pushed > _ZERO:
            graph[node][idx][_ONE] -= pushed
            graph[nxt][rev][_ONE] += pushed
            return pushed
    return _ZERO


def _max_flow(graph: List[List[List[int]]], source: int, sink: int) -> int:
    flow = _ZERO
    max_cap = sum(edge[_ONE] for edge in graph[source])
    while True:
        level = _flow_bfs_level(graph, source, sink)
        if level[sink] < _ZERO:
            return flow
        iters = [_ZERO] * len(graph)
        while True:
            pushed = _flow_dfs(graph, source, sink, max_cap, level, iters)
            if pushed == _ZERO:
                break
            flow += pushed


def _edge_disjoint_paths(
    node_count: int,
    edges: Sequence[Tuple[int, int]],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Optional[int]:
    if not sources or not targets or not edges:
        return None
    required = max(sys.getrecursionlimit(), node_count * _RECURSION_FACTOR)
    if required > sys.getrecursionlimit():
        sys.setrecursionlimit(max(required, _RECURSION_MIN))
    graph, source, sink = _flow_build_graph(node_count, edges, sources, targets)
    return _max_flow(graph, source, sink)


def _init_counts() -> Dict[str, int]:
    return {key: _ZERO for key in _COUNTS_KEYS}


def _init_metrics() -> Dict[str, SummaryStats]:
    return {key: SummaryStats() for key in _METRIC_KEYS}


def _init_totals() -> Dict[str, int]:
    return {key: _ZERO for key in _TOTAL_KEYS}


def _init_unique_sets() -> Dict[str, set]:
    return {key: set() for key in _UNIQUE_KEYS}


def _init_split_stats(include_edge_disjoint_dist: bool) -> Dict[str, Any]:
    edge_dist = Counter() if include_edge_disjoint_dist else None
    return {
        "counts": _init_counts(),
        "totals": _init_totals(),
        "unique_sets": _init_unique_sets(),
        "metrics": _init_metrics(),
        "answer_count_buckets": Counter(),
        "question_hop_dist": Counter(),
        "question_hop_buckets": Counter(),
        "shortest_path_dist": Counter(),
        "edge_disjoint_dist": edge_dist,
    }


def _bucket_answer_count(answer_count: int) -> str:
    if answer_count == _ZERO:
        return "ans_eq_0"
    if answer_count == _ANS_BUCKET_EQ_1:
        return "ans_eq_1"
    if _ANS_BUCKET_2_MIN <= answer_count <= _ANS_BUCKET_2_MAX:
        return "ans_2_4"
    if _ANS_BUCKET_3_MIN <= answer_count <= _ANS_BUCKET_3_MAX:
        return "ans_5_9"
    if answer_count >= _ANS_BUCKET_4_MIN:
        return "ans_ge_10"
    return "ans_eq_0"


def _bucket_question_hop(hop: Optional[int]) -> str:
    if hop is None:
        return "hop_unreachable"
    if hop == _ZERO:
        return "hop_0"
    if _HOP_BUCKET_MIN <= hop <= _HOP_BUCKET_MAX:
        return f"hop_{hop}"
    return "hop_gt5"


def _update_counts(
    counts: Dict[str, int],
    missing_start: bool,
    missing_answer: bool,
    start_not_in_graph: bool,
    answer_not_in_graph: bool,
) -> None:
    counts["total_samples"] += _ONE
    if missing_start:
        counts["missing_start"] += _ONE
    if missing_answer:
        counts["missing_answer"] += _ONE
    if start_not_in_graph:
        counts["start_not_in_graph"] += _ONE
    if answer_not_in_graph:
        counts["answer_not_in_graph"] += _ONE


def _update_totals(totals: Dict[str, int], metrics: Dict[str, Any]) -> None:
    totals["total_nodes"] += int(metrics["node_count"])
    totals["total_edges"] += int(metrics["edge_count"])
    text_count = metrics.get("text_entity_count")
    cvt_count = metrics.get("cvt_entity_count")
    if text_count is not None:
        totals["total_text_entities"] += int(text_count)
    if cvt_count is not None:
        totals["total_cvt_entities"] += int(cvt_count)


def _update_metric_summaries(metrics: Dict[str, SummaryStats], values: Dict[str, Any]) -> None:
    for key, value in values.items():
        metrics[key].update(value)


def _update_shortest_path(
    stats: Dict[str, Any], distance: Optional[int], one_hop: bool
) -> None:
    if distance is None:
        return
    stats["counts"]["bfs_attempted"] += _ONE
    if distance == _DIST_UNREACHABLE:
        stats["counts"]["no_path"] += _ONE
        return
    stats["shortest_path_dist"][distance] += _ONE
    if one_hop:
        stats["counts"]["one_hop"] += _ONE


def _update_answer_buckets(stats: Dict[str, Any], answer_count: int) -> None:
    bucket = _bucket_answer_count(answer_count)
    stats["answer_count_buckets"][bucket] += _ONE


def _update_question_hops(
    stats: Dict[str, Any], hop: Optional[int], bfs_attempted: bool
) -> None:
    if not bfs_attempted:
        return
    bucket = _bucket_question_hop(hop)
    stats["question_hop_buckets"][bucket] += _ONE
    if hop is not None:
        stats["question_hop_dist"][hop] += _ONE


def _update_edge_disjoint(stats: Dict[str, Any], value: Optional[int]) -> None:
    if value is None:
        return
    stats["counts"]["edge_disjoint_attempted"] += _ONE
    dist = stats.get("edge_disjoint_dist")
    if dist is not None:
        dist[value] += _ONE


def _to_ordered(counter: Counter) -> OrderedDict:
    return OrderedDict(sorted(counter.items(), key=lambda item: item[_ZERO]))


def _finalize_unique(unique_sets: Dict[str, set]) -> Dict[str, int]:
    return {key: len(unique_sets.get(key, set())) for key in _UNIQUE_KEYS}


def _stat_with_totals(
    summary: SummaryStats, total: Optional[int], unique: Optional[int]
) -> Dict[str, Optional[float]]:
    payload = summary.to_dict()
    if total is not None:
        payload["total"] = total
    if unique is not None:
        payload["unique"] = unique
    return payload


def _bucket_payload(counter: Counter, keys: Sequence[str]) -> OrderedDict:
    payload = OrderedDict()
    for key in keys:
        payload[key] = counter.get(key, _ZERO)
    return payload


def _dist_stats(counter: Counter) -> Dict[str, Optional[float]]:
    count = sum(counter.values())
    if count == _ZERO:
        return {"count": _ZERO, "mean": None, "min": None, "max": None}
    total = _ZERO_FLOAT
    for value, freq in counter.items():
        total += float(value) * float(freq)
    return {
        "count": count,
        "mean": total / float(count),
        "min": float(min(counter.keys())),
        "max": float(max(counter.keys())),
    }


def _split_graph_section(
    metrics: Dict[str, SummaryStats], totals: Dict[str, int], unique: Dict[str, int]
) -> Dict[str, Any]:
    node_type_unique = (
        unique["unique_node_types"] if metrics["node_type_count"].count > _ZERO else None
    )
    return {
        "nodes": _stat_with_totals(
            metrics["node_count"], totals["total_nodes"], unique["unique_entities"]
        ),
        "node_types": _stat_with_totals(
            metrics["node_type_count"], None, node_type_unique
        ),
        "edges": _stat_with_totals(
            metrics["edge_count"], totals["total_edges"], None
        ),
        "edge_types": _stat_with_totals(
            metrics["edge_type_count"], None, unique["unique_relations"]
        ),
        "text_entities": _stat_with_totals(
            metrics["text_entity_count"],
            totals["total_text_entities"],
            unique["unique_text_entities"],
        ),
        "cvt_entities": _stat_with_totals(
            metrics["cvt_entity_count"],
            totals["total_cvt_entities"],
            unique["unique_cvt_entities"],
        ),
        "triples": {"total": totals["total_edges"]},
    }


def _split_start_section(metrics: Dict[str, SummaryStats]) -> Dict[str, Any]:
    return {
        "count": metrics["start_count"].to_dict(),
        "degree_mean": {
            "out": metrics["start_out_degree_mean"].to_dict()["mean"],
            "in": metrics["start_in_degree_mean"].to_dict()["mean"],
        },
    }


def _split_answer_section(
    metrics: Dict[str, SummaryStats], answer_buckets: Counter
) -> Dict[str, Any]:
    return {
        "count": metrics["answer_count"].to_dict(),
        "degree_mean": {"in": metrics["answer_in_degree_mean"].to_dict()["mean"]},
        "count_buckets": _bucket_payload(answer_buckets, _ANSWER_BUCKET_KEYS),
    }


def _split_connectivity_section(
    stats: Dict[str, Any], metrics: Dict[str, SummaryStats]
) -> Dict[str, Any]:
    dist = _to_ordered(stats["shortest_path_dist"])
    return {
        "bfs_attempted": stats["counts"]["bfs_attempted"],
        "no_path": stats["counts"]["no_path"],
        "one_hop": stats["counts"]["one_hop"],
        "connected_pair_count": metrics["connected_pair_count"].to_dict(),
        "shortest_path": {
            "dist": dist,
            "buckets": _bucket_payload(stats["question_hop_buckets"], _QUESTION_HOP_BUCKET_KEYS),
            "stats": _dist_stats(dist),
        },
        "edge_disjoint_paths": metrics["edge_disjoint_paths"].to_dict(),
        "edge_disjoint_attempted": stats["counts"]["edge_disjoint_attempted"],
    }


def _finalize_split(stats: Dict[str, Any]) -> Dict[str, Any]:
    totals = dict(stats["totals"])
    totals["total_triples"] = totals["total_edges"]
    unique = _finalize_unique(stats["unique_sets"])
    totals["total_relation_types"] = unique["unique_relations"]
    metrics = stats["metrics"]
    return {
        "samples": {
            "total": stats["counts"]["total_samples"],
            "missing_start": stats["counts"]["missing_start"],
            "missing_answer": stats["counts"]["missing_answer"],
            "start_not_in_graph": stats["counts"]["start_not_in_graph"],
            "answer_not_in_graph": stats["counts"]["answer_not_in_graph"],
        },
        "graph": _split_graph_section(metrics, totals, unique),
        "start": _split_start_section(metrics),
        "answer": _split_answer_section(metrics, stats["answer_count_buckets"]),
        "connectivity": _split_connectivity_section(stats, metrics),
        "signal_noise": {
            "gold_edge_ratio": metrics["gold_edge_ratio"].to_dict(),
            "dead_end_rate": metrics["dead_end_rate"].to_dict(),
        },
    }


def _finalize_summary(
    stats_by_split: Dict[str, Any], overall: Dict[str, Any]
) -> Dict[str, Any]:
    overall_totals = dict(overall["totals"])
    overall_totals["total_triples"] = overall_totals["total_edges"]
    unique = _finalize_unique(overall["unique_sets"])
    overall_totals["total_relation_types"] = unique["unique_relations"]
    node_type_unique = (
        unique["unique_node_types"]
        if overall["metrics"]["node_type_count"].count > _ZERO
        else None
    )
    samples_by_split = OrderedDict(
        (split, stats["counts"]["total_samples"]) for split, stats in stats_by_split.items()
    )
    return {
        "samples": {
            "total": overall["counts"]["total_samples"],
            "by_split": samples_by_split,
        },
        "graph": {
            "nodes": {
                "total": overall_totals["total_nodes"],
                "unique": unique["unique_entities"],
            },
            "node_types": {"unique": node_type_unique},
            "edges": {"total": overall_totals["total_edges"]},
            "edge_types": {"unique": unique["unique_relations"]},
            "text_entities": {
                "total": overall_totals["total_text_entities"],
                "unique": unique["unique_text_entities"],
            },
            "cvt_entities": {
                "total": overall_totals["total_cvt_entities"],
                "unique": unique["unique_cvt_entities"],
            },
            "triples": {"total": overall_totals["total_edges"]},
        },
    }


def _should_log(progress_interval: int, processed: int) -> bool:
    if progress_interval == _PROGRESS_DISABLED:
        return False
    return processed % progress_interval == _ZERO


def _compute_sample_metrics(
    graph: Sequence[Sequence[Any]],
    starts: Sequence[Any],
    answers: Sequence[Any],
    node_types: Optional[Sequence[Any]],
    edge_types: Optional[Sequence[Any]],
    text_classifier: Optional[callable],
    unique_sets: Optional[Sequence[Dict[str, set]]],
    path_mode: str,
    head_idx: int,
    rel_idx: int,
    tail_idx: int,
    compute_edge_disjoint: bool,
) -> Tuple[Dict[str, Any], Dict[str, bool], Optional[int], bool, Optional[int], Optional[int]]:
    graph_data = _build_graph(graph, head_idx, rel_idx, tail_idx)
    start_list, answer_list, start_nodes, answer_nodes = _resolve_entity_lists(
        starts, answers, graph_data.node_to_idx
    )
    flags = _presence_flags(start_list, answer_list, start_nodes, answer_nodes)
    degrees = _degree_metrics(graph_data, start_nodes, answer_nodes)
    connectivity = _connectivity_metrics(
        graph_data, start_nodes, answer_nodes, path_mode
    )
    node_type_count, edge_type_count, text_count, cvt_count = _type_entity_stats(
        graph_data, node_types, edge_types, text_classifier, unique_sets
    )
    edge_disjoint = _edge_disjoint_paths(
        len(graph_data.adjacency), graph_data.edges, start_nodes, answer_nodes
    ) if compute_edge_disjoint else None
    metrics = _assemble_metrics(
        graph_data,
        start_list,
        answer_list,
        node_type_count,
        edge_type_count,
        text_count,
        cvt_count,
        degrees,
        connectivity,
        edge_disjoint,
    )
    return (
        metrics,
        flags,
        connectivity["shortest_dist"],
        connectivity["one_hop"],
        edge_disjoint,
        connectivity["question_hop"],
    )


def _resolve_hf_root(value: Optional[str]) -> Path:
    if value:
        return Path(value)
    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        return Path(env_cache)
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home) / _DATASETS_DIRNAME
    return _HF_ROOT_DEFAULT


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name.")
    parser.add_argument("--config", type=str, default=None, help="HF dataset config.")
    parser.add_argument("--hf-root", type=str, default=None, help="HF cache root.")


def _add_field_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-field", type=str, default="graph", help="Graph field.")
    parser.add_argument("--start-field", type=str, default="q_entity", help="Start field.")
    parser.add_argument("--answer-field", type=str, default="a_entity", help="Answer field.")
    parser.add_argument("--node-type-field", type=str, default=None, help="Node type field.")
    parser.add_argument("--edge-type-field", type=str, default=None, help="Edge type field.")


def _add_graph_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--head-index", type=int, default=_HEAD_DEFAULT, help="Head index.")
    parser.add_argument("--rel-index", type=int, default=_REL_DEFAULT, help="Rel index.")
    parser.add_argument("--tail-index", type=int, default=_TAIL_DEFAULT, help="Tail index.")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=_PROGRESS_INTERVAL_DEFAULT,
        help="Progress logging interval (0 disables).",
    )
    parser.add_argument(
        "--shortest-path-mode",
        choices=_PATH_MODES,
        default=_PATH_MODE_Q_TO_A,
        help="Shortest path definition (q_to_a or subgraph_rag).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Sample cap.")


def _add_edge_disjoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-edge-disjoint",
        action="store_true",
        help="Skip edge-disjoint path computation.",
    )
    parser.add_argument(
        "--edge-disjoint-dist-path",
        type=str,
        default=None,
        help="Optional output path for edge-disjoint distribution JSON.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path (default uses dataset name).",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF G_sub stats (dataset-agnostic).")
    _add_dataset_args(parser)
    _add_field_args(parser)
    _add_graph_index_args(parser)
    _add_runtime_args(parser)
    _add_edge_disjoint_args(parser)
    _add_output_args(parser)
    return parser.parse_args()


def _extract_batch_fields(
    batch: Dict[str, list], args: argparse.Namespace
) -> Tuple[list, list, list, Optional[list], Optional[list]]:
    node_types_list = batch.get(args.node_type_field) if args.node_type_field else None
    edge_types_list = batch.get(args.edge_type_field) if args.edge_type_field else None
    graph_list = batch[args.graph_field]
    starts_list = batch[args.start_field]
    answers_list = batch[args.answer_field]
    return graph_list, starts_list, answers_list, node_types_list, edge_types_list


def _update_stats_for_sample(
    split_stats: Dict[str, Any],
    overall: Dict[str, Any],
    metrics: Dict[str, Any],
    flags: Dict[str, bool],
    dist: Optional[int],
    one_hop: bool,
    question_hop: Optional[int],
    edge_disjoint: Optional[int],
) -> None:
    _update_counts(
        split_stats["counts"],
        flags["missing_start"],
        flags["missing_answer"],
        flags["start_not_in_graph"],
        flags["answer_not_in_graph"],
    )
    _update_counts(
        overall["counts"],
        flags["missing_start"],
        flags["missing_answer"],
        flags["start_not_in_graph"],
        flags["answer_not_in_graph"],
    )
    _update_metric_summaries(split_stats["metrics"], metrics)
    _update_metric_summaries(overall["metrics"], metrics)
    _update_totals(split_stats["totals"], metrics)
    _update_totals(overall["totals"], metrics)
    _update_answer_buckets(split_stats, int(metrics["answer_count"]))
    _update_answer_buckets(overall, int(metrics["answer_count"]))
    bfs_attempted = dist is not None
    _update_question_hops(split_stats, question_hop, bfs_attempted)
    _update_question_hops(overall, question_hop, bfs_attempted)
    _update_shortest_path(split_stats, dist, one_hop)
    _update_shortest_path(overall, dist, one_hop)
    _update_edge_disjoint(split_stats, edge_disjoint)
    _update_edge_disjoint(overall, edge_disjoint)


def _consume_batch(
    split: str,
    batch: Dict[str, list],
    args: argparse.Namespace,
    stats_by_split: Dict[str, Any],
    overall: Dict[str, Any],
    processed: int,
    text_classifier: Optional[callable],
) -> Tuple[int, bool]:
    graph_list, starts_list, answers_list, node_types_list, edge_types_list = _extract_batch_fields(
        batch, args
    )
    split_stats = stats_by_split[split]
    unique_sets = [split_stats["unique_sets"], overall["unique_sets"]]
    for idx in range(len(graph_list)):
        metrics, flags, dist, one_hop, edge_disjoint, question_hop = _compute_sample_metrics(
            graph_list[idx],
            starts_list[idx],
            answers_list[idx],
            node_types_list[idx] if node_types_list else None,
            edge_types_list[idx] if edge_types_list else None,
            text_classifier,
            unique_sets,
            args.shortest_path_mode,
            args.head_index,
            args.rel_index,
            args.tail_index,
            not args.skip_edge_disjoint,
        )
        _update_stats_for_sample(
            split_stats,
            overall,
            metrics,
            flags,
            dist,
            one_hop,
            question_hop,
            edge_disjoint,
        )
        processed += _ONE
        if args.max_samples and processed >= args.max_samples:
            return processed, True
        if _should_log(args.progress_interval, processed):
            print(f"processed={processed}", file=sys.stderr)
    return processed, False


def _process_batches(
    files_by_split: Dict[str, List[Path]],
    args: argparse.Namespace,
    text_classifier: Optional[callable],
    include_edge_disjoint_dist: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    required = [args.graph_field, args.start_field, args.answer_field]
    optional = [args.node_type_field, args.edge_type_field]
    stats_by_split = defaultdict(lambda: _init_split_stats(include_edge_disjoint_dist))
    overall = _init_split_stats(include_edge_disjoint_dist)
    processed = _ZERO
    for split, batch in _iter_batches(files_by_split, required + optional):
        processed, stop = _consume_batch(
            split, batch, args, stats_by_split, overall, processed, text_classifier
        )
        if stop:
            break
    return stats_by_split, overall


def _build_report(
    args: argparse.Namespace,
    dataset_root: Path,
    stats_by_split: Dict[str, Any],
    overall: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "config": args.config,
        "dataset_root": str(dataset_root),
        "fields": {
            "graph_field": args.graph_field,
            "start_field": args.start_field,
            "answer_field": args.answer_field,
            "node_type_field": args.node_type_field,
            "edge_type_field": args.edge_type_field,
            "shortest_path_mode": args.shortest_path_mode,
            "cvt_prefixes": list(_CVT_PREFIXES),
        },
        "summary": _finalize_summary(stats_by_split, overall),
        "splits": {name: _finalize_split(stats) for name, stats in stats_by_split.items()},
    }


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=True, indent=_TWO))


def _write_edge_disjoint_dist(
    path: Path, stats_by_split: Dict[str, Any], overall: Dict[str, Any]
) -> None:
    def _dist_payload(stats: Dict[str, Any]) -> OrderedDict:
        dist = stats.get("edge_disjoint_dist")
        if dist is None:
            return OrderedDict()
        return _to_ordered(dist)

    payload = {
        "overall": _dist_payload(overall),
        "splits": {split: _dist_payload(stats) for split, stats in stats_by_split.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=_TWO))


def main() -> None:
    args = _parse_args()
    hf_root = _resolve_hf_root(args.hf_root)
    dataset_root, info = _find_dataset_root(hf_root, args.dataset, args.config)
    split_names = _split_names_from_info(info)
    if not split_names:
        raise RuntimeError("No splits found in dataset_info.json.")
    files_by_split = _collect_arrow_files(dataset_root, split_names)
    text_classifier = _build_text_entity_classifier()
    include_edge_dist = bool(args.edge_disjoint_dist_path)
    stats_by_split, overall = _process_batches(
        files_by_split, args, text_classifier, include_edge_dist
    )
    report = _build_report(args, dataset_root, stats_by_split, overall)
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = _default_output_path(args.dataset, args.config)
    _write_report(output_path, report)
    if args.edge_disjoint_dist_path:
        _write_edge_disjoint_dist(
            Path(args.edge_disjoint_dist_path), stats_by_split, overall
        )
    print(json.dumps(report["summary"], ensure_ascii=True, indent=_TWO))


if __name__ == "__main__":
    main()
