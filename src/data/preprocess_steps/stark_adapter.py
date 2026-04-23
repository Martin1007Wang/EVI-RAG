from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch

# 🎯 对齐之前的重构契约
from .samples import RawSample

log = logging.getLogger(__name__)

# --- 核心配置常量 ---
_PRIME_DATASET = "prime"
_SOURCE_SPLITS = {"train": "train", "validation": "val", "test": "test"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_SUPPORTED_LINKER_BACKENDS = {"keyword"}
_SUPPORTED_LOCAL_GRAPH_DIRECTIONS = {"out", "in", "both"}

# ==============================================================================
# 1. 结构化中间表示
# ==============================================================================


@dataclass(frozen=True)
class PrimeNodeRecord:
    node_id: int
    entity_id: str
    name: str
    node_type: str
    source: str
    summary: str
    search_tokens: frozenset[str]


@dataclass(frozen=True)
class LinkerCandidate:
    node_id: int
    score: float
    record: PrimeNodeRecord


@dataclass(frozen=True)
class LocalGraphConfig:
    num_hops: int
    direction: str
    max_nodes: int
    max_edges: int


def _resolve_linker_backend(linker_cfg: Mapping[str, Any]) -> str:
    backend = str(linker_cfg.get("backend", "keyword")).strip().lower()
    if backend not in _SUPPORTED_LINKER_BACKENDS:
        raise ValueError(
            f"Unsupported STARK linker backend {backend!r}; expected one of "
            f"{sorted(_SUPPORTED_LINKER_BACKENDS)}."
        )
    return backend


def _resolve_local_graph_config(
    local_graph_cfg: Mapping[str, Any] | None,
) -> LocalGraphConfig:
    cfg = dict(local_graph_cfg or {})
    num_hops = int(cfg.get("num_hops", 2))
    max_nodes = int(cfg.get("max_nodes", 64))
    max_edges = int(cfg.get("max_edges", 256))
    direction = str(cfg.get("direction", "both")).strip().lower()

    if num_hops < 0:
        raise ValueError(f"stark.local_graph.num_hops must be >= 0, got {num_hops}.")
    if max_nodes < 1:
        raise ValueError(f"stark.local_graph.max_nodes must be >= 1, got {max_nodes}.")
    if max_edges < 1:
        raise ValueError(f"stark.local_graph.max_edges must be >= 1, got {max_edges}.")
    if direction not in _SUPPORTED_LOCAL_GRAPH_DIRECTIONS:
        raise ValueError(
            f"Unsupported stark.local_graph.direction={direction!r}; expected one of "
            f"{sorted(_SUPPORTED_LOCAL_GRAPH_DIRECTIONS)}."
        )

    return LocalGraphConfig(
        num_hops=num_hops,
        direction=direction,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


# ==============================================================================
# 2. 实体链接引擎 (Linker)
# ==============================================================================


class StarkLinker:
    """负责将问题文本映射到图谱节点（锚点选择）"""

    def __init__(
        self, node_records: Dict[int, PrimeNodeRecord], linker_cfg: Mapping[str, Any]
    ):
        self.records = node_records
        self.cfg = linker_cfg
        self.token_to_ids: Dict[str, List[int]] = defaultdict(list)
        self.token_idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self):
        """构建轻量级 TF-IDF 索引"""
        log.info("Building Stark keyword index...")
        num_docs = len(self.records) or 1
        counts = defaultdict(int)
        for r in self.records.values():
            for t in r.search_tokens:
                self.token_to_ids[t].append(r.node_id)
                counts[t] += 1
        for t, c in counts.items():
            self.token_idf[t] = math.log((1.0 + num_docs) / (1.0 + c)) + 1.0

    def retrieve_anchors(self, question: str) -> List[int]:
        """执行链接策略"""
        backend = _resolve_linker_backend(self.cfg)

        # 1. 粗排检索
        tokens = set(_tokenize(question))
        scores = defaultdict(float)
        for t in tokens:
            if t in self.token_to_ids:
                weight = self.token_idf.get(t, 1.0)
                for nid in self.token_to_ids[t]:
                    scores[nid] += weight

        # 2. 排序取 Top-K
        candidates = []
        for nid, s in scores.items():
            r = self.records[nid]
            # 基础重叠分 + 语义分
            overlap_bonus = 0.5 * len(tokens & r.search_tokens)
            candidates.append(LinkerCandidate(nid, s + overlap_bonus, r))

        candidates.sort(key=lambda x: -x.score)
        top_candidates = candidates[: self.cfg.get("max_candidates", 12)]

        # 3. 精排（LLM 路径或启发式路径）
        return [c.node_id for c in top_candidates[: self.cfg.get("max_entities", 3)]]


# ==============================================================================
# 3. 主适配器 (Adapter Orchestrator)
# ==============================================================================


class StarkPrimeAdapter:
    def __init__(self, dataset: str, kb: str, stark_cfg: Mapping[str, Any]):
        self.dataset_id = dataset
        self.kb_id = kb
        self.local_graph_cfg = _resolve_local_graph_config(stark_cfg.get("local_graph"))
        if bool(stark_cfg.get("indirected", False)):
            raise ValueError(
                "dataset.stark.indirected is not supported: PRIME knowledge graphs "
                "must preserve directed relations."
            )

        # 资源加载
        self._qa, self._skb = _load_prime_resources(
            stark_cfg.get("root"),
            stark_cfg.get("download_processed", True),
            False,
        )

        # 节点属性归档
        self.node_records = self._init_node_records()

        # 链接引擎初始化
        self.linker = StarkLinker(self.node_records, stark_cfg.get("linker", {}))

        # 邻接表缓存
        self.adj_out, self.adj_in = self._init_adjacency()

    def iter_samples(self, splits: Sequence[str]) -> Iterator[RawSample]:
        idx_splits = self._qa.get_idx_split()

        for split in splits:
            source_key = _SOURCE_SPLITS.get(split, "train")
            indices = idx_splits.get(source_key, [])

            for idx in indices:
                query, q_id, ans_ids, _ = self._qa[int(idx)]

                # 1. 链接锚点
                anchor_ids = self.linker.retrieve_anchors(query)

                # 2. BFS 漫步提取子图
                graph_triples = self._walk_subgraph(anchor_ids)

                # 3. 封装为 RawSample (名副其实的数据载体)
                yield RawSample(
                    dataset=self.dataset_id,
                    split=split,
                    question_id=str(q_id),
                    kb=self.kb_id,
                    question=str(query),
                    graph=tuple(graph_triples),
                    question_entities=tuple(
                        self.node_records[nid].entity_id for nid in anchor_ids
                    ),
                    answer_entities=tuple(
                        self.node_records[int(aid)].entity_id
                        for aid in ans_ids
                        if int(aid) in self.node_records
                    ),
                    answer_texts=tuple(
                        self.node_records[int(aid)].name
                        for aid in ans_ids
                        if int(aid) in self.node_records
                    ),
                )

    def _init_node_records(self) -> Dict[int, PrimeNodeRecord]:
        """预处理所有节点，提升后续访问速度"""
        records = {}
        for nid in range(self._skb.num_nodes()):
            info = self._skb.node_info[nid]
            name = str(info.get("name", f"node_{nid}"))
            node_type = str(info.get("type", "node"))

            records[nid] = PrimeNodeRecord(
                node_id=nid,
                entity_id=f"prime[{node_type}] {name} <id={nid}>",
                name=name,
                node_type=node_type,
                source=str(info.get("source", "")),
                summary=str(info.get("summary", ""))[:200],
                search_tokens=frozenset(_tokenize(name) + _tokenize(node_type)),
            )
        return records

    def _init_adjacency(
        self,
    ) -> tuple[Dict[int, List[tuple[int, int]]], Dict[int, List[tuple[int, int]]]]:
        """构建高效邻接表"""
        adj_out = defaultdict(list)
        adj_in = defaultdict(list)
        edge_index = self._skb.edge_index
        edge_types = self._skb.edge_types
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            rel = int(edge_types[i])
            adj_out[u].append((v, rel))
            adj_in[v].append((u, rel))
        return adj_out, adj_in

    def _iter_traversal_neighbors(self, node_id: int) -> List[int]:
        direction = self.local_graph_cfg.direction
        neighbor_ids: List[int] = []
        if direction in {"out", "both"}:
            neighbor_ids.extend(v for v, _ in self.adj_out.get(node_id, []))
        if direction in {"in", "both"}:
            neighbor_ids.extend(v for v, _ in self.adj_in.get(node_id, []))
        return neighbor_ids

    def _append_triple(
        self,
        triples: List[tuple[str, str, str]],
        seen: set[tuple[str, str, str]],
        triple: tuple[str, str, str],
    ) -> bool:
        if triple in seen:
            return False
        seen.add(triple)
        triples.append(triple)
        return len(triples) >= self.local_graph_cfg.max_edges

    def _walk_subgraph(self, anchors: List[int]) -> List[tuple[str, str, str]]:
        """执行 BFS 提取局部上下文"""
        queue = deque([(nid, 0) for nid in anchors])
        triples: List[tuple[str, str, str]] = []

        # 1. 节点发现
        node_context = set()
        node_order: List[int] = []
        while queue and len(node_context) < self.local_graph_cfg.max_nodes:
            u, dist = queue.popleft()
            if u in node_context:
                continue
            node_context.add(u)
            node_order.append(u)
            if dist < self.local_graph_cfg.num_hops:
                for v in self._iter_traversal_neighbors(u):
                    if v not in node_context:
                        queue.append((v, dist + 1))

        # 2. 边提取
        seen_triples: set[tuple[str, str, str]] = set()
        for u in node_order:
            u_ent = self.node_records[u].entity_id
            for v, rel_id in self.adj_out.get(u, []):
                if v in node_context:
                    rel_label = str(self._skb.get_edge_type_by_id(rel_id))
                    tail_ent = self.node_records[v].entity_id
                    if self._append_triple(
                        triples, seen_triples, (u_ent, rel_label, tail_ent)
                    ):
                        return triples

        return triples


# ==============================================================================
# 4. 辅助函数
# ==============================================================================


def _tokenize(text: str) -> List[str]:
    return [t.group(0).lower() for t in _TOKEN_RE.finditer(text) if len(t.group(0)) > 1]


def _load_prime_resources(root, download, indirected):
    from stark_qa import load_qa, load_skb

    return load_qa(_PRIME_DATASET, root=root), load_skb(
        _PRIME_DATASET, root=root, download_processed=download, indirected=indirected
    )


def iter_stark_samples(*args, **kwargs):
    # 这里保持与 source.py 的接口一致
    adapter = StarkPrimeAdapter(kwargs["dataset"], kwargs["kb"], kwargs["stark_cfg"])
    return adapter.iter_samples(kwargs["splits"])
