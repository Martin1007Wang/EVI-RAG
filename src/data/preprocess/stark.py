from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .samples import RawSample

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class StarkAdapterConfig:
    name: str
    root: str | None
    download_processed: bool
    anchor_top_k: int
    anchor_index_limit: int
    num_hops: int
    ppr_alpha: float
    ppr_iterations: int
    ppr_top_nodes: int
    max_edges: int
    candidate_pool: int
    include_answer_when_unreachable: bool


def iter_stark_samples(
    *,
    dataset: str,
    split_mapping: Mapping[str, str],
    options: Mapping[str, Any] | None = None,
) -> Iterable[RawSample]:
    cfg = _build_config(dataset=dataset, options=options or {})
    load_qa, load_skb = _import_stark()
    qa_dataset = load_qa(cfg.name, root=cfg.root)
    skb = load_skb(
        cfg.name,
        root=cfg.root,
        download_processed=cfg.download_processed,
    )
    linker = _AnchorLinker.from_skb(skb, index_limit=cfg.anchor_index_limit)
    graph_index = _StarkGraphIndex.from_skb(skb)
    split_indices = qa_dataset.get_idx_split()
    stats = _StarkStats()

    for logical_split, source_split in split_mapping.items():
        if source_split not in split_indices:
            raise ValueError(
                f"STaRK split {source_split!r} is unavailable. "
                f"Available splits: {sorted(split_indices)}"
            )
        for idx in split_indices[source_split].tolist():
            query, qid, answer_ids, _ = qa_dataset[int(idx)]
            sample = _build_sample(
                dataset=dataset,
                split=str(logical_split),
                question_id=str(qid),
                question=str(query),
                answer_ids=[int(aid) for aid in answer_ids],
                skb=skb,
                graph_index=graph_index,
                linker=linker,
                cfg=cfg,
                stats=stats,
            )
            if sample is not None:
                yield sample

    log.info(
        "STaRK adapter complete: dataset=%s samples=%d no_anchor=%d empty_subgraph=%d "
        "answer_missing=%d avg_nodes=%.1f avg_edges=%.1f",
        cfg.name,
        stats.samples,
        stats.no_anchor,
        stats.empty_subgraph,
        stats.answer_missing,
        stats.avg_nodes(),
        stats.avg_edges(),
    )


@dataclass
class _StarkStats:
    samples: int = 0
    no_anchor: int = 0
    empty_subgraph: int = 0
    answer_missing: int = 0
    total_nodes: int = 0
    total_edges: int = 0

    def avg_nodes(self) -> float:
        return self.total_nodes / self.samples if self.samples else 0.0

    def avg_edges(self) -> float:
        return self.total_edges / self.samples if self.samples else 0.0


class _AnchorLinker:
    def __init__(self, postings: Mapping[str, Mapping[int, int]], doc_count: int):
        self._postings = postings
        self._doc_count = max(1, doc_count)

    @classmethod
    def from_skb(cls, skb: Any, *, index_limit: int) -> "_AnchorLinker":
        limit = min(int(index_limit), int(skb.num_nodes()))
        postings: dict[str, Counter[int]] = defaultdict(Counter)
        for node_id in range(limit):
            text = _node_text(skb, node_id)
            for token, count in Counter(_tokenize(text)).items():
                postings[token][node_id] = count
        return cls(postings=postings, doc_count=limit)

    def link(self, query: str, *, top_k: int) -> list[int]:
        scores: Counter[int] = Counter()
        query_counts = Counter(_tokenize(query))
        for token, query_tf in query_counts.items():
            posting = self._postings.get(token)
            if not posting:
                continue
            idf = math.log((self._doc_count + 1) / (len(posting) + 1)) + 1.0
            for node_id, node_tf in posting.items():
                scores[node_id] += query_tf * node_tf * idf
        if not scores:
            return []
        return [node_id for node_id, _ in scores.most_common(max(1, top_k))]


class _StarkGraphIndex:
    def __init__(self, edge_ids_by_node: Mapping[int, Sequence[int]]):
        self._edge_ids_by_node = edge_ids_by_node

    @classmethod
    def from_skb(cls, skb: Any) -> "_StarkGraphIndex":
        edge_ids_by_node: dict[int, list[int]] = defaultdict(list)
        edge_index = skb.edge_index
        for edge_id in range(int(edge_index.size(1))):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            edge_ids_by_node[src].append(edge_id)
            edge_ids_by_node[dst].append(edge_id)
        return cls(edge_ids_by_node=edge_ids_by_node)

    def incident_edge_ids(self, node_ids: Iterable[int]) -> Iterable[int]:
        seen: set[int] = set()
        for node_id in node_ids:
            for edge_id in self._edge_ids_by_node.get(int(node_id), ()):
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                yield edge_id


def _build_sample(
    *,
    dataset: str,
    split: str,
    question_id: str,
    question: str,
    answer_ids: Sequence[int],
    skb: Any,
    graph_index: _StarkGraphIndex,
    linker: _AnchorLinker,
    cfg: StarkAdapterConfig,
    stats: _StarkStats,
) -> RawSample | None:
    anchors = linker.link(question, top_k=cfg.anchor_top_k)
    if not anchors:
        stats.no_anchor += 1
        return None

    candidate_nodes = _two_hop_candidates(
        skb=skb,
        seeds=anchors,
        num_hops=cfg.num_hops,
        limit=max(cfg.candidate_pool, cfg.ppr_top_nodes),
    )
    if not candidate_nodes:
        stats.empty_subgraph += 1
        return None

    selected_nodes = _ppr_select(
        skb=skb,
        seeds=anchors,
        candidate_nodes=candidate_nodes,
        alpha=cfg.ppr_alpha,
        iterations=cfg.ppr_iterations,
        top_nodes=cfg.ppr_top_nodes,
    )
    if cfg.include_answer_when_unreachable:
        selected_nodes.update(int(aid) for aid in answer_ids)

    edge_ids = _select_edges(
        skb=skb,
        graph_index=graph_index,
        selected_nodes=selected_nodes,
        max_edges=cfg.max_edges,
    )
    if not edge_ids:
        stats.empty_subgraph += 1
        return None

    graph = tuple(_edge_to_triple(skb, edge_id) for edge_id in edge_ids)
    node_names = {_entity_name(skb, node_id) for edge_id in edge_ids for node_id in _edge_nodes(skb, edge_id)}
    question_entities = tuple(
        name for node_id in anchors if (name := _entity_name(skb, node_id)) in node_names
    )
    answer_entities = tuple(
        name for node_id in answer_ids if (name := _entity_name(skb, int(node_id))) in node_names
    )
    if not answer_entities:
        stats.answer_missing += 1

    stats.samples += 1
    stats.total_nodes += len(node_names)
    stats.total_edges += len(graph)
    return RawSample(
        dataset=dataset,
        split=split,
        question_id=question_id,
        question=question,
        graph=graph,
        question_entities=question_entities,
        answer_entities=answer_entities,
    )


def _two_hop_candidates(
    *,
    skb: Any,
    seeds: Sequence[int],
    num_hops: int,
    limit: int,
) -> set[int]:
    seen = set(int(seed) for seed in seeds)
    queue = deque((int(seed), 0) for seed in seeds)
    while queue and len(seen) < limit:
        node_id, depth = queue.popleft()
        if depth >= num_hops:
            continue
        for neighbor in skb.get_neighbor_nodes(node_id, edge_type="*"):
            neighbor = int(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, depth + 1))
            if len(seen) >= limit:
                break
    return seen


def _ppr_select(
    *,
    skb: Any,
    seeds: Sequence[int],
    candidate_nodes: set[int],
    alpha: float,
    iterations: int,
    top_nodes: int,
) -> set[int]:
    seeds = [int(seed) for seed in seeds if int(seed) in candidate_nodes]
    if not seeds:
        return set()
    seed_mass = 1.0 / len(seeds)
    scores = {seed: seed_mass for seed in seeds}
    seed_scores = {seed: seed_mass for seed in seeds}
    for _ in range(max(1, iterations)):
        next_scores = defaultdict(float)
        for seed, value in seed_scores.items():
            next_scores[seed] += alpha * value
        for node_id, value in scores.items():
            neighbors = [
                int(n)
                for n in skb.get_neighbor_nodes(node_id, edge_type="*")
                if int(n) in candidate_nodes
            ]
            if not neighbors:
                next_scores[node_id] += (1.0 - alpha) * value
                continue
            share = (1.0 - alpha) * value / len(neighbors)
            for neighbor in neighbors:
                next_scores[neighbor] += share
        scores = dict(next_scores)
    return set(heapq.nlargest(max(1, top_nodes), scores, key=scores.get))


def _select_edges(
    *,
    skb: Any,
    graph_index: _StarkGraphIndex,
    selected_nodes: set[int],
    max_edges: int,
) -> list[int]:
    edge_ids: list[int] = []
    for edge_id in graph_index.incident_edge_ids(selected_nodes):
        src, dst = _edge_nodes(skb, edge_id)
        if src in selected_nodes and dst in selected_nodes:
            edge_ids.append(edge_id)
            if len(edge_ids) >= max_edges:
                break
    return edge_ids


def _edge_nodes(skb: Any, edge_id: int) -> tuple[int, int]:
    edge_index = skb.edge_index
    return int(edge_index[0, edge_id].item()), int(edge_index[1, edge_id].item())


def _edge_to_triple(skb: Any, edge_id: int) -> tuple[str, str, str]:
    src, dst = _edge_nodes(skb, edge_id)
    return (_entity_name(skb, src), _relation_name(skb, edge_id), _entity_name(skb, dst))


def _entity_name(skb: Any, node_id: int) -> str:
    node_type = ""
    if getattr(skb, "node_types", None) is not None:
        try:
            node_type = str(skb.get_node_type_by_id(int(node_id)))
        except Exception:
            node_type = ""
    text = _node_title(skb, int(node_id)) or f"node_{int(node_id)}"
    return f"{node_type}:{text}" if node_type else text


def _relation_name(skb: Any, edge_id: int) -> str:
    try:
        return str(skb.get_edge_type_by_id(int(edge_id)))
    except Exception:
        edge_types = getattr(skb, "edge_types", None)
        if edge_types is not None:
            return f"rel_{int(edge_types[edge_id].item())}"
        return "related_to"


def _node_text(skb: Any, node_id: int) -> str:
    title = _node_title(skb, node_id)
    try:
        doc = str(skb.get_doc_info(node_id, add_rel=False, compact=True))
    except Exception:
        doc = ""
    return f"{title} {doc}".strip()


def _node_title(skb: Any, node_id: int) -> str:
    info = getattr(skb, "node_info", {}).get(int(node_id), {})
    values = _flatten_values(info)
    for value in values:
        if value:
            return value
    return f"node_{int(node_id)}"


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        preferred = []
        for key in ("name", "title", "display_name", "product_title", "drug_name"):
            if key in value:
                preferred.extend(_flatten_values(value[key]))
        if preferred:
            return preferred
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_values(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _build_config(*, dataset: str, options: Mapping[str, Any]) -> StarkAdapterConfig:
    name = str(options.get("dataset", dataset)).strip()
    root_raw = options.get("root")
    root = None if root_raw in (None, "") else str(root_raw)
    return StarkAdapterConfig(
        name=name,
        root=root,
        download_processed=bool(options.get("download_processed", True)),
        anchor_top_k=int(_nested_get(options, ("linker", "max_entities"), 4)),
        anchor_index_limit=int(_nested_get(options, ("linker", "index_limit"), 200000)),
        num_hops=int(_nested_get(options, ("local_graph", "num_hops"), 2)),
        ppr_alpha=float(_nested_get(options, ("local_graph", "ppr_alpha"), 0.15)),
        ppr_iterations=int(_nested_get(options, ("local_graph", "ppr_iterations"), 20)),
        ppr_top_nodes=int(_nested_get(options, ("local_graph", "ppr_top_nodes"), 128)),
        max_edges=int(_nested_get(options, ("local_graph", "max_edges"), 512)),
        candidate_pool=int(_nested_get(options, ("local_graph", "candidate_pool"), 4096)),
        include_answer_when_unreachable=bool(
            _nested_get(options, ("local_graph", "include_answer_when_unreachable"), False)
        ),
    )


def _nested_get(options: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    value: Any = options
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _import_stark() -> tuple[Any, Any]:
    try:
        from stark_qa import load_qa, load_skb
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "dataset_source=stark requires the optional package `stark-qa`. "
            "Install it with `pip install stark-qa` or use the provided setup command."
        ) from exc
    return load_qa, load_skb


__all__ = ["StarkAdapterConfig", "iter_stark_samples"]
