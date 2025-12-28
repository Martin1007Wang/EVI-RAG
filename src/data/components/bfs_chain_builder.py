from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.g_agent_dataset import GAgentSample, load_g_agent_samples

DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1


@dataclass(frozen=True)
class _OrientedEdge:
    edge_id: int
    src_local: int
    dst_local: int
    head_local: int
    tail_local: int
    relation_id: int
    score: float
    direction: int


@dataclass
class _Chain:
    edges: List[_OrientedEdge]
    last_node: int
    score: float
    used_edge_ids: Optional[set[int]] = None
    visited_nodes: Optional[set[int]] = None


@dataclass
class BFSChainSettings:
    max_chain_length: int
    min_chain_length: int = 1
    max_chains_per_sample: int = 100
    max_total_chains: int = 5000
    allow_backward: bool = True
    max_branch_per_node: Optional[int] = None
    forbid_edge_revisit: bool = True
    forbid_node_revisit: bool = False


def build_bfs_candidate_chains(sample: GAgentSample, *, settings: BFSChainSettings) -> List[Dict[str, Any]]:
    start_nodes = sample.start_node_locals.view(-1).tolist()
    if not start_nodes:
        return []
    max_len = int(settings.max_chain_length)
    if max_len <= 0:
        raise ValueError("max_chain_length must be a positive integer.")

    heads = sample.edge_head_locals.view(-1).tolist()
    tails = sample.edge_tail_locals.view(-1).tolist()
    relations = sample.edge_relations.view(-1).tolist()
    scores = sample.edge_scores.view(-1).tolist()
    node_entity_ids = sample.node_entity_ids.view(-1).tolist()
    num_nodes = len(node_entity_ids)

    adj = _build_oriented_adjacency(
        num_nodes=num_nodes,
        heads=heads,
        tails=tails,
        relations=relations,
        scores=scores,
        allow_backward=settings.allow_backward,
        max_branch_per_node=settings.max_branch_per_node,
    )
    raw_chains = _expand_chains(
        adj=adj,
        start_nodes=start_nodes,
        max_chain_length=max_len,
        min_chain_length=int(settings.min_chain_length),
        max_total_chains=int(settings.max_total_chains),
        forbid_edge_revisit=settings.forbid_edge_revisit,
        forbid_node_revisit=settings.forbid_node_revisit,
    )
    candidates = _dedup_chains(
        raw_chains,
        node_entity_ids=node_entity_ids,
    )
    if settings.max_chains_per_sample is not None:
        limit = int(settings.max_chains_per_sample)
        candidates = candidates[:max(limit, 0)]
    for rank, cand in enumerate(candidates, 1):
        cand["rank"] = rank
    return candidates


def export_bfs_chain_cache(
    *,
    g_agent_path: str | Path,
    output_dir: str | Path,
    split: str,
    settings: BFSChainSettings,
    artifact_name: str = "eval_bfs",
    schema_version: int = 1,
    overwrite: bool = True,
    drop_unreachable: bool = False,
) -> Tuple[Path, int]:
    g_agent_path = Path(g_agent_path).expanduser().resolve()
    if not g_agent_path.exists():
        raise FileNotFoundError(f"g_agent_path not found: {g_agent_path}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"
    manifest_path = output_dir / f"{split}.manifest.json"
    if overwrite:
        output_path.write_text("", encoding="utf-8")

    samples = load_g_agent_samples(g_agent_path, drop_unreachable=drop_unreachable)
    total = 0
    with output_path.open("a", encoding="utf-8") as f:
        for sample in samples:
            chains = build_bfs_candidate_chains(sample, settings=settings)
            record = {
                "sample_id": str(sample.sample_id),
                "question": str(sample.question),
                "candidate_chains": chains,
            }
            f.write(json.dumps(record) + "\n")
            total += 1

    manifest = {
        "artifact": str(artifact_name).strip(),
        "schema_version": int(schema_version),
        "file": output_path.name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "producer": "bfs_chain_builder",
        "settings": {
            "max_chain_length": int(settings.max_chain_length),
            "min_chain_length": int(settings.min_chain_length),
            "max_chains_per_sample": int(settings.max_chains_per_sample),
            "max_total_chains": int(settings.max_total_chains),
            "allow_backward": bool(settings.allow_backward),
            "max_branch_per_node": settings.max_branch_per_node,
            "forbid_edge_revisit": bool(settings.forbid_edge_revisit),
            "forbid_node_revisit": bool(settings.forbid_node_revisit),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path, total


def _build_oriented_adjacency(
    *,
    num_nodes: int,
    heads: Sequence[int],
    tails: Sequence[int],
    relations: Sequence[int],
    scores: Sequence[float],
    allow_backward: bool,
    max_branch_per_node: Optional[int],
) -> List[List[_OrientedEdge]]:
    adj: List[List[_OrientedEdge]] = [[] for _ in range(num_nodes)]
    for edge_id, (head, tail, rel, score) in enumerate(zip(heads, tails, relations, scores)):
        adj[int(head)].append(
            _OrientedEdge(
                edge_id=int(edge_id),
                src_local=int(head),
                dst_local=int(tail),
                head_local=int(head),
                tail_local=int(tail),
                relation_id=int(rel),
                score=float(score),
                direction=DIRECTION_FORWARD,
            )
        )
        if allow_backward:
            adj[int(tail)].append(
                _OrientedEdge(
                    edge_id=int(edge_id),
                    src_local=int(tail),
                    dst_local=int(head),
                    head_local=int(head),
                    tail_local=int(tail),
                    relation_id=int(rel),
                    score=float(score),
                    direction=DIRECTION_BACKWARD,
                )
            )
    for edges in adj:
        edges.sort(key=lambda e: (-e.score, e.edge_id, e.direction))
        if max_branch_per_node is not None:
            keep = int(max_branch_per_node)
            if keep >= 0:
                del edges[keep:]
    return adj


def _expand_chains(
    *,
    adj: List[List[_OrientedEdge]],
    start_nodes: Iterable[int],
    max_chain_length: int,
    min_chain_length: int,
    max_total_chains: int,
    forbid_edge_revisit: bool,
    forbid_node_revisit: bool,
) -> List[_Chain]:
    queue: deque[_Chain] = deque()
    for s in start_nodes:
        if s < 0 or s >= len(adj):
            continue
        for e in adj[int(s)]:
            used = {e.edge_id} if forbid_edge_revisit else None
            visited = {int(s), e.dst_local} if forbid_node_revisit else None
            queue.append(_Chain(edges=[e], last_node=e.dst_local, score=e.score, used_edge_ids=used, visited_nodes=visited))

    chains: List[_Chain] = []
    while queue:
        chain = queue.popleft()
        if len(chain.edges) >= min_chain_length:
            chains.append(chain)
            if max_total_chains > 0 and len(chains) >= max_total_chains:
                break
        if len(chain.edges) >= max_chain_length:
            continue

        for e in adj[int(chain.last_node)]:
            if forbid_edge_revisit and chain.used_edge_ids is not None and e.edge_id in chain.used_edge_ids:
                continue
            if forbid_node_revisit and chain.visited_nodes is not None and e.dst_local in chain.visited_nodes:
                continue

            used = None
            if forbid_edge_revisit and chain.used_edge_ids is not None:
                used = set(chain.used_edge_ids)
                used.add(e.edge_id)

            visited = None
            if forbid_node_revisit and chain.visited_nodes is not None:
                visited = set(chain.visited_nodes)
                visited.add(e.dst_local)

            queue.append(
                _Chain(
                    edges=[*chain.edges, e],
                    last_node=e.dst_local,
                    score=chain.score + e.score,
                    used_edge_ids=used,
                    visited_nodes=visited,
                )
            )
    return chains


def _dedup_chains(chains: Iterable[_Chain], *, node_entity_ids: Sequence[int]) -> List[Dict[str, Any]]:
    chain_stats: Dict[Tuple[Tuple[int, int, int], ...], Dict[str, Any]] = {}
    for chain in chains:
        sig = tuple(
            (int(node_entity_ids[e.src_local]), int(e.relation_id), int(node_entity_ids[e.dst_local]))
            for e in chain.edges
        )
        if not sig:
            continue
        stat = chain_stats.get(sig)
        if stat is None:
            chain_stats[sig] = {
                "frequency": 1,
                "score": float(chain.score),
                "edges": chain.edges,
            }
        else:
            stat["frequency"] += 1
            if float(chain.score) > float(stat["score"]):
                stat["score"] = float(chain.score)
                stat["edges"] = chain.edges

    candidates: List[Dict[str, Any]] = []
    for sig, stat in chain_stats.items():
        edges = stat["edges"]
        chain_edges = [
            _edge_to_dict(e, node_entity_ids=node_entity_ids)
            for e in edges
        ]
        candidates.append(
            {
                "signature": sig,
                "length": len(edges),
                "frequency": int(stat["frequency"]),
                "score": float(stat["score"]),
                "edge_local_ids": [int(e.edge_id) for e in edges],
                "chain_edges": chain_edges,
            }
        )

    candidates.sort(key=lambda c: (-int(c["frequency"]), -int(c["length"]), -float(c["score"])))
    return candidates


def _edge_to_dict(edge: _OrientedEdge, *, node_entity_ids: Sequence[int]) -> Dict[str, Any]:
    head_entity = int(node_entity_ids[edge.head_local])
    tail_entity = int(node_entity_ids[edge.tail_local])
    src_entity = int(node_entity_ids[edge.src_local])
    dst_entity = int(node_entity_ids[edge.dst_local])
    return {
        "edge_id": int(edge.edge_id),
        "head_entity_id": head_entity,
        "tail_entity_id": tail_entity,
        "relation_id": int(edge.relation_id),
        "src_entity_id": src_entity,
        "dst_entity_id": dst_entity,
        "src_node_local": int(edge.src_local),
        "dst_node_local": int(edge.dst_local),
        "direction": int(edge.direction),
    }


__all__ = [
    "BFSChainSettings",
    "build_bfs_candidate_chains",
    "export_bfs_chain_cache",
]
