from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Set

import torch

_DIST_UNREACHABLE = -1
_PATH_MODE_UNDIRECTED = "undirected"
_PATH_MODE_QA_DIRECTED = "qa_directed"


@dataclass(frozen=True)
class ShortestPathLabels:
    """SubgraphRAG 风格的弱监督标签"""

    num_edges: int
    positive_edge_ids: torch.Tensor  # 位于最短路径上的边索引
    reachable_target_node_ids: torch.Tensor
    max_path_length: Optional[int]


def compute_shortest_path_labels(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    is_target_mask: torch.Tensor,
    num_nodes: int,
    path_mode: str = _PATH_MODE_QA_DIRECTED,
) -> ShortestPathLabels:
    """
    计算严格的最短路径标签。
    算法逻辑：
    1. 建立有向图索引。
    2. 计算所有锚点到所有答案的双向最短距离。
    3. 标记所有满足 d(anchor, u) + 1 + d(v, answer) == d(anchor, answer) 的边 (u, v)。
    """
    path_mode = str(path_mode or _PATH_MODE_QA_DIRECTED).strip().lower()
    if path_mode not in {_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of "
            f"{(_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)}."
        )

    # 1. 验证输入
    if edge_index.numel() == 0 or num_nodes <= 0:
        return ShortestPathLabels(
            num_edges=0,
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            max_path_length=None,
        )

    # 2. 提取局部索引
    anchor_nodes = torch.nonzero(is_anchor_mask.view(-1)).view(-1).tolist()
    answer_nodes = torch.nonzero(is_target_mask.view(-1)).view(-1).tolist()

    if not anchor_nodes or not answer_nodes:
        return ShortestPathLabels(
            num_edges=edge_index.size(1),
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            max_path_length=None,
        )

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # 3. 构建邻接表，并统一 path_mode 语义。
    adj, rev_adj = _build_adjacency(
        num_nodes=num_nodes,
        src=src,
        dst=dst,
        path_mode=path_mode,
    )

    # 4. 批量计算距离场
    # dist_from_anchors[a][n]: 锚点 a 到节点 n 的最短距离
    dist_from_anchors = {a: _bfs_dist(adj, a) for a in anchor_nodes}

    reachable_target_ids = sorted(
        {
            target
            for target in answer_nodes
            if any(
                dist_from_anchors[anchor][target] != _DIST_UNREACHABLE
                for anchor in anchor_nodes
            )
        }
    )
    if not reachable_target_ids:
        return ShortestPathLabels(
            num_edges=edge_index.size(1),
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            max_path_length=None,
        )

    # dist_to_answers[ans][n]: 节点 n 到答案 ans 的最短距离
    dist_to_answers = {ans: _bfs_dist(rev_adj, ans) for ans in reachable_target_ids}

    # 5. 寻找最短路径上的边
    positive_edge_ids: Set[int] = set()
    max_len: Optional[int] = None
    undirected = path_mode == _PATH_MODE_UNDIRECTED

    for a in anchor_nodes:
        for ans in reachable_target_ids:
            d_total = dist_from_anchors[a][ans]
            if d_total == _DIST_UNREACHABLE:
                continue

            max_len = d_total if max_len is None else max(max_len, d_total)

            # 遍历所有边，检查是否在 a -> ans 的最短路径上
            for e_id, (u, v) in enumerate(zip(src, dst)):
                if (
                    dist_from_anchors[a][u] != _DIST_UNREACHABLE
                    and dist_to_answers[ans][v] != _DIST_UNREACHABLE
                    and dist_from_anchors[a][u] + 1 + dist_to_answers[ans][v] == d_total
                ):
                    positive_edge_ids.add(e_id)
                    continue
                if (
                    undirected
                    and dist_from_anchors[a][v] != _DIST_UNREACHABLE
                    and dist_to_answers[ans][u] != _DIST_UNREACHABLE
                    and dist_from_anchors[a][v] + 1 + dist_to_answers[ans][u] == d_total
                ):
                    positive_edge_ids.add(e_id)

    return ShortestPathLabels(
        num_edges=edge_index.size(1),
        positive_edge_ids=torch.as_tensor(sorted(positive_edge_ids), dtype=torch.long),
        reachable_target_node_ids=torch.as_tensor(
            reachable_target_ids, dtype=torch.long
        ),
        max_path_length=max_len,
    )


def _build_adjacency(
    *,
    num_nodes: int,
    src: Sequence[int],
    dst: Sequence[int],
    path_mode: str,
) -> tuple[list[list[int]], list[list[int]]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    reverse_adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    undirected = path_mode == _PATH_MODE_UNDIRECTED
    for src_raw, dst_raw in zip(src, dst):
        u = int(src_raw)
        v = int(dst_raw)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)
        if undirected and u != v:
            adjacency[v].append(u)
            reverse_adjacency[u].append(v)
    return adjacency, reverse_adjacency


def _bfs_dist(adjacency: list[list[int]], start: int) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        d_next = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] == _DIST_UNREACHABLE:
                dist[v] = d_next
                q.append(v)
    return dist


__all__ = ["compute_shortest_path_labels", "ShortestPathLabels"]
