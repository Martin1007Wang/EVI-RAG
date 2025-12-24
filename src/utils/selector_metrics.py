from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

import torch


def edge_precision_recall_f1(
    *,
    selected_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    allow_empty_positive: bool,
    sample_id: str,
) -> Tuple[float, float, float]:
    if selected_mask.numel() != positive_mask.numel():
        raise ValueError(f"selected_mask/positive_mask length mismatch for {sample_id}")
    tp = int((selected_mask & positive_mask).sum().item())
    pred = int(selected_mask.sum().item())
    pos = int(positive_mask.sum().item())
    if pos == 0:
        if not allow_empty_positive:
            raise ValueError(f"positive edge set is empty for {sample_id}")
        recall = 1.0
        precision = 1.0 if pred == 0 else 0.0
    else:
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / pos
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def path_hit_from_selected(
    *,
    edge_head_locals: torch.Tensor,
    edge_tail_locals: torch.Tensor,
    selected_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    start_node_locals: torch.Tensor,
    answer_node_locals: torch.Tensor,
    num_nodes: int,
    sample_id: str,
) -> bool:
    if edge_head_locals.numel() != edge_tail_locals.numel():
        raise ValueError(f"edge_head_locals/edge_tail_locals length mismatch for {sample_id}")
    if selected_mask.numel() != edge_head_locals.numel():
        raise ValueError(f"selected_mask length mismatch for {sample_id}")
    if start_node_locals.numel() == 0 or answer_node_locals.numel() == 0:
        return False
    if torch.isin(start_node_locals, answer_node_locals).any().item():
        return True

    active = selected_mask & positive_mask
    if not bool(active.any().item()):
        return False
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive for {sample_id}")

    heads = edge_head_locals[active].tolist()
    tails = edge_tail_locals[active].tolist()
    adj = [[] for _ in range(num_nodes)]
    for h, t in zip(heads, tails):
        if 0 <= h < num_nodes and 0 <= t < num_nodes:
            adj[h].append(t)

    answers = set(int(x) for x in answer_node_locals.tolist())
    visited = [False] * num_nodes
    q: deque[int] = deque()
    for s in start_node_locals.tolist():
        if 0 <= s < num_nodes and not visited[s]:
            visited[s] = True
            q.append(int(s))

    while q:
        u = q.popleft()
        if u in answers:
            return True
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    return False


def ensure_int_list(values: Iterable[int], *, sample_id: str, field: str) -> list[int]:
    out: list[int] = []
    for idx, val in enumerate(values):
        if not isinstance(val, int):
            raise ValueError(f"{field}[{idx}] must be int for {sample_id}, got {type(val).__name__}")
        out.append(int(val))
    return out
