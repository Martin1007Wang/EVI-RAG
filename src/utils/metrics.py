from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _to_iterable(raw: Any) -> Iterable[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set, range)):
        return raw
    if isinstance(raw, str):
        return [raw]
    try:
        return list(raw)
    except TypeError:
        return [raw]


def normalize_k_values(raw_values: Any, default: Optional[Sequence[int]] = None) -> List[int]:
    normalized: List[int] = []
    seen = set()
    for item in _to_iterable(raw_values):
        try:
            k = int(item)
        except (TypeError, ValueError):
            continue
        if k <= 0 or k in seen:
            continue
        normalized.append(k)
        seen.add(k)
    if not normalized and default is not None:
        return normalize_k_values(default, default=None)
    normalized.sort()
    return normalized


# --------------------------------------------------------------------------- #
# Sample helpers
# --------------------------------------------------------------------------- #
def extract_sample_ids(batch) -> List[str]:
    raw = getattr(batch, "sample_id", None)
    if raw is None:
        num_graphs = int(getattr(batch, "num_graphs", 0))
        return [str(i) for i in range(num_graphs)]
    if isinstance(raw, (list, tuple)):
        return [str(s) for s in raw]
    if torch.is_tensor(raw):
        return [str(s.item()) for s in raw]
    return [str(raw)]


def extract_answer_entity_ids(
    batch,
    sample_idx: int,
    node_ptr: Optional[torch.Tensor],
    node_ids: torch.Tensor,
) -> torch.Tensor:
    def _get_attr(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    if node_ptr is None or node_ptr.numel() <= sample_idx + 1:
        return torch.empty(0, dtype=torch.long)

    attr = _get_attr(batch, "a_local_indices")
    if attr is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(attr, (list, tuple)):
        local = attr[sample_idx] if sample_idx < len(attr) else []
        local_tensor = torch.as_tensor(local, dtype=torch.long)
    elif torch.is_tensor(attr):
        ptr = _get_attr(batch, "a_local_indices_ptr")
        if ptr is None and hasattr(batch, "_slice_dict"):
            ptr = batch._slice_dict.get("a_local_indices")
        if ptr is None or ptr.numel() <= sample_idx + 1:
            return torch.empty(0, dtype=torch.long)
        start = int(ptr[sample_idx].item())
        end = int(ptr[sample_idx + 1].item())
        local_tensor = attr[start:end].long()
    else:
        local_tensor = torch.empty(0, dtype=torch.long)
    if local_tensor.numel() == 0:
        return local_tensor
    node_start = int(node_ptr[sample_idx].item())
    node_end = int(node_ptr[sample_idx + 1].item())
    node_slice = node_ids[node_start:node_end]
    local_tensor = local_tensor.clamp(min=0, max=max(0, node_slice.numel() - 1))
    if node_slice.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return node_slice[local_tensor]


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
@dataclass
class RankingStats:
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float


def compute_ranking_metrics(samples: Iterable[Dict[str, torch.Tensor]], k_values: Sequence[int]) -> RankingStats:
    ks = normalize_k_values(k_values, default=[1])
    totals = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0, "count": 0.0} for k in ks}
    mrr_sum = 0.0
    mrr_count = 0
    for sample in samples:
        scores = sample["scores"]
        labels = sample["labels"]
        positives = int(labels.sum().item())
        if positives <= 0:
            continue
        order = torch.argsort(scores, descending=True)
        ranked_labels = labels[order]
        positives_idx = torch.nonzero(ranked_labels > 0.5, as_tuple=False)
        if positives_idx.numel() > 0:
            mrr_sum += 1.0 / float(positives_idx[0].item() + 1)
            mrr_count += 1
        for k in ks:
            topk = ranked_labels[:k]
            hits = float(topk.sum().item())
            precision = hits / float(k)
            recall = hits / float(positives)
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            ndcg = _ndcg(ranked_labels, k)
            stat = totals[k]
            stat["precision"] += precision
            stat["recall"] += recall
            stat["f1"] += f1
            stat["ndcg"] += ndcg
            stat["count"] += 1.0
    precision_at_k: Dict[int, float] = {}
    recall_at_k: Dict[int, float] = {}
    f1_at_k: Dict[int, float] = {}
    ndcg_at_k: Dict[int, float] = {}
    for k, stat in totals.items():
        count = stat["count"] or 1.0
        precision_at_k[k] = stat["precision"] / count
        recall_at_k[k] = stat["recall"] / count
        f1_at_k[k] = stat["f1"] / count
        ndcg_at_k[k] = stat["ndcg"] / count
    mrr = mrr_sum / mrr_count if mrr_count > 0 else 0.0
    return RankingStats(precision_at_k, recall_at_k, f1_at_k, ndcg_at_k, mrr)


def _ndcg(ranked_labels: torch.Tensor, k: int) -> float:
    trunc = ranked_labels[:k]
    if trunc.numel() == 0:
        return 0.0
    positions = torch.arange(1, trunc.numel() + 1, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1.0)
    dcg = float((trunc * discounts).sum().item())
    ideal = torch.sort(ranked_labels, descending=True).values[:k]
    if ideal.numel() == 0:
        return 0.0
    ideal_dcg = float((ideal * discounts[: ideal.numel()]).sum().item())
    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def compute_answer_recall(samples: Iterable[Dict[str, torch.Tensor]], k_values: Sequence[int]) -> Dict[str, float]:
    ks = normalize_k_values(k_values)
    if not ks:
        return {}
    max_k = max(ks)
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    for sample in samples:
        answer_ids = sample.get("answer_ids")
        if answer_ids is None or answer_ids.numel() == 0:
            continue
        answers = set(int(x) for x in answer_ids.tolist())
        if not answers:
            continue
        scores = sample["scores"]
        order = torch.argsort(scores, descending=True)
        head_ids = sample["head_ids"].tolist()
        tail_ids = sample["tail_ids"].tolist()
        found: set[int] = set()
        k_pointer = 0
        for rank_idx, edge_idx in enumerate(order.tolist()[:max_k], start=1):
            if edge_idx < len(head_ids) and head_ids[edge_idx] in answers:
                found.add(head_ids[edge_idx])
            if edge_idx < len(tail_ids) and tail_ids[edge_idx] in answers:
                found.add(tail_ids[edge_idx])
            while k_pointer < len(ks) and rank_idx == ks[k_pointer]:
                recalls[ks[k_pointer]].append(len(found) / len(answers))
                k_pointer += 1
        last_recall = len(found) / len(answers)
        while k_pointer < len(ks):
            recalls[ks[k_pointer]].append(last_recall)
            k_pointer += 1
    return {f"answer_recall@{k}": float(sum(values) / len(values)) if values else 0.0 for k, values in recalls.items()}


# --------------------------------------------------------------------------- #
# Uncertainty summaries
# --------------------------------------------------------------------------- #
def summarize_uncertainty(values: Iterable[torch.Tensor], quantile: float = 0.95) -> Tuple[float, float]:
    tensors = [v for v in values if isinstance(v, torch.Tensor) and v.numel() > 0]
    if not tensors:
        return 0.0, 0.0
    concat = torch.cat(tensors)
    mean = float(concat.mean().item())
    quant = float(concat.quantile(quantile).item())
    return mean, quant


__all__ = [
    "normalize_k_values",
    "extract_sample_ids",
    "extract_answer_entity_ids",
    "RankingStats",
    "compute_ranking_metrics",
    "compute_answer_recall",
    "summarize_uncertainty",
]
