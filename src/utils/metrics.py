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
    if node_ptr is None or node_ptr.numel() <= sample_idx + 1:
        return torch.empty(0, dtype=torch.long)
    attr = getattr(batch, "a_local_indices", None)
    if attr is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(attr, (list, tuple)):
        local = attr[sample_idx] if sample_idx < len(attr) else []
        local_tensor = torch.as_tensor(local, dtype=torch.long)
    elif torch.is_tensor(attr):
        ptr = getattr(batch, "a_local_indices_ptr", None)
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


# --------------------------------------------------------------------------- #
# Selective rejection metrics
# --------------------------------------------------------------------------- #
def compute_selective_metrics(
    labels: torch.Tensor,
    confidence: torch.Tensor,
    *,
    partial_coverages: Sequence[float] = (0.8, 0.9, 0.95),
    reliability_bins: int = 10,
) -> Dict[str, Any]:
    if labels.numel() == 0:
        return {}
    labels = labels.float()
    confidence = confidence.float()
    order = torch.argsort(confidence, descending=True)
    sorted_labels = labels[order]
    errors = 1.0 - sorted_labels
    idx = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    coverage = idx / float(sorted_labels.numel())
    cum_errors = torch.cumsum(errors, dim=0)
    risk = cum_errors / idx

    aurc = float(torch.trapz(risk, coverage).item())
    metrics: Dict[str, Any] = {"aurc": aurc}

    # Exact AUGRC per NeurIPS definition
    aug_rc = _compute_augrc(labels, confidence)
    metrics["aug_rc"] = aug_rc

    tp = torch.cumsum(sorted_labels, dim=0)
    accuracy = tp / idx
    rejection = 1.0 - coverage
    arc_auc = float(torch.trapz(accuracy.flip(0), rejection.flip(0)).item())
    metrics["arc_auc"] = arc_auc

    total_pos = max(sorted_labels.sum().item(), 1.0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / total_pos
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_auc = float(torch.trapz(f1, coverage).item())
    metrics["f1_auc"] = f1_auc

    for cov in partial_coverages:
        cov = float(cov)
        if cov <= 0 or cov > 1:
            continue
        mask = coverage <= cov
        if mask.sum().item() == 0:
            continue
        cov_axis = coverage[mask]
        risk_axis = risk[mask]
        partial_area = float(torch.trapz(risk_axis, cov_axis).item())
        metrics[f"aurc_partial@{cov}"] = partial_area
        acc_mask = accuracy[mask]
        risk_mask = risk[mask]
        if acc_mask.numel() > 0:
            metrics[f"accuracy@{cov}"] = float(acc_mask[-1].item())
        if risk_mask.numel() > 0:
            metrics[f"risk@{cov}"] = float(risk_mask[-1].item())

    # Error (OOD) detection AUROC treating errors as positives
    error_flags = 1.0 - labels
    pos = int(error_flags.sum().item())
    neg = int(labels.sum().item())
    if pos > 0 and neg > 0:
        auc = _rank_statistic_auc(error_flags, -confidence)
        metrics["error_detection_auroc"] = auc

    # Reliability diagram data
    reliability = compute_reliability_diagram(labels, confidence, n_bins=reliability_bins)
    metrics["ece_bin"] = float(_expected_calibration_error(reliability))
    metrics["mce"] = float(_maximum_calibration_error(reliability))
    metrics["brier"] = float(torch.mean((confidence - labels) ** 2).item())
    metrics["reliability"] = reliability

    return metrics


def compute_reliability_diagram(
    labels: torch.Tensor,
    predicted: torch.Tensor,
    *,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    labels = labels.float()
    predicted = predicted.float()
    if labels.numel() == 0 or n_bins <= 0:
        return {"predicted": [], "observed": [], "count": []}
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=predicted.device)
    bin_ids = torch.bucketize(predicted, edges) - 1
    bin_ids = bin_ids.clamp(0, n_bins - 1)
    bin_conf: List[float] = []
    bin_cnt: List[float] = []
    bin_edges: List[float] = [float(e.item()) for e in edges]
    bin_centers: List[float] = []
    bin_acc: List[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum().item())
        center = (bin_edges[b] + bin_edges[b + 1]) / 2.0
        bin_centers.append(center)
        if count == 0:
            bin_conf.append(center)
            bin_acc.append(0.0)
            bin_cnt.append(0.0)
            continue
        bin_conf.append(float(predicted[mask].mean().item()))
        bin_acc.append(float(labels[mask].mean().item()))
        bin_cnt.append(float(count))
    return {
        "edges": bin_edges,
        "centers": bin_centers,
        "predicted": bin_conf,
        "observed": bin_acc,
        "count": bin_cnt,
    }


def _rank_statistic_auc(pos_flags: torch.Tensor, scores: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32, device=scores.device)
    pos_mask = pos_flags > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = float(ranks[pos_mask].sum().item())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _compute_augrc(labels: torch.Tensor, confidence: torch.Tensor) -> float:
    n = labels.numel()
    if n == 0:
        return 0.0
    order = torch.argsort(confidence, descending=False)
    ranked_errors = (1.0 - labels)[order]
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=labels.device)
    cdf = ranks / (n + 1)
    alpha = -torch.log1p(-cdf)
    aug = float((alpha * ranked_errors).mean().item())
    return aug


def _expected_calibration_error(reliability: Dict[str, List[float]]) -> float:
    counts = torch.tensor(reliability["count"], dtype=torch.float32)
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    pred = torch.tensor(reliability["predicted"], dtype=torch.float32)
    obs = torch.tensor(reliability["observed"], dtype=torch.float32)
    return float((counts * (pred - obs).abs()).sum().item() / total)


def _maximum_calibration_error(reliability: Dict[str, List[float]]) -> float:
    if not reliability["predicted"]:
        return 0.0
    pred = torch.tensor(reliability["predicted"], dtype=torch.float32)
    obs = torch.tensor(reliability["observed"], dtype=torch.float32)
    return float((pred - obs).abs().max().item())


__all__ = [
    "normalize_k_values",
    "extract_sample_ids",
    "extract_answer_entity_ids",
    "RankingStats",
    "compute_ranking_metrics",
    "compute_answer_recall",
    "summarize_uncertainty",
    "compute_selective_metrics",
]
