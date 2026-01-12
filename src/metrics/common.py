from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_ZERO = 0
_ONE = 1
_FLOAT_ZERO = 0.0
_FLOAT_ONE = 1.0
_DEFAULT_QUANTILE = 0.95


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
        if k <= _ZERO or k in seen:
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
def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


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


def _slice_answer_indices(batch: Any, sample_idx: int) -> torch.Tensor:
    attr = _get_attr(batch, "a_local_indices")
    if attr is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(attr, (list, tuple)):
        data = attr[sample_idx] if sample_idx < len(attr) else []
        return torch.as_tensor(data, dtype=torch.long).view(-_ONE)
    if torch.is_tensor(attr):
        ptr = _get_attr(batch, "a_local_indices_ptr")
        if ptr is None and hasattr(batch, "_slice_dict"):
            ptr = batch._slice_dict.get("a_local_indices")
        if ptr is None:
            return torch.empty(0, dtype=torch.long)
        ptr = torch.as_tensor(ptr, dtype=torch.long)
        if ptr.numel() <= sample_idx + _ONE:
            return torch.empty(0, dtype=torch.long)
        start = int(ptr[sample_idx].item())
        end = int(ptr[sample_idx + _ONE].item())
        return attr[start:end].to(dtype=torch.long).view(-_ONE)
    return torch.empty(0, dtype=torch.long)


def _map_answer_indices(
    indices: torch.Tensor,
    node_ptr: torch.Tensor,
    node_ids: torch.Tensor,
    sample_idx: int,
) -> torch.Tensor:
    if indices.numel() == _ZERO:
        return indices.to(device=node_ids.device, dtype=torch.long)
    node_start = int(node_ptr[sample_idx].item())
    node_end = int(node_ptr[sample_idx + _ONE].item())
    if node_end <= node_start:
        return torch.empty(0, dtype=torch.long, device=node_ids.device)
    if node_end > int(node_ids.numel()):
        raise ValueError("node_ptr exceeds node_ids length in extract_answer_entity_ids.")
    num_nodes = node_end - node_start
    indices = indices.to(device=node_ids.device, dtype=torch.long)
    in_global = (indices >= node_start) & (indices < node_end)
    if bool(in_global.all().item()):
        return node_ids.index_select(0, indices)
    in_local = (indices >= _ZERO) & (indices < num_nodes)
    if bool(in_local.all().item()):
        node_slice = node_ids[node_start:node_end]
        return node_slice.index_select(0, indices)
    raise ValueError(
        "a_local_indices out of range for extract_answer_entity_ids: "
        f"sample_idx={sample_idx} node_start={node_start} node_end={node_end} "
        f"indices_min={int(indices.min().item())} indices_max={int(indices.max().item())}"
    )


def extract_answer_entity_ids(
    batch,
    sample_idx: int,
    node_ptr: Optional[torch.Tensor],
    node_ids: torch.Tensor,
) -> torch.Tensor:
    if node_ptr is None or node_ptr.numel() <= sample_idx + _ONE:
        return torch.empty(0, dtype=torch.long)

    indices = _slice_answer_indices(batch, sample_idx)
    return _map_answer_indices(indices, node_ptr, node_ids, sample_idx)


def compute_answer_recall(samples: Iterable[Dict[str, torch.Tensor]], k_values: Sequence[int]) -> Dict[str, float]:
    ks = normalize_k_values(k_values)
    if not ks:
        return {}
    max_k = max(ks)
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    for sample in samples:
        prepared = _prepare_topk_edges(sample, max_k)
        if prepared is None:
            continue
        answers, head_top, tail_top, max_k_eff = prepared
        num_answers = int(answers.numel())
        if max_k_eff == _ZERO:
            for k in ks:
                recalls[k].append(_FLOAT_ZERO)
            continue
        for k in ks:
            k_eff = min(k, max_k_eff)
            if k_eff == _ZERO:
                recalls[k].append(_FLOAT_ZERO)
                continue
            prefix = torch.cat([head_top[:k_eff], tail_top[:k_eff]])
            hits = torch.isin(prefix, answers)
            found = torch.unique(prefix[hits])
            recalls[k].append(float(found.numel()) / float(num_answers))
    return {
        f"answer_recall@{k}": float(sum(values) / len(values)) if values else _FLOAT_ZERO
        for k, values in recalls.items()
    }


def compute_answer_hit(samples: Iterable[Dict[str, torch.Tensor]], k_values: Sequence[int]) -> Dict[str, float]:
    """Compute answer hit@k (any answer entity appears in top-k edges)."""
    ks = normalize_k_values(k_values)
    if not ks:
        return {}
    max_k = max(ks)
    hits: Dict[int, List[float]] = {k: [] for k in ks}
    for sample in samples:
        prepared = _prepare_topk_edges(sample, max_k)
        if prepared is None:
            continue
        answers, head_top, tail_top, max_k_eff = prepared
        if max_k_eff == _ZERO:
            for k in ks:
                hits[k].append(_FLOAT_ZERO)
            continue
        head_hit = torch.isin(head_top, answers)
        tail_hit = torch.isin(tail_top, answers)
        edge_hit = head_hit | tail_hit
        prefix_any = edge_hit.cumsum(0) > _ZERO
        for k in ks:
            k_eff = min(k, max_k_eff)
            if k_eff == _ZERO:
                hits[k].append(_FLOAT_ZERO)
                continue
            found_any = bool(prefix_any[k_eff - _ONE].item())
            hits[k].append(_FLOAT_ONE if found_any else _FLOAT_ZERO)
    return {
        f"answer_hit@{k}": float(sum(values) / len(values)) if values else _FLOAT_ZERO
        for k, values in hits.items()
    }


def _prepare_topk_edges(
    sample: Dict[str, torch.Tensor],
    max_k: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    answer_ids = sample.get("answer_ids")
    if answer_ids is None:
        return None
    if not torch.is_tensor(answer_ids):
        answer_ids = torch.as_tensor(answer_ids, dtype=torch.long)
    if answer_ids.numel() == _ZERO:
        return None
    answers = answer_ids.to(dtype=torch.long).view(-1).unique()
    if answers.numel() == _ZERO:
        return None
    scores = sample["scores"]
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)
    order = torch.argsort(scores, descending=True)
    max_k_eff = min(max_k, int(order.numel()))
    if max_k_eff == _ZERO:
        empty = order.new_empty((_ZERO,), dtype=torch.long)
        return answers.to(device=order.device), empty, empty, max_k_eff
    topk = order[:max_k_eff]
    head_ids = sample["head_ids"]
    tail_ids = sample["tail_ids"]
    if not torch.is_tensor(head_ids):
        head_ids = torch.as_tensor(head_ids, device=topk.device, dtype=torch.long)
    else:
        head_ids = head_ids.to(device=topk.device, dtype=torch.long)
    if not torch.is_tensor(tail_ids):
        tail_ids = torch.as_tensor(tail_ids, device=topk.device, dtype=torch.long)
    else:
        tail_ids = tail_ids.to(device=topk.device, dtype=torch.long)
    answers = answers.to(device=topk.device)
    head_top = head_ids.index_select(0, topk)
    tail_top = tail_ids.index_select(0, topk)
    return answers, head_top, tail_top, max_k_eff


# --------------------------------------------------------------------------- #
# Uncertainty summaries
# --------------------------------------------------------------------------- #
def summarize_uncertainty(values: Iterable[torch.Tensor], quantile: float = _DEFAULT_QUANTILE) -> Tuple[float, float]:
    tensors = [v for v in values if isinstance(v, torch.Tensor) and v.numel() > _ZERO]
    if not tensors:
        return _FLOAT_ZERO, _FLOAT_ZERO
    concat = torch.cat(tensors)
    mean = float(concat.mean().item())
    quant = float(concat.quantile(quantile).item())
    return mean, quant


__all__ = [
    "normalize_k_values",
    "extract_sample_ids",
    "extract_answer_entity_ids",
    "compute_answer_recall",
    "compute_answer_hit",
    "summarize_uncertainty",
]
