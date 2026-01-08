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
        answer_ids = sample.get("answer_ids")
        if answer_ids is None or answer_ids.numel() == _ZERO:
            continue
        answers = set(int(x) for x in answer_ids.tolist())
        if not answers:
            continue
        scores = sample["scores"]
        order = torch.argsort(scores, descending=True)
        head_ids = sample["head_ids"].tolist()
        tail_ids = sample["tail_ids"].tolist()
        found: set[int] = set()
        k_pointer = _ZERO
        for rank_idx, edge_idx in enumerate(order.tolist()[:max_k], start=_ONE):
            if edge_idx < len(head_ids) and head_ids[edge_idx] in answers:
                found.add(head_ids[edge_idx])
            if edge_idx < len(tail_ids) and tail_ids[edge_idx] in answers:
                found.add(tail_ids[edge_idx])
            while k_pointer < len(ks) and rank_idx == ks[k_pointer]:
                recalls[ks[k_pointer]].append(len(found) / len(answers))
                k_pointer += _ONE
        last_recall = len(found) / len(answers)
        while k_pointer < len(ks):
            recalls[ks[k_pointer]].append(last_recall)
            k_pointer += _ONE
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
        answer_ids = sample.get("answer_ids")
        if answer_ids is None or answer_ids.numel() == _ZERO:
            continue
        answers = set(int(x) for x in answer_ids.tolist())
        if not answers:
            continue
        scores = sample["scores"]
        order = torch.argsort(scores, descending=True)
        head_ids = sample["head_ids"].tolist()
        tail_ids = sample["tail_ids"].tolist()
        found_any = False
        k_pointer = _ZERO
        for rank_idx, edge_idx in enumerate(order.tolist()[:max_k], start=_ONE):
            if edge_idx < len(head_ids) and head_ids[edge_idx] in answers:
                found_any = True
            if edge_idx < len(tail_ids) and tail_ids[edge_idx] in answers:
                found_any = True
            while k_pointer < len(ks) and rank_idx == ks[k_pointer]:
                hits[ks[k_pointer]].append(_FLOAT_ONE if found_any else _FLOAT_ZERO)
                k_pointer += _ONE
        last_val = _FLOAT_ONE if found_any else _FLOAT_ZERO
        while k_pointer < len(ks):
            hits[ks[k_pointer]].append(last_val)
            k_pointer += _ONE
    return {
        f"answer_hit@{k}": float(sum(values) / len(values)) if values else _FLOAT_ZERO
        for k, values in hits.items()
    }


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
