from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch

_ZERO = 0
_ONE = 1


def build_relation_id_map(
    relation_ids: Sequence[int],
    relation_kg_ids: Sequence[str],
) -> Dict[str, int]:
    if len(relation_ids) != len(relation_kg_ids):
        raise ValueError("relation_ids and relation_kg_ids length mismatch.")
    return {
        str(kg_id): int(rel_id)
        for rel_id, kg_id in zip(relation_ids, relation_kg_ids)
        if kg_id is not None
    }


def extract_inverse_relation_pairs(payload: object) -> Dict[str, str]:
    if isinstance(payload, dict):
        inner = payload.get("inverse_relations", payload)
        if isinstance(inner, list):
            return {
                str(item["forward"]): str(item["inverse_relation"])
                for item in inner
                if isinstance(item, dict)
                and item.get("forward")
                and item.get("inverse_relation")
            }
        if isinstance(inner, dict):
            out: Dict[str, str] = {}
            for key, value in inner.items():
                if isinstance(value, Mapping):
                    inv_rel = value.get("inverse_relation")
                    if inv_rel:
                        out[str(key)] = str(inv_rel)
            return out
    if isinstance(payload, list):
        return {
            str(item["forward"]): str(item["inverse_relation"])
            for item in payload
            if isinstance(item, dict)
            and item.get("forward")
            and item.get("inverse_relation")
        }
    raise ValueError("inverse_relations payload must be a dict or list.")


def load_inverse_relation_pairs(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return extract_inverse_relation_pairs(payload)


def extract_inverse_prefix_pairs(payload: object, *, prefix: str) -> Dict[str, str]:
    if not prefix:
        raise ValueError("inverse relation prefix must be non-empty.")
    entries: Sequence[Mapping[str, object]]
    if isinstance(payload, dict):
        inner = payload.get("inverse_relations", payload)
        if isinstance(inner, dict):
            entries = [
                {"forward": str(key), **(value if isinstance(value, Mapping) else {})}
                for key, value in inner.items()
            ]
        elif isinstance(inner, list):
            entries = [item for item in inner if isinstance(item, Mapping)]
        else:
            raise ValueError("inverse_relations payload must be a dict or list.")
    elif isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, Mapping)]
    else:
        raise ValueError("inverse_relations payload must be a dict or list.")
    out: Dict[str, str] = {}
    for item in entries:
        forward = item.get("forward")
        inv = item.get("inverse_relation")
        if not forward or not inv:
            continue
        fwd = str(forward)
        inv = str(inv)
        if inv.startswith(prefix):
            out[fwd] = inv
    return out


def build_generated_inverse_pairs(
    payload: object,
    *,
    prefix: str = "",
    suffix: str = "",
) -> Dict[str, str]:
    pairs = extract_inverse_relation_pairs(payload)
    if prefix:
        return {fwd: inv for fwd, inv in pairs.items() if inv.startswith(prefix)}
    if suffix:
        return {fwd: inv for fwd, inv in pairs.items() if inv == f"{fwd}{suffix}"}
    raise ValueError("build_generated_inverse_pairs requires a non-empty prefix or suffix.")


def tie_inverse_relation_embeddings(
    relation_embeddings: torch.Tensor,
    relation_id_map: Mapping[str, int],
    inverse_relations_map: Mapping[str, str],
) -> Tuple[torch.Tensor, int, int]:
    if not inverse_relations_map:
        return relation_embeddings, _ZERO, _ZERO
    forward_ids = []
    inverse_ids = []
    for forward_rel, inverse_rel in inverse_relations_map.items():
        f_id = relation_id_map.get(forward_rel)
        i_id = relation_id_map.get(inverse_rel)
        if f_id is None or i_id is None:
            raise ValueError(f"Missing relation id for {forward_rel!r} -> {inverse_rel!r}.")
        if f_id == i_id:
            raise ValueError(f"inverse relation id matches forward for {forward_rel!r}.")
        forward_ids.append(int(f_id))
        inverse_ids.append(int(i_id))
    if not forward_ids:
        return relation_embeddings, _ZERO, _ZERO
    device = relation_embeddings.device
    forward_idx = torch.tensor(forward_ids, device=device, dtype=torch.long)
    inverse_idx = torch.tensor(inverse_ids, device=device, dtype=torch.long)
    src = relation_embeddings.index_select(0, forward_idx)
    sums = torch.zeros_like(relation_embeddings)
    counts = torch.zeros((relation_embeddings.size(0), _ONE), device=device, dtype=relation_embeddings.dtype)
    sums.index_add_(0, inverse_idx, src)
    ones = torch.ones((inverse_idx.size(0), _ONE), device=device, dtype=relation_embeddings.dtype)
    counts.index_add_(0, inverse_idx, ones)
    mask = counts.squeeze(-1) > float(_ZERO)
    tied = relation_embeddings.clone()
    tied[mask] = sums[mask] / counts[mask]
    unique_targets = int(torch.unique(inverse_idx).numel())
    return tied, int(len(forward_ids)), unique_targets


__all__ = [
    "build_relation_id_map",
    "build_generated_inverse_pairs",
    "extract_inverse_prefix_pairs",
    "extract_inverse_relation_pairs",
    "load_inverse_relation_pairs",
    "tie_inverse_relation_embeddings",
]
