from __future__ import annotations

import torch

_GUMBEL_EPS = 1e-10


def neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def _normalize_num_segments(num_segments: int | torch.Tensor, *, device: torch.device) -> int | torch.Tensor:
    if torch.is_tensor(num_segments):
        if num_segments.device != device:
            return num_segments.to(device=device)
        return num_segments
    return int(num_segments)


def segment_max(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-wise max for 1D tensors with argmax indices."""
    num_segments = _normalize_num_segments(num_segments, device=src.device)
    if src.numel() == 0:
        max_per = torch.full((num_segments,), neg_inf_value(src), device=src.device, dtype=src.dtype)
        argmax = torch.zeros((num_segments,), device=src.device, dtype=torch.long)
        return max_per, argmax

    src = src.view(-1)
    if not torch.isfinite(src).all():
        neg_inf = neg_inf_value(src)
        finfo = torch.finfo(src.dtype)
        src = torch.nan_to_num(src, nan=neg_inf, posinf=float(finfo.max), neginf=neg_inf)

    segment_ids = segment_ids.to(device=src.device, dtype=torch.long).view(-1)
    max_per = torch.full((num_segments,), neg_inf_value(src), device=src.device, dtype=src.dtype)
    max_per.scatter_reduce_(0, segment_ids, src, reduce="amax", include_self=True)

    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    is_max = src == max_per.index_select(0, segment_ids)
    sentinel = src.numel()
    candidate = torch.where(is_max, positions, torch.full_like(positions, sentinel))
    argmin = torch.full((num_segments,), sentinel, device=src.device, dtype=torch.long)
    argmin.scatter_reduce_(0, segment_ids, candidate, reduce="amin", include_self=True)
    argmax = torch.where(argmin == sentinel, torch.zeros_like(argmin), argmin)
    return max_per, argmax


def segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int | torch.Tensor) -> torch.Tensor:
    num_segments = _normalize_num_segments(num_segments, device=logits.device)
    if logits.numel() == 0:
        return torch.full((num_segments,), neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    calc_dtype = logits.dtype
    neg_inf = torch.finfo(calc_dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=calc_dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=calc_dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(calc_dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


def gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u.clamp(min=_GUMBEL_EPS, max=1.0 - _GUMBEL_EPS)))


__all__ = [
    "gumbel_noise_like",
    "neg_inf_value",
    "segment_logsumexp_1d",
    "segment_max",
]
