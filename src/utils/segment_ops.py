from __future__ import annotations

import torch


def _normalize_segment_ids(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    op_name: str,
) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError(
            f"{op_name} expects 1D `values`, got shape={tuple(values.shape)}."
        )
    if segment_ids.dim() != 1:
        raise ValueError(
            f"{op_name} expects 1D `segment_ids`, got shape={tuple(segment_ids.shape)}."
        )
    if int(values.numel()) != int(segment_ids.numel()):
        raise ValueError(
            f"{op_name} size mismatch between values and segment_ids: "
            f"values={int(values.numel())}, segment_ids={int(segment_ids.numel())}."
        )
    if num_segments < 0:
        raise ValueError(f"{op_name} requires num_segments >= 0, got {num_segments}.")
    ids = segment_ids.to(device=values.device, dtype=torch.long)
    if int(ids.numel()) == 0:
        return ids
    if bool((ids < 0).any().item()) or bool((ids >= num_segments).any().item()):
        raise ValueError(
            f"{op_name} received out-of-range segment_ids for num_segments={num_segments}."
        )
    return ids


def compute_has_finite_edges(
    *,
    edge_logits: torch.Tensor,
    out_degrees: torch.Tensor,
) -> torch.Tensor:
    num_agents_total = int(out_degrees.numel())
    if num_agents_total == 0:
        return out_degrees.new_zeros((0,), dtype=torch.bool)
    if edge_logits.numel() == 0:
        return out_degrees.new_zeros((num_agents_total,), dtype=torch.bool)
    agent_ids = torch.arange(
        num_agents_total, device=out_degrees.device, dtype=torch.long
    )
    edge_agent_ids = agent_ids.repeat_interleave(out_degrees)
    finite_edges = torch.isfinite(edge_logits).to(dtype=torch.int32)
    has_finite = torch.zeros(
        (num_agents_total,), device=out_degrees.device, dtype=torch.int32
    )
    has_finite.scatter_reduce_(
        0, edge_agent_ids, finite_edges, reduce="amax", include_self=True
    )
    return has_finite > 0


def mask_stop_logits_for_min_steps(
    *,
    policy_out: dict[str, torch.Tensor],
    active_flat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out_degrees_flat = policy_out["out_degrees"].view(-1)
    has_finite_edges = compute_has_finite_edges(
        edge_logits=policy_out["edge_logits"],
        out_degrees=out_degrees_flat,
    )
    ban_stop = active_flat & (out_degrees_flat > 0) & has_finite_edges
    stop_logits_flat = policy_out["stop_logits"].view(-1)
    masked_stop = stop_logits_flat.masked_fill(
        ban_stop,
        torch.tensor(
            float("-inf"),
            device=stop_logits_flat.device,
            dtype=stop_logits_flat.dtype,
        ),
    )
    patched = dict(policy_out)
    patched["stop_logits"] = masked_stop.view_as(policy_out["stop_logits"])
    return patched


def segment_mean_1d(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    ids = _normalize_segment_ids(
        values=values,
        segment_ids=segment_ids,
        num_segments=num_segments,
        op_name="segment_mean_1d",
    )
    if num_segments == 0:
        return torch.empty((0,), device=values.device, dtype=dtype)

    out = torch.zeros((num_segments,), device=values.device, dtype=torch.float32)
    counts = torch.zeros((num_segments,), device=values.device, dtype=torch.float32)
    if int(values.numel()) == 0:
        return out.to(dtype=dtype)

    finite_mask = torch.isfinite(values)
    if not bool(finite_mask.any().item()):
        return out.to(dtype=dtype)

    finite_ids = ids[finite_mask]
    finite_values = values[finite_mask].to(dtype=torch.float32)
    out.scatter_add_(0, finite_ids, finite_values)
    counts.scatter_add_(0, finite_ids, torch.ones_like(finite_values))
    mean = out / counts.clamp(min=1.0)
    mean = torch.where(counts > 0, mean, torch.zeros_like(mean))
    return mean.to(dtype=dtype)


def segment_logsumexp_1d(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    dtype: torch.dtype,
    ignore_non_finite: bool,
    empty_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = _normalize_segment_ids(
        values=values,
        segment_ids=segment_ids,
        num_segments=num_segments,
        op_name="segment_logsumexp_1d",
    )
    if num_segments == 0:
        empty = torch.empty((0,), device=values.device, dtype=dtype)
        return empty, torch.empty((0,), device=values.device, dtype=torch.bool)

    out = torch.full(
        (num_segments,),
        fill_value=float(empty_value),
        device=values.device,
        dtype=torch.float32,
    )
    has_values = torch.zeros((num_segments,), device=values.device, dtype=torch.bool)
    if int(values.numel()) == 0:
        return out.to(dtype=dtype), has_values

    finite_mask = torch.isfinite(values)
    if not ignore_non_finite and not bool(finite_mask.all().item()):
        raise ValueError(
            "segment_logsumexp_1d received non-finite values while ignore_non_finite=False."
        )
    considered_mask = (
        finite_mask
        if ignore_non_finite
        else torch.ones_like(finite_mask, dtype=torch.bool)
    )
    if not bool(considered_mask.any().item()):
        return out.to(dtype=dtype), has_values

    considered_ids = ids[considered_mask]
    considered_values = values[considered_mask].to(dtype=torch.float32)
    has_values.scatter_(0, considered_ids, True)

    neg_inf = float("-inf")
    max_per_segment = torch.full(
        (num_segments,), fill_value=neg_inf, device=values.device, dtype=torch.float32
    )
    max_per_segment.scatter_reduce_(
        0, considered_ids, considered_values, reduce="amax", include_self=True
    )
    safe_max = torch.where(
        has_values, max_per_segment, torch.zeros_like(max_per_segment)
    )

    shifted = torch.exp(considered_values - safe_max.index_select(0, considered_ids))
    sum_per_segment = torch.zeros(
        (num_segments,), device=values.device, dtype=torch.float32
    )
    sum_per_segment.scatter_add_(0, considered_ids, shifted)
    lse = safe_max + torch.log(
        sum_per_segment.clamp(min=torch.finfo(torch.float32).tiny)
    )
    out = torch.where(has_values, lse, out)
    return out.to(dtype=dtype), has_values


__all__ = [
    "compute_has_finite_edges",
    "mask_stop_logits_for_min_steps",
    "segment_logsumexp_1d",
    "segment_mean_1d",
]
