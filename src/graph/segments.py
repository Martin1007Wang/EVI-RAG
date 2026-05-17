from __future__ import annotations

import torch

try:
    from torch_scatter import scatter_max
except ModuleNotFoundError:
    def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        *,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise ValueError("fallback scatter_max only supports dim=0.")
        src = src.view(-1)
        index = index.to(device=src.device, dtype=torch.long).view(-1)
        if src.shape != index.shape:
            raise ValueError(
                "src and index must have matching shape: "
                f"{tuple(src.shape)} != {tuple(index.shape)}."
            )
        if dim_size is not None:
            size = int(dim_size)
        elif index.numel() == 0:
            size = 0
        else:
            size = int(index.max().item()) + 1
        values = src.new_full((size,), -torch.inf)
        if src.numel() > 0:
            values.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
        positions = torch.full((size,), -1, dtype=torch.long, device=src.device)
        if src.numel() > 0:
            pos = torch.arange(src.numel(), dtype=torch.long, device=src.device)
            is_max = src.eq(values.index_select(0, index))
            candidates = torch.where(is_max, pos, torch.full_like(pos, src.numel()))
            positions.scatter_reduce_(0, index, candidates, reduce="amin", include_self=True)
            positions = torch.where(
                positions.eq(src.numel()),
                torch.full_like(positions, -1),
                positions,
            )
        return values, positions


def _validate_segmented_inputs(
    *,
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    active_segments: torch.Tensor | None = None,
) -> None:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape={tuple(logits.shape)}.")
    if segment_ids.ndim != 1:
        raise ValueError(
            f"segment_ids must be 1D, got shape={tuple(segment_ids.shape)}."
        )
    if logits.numel() != segment_ids.numel():
        raise ValueError(
            "logits and segment_ids length mismatch: "
            f"{logits.numel()} != {segment_ids.numel()}."
        )
    if num_segments < 0:
        raise ValueError(f"num_segments must be non-negative, got {num_segments}.")
    if segment_ids.numel() > 0:
        if bool((segment_ids < 0).any()):
            raise ValueError("segment_ids contains negative segment ids.")
        if bool((segment_ids >= int(num_segments)).any()):
            raise ValueError(
                f"segment_ids contains ids outside [0, {int(num_segments)})."
            )
    if active_segments is not None:
        if active_segments.ndim != 1:
            raise ValueError(
                f"active_segments must be 1D, got shape={tuple(active_segments.shape)}."
            )
        if active_segments.numel() > 0:
            if bool((active_segments < 0).any()):
                raise ValueError("active_segments contains negative segment ids.")
            if bool((active_segments >= int(num_segments)).any()):
                raise ValueError(
                    f"active_segments contains ids outside [0, {int(num_segments)})."
                )


def scatter_log_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute log-softmax independently inside each segment.
    Returns:
        log_probs[i] = logits[i] - logsumexp(logits[j])
        where segment_ids[j] == segment_ids[i].
    """
    segment_ids = segment_ids.to(device=logits.device, dtype=torch.long)
    _validate_segmented_inputs(
        logits=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
    )
    if logits.numel() == 0:
        return logits.new_empty((0,))
    log_z = segment_logsumexp(
        values=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
    )
    return subtract_log_normalizer(
        values=logits,
        log_normalizer=log_z.index_select(0, segment_ids),
    )


def segment_log_softmax(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int | None = None,
) -> torch.Tensor:
    if num_segments is None:
        if segment_ids.numel() == 0:
            num_segments = 0
        else:
            num_segments = int(segment_ids.max().item()) + 1
    return scatter_log_softmax(
        values,
        segment_ids,
        int(num_segments),
    )


def subtract_log_normalizer(
    *,
    values: torch.Tensor,
    log_normalizer: torch.Tensor,
    fallback: float = -torch.inf,
) -> torch.Tensor:
    """
    Return values - log_normalizer without materializing inf - inf.

    Non-finite normalizers represent empty/all-masked log-domain groups. Those
    outputs are assigned fallback, which keeps both forward values and backward
    gradients finite for masked logits.
    """
    if values.shape != log_normalizer.shape:
        raise ValueError(
            "values and log_normalizer must have matching shapes: "
            f"{tuple(values.shape)} != {tuple(log_normalizer.shape)}."
        )
    finite_normalizer = torch.isfinite(log_normalizer)
    safe_normalizer = torch.where(
        finite_normalizer,
        log_normalizer,
        torch.zeros_like(log_normalizer),
    )
    return torch.where(
        finite_normalizer,
        values - safe_normalizer,
        values.new_full(values.shape, float(fallback)),
    )


def segment_logsumexp(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_full((int(num_segments),), -torch.inf)

    max_values = values.new_full((int(num_segments),), -torch.inf).scatter_reduce(
        0,
        segment_ids,
        values,
        reduce="amax",
        include_self=True,
    )
    max_values = max_values.detach()
    selected_max = max_values.index_select(0, segment_ids)
    finite_max = torch.isfinite(selected_max)
    safe_selected_max = torch.where(
        finite_max,
        selected_max,
        torch.zeros_like(selected_max),
    )
    shifted = subtract_log_normalizer(
        values=values,
        log_normalizer=safe_selected_max,
        fallback=-torch.inf,
    ).exp()
    shifted = torch.where(finite_max, shifted, torch.zeros_like(shifted))
    sums = values.new_zeros((int(num_segments),)).scatter_add(0, segment_ids, shifted)
    return max_values + sums.clamp_min(torch.finfo(values.dtype).tiny).log()


def segment_softmax(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values

    segment_ids = segment_ids.to(device=values.device, dtype=torch.long).view(-1)
    log_probs = scatter_log_softmax(
        values,
        segment_ids,
        num_segments=int(num_segments),
    )
    return torch.where(
        torch.isfinite(log_probs),
        log_probs.exp(),
        torch.zeros_like(log_probs),
    )


def segment_has_any(
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or segment_ids.device
    if segment_ids.numel() == 0:
        return torch.zeros(int(num_segments), dtype=torch.bool, device=device)
    counts = torch.bincount(
        segment_ids.to(device=device, dtype=torch.long),
        minlength=int(num_segments),
    )
    return counts.gt(0)


def segment_topk_positions(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    k: int,
    active_segments: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return up to k positions per segment without moving segment ids to Python.
    """
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    values = values.view(-1)
    segment_ids = segment_ids.to(device=values.device, dtype=torch.long).view(-1)
    _validate_segmented_inputs(
        logits=values,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
        active_segments=active_segments,
    )
    if values.numel() == 0:
        return segment_ids.new_empty((0,))

    work = values.to(dtype=torch.float32)
    if active_segments is not None:
        active_segments = active_segments.to(device=values.device, dtype=torch.long).view(-1)
        active_mask = torch.zeros(int(num_segments), dtype=torch.bool, device=values.device)
        if active_segments.numel() > 0:
            active_mask[active_segments] = True
        work = torch.where(
            active_mask.index_select(0, segment_ids),
            work,
            work.new_full(work.shape, -torch.inf),
        )

    selected: list[torch.Tensor] = []
    for _ in range(k):
        max_values, positions = scatter_max(
            work,
            segment_ids,
            dim=0,
            dim_size=int(num_segments),
        )
        valid = (
            positions.ge(0)
            & positions.lt(work.numel())
            & torch.isfinite(max_values)
        )
        if bool(valid.any()):
            valid_segment_ids = segment_ids.index_select(0, positions[valid])
            valid_positions = valid.nonzero(as_tuple=False).view(-1)
            valid[valid_positions] &= valid_segment_ids.eq(valid_positions)
        current = positions[valid]
        if current.numel() == 0:
            break
        selected.append(current)
        work = work.clone()
        work[current] = -torch.inf

    if not selected:
        return segment_ids.new_empty((0,))
    return torch.cat(selected, dim=0)


def _sample_positions_by_segment(
    *,
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    active_segments: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if active_segments.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    if logits.numel() == 0:
        raise RuntimeError("Cannot sample from an empty candidate list.")
    gumbel = -torch.empty_like(logits).exponential_().log()
    scores = logits / float(temperature) + gumbel
    positions_by_segment = scatter_max(
        scores,
        segment_ids,
        dim=0,
        dim_size=int(num_segments),
    )[1]
    positions = positions_by_segment.index_select(0, active_segments)
    if bool((positions < 0).any()):
        missing = active_segments[positions < 0]
        raise RuntimeError(
            f"Some active segments have no candidates: {missing.tolist()}."
        )
    return positions


def sample_segmented_categorical(
    *,
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    active_segments: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one item from each active segment.
    Sampling distribution:
        softmax(logits / temperature)
    Returned log-probability:
        log softmax(logits), without temperature scaling.
    Temperature affects sampling only. Returned log-probs are computed from
    the untempered logits.
    """
    segment_ids = segment_ids.to(device=logits.device, dtype=torch.long)
    active_segments = active_segments.to(device=logits.device, dtype=torch.long)
    _validate_segmented_inputs(
        logits=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
        active_segments=active_segments,
    )
    sampled_positions = _sample_positions_by_segment(
        logits=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
        active_segments=active_segments,
        temperature=float(temperature),
    )
    if sampled_positions.numel() == 0:
        return sampled_positions, logits.new_empty((0,))
    target_log_probs = scatter_log_softmax(
        logits,
        segment_ids,
        num_segments=int(num_segments),
    )
    return sampled_positions, target_log_probs.index_select(0, sampled_positions)


def sample_segmented_positions(
    *,
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    active_segments: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Sample one item position from each active segment without computing log-probs.
    """
    segment_ids = segment_ids.to(device=logits.device, dtype=torch.long)
    active_segments = active_segments.to(device=logits.device, dtype=torch.long)
    _validate_segmented_inputs(
        logits=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
        active_segments=active_segments,
    )
    return _sample_positions_by_segment(
        logits=logits,
        segment_ids=segment_ids,
        num_segments=int(num_segments),
        active_segments=active_segments,
        temperature=float(temperature),
    )


__all__ = [
    "segment_log_softmax",
    "scatter_log_softmax",
    "sample_segmented_categorical",
    "sample_segmented_positions",
    "segment_has_any",
    "segment_logsumexp",
    "segment_softmax",
    "segment_topk_positions",
    "subtract_log_normalizer",
]
