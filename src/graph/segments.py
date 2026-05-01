from __future__ import annotations

import torch
from torch_scatter import scatter_max


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
    log_z = torch.stack(
        [
            torch.logsumexp(logits[segment_ids == segment_id], dim=0)
            if bool((segment_ids == segment_id).any())
            else logits.new_full((), -torch.inf)
            for segment_id in range(int(num_segments))
        ],
        dim=0,
    )
    return logits - log_z.index_select(0, segment_ids)


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
    "scatter_log_softmax",
    "sample_segmented_categorical",
    "sample_segmented_positions",
]
