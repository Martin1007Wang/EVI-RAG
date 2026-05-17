from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.graph.segments import segment_log_softmax

if TYPE_CHECKING:
    from src.weaver.policy import PolicyOutput


@dataclass(frozen=True, slots=True)
class SampledAction:
    stop_rows: torch.Tensor
    stop_logprob: torch.Tensor
    forced_stop: torch.Tensor
    expand_rows: torch.Tensor
    expand_edge_ids: torch.Tensor
    expand_logprob: torch.Tensor


def sample_action(
    *,
    policy_out: PolicyOutput,
    temperature: float = 1.0,
) -> SampledAction:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    tensors = _policy_tensors(policy_out)
    _validate_policy_tensors(tensors)

    device = tensors.stop_logit.device
    dtype = tensors.stop_logit.dtype
    num_rows = int(tensors.stop_logit.numel())

    has_frontier = _has_frontier(
        row_ids=tensors.row_ids,
        num_rows=num_rows,
        device=device,
    )

    forced_rows = (~has_frontier).nonzero(as_tuple=False).flatten()
    active_rows = has_frontier.nonzero(as_tuple=False).flatten()

    stop_row_parts: list[torch.Tensor] = []
    stop_logprob_parts: list[torch.Tensor] = []
    forced_stop_parts: list[torch.Tensor] = []
    expand_row_parts: list[torch.Tensor] = []
    expand_edge_parts: list[torch.Tensor] = []
    expand_logprob_parts: list[torch.Tensor] = []

    if forced_rows.numel() > 0:
        stop_row_parts.append(forced_rows)
        stop_logprob_parts.append(torch.zeros(forced_rows.numel(), device=device, dtype=dtype))
        forced_stop_parts.append(torch.ones(forced_rows.numel(), device=device, dtype=torch.bool))

    if active_rows.numel() > 0:
        sampled = _sample_nonforced_rows(
            tensors=tensors,
            active_rows=active_rows,
            temperature=temperature,
        )
        if sampled.stop_rows.numel() > 0:
            stop_row_parts.append(sampled.stop_rows)
            stop_logprob_parts.append(sampled.stop_logprob)
            forced_stop_parts.append(torch.zeros(sampled.stop_rows.numel(), device=device, dtype=torch.bool))
        if sampled.expand_rows.numel() > 0:
            expand_row_parts.append(sampled.expand_rows)
            expand_edge_parts.append(sampled.expand_edge_ids)
            expand_logprob_parts.append(sampled.expand_logprob)

    return SampledAction(
        stop_rows=_cat_long(stop_row_parts, device=device),
        stop_logprob=_cat_float(stop_logprob_parts, device=device, dtype=dtype),
        forced_stop=_cat_bool(forced_stop_parts, device=device),
        expand_rows=_cat_long(expand_row_parts, device=device),
        expand_edge_ids=_cat_long(expand_edge_parts, device=device),
        expand_logprob=_cat_float(expand_logprob_parts, device=device, dtype=dtype),
    )


@dataclass(frozen=True, slots=True)
class _PolicyTensors:
    stop_logit: torch.Tensor
    stop_log_prob: torch.Tensor
    continue_log_prob: torch.Tensor
    edge_logits: torch.Tensor
    transition_log_prob: torch.Tensor
    row_ids: torch.Tensor
    edge_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SampledNonforced:
    stop_rows: torch.Tensor
    stop_logprob: torch.Tensor
    expand_rows: torch.Tensor
    expand_edge_ids: torch.Tensor
    expand_logprob: torch.Tensor


def _policy_tensors(policy_out: PolicyOutput) -> _PolicyTensors:
    stop_logit = policy_out.stop_logit.float().view(-1)
    stop_log_prob = policy_out.stop_log_prob.to(
        device=stop_logit.device,
        dtype=stop_logit.dtype,
    ).view(-1)
    continue_log_prob = policy_out.continue_log_prob.to(
        device=stop_logit.device,
        dtype=stop_logit.dtype,
    ).view(-1)
    edge_logits = policy_out.edge_logits.to(
        device=stop_logit.device,
        dtype=stop_logit.dtype,
    ).view(-1)
    transition_log_prob = policy_out.transition_log_prob.to(
        device=stop_logit.device,
        dtype=stop_logit.dtype,
    ).view(-1)
    row_ids = policy_out.frontier.row_ids.to(device=stop_logit.device, dtype=torch.long).view(-1)
    edge_ids = policy_out.frontier.edge_ids.to(device=stop_logit.device, dtype=torch.long).view(-1)
    return _PolicyTensors(
        stop_logit=stop_logit,
        stop_log_prob=stop_log_prob,
        continue_log_prob=continue_log_prob,
        edge_logits=edge_logits,
        transition_log_prob=transition_log_prob,
        row_ids=row_ids,
        edge_ids=edge_ids,
    )


def _validate_policy_tensors(tensors: _PolicyTensors) -> None:
    num_rows = int(tensors.stop_logit.numel())
    if tensors.stop_log_prob.numel() != num_rows:
        raise ValueError("stop_log_prob must have one value per state row.")
    if tensors.continue_log_prob.numel() != num_rows:
        raise ValueError("continue_log_prob must have one value per state row.")
    if tensors.row_ids.shape != tensors.edge_ids.shape:
        raise ValueError("frontier row_ids and edge_ids must have same shape.")
    if tensors.edge_logits.numel() != tensors.edge_ids.numel():
        raise ValueError("edge_logits must have one value per frontier edge.")
    if tensors.transition_log_prob.numel() != tensors.edge_ids.numel():
        raise ValueError("transition_log_prob must have one value per frontier edge.")
    if tensors.row_ids.numel() > 0:
        bad_rows = tensors.row_ids.lt(0) | tensors.row_ids.ge(num_rows)
        if bool(bad_rows.any()):
            raise ValueError("frontier.row_ids contains ids outside active row range.")
    for name, tensor in {
        "stop_logit": tensors.stop_logit,
        "stop_log_prob": tensors.stop_log_prob,
        "continue_log_prob": tensors.continue_log_prob,
        "edge_logits": tensors.edge_logits,
        "transition_log_prob": tensors.transition_log_prob,
    }.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite.")


def _has_frontier(
    *,
    row_ids: torch.Tensor,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    has_frontier = torch.zeros(int(num_rows), dtype=torch.bool, device=device)
    if row_ids.numel() > 0:
        has_frontier.scatter_(0, row_ids, torch.ones_like(row_ids, dtype=torch.bool))
    return has_frontier


def _sample_nonforced_rows(
    *,
    tensors: _PolicyTensors,
    active_rows: torch.Tensor,
    temperature: float,
) -> _SampledNonforced:
    device = tensors.stop_logit.device
    dtype = tensors.stop_logit.dtype
    stop_behavior_scores = tensors.stop_logit.index_select(0, active_rows) / temperature
    stop_draw = torch.sigmoid(stop_behavior_scores)
    stop_choice = torch.bernoulli(stop_draw).to(dtype=torch.bool)

    stop_rows = active_rows[stop_choice]
    stop_logprob = tensors.stop_log_prob.index_select(0, stop_rows) if stop_rows.numel() > 0 else torch.empty(0, device=device, dtype=dtype)

    continue_rows = active_rows[~stop_choice]
    if continue_rows.numel() == 0:
        return _SampledNonforced(
            stop_rows=stop_rows.to(device=device, dtype=torch.long),
            stop_logprob=stop_logprob.to(device=device, dtype=dtype),
            expand_rows=torch.empty(0, dtype=torch.long, device=device),
            expand_edge_ids=torch.empty(0, dtype=torch.long, device=device),
            expand_logprob=torch.empty(0, dtype=dtype, device=device),
        )

    active_frontier_mask = tensors.row_ids.unsqueeze(0).eq(continue_rows.unsqueeze(1)).any(dim=0)
    cont_row_ids = tensors.row_ids[active_frontier_mask]
    cont_edge_ids = tensors.edge_ids[active_frontier_mask]
    cont_edge_logits = tensors.edge_logits[active_frontier_mask] / temperature
    cont_positions, cont_edge_log_prob = _sample_segmented_edges(
        logits=cont_edge_logits,
        row_ids=cont_row_ids,
        num_rows=int(tensors.stop_logit.numel()),
    )
    sampled_positions = cont_positions.index_select(0, continue_rows)
    if bool(sampled_positions.lt(0).any()):
        raise RuntimeError("Rows without sampled edge action.")
    expand_edge_ids = cont_edge_ids.index_select(0, sampled_positions)
    expand_rows = continue_rows
    expand_logprob = (
        tensors.continue_log_prob.index_select(0, expand_rows)
        + cont_edge_log_prob.index_select(0, sampled_positions)
    )
    return _SampledNonforced(
        stop_rows=stop_rows.to(device=device, dtype=torch.long),
        stop_logprob=stop_logprob.to(device=device, dtype=dtype),
        expand_rows=expand_rows.to(device=device, dtype=torch.long),
        expand_edge_ids=expand_edge_ids.to(device=device, dtype=torch.long),
        expand_logprob=expand_logprob.to(device=device, dtype=dtype),
    )


def _sample_segmented_edges(
    *,
    logits: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.numel() == 0:
        return (
            torch.full((int(num_rows),), -1, dtype=torch.long, device=logits.device),
            logits.new_empty((0,)),
        )
    edge_log_prob = segment_log_softmax(
        logits,
        row_ids,
        num_segments=int(num_rows),
    )
    sampled_positions = _sample_gumbel_argmax_by_row(
        values=logits.detach(),
        row_ids=row_ids,
        num_rows=num_rows,
    )
    return sampled_positions, edge_log_prob


def _sample_gumbel_argmax_by_row(
    *,
    values: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    gumbel = -torch.empty_like(values).exponential_().log()
    scores = values + gumbel
    return _scatter_argmax_by_row(
        values=scores,
        row_ids=row_ids,
        num_rows=num_rows,
    )


def _scatter_argmax_by_row(
    *,
    values: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    maxima = torch.full(
        (int(num_rows),),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    maxima.scatter_reduce_(
        0,
        row_ids,
        values,
        reduce="amax",
        include_self=True,
    )
    is_max = values.eq(maxima.index_select(0, row_ids))
    positions = torch.arange(
        values.numel(),
        dtype=torch.long,
        device=values.device,
    )
    sentinel = torch.full_like(positions, values.numel())
    candidates = torch.where(is_max, positions, sentinel)
    out = torch.full(
        (int(num_rows),),
        values.numel(),
        dtype=torch.long,
        device=values.device,
    )
    out.scatter_reduce_(
        0,
        row_ids,
        candidates,
        reduce="amin",
        include_self=True,
    )
    return torch.where(
        out.eq(values.numel()),
        torch.full_like(out, -1),
        out,
    )


def _cat_long(
    values: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = [value for value in values if value.numel() > 0]
    if not values:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(values, dim=0).to(device=device, dtype=torch.long)


def _cat_float(
    values: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [value for value in values if value.numel() > 0]
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(values, dim=0).to(device=device, dtype=dtype)


def _cat_bool(
    values: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = [value for value in values if value.numel() > 0]
    if not values:
        return torch.empty(0, dtype=torch.bool, device=device)
    return torch.cat(values, dim=0).to(device=device, dtype=torch.bool)


__all__ = ["SampledAction", "sample_action"]
