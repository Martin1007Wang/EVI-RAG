from __future__ import annotations

from dataclasses import dataclass

import torch

CONTINUE_ACTION = 0
STOP_ACTION = 1

STOP_OPTION_INDEX = 0
EXPAND_OPTION_INDEX = 1


@dataclass(frozen=True, slots=True)
class ActionSample:
    """
    One sampled action per rollout row.

    action_type:
        [B], 0 for Continue/Expand(edge), 1 for Stop.

    chosen_edges:
        [B], physical edge ids for Continue actions, -1 otherwise.

    target_log_prob:
        [B], log P_target(action | state).

        For Stop:
            log P_target(Stop | s)

        For Expand(edge):
            log P_target(Expand(edge) | s)

        For environment-forced Stop, e.g. inactive graph, no frontier, or
        exhausted budget, this is zero. Forced Stop should not train the Stop
        scorer as evidence sufficiency.
    """

    action_type: torch.Tensor
    chosen_edges: torch.Tensor
    target_log_prob: torch.Tensor


def sample_policy_actions(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    active: torch.Tensor,
    can_expand: torch.Tensor,
    temperature: float,
    batch_size: int,
) -> ActionSample:
    """
    Sample one behavior action per graph.

    Behavior distribution uses temperature over the flat action space:

        A(s) = {Stop} union {Expand(e): e in C(s)}

    Returned log-probabilities are always from the untempered target policy:

        log P_target(Stop | s)
        log P_target(Expand(e) | s)
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    device = stop_logits.device

    stop_logits = stop_logits.to(device=device, dtype=torch.float32).view(batch_size)
    edge_logits = edge_logits.to(device=device, dtype=torch.float32).view(-1)
    candidate_edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    candidate_batch_ids = candidate_batch_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)
    active = active.to(device=device, dtype=torch.bool).view(batch_size)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_inputs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_edge_ids=candidate_edge_ids,
        candidate_batch_ids=candidate_batch_ids,
        active=active,
        can_expand=can_expand,
        batch_size=batch_size,
        device=device,
    )

    target_stop_logp, target_edge_logp = action_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
    )

    behavior_stop_logp, behavior_edge_logp = action_log_probs(
        stop_logits=stop_logits / float(temperature),
        edge_logits=edge_logits / float(temperature),
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
    )

    action_type = torch.full(
        (batch_size,),
        STOP_ACTION,
        dtype=torch.long,
        device=device,
    )
    chosen_edges = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=device,
    )
    target_log_prob = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=device,
    )

    learnable_graphs = (active & can_expand).nonzero(as_tuple=False).flatten()
    if learnable_graphs.numel() == 0:
        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    _sample_flat_actions(
        active_graphs=learnable_graphs,
        behavior_stop_logp=behavior_stop_logp,
        behavior_edge_logp=behavior_edge_logp,
        target_stop_logp=target_stop_logp,
        target_edge_logp=target_edge_logp,
        candidate_edge_ids=candidate_edge_ids,
        candidate_batch_ids=candidate_batch_ids,
        action_type=action_type,
        chosen_edges=chosen_edges,
        target_log_prob=target_log_prob,
    )

    return ActionSample(
        action_type=action_type,
        chosen_edges=chosen_edges,
        target_log_prob=target_log_prob,
    )


def action_log_probs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return flat action log-probs over {Stop} union {Expand(edge)}.

    stop_logp:
        [B], log P(Stop | s).

    edge_logp:
        [C], log P(Expand(edge) | s), normalized jointly with Stop inside each
        candidate graph.

    For graphs where can_expand=False:
        P(Stop)=1 and candidate edge log-probs belonging to those graphs are
        -inf.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    device = stop_logits.device

    stop_logits = stop_logits.to(device=device, dtype=torch.float32).view(batch_size)
    edge_logits = edge_logits.to(device=device, dtype=torch.float32).view(-1)
    candidate_batch_ids = candidate_batch_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_option_inputs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
        device=device,
    )

    if edge_logits.numel() == 0:
        stop_logp = torch.zeros_like(stop_logits)
        return stop_logp, stop_logits.new_empty((0,), dtype=torch.float32)

    log_z = _flat_action_log_normalizer(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
    )
    stop_logp = torch.where(
        can_expand,
        stop_logits - log_z,
        torch.zeros_like(stop_logits),
    )
    edge_logp = edge_logits - log_z.index_select(0, candidate_batch_ids)

    edge_allowed = can_expand.index_select(0, candidate_batch_ids)
    edge_logp = torch.where(
        edge_allowed,
        edge_logp,
        edge_logp.new_full(edge_logp.shape, -torch.inf),
    )

    return stop_logp, edge_logp


def action_probs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
        stop_prob:       [B]
        expand_prob_sum: [B], sum_e P(Expand(e) | s)
        edge_prob:       [C], action-level P(Expand(edge) | s)
    """
    stop_logp, edge_logp = action_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=int(batch_size),
    )

    stop_prob = stop_logp.exp()
    edge_prob = torch.where(
        torch.isfinite(edge_logp),
        edge_logp.exp(),
        edge_logp.new_zeros(edge_logp.shape),
    )
    expand_prob = torch.zeros(
        int(batch_size),
        dtype=edge_prob.dtype,
        device=edge_prob.device,
    )
    if edge_prob.numel() > 0:
        expand_prob = torch.bincount(
            candidate_batch_ids.to(device=edge_prob.device, dtype=torch.long),
            weights=edge_prob,
            minlength=int(batch_size),
        ).to(dtype=edge_prob.dtype)

    return stop_prob, expand_prob, edge_prob


def option_action_probs(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward-compatible wrapper for the removed Stop/Expand conditional policy.

    Return:
        stop_prob:   [B]
        expand_prob: [B], sum_e P(Expand(e) | s)
        edge_prob:   [C], action-level P(Expand(edge) | s)
    """
    del expand_logits
    return action_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=int(batch_size),
    )


def option_action_log_probs(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-compatible wrapper returning [Stop, Expand-sum] log-probs plus
    action-level edge log-probs.
    """
    del expand_logits
    stop_logp, edge_logp = action_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=can_expand,
        batch_size=int(batch_size),
    )
    expand_prob = torch.zeros_like(stop_logp)
    if edge_logp.numel() > 0:
        edge_prob = torch.where(
            torch.isfinite(edge_logp),
            edge_logp.exp(),
            edge_logp.new_zeros(edge_logp.shape),
        )
        expand_prob = torch.bincount(
            candidate_batch_ids.to(device=edge_logp.device, dtype=torch.long),
            weights=edge_prob,
            minlength=int(batch_size),
        ).to(dtype=stop_logp.dtype)
    expand_logp = torch.where(
        expand_prob.gt(0.0),
        expand_prob.clamp_min(torch.finfo(stop_logp.dtype).tiny).log(),
        stop_logp.new_full(stop_logp.shape, -torch.inf),
    )
    return torch.stack([stop_logp, expand_logp], dim=-1), edge_logp


def _sample_flat_actions(
    *,
    active_graphs: torch.Tensor,
    behavior_stop_logp: torch.Tensor,
    behavior_edge_logp: torch.Tensor,
    target_stop_logp: torch.Tensor,
    target_edge_logp: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    action_type: torch.Tensor,
    chosen_edges: torch.Tensor,
    target_log_prob: torch.Tensor,
) -> None:
    """
    Sample Stop or one Expand(edge) action per active rollout row.
    """
    for graph_id_tensor in active_graphs:
        graph_id = int(graph_id_tensor.item())
        candidate_pos = (
            candidate_batch_ids.eq(graph_id).nonzero(as_tuple=False).view(-1)
        )

        if candidate_pos.numel() == 0:
            raise RuntimeError(
                "Sampled Expand for rollout row with no candidate edges: "
                f"rollout row id={graph_id}."
            )

        logits = torch.cat(
            [
                behavior_stop_logp[graph_id].view(1),
                behavior_edge_logp.index_select(0, candidate_pos),
            ],
            dim=0,
        )
        local_choice = torch.distributions.Categorical(logits=logits).sample()
        local_idx = int(local_choice.item())
        if local_idx == 0:
            target_log_prob[graph_id] = target_stop_logp[graph_id]
            continue

        chosen_pos = candidate_pos[local_idx - 1]

        action_type[graph_id] = CONTINUE_ACTION
        chosen_edges[graph_id] = candidate_edge_ids[chosen_pos]
        target_log_prob[graph_id] = target_edge_logp[chosen_pos]


def _flat_action_log_normalizer(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    log_z = stop_logits.clone()
    if edge_logits.numel() == 0:
        return log_z
    edge_log_z = torch.full_like(stop_logits, -torch.inf)
    for graph_id in range(int(batch_size)):
        mask = candidate_batch_ids.eq(graph_id)
        if bool(mask.any()):
            edge_log_z[graph_id] = torch.logsumexp(edge_logits[mask], dim=0)
    edge_log_z = torch.where(
        can_expand,
        edge_log_z,
        edge_log_z.new_full(edge_log_z.shape, -torch.inf),
    )
    return torch.logaddexp(stop_logits, edge_log_z)


def _validate_inputs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    active: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    expected_shape = (int(batch_size),)

    for name, tensor in {
        "stop_logits": stop_logits,
        "active": active,
        "can_expand": can_expand,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )

    if bool((can_expand & ~active).any()):
        raise ValueError("can_expand cannot be true for inactive graphs.")

    _validate_candidate_tensors(
        edge_logits=edge_logits,
        candidate_edge_ids=candidate_edge_ids,
        candidate_batch_ids=candidate_batch_ids,
        batch_size=batch_size,
        device=device,
    )


def _validate_option_inputs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    expected_shape = (int(batch_size),)

    for name, tensor in {
        "stop_logits": stop_logits,
        "can_expand": can_expand,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )

    if edge_logits.ndim != 1:
        raise ValueError(f"edge_logits must be 1D, got {tuple(edge_logits.shape)}.")
    if candidate_batch_ids.ndim != 1:
        raise ValueError(
            f"candidate_batch_ids must be 1D, got {tuple(candidate_batch_ids.shape)}."
        )
    if edge_logits.numel() != candidate_batch_ids.numel():
        raise ValueError(
            "edge_logits and candidate_batch_ids must have matching length: "
            f"{edge_logits.numel()} != {candidate_batch_ids.numel()}."
        )
    if candidate_batch_ids.device != device:
        raise ValueError(
            f"candidate_batch_ids is on {candidate_batch_ids.device}, expected {device}."
        )
    if candidate_batch_ids.dtype != torch.long:
        raise TypeError(
            f"candidate_batch_ids must be torch.long, got {candidate_batch_ids.dtype}."
        )

    _validate_graph_ids(
        candidate_batch_ids,
        batch_size=batch_size,
        name="candidate_batch_ids",
    )


def _validate_candidate_tensors(
    *,
    edge_logits: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    if edge_logits.device != device:
        raise ValueError(f"edge_logits is on {edge_logits.device}, expected {device}.")
    if candidate_edge_ids.device != device:
        raise ValueError(
            f"candidate_edge_ids is on {candidate_edge_ids.device}, expected {device}."
        )
    if candidate_batch_ids.device != device:
        raise ValueError(
            f"candidate_batch_ids is on {candidate_batch_ids.device}, expected {device}."
        )

    if edge_logits.ndim != 1:
        raise ValueError(f"edge_logits must be 1D, got {tuple(edge_logits.shape)}.")
    if candidate_edge_ids.ndim != 1:
        raise ValueError(
            f"candidate_edge_ids must be 1D, got {tuple(candidate_edge_ids.shape)}."
        )
    if candidate_batch_ids.ndim != 1:
        raise ValueError(
            f"candidate_batch_ids must be 1D, got {tuple(candidate_batch_ids.shape)}."
        )

    if candidate_edge_ids.dtype != torch.long:
        raise TypeError(
            f"candidate_edge_ids must be torch.long, got {candidate_edge_ids.dtype}."
        )
    if candidate_batch_ids.dtype != torch.long:
        raise TypeError(
            f"candidate_batch_ids must be torch.long, got {candidate_batch_ids.dtype}."
        )

    if not (
        edge_logits.numel() == candidate_edge_ids.numel() == candidate_batch_ids.numel()
    ):
        raise ValueError(
            "candidate tensors must have matching length: "
            f"edge_logits={edge_logits.numel()}, "
            f"candidate_edge_ids={candidate_edge_ids.numel()}, "
            f"candidate_batch_ids={candidate_batch_ids.numel()}."
        )

    _validate_graph_ids(
        candidate_batch_ids,
        batch_size=batch_size,
        name="candidate_batch_ids",
    )


def _validate_graph_ids(
    graph_ids: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> None:
    if graph_ids.numel() == 0:
        return

    if bool((graph_ids < 0).any()):
        raise ValueError(f"{name} contains negative graph ids.")
    if bool((graph_ids >= int(batch_size)).any()):
        raise ValueError(f"{name} contains ids outside [0, {int(batch_size)}).")


__all__ = [
    "CONTINUE_ACTION",
    "STOP_ACTION",
    "STOP_OPTION_INDEX",
    "EXPAND_OPTION_INDEX",
    "ActionSample",
    "action_log_probs",
    "action_probs",
    "option_action_log_probs",
    "option_action_probs",
    "sample_policy_actions",
]
