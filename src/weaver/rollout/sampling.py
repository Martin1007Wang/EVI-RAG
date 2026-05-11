from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_scatter import scatter_max

from src.graph.segments import segment_logsumexp

CONTINUE_ACTION = 0
STOP_ACTION = 1

@dataclass(frozen=True, slots=True)
class ActionSample:
    """
    One sampled action per rollout row.

    action_type:
        [B], 0 for Continue/Expand(edge), 1 for Stop.

    chosen_edges:
        [B], physical edge ids for Continue actions, -1 otherwise.

    target_log_prob:
        [B], log P_target(action | state). "target" means the untempered
        policy probability used for training traces; it is not a target network
        or EMA teacher.

        For Stop:
            log P_target(Stop | s)

        For Expand(edge):
            log P_target(Expand(edge) | s)

        For environment-forced Stop, e.g. inactive graph, no frontier, or
        exhausted budget, this is zero. Forced Stop should not train the Stop
        stop head as evidence sufficiency.
    """

    action_type: torch.Tensor
    chosen_edges: torch.Tensor
    target_log_prob: torch.Tensor


def sample_action_for_generation(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    active: torch.Tensor,
    can_expand: torch.Tensor,
    temperature: float,
    batch_size: int,
) -> ActionSample:
    """
    Sample one behavior action per graph.

    Behavior sampling applies temperature to the learned Stop and frontier
    edge logits, then samples from the action softmax.

    Returned log-probabilities are always from the untempered policy. The
    historical "target" name here does not mean target network or EMA teacher:

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
    frontier_edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
    frontier_batch_ids = frontier_batch_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)
    active = active.to(device=device, dtype=torch.bool).view(batch_size)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_inputs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        frontier_edge_ids=frontier_edge_ids,
        frontier_batch_ids=frontier_batch_ids,
        active=active,
        can_expand=can_expand,
        batch_size=batch_size,
        device=device,
    )

    target_stop_logp, target_edge_logp = action_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        frontier_batch_ids=frontier_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
    )

    with torch.no_grad():
        behavior_stop_logp, behavior_edge_logp = action_log_probs(
            stop_logits=stop_logits.detach() / float(temperature),
            edge_logits=edge_logits.detach() / float(temperature),
            frontier_batch_ids=frontier_batch_ids,
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

    _sample_hazard_actions(
        active_graphs=learnable_graphs,
        behavior_stop_logp=behavior_stop_logp,
        behavior_edge_logp=behavior_edge_logp,
        target_stop_logp=target_stop_logp,
        target_edge_logp=target_edge_logp,
        frontier_edge_ids=frontier_edge_ids,
        frontier_batch_ids=frontier_batch_ids,
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
    frontier_batch_ids: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return action log-probs.

    edge_logits and stop_logits are learned action logits normalized together.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    device = stop_logits.device

    stop_logits = stop_logits.to(device=device, dtype=torch.float32).view(batch_size)
    edge_logits = edge_logits.to(device=device, dtype=torch.float32).view(-1)
    frontier_batch_ids = frontier_batch_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_option_inputs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        frontier_batch_ids=frontier_batch_ids,
        can_expand=can_expand,
        batch_size=batch_size,
        device=device,
    )

    if edge_logits.numel() == 0:
        return torch.zeros_like(stop_logits), stop_logits.new_empty((0,), dtype=torch.float32)

    edge_log_z = segment_logsumexp(
        values=edge_logits,
        segment_ids=frontier_batch_ids,
        num_segments=int(batch_size),
    )
    action_log_z = torch.logaddexp(stop_logits, edge_log_z)
    stop_logp = torch.where(
        can_expand,
        stop_logits - action_log_z,
        torch.zeros_like(stop_logits),
    )
    edge_logp = edge_logits - action_log_z.index_select(0, frontier_batch_ids)

    edge_allowed = can_expand.index_select(0, frontier_batch_ids)
    edge_logp = torch.where(
        edge_allowed,
        edge_logp,
        edge_logp.new_full(edge_logp.shape, -torch.inf),
    )
    prob_sum = stop_logp.exp()
    finite_edge_prob = torch.where(
        torch.isfinite(edge_logp),
        edge_logp.exp(),
        edge_logp.new_zeros(edge_logp.shape),
    )
    prob_sum = prob_sum.scatter_add(0, frontier_batch_ids, finite_edge_prob)
    if not bool(torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1.0e-5, rtol=1.0e-5)):
        raise RuntimeError("Action probabilities must sum to 1 for every state.")

    return stop_logp, edge_logp


def action_probs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
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
        frontier_batch_ids=frontier_batch_ids,
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
            frontier_batch_ids.to(device=edge_prob.device, dtype=torch.long),
            weights=edge_prob,
            minlength=int(batch_size),
        ).to(dtype=edge_prob.dtype)

    return stop_prob, expand_prob, edge_prob


def stop_continue_log_probs(
    *,
    stop_logits: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Return [log P(Stop | s), log P(Continue | s)] under the stop hazard.
    """
    stop_logits = stop_logits.to(device=stop_logits.device, dtype=torch.float32).view(
        int(batch_size)
    )
    can_expand = can_expand.to(device=stop_logits.device, dtype=torch.bool).view(
        int(batch_size)
    )
    stop_logp = torch.where(
        can_expand,
        F.logsigmoid(stop_logits),
        torch.zeros_like(stop_logits),
    )
    continue_logp = torch.where(
        can_expand,
        F.logsigmoid(-stop_logits),
        stop_logp.new_full(stop_logp.shape, -torch.inf),
    )
    return torch.stack([stop_logp, continue_logp], dim=-1)


def _sample_hazard_actions(
    *,
    active_graphs: torch.Tensor,
    behavior_stop_logp: torch.Tensor,
    behavior_edge_logp: torch.Tensor,
    target_stop_logp: torch.Tensor,
    target_edge_logp: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    action_type: torch.Tensor,
    chosen_edges: torch.Tensor,
    target_log_prob: torch.Tensor,
) -> None:
    """
    Sample Stop or one Expand(edge) action per active rollout row.
    """
    device = behavior_stop_logp.device
    batch_size = int(behavior_stop_logp.numel())
    active_graphs = active_graphs.to(device=device, dtype=torch.long).view(-1)
    if active_graphs.numel() == 0:
        return

    frontier_batch_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
    frontier_edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)

    edge_active = torch.zeros(batch_size, dtype=torch.bool, device=device)
    edge_active[active_graphs] = True
    edge_option_pos = edge_active.index_select(0, frontier_batch_ids).nonzero(
        as_tuple=False
    ).view(-1)

    option_graph_ids = torch.cat(
        [
            active_graphs,
            frontier_batch_ids.index_select(0, edge_option_pos),
        ],
        dim=0,
    )
    option_action_type = torch.cat(
        [
            torch.full_like(active_graphs, STOP_ACTION),
            torch.full(
                (edge_option_pos.numel(),),
                CONTINUE_ACTION,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=0,
    )
    option_edge_ids = torch.cat(
        [
            torch.full_like(active_graphs, -1),
            frontier_edge_ids.index_select(0, edge_option_pos),
        ],
        dim=0,
    )
    option_target_logp = torch.cat(
        [
            target_stop_logp.index_select(0, active_graphs),
            target_edge_logp.index_select(0, edge_option_pos),
        ],
        dim=0,
    )
    option_behavior_logp = torch.cat(
        [
            behavior_stop_logp.index_select(0, active_graphs),
            behavior_edge_logp.index_select(0, edge_option_pos),
        ],
        dim=0,
    )

    gumbel = -torch.empty_like(option_behavior_logp).exponential_().log()
    sampled_by_graph = scatter_max(
        option_behavior_logp + gumbel,
        option_graph_ids,
        dim=0,
        dim_size=batch_size,
    )[1]
    sampled_pos = sampled_by_graph.index_select(0, active_graphs)
    if bool((sampled_pos < 0).any()):
        missing = active_graphs[sampled_pos < 0]
        raise RuntimeError(
            "Some active rollout rows had no action options: "
            f"rollout row ids={missing.tolist()}."
        )

    sampled_action_type = option_action_type.index_select(0, sampled_pos)
    sampled_edge_ids = option_edge_ids.index_select(0, sampled_pos)

    action_type[active_graphs] = sampled_action_type
    chosen_edges[active_graphs] = sampled_edge_ids
    target_log_prob[active_graphs] = option_target_logp.index_select(0, sampled_pos)


def _validate_inputs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
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

    _validate_frontier_tensors(
        edge_logits=edge_logits,
        frontier_edge_ids=frontier_edge_ids,
        frontier_batch_ids=frontier_batch_ids,
        batch_size=batch_size,
        device=device,
    )


def _validate_option_inputs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
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
    if frontier_batch_ids.ndim != 1:
        raise ValueError(
            f"frontier_batch_ids must be 1D, got {tuple(frontier_batch_ids.shape)}."
        )
    if edge_logits.numel() != frontier_batch_ids.numel():
        raise ValueError(
            "edge_logits and frontier_batch_ids must have matching length: "
            f"{edge_logits.numel()} != {frontier_batch_ids.numel()}."
        )
    if frontier_batch_ids.device != device:
        raise ValueError(
            f"frontier_batch_ids is on {frontier_batch_ids.device}, expected {device}."
        )
    if frontier_batch_ids.dtype != torch.long:
        raise TypeError(
            f"frontier_batch_ids must be torch.long, got {frontier_batch_ids.dtype}."
        )

    _validate_graph_ids(
        frontier_batch_ids,
        batch_size=batch_size,
        name="frontier_batch_ids",
    )


def _validate_frontier_tensors(
    *,
    edge_logits: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    if edge_logits.device != device:
        raise ValueError(f"edge_logits is on {edge_logits.device}, expected {device}.")
    if frontier_edge_ids.device != device:
        raise ValueError(
            f"frontier_edge_ids is on {frontier_edge_ids.device}, expected {device}."
        )
    if frontier_batch_ids.device != device:
        raise ValueError(
            f"frontier_batch_ids is on {frontier_batch_ids.device}, expected {device}."
        )

    if edge_logits.ndim != 1:
        raise ValueError(f"edge_logits must be 1D, got {tuple(edge_logits.shape)}.")
    if frontier_edge_ids.ndim != 1:
        raise ValueError(
            f"frontier_edge_ids must be 1D, got {tuple(frontier_edge_ids.shape)}."
        )
    if frontier_batch_ids.ndim != 1:
        raise ValueError(
            f"frontier_batch_ids must be 1D, got {tuple(frontier_batch_ids.shape)}."
        )

    if frontier_edge_ids.dtype != torch.long:
        raise TypeError(
            f"frontier_edge_ids must be torch.long, got {frontier_edge_ids.dtype}."
        )
    if frontier_batch_ids.dtype != torch.long:
        raise TypeError(
            f"frontier_batch_ids must be torch.long, got {frontier_batch_ids.dtype}."
        )

    if not (
        edge_logits.numel() == frontier_edge_ids.numel() == frontier_batch_ids.numel()
    ):
        raise ValueError(
            "frontier tensors must have matching length: "
            f"edge_logits={edge_logits.numel()}, "
            f"frontier_edge_ids={frontier_edge_ids.numel()}, "
            f"frontier_batch_ids={frontier_batch_ids.numel()}."
        )

    _validate_graph_ids(
        frontier_batch_ids,
        batch_size=batch_size,
        name="frontier_batch_ids",
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
    "ActionSample",
    "action_log_probs",
    "action_probs",
    "sample_action_for_generation",
    "stop_continue_log_probs",
]
