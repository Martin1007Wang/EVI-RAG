from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph.segments import scatter_log_softmax
from src.weaver.policy import CandidateEdges


CONTINUE_ACTION = 0
STOP_ACTION = 1

STOP_OPTION_INDEX = 0
EXPAND_OPTION_INDEX = 1


@dataclass(frozen=True)
class ActionSample:
    """
    One sampled action per physical graph.

    action_type:
        [B], 0 for Continue/Expand(edge), 1 for Stop.

    chosen_edges:
        [B], physical edge ids for Continue actions, -1 for Stop.

    target_log_prob:
        [B], log P_target(action | state).

        For Stop:
            log P_target(Stop | s)

        For Expand(edge):
            log P_target(Expand | s)
            + log P_target(edge | s, Expand)

        For environment-forced Stop, e.g. inactive graph, no frontier, or
        exhausted budget, this is zero. Such forced termination should not
        train the Stop scorer as evidence sufficiency.
    """

    action_type: torch.Tensor
    chosen_edges: torch.Tensor
    target_log_prob: torch.Tensor


def sample_policy_actions(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    candidates: CandidateEdges,
    active: torch.Tensor,
    can_expand: torch.Tensor,
    temperature: float,
    batch_size: int,
) -> ActionSample:
    """
    Sample one target-policy action per graph.

    The behavior distribution uses temperature:

        P_behavior(o | s) ∝ exp(z_o / T)
        P_behavior(e | s, Expand) ∝ exp(z_e / T)

    Returned log-probabilities are always from the untempered target policy:

        log P_target(Stop | s)
        log P_target(Expand | s) + log P_target(e | s, Expand)

    This function has no proposal intervention, no coverage guide, and no
    teacher override. VIGOR is a loss-level auxiliary, not a behavior sampler.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    device = stop_logits.device

    stop_logits = stop_logits.to(device=device, dtype=torch.float32).view(batch_size)
    expand_logits = expand_logits.to(device=device, dtype=torch.float32).view(
        batch_size
    )
    candidates = _candidate_edges_to_device(candidates, device=device)
    active = active.to(device=device, dtype=torch.bool).view(batch_size)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_option_inputs(
        stop_logits=stop_logits,
        expand_logits=expand_logits,
        candidates=candidates,
        active=active,
        can_expand=can_expand,
        batch_size=batch_size,
        device=device,
    )

    target_type_logp, target_edge_logp = option_action_log_probs(
        stop_logits=stop_logits,
        expand_logits=expand_logits,
        candidates=candidates,
        can_expand=can_expand,
        batch_size=batch_size,
    )

    behavior_candidates = CandidateEdges(
        edge_ids=candidates.edge_ids,
        expand_logits=candidates.expand_logits / float(temperature),
        batch_index=candidates.batch_index,
    )
    behavior_type_logp, behavior_edge_logp = option_action_log_probs(
        stop_logits=stop_logits / float(temperature),
        expand_logits=expand_logits / float(temperature),
        candidates=behavior_candidates,
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

    # Only active graphs that are legally expandable sample between Stop/Expand.
    # All other active graphs are environment-forced Stop with log-probability 0.
    learnable_graphs = (active & can_expand).nonzero(as_tuple=False).view(-1)
    if learnable_graphs.numel() == 0:
        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    sampled_options = torch.distributions.Categorical(
        logits=behavior_type_logp.index_select(0, learnable_graphs),
    ).sample()

    target_log_prob[learnable_graphs] = (
        target_type_logp.index_select(
            0,
            learnable_graphs,
        )
        .gather(
            dim=1,
            index=sampled_options.view(-1, 1),
        )
        .squeeze(1)
    )

    expand_mask = sampled_options.eq(EXPAND_OPTION_INDEX)
    if bool(expand_mask.any()):
        expand_graphs = learnable_graphs[expand_mask]

        for graph_id_tensor in expand_graphs:
            graph_id = int(graph_id_tensor.item())
            candidate_pos = (
                candidates.batch_index.eq(graph_id).nonzero(as_tuple=False).view(-1)
            )
            if candidate_pos.numel() == 0:
                raise RuntimeError(
                    "Sampled Expand for graph with no candidate edges: "
                    f"physical graph id={graph_id}."
                )

            local_behavior_logits = behavior_edge_logp.index_select(0, candidate_pos)
            local_choice = torch.distributions.Categorical(
                logits=local_behavior_logits,
            ).sample()

            chosen_pos = candidate_pos[int(local_choice.item())]

            action_type[graph_id] = CONTINUE_ACTION
            chosen_edges[graph_id] = candidates.edge_ids[chosen_pos]
            target_log_prob[graph_id] = (
                target_log_prob[graph_id] + target_edge_logp[chosen_pos]
            )

    # Sampled Stop graphs keep chosen_edges=-1 and already have target option log-prob.
    return ActionSample(
        action_type=action_type,
        chosen_edges=chosen_edges,
        target_log_prob=target_log_prob,
    )


def option_action_log_probs(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    candidates: CandidateEdges,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return option log-probs and conditional edge log-probs.

    type_logp:
        [B, 2], columns are [Stop, Expand].

    edge_logp:
        [num_candidates], conditional log P(edge | s, Expand), normalized
        within each candidate graph.

    For graphs where can_expand=False:
        P(Stop)=1, P(Expand)=0.
        Candidate edge log-probs belonging to those graphs, if any, are -inf.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    device = stop_logits.device

    stop_logits = stop_logits.to(device=device, dtype=torch.float32).view(batch_size)
    expand_logits = expand_logits.to(device=device, dtype=torch.float32).view(
        batch_size
    )
    candidates = _candidate_edges_to_device(candidates, device=device)
    can_expand = can_expand.to(device=device, dtype=torch.bool).view(batch_size)

    _validate_candidates_for_option(
        candidates=candidates,
        batch_size=batch_size,
        device=device,
    )

    type_logits = torch.stack([stop_logits, expand_logits], dim=-1)
    type_logits = type_logits.clone()

    # Environment-forced Stop. This makes target log-prob for Stop exactly 0
    # when the graph cannot legally expand.
    type_logits[~can_expand, STOP_OPTION_INDEX] = 0.0
    type_logits[~can_expand, EXPAND_OPTION_INDEX] = -torch.inf

    type_logp = torch.log_softmax(type_logits, dim=-1)

    if len(candidates) == 0:
        return type_logp, stop_logits.new_empty((0,), dtype=torch.float32)

    edge_logp = scatter_log_softmax(
        candidates.expand_logits.to(dtype=torch.float32),
        candidates.batch_index,
        num_segments=batch_size,
    )

    edge_allowed = can_expand.index_select(0, candidates.batch_index)
    edge_logp = torch.where(
        edge_allowed,
        edge_logp,
        edge_logp.new_full(edge_logp.shape, -torch.inf),
    )

    return type_logp, edge_logp


def option_action_probs(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    candidates: CandidateEdges,
    can_expand: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
        stop_prob:   [B]
        expand_prob: [B]
        edge_prob:   [num_candidates], conditional P(edge | s, Expand)
    """
    type_logp, edge_logp = option_action_log_probs(
        stop_logits=stop_logits,
        expand_logits=expand_logits,
        candidates=candidates,
        can_expand=can_expand,
        batch_size=int(batch_size),
    )

    edge_prob = torch.where(
        torch.isfinite(edge_logp),
        edge_logp.exp(),
        edge_logp.new_zeros(edge_logp.shape),
    )

    return (
        type_logp[:, STOP_OPTION_INDEX].exp(),
        type_logp[:, EXPAND_OPTION_INDEX].exp(),
        edge_prob,
    )


def _candidate_edges_to_device(
    candidates: CandidateEdges,
    *,
    device: torch.device,
) -> CandidateEdges:
    return CandidateEdges(
        edge_ids=candidates.edge_ids.to(device=device, dtype=torch.long),
        expand_logits=candidates.expand_logits.to(device=device, dtype=torch.float32),
        batch_index=candidates.batch_index.to(device=device, dtype=torch.long),
    )


def _validate_option_inputs(
    *,
    stop_logits: torch.Tensor,
    expand_logits: torch.Tensor,
    candidates: CandidateEdges,
    active: torch.Tensor,
    can_expand: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    expected_shape = (int(batch_size),)

    for name, tensor in {
        "stop_logits": stop_logits,
        "expand_logits": expand_logits,
        "active": active,
        "can_expand": can_expand,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )

    if bool((can_expand & ~active).any()):
        raise ValueError("can_expand cannot be true for inactive graphs.")

    _validate_candidates_for_option(
        candidates=candidates,
        batch_size=batch_size,
        device=device,
    )


def _validate_candidates_for_option(
    *,
    candidates: CandidateEdges,
    batch_size: int,
    device: torch.device,
) -> None:
    batch_size = int(batch_size)

    if candidates.edge_ids.device != device:
        raise ValueError(
            f"candidates.edge_ids is on {candidates.edge_ids.device}, expected {device}."
        )
    if candidates.expand_logits.device != device:
        raise ValueError(
            "candidates.expand_logits is on "
            f"{candidates.expand_logits.device}, expected {device}."
        )
    if candidates.batch_index.device != device:
        raise ValueError(
            "candidates.batch_index is on "
            f"{candidates.batch_index.device}, expected {device}."
        )

    if candidates.edge_ids.dtype != torch.long:
        raise TypeError(
            f"candidates.edge_ids must be torch.long, got {candidates.edge_ids.dtype}."
        )
    if candidates.batch_index.dtype != torch.long:
        raise TypeError(
            "candidates.batch_index must be torch.long, "
            f"got {candidates.batch_index.dtype}."
        )

    if candidates.edge_ids.ndim != 1:
        raise ValueError(
            f"candidates.edge_ids must be 1D, got {tuple(candidates.edge_ids.shape)}."
        )
    if candidates.expand_logits.ndim != 1:
        raise ValueError(
            "candidates.expand_logits must be 1D, "
            f"got {tuple(candidates.expand_logits.shape)}."
        )
    if candidates.batch_index.ndim != 1:
        raise ValueError(
            "candidates.batch_index must be 1D, "
            f"got {tuple(candidates.batch_index.shape)}."
        )

    if not (
        candidates.edge_ids.numel()
        == candidates.expand_logits.numel()
        == candidates.batch_index.numel()
    ):
        raise ValueError(
            "CandidateEdges fields must have matching lengths: "
            f"edge_ids={candidates.edge_ids.numel()}, "
            f"expand_logits={candidates.expand_logits.numel()}, "
            f"batch_index={candidates.batch_index.numel()}."
        )

    if candidates.batch_index.numel() == 0:
        return

    if bool((candidates.batch_index < 0).any()):
        raise ValueError("candidates.batch_index contains negative graph ids.")
    if bool((candidates.batch_index >= batch_size).any()):
        raise ValueError(
            f"candidates.batch_index contains ids outside [0, {batch_size})."
        )


__all__ = [
    "CONTINUE_ACTION",
    "STOP_ACTION",
    "STOP_OPTION_INDEX",
    "EXPAND_OPTION_INDEX",
    "ActionSample",
    "option_action_log_probs",
    "option_action_probs",
    "sample_policy_actions",
]
