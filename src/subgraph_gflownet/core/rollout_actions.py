from __future__ import annotations

import torch

from .actor import HierarchicalStateActionDistribution
from .state import SubgraphAction


def log_softmax_choice(logits: torch.Tensor, index: int) -> torch.Tensor:
    return torch.log_softmax(logits.to(dtype=torch.float32), dim=0)[int(index)]


def sample_index(
    logits: torch.Tensor, *, temperature: float
) -> tuple[int, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    scaled_logits = logits.to(dtype=torch.float32) / float(temperature)
    probabilities = torch.softmax(scaled_logits, dim=0)
    sample = torch.multinomial(probabilities, num_samples=1)
    return int(sample.item()), scaled_logits


def gate_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    return torch.stack(
        (state_distribution.stop_logit, state_distribution.continue_logit), dim=0
    ).to(dtype=torch.float32)


def node_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    return state_distribution.node_choice_logits.to(dtype=torch.float32)


def relation_logits(
    state_distribution: HierarchicalStateActionDistribution,
    node_choice_idx: int,
) -> torch.Tensor:
    relation_slice = state_distribution.relation_slice(int(node_choice_idx))
    return state_distribution.relation_choice_logits[relation_slice].to(
        dtype=torch.float32
    )


def edge_logits(
    state_distribution: HierarchicalStateActionDistribution,
    relation_choice_idx: int,
) -> torch.Tensor:
    edge_slice = state_distribution.edge_slice(int(relation_choice_idx))
    return state_distribution.edge_choice_logits[edge_slice].to(dtype=torch.float32)


def stop_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    if int(state_distribution.stop_choice_logits.numel()) <= 0:
        raise RuntimeError("Stop distributions must expose at least one stop choice.")
    return state_distribution.stop_choice_logits.to(dtype=torch.float32)


def select_stop_choice(
    *, state_distribution: HierarchicalStateActionDistribution
) -> int:
    if int(state_distribution.stop_choice_logits.numel()) <= 0:
        raise RuntimeError("Stop distributions must expose at least one stop choice.")
    return 0


def sample_state_action(
    *,
    state_distribution: HierarchicalStateActionDistribution,
    temperature: float,
) -> tuple[SubgraphAction, torch.Tensor, int]:
    gate = gate_logits(state_distribution)
    gate_idx, _ = sample_index(gate, temperature=temperature)
    gate_log_prob = log_softmax_choice(gate, gate_idx)
    if gate_idx == 0 or int(state_distribution.node_choice_logits.numel()) <= 0:
        stop = stop_logits(state_distribution)
        stop_idx, _ = sample_index(stop, temperature=temperature)
        stop_log_prob = log_softmax_choice(stop, stop_idx)
        return (
            state_distribution.build_stop_action(int(stop_idx)),
            gate_log_prob + stop_log_prob,
            state_distribution.current_component_count,
        )
    nodes = node_logits(state_distribution)
    node_idx, _ = sample_index(nodes, temperature=temperature)
    node_log_prob = log_softmax_choice(nodes, node_idx)
    relations = relation_logits(state_distribution, int(node_idx))
    relation_idx, _ = sample_index(relations, temperature=temperature)
    relation_log_prob = log_softmax_choice(relations, relation_idx)
    relation_choice_slice = state_distribution.relation_slice(int(node_idx))
    relation_choice_idx = int(relation_choice_slice.start + int(relation_idx))
    edges = edge_logits(state_distribution, relation_choice_idx)
    edge_idx, _ = sample_index(edges, temperature=temperature)
    edge_log_prob = log_softmax_choice(edges, edge_idx)
    edge_choice_slice = state_distribution.edge_slice(relation_choice_idx)
    edge_choice_idx = int(edge_choice_slice.start + int(edge_idx))
    return (
        state_distribution.build_edge_action(edge_choice_idx),
        gate_log_prob + node_log_prob + relation_log_prob + edge_log_prob,
        state_distribution.edge_next_component_count(edge_choice_idx),
    )


def teacher_forced_state_action(
    *,
    state_distribution: HierarchicalStateActionDistribution,
    planned_edge_id: int | None,
) -> tuple[SubgraphAction, torch.Tensor, int]:
    gate = gate_logits(state_distribution)
    if planned_edge_id is None:
        stop_idx = select_stop_choice(state_distribution=state_distribution)
        stop = stop_logits(state_distribution)
        return (
            state_distribution.build_stop_action(int(stop_idx)),
            log_softmax_choice(gate, 0) + log_softmax_choice(stop, stop_idx),
            state_distribution.current_component_count,
        )
    if int(state_distribution.node_choice_logits.numel()) <= 0:
        raise RuntimeError(
            "Teacher-forced subgraph replay could not resolve any expandable node."
        )
    matching_edge_indices = torch.nonzero(
        state_distribution.edge_choice_edge_ids == int(planned_edge_id),
        as_tuple=False,
    ).view(-1)
    if int(matching_edge_indices.numel()) > 0:
        edge_choice_idx = int(matching_edge_indices[0].item())
        relation_choice_idx = int(
            state_distribution.edge_choice_relation_choice_indices[
                edge_choice_idx
            ].item()
        )
        node_choice_idx = int(
            state_distribution.relation_choice_node_choice_indices[
                relation_choice_idx
            ].item()
        )
        gate_log_prob = log_softmax_choice(gate, 1)
        node_log_prob = log_softmax_choice(
            node_logits(state_distribution), node_choice_idx
        )
        relation_choice_slice = state_distribution.relation_slice(node_choice_idx)
        relation_local_idx = int(relation_choice_idx - relation_choice_slice.start)
        relation_log_prob = log_softmax_choice(
            relation_logits(state_distribution, node_choice_idx),
            relation_local_idx,
        )
        edge_choice_slice = state_distribution.edge_slice(relation_choice_idx)
        edge_local_idx = int(edge_choice_idx - edge_choice_slice.start)
        edge_log_prob = log_softmax_choice(
            edge_logits(state_distribution, relation_choice_idx),
            edge_local_idx,
        )
        return (
            state_distribution.build_edge_action(edge_choice_idx),
            gate_log_prob + node_log_prob + relation_log_prob + edge_log_prob,
            state_distribution.edge_next_component_count(edge_choice_idx),
        )
    raise RuntimeError(
        "Teacher-forced subgraph replay could not resolve the planned edge under the "
        f"semantic hierarchical policy. planned_edge_id={planned_edge_id}"
    )


__all__ = [
    "sample_state_action",
    "teacher_forced_state_action",
]
