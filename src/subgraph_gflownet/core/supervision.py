from __future__ import annotations

from dataclasses import dataclass

import torch

from .actor import HierarchicalStateActionDistribution
from .policy import SubgraphPolicy
from .rollout_engine import SubgraphTrajectorySampleBatch
from .state import SubgraphState
from .subgraph_batch import SubgraphBatch


def canonicalize_teacher_sequence_bank(
    sequence_bank: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], ...]:
    if not sequence_bank:
        return ()
    return tuple(
        tuple(int(edge_id) for edge_id in edge_ids) for edge_ids in sequence_bank
    )


def teacher_subgraph_bank_from_sequences(
    sequence_bank: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], ...]:
    canonical = canonicalize_teacher_sequence_bank(sequence_bank)
    return tuple(
        tuple(sorted({int(edge_id) for edge_id in edge_ids})) for edge_ids in canonical
    )


def representative_teacher_sequence(
    sequence_bank: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
) -> tuple[int, ...] | None:
    canonical = canonicalize_teacher_sequence_bank(sequence_bank)
    if not canonical:
        return None
    return tuple(int(edge_id) for edge_id in canonical[0])


def remaining_success_edge_ids(
    *,
    state: SubgraphState,
    success_edge_sets: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> tuple[int, ...]:
    selected = {int(edge_id) for edge_id in state.edge_ids}
    remaining = {
        int(edge_id)
        for edge_set in success_edge_sets
        for edge_id in edge_set
        if int(edge_id) not in selected
    }
    return tuple(sorted(remaining))


def _log_softmax_choice(logits: torch.Tensor, index: int) -> torch.Tensor:
    return torch.log_softmax(logits.to(dtype=torch.float32), dim=0)[int(index)]


def planned_edge_log_prob(
    *,
    distribution: HierarchicalStateActionDistribution,
    planned_edge_id: int,
) -> torch.Tensor | None:
    matching_edge_indices = torch.nonzero(
        distribution.edge_choice_edge_ids == int(planned_edge_id), as_tuple=False
    ).view(-1)
    if int(matching_edge_indices.numel()) <= 0:
        return None
    edge_choice_idx = int(matching_edge_indices[0].item())
    relation_choice_idx = int(
        distribution.edge_choice_relation_choice_indices[edge_choice_idx].item()
    )
    node_choice_idx = int(
        distribution.relation_choice_node_choice_indices[relation_choice_idx].item()
    )
    gate_logits = torch.stack(
        (distribution.stop_logit, distribution.continue_logit), dim=0
    ).to(dtype=torch.float32)
    node_logits = distribution.node_choice_logits.to(dtype=torch.float32)
    relation_slice = distribution.relation_slice(int(node_choice_idx))
    relation_local_idx = int(relation_choice_idx - relation_slice.start)
    relation_logits = distribution.relation_choice_logits[relation_slice].to(
        dtype=torch.float32
    )
    edge_slice = distribution.edge_slice(int(relation_choice_idx))
    edge_local_idx = int(edge_choice_idx - edge_slice.start)
    edge_logits = distribution.edge_choice_logits[edge_slice].to(dtype=torch.float32)
    return (
        _log_softmax_choice(gate_logits, 1)
        + _log_softmax_choice(node_logits, node_choice_idx)
        + _log_softmax_choice(relation_logits, relation_local_idx)
        + _log_softmax_choice(edge_logits, edge_local_idx)
    )


def success_action_targets(
    *,
    distribution: HierarchicalStateActionDistribution,
    positive_edge_ids: tuple[int, ...] | list[int] | set[int],
) -> torch.Tensor:
    positives = {int(edge_id) for edge_id in positive_edge_ids}
    candidate_edge_ids = distribution.edge_choice_edge_ids.detach().cpu().tolist()
    return torch.tensor(
        [1.0 if int(edge_id) in positives else 0.0 for edge_id in candidate_edge_ids],
        device=distribution.edge_choice_logits.device,
        dtype=torch.float32,
    )


@dataclass(frozen=True)
class SequenceSupervisionLossOutput:
    imitation_loss: torch.Tensor
    success_action_loss: torch.Tensor
    prefix_count: int
    sequence_count: int
    positive_edge_count: int
    candidate_edge_count: int


def build_single_state_distribution(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    graph_idx: int,
    state: SubgraphState,
) -> tuple[object, HierarchicalStateActionDistribution]:
    graph_ids = torch.tensor(
        [int(graph_idx)], device=prepared_batch.device, dtype=torch.long
    )
    rollout_batch = policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    rollout_batch = type(rollout_batch)(
        graph_ids=graph_ids,
        states=(state,),
        done_mask=torch.zeros_like(graph_ids, dtype=torch.bool),
        view_shape=(1, 1),
    )
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=int(graph_idx),
        state=state,
    )
    distribution = policy.compute_action_distribution(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
        analyses=(analysis,),
        action_pruning=None,
    )
    return analysis, distribution.state_distributions[0]


def compute_sequence_supervision_losses(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    sequence_banks: tuple[tuple[tuple[int, ...], ...], ...],
    success_edge_banks: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
) -> SequenceSupervisionLossOutput:
    device = prepared_batch.device
    resolved_success_edge_banks = (
        prepared_batch.graph_teacher_subgraph_bank
        if success_edge_banks is None
        else success_edge_banks
    )
    imitation_terms: list[torch.Tensor] = []
    success_action_terms: list[torch.Tensor] = []
    prefix_count = 0
    sequence_count = 0
    positive_edge_count = 0
    candidate_edge_count = 0
    for graph_idx, sequence_bank in enumerate(sequence_banks):
        representative_sequence = representative_teacher_sequence(sequence_bank)
        if representative_sequence is None:
            continue
        success_edge_sets = resolved_success_edge_banks[int(graph_idx)]
        if not success_edge_sets:
            continue
        sequence_count += 1
        state = policy.initial_state()
        for planned_edge_id in representative_sequence:
            _analysis, distribution = build_single_state_distribution(
                policy=policy,
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                state=state,
            )
            planned_log_prob = planned_edge_log_prob(
                distribution=distribution,
                planned_edge_id=int(planned_edge_id),
            )
            if planned_log_prob is not None:
                imitation_terms.append(-planned_log_prob.to(dtype=torch.float32))
            positive_edge_ids = remaining_success_edge_ids(
                state=state,
                success_edge_sets=success_edge_sets,
            )
            if int(distribution.edge_choice_logits.numel()) > 0:
                targets = success_action_targets(
                    distribution=distribution,
                    positive_edge_ids=positive_edge_ids,
                )
                if int(targets.numel()) > 0:
                    success_action_terms.append(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            distribution.edge_choice_logits.to(dtype=torch.float32),
                            targets,
                            reduction="mean",
                        )
                    )
                    positive_edge_count += int(targets.sum().item())
                    candidate_edge_count += int(targets.numel())
            prefix_count += 1
            state = state.with_edge(int(planned_edge_id))
    imitation_loss = (
        torch.stack(imitation_terms).mean()
        if imitation_terms
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    success_action_loss = (
        torch.stack(success_action_terms).mean()
        if success_action_terms
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    return SequenceSupervisionLossOutput(
        imitation_loss=imitation_loss,
        success_action_loss=success_action_loss,
        prefix_count=int(prefix_count),
        sequence_count=int(sequence_count),
        positive_edge_count=int(positive_edge_count),
        candidate_edge_count=int(candidate_edge_count),
    )


def compute_expand_imitation_loss(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    sample_batch: SubgraphTrajectorySampleBatch,
    from_anchor_bonus: float,
    answer_finish_bonus: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weighted_log_probs: list[torch.Tensor] = []
    weight_values: list[float] = []
    from_anchor_steps = 0.0
    answer_finish_steps = 0.0
    if sample_batch.chosen_source_graph_nodes is None:
        raise RuntimeError(
            "Replay expand imitation requires chosen_source_graph_nodes metadata."
        )
    for graph_idx in range(int(sample_batch.num_graphs)):
        anchor_nodes = {
            int(node_id)
            for node_id in prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)]
        }
        for rollout_idx in range(int(sample_batch.num_rollouts)):
            state = policy.initial_state()
            for action_step in range(int(policy.max_steps)):
                edge_id = int(
                    sample_batch.chosen_edge_ids[graph_idx, rollout_idx, action_step]
                    .detach()
                    .item()
                )
                if edge_id < 0:
                    continue
                source_graph_node = int(
                    sample_batch.chosen_source_graph_nodes[
                        graph_idx, rollout_idx, action_step
                    ]
                    .detach()
                    .item()
                )
                if source_graph_node < 0:
                    continue
                current_analysis = policy.analyze_state(
                    prepared_batch=prepared_batch,
                    graph_idx=int(graph_idx),
                    state=state,
                )
                next_state = state.with_edge(int(edge_id))
                next_analysis = policy.analyze_state(
                    prepared_batch=prepared_batch,
                    graph_idx=int(graph_idx),
                    state=next_state,
                )
                weight = 1.0
                src = int(source_graph_node)
                if src in anchor_nodes:
                    weight += float(from_anchor_bonus)
                    from_anchor_steps += 1.0
                current_gold_answer_count, _ = policy.count_gold_answers_in_graph(
                    prepared_batch=prepared_batch,
                    graph_idx=int(graph_idx),
                    analysis=current_analysis,
                )
                next_gold_answer_count, _ = policy.count_gold_answers_in_graph(
                    prepared_batch=prepared_batch,
                    graph_idx=int(graph_idx),
                    analysis=next_analysis,
                )
                if int(next_gold_answer_count) > int(current_gold_answer_count):
                    weight += float(answer_finish_bonus)
                    answer_finish_steps += 1.0
                weighted_log_probs.append(
                    sample_batch.log_pf_actions[graph_idx, rollout_idx, action_step].to(
                        dtype=torch.float32
                    )
                )
                weight_values.append(float(weight))
                state = next_state
    if not weight_values:
        return sample_batch.log_pf_actions.new_zeros((), dtype=torch.float32), {
            "from_anchor_steps": 0.0,
            "answer_finish_steps": 0.0,
            "mean_weight": 0.0,
        }
    weight_tensor = torch.tensor(
        weight_values,
        device=sample_batch.log_pf_actions.device,
        dtype=torch.float32,
    )
    log_prob_tensor = torch.stack(weighted_log_probs).to(dtype=torch.float32)
    loss = -(log_prob_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)
    return loss, {
        "from_anchor_steps": float(from_anchor_steps),
        "answer_finish_steps": float(answer_finish_steps),
        "mean_weight": float(weight_tensor.mean().item()),
    }


__all__ = [
    "SequenceSupervisionLossOutput",
    "build_single_state_distribution",
    "compute_expand_imitation_loss",
    "compute_sequence_supervision_losses",
    "canonicalize_teacher_sequence_bank",
    "planned_edge_log_prob",
    "remaining_success_edge_ids",
    "representative_teacher_sequence",
    "success_action_targets",
    "teacher_subgraph_bank_from_sequences",
]
