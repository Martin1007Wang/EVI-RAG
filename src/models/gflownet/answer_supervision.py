from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.segment_ops import segment_logsumexp_1d


@dataclass(frozen=True)
class AnswerSupervisionMetadata:
    entity_offset: torch.Tensor
    key_base: torch.Tensor
    gold_keys: torch.Tensor


@dataclass(frozen=True)
class UniqueGraphAnswers:
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor


def graph_ids_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
    counts = ptr[1:] - ptr[:-1]
    if int(counts.numel()) == 0:
        return torch.empty((0,), device=ptr.device, dtype=torch.long)
    graph_ids = torch.arange(int(counts.numel()), device=ptr.device, dtype=torch.long)
    return graph_ids.repeat_interleave(counts.to(device=ptr.device, dtype=torch.long))


def build_answer_supervision_metadata(
    *,
    node_entity_ids: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    answer_ptr: torch.Tensor,
    device: torch.device,
) -> AnswerSupervisionMetadata:
    node_entity_ids = node_entity_ids.to(device=device, dtype=torch.long)
    answer_entity_ids = answer_entity_ids.to(device=device, dtype=torch.long)
    zero = torch.zeros((), device=device, dtype=torch.long)
    if int(node_entity_ids.numel()) > 0:
        min_entity = node_entity_ids.min()
        max_entity = node_entity_ids.max()
    else:
        min_entity = zero
        max_entity = zero
    if int(answer_entity_ids.numel()) > 0:
        min_entity = torch.minimum(min_entity, answer_entity_ids.min())
        max_entity = torch.maximum(max_entity, answer_entity_ids.max())
    entity_offset = (-torch.minimum(min_entity, zero)).to(dtype=torch.long)
    key_base = (max_entity + entity_offset + 1).clamp_min(1).to(dtype=torch.long)

    gold_graph_ids = graph_ids_from_ptr(answer_ptr.to(device=device, dtype=torch.long))
    if int(answer_entity_ids.numel()) > 0:
        gold_keys = torch.unique(
            gold_graph_ids * key_base + (answer_entity_ids + entity_offset),
            sorted=True,
        )
    else:
        gold_keys = torch.empty((0,), device=device, dtype=torch.long)
    return AnswerSupervisionMetadata(
        entity_offset=entity_offset,
        key_base=key_base,
        gold_keys=gold_keys,
    )


def build_unique_graph_answers(
    *,
    answer_entity_ids: torch.Tensor,
    answer_ptr: torch.Tensor,
    device: torch.device,
) -> UniqueGraphAnswers:
    flat_unique_answers: list[torch.Tensor] = []
    ptr_values = [0]
    answer_entity_ids = answer_entity_ids.to(device=device, dtype=torch.long)
    answer_ptr = answer_ptr.to(device=device, dtype=torch.long)
    for graph_idx in range(max(int(answer_ptr.numel()) - 1, 0)):
        start = int(answer_ptr[graph_idx].item())
        end = int(answer_ptr[graph_idx + 1].item())
        unique_answers = torch.unique(answer_entity_ids[start:end], sorted=True)
        flat_unique_answers.append(unique_answers)
        ptr_values.append(ptr_values[-1] + int(unique_answers.numel()))
    if flat_unique_answers:
        flat_answer_ids = torch.cat(flat_unique_answers, dim=0)
    else:
        flat_answer_ids = torch.empty((0,), device=device, dtype=torch.long)
    return UniqueGraphAnswers(
        answer_entity_ids=flat_answer_ids,
        answer_ptr=torch.tensor(ptr_values, device=device, dtype=torch.long),
    )


def lookup_answer_local_index(
    *,
    unique_answers: UniqueGraphAnswers,
    graph_ids: torch.Tensor,
    entity_ids: torch.Tensor,
) -> torch.Tensor:
    graph_ids = graph_ids.to(dtype=torch.long)
    entity_ids = entity_ids.to(dtype=torch.long)
    if tuple(graph_ids.shape) != tuple(entity_ids.shape):
        raise ValueError(
            "graph_ids and entity_ids must share the same shape for answer lookup. "
            f"graph_ids={tuple(graph_ids.shape)} entity_ids={tuple(entity_ids.shape)}."
        )
    local_index = torch.full_like(entity_ids, fill_value=-1, dtype=torch.long)
    for graph_idx in range(max(int(unique_answers.answer_ptr.numel()) - 1, 0)):
        graph_mask = graph_ids == int(graph_idx)
        if not bool(graph_mask.any().item()):
            continue
        start = int(unique_answers.answer_ptr[graph_idx].item())
        end = int(unique_answers.answer_ptr[graph_idx + 1].item())
        graph_answers = unique_answers.answer_entity_ids[start:end]
        if int(graph_answers.numel()) == 0:
            continue
        graph_entities = entity_ids[graph_mask]
        answer_pos = torch.searchsorted(graph_answers, graph_entities)
        in_range = answer_pos < int(graph_answers.numel())
        matched = torch.zeros_like(graph_entities, dtype=torch.bool)
        matched[in_range] = (
            graph_answers.index_select(0, answer_pos[in_range])
            == graph_entities[in_range]
        )
        local_values = torch.full_like(graph_entities, fill_value=-1, dtype=torch.long)
        local_values[matched] = answer_pos[matched]
        local_index[graph_mask] = local_values
    return local_index.view_as(entity_ids)


def compute_answer_sink_targets(
    *,
    unique_answers: UniqueGraphAnswers,
    graph_ids: torch.Tensor,
    entity_ids: torch.Tensor,
    non_gold_terminal_log_reward: float,
    gold_reward_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gold_reward_mode not in {"shared", "unit"}:
        raise ValueError(
            "gold_reward_mode must be one of {'shared', 'unit'} when computing answer sinks."
        )
    graph_ids = graph_ids.to(dtype=torch.long)
    entity_ids = entity_ids.to(dtype=torch.long)
    gold_answer_counts_per_graph = (
        unique_answers.answer_ptr[1:] - unique_answers.answer_ptr[:-1]
    ).to(device=entity_ids.device, dtype=torch.long)
    if int(gold_answer_counts_per_graph.numel()) == 0:
        gold_answer_counts_per_graph = torch.zeros(
            (int(graph_ids.max().item()) + 1 if int(graph_ids.numel()) > 0 else 0,),
            device=entity_ids.device,
            dtype=torch.long,
        )
    local_answer_index = lookup_answer_local_index(
        unique_answers=unique_answers,
        graph_ids=graph_ids,
        entity_ids=entity_ids,
    )
    gold_answer_counts = gold_answer_counts_per_graph.index_select(0, graph_ids)
    is_gold = local_answer_index >= 0
    sink_ids = torch.where(is_gold, local_answer_index, gold_answer_counts)
    sink_log_rewards = torch.full(
        entity_ids.shape,
        fill_value=float(non_gold_terminal_log_reward),
        device=entity_ids.device,
        dtype=torch.float32,
    )
    if bool(is_gold.any().item()):
        if gold_reward_mode == "shared":
            sink_log_rewards[is_gold] = -torch.log(
                gold_answer_counts[is_gold].to(dtype=torch.float32).clamp_min(1.0)
            )
        else:
            sink_log_rewards[is_gold] = 0.0
    return (
        sink_ids.view_as(entity_ids),
        sink_log_rewards.view_as(entity_ids),
        gold_answer_counts_per_graph,
    )


def build_node_answer_sink_tensors(
    *,
    node_ptr: torch.Tensor,
    node_entity_ids: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    answer_ptr: torch.Tensor,
    non_gold_terminal_log_reward: float,
    gold_reward_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_graph_ids = graph_ids_from_ptr(
        node_ptr.to(device=node_ptr.device, dtype=torch.long)
    )
    unique_answers = build_unique_graph_answers(
        answer_entity_ids=answer_entity_ids,
        answer_ptr=answer_ptr,
        device=node_ptr.device,
    )
    sink_ids, sink_log_rewards, gold_answer_counts_per_graph = (
        compute_answer_sink_targets(
            unique_answers=unique_answers,
            graph_ids=node_graph_ids,
            entity_ids=node_entity_ids.to(device=node_ptr.device, dtype=torch.long),
            non_gold_terminal_log_reward=non_gold_terminal_log_reward,
            gold_reward_mode=gold_reward_mode,
        )
    )
    return node_graph_ids, sink_ids, sink_log_rewards, gold_answer_counts_per_graph


def compute_gold_entity_ranking_loss(
    *,
    graph_ids: torch.Tensor,
    entity_ids: torch.Tensor,
    entity_scores: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    answer_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_ids = graph_ids.to(dtype=torch.long)
    entity_ids = entity_ids.to(dtype=torch.long)
    entity_scores = entity_scores.to(dtype=torch.float32)
    answer_entity_ids = answer_entity_ids.to(dtype=torch.long)
    answer_ptr = answer_ptr.to(dtype=torch.long)
    if tuple(graph_ids.shape) != tuple(entity_ids.shape) or tuple(
        graph_ids.shape
    ) != tuple(entity_scores.shape):
        raise ValueError(
            "graph_ids, entity_ids, and entity_scores must share the same shape for "
            "direct gold-entity ranking supervision. "
            f"graph_ids={tuple(graph_ids.shape)} entity_ids={tuple(entity_ids.shape)} "
            f"entity_scores={tuple(entity_scores.shape)}."
        )
    num_graphs = max(int(answer_ptr.numel()) - 1, 0)
    zero = entity_scores.new_zeros(())
    if int(graph_ids.numel()) == 0 or int(entity_ids.numel()) == 0 or num_graphs == 0:
        return zero, zero, zero
    if int(answer_entity_ids.numel()) == 0:
        return zero, zero, zero

    min_entity = torch.minimum(entity_ids.min(), answer_entity_ids.min())
    max_entity = torch.maximum(entity_ids.max(), answer_entity_ids.max())
    entity_offset = (-torch.minimum(min_entity, torch.zeros_like(min_entity))).to(
        dtype=torch.long
    )
    key_base = (max_entity + entity_offset + 1).clamp_min(1).to(dtype=torch.long)

    entity_group_keys = graph_ids * key_base + (entity_ids + entity_offset)
    unique_keys, inverse_ids = torch.unique(
        entity_group_keys, sorted=True, return_inverse=True
    )
    entity_log_scores, has_entity_values = segment_logsumexp_1d(
        values=entity_scores,
        segment_ids=inverse_ids,
        num_segments=int(unique_keys.numel()),
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    if not bool(has_entity_values.any().item()):
        return zero, zero, zero
    unique_keys = unique_keys[has_entity_values]
    entity_log_scores = entity_log_scores[has_entity_values]
    entity_graph_ids = unique_keys // key_base

    gold_graph_ids = graph_ids_from_ptr(answer_ptr)
    gold_keys = torch.unique(
        gold_graph_ids * key_base + (answer_entity_ids + entity_offset),
        sorted=True,
    )
    gold_match_idx = torch.searchsorted(gold_keys, unique_keys)
    gold_in_range = gold_match_idx < int(gold_keys.numel())
    gold_entity_mask = torch.zeros_like(unique_keys, dtype=torch.bool)
    if bool(gold_in_range.any().item()):
        gold_entity_mask[gold_in_range] = (
            gold_keys.index_select(0, gold_match_idx[gold_in_range])
            == unique_keys[gold_in_range]
        )
    if not bool(gold_entity_mask.any().item()):
        return (
            zero,
            zero,
            entity_log_scores.new_tensor(float(entity_log_scores.numel())),
        )

    total_entity_log_mass, has_graph_entities = segment_logsumexp_1d(
        values=entity_log_scores,
        segment_ids=entity_graph_ids,
        num_segments=num_graphs,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    gold_entity_scores = entity_log_scores[gold_entity_mask]
    gold_entity_graph_ids = entity_graph_ids[gold_entity_mask]
    gold_entity_log_mass, has_gold_entities = segment_logsumexp_1d(
        values=gold_entity_scores,
        segment_ids=gold_entity_graph_ids,
        num_segments=num_graphs,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    valid_graphs = has_graph_entities & has_gold_entities
    if not bool(valid_graphs.any().item()):
        return (
            zero,
            zero,
            entity_log_scores.new_tensor(float(entity_log_scores.numel())),
        )

    gold_log_probs = (
        gold_entity_log_mass[valid_graphs] - total_entity_log_mass[valid_graphs]
    )
    gold_mass = gold_log_probs.exp().mean()
    loss = (-gold_log_probs).mean()
    return (
        loss,
        gold_mass,
        entity_log_scores.new_tensor(float(entity_log_scores.numel())),
    )


def lookup_is_gold(
    *,
    metadata: AnswerSupervisionMetadata,
    graph_ids: torch.Tensor,
    entity_ids: torch.Tensor,
) -> torch.Tensor:
    graph_ids = graph_ids.to(dtype=torch.long)
    entity_ids = entity_ids.to(dtype=torch.long)
    lookup_keys = graph_ids * metadata.key_base + (entity_ids + metadata.entity_offset)
    is_gold = torch.zeros_like(lookup_keys, dtype=torch.bool)
    if int(metadata.gold_keys.numel()) == 0:
        return is_gold.view_as(entity_ids)
    gold_match_idx = torch.searchsorted(metadata.gold_keys, lookup_keys)
    gold_in_range = gold_match_idx < int(metadata.gold_keys.numel())
    is_gold[gold_in_range] = (
        metadata.gold_keys.index_select(0, gold_match_idx[gold_in_range])
        == lookup_keys[gold_in_range]
    )
    return is_gold.view_as(entity_ids)


def build_answer_mask(
    *,
    node_ptr: torch.Tensor,
    node_entity_ids: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    answer_ptr: torch.Tensor,
) -> torch.Tensor:
    num_nodes_total = int(node_ptr[-1].item()) if int(node_ptr.numel()) > 0 else 0
    answer_mask = torch.zeros(
        (num_nodes_total,), device=node_ptr.device, dtype=torch.bool
    )
    if int(answer_mask.numel()) == 0 or int(answer_entity_ids.numel()) == 0:
        return answer_mask
    metadata = build_answer_supervision_metadata(
        node_entity_ids=node_entity_ids,
        answer_entity_ids=answer_entity_ids,
        answer_ptr=answer_ptr,
        device=node_ptr.device,
    )
    node_graph_ids = graph_ids_from_ptr(
        node_ptr.to(device=node_ptr.device, dtype=torch.long)
    )
    answer_mask = lookup_is_gold(
        metadata=metadata,
        graph_ids=node_graph_ids,
        entity_ids=node_entity_ids.to(device=node_ptr.device, dtype=torch.long),
    )
    return answer_mask.to(dtype=torch.bool)


__all__ = [
    "AnswerSupervisionMetadata",
    "UniqueGraphAnswers",
    "build_answer_mask",
    "build_answer_supervision_metadata",
    "build_node_answer_sink_tensors",
    "build_unique_graph_answers",
    "compute_gold_entity_ranking_loss",
    "compute_answer_sink_targets",
    "graph_ids_from_ptr",
    "lookup_answer_local_index",
    "lookup_is_gold",
]
