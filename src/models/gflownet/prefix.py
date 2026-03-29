from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .path import append_relation_and_node_tokens_inplace, append_stop_token_inplace


@dataclass(frozen=True)
class PrefixKey:
    sample_id: str
    current_node: int
    num_steps: int
    is_absorbing: bool
    token_ids: tuple[int, ...]
    visited_entity_ids: tuple[int, ...]


def _resolve_flat_prefix_node_rows(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(flat_current_abs_nodes.shape) != tuple(flat_num_steps.shape):
        raise ValueError(
            "flat_current_abs_nodes and flat_num_steps must align for prefix helpers. "
            f"current={tuple(flat_current_abs_nodes.shape)} steps={tuple(flat_num_steps.shape)}."
        )
    if flat_path_token_ids is None:
        if bool((flat_num_steps != 0).any().item()):
            raise ValueError(
                "Exact prefix helpers require path_token_ids for non-root rows."
            )
        return flat_current_abs_nodes.unsqueeze(1), torch.ones_like(
            flat_current_abs_nodes.unsqueeze(1), dtype=torch.bool
        )
    if int(flat_path_token_ids.size(0)) != int(flat_current_abs_nodes.numel()):
        raise ValueError(
            "flat_path_token_ids rows must align with flat_current_abs_nodes in prefix helpers. "
            f"path_rows={int(flat_path_token_ids.size(0))} states={int(flat_current_abs_nodes.numel())}."
        )
    prefix_node_rows = flat_path_token_ids[:, 0::2].to(dtype=torch.long)
    slot_ids = torch.arange(
        int(prefix_node_rows.size(1)),
        device=prefix_node_rows.device,
        dtype=torch.long,
    )
    valid_mask = slot_ids.unsqueeze(0) <= flat_num_steps.unsqueeze(1).to(
        dtype=torch.long
    )
    return prefix_node_rows, valid_mask


def build_flat_visited_entity_id_rows(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_node_rows, valid_mask = _resolve_flat_prefix_node_rows(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
    )
    if int(prefix_node_rows.numel()) == 0:
        return prefix_node_rows, valid_mask
    safe_nodes = prefix_node_rows.clamp(min=0, max=max(int(num_nodes) - 1, 0))
    entity_rows = node_entity_ids_by_abs_node.index_select(
        0, safe_nodes.reshape(-1)
    ).view_as(prefix_node_rows)
    invalid_entity = torch.full_like(entity_rows, fill_value=-1)
    entity_rows = torch.where(
        valid_mask, entity_rows.to(dtype=torch.long), invalid_entity
    )
    unique_mask = valid_mask.clone()
    for slot_idx in range(int(entity_rows.size(1))):
        if slot_idx == 0:
            continue
        duplicate_mask = (
            entity_rows[:, :slot_idx] == entity_rows[:, slot_idx : slot_idx + 1]
        ).any(dim=1)
        unique_mask[:, slot_idx] = unique_mask[:, slot_idx] & (~duplicate_mask)
    unique_rows = torch.where(unique_mask, entity_rows, invalid_entity)
    return unique_rows, unique_mask


def build_flat_visited_entity_sketch(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor,
    num_nodes: int,
    sketch_dim: int,
    num_hashes: int,
) -> torch.Tensor:
    if sketch_dim < 1:
        raise ValueError("sketch_dim must be >= 1 for visited-entity sketches.")
    if num_hashes < 1:
        raise ValueError("num_hashes must be >= 1 for visited-entity sketches.")
    total_states = int(flat_current_abs_nodes.numel())
    sketch = torch.zeros(
        (total_states, int(sketch_dim)),
        device=flat_current_abs_nodes.device,
        dtype=torch.float32,
    )
    if total_states == 0:
        return sketch
    entity_rows, unique_mask = build_flat_visited_entity_id_rows(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
    )
    row_ids, slot_ids = torch.nonzero(unique_mask, as_tuple=True)
    if int(row_ids.numel()) == 0:
        return sketch
    entity_ids = entity_rows[row_ids, slot_ids].to(dtype=torch.long)
    hash_positions = []
    hash_signs = []
    for hash_idx in range(int(num_hashes)):
        pos = torch.remainder(
            entity_ids * (1315423911 + (104729 * hash_idx)) + (2654435761 + hash_idx),
            int(sketch_dim),
        ).to(dtype=torch.long)
        sign_bit = torch.remainder(
            entity_ids * (40503 + (2909 * hash_idx)) + (7919 + hash_idx),
            2,
        )
        sign = torch.where(
            sign_bit == 0,
            torch.ones_like(entity_ids, dtype=torch.float32),
            -torch.ones_like(entity_ids, dtype=torch.float32),
        )
        hash_positions.append(pos)
        hash_signs.append(sign)
    for pos, sign in zip(hash_positions, hash_signs):
        sketch.index_put_((row_ids, pos), sign, accumulate=True)
    return sketch / math.sqrt(float(num_hashes))


def build_child_prefix_token_ids(
    *,
    parent_path_token_ids: torch.Tensor,
    parent_num_steps: torch.Tensor,
    relation_ids: torch.Tensor,
    target_nodes: torch.Tensor,
    graph_move_mask: torch.Tensor,
    stop_mask: torch.Tensor,
) -> torch.Tensor:
    child_path_token_ids = parent_path_token_ids.clone()
    if bool(graph_move_mask.any().item()):
        child_path_token_ids = append_relation_and_node_tokens_inplace(
            path_token_ids=child_path_token_ids,
            num_steps=parent_num_steps,
            relation_ids=relation_ids,
            target_nodes=target_nodes,
            active_mask=graph_move_mask,
        )
    if bool(stop_mask.any().item()):
        child_path_token_ids = append_stop_token_inplace(
            path_token_ids=child_path_token_ids,
            num_steps=parent_num_steps,
            active_mask=stop_mask,
        )
    return child_path_token_ids


def build_flat_prefix_keys(
    *,
    sample_ids: list[str],
    graph_ids: torch.Tensor,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    flat_absorbing_mask: torch.Tensor,
    node_entity_ids_by_abs_node: torch.Tensor,
    num_nodes: int,
) -> list[PrefixKey]:
    total_states = int(flat_current_abs_nodes.numel())
    if tuple(graph_ids.shape) != tuple(flat_current_abs_nodes.shape):
        raise ValueError(
            "graph_ids must align with flat_current_abs_nodes when building prefix keys. "
            f"graph_ids={tuple(graph_ids.shape)} states={tuple(flat_current_abs_nodes.shape)}."
        )
    if tuple(flat_absorbing_mask.shape) != tuple(flat_current_abs_nodes.shape):
        raise ValueError(
            "flat_absorbing_mask must align with flat_current_abs_nodes when building prefix keys. "
            f"absorbing={tuple(flat_absorbing_mask.shape)} states={tuple(flat_current_abs_nodes.shape)}."
        )
    if total_states == 0:
        return []
    if flat_path_token_ids is None:
        if bool((flat_num_steps != 0).any().item()):
            raise ValueError(
                "Exact prefix keys require path_token_ids for non-root rows."
            )
        flat_path_token_ids = flat_current_abs_nodes.unsqueeze(1)
    assert flat_path_token_ids is not None
    entity_rows, entity_mask = build_flat_visited_entity_id_rows(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
    )
    graph_ids_cpu = graph_ids.detach().cpu().to(dtype=torch.long).tolist()
    current_nodes_cpu = (
        flat_current_abs_nodes.detach().cpu().to(dtype=torch.long).tolist()
    )
    num_steps_cpu = flat_num_steps.detach().cpu().to(dtype=torch.long).tolist()
    absorbing_cpu = flat_absorbing_mask.detach().cpu().to(dtype=torch.bool).tolist()
    path_rows_cpu = flat_path_token_ids.detach().cpu().to(dtype=torch.long).tolist()
    entity_rows_cpu = entity_rows.detach().cpu().to(dtype=torch.long).tolist()
    entity_mask_cpu = entity_mask.detach().cpu().to(dtype=torch.bool).tolist()
    keys: list[PrefixKey] = []
    for row_idx in range(total_states):
        prefix_length = (
            (2 * int(num_steps_cpu[row_idx])) + 1 + int(absorbing_cpu[row_idx])
        )
        token_ids = tuple(
            int(value) for value in path_rows_cpu[row_idx][:prefix_length]
        )
        visited_entity_ids = tuple(
            int(entity)
            for entity, keep in zip(entity_rows_cpu[row_idx], entity_mask_cpu[row_idx])
            if keep
        )
        graph_idx = int(graph_ids_cpu[row_idx])
        keys.append(
            PrefixKey(
                sample_id=str(sample_ids[graph_idx]),
                current_node=int(current_nodes_cpu[row_idx]),
                num_steps=int(num_steps_cpu[row_idx]),
                is_absorbing=bool(absorbing_cpu[row_idx]),
                token_ids=token_ids,
                visited_entity_ids=visited_entity_ids,
            )
        )
    return keys


__all__ = [
    "PrefixKey",
    "build_child_prefix_token_ids",
    "build_flat_prefix_keys",
    "build_flat_visited_entity_id_rows",
    "build_flat_visited_entity_sketch",
]
