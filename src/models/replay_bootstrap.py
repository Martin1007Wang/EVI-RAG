from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from src.models.replay import TrajectoryPlan, TrajectoryTrace, reward_priority_from_log_reward
from src.models.rollout.traces import resolve_batch_sample_ids, resolve_edge_ptr
from src.utils.reward_utils import build_anchor_induced_edge_mask

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch
    from src.models.reward import RewardModel


def build_teacher_edge_trace(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    is_target_mask: torch.Tensor,
    positive_edge_mask: torch.Tensor,
    node_to_target_distance: torch.Tensor,
    max_steps: int,
    undirected: bool = True,
) -> tuple[int, ...] | None:
    """Build one legal shortest-path teacher trace for a single graph.

    The returned edge ids are graph-local and ordered by rollout expansion time.
    If no legal shortest-path rollout can reach a target within ``max_steps``,
    the function returns ``None``.
    """

    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
        )

    num_nodes = int(is_anchor_mask.numel())
    if is_target_mask.numel() != num_nodes:
        raise ValueError(
            "is_target_mask length must equal num_nodes, got "
            f"{is_target_mask.numel()} and {num_nodes}."
        )
    num_edges = int(edge_index.shape[1])
    if positive_edge_mask.numel() != num_edges:
        raise ValueError(
            "positive_edge_mask length must equal num_edges, got "
            f"{positive_edge_mask.numel()} and {num_edges}."
        )
    if node_to_target_distance.numel() != num_nodes:
        raise ValueError(
            "node_to_target_distance length must equal num_nodes, got "
            f"{node_to_target_distance.numel()} and {num_nodes}."
        )

    active_nodes = is_anchor_mask.clone()
    active_edges = build_anchor_induced_edge_mask(edge_index, active_nodes)
    if bool((active_nodes & is_target_mask).any().item()):
        return ()

    src = edge_index[0]
    dst = edge_index[1]
    trace: list[int] = []
    for _ in range(max_steps):
        src_active = active_nodes.index_select(0, src)
        dst_active = active_nodes.index_select(0, dst)
        src_dist = node_to_target_distance.index_select(0, src)
        dst_dist = node_to_target_distance.index_select(0, dst)

        expand_from_src = (
            src_active
            & ~dst_active
            & src_dist.ge(1)
            & dst_dist.ge(0)
            & src_dist.eq(dst_dist + 1)
        )
        expand_from_dst = (
            dst_active
            & ~src_active
            & dst_dist.ge(1)
            & src_dist.ge(0)
            & dst_dist.eq(src_dist + 1)
        )
        valid_teacher = (
            positive_edge_mask & ~active_edges & (expand_from_src | expand_from_dst)
        )
        if undirected:
            valid_teacher = valid_teacher & src.lt(dst)
        candidate_ids = torch.nonzero(valid_teacher, as_tuple=False).view(-1)
        if candidate_ids.numel() == 0:
            return None

        candidate_new_dist = torch.where(
            expand_from_src.index_select(0, candidate_ids),
            dst_dist.index_select(0, candidate_ids),
            src_dist.index_select(0, candidate_ids),
        )
        best_idx = candidate_new_dist.argmin()
        chosen_edge = int(candidate_ids[best_idx].item())
        trace.append(chosen_edge)
        active_edges[chosen_edge] = True
        active_nodes[src[chosen_edge]] = True
        active_nodes[dst[chosen_edge]] = True
        if bool((active_nodes & is_target_mask).any().item()):
            return tuple(trace)

    return None


def bootstrap_teacher_traces(
    *,
    datamodule: Any,
    reward_model: "RewardModel",
    max_steps: int,
    count: int,
    batch_size: int,
    insert_step: int,
    priority_epsilon: float,
    max_priority_log_reward: float,
    teacher_priority_multiplier: float,
    device: torch.device,
    undirected: bool,
) -> list[TrajectoryTrace]:
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if count == 0:
        return []

    train_dataset = getattr(datamodule, "train_dataset", None)
    if train_dataset is None:
        raise RuntimeError(
            "Teacher replay bootstrap requires datamodule.train_dataset. "
            "Did you call setup('fit')?"
        )
    sample_ids = list(getattr(train_dataset, "sample_ids", []))
    if not sample_ids:
        return []

    traces: list[TrajectoryTrace] = []
    offset = 0
    while offset < len(sample_ids) and len(traces) < count:
        batch_ids = sample_ids[offset : offset + batch_size]
        offset += batch_size
        base_graph = datamodule.build_train_batch_from_ids(batch_ids).to(device)
        traces.extend(
            _build_teacher_traces_for_batch(
                base_graph=base_graph,
                reward_model=reward_model,
                max_steps=max_steps,
                insert_step=insert_step,
                priority_epsilon=priority_epsilon,
                max_priority_log_reward=max_priority_log_reward,
                teacher_priority_multiplier=teacher_priority_multiplier,
                undirected=undirected,
            )
        )
    return traces[:count]


def build_teacher_plans_for_batch(
    *,
    base_graph: "RetrievalBatch",
    max_steps: int,
    prefix_len: int,
    undirected: bool,
    source: str = "guided",
) -> tuple[TrajectoryPlan | None, ...]:
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}.")
    sample_ids = resolve_batch_sample_ids(base_graph)
    if sample_ids is None:
        raise RuntimeError("Guided teacher plans require RetrievalBatch.sample_id.")
    if prefix_len == 0:
        return tuple(None for _ in range(int(base_graph.num_graphs)))

    edge_ptr = resolve_edge_ptr(base_graph)
    plans: list[TrajectoryPlan | None] = []
    for graph_id in range(int(base_graph.num_graphs)):
        node_start = int(base_graph.ptr[graph_id].item())
        node_end = int(base_graph.ptr[graph_id + 1].item())
        edge_start = int(edge_ptr[graph_id].item())
        edge_end = int(edge_ptr[graph_id + 1].item())
        local_trace = build_teacher_edge_trace(
            edge_index=base_graph.edge_index[:, edge_start:edge_end] - node_start,
            is_anchor_mask=base_graph.is_anchor_mask[node_start:node_end],
            is_target_mask=base_graph.is_target_mask[node_start:node_end],
            positive_edge_mask=base_graph.positive_edge_mask[edge_start:edge_end],
            node_to_target_distance=base_graph.node_to_target_distance[node_start:node_end],
            max_steps=max_steps,
            undirected=undirected,
        )
        if local_trace is None or len(local_trace) == 0:
            plans.append(None)
            continue
        plans.append(
            TrajectoryPlan(
                sample_id=sample_ids[graph_id],
                edge_trace_local=local_trace,
                mode="prefix",
                forced_prefix_len=prefix_len,
                source=source,
            )
        )
    return tuple(plans)


def _build_teacher_traces_for_batch(
    *,
    base_graph: "RetrievalBatch",
    reward_model: "RewardModel",
    max_steps: int,
    insert_step: int,
    priority_epsilon: float,
    max_priority_log_reward: float,
    teacher_priority_multiplier: float,
    undirected: bool,
) -> list[TrajectoryTrace]:
    sample_ids = resolve_batch_sample_ids(base_graph)
    if sample_ids is None:
        raise RuntimeError(
            "Teacher replay bootstrap requires RetrievalBatch.sample_id."
        )

    edge_ptr = resolve_edge_ptr(base_graph)
    src = base_graph.edge_index[0]
    dst = base_graph.edge_index[1]
    active_nodes = base_graph.is_anchor_mask.clone()
    active_edges = build_anchor_induced_edge_mask(base_graph.edge_index, active_nodes)

    graph_ids: list[int] = []
    edge_traces_local: list[tuple[int, ...]] = []
    for graph_id in range(int(base_graph.num_graphs)):
        node_start = int(base_graph.ptr[graph_id].item())
        node_end = int(base_graph.ptr[graph_id + 1].item())
        edge_start = int(edge_ptr[graph_id].item())
        edge_end = int(edge_ptr[graph_id + 1].item())
        local_trace = build_teacher_edge_trace(
            edge_index=base_graph.edge_index[:, edge_start:edge_end] - node_start,
            is_anchor_mask=base_graph.is_anchor_mask[node_start:node_end],
            is_target_mask=base_graph.is_target_mask[node_start:node_end],
            positive_edge_mask=base_graph.positive_edge_mask[edge_start:edge_end],
            node_to_target_distance=base_graph.node_to_target_distance[
                node_start:node_end
            ],
            max_steps=max_steps,
            undirected=undirected,
        )
        if local_trace is None:
            continue
        graph_ids.append(graph_id)
        edge_traces_local.append(local_trace)
        if local_trace:
            global_edge_ids = torch.tensor(
                [edge_start + edge_id for edge_id in local_trace],
                dtype=torch.long,
                device=base_graph.edge_index.device,
            )
            active_edges[global_edge_ids] = True
            active_nodes[src.index_select(0, global_edge_ids)] = True
            active_nodes[dst.index_select(0, global_edge_ids)] = True

    if not graph_ids:
        return []

    rewards = reward_model(
        base_graph=base_graph,
        active_nodes=active_nodes,
        active_edges=active_edges,
    )
    graph_index = torch.tensor(graph_ids, dtype=torch.long, device=rewards.device)
    graph_rewards = rewards.index_select(0, graph_index)
    priorities = reward_priority_from_log_reward(
        graph_rewards,
        epsilon=priority_epsilon,
        max_priority_log_reward=max_priority_log_reward,
    )
    priorities = priorities * float(teacher_priority_multiplier)
    return [
        TrajectoryTrace(
            sample_id=sample_ids[graph_id],
            edge_trace_local=trace_local,
            traj_len=len(trace_local) + 1,
            terminal_log_reward=float(graph_rewards[idx].detach().cpu()),
            priority=float(priorities[idx].detach().cpu()),
            insert_step=insert_step,
            source="teacher_bootstrap",
            positive_edge_hit_count=len(trace_local),
            positive_prefix_hit_len=len(trace_local),
        )
        for idx, (graph_id, trace_local) in enumerate(zip(graph_ids, edge_traces_local))
    ]


__all__ = [
    "bootstrap_teacher_traces",
    "build_teacher_edge_trace",
    "build_teacher_plans_for_batch",
]
