from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import State
from src.weaver.transition import (
    ExpansionBatch,
    SampleMeta,
    SRC_UNKNOWN,
    TerminalBatch,
    TrainingBatch,
)


@dataclass(frozen=True, slots=True)
class ReplayTrajectory:
    graph_id: int
    edge_ids: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.edge_ids) == 0


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    trajectories: tuple[ReplayTrajectory, ...]

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True, slots=True)
class ReplaySampleBudget:
    policy_rollout: int
    replay_expand: int

    @property
    def total(self) -> int:
        return int(self.policy_rollout + self.replay_expand)


@dataclass(frozen=True, slots=True)
class ReplayTargetView:
    target_node_id: int
    graph_id: int
    node_start: int
    edge_start: int
    anchor_target_distance: int
    node_distances: torch.Tensor
    edge_counts: torch.Tensor

    def node_distance(self, node_id: int) -> int:
        local_node_id = int(node_id) - int(self.node_start)
        return int(self.node_distances[local_node_id].item())

    def node_distances_for(self, node_ids: torch.Tensor) -> torch.Tensor:
        local_node_ids = node_ids.to(
            device=self.node_distances.device,
            dtype=torch.long,
        ) - int(self.node_start)
        return self.node_distances.index_select(0, local_node_ids)

    def edge_counts_for(self, edge_ids: torch.Tensor) -> torch.Tensor:
        local_edge_ids = edge_ids.to(
            device=self.edge_counts.device,
            dtype=torch.long,
        ) - int(self.edge_start)
        return self.edge_counts.index_select(0, local_edge_ids)


class ReplaySource:
    def __init__(
        self,
        *,
        expand_budget: int,
    ) -> None:
        self.expand_budget = int(expand_budget)

    @torch.no_grad()
    def sample_from_rollouts(
        self,
        *,
        batch: RetrievalBatch,
        context: GraphContext,
        rollouts: Sequence[RolloutResult],
        num_trajectories: int,
    ) -> ReplayBatch | None:
        num_trajectories = int(num_trajectories)
        if num_trajectories <= 0:
            return None

        trajectories = replay_trajectories(
            batch=batch,
            context=context,
            rollouts=rollouts,
            budget=self.expand_budget,
            max_trajectories=num_trajectories,
        )
        if not trajectories:
            return None
        return ReplayBatch(trajectories=tuple(trajectories))


class ReplayBuilder:
    def __init__(
        self,
        *,
        expand_budget: int,
    ) -> None:
        self.expand_budget = int(expand_budget)

    def build(
        self,
        *,
        graph: GraphContext,
        trajectories: ReplayBatch,
    ) -> TrainingBatch:
        return training_from_trajectories(
            trajectories=trajectories.trajectories,
            graph=graph,
            budget=self.expand_budget,
        )


def training_from_rollouts(
    *,
    rollouts: Sequence[RolloutResult],
    budget: int,
    context: GraphContext,
) -> TrainingBatch | None:
    batches: list[TrainingBatch] = []
    trajectory_offset = 0

    for rollout in rollouts:
        graph_ids = rollout.source_graph_id.to(
            device=context.device,
            dtype=torch.long,
        ).view(-1)
        current = initial_state_for_graph_ids(
            context=context,
            graph_ids=graph_ids,
        )
        rollout_trajectory_ids = torch.arange(
            graph_ids.numel(),
            dtype=torch.long,
            device=context.device,
        ) + trajectory_offset
        trajectory_offset += graph_ids.numel()

        exp_parts: list[ExpansionBatch] = []
        term_parts: list[TerminalBatch] = []

        for step in range(int(rollout.max_steps)):
            expand_rows = rollout.expand_mask[:, step].to(
                device=context.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).flatten()
            if expand_rows.numel() > 0:
                edge_ids = rollout.selected_edge_ids[:, step].to(
                    device=context.device,
                    dtype=torch.long,
                ).index_select(0, expand_rows)
                parent = current.select_rows(expand_rows)
                child = parent.expand(
                    graph=context,
                    rows=torch.arange(
                        parent.num_rows,
                        dtype=torch.long,
                        device=context.device,
                    ),
                    edge_ids=edge_ids,
                    expand_budget=int(budget),
                )
                exp_parts.append(
                    ExpansionBatch(
                        parent=parent,
                        child=child,
                        edge_ids=edge_ids,
                        meta=SampleMeta(
                            trajectory_ids=rollout_trajectory_ids.index_select(0, expand_rows),
                            step_ids=torch.full(
                                (expand_rows.numel(),),
                                step,
                                dtype=torch.long,
                                device=context.device,
                            ),
                            source_ids=torch.full(
                                (expand_rows.numel(),),
                                SRC_UNKNOWN,
                                dtype=torch.long,
                                device=context.device,
                            ),
                        ),
                    )
                )
                current = current.expand(
                    graph=context,
                    rows=expand_rows,
                    edge_ids=edge_ids,
                    expand_budget=int(budget),
                )

            stop_rows = rollout.stop_mask[:, step].to(
                device=context.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).flatten()
            if stop_rows.numel() > 0:
                hit_continue_steps = rollout_hit_continue_steps(
                    rollout=rollout,
                    context=context,
                    stop_rows=stop_rows,
                )
                term_parts.append(
                    TerminalBatch(
                        state=current.select_rows(stop_rows),
                        meta=SampleMeta(
                            trajectory_ids=rollout_trajectory_ids.index_select(0, stop_rows),
                            step_ids=torch.full(
                                (stop_rows.numel(),),
                                step,
                                dtype=torch.long,
                                device=context.device,
                            ),
                            source_ids=torch.full(
                                (stop_rows.numel(),),
                                SRC_UNKNOWN,
                                dtype=torch.long,
                                device=context.device,
                            ),
                        ),
                        forced_stop=rollout.forced_stop.to(
                            device=context.device,
                            dtype=torch.bool,
                        ).index_select(0, stop_rows),
                        hit_continue_steps=hit_continue_steps,
                    )
                )

        if not exp_parts and not term_parts:
            continue

        empty_state = initial_state_for_graph_ids(
            context=context,
            graph_ids=torch.empty(0, dtype=torch.long, device=context.device),
        )
        batches.append(
            TrainingBatch(
                expansions=ExpansionBatch.concat(exp_parts) if exp_parts else ExpansionBatch.empty_like(graph_like=empty_state),
                terminals=TerminalBatch.concat(term_parts) if term_parts else TerminalBatch.empty_like(graph_like=empty_state),
            )
        )

    if not batches:
        return None
    return TrainingBatch.concat_reindex_trajectories(batches)


def replay_trajectories(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    budget: int,
    max_trajectories: int | None = None,
    rollouts: Sequence[RolloutResult] = (),
) -> list[ReplayTrajectory]:
    max_trajectories = None if max_trajectories is None else int(max_trajectories)
    if max_trajectories is not None and max_trajectories <= 0:
        return []

    targets = batch.reachable_target_node_ids.to(
        device=context.device,
        dtype=torch.long,
    ).view(-1)
    if targets.numel() == 0:
        return []

    target_graph = context.node_to_graph.index_select(0, targets)
    eligible_graphs = replay_graph_ids(
        targets=targets,
        target_graph=target_graph,
        context=context,
        rollouts=rollouts,
        budget=int(budget),
    )
    if not eligible_graphs:
        return []

    trajectories: list[ReplayTrajectory] = []
    target_views = build_replay_target_views(
        batch=batch,
        context=context,
        targets=targets,
        target_graph=target_graph,
    )

    for graph_id in range(int(context.num_graphs)):
        if graph_id not in eligible_graphs:
            continue
        graph_target_positions = target_graph.eq(int(graph_id)).nonzero(as_tuple=False).view(-1)
        if graph_target_positions.numel() == 0:
            continue
        for target_pos in graph_target_positions.tolist():
            edge_path = precomputed_shortest_edge_path(
                batch=batch,
                context=context,
                target_view=target_views[int(target_pos)],
                budget=int(budget),
            )
            if not edge_path:
                continue
            trajectories.append(
                ReplayTrajectory(
                    graph_id=int(graph_id),
                    edge_ids=tuple(int(edge_id) for edge_id in edge_path[: int(budget)]),
                )
            )

    if max_trajectories is not None and len(trajectories) > max_trajectories:
        order = torch.randperm(len(trajectories), device=context.device)[:max_trajectories].cpu()
        trajectories = [trajectories[int(i)] for i in order.tolist()]
    return trajectories


def training_from_trajectories(
    *,
    trajectories: Sequence[ReplayTrajectory],
    graph: GraphContext,
    budget: int,
) -> TrainingBatch:
    batches: list[TrainingBatch] = []
    for trajectory_id, trajectory in enumerate(trajectories):
        if trajectory.is_empty:
            continue
        current = initial_state_for_graph_ids(
            context=graph,
            graph_ids=torch.tensor([int(trajectory.graph_id)], dtype=torch.long, device=graph.device),
        )
        exp_parts: list[ExpansionBatch] = []
        for step, edge_id in enumerate(trajectory.edge_ids):
            edge_ids = torch.tensor([int(edge_id)], dtype=torch.long, device=graph.device)
            parent = current
            child = current.expand(
                graph=graph,
                rows=torch.zeros(1, dtype=torch.long, device=graph.device),
                edge_ids=edge_ids,
                expand_budget=int(budget),
            )
            exp_parts.append(
                ExpansionBatch(
                    parent=parent,
                    child=child,
                    edge_ids=edge_ids,
                    meta=SampleMeta(
                        trajectory_ids=torch.tensor([trajectory_id], dtype=torch.long, device=graph.device),
                        step_ids=torch.tensor([step], dtype=torch.long, device=graph.device),
                        source_ids=torch.full((1,), SRC_UNKNOWN, dtype=torch.long, device=graph.device),
                    ),
                )
            )
            current = child

        term = TerminalBatch(
            state=current,
            meta=SampleMeta(
                trajectory_ids=torch.tensor([trajectory_id], dtype=torch.long, device=graph.device),
                step_ids=torch.tensor([len(trajectory.edge_ids)], dtype=torch.long, device=graph.device),
                source_ids=torch.full((1,), SRC_UNKNOWN, dtype=torch.long, device=graph.device),
            ),
            forced_stop=torch.zeros(1, dtype=torch.bool, device=graph.device),
            hit_continue_steps=torch.zeros(1, dtype=torch.long, device=graph.device),
        )
        batches.append(
            TrainingBatch(
                expansions=ExpansionBatch.concat(exp_parts),
                terminals=term,
            )
        )

    if not batches:
        empty_state = initial_state_for_graph_ids(
            context=graph,
            graph_ids=torch.empty(0, dtype=torch.long, device=graph.device),
        )
        return TrainingBatch(
            expansions=ExpansionBatch.empty_like(graph_like=empty_state),
            terminals=TerminalBatch.empty_like(graph_like=empty_state),
        )
    return TrainingBatch.concat_reindex_trajectories(batches)


def rollout_hit_continue_steps(
    *,
    rollout: RolloutResult,
    context: GraphContext,
    stop_rows: torch.Tensor,
) -> torch.Tensor:
    stop_rows = stop_rows.to(device=context.device, dtype=torch.long).view(-1)
    if stop_rows.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=context.device)

    selected = rollout.selected_edge_ids.to(device=context.device, dtype=torch.long)
    expand = rollout.expand_mask.to(device=context.device, dtype=torch.bool)
    edge_index = context.edge_index.to(device=context.device, dtype=torch.long)

    active = anchor_mask_for_graph_rows(
        graph=context,
        graph_ids=rollout.source_graph_id.to(device=context.device, dtype=torch.long),
    )
    target_mask = context.anchor_mask.new_zeros(
        int(context.num_nodes),
        dtype=torch.bool,
        device=context.device,
    )
    hit_continue_steps = torch.zeros(
        int(rollout.source_graph_id.numel()),
        dtype=torch.long,
        device=context.device,
    )
    target_rows = stop_rows.new_empty((0,))
    del target_rows
    return hit_continue_steps.index_select(0, stop_rows)


def replay_graph_ids(
    *,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
    context: GraphContext,
    rollouts: Sequence[RolloutResult],
    budget: int,
) -> set[int]:
    del budget
    graphs_with_targets = {int(x) for x in target_graph.tolist()}
    if not rollouts:
        return graphs_with_targets
    hit_graphs = rollout_hit_graph_ids(
        rollouts=rollouts,
        targets=targets,
        context=context,
    )
    return graphs_with_targets - hit_graphs


def rollout_hit_graph_ids(
    *,
    rollouts: Sequence[RolloutResult],
    targets: torch.Tensor,
    context: GraphContext,
) -> set[int]:
    target_mask = torch.zeros(
        int(context.num_nodes),
        dtype=torch.bool,
        device=context.device,
    )
    if targets.numel() > 0:
        target_mask[targets.to(device=context.device, dtype=torch.long)] = True

    hit_graphs: set[int] = set()
    for rollout in rollouts:
        graph_ids = rollout.source_graph_id.to(
            device=context.device,
            dtype=torch.long,
        ).view(-1)
        if graph_ids.numel() == 0:
            continue
        state = initial_state_for_graph_ids(
            context=context,
            graph_ids=graph_ids,
        )
        for step in range(int(rollout.max_steps)):
            expand_rows = rollout.expand_mask[:, step].to(
                device=context.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).flatten()
            if expand_rows.numel() == 0:
                continue
            edge_ids = rollout.selected_edge_ids[:, step].to(
                device=context.device,
                dtype=torch.long,
            ).index_select(0, expand_rows)
            state = state.expand(
                graph=context,
                rows=expand_rows,
                edge_ids=edge_ids,
                expand_budget=int(rollout.expand_budget),
            )
        has_target = (state.active_node_mask & target_mask.view(1, -1)).any(dim=1)
        for graph_id in graph_ids[has_target].tolist():
            hit_graphs.add(int(graph_id))
    return hit_graphs


def initial_state_for_graph_ids(
    *,
    context: GraphContext,
    graph_ids: torch.Tensor,
) -> State:
    return State.initial(
        graph=context,
        graph_ids=graph_ids.to(
            device=context.device,
            dtype=torch.long,
        ).view(-1),
    )


def build_replay_target_views(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
) -> tuple[ReplayTargetView, ...]:
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [T], got {tuple(targets.shape)}.")
    if target_graph.shape != targets.shape:
        raise ValueError(
            "target_graph must have the same shape as targets: "
            f"{tuple(target_graph.shape)} != {tuple(targets.shape)}."
        )

    device = context.device
    node_ptr = batch.ptr.to(device=device, dtype=torch.long)
    edge_ptr = _edge_ptr_from_edge_batch(
        edge_batch=batch.edge_batch,
        num_graphs=int(context.num_graphs),
        device=device,
    )

    node_target_distances = batch.node_target_distances_flat.to(
        device=device,
        dtype=torch.long,
    )
    node_target_edge_counts = batch.node_target_shortest_path_edge_count_flat.to(
        device=device,
        dtype=torch.float32,
    )
    anchor_forward = batch.anchor_node_forward_distances_flat.to(
        device=device,
        dtype=torch.long,
    )
    target_batch = _target_batch_from_reachable_targets(
        batch=batch,
        targets=targets,
        target_graph=target_graph,
        device=device,
    )
    target_graph_ptr = _graph_ptr_from_item_batch(
        item_batch=target_batch,
        num_graphs=int(context.num_graphs),
        device=device,
    )
    target_item_offsets = _item_offsets_within_graph(
        item_batch=target_batch,
        num_graphs=int(context.num_graphs),
        device=device,
    )
    node_target_graph_ptr = _graph_ptr_from_flat_supervision(
        item_batch=target_batch,
        graph_sizes=node_ptr[1:] - node_ptr[:-1],
        device=device,
    )
    edge_target_graph_ptr = _graph_ptr_from_flat_supervision(
        item_batch=target_batch,
        graph_sizes=edge_ptr[1:] - edge_ptr[:-1],
        device=device,
    )

    views: list[ReplayTargetView] = []
    for target_pos, (target_id, graph_id) in enumerate(
        zip(targets.tolist(), target_graph.tolist(), strict=True)
    ):
        node_start = int(node_ptr[graph_id].item())
        node_end = int(node_ptr[graph_id + 1].item())
        edge_start = int(edge_ptr[graph_id].item())
        edge_end = int(edge_ptr[graph_id + 1].item())
        local_target_pos = int(target_item_offsets[target_pos].item())
        graph_target_count = int(
            target_graph_ptr[graph_id + 1].item() - target_graph_ptr[graph_id].item()
        )
        graph_num_nodes = node_end - node_start
        graph_num_edges = edge_end - edge_start

        node_graph_start = int(node_target_graph_ptr[graph_id].item())
        node_graph_end = int(node_target_graph_ptr[graph_id + 1].item())
        edge_graph_start = int(edge_target_graph_ptr[graph_id].item())
        edge_graph_end = int(edge_target_graph_ptr[graph_id + 1].item())

        if node_graph_end - node_graph_start != graph_target_count * graph_num_nodes:
            raise ValueError(
                "node_target_distances_flat graph block has inconsistent size: "
                f"graph={graph_id}, targets={graph_target_count}, nodes={graph_num_nodes}."
            )
        if edge_graph_end - edge_graph_start != graph_target_count * graph_num_edges:
            raise ValueError(
                "node_target_shortest_path_edge_count_flat graph block has inconsistent size: "
                f"graph={graph_id}, targets={graph_target_count}, edges={graph_num_edges}."
            )

        node_slice_start = node_graph_start + local_target_pos * graph_num_nodes
        node_slice_end = node_slice_start + graph_num_nodes
        edge_slice_start = edge_graph_start + local_target_pos * graph_num_edges
        edge_slice_end = edge_slice_start + graph_num_edges

        node_distances = node_target_distances[node_slice_start:node_slice_end]
        edge_counts = node_target_edge_counts[edge_slice_start:edge_slice_end]
        if int(node_distances.numel()) != node_end - node_start:
            raise ValueError(
                "node_target_distances_flat is inconsistent with graph node slices: "
                f"target={target_id}, graph={graph_id}."
            )
        if int(edge_counts.numel()) != edge_end - edge_start:
            raise ValueError(
                "node_target_shortest_path_edge_count_flat is inconsistent with graph edge slices: "
                f"target={target_id}, graph={graph_id}."
            )

        views.append(
            ReplayTargetView(
                target_node_id=int(target_id),
                graph_id=int(graph_id),
                node_start=node_start,
                edge_start=edge_start,
                anchor_target_distance=int(anchor_forward[int(target_id)].item()),
                node_distances=node_distances,
                edge_counts=edge_counts,
            )
        )

    return tuple(views)


def _edge_ptr_from_edge_batch(
    *,
    edge_batch: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    edge_batch = edge_batch.to(device=device, dtype=torch.long).view(-1)
    edge_counts = torch.bincount(edge_batch, minlength=int(num_graphs))
    edge_ptr = torch.empty(
        int(num_graphs) + 1,
        dtype=torch.long,
        device=device,
    )
    edge_ptr[0] = 0
    edge_ptr[1:] = torch.cumsum(edge_counts, dim=0)
    return edge_ptr


def _target_ptr_from_target_graph(
    *,
    target_graph: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    target_graph = target_graph.to(device=device, dtype=torch.long).view(-1)
    target_counts = torch.bincount(target_graph, minlength=int(num_graphs))
    target_ptr = torch.empty(
        int(num_graphs) + 1,
        dtype=torch.long,
        device=device,
    )
    target_ptr[0] = 0
    target_ptr[1:] = torch.cumsum(target_counts, dim=0)
    return target_ptr


def _target_batch_from_reachable_targets(
    *,
    batch: RetrievalBatch,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    target_batch = getattr(batch, "reachable_target_node_ids_batch", None)
    if target_batch is None:
        return target_graph.to(device=device, dtype=torch.long).view(-1)

    target_batch = target_batch.to(device=device, dtype=torch.long).view(-1)
    if target_batch.shape != targets.shape:
        raise ValueError(
            "reachable_target_node_ids_batch must match reachable_target_node_ids shape: "
            f"{tuple(target_batch.shape)} != {tuple(targets.shape)}."
        )
    return target_batch


def _graph_ptr_from_item_batch(
    *,
    item_batch: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    item_batch = item_batch.to(device=device, dtype=torch.long).view(-1)
    counts = torch.bincount(item_batch, minlength=int(num_graphs))
    ptr = torch.empty(
        int(num_graphs) + 1,
        dtype=torch.long,
        device=device,
    )
    ptr[0] = 0
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def _item_offsets_within_graph(
    *,
    item_batch: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    item_batch = item_batch.to(device=device, dtype=torch.long).view(-1)
    offsets = torch.empty_like(item_batch)
    counts = torch.zeros(int(num_graphs), dtype=torch.long, device=device)
    for idx in range(int(item_batch.numel())):
        graph_id = int(item_batch[idx].item())
        offsets[idx] = counts[graph_id]
        counts[graph_id] += 1
    return offsets


def _graph_ptr_from_flat_supervision(
    *,
    item_batch: torch.Tensor,
    graph_sizes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    item_batch = item_batch.to(device=device, dtype=torch.long).view(-1)
    graph_sizes = graph_sizes.to(device=device, dtype=torch.long).view(-1)
    if graph_sizes.numel() == 0:
        ptr = torch.zeros(1, dtype=torch.long, device=device)
        return ptr

    counts = torch.bincount(item_batch, minlength=int(graph_sizes.numel()))
    flat_sizes = counts * graph_sizes
    ptr = torch.empty(
        int(graph_sizes.numel()) + 1,
        dtype=torch.long,
        device=device,
    )
    ptr[0] = 0
    ptr[1:] = torch.cumsum(flat_sizes, dim=0)
    return ptr


def precomputed_shortest_edge_path(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    target_view: ReplayTargetView,
    budget: int,
) -> list[int]:
    graph_id = int(target_view.graph_id)
    target_node_id = int(target_view.target_node_id)

    anchors = context.anchor_mask.nonzero(as_tuple=True)[0]
    anchor_graph = context.node_to_graph.index_select(0, anchors)
    graph_anchors = anchors[anchor_graph.eq(graph_id)]
    if graph_anchors.numel() == 0:
        return []

    current = choose_best_anchor(
        anchors=graph_anchors,
        target_view=target_view,
    )
    path: list[int] = []

    for _ in range(int(budget)):
        if current == target_node_id:
            break
        current_dist = target_view.node_distance(current)
        next_edge = choose_next_edge(
            context=context,
            target_view=target_view,
            current_node_id=current,
            current_distance=current_dist,
        )
        if next_edge is None:
            return []
        path.append(int(next_edge))
        current = int(context.edge_index[1, next_edge].item())

    return path if current == target_node_id else []


def choose_best_anchor(
    *,
    anchors: torch.Tensor,
    target_view: ReplayTargetView,
) -> int:
    distances = target_view.node_distances_for(anchors)
    best_pos = int(torch.argmin(distances).item())
    return int(anchors[best_pos].item())


def choose_next_edge(
    *,
    context: GraphContext,
    target_view: ReplayTargetView,
    current_node_id: int,
    current_distance: int,
) -> int | None:
    out_start = int(context.adjacency.out_ptr[current_node_id].item())
    out_end = int(context.adjacency.out_ptr[current_node_id + 1].item())
    if out_end <= out_start:
        return None

    edge_ids = context.edge_ids_by_src[out_start:out_end]
    dst_ids = context.edge_index[1].index_select(0, edge_ids)
    dst_dist = target_view.node_distances_for(dst_ids)
    edge_counts = target_view.edge_counts_for(edge_ids)

    valid = dst_dist.eq(current_distance - 1) & edge_counts.gt(0)
    if not bool(valid.any()):
        return None

    valid_edges = edge_ids[valid]
    valid_counts = edge_counts[valid]
    best_pos = int(torch.argmax(valid_counts).item())
    return int(valid_edges[best_pos].item())


__all__ = [
    "ReplayBatch",
    "ReplayBuilder",
    "ReplayTrajectory",
    "ReplaySampleBudget",
    "ReplaySource",
    "ReplayTargetView",
    "replay_trajectories",
    "replay_graph_ids",
    "rollout_hit_graph_ids",
    "training_from_trajectories",
    "training_from_rollouts",
]
