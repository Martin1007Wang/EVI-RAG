from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import State
from src.weaver.transition import ExpansionBatch, SampleMeta, SRC_UNKNOWN, TerminalBatch, TrainingBatch
from src.weaver.utility import TrueTerminalReward


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
    stats: ReplayStats = field(default_factory=lambda: ReplayStats())

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True, slots=True)
class ReplayStats:
    eligible_graphs: int = 0
    skipped_by_reward: int = 0
    generated_trajectories: int = 0
    covered_graphs: int = 0
    oracle_reward_mean: float = 0.0
    policy_best_reward_mean: float = 0.0
    reward_gap_mean: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplaySampleBudget:
    policy_rollout: int
    replay_expand: int

    @property
    def total(self) -> int:
        return int(self.policy_rollout + self.replay_expand)


@dataclass(frozen=True, slots=True)
class ReplayGraphLabelView:
    graph_id: int
    target_node_ids: torch.Tensor
    admissible_edge_ids: torch.Tensor
    admissible_edge_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayStateNode:
    key: tuple[int, ...]
    state: State
    reward: float
    predecessor_keys: tuple[tuple[int, ...], ...]
    predecessor_edge_ids: tuple[int, ...]
    trajectory_count: int


class ReplaySource:
    def __init__(
        self,
        *,
        expand_budget: int,
        skip_warmup_progress: float = 0.05,
        skip_gap_ema_beta: float = 0.9,
        skip_gap_margin: float = 0.0,
    ) -> None:
        self.expand_budget = int(expand_budget)
        self.skip_warmup_progress = float(skip_warmup_progress)
        self.skip_gap_ema_beta = float(skip_gap_ema_beta)
        self.skip_gap_margin = float(skip_gap_margin)
        self._reward_gap_ema: float | None = None

    @torch.no_grad()
    def sample_from_rollouts(
        self,
        *,
        batch: RetrievalBatch,
        context: GraphContext,
        rollouts: Sequence[RolloutResult],
        num_trajectories: int,
        reward_model: TrueTerminalReward | None = None,
        target_context: TargetContext | None = None,
        progress: float = 0.0,
    ) -> ReplayBatch | None:
        trajectories_per_graph = int(num_trajectories)
        if trajectories_per_graph <= 0:
            return None

        allow_reward_skip = (
            float(progress) >= self.skip_warmup_progress
            and self._reward_gap_ema is not None
            and self._reward_gap_ema <= self.skip_gap_margin
        )
        trajectories, stats = replay_trajectories_with_stats(
            batch=batch,
            context=context,
            rollouts=rollouts,
            budget=self.expand_budget,
            max_trajectories_per_graph=trajectories_per_graph,
            reward_model=reward_model,
            target_context=target_context,
            allow_reward_skip=allow_reward_skip,
            skip_gap_margin=self.skip_gap_margin,
        )
        if stats.eligible_graphs > 0:
            if self._reward_gap_ema is None:
                self._reward_gap_ema = float(stats.reward_gap_mean)
            else:
                beta = self.skip_gap_ema_beta
                self._reward_gap_ema = beta * self._reward_gap_ema + (1.0 - beta) * float(stats.reward_gap_mean)
        if not trajectories:
            return ReplayBatch(trajectories=(), stats=stats)
        return ReplayBatch(trajectories=tuple(trajectories), stats=stats)


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


def replay_trajectories(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    budget: int,
    max_trajectories: int | None = None,
    rollouts: Sequence[RolloutResult] = (),
) -> list[ReplayTrajectory]:
    max_per_graph = None if max_trajectories is None else int(max_trajectories)
    trajectories, _ = replay_trajectories_with_stats(
        batch=batch,
        context=context,
        budget=budget,
        max_trajectories_per_graph=max_per_graph,
        rollouts=rollouts,
    )
    return trajectories


def replay_trajectories_with_stats(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    budget: int,
    max_trajectories_per_graph: int | None = None,
    rollouts: Sequence[RolloutResult] = (),
    reward_model: TrueTerminalReward | None = None,
    target_context: TargetContext | None = None,
    allow_reward_skip: bool = False,
    skip_gap_margin: float = 0.0,
) -> tuple[list[ReplayTrajectory], ReplayStats]:
    max_per_graph = None if max_trajectories_per_graph is None else int(max_trajectories_per_graph)
    if max_per_graph is not None and max_per_graph <= 0:
        return [], ReplayStats()

    targets = batch.reachable_target_node_ids.to(
        device=context.device,
        dtype=torch.long,
    ).view(-1)
    if targets.numel() == 0:
        return [], ReplayStats()

    target_graph = context.node_to_graph.index_select(0, targets)
    eligible_graphs = replay_graph_ids(
        targets=targets,
        target_graph=target_graph,
        context=context,
    )
    if not eligible_graphs:
        return [], ReplayStats()

    if reward_model is None or target_context is None:
        raise ValueError("Reward-first replay requires reward_model and target_context.")

    graph_views = build_replay_graph_label_views(
        batch=batch,
        context=context,
        targets=targets,
        target_graph=target_graph,
    )
    rollout_reward = best_rollout_reward_by_graph(
        rollouts=rollouts,
        context=context,
        target_context=target_context,
        reward_model=reward_model,
    )

    trajectories: list[ReplayTrajectory] = []
    skipped_by_reward = 0
    covered_graphs: set[int] = set()
    oracle_rewards: list[float] = []
    policy_rewards: list[float] = []
    reward_gaps: list[float] = []

    for view in graph_views:
        graph_id = int(view.graph_id)
        if graph_id not in eligible_graphs:
            continue

        replay_states = enumerate_replay_state_dag(
            context=context,
            graph_view=view,
            budget=int(budget),
        )
        if not replay_states:
            continue

        terminal_nodes = score_replay_states(
            replay_states=replay_states,
            context=context,
            target_context=target_context,
            reward_model=reward_model,
        )
        if not terminal_nodes:
            continue

        best_oracle = max(node.reward for node in terminal_nodes)
        best_policy = rollout_reward.get(graph_id)
        oracle_rewards.append(float(best_oracle))
        if best_policy is not None:
            policy_rewards.append(float(best_policy))
            reward_gaps.append(max(0.0, float(best_oracle) - float(best_policy)))
        if (
            allow_reward_skip
            and best_policy is not None
            and best_policy >= best_oracle - float(skip_gap_margin)
        ):
            skipped_by_reward += 1
            continue

        sampled = sample_oracle_trajectories(
            graph_id=graph_id,
            replay_states=replay_states,
            terminal_nodes=terminal_nodes,
            max_trajectories=max_per_graph,
        )
        if not sampled:
            continue

        trajectories.extend(sampled)
        covered_graphs.add(graph_id)

    stats = ReplayStats(
        eligible_graphs=len(eligible_graphs),
        skipped_by_reward=skipped_by_reward,
        generated_trajectories=len(trajectories),
        covered_graphs=len(covered_graphs),
        oracle_reward_mean=float(sum(oracle_rewards) / len(oracle_rewards)) if oracle_rewards else 0.0,
        policy_best_reward_mean=float(sum(policy_rewards) / len(policy_rewards)) if policy_rewards else 0.0,
        reward_gap_mean=float(sum(reward_gaps) / len(reward_gaps)) if reward_gaps else 0.0,
    )
    return trajectories, stats


def training_from_trajectories(
    *,
    trajectories: Sequence[ReplayTrajectory],
    graph: GraphContext,
    budget: int,
) -> TrainingBatch:
    batches: list[TrainingBatch] = []
    for trajectory_id, trajectory in enumerate(trajectories):
        current = initial_state_for_graph_ids(
            context=graph,
            graph_ids=torch.tensor([int(trajectory.graph_id)], dtype=torch.long, device=graph.device),
            expand_budget=int(budget),
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
            stop_reason=torch.full(
                (1,),
                RolloutResult.POLICY_STOP,
                dtype=torch.long,
                device=graph.device,
            ),
        )
        batches.append(
            TrainingBatch(
                expansions=ExpansionBatch.concat(exp_parts) if exp_parts else ExpansionBatch.empty_like(graph_like=current),
                terminals=term,
            )
        )

    if not batches:
        empty_state = initial_state_for_graph_ids(
            context=graph,
            graph_ids=torch.empty(0, dtype=torch.long, device=graph.device),
            expand_budget=int(budget),
        )
        return TrainingBatch(
            expansions=ExpansionBatch.empty_like(graph_like=empty_state),
            terminals=TerminalBatch.empty_like(graph_like=empty_state),
        )
    return TrainingBatch.concat_reindex_trajectories(batches)


def replay_graph_ids(
    *,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
    context: GraphContext,
) -> set[int]:
    del targets, context
    return {int(x) for x in target_graph.tolist()}


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
            expand_budget=int(rollout.expand_budget),
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


@torch.no_grad()
def best_rollout_reward_by_graph(
    *,
    rollouts: Sequence[RolloutResult],
    context: GraphContext,
    target_context: TargetContext | None,
    reward_model: TrueTerminalReward | None,
) -> dict[int, float]:
    if not rollouts or reward_model is None or target_context is None:
        return {}

    out: dict[int, float] = {}
    for rollout in rollouts:
        state = terminal_state_from_rollout(
            rollout=rollout,
            context=context,
        )
        if state.num_rows == 0:
            continue
        reward = reward_model(
            state=state,
            graph_context=context,
            target_context=target_context,
        ).log_reward
        for graph_id, value in zip(state.graph_ids.tolist(), reward.tolist(), strict=True):
            current = out.get(int(graph_id))
            value = float(value)
            if current is None or value > current:
                out[int(graph_id)] = value
    return out


def terminal_state_from_rollout(
    *,
    rollout: RolloutResult,
    context: GraphContext,
) -> State:
    graph_ids = rollout.source_graph_id.to(
        device=context.device,
        dtype=torch.long,
    ).view(-1)
    state = initial_state_for_graph_ids(
        context=context,
        graph_ids=graph_ids,
        expand_budget=int(rollout.expand_budget),
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
    return state


@torch.no_grad()
def best_replay_reward(
    *,
    trajectories: Sequence[ReplayTrajectory],
    context: GraphContext,
    target_context: TargetContext,
    reward_model: TrueTerminalReward,
    budget: int,
) -> float:
    training = training_from_trajectories(
        trajectories=trajectories,
        graph=context,
        budget=int(budget),
    )
    if training.terminals.num_items <= 0:
        return float("-inf")
    reward = reward_model(
        state=training.terminals.state,
        graph_context=context,
        target_context=target_context,
    ).log_reward
    if reward.numel() == 0:
        return float("-inf")
    return float(reward.max().item())


def initial_state_for_graph_ids(
    *,
    context: GraphContext,
    graph_ids: torch.Tensor,
    expand_budget: int,
) -> State:
    return State.initial(
        graph=context,
        graph_ids=graph_ids.to(
            device=context.device,
            dtype=torch.long,
        ).view(-1),
        expand_budget=int(expand_budget),
    )


def build_replay_graph_label_views(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
) -> tuple[ReplayGraphLabelView, ...]:
    device = context.device
    target_mask_blocks = unflatten_target_edge_mask(
        batch=batch,
        context=context,
        targets=targets,
        target_graph=target_graph,
    )
    views: list[ReplayGraphLabelView] = []
    for graph_id in range(int(context.num_graphs)):
        graph_targets = target_graph.eq(graph_id).nonzero(as_tuple=False).flatten()
        if graph_targets.numel() == 0:
            continue
        edge_mask = target_mask_blocks.index_select(0, graph_targets)
        admissible_edge_mask = edge_mask.any(dim=0)
        admissible_edge_ids = admissible_edge_mask.nonzero(as_tuple=False).flatten()
        if admissible_edge_ids.numel() == 0:
            continue
        views.append(
            ReplayGraphLabelView(
                graph_id=graph_id,
                target_node_ids=targets.index_select(0, graph_targets),
                admissible_edge_ids=admissible_edge_ids,
                admissible_edge_mask=admissible_edge_mask,
            )
        )
    return tuple(views)


def unflatten_target_edge_mask(
    *,
    batch: RetrievalBatch,
    context: GraphContext,
    targets: torch.Tensor,
    target_graph: torch.Tensor,
) -> torch.Tensor:
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [T], got {tuple(targets.shape)}.")
    if target_graph.shape != targets.shape:
        raise ValueError(
            "target_graph must have the same shape as targets: "
            f"{tuple(target_graph.shape)} != {tuple(targets.shape)}."
        )

    edge_ptr = _edge_ptr_from_edge_batch(
        edge_batch=batch.edge_batch,
        num_graphs=int(context.num_graphs),
        device=context.device,
    )
    target_batch = _target_batch_from_reachable_targets(
        batch=batch,
        targets=targets,
        target_graph=target_graph,
        device=context.device,
    )
    target_offsets = _item_offsets_within_graph(
        item_batch=target_batch,
        num_graphs=int(context.num_graphs),
        device=context.device,
    )
    edge_target_graph_ptr = _graph_ptr_from_flat_supervision(
        item_batch=target_batch,
        graph_sizes=edge_ptr[1:] - edge_ptr[:-1],
        device=context.device,
    )

    flat = batch.node_target_shortest_path_edge_mask_flat.to(
        device=context.device,
        dtype=torch.bool,
    )
    out = torch.zeros(
        (int(targets.numel()), int(context.num_edges)),
        dtype=torch.bool,
        device=context.device,
    )

    for target_pos, graph_id in enumerate(target_graph.tolist()):
        edge_start = int(edge_ptr[graph_id].item())
        edge_end = int(edge_ptr[graph_id + 1].item())
        graph_num_edges = edge_end - edge_start
        local_target_pos = int(target_offsets[target_pos].item())
        graph_edge_flat_start = int(edge_target_graph_ptr[graph_id].item())
        slice_start = graph_edge_flat_start + local_target_pos * graph_num_edges
        slice_end = slice_start + graph_num_edges
        out[target_pos, edge_start:edge_end] = flat[slice_start:slice_end]

    return out


def enumerate_replay_state_dag(
    *,
    context: GraphContext,
    graph_view: ReplayGraphLabelView,
    budget: int,
) -> dict[tuple[int, ...], ReplayStateNode]:
    graph_id = int(graph_view.graph_id)
    initial = initial_state_for_graph_ids(
        context=context,
        graph_ids=torch.tensor([graph_id], dtype=torch.long, device=context.device),
        expand_budget=int(budget),
    )
    initial_key = state_key(initial)
    states: dict[tuple[int, ...], State] = {initial_key: initial}
    predecessors: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {initial_key: {}}
    current_layer: list[tuple[int, ...]] = [initial_key]

    for _ in range(int(budget)):
        next_layer: list[tuple[int, ...]] = []
        for key in current_layer:
            parent = states[key]
            frontier = parent.frontier(
                context,
                expand_budget=int(budget),
            )
            if frontier.edge_ids.numel() == 0:
                continue
            local_rows = frontier.row_ids.eq(0)
            edge_ids = frontier.edge_ids[local_rows]
            if edge_ids.numel() == 0:
                continue
            admissible = graph_view.admissible_edge_mask.index_select(0, edge_ids)
            edge_ids = edge_ids[admissible]
            if edge_ids.numel() == 0:
                continue
            for edge_id in edge_ids.tolist():
                child = parent.expand(
                    graph=context,
                    rows=torch.zeros(1, dtype=torch.long, device=context.device),
                    edge_ids=torch.tensor([int(edge_id)], dtype=torch.long, device=context.device),
                    expand_budget=int(budget),
                )
                child_key = state_key(child)
                pred_map = predecessors.setdefault(child_key, {})
                pred_map[key] = int(edge_id)
                if child_key in states:
                    continue
                states[child_key] = child
                next_layer.append(child_key)
        current_layer = next_layer

    trajectory_count = count_state_trajectories(
        predecessor_map=predecessors,
        root_key=initial_key,
        state_depths={key: len(key) for key in states},
    )

    nodes: dict[tuple[int, ...], ReplayStateNode] = {}
    for key, state in states.items():
        pred_items = sorted(predecessors.get(key, {}).items())
        predecessor_keys = tuple(parent_key for parent_key, _ in pred_items)
        predecessor_edge_ids = tuple(edge_id for _, edge_id in pred_items)
        nodes[key] = ReplayStateNode(
            key=key,
            state=state,
            reward=float("-inf"),
            predecessor_keys=predecessor_keys,
            predecessor_edge_ids=predecessor_edge_ids,
            trajectory_count=int(trajectory_count.get(key, 0)),
        )
    return nodes


def count_state_trajectories(
    *,
    predecessor_map: dict[tuple[int, ...], dict[tuple[int, ...], int]],
    root_key: tuple[int, ...],
    state_depths: dict[tuple[int, ...], int],
) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {root_key: 1}
    ordered_keys = sorted(state_depths, key=state_depths.__getitem__)
    for key in ordered_keys:
        if key == root_key:
            continue
        preds = predecessor_map.get(key, {})
        counts[key] = sum(counts[parent_key] for parent_key in preds)
    return counts


def score_replay_states(
    *,
    replay_states: dict[tuple[int, ...], ReplayStateNode],
    context: GraphContext,
    target_context: TargetContext,
    reward_model: TrueTerminalReward,
) -> list[ReplayStateNode]:
    ordered_keys = sorted(replay_states.keys(), key=lambda key: (len(key), key))
    states = State.concat([replay_states[key].state for key in ordered_keys])
    reward = reward_model(
        state=states,
        graph_context=context,
        target_context=target_context,
    ).log_reward
    best = float(reward.max().item()) if reward.numel() > 0 else float("-inf")
    terminal_nodes: list[ReplayStateNode] = []
    for idx, key in enumerate(ordered_keys):
        node = replay_states[key]
        scored = ReplayStateNode(
            key=node.key,
            state=node.state,
            reward=float(reward[idx].item()),
            predecessor_keys=node.predecessor_keys,
            predecessor_edge_ids=node.predecessor_edge_ids,
            trajectory_count=node.trajectory_count,
        )
        replay_states[key] = scored
        if scored.reward == best:
            terminal_nodes.append(scored)
    return terminal_nodes


def sample_oracle_trajectories(
    *,
    graph_id: int,
    replay_states: dict[tuple[int, ...], ReplayStateNode],
    terminal_nodes: Sequence[ReplayStateNode],
    max_trajectories: int | None,
) -> list[ReplayTrajectory]:
    if not terminal_nodes:
        return []

    limit = None if max_trajectories is None else int(max_trajectories)
    sequences: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    ordered_terminal_keys = sorted(
        (node.key for node in terminal_nodes),
        key=lambda key: (
            -replay_states[key].trajectory_count,
            len(key),
            key,
        ),
    )
    for key in ordered_terminal_keys:
        for sequence in enumerate_action_sequences(
            nodes_by_key=replay_states,
            terminal_key=key,
        ):
            if sequence in seen:
                continue
            seen.add(sequence)
            sequences.append(sequence)
    sequences.sort(key=lambda seq: (len(seq), seq))
    if limit is not None:
        sequences = sequences[:limit]
    return [
        ReplayTrajectory(
            graph_id=int(graph_id),
            edge_ids=sequence,
        )
        for sequence in sequences
    ]


def state_key(state: State) -> tuple[int, ...]:
    if state.num_rows != 1:
        raise ValueError(f"Replay state_key expects exactly one row, got {state.num_rows}.")
    return tuple(int(edge_id) for edge_id in state.edge_mask[0].nonzero(as_tuple=True)[0].tolist())


def enumerate_action_sequences(
    *,
    nodes_by_key: dict[tuple[int, ...], ReplayStateNode],
    terminal_key: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    memo: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}

    def visit(key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        cached = memo.get(key)
        if cached is not None:
            return cached
        node = nodes_by_key[key]
        if not node.predecessor_keys:
            memo[key] = ((),)
            return memo[key]

        out: list[tuple[int, ...]] = []
        for parent_key, edge_id in zip(node.predecessor_keys, node.predecessor_edge_ids, strict=True):
            parent_sequences = visit(parent_key)
            for parent_sequence in parent_sequences:
                out.append((*parent_sequence, int(edge_id)))
        deduped = tuple(sorted(set(out), key=lambda seq: (len(seq), seq)))
        memo[key] = deduped
        return deduped

    return visit(terminal_key)


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
        return torch.zeros(1, dtype=torch.long, device=device)

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


__all__ = [
    "ReplayBatch",
    "ReplayBuilder",
    "ReplayGraphLabelView",
    "ReplaySampleBudget",
    "ReplaySource",
    "ReplayStateNode",
    "ReplayStats",
    "ReplayTrajectory",
    "best_replay_reward",
    "best_rollout_reward_by_graph",
    "build_replay_graph_label_views",
    "enumerate_replay_state_dag",
    "replay_graph_ids",
    "replay_trajectories",
    "replay_trajectories_with_stats",
    "rollout_hit_graph_ids",
    "score_replay_states",
    "training_from_trajectories",
]
