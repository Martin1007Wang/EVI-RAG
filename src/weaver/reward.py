from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import ExpansionBatch, StateBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class RewardOutput:
    """
    Terminal reward for StateBatch rows.

    Contract:
    - one scalar log_reward per state row;
    - reward is terminal utility over a prefix state;
    - no RL step reward;
    - no parsing of raw SampleFields here;
    - StateBatch.edge_ids is the only selected-edge truth source.
    """

    log_reward: Tensor  # [S]
    raw_log_reward: Tensor  # [S]

    answer_count: Tensor  # [S]
    target_count: Tensor  # [S]
    target_recall: Tensor  # [S]

    target_proximity: Tensor  # [S]
    path_edge_count: Tensor  # [S]
    path_edge_precision: Tensor  # [S]

    edge_count: Tensor  # [S]
    valid_mask: Tensor  # [S]
    success_mask: Tensor  # [S]

    metrics: dict[str, Tensor] = field(default_factory=dict)


class EvidenceSubgraphReward(nn.Module):
    """
    Dense terminal utility for evidence-subgraph prefix states.

    This is still a GFlowNet terminal reward:

        log R(z)

    not an RL-style accumulated step reward.

    Main semantics:

        success branch:
            answer recall
            + target-distance proximity
            + weak shortest-path edge fidelity
            - edge cost

        failure branch:
            large fail penalty
            + weak target-distance proximity
            + weak shortest-path edge fidelity
            - edge cost

    The failure branch exists only to order failed states for credit assignment.
    It must not give failed subgraphs reward mass comparable to successful states.
    """

    def __init__(
        self,
        *,
        answer_weight: float = 4.0,
        proximity_weight: float = 1.0,
        path_weight: float = 0.2,
        fail_cost: float = 8.0,
        fail_proximity_weight: float = 0.5,
        fail_path_weight: float = 0.1,
        edge_cost: float = 0.1,
        redundant_edge_cost: float = 0.0,
        success_bias: float = 0.0,
        distance_temperature: float = 1.0,
        reward_temperature: float = 1.0,
        unreachable_distance: int = -1,
    ) -> None:
        super().__init__()

        _check_non_negative("answer_weight", answer_weight)
        _check_non_negative("proximity_weight", proximity_weight)
        _check_non_negative("path_weight", path_weight)
        _check_non_negative("fail_cost", fail_cost)
        _check_non_negative("fail_proximity_weight", fail_proximity_weight)
        _check_non_negative("fail_path_weight", fail_path_weight)
        _check_non_negative("edge_cost", edge_cost)
        _check_non_negative("redundant_edge_cost", redundant_edge_cost)

        if distance_temperature <= 0.0:
            raise ValueError(f"distance_temperature must be positive, got {distance_temperature}.")
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")

        self.answer_weight = float(answer_weight)
        self.proximity_weight = float(proximity_weight)
        self.path_weight = float(path_weight)

        self.fail_cost = float(fail_cost)
        self.fail_proximity_weight = float(fail_proximity_weight)
        self.fail_path_weight = float(fail_path_weight)

        self.edge_cost = float(edge_cost)
        self.redundant_edge_cost = float(redundant_edge_cost)
        self.success_bias = float(success_bias)

        self.distance_temperature = float(distance_temperature)
        self.reward_temperature = float(reward_temperature)
        self.unreachable_distance = int(unreachable_distance)

    @torch.no_grad()
    def forward(
        self,
        *,
        state: StateBatch,
        graph_context: GraphContext,
        target_context: TargetContext,
    ) -> RewardOutput:
        _check_base_target_context(
            state=state,
            target_context=target_context,
        )

        node_target_distance = _require_tensor(
            target_context,
            "node_target_distance",
        )
        _check_node_target_distance(
            state=state,
            node_target_distance=node_target_distance,
        )

        if self.path_weight > 0.0 or self.fail_path_weight > 0.0:
            shortest_path_edge_mask = _require_tensor(
                target_context,
                "shortest_path_edge_mask",
            )
            _check_shortest_path_edge_mask(
                state=state,
                graph=graph_context,
                shortest_path_edge_mask=shortest_path_edge_mask,
            )
        else:
            shortest_path_edge_mask = None

        answer_count = count_answers_in_state(
            state=state,
            graph=graph_context,
            target_mask=target_context.target_mask,
        ).to(dtype=torch.float)

        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.graph_ids,
        ).to(dtype=torch.float)

        valid_mask = target_count.gt(0)
        success_mask = answer_count.gt(0) & valid_mask

        target_recall = answer_count / target_count.clamp_min(1.0)

        target_proximity = compute_target_proximity(
            state=state,
            graph=graph_context,
            node_target_distance=node_target_distance,
            distance_temperature=self.distance_temperature,
            unreachable_distance=self.unreachable_distance,
        )

        if shortest_path_edge_mask is None:
            path_edge_count = state.edge_count.new_zeros(
                state.num_states,
                dtype=torch.float,
            )
            path_edge_precision = state.edge_count.new_zeros(
                state.num_states,
                dtype=torch.float,
            )
        else:
            path_edge_count, path_edge_precision = compute_path_edge_fidelity(
                state=state,
                shortest_path_edge_mask=shortest_path_edge_mask,
            )

        edge_count = state.edge_count.to(dtype=torch.float)
        edge_penalty = self.edge_cost * edge_count
        if self.redundant_edge_cost > 0.0:
            zero_gain_edge_count = compute_zero_gain_edge_count(
                state=state,
                graph=graph_context,
                target_mask=target_context.target_mask,
            )
        else:
            zero_gain_edge_count = edge_count.new_zeros(edge_count.shape)
        redundant_penalty = float(self.redundant_edge_cost) * zero_gain_edge_count

        success_log_reward = (
            self.success_bias
            + self.answer_weight * target_recall
            + self.proximity_weight * target_proximity
            + self.path_weight * path_edge_precision
            - edge_penalty
            - redundant_penalty
        )

        failure_log_reward = (
            -self.fail_cost
            + self.fail_proximity_weight * target_proximity
            + self.fail_path_weight * path_edge_precision
            - edge_penalty
            - redundant_penalty
        )

        raw_log_reward = torch.where(
            success_mask,
            success_log_reward,
            failure_log_reward,
        )

        raw_log_reward = torch.where(
            valid_mask,
            raw_log_reward,
            raw_log_reward.new_full(raw_log_reward.shape, -self.fail_cost),
        )

        log_reward = raw_log_reward / self.reward_temperature

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            answer_count=answer_count,
            target_count=target_count,
            target_recall=target_recall,
            target_proximity=target_proximity,
            path_edge_count=path_edge_count,
            path_edge_precision=path_edge_precision,
            edge_count=edge_count,
            valid_mask=valid_mask,
            success_mask=success_mask,
            metrics={
                "reward/log_reward_mean": _mean(log_reward),
                "reward/log_reward_min": _min(log_reward),
                "reward/log_reward_max": _max(log_reward),
                "reward/raw_log_reward_mean": _mean(raw_log_reward),
                "reward/raw_log_reward_min": _min(raw_log_reward),
                "reward/raw_log_reward_max": _max(raw_log_reward),
                "reward/success_log_reward_mean": _masked_mean(
                    success_log_reward,
                    valid_mask,
                ),
                "reward/failure_log_reward_mean": _masked_mean(
                    failure_log_reward,
                    valid_mask,
                ),
                "reward/target_recall_mean": _masked_mean(
                    target_recall,
                    valid_mask,
                ),
                "reward/target_proximity_mean": _masked_mean(
                    target_proximity,
                    valid_mask,
                ),
                "reward/path_edge_count_mean": _masked_mean(
                    path_edge_count,
                    valid_mask,
                ),
                "reward/path_edge_precision_mean": _masked_mean(
                    path_edge_precision,
                    valid_mask,
                ),
                "reward/answer_count_mean": _masked_mean(
                    answer_count,
                    valid_mask,
                ),
                "reward/target_count_mean": _masked_mean(
                    target_count,
                    valid_mask,
                ),
                "reward/edge_count_mean": _masked_mean(
                    edge_count,
                    valid_mask,
                ),
                "reward/zero_gain_edge_count_mean": _masked_mean(
                    zero_gain_edge_count,
                    valid_mask,
                ),
                "reward/redundant_penalty_mean": _masked_mean(
                    redundant_penalty,
                    valid_mask,
                ),
                "reward/hit_rate": _mean(
                    success_mask.to(dtype=torch.float),
                ),
                "reward/zero_recall_rate": _masked_mean(
                    target_recall.le(0).to(dtype=torch.float),
                    valid_mask,
                ),
                "reward/positive_recall_rate": _masked_mean(
                    target_recall.gt(0).to(dtype=torch.float),
                    valid_mask,
                ),
                "reward/valid_rate": _mean(
                    valid_mask.to(dtype=torch.float),
                ),
                "reward/success_reward_mean": _masked_mean(
                    torch.exp(log_reward),
                    success_mask,
                ),
                "reward/failure_reward_mean": _masked_mean(
                    torch.exp(log_reward),
                    valid_mask & ~success_mask,
                ),
            },
        )


class TerminalRecallReward(nn.Module):
    """
    Legacy sparse terminal reward.

        log R(z) = log(epsilon + recall(z)) - edge_cost * |E_z|

    Keep this class for ablation only.
    The main model should use EvidenceSubgraphReward.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-6,
        edge_cost: float = 0.05,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        _check_non_negative("edge_cost", edge_cost)
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")

        self.epsilon = float(epsilon)
        self.edge_cost = float(edge_cost)
        self.reward_temperature = float(reward_temperature)

    @torch.no_grad()
    def forward(
        self,
        *,
        state: StateBatch,
        graph_context: GraphContext,
        target_context: TargetContext,
    ) -> RewardOutput:
        _check_base_target_context(
            state=state,
            target_context=target_context,
        )

        answer_count = count_answers_in_state(
            state=state,
            graph=graph_context,
            target_mask=target_context.target_mask,
        ).to(dtype=torch.float)

        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.graph_ids,
        ).to(dtype=torch.float)

        valid_mask = target_count.gt(0)
        success_mask = answer_count.gt(0) & valid_mask

        target_recall = answer_count / target_count.clamp_min(1.0)
        edge_count = state.edge_count.to(dtype=torch.float)

        raw_log_reward = torch.log(target_recall + self.epsilon) - self.edge_cost * edge_count

        raw_log_reward = torch.where(
            valid_mask,
            raw_log_reward,
            raw_log_reward.new_full(raw_log_reward.shape, torch.log(raw_log_reward.new_tensor(self.epsilon)).item()),
        )

        log_reward = raw_log_reward / self.reward_temperature

        zeros = edge_count.new_zeros(edge_count.shape)

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            answer_count=answer_count,
            target_count=target_count,
            target_recall=target_recall,
            target_proximity=zeros,
            path_edge_count=zeros,
            path_edge_precision=zeros,
            edge_count=edge_count,
            valid_mask=valid_mask,
            success_mask=success_mask,
            metrics={
                "reward/log_reward_mean": _mean(log_reward),
                "reward/log_reward_min": _min(log_reward),
                "reward/log_reward_max": _max(log_reward),
                "reward/raw_log_reward_mean": _mean(raw_log_reward),
                "reward/target_recall_mean": _masked_mean(
                    target_recall,
                    valid_mask,
                ),
                "reward/answer_count_mean": _masked_mean(
                    answer_count,
                    valid_mask,
                ),
                "reward/target_count_mean": _masked_mean(
                    target_count,
                    valid_mask,
                ),
                "reward/edge_count_mean": _masked_mean(
                    edge_count,
                    valid_mask,
                ),
                "reward/hit_rate": _mean(
                    success_mask.to(dtype=torch.float),
                ),
                "reward/zero_recall_rate": _masked_mean(
                    target_recall.le(0).to(dtype=torch.float),
                    valid_mask,
                ),
                "reward/positive_recall_rate": _masked_mean(
                    target_recall.gt(0).to(dtype=torch.float),
                    valid_mask,
                ),
                "reward/valid_rate": _mean(
                    valid_mask.to(dtype=torch.float),
                ),
            },
        )


def count_answers_in_state(
    *,
    state: StateBatch,
    graph: GraphContext,
    target_mask: Tensor,
) -> Tensor:
    """
    Count reachable target nodes covered by each state.

    StateBatch.covered_node_pairs(graph) must return unique
    (state_id, node_id) pairs. Therefore scatter_add counts each covered
    target node once.
    """

    state_ids, node_ids = state.covered_node_pairs(graph)

    answer_count = torch.zeros(
        state.num_states,
        dtype=torch.long,
        device=state.device,
    )

    if int(node_ids.numel()) == 0:
        return answer_count

    is_answer = target_mask.index_select(0, node_ids).to(dtype=torch.long)

    answer_count.scatter_add_(
        dim=0,
        index=state_ids,
        src=is_answer,
    )

    return answer_count


def compute_target_proximity(
    *,
    state: StateBatch,
    graph: GraphContext,
    node_target_distance: Tensor,
    distance_temperature: float,
    unreachable_distance: int,
) -> Tensor:
    """
    Dense progress signal from covered nodes to nearest reachable target.

    node_target_distance is indexed by physical node id.

    Expected convention:
    - distance == 0 for target nodes;
    - distance > 0 for nodes that can reach a target;
    - distance == unreachable_distance for nodes that cannot reach a target.

    Output:
    - max_{v in covered(z)} exp(-distance(v) / tau)
    - unreachable covered nodes contribute 0.
    """

    state_ids, node_ids = state.covered_node_pairs(graph)

    proximity = torch.zeros(
        state.num_states,
        dtype=torch.float,
        device=state.device,
    )

    if int(node_ids.numel()) == 0:
        return proximity

    distance = node_target_distance.index_select(0, node_ids)

    reachable = distance.ne(int(unreachable_distance))
    non_negative = distance.ge(0)
    valid = reachable & non_negative

    pair_score = torch.zeros(
        node_ids.numel(),
        dtype=torch.float,
        device=state.device,
    )

    if bool(valid.any()):
        pair_score[valid] = torch.exp(-distance[valid].to(dtype=torch.float) / float(distance_temperature))

    proximity.scatter_reduce_(
        dim=0,
        index=state_ids,
        src=pair_score,
        reduce="amax",
        include_self=True,
    )

    return proximity


def compute_path_edge_fidelity(
    *,
    state: StateBatch,
    shortest_path_edge_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Count selected edges that lie on at least one target-reaching shortest path.

    shortest_path_edge_mask is indexed by physical edge id.

    This is intentionally weak evidence shaping. It must not dominate
    answer_recall.
    """

    edge_ids = state.edge_ids

    if edge_ids.ndim != 2:
        raise ValueError("state.edge_ids must have shape [num_states, budget], " f"got {tuple(edge_ids.shape)}.")

    selected = edge_ids.ge(0)

    path_edge_count = torch.zeros(
        state.num_states,
        dtype=torch.float,
        device=state.device,
    )

    if not bool(selected.any()):
        path_edge_precision = path_edge_count.clone()
        return path_edge_count, path_edge_precision

    row_ids = (
        torch.arange(
            state.num_states,
            dtype=torch.long,
            device=state.device,
        )
        .unsqueeze(1)
        .expand_as(edge_ids)
    )

    selected_rows = row_ids[selected]
    selected_edges = edge_ids[selected]

    is_path_edge = shortest_path_edge_mask.index_select(
        0,
        selected_edges,
    ).to(dtype=torch.float)

    path_edge_count.scatter_add_(
        dim=0,
        index=selected_rows,
        src=is_path_edge,
    )

    edge_count = state.edge_count.to(dtype=torch.float)
    path_edge_precision = path_edge_count / edge_count.clamp_min(1.0)
    path_edge_precision = torch.where(
        edge_count.gt(0),
        path_edge_precision,
        path_edge_precision.new_zeros(path_edge_precision.shape),
    )

    return path_edge_count, path_edge_precision


def compute_zero_gain_edge_count(
    *,
    state: StateBatch,
    graph: GraphContext,
    target_mask: Tensor,
) -> Tensor:
    """
    Count selected edges that do not increase covered target count at selection time.

    StateBatch stores selected edges in trajectory order, so this is a prefix-time
    redundancy proxy rather than a global minimal-subgraph computation.
    """

    zero_gain = torch.zeros(
        state.num_states,
        dtype=torch.float,
        device=state.device,
    )

    if state.num_states == 0 or int(state.budget) == 0:
        return zero_gain

    current = StateBatch.initial(
        graph_ids=state.graph_ids,
        budget=int(state.budget),
    )
    current_answers = count_answers_in_state(
        state=current,
        graph=graph,
        target_mask=target_mask,
    ).to(dtype=torch.float)

    for step in range(int(state.budget)):
        rows = state.edge_count.gt(step).nonzero(as_tuple=False).flatten()
        if int(rows.numel()) == 0:
            continue

        edge_ids = state.edge_ids.index_select(0, rows)[:, step]
        current = current.advance(
            ExpansionBatch(
                state_ids=rows,
                edge_ids=edge_ids,
            )
        )
        next_answers = count_answers_in_state(
            state=current,
            graph=graph,
            target_mask=target_mask,
        ).to(dtype=torch.float)

        gain = next_answers.index_select(0, rows) - current_answers.index_select(0, rows)
        zero_gain[rows] = zero_gain.index_select(0, rows) + gain.le(0.0).to(dtype=torch.float)
        current_answers = next_answers

    return zero_gain


def _check_base_target_context(
    *,
    state: StateBatch,
    target_context: TargetContext,
) -> None:
    if target_context.target_mask.ndim != 1:
        raise ValueError("target_context.target_mask must have shape [num_nodes], " f"got {tuple(target_context.target_mask.shape)}.")

    if target_context.target_mask.device != state.device:
        raise ValueError("target_mask and state must be on the same device.")

    if target_context.target_count_by_graph.ndim != 1:
        raise ValueError(
            "target_context.target_count_by_graph must have shape [num_graphs], " f"got {tuple(target_context.target_count_by_graph.shape)}."
        )

    if target_context.target_count_by_graph.device != state.device:
        raise ValueError("target_count_by_graph and state must be on the same device.")


def _check_node_target_distance(
    *,
    state: StateBatch,
    node_target_distance: Tensor,
) -> None:
    if node_target_distance.ndim != 1:
        raise ValueError("target_context.node_target_distance must have shape [num_nodes], " f"got {tuple(node_target_distance.shape)}.")

    if node_target_distance.device != state.device:
        raise ValueError("node_target_distance and state must be on the same device.")


def _check_shortest_path_edge_mask(
    *,
    state: StateBatch,
    graph: GraphContext,
    shortest_path_edge_mask: Tensor,
) -> None:
    if shortest_path_edge_mask.ndim != 1:
        raise ValueError("target_context.shortest_path_edge_mask must have shape [num_edges], " f"got {tuple(shortest_path_edge_mask.shape)}.")

    if shortest_path_edge_mask.device != state.device:
        raise ValueError("shortest_path_edge_mask and state must be on the same device.")

    num_graph_edges = int(graph.edge_index.size(1))
    if int(shortest_path_edge_mask.numel()) != num_graph_edges:
        raise ValueError(
            "shortest_path_edge_mask length must equal graph.edge_index.size(1): " f"got {int(shortest_path_edge_mask.numel())} vs {num_graph_edges}."
        )


def _require_tensor(
    obj: object,
    name: str,
) -> Tensor:
    value = getattr(obj, name, None)
    if value is None:
        raise AttributeError(f"TargetContext must provide {name!r} for EvidenceSubgraphReward.")
    if not torch.is_tensor(value):
        raise TypeError(f"TargetContext.{name} must be a Tensor.")
    return value


def _check_non_negative(
    name: str,
    value: float,
) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _mean(value: Tensor) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    return value.mean().detach()


def _masked_mean(
    value: Tensor,
    mask: Tensor,
) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    if int(mask.numel()) != int(value.numel()):
        raise ValueError("mask and value must have the same number of elements: " f"got {int(mask.numel())} vs {int(value.numel())}.")
    mask = mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        return value.new_zeros(())
    return value[mask].mean().detach()


def _min(value: Tensor) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    return value.min().detach()


def _max(value: Tensor) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    return value.max().detach()


__all__ = [
    "RewardOutput",
    "EvidenceSubgraphReward",
    "TerminalRecallReward",
    "count_answers_in_state",
    "compute_target_proximity",
    "compute_path_edge_fidelity",
    "compute_zero_gain_edge_count",
]
