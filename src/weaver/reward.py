from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import StateBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class RewardOutput:
    """
    Terminal reward for StateBatch rows.

    Contract:
    - one scalar log_reward per state row;
    - reward is terminal utility over an evidence-subgraph state;
    - no RL-style accumulated step reward;
    - no shortest-path fidelity term;
    - StateBatch.edge_ids / covered_node_pairs are the only state truth sources.

    Main semantics:

        log R(z)
        =
        answer_weight * log(epsilon + F_beta(z))
        - edge_cost * |E_z|

    where F_beta is computed from:
    - answer precision over covered candidate nodes;
    - target recall over reachable target nodes.
    """

    log_reward: Tensor  # [S]
    raw_log_reward: Tensor  # [S]

    answer_count: Tensor  # [S]
    candidate_count: Tensor  # [S]
    target_count: Tensor  # [S]

    answer_precision: Tensor  # [S]
    target_recall: Tensor  # [S]
    answer_f_score: Tensor  # [S]

    edge_count: Tensor  # [S]
    valid_mask: Tensor  # [S]
    success_mask: Tensor  # [S]

    metrics: dict[str, Tensor] = field(default_factory=dict)


class EvidenceSubgraphReward(nn.Module):
    """
    Precision-aware terminal reward for evidence-subgraph states.

    This is a GFlowNet terminal reward:

        log R(z)

    not a step reward.

    Reward definition:

        log R(z)
        =
        answer_weight * log(epsilon + F_beta(z))
        - edge_cost * |E_z|

    F_beta balances:
    - precision: how selectively the state exposes answer nodes among candidate nodes;
    - recall: how many reachable target answers are covered.

    Important design constraint:
    - shortest-path weak labels must be used by replay / off-policy sampling only;
    - they must not appear in terminal reward.
    """

    def __init__(
        self,
        *,
        beta: float = 2.0,
        epsilon: float = 1e-6,
        answer_weight: float = 1.0,
        edge_cost: float = 0.05,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        _check_non_negative("answer_weight", answer_weight)
        _check_non_negative("edge_cost", edge_cost)
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")

        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.answer_weight = float(answer_weight)
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

        candidate_mask = _optional_node_mask(
            target_context,
            names=("answer_candidate_mask", "candidate_mask"),
            state=state,
        )

        answer_count = count_answers_in_state(
            state=state,
            graph=graph_context,
            target_mask=target_context.target_mask,
        ).to(dtype=torch.float)

        candidate_count = count_candidate_nodes_in_state(
            state=state,
            graph=graph_context,
            candidate_mask=candidate_mask,
        ).to(dtype=torch.float)

        # Answers are candidates by definition. This protects against a
        # user-provided candidate_mask that accidentally excludes target nodes.
        candidate_count = torch.maximum(candidate_count, answer_count)

        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.graph_ids,
        ).to(dtype=torch.float)

        valid_mask = target_count.gt(0)
        success_mask = answer_count.gt(0) & valid_mask

        answer_precision = answer_count / candidate_count.clamp_min(1.0)
        target_recall = answer_count / target_count.clamp_min(1.0)

        answer_f_score = f_beta_score(
            precision=answer_precision,
            recall=target_recall,
            beta=self.beta,
            epsilon=self.epsilon,
        )

        # Invalid samples should normally not appear. They receive the floor
        # answer score while still paying compactness cost.
        answer_f_score = torch.where(
            valid_mask,
            answer_f_score,
            answer_f_score.new_zeros(answer_f_score.shape),
        )

        edge_count = state.edge_count.to(dtype=torch.float)
        edge_penalty = self.edge_cost * edge_count

        raw_log_reward = self.answer_weight * torch.log(answer_f_score + self.epsilon) - edge_penalty

        log_reward = raw_log_reward / self.reward_temperature

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            answer_count=answer_count,
            candidate_count=candidate_count,
            target_count=target_count,
            answer_precision=answer_precision,
            target_recall=target_recall,
            answer_f_score=answer_f_score,
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
                "reward/answer_f_score_mean": _masked_mean(
                    answer_f_score,
                    valid_mask,
                ),
                "reward/answer_precision_mean": _masked_mean(
                    answer_precision,
                    valid_mask,
                ),
                "reward/target_recall_mean": _masked_mean(
                    target_recall,
                    valid_mask,
                ),
                "reward/answer_count_mean": _masked_mean(
                    answer_count,
                    valid_mask,
                ),
                "reward/candidate_count_mean": _masked_mean(
                    candidate_count,
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
                "reward/edge_penalty_mean": _masked_mean(
                    edge_penalty,
                    valid_mask,
                ),
                "reward/hit_rate": _mean(
                    success_mask.to(dtype=torch.float),
                ),
                "reward/zero_f_score_rate": _masked_mean(
                    answer_f_score.le(0).to(dtype=torch.float),
                    valid_mask,
                ),
                "reward/positive_f_score_rate": _masked_mean(
                    answer_f_score.gt(0).to(dtype=torch.float),
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


def f_beta_score(
    *,
    precision: Tensor,
    recall: Tensor,
    beta: float,
    epsilon: float,
) -> Tensor:
    """
    Compute F_beta from precision and recall.

        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

    beta > 1 emphasizes recall.
    beta < 1 emphasizes precision.
    """

    beta2 = float(beta) ** 2
    numerator = (1.0 + beta2) * precision * recall
    denominator = beta2 * precision + recall

    return numerator / denominator.clamp_min(float(epsilon))


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


def count_candidate_nodes_in_state(
    *,
    state: StateBatch,
    graph: GraphContext,
    candidate_mask: Tensor | None,
) -> Tensor:
    """
    Count candidate answer nodes covered by each state.

    If target_context provides `answer_candidate_mask` or `candidate_mask`,
    candidates are restricted by that mask.

    If no candidate mask is provided, this falls back to all covered nodes.
    That fallback is intentionally conservative: it penalizes states that expose
    many non-answer nodes.

    Recommended future TargetContext field:
        answer_candidate_mask: BoolTensor[num_nodes]

    It should mark nodes that are valid answer candidates, e.g. non-CVT entities,
    type-compatible entities, or leaf/end entities depending on the dataset.
    """

    state_ids, node_ids = state.covered_node_pairs(graph)

    candidate_count = torch.zeros(
        state.num_states,
        dtype=torch.long,
        device=state.device,
    )

    if int(node_ids.numel()) == 0:
        return candidate_count

    if candidate_mask is None:
        is_candidate = torch.ones(
            node_ids.numel(),
            dtype=torch.long,
            device=state.device,
        )
    else:
        is_candidate = candidate_mask.index_select(0, node_ids).to(dtype=torch.long)

    candidate_count.scatter_add_(
        dim=0,
        index=state_ids,
        src=is_candidate,
    )

    return candidate_count


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


def _optional_node_mask(
    obj: object,
    *,
    names: tuple[str, ...],
    state: StateBatch,
) -> Tensor | None:
    """
    Return the first available optional node mask from TargetContext.

    This keeps reward.py independent of raw SampleFields while allowing a
    cleaner answer-candidate projection when TargetContext provides one.
    """

    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"TargetContext.{name} must be a Tensor.")
        if value.ndim != 1:
            raise ValueError(f"TargetContext.{name} must have shape [num_nodes], " f"got {tuple(value.shape)}.")
        if value.device != state.device:
            raise ValueError(f"TargetContext.{name} and state must be on the same device.")
        return value.to(dtype=torch.bool)

    return None


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
    "f_beta_score",
    "count_answers_in_state",
    "count_candidate_nodes_in_state",
]
