from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.policy import PolicyOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput, target_ids
from src.weaver.rollout.buffer import RolloutBuffer
from src.weaver.rollout.executor import StepContext
from src.weaver.state import State


@dataclass(frozen=True, slots=True)
class StopAdvantageConfig:
    """
    Optimal-stopping auxiliary configuration.

    The auxiliary estimates whether the current state should Stop by comparing:

        J_stop(s) = log R_stop(s)

    with a small pool of one-step children:

        J_continue(s) = pool_{e in C_k(s)} log R_stop(s + e)

    Then it writes a soft Stop target:

        y_stop(s) = sigmoid((J_stop(s) - J_continue(s)) / label_temperature)

    The loss itself is computed later in loss.py from:

        stop_log_pf + stop_adv_target

    This module must not precompute or detach the BCE loss.
    """

    enabled: bool = False
    topk_by_semantic: int = 8
    topk_by_final: int = 8
    random_k: int = 8

    # "max" is recommended for stopping: continue if any candidate child improves reward.
    # "softmax" is smoother but may overvalue many mediocre candidates.
    # "logmeanexp" is usually too conservative when random candidates are included.
    continue_pool: str = "max"
    continue_pool_temperature: float = 0.5
    label_temperature: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict[str, object] | None) -> "StopAdvantageConfig":
        cfg = dict(cfg or {})
        defaults = cls()

        enabled = bool(cfg.pop("enabled", False))
        topk_by_semantic = _non_negative_int(
            cfg.pop("topk_by_semantic", defaults.topk_by_semantic),
            "topk_by_semantic",
        )
        topk_by_final = _non_negative_int(
            cfg.pop("topk_by_final", defaults.topk_by_final),
            "topk_by_final",
        )
        random_k = _non_negative_int(
            cfg.pop("random_k", defaults.random_k),
            "random_k",
        )

        continue_pool = str(cfg.pop("continue_pool", defaults.continue_pool))
        if continue_pool not in {"max", "softmax", "logmeanexp"}:
            raise ValueError(
                "continue_pool must be one of {'max', 'softmax', 'logmeanexp'}, "
                f"got {continue_pool!r}."
            )

        continue_pool_temperature = _positive_float(
            cfg.pop(
                "continue_pool_temperature",
                defaults.continue_pool_temperature,
            ),
            "continue_pool_temperature",
        )
        label_temperature = _positive_float(
            cfg.pop("label_temperature", defaults.label_temperature),
            "label_temperature",
        )

        if cfg:
            raise ValueError(f"Unused stop_adv config keys: {sorted(cfg)}.")

        return cls(
            enabled=enabled,
            topk_by_semantic=topk_by_semantic,
            topk_by_final=topk_by_final,
            random_k=random_k,
            continue_pool=continue_pool,
            continue_pool_temperature=continue_pool_temperature,
            label_temperature=label_temperature,
        )


class StopAdvantageAuxiliary:
    """
    Training-time optimal-stopping target writer.

    It compares the current stop-now reward with one-step child stop-now rewards
    and writes only target tensors into RolloutBuffer.

    It does not:
        - change policy logits;
        - sample actions;
        - compute final StopAdv BCE loss;
        - supervise edge selection;
        - run at inference.

    Required reward form:
        log R(x) = log(eps + support(x)) - edge_cost * expanded_edges
    """

    def __init__(self, cfg: StopAdvantageConfig) -> None:
        self.cfg = cfg

        if cfg.enabled and (
            cfg.topk_by_semantic + cfg.topk_by_final + cfg.random_k <= 0
        ):
            raise ValueError(
                "At least one StopAdvantage candidate source must be positive."
            )

    @property
    def requires_stop_now_reward(self) -> bool:
        return bool(self.cfg.enabled)

    def write_step(
        self,
        *,
        buffer: RolloutBuffer,
        t: int,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        state: State,
        step_out: PolicyOutput,
        step_context: StepContext,
        stop_now_reward: TerminalRewardOutput,
    ) -> None:
        if not self.cfg.enabled:
            return

        device = step_out.stop_logits.device
        num_graphs = int(retrieval_batch.num_graphs)

        active = step_context.active_mask.to(device=device, dtype=torch.bool).view(
            num_graphs
        )
        valid = step_context.can_expand.to(device=device, dtype=torch.bool).view(
            num_graphs
        )

        target = step_out.stop_logits.new_zeros(num_graphs)
        continue_log_reward = step_out.stop_logits.new_zeros(num_graphs)

        if not bool(valid.any()):
            self._write_empty(
                buffer=buffer,
                t=t,
                active=active,
                target=target,
                valid=valid,
                continue_log_reward=continue_log_reward,
            )
            return

        selected_pos = self._select_candidate_positions(
            step_out=step_out,
            num_graphs=num_graphs,
            valid_graphs=valid,
        )

        if selected_pos.numel() == 0:
            valid = torch.zeros_like(valid)
            self._write_empty(
                buffer=buffer,
                t=t,
                active=active,
                target=target,
                valid=valid,
                continue_log_reward=continue_log_reward,
            )
            return

        selected_edge_ids = step_out.candidate_edge_ids.index_select(0, selected_pos)
        selected_graph_ids = step_out.candidate_batch_ids.index_select(0, selected_pos)

        child = evaluate_one_step_child_rewards(
            batch=retrieval_batch,
            state=state,
            reward_model=reward_model,
            stop_now_reward=stop_now_reward,
            candidate_edge_ids=selected_edge_ids,
        )

        for graph_id_tensor in valid.nonzero(as_tuple=False).view(-1):
            graph_id = int(graph_id_tensor.item())

            graph_pos = selected_graph_ids.eq(graph_id).nonzero(as_tuple=False).view(-1)
            if graph_pos.numel() == 0:
                valid[graph_id] = False
                continue

            child_values = child.log_reward.index_select(0, graph_pos)
            pooled_continue = pool_continue_value(
                child_values,
                mode=self.cfg.continue_pool,
                temperature=self.cfg.continue_pool_temperature,
            )

            stop_value = stop_now_reward.log_reward[graph_id].to(
                device=device,
                dtype=pooled_continue.dtype,
            )

            continue_log_reward[graph_id] = pooled_continue
            target[graph_id] = torch.sigmoid(
                (stop_value - pooled_continue) / float(self.cfg.label_temperature)
            )

        buffer.write_stop_advantage(
            t=t,
            active=active,
            target=target.detach(),
            valid_mask=valid,
            continue_log_reward=continue_log_reward.detach(),
        )

    @staticmethod
    def _write_empty(
        *,
        buffer: RolloutBuffer,
        t: int,
        active: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        continue_log_reward: torch.Tensor,
    ) -> None:
        buffer.write_stop_advantage(
            t=t,
            active=active,
            target=target.detach(),
            valid_mask=valid,
            continue_log_reward=continue_log_reward.detach(),
        )

    def _select_candidate_positions(
        self,
        *,
        step_out: PolicyOutput,
        num_graphs: int,
        valid_graphs: torch.Tensor,
    ) -> torch.Tensor:
        candidate_batch = step_out.candidate_batch_ids
        if candidate_batch.numel() == 0:
            return candidate_batch.new_empty((0,))

        semantic_logits = step_out.edge_logits
        if step_out.edge_score_breakdown is not None:
            semantic_logits = step_out.edge_score_breakdown.semantic_logits

        positions: list[torch.Tensor] = []

        for graph_id in range(int(num_graphs)):
            if not bool(valid_graphs[graph_id]):
                continue

            graph_pos = candidate_batch.eq(graph_id).nonzero(as_tuple=False).view(-1)
            if graph_pos.numel() == 0:
                continue

            positions.append(
                _topk_positions(
                    graph_pos=graph_pos,
                    scores=semantic_logits,
                    k=self.cfg.topk_by_semantic,
                )
            )
            positions.append(
                _topk_positions(
                    graph_pos=graph_pos,
                    scores=step_out.edge_logits,
                    k=self.cfg.topk_by_final,
                )
            )
            positions.append(
                _random_positions(
                    graph_pos=graph_pos,
                    k=self.cfg.random_k,
                )
            )

        if not positions:
            return candidate_batch.new_empty((0,))

        non_empty = [pos for pos in positions if pos.numel() > 0]
        if not non_empty:
            return candidate_batch.new_empty((0,))

        return torch.unique(torch.cat(non_empty, dim=0))


@dataclass(frozen=True, slots=True)
class OneStepChildReward:
    log_reward: torch.Tensor
    supported_answer_recall: torch.Tensor


@torch.no_grad()
def evaluate_one_step_child_rewards(
    *,
    batch: RetrievalBatch,
    state: State,
    reward_model: RewardModel,
    stop_now_reward: TerminalRewardOutput,
    candidate_edge_ids: torch.Tensor,
) -> OneStepChildReward:
    """
    Fast one-step stop reward evaluator for s + e.

    The function evaluates only selected candidate edges, not the full frontier.
    """
    device = state.active_nodes.device

    candidate_edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)

    if candidate_edge_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return OneStepChildReward(log_reward=empty, supported_answer_recall=empty)

    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
    node_batch = batch.batch.to(device=device, dtype=torch.long)

    num_nodes = int(batch.num_nodes_total)

    graph_ids = edge_batch.index_select(0, candidate_edge_ids).long()

    current_support = stop_now_reward.supported_answer_recall.index_select(
        0,
        graph_ids,
    ).to(device=device, dtype=torch.float32)
    child_support = current_support.clone()

    root_edges = state.root_edges.to(device=device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=device, dtype=torch.bool)

    expanded_count = stop_now_reward.expanded_edge_count.index_select(0, graph_ids).to(
        device=device,
        dtype=torch.float32,
    )

    newly_expanded = (
        (~root_edges.index_select(0, candidate_edge_ids))
        & (~active_edges.index_select(0, candidate_edge_ids))
    ).to(dtype=torch.float32)

    expanded_count = expanded_count + newly_expanded

    target_mask = _node_mask(
        target_ids(batch),
        num_nodes=num_nodes,
        device=device,
    )
    anchor_mask = _node_mask(
        batch.anchor_node_ids,
        num_nodes=num_nodes,
        device=device,
    )
    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)

    for local_idx, edge_id_tensor in enumerate(candidate_edge_ids):
        edge_id = int(edge_id_tensor.item())
        graph_id = int(edge_batch[edge_id].item())

        graph_targets = (
            (target_mask & node_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )

        if graph_targets.numel() == 0:
            continue

        graph_anchors = (
            (anchor_mask & active_nodes & node_batch.eq(graph_id))
            .nonzero(as_tuple=False)
            .view(-1)
        )

        if graph_anchors.numel() == 0:
            continue

        graph_active_edges = (
            (active_edges & edge_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )

        graph_edge_ids = torch.unique(
            torch.cat([graph_active_edges, edge_id_tensor.view(1)], dim=0)
        )

        reached = _connected_nodes_from_anchors(
            edge_index=edge_index,
            edge_ids=graph_edge_ids,
            anchors=graph_anchors,
        )

        hits = sum(int(node_id) in reached for node_id in graph_targets.tolist())
        support = float(hits) / float(max(1, int(graph_targets.numel())))

        # Adding an edge should not reduce undirected anchor-supported coverage.
        child_support[local_idx] = max(float(current_support[local_idx]), support)

    if getattr(reward_model, "score_mode", "supported_recall") == "f_beta":
        # Approximate one-step precision by preserving the current supported
        # precision unless the child reaches a new answer. This helper remains
        # an auxiliary oracle, not the terminal reward source of truth.
        current_precision = stop_now_reward.supported_answer_precision.index_select(
            0,
            graph_ids,
        ).to(device=device, dtype=torch.float32)
        beta = float(getattr(reward_model, "beta", 2.0))
        beta_sq = beta * beta
        denom = beta_sq * current_precision + child_support
        child_utility = torch.where(
            (current_precision > 0.0) & (child_support > 0.0) & denom.gt(0.0),
            (1.0 + beta_sq) * current_precision * child_support / denom,
            torch.zeros_like(child_support),
        )
    else:
        child_utility = child_support

    log_reward = (
        (child_utility + float(reward_model.utility_epsilon)).log()
        - float(reward_model.edge_cost) * expanded_count
    ).clamp_min(float(reward_model.log_reward_clip_min))

    return OneStepChildReward(
        log_reward=log_reward.to(dtype=torch.float32),
        supported_answer_recall=child_support.to(dtype=torch.float32),
    )


def pool_continue_value(
    values: torch.Tensor,
    *,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    """
    Pool one-step child rewards into J_continue(s).

    Recommended:
        mode="max"

    max:
        Continue is valuable if at least one child improves reward.

    softmax:
        Smooth max, no -log(K) normalization.

    logmeanexp:
        Mean-like soft pooling. Usually too conservative when random candidates
        are mixed into the pool.
    """
    if values.numel() == 0:
        raise ValueError("Cannot pool an empty child reward tensor.")

    mode = str(mode)

    if mode == "max":
        return values.max()

    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temp}.")

    if mode == "softmax":
        return temp * torch.logsumexp(values / temp, dim=0)

    if mode == "logmeanexp":
        normalizer = torch.log(values.new_tensor(float(values.numel())))
        return temp * (torch.logsumexp(values / temp, dim=0) - normalizer)

    raise ValueError(
        f"mode must be one of {{'max', 'softmax', 'logmeanexp'}}, got {mode!r}."
    )


def _topk_positions(
    *,
    graph_pos: torch.Tensor,
    scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0 or graph_pos.numel() == 0:
        return graph_pos.new_empty((0,))

    count = min(int(k), int(graph_pos.numel()))
    values = scores.index_select(0, graph_pos)
    _, order = torch.topk(values, k=count)
    return graph_pos.index_select(0, order)


def _random_positions(
    *,
    graph_pos: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0 or graph_pos.numel() == 0:
        return graph_pos.new_empty((0,))

    count = min(int(k), int(graph_pos.numel()))
    order = torch.randperm(int(graph_pos.numel()), device=graph_pos.device)[:count]
    return graph_pos.index_select(0, order)


def _node_mask(
    ids: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)

    ids = ids.to(device=device, dtype=torch.long).view(-1)
    valid = ids.ge(0) & ids.lt(int(num_nodes))

    if bool(valid.any()):
        mask[ids[valid]] = True

    return mask


def _connected_nodes_from_anchors(
    *,
    edge_index: torch.Tensor,
    edge_ids: torch.Tensor,
    anchors: torch.Tensor,
) -> set[int]:
    visited = {int(node_id) for node_id in anchors.tolist()}
    frontier = list(visited)
    adjacency: dict[int, list[int]] = {}

    if edge_ids.numel() > 0:
        src = edge_index[0].index_select(0, edge_ids).tolist()
        dst = edge_index[1].index_select(0, edge_ids).tolist()

        for left, right in zip(src, dst):
            left_id = int(left)
            right_id = int(right)
            adjacency.setdefault(left_id, []).append(right_id)
            adjacency.setdefault(right_id, []).append(left_id)

    while frontier:
        current = frontier.pop()

        for neighbor in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)

    return visited


def _non_negative_int(value: object, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}.")
    return out


def _positive_float(value: object, name: str) -> float:
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}.")
    return out


__all__ = [
    "OneStepChildReward",
    "StopAdvantageAuxiliary",
    "StopAdvantageConfig",
    "evaluate_one_step_child_rewards",
    "pool_continue_value",
]
