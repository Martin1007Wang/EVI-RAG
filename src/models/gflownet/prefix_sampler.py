from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.graph import TrajectoryBatch
from src.utils.segment_ops import sample_segmented_one_1d

from .answer_supervision import (
    AnswerSupervisionMetadata,
    build_answer_mask,
    build_answer_supervision_metadata,
    build_unique_graph_answers,
    compute_answer_sink_targets,
    graph_ids_from_ptr,
    lookup_is_gold,
)
from .prefix_policy import _temper_segmented_log_probs
from .path import (
    append_relation_and_node_tokens_inplace,
    append_stop_token_inplace,
    initialize_path_token_ids,
)
from .transitions import apply_forward_constraints
from .prefix_state import (
    ForwardActionDistribution,
    GFlowNetPolicyProtocol,
    PreparedGFlowNetBatch,
    RootActionDistribution,
    SearchState,
)


class TrajectoryRolloutSupervisorProtocol(Protocol):
    def build_terminal_target_mask(self, *, batch: TrajectoryBatch) -> torch.Tensor: ...

    def resolve_terminal_transitions(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
    ) -> "TerminalTransitionBatch": ...

    def compute_terminal_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# Fixed terminal energy reward keeps supervision entirely in log-reward space.
_GOLD_TERMINAL_LOG_REWARD = 0.0


@dataclass(frozen=True)
class TerminalTransitionBatch:
    terminal_entity_ids: torch.Tensor
    terminal_is_gold: torch.Tensor
    terminal_sink_ids: torch.Tensor
    terminal_sink_log_rewards: torch.Tensor
    gold_answer_counts: torch.Tensor
    terminal_rewards: torch.Tensor
    terminal_log_rewards: torch.Tensor
    terminal_backward_log_probs: torch.Tensor


class AnswerReachabilityTrajectorySupervisor:
    """Base terminal-energy supervisor for answer reachability.

    The path-level terminal reward still depends only on whether the reached
    entity is gold, but the supervisor also emits answer-sink metadata so the
    loss can aggregate terminal mass at the answer-equivalence level.
    """

    def __init__(
        self,
        *,
        non_gold_terminal_log_reward: float = -3.0,
        gold_reward_mode: str = "shared",
    ) -> None:
        if float(non_gold_terminal_log_reward) > 0.0:
            raise ValueError(
                "non_gold_terminal_log_reward must be <= 0 for terminal supervision."
            )
        if gold_reward_mode not in {"shared", "unit"}:
            raise ValueError(
                "gold_reward_mode must be one of {'shared', 'unit'} for terminal supervision."
            )
        self.non_gold_terminal_log_reward = float(non_gold_terminal_log_reward)
        self.gold_reward_mode = str(gold_reward_mode)

    @staticmethod
    def _graph_ids_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        return graph_ids_from_ptr(ptr)

    @staticmethod
    def _build_answer_supervision_metadata(
        batch: TrajectoryBatch,
    ) -> AnswerSupervisionMetadata:
        return build_answer_supervision_metadata(
            node_entity_ids=batch.node_entity_ids,
            answer_entity_ids=batch.answer_entity_ids,
            answer_ptr=batch.answer_ptr,
            device=batch.node_ptr.device,
        )

    @staticmethod
    def _flatten_terminal_graph_ids(terminal_values: torch.Tensor) -> torch.Tensor:
        graph_ids = torch.arange(
            int(terminal_values.size(0)),
            device=terminal_values.device,
            dtype=torch.long,
        )
        view_shape = (int(terminal_values.size(0)),) + (1,) * (
            terminal_values.dim() - 1
        )
        return graph_ids.view(view_shape).expand_as(terminal_values).reshape(-1)

    @staticmethod
    def _lookup_terminal_is_gold(
        *,
        metadata: AnswerSupervisionMetadata,
        terminal_entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        flat_graph_ids = (
            AnswerReachabilityTrajectorySupervisor._flatten_terminal_graph_ids(
                terminal_entity_ids
            )
        )
        return lookup_is_gold(
            metadata=metadata,
            graph_ids=flat_graph_ids,
            entity_ids=terminal_entity_ids.reshape(-1),
        ).view_as(terminal_entity_ids)

    def _compute_terminal_energy_log_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_entity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self._build_answer_supervision_metadata(batch)
        is_gold = self._lookup_terminal_is_gold(
            metadata=metadata,
            terminal_entity_ids=terminal_entity_ids,
        )
        log_rewards = torch.where(
            is_gold,
            torch.full_like(
                terminal_entity_ids,
                fill_value=_GOLD_TERMINAL_LOG_REWARD,
                dtype=torch.float32,
            ),
            torch.full_like(
                terminal_entity_ids,
                fill_value=self.non_gold_terminal_log_reward,
                dtype=torch.float32,
            ),
        )
        del batch
        return log_rewards, is_gold.to(dtype=torch.bool)

    def _compute_terminal_answer_sink_targets(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_entity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_answers = build_unique_graph_answers(
            answer_entity_ids=batch.answer_entity_ids,
            answer_ptr=batch.answer_ptr,
            device=terminal_entity_ids.device,
        )
        flat_graph_ids = self._flatten_terminal_graph_ids(terminal_entity_ids)
        flat_terminal_entities = terminal_entity_ids.reshape(-1)
        sink_ids, sink_log_rewards, gold_answer_counts_per_graph = (
            compute_answer_sink_targets(
                unique_answers=unique_answers,
                graph_ids=flat_graph_ids,
                entity_ids=flat_terminal_entities,
                non_gold_terminal_log_reward=self.non_gold_terminal_log_reward,
                gold_reward_mode=self.gold_reward_mode,
            )
        )
        return (
            sink_ids.view_as(terminal_entity_ids),
            sink_log_rewards.view_as(terminal_entity_ids),
            gold_answer_counts_per_graph,
        )

    @staticmethod
    def _resolve_terminal_entity_ids(
        *, batch: TrajectoryBatch, terminal_nodes: torch.Tensor
    ) -> torch.Tensor:
        node_entity_ids = batch.node_entity_ids.to(dtype=torch.long)
        flat_terminal_nodes = terminal_nodes.reshape(-1)
        terminal_entity_ids = node_entity_ids.index_select(0, flat_terminal_nodes)
        return terminal_entity_ids.view_as(terminal_nodes)

    def resolve_terminal_transitions(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
    ) -> TerminalTransitionBatch:
        terminal_entity_ids = self._resolve_terminal_entity_ids(
            batch=batch,
            terminal_nodes=terminal_nodes,
        )
        terminal_log_rewards, terminal_is_gold = (
            self._compute_terminal_energy_log_rewards(
                batch=batch,
                terminal_entity_ids=terminal_entity_ids,
            )
        )
        (
            terminal_sink_ids,
            terminal_sink_log_rewards,
            gold_answer_counts,
        ) = self._compute_terminal_answer_sink_targets(
            batch=batch,
            terminal_entity_ids=terminal_entity_ids,
        )
        terminal_backward_log_probs = torch.zeros_like(
            terminal_log_rewards, dtype=torch.float32
        )
        terminal_rewards = terminal_log_rewards.exp()
        return TerminalTransitionBatch(
            terminal_entity_ids=terminal_entity_ids,
            terminal_is_gold=terminal_is_gold,
            terminal_sink_ids=terminal_sink_ids,
            terminal_sink_log_rewards=terminal_sink_log_rewards,
            gold_answer_counts=gold_answer_counts,
            terminal_rewards=terminal_rewards,
            terminal_log_rewards=terminal_log_rewards,
            terminal_backward_log_probs=terminal_backward_log_probs,
        )

    def build_terminal_target_mask(self, *, batch: TrajectoryBatch) -> torch.Tensor:
        return build_answer_mask(
            node_ptr=batch.node_ptr,
            node_entity_ids=batch.node_entity_ids,
            answer_entity_ids=batch.answer_entity_ids,
            answer_ptr=batch.answer_ptr,
        )

    def compute_terminal_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terminal_transition = self.resolve_terminal_transitions(
            batch=batch,
            terminal_nodes=terminal_nodes,
        )
        return (
            terminal_transition.terminal_rewards,
            terminal_transition.terminal_log_rewards,
        )


def _apply_terminal_stop_action_backward_log_probs(
    *,
    log_pb_steps: torch.Tensor,
    termination_action_steps: torch.Tensor,
    terminal_num_steps: torch.Tensor,
    terminal_backward_log_probs: torch.Tensor,
) -> torch.Tensor:
    stopped_mask = termination_action_steps > terminal_num_steps
    if not bool(stopped_mask.any().item()):
        return log_pb_steps
    max_actions = int(log_pb_steps.size(-1))
    flat_log_pb_steps = log_pb_steps.view(-1, max_actions)
    flat_stopped_mask = stopped_mask.reshape(-1)
    row_idx = torch.nonzero(flat_stopped_mask, as_tuple=False).view(-1)
    stop_action_step_idx = (
        termination_action_steps.reshape(-1).index_select(0, row_idx) - 1
    )
    flat_terminal_backward = terminal_backward_log_probs.reshape(-1).index_select(
        0, row_idx
    )
    flat_log_pb_steps[row_idx, stop_action_step_idx] = flat_terminal_backward
    return log_pb_steps


def _mask_terminal_stop_action_backward_log_probs(
    *,
    termination_action_steps: torch.Tensor,
    terminal_num_steps: torch.Tensor,
    terminal_backward_log_probs: torch.Tensor,
) -> torch.Tensor:
    stopped_mask = termination_action_steps > terminal_num_steps
    return torch.where(
        stopped_mask,
        terminal_backward_log_probs,
        torch.zeros_like(terminal_backward_log_probs),
    )


def _policy_uses_answer_sink_terminal_targets(policy: GFlowNetPolicyProtocol) -> bool:
    base_policy = getattr(policy, "base_policy", policy)
    return bool(getattr(base_policy, "answer_quotient_allocate_stop_mass", False))


def _resolve_stop_branch_base_log_rewards(
    *,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    search_state: SearchState,
    distribution: ForwardActionDistribution,
    total_agents: int,
    on_target: torch.Tensor,
    answer_stop_log_reward_bonus: float,
) -> torch.Tensor:
    base_policy = getattr(policy, "base_policy", policy)
    if hasattr(base_policy, "build_state_features") and hasattr(
        base_policy, "_compute_stop_branch_logits"
    ):
        state_features = base_policy.build_state_features(prepared_batch, search_state)
        semantic_stop_logits = base_policy._compute_stop_branch_logits(
            prepared_batch=prepared_batch,
            current_nodes=search_state.flatten_current_nodes(),
            state_features=state_features.reshape(-1, int(state_features.size(-1))),
            graph_ids=search_state.flatten_graph_index(),
        ).to(dtype=torch.float32)
        stop_bonus = on_target.to(dtype=torch.float32) * float(
            answer_stop_log_reward_bonus
        )
        return semantic_stop_logits - stop_bonus
    base_log_rewards = torch.zeros(
        (total_agents,), device=distribution.edge_logits.device, dtype=torch.float32
    )
    stop_mask = distribution.is_stop_action
    if stop_mask is None or not bool(stop_mask.any().item()):
        return base_log_rewards
    stop_agents = distribution.edge_agent_batch[stop_mask]
    stop_branch_logits = distribution.edge_logits[stop_mask].to(dtype=torch.float32)
    stop_bonus = on_target.index_select(0, stop_agents).to(dtype=torch.float32) * float(
        answer_stop_log_reward_bonus
    )
    base_log_rewards[stop_agents] = stop_branch_logits - stop_bonus
    return base_log_rewards


def _apply_terminal_log_reward_overrides(
    *,
    policy: GFlowNetPolicyProtocol,
    terminal_transition: TerminalTransitionBatch,
    explicit_stop_mask: torch.Tensor,
    explicit_stop_log_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _policy_uses_answer_sink_terminal_targets(policy):
        terminal_log_rewards = terminal_transition.terminal_sink_log_rewards.to(
            dtype=torch.float32
        )
    else:
        terminal_log_rewards = terminal_transition.terminal_log_rewards.to(
            dtype=torch.float32
        )
    if bool(explicit_stop_mask.any().item()):
        terminal_log_rewards = terminal_log_rewards.clone()
        terminal_log_rewards[explicit_stop_mask] = explicit_stop_log_rewards[
            explicit_stop_mask
        ].to(dtype=torch.float32)
    return terminal_log_rewards, terminal_log_rewards.exp()


def _resolve_chosen_relation_ids(
    *,
    edge_type: torch.Tensor,
    chosen_edge_ids: torch.Tensor,
    view_shape: torch.Size,
) -> torch.Tensor:
    relation_ids = torch.zeros_like(chosen_edge_ids, dtype=torch.long)
    graph_move_mask = chosen_edge_ids >= 0
    if bool(graph_move_mask.any().item()):
        relation_ids[graph_move_mask] = edge_type.index_select(
            0,
            chosen_edge_ids[graph_move_mask],
        )
    return relation_ids.view(view_shape)


def _compute_move_log_reward_shaping(
    *,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    current_nodes: torch.Tensor,
    next_nodes: torch.Tensor,
    move_mask: torch.Tensor,
) -> torch.Tensor:
    shaping = torch.zeros_like(current_nodes, dtype=torch.float32)
    if not bool(move_mask.any().item()):
        return shaping
    base_policy = getattr(policy, "base_policy", policy)
    if not hasattr(base_policy, "compute_move_log_reward_shaping"):
        return shaping
    flat_move_mask = move_mask.reshape(-1)
    flat_current_nodes = current_nodes.reshape(-1)
    flat_next_nodes = next_nodes.reshape(-1)
    shaping.view(-1)[flat_move_mask] = base_policy.compute_move_log_reward_shaping(
        prepared_batch=prepared_batch,
        current_nodes=flat_current_nodes[flat_move_mask],
        next_nodes=flat_next_nodes[flat_move_mask],
    )
    return shaping


def _build_step_log_rewards(
    *,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    current_nodes: torch.Tensor,
    next_nodes: torch.Tensor,
    move_mask: torch.Tensor,
    gold_stop_mask: torch.Tensor,
    step_log_penalty: float,
    answer_stop_log_reward_bonus: float,
) -> torch.Tensor:
    log_reward_steps = move_mask.to(dtype=torch.float32) * float(step_log_penalty)
    log_reward_steps = log_reward_steps + _compute_move_log_reward_shaping(
        policy=policy,
        prepared_batch=prepared_batch,
        current_nodes=current_nodes,
        next_nodes=next_nodes,
        move_mask=move_mask,
    )
    if float(answer_stop_log_reward_bonus) <= 0.0:
        return log_reward_steps
    return log_reward_steps + gold_stop_mask.to(dtype=torch.float32) * float(
        answer_stop_log_reward_bonus
    )


def _sample_edges(
    *,
    distribution: ForwardActionDistribution,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_agents = int(distribution.out_degrees.numel())
    selected_positions = torch.full(
        (total_agents,),
        fill_value=-1,
        device=distribution.edge_logits.device,
        dtype=torch.long,
    )
    chosen_edge_ids = torch.full(
        (total_agents,),
        fill_value=-1,
        device=distribution.edge_ids.device,
        dtype=torch.long,
    )
    chosen_target_nodes = torch.zeros(
        (total_agents,), device=distribution.target_nodes.device, dtype=torch.long
    )
    chosen_log_probs = torch.zeros(
        (total_agents,), device=distribution.edge_logits.device, dtype=torch.float32
    )
    chosen_is_stop_action = torch.zeros(
        (total_agents,), device=distribution.edge_logits.device, dtype=torch.bool
    )
    if total_agents == 0:
        return (
            selected_positions,
            chosen_edge_ids,
            chosen_target_nodes,
            chosen_log_probs,
            chosen_is_stop_action,
        )
    stop_action_mask = (
        distribution.is_stop_action.to(dtype=torch.bool)
        if distribution.is_stop_action is not None
        else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
    )
    selected_positions, chosen_log_probs, has_values = sample_segmented_one_1d(
        logits=distribution.edge_logits,
        segment_ids=distribution.edge_agent_batch,
        num_segments=total_agents,
        temperature=float(temperature),
    )
    valid_positions = selected_positions[has_values]
    chosen_edge_ids[has_values] = distribution.edge_ids.index_select(0, valid_positions)
    chosen_target_nodes[has_values] = distribution.target_nodes.index_select(
        0, valid_positions
    )
    chosen_is_stop_action[has_values] = stop_action_mask.index_select(
        0, valid_positions
    )
    return (
        selected_positions,
        chosen_edge_ids,
        chosen_target_nodes,
        chosen_log_probs,
        chosen_is_stop_action,
    )


def _action_lookup_base(
    *, distribution: ForwardActionDistribution, selected_edge_ids: torch.Tensor
) -> torch.Tensor:
    max_distribution_edge = selected_edge_ids.new_tensor(-1, dtype=torch.long)
    if int(distribution.edge_ids.numel()) > 0:
        max_distribution_edge = distribution.edge_ids.to(dtype=torch.long).max()
    non_stop_selected = selected_edge_ids[selected_edge_ids >= 0]
    max_selected_edge = selected_edge_ids.new_tensor(-1, dtype=torch.long)
    if int(non_stop_selected.numel()) > 0:
        max_selected_edge = non_stop_selected.to(dtype=torch.long).max()
    return torch.maximum(max_distribution_edge, max_selected_edge) + 2


def _resolve_selected_action_positions(
    *,
    distribution: ForwardActionDistribution,
    selected_edge_ids: torch.Tensor,
    selected_is_stop_action: torch.Tensor,
    active_mask: torch.Tensor,
    error_prefix: str,
) -> torch.Tensor:
    total_agents = int(selected_edge_ids.numel())
    selected_positions = torch.full(
        (total_agents,),
        fill_value=-1,
        device=distribution.edge_logits.device,
        dtype=torch.long,
    )
    if total_agents == 0 or not bool(active_mask.any().item()):
        return selected_positions

    active_agents = torch.nonzero(active_mask, as_tuple=False).view(-1)
    active_edge_ids = selected_edge_ids.index_select(0, active_agents)
    active_stop_mask = selected_is_stop_action.index_select(0, active_agents)
    invalid_edge_ids = (~active_stop_mask) & (active_edge_ids < 0)
    if bool(invalid_edge_ids.any().item()):
        invalid_agents = active_agents[invalid_edge_ids].tolist()
        raise ValueError(
            f"{error_prefix} is missing an edge id for an active step. "
            f"agent_idx={invalid_agents}."
        )

    stop_action_mask = (
        distribution.is_stop_action.to(dtype=torch.bool)
        if distribution.is_stop_action is not None
        else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
    )
    lookup_base = _action_lookup_base(
        distribution=distribution,
        selected_edge_ids=selected_edge_ids,
    )
    action_edge_ids = torch.where(
        stop_action_mask,
        torch.full_like(distribution.edge_ids, fill_value=-1),
        distribution.edge_ids,
    )
    action_keys = (
        ((distribution.edge_agent_batch * 2) + stop_action_mask.to(dtype=torch.long))
        * lookup_base
    ) + (action_edge_ids + 1)
    sorted_keys, order = torch.sort(action_keys)
    selected_keys = (
        ((active_agents * 2) + active_stop_mask.to(dtype=torch.long)) * lookup_base
    ) + (
        torch.where(
            active_stop_mask, torch.full_like(active_edge_ids, -1), active_edge_ids
        )
        + 1
    )
    match_idx = torch.searchsorted(sorted_keys, selected_keys)
    in_range = match_idx < int(sorted_keys.numel())
    exact_match = torch.zeros_like(in_range)
    if bool(in_range.any().item()):
        exact_match[in_range] = (
            sorted_keys.index_select(0, match_idx[in_range]) == selected_keys[in_range]
        )
    duplicate_match = torch.zeros_like(in_range)
    if bool(in_range.any().item()):
        right_idx = match_idx[in_range]
        duplicate_match[in_range] = (
            ((right_idx + 1) < int(sorted_keys.numel()))
            & (
                sorted_keys.index_select(
                    0, (right_idx + 1).clamp_max(int(sorted_keys.numel()) - 1)
                )
                == selected_keys[in_range]
            )
        ) | (
            (right_idx > 0)
            & (
                sorted_keys.index_select(0, (right_idx - 1).clamp_min(0))
                == selected_keys[in_range]
            )
        )
    valid_match = in_range & exact_match & (~duplicate_match)
    if not bool(valid_match.all().item()):
        invalid_agents = active_agents[~valid_match]
        invalid_edges = active_edge_ids[~valid_match]
        invalid_stop_actions = active_stop_mask[~valid_match]
        raise ValueError(
            f"{error_prefix} edge is invalid under the current policy state. "
            f"agent_idx={invalid_agents.tolist()} edge_id={invalid_edges.tolist()} "
            f"is_stop_action={invalid_stop_actions.tolist()}."
        )
    selected_positions[active_agents] = order.index_select(0, match_idx)
    return selected_positions


def _select_edge_log_probs(
    *,
    distribution: ForwardActionDistribution,
    selected_edge_ids: torch.Tensor,
    selected_is_stop_action: torch.Tensor,
    active_mask: torch.Tensor,
    policy: GFlowNetPolicyProtocol,
    error_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_nodes = torch.zeros_like(selected_edge_ids)
    selected_log_probs = torch.zeros(
        selected_edge_ids.shape,
        device=distribution.edge_logits.device,
        dtype=torch.float32,
    )
    move_log_probs, _, _ = policy.compute_move_log_probs(distribution)
    selected_positions = _resolve_selected_action_positions(
        distribution=distribution,
        selected_edge_ids=selected_edge_ids,
        selected_is_stop_action=selected_is_stop_action,
        active_mask=active_mask,
        error_prefix=error_prefix,
    )
    active_positions = selected_positions[active_mask]
    selected_nodes[active_mask] = distribution.target_nodes.index_select(
        0, active_positions
    )
    selected_log_probs[active_mask] = move_log_probs.index_select(0, active_positions)
    return selected_nodes, selected_log_probs


def _resolve_selected_start_values(
    *,
    prepared_batch: PreparedGFlowNetBatch,
    policy: GFlowNetPolicyProtocol,
    start_nodes: torch.Tensor,
    start_distribution: RootActionDistribution | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    RootActionDistribution,
]:
    if start_distribution is None:
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
    start_log_probs = torch.zeros_like(start_nodes, dtype=torch.float32)
    start_log_flows = torch.zeros_like(start_nodes, dtype=torch.float32)
    start_log_rewards = torch.zeros_like(start_nodes, dtype=torch.float32)
    num_graphs, num_rollouts = start_nodes.shape
    if num_graphs == 0 or num_rollouts == 0:
        return (
            start_log_probs,
            start_log_flows,
            start_log_rewards,
            start_distribution.graph_log_z.to(dtype=torch.float32),
            start_distribution,
        )
    if int(start_distribution.candidate_nodes_abs.numel()) == 0:
        raise ValueError(
            "Sampled start node is not a valid target-policy start candidate. "
            "The batch has no start candidates."
        )
    flat_graph_ids = (
        torch.arange(num_graphs, device=start_nodes.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(start_nodes)
        .reshape(-1)
    )
    flat_start_nodes = start_nodes.reshape(-1)
    lookup_base = (
        torch.maximum(
            start_distribution.candidate_nodes_abs.to(dtype=torch.long).max(),
            flat_start_nodes.to(dtype=torch.long).max(),
        )
        + 1
    )
    candidate_keys = (
        start_distribution.candidate_graph_ids.to(dtype=torch.long) * lookup_base
    ) + start_distribution.candidate_nodes_abs.to(dtype=torch.long)
    sorted_keys, order = torch.sort(candidate_keys)
    selected_keys = (flat_graph_ids * lookup_base) + flat_start_nodes.to(
        dtype=torch.long
    )
    match_idx = torch.searchsorted(sorted_keys, selected_keys)
    in_range = match_idx < int(sorted_keys.numel())
    exact_match = torch.zeros_like(in_range)
    exact_match[in_range] = (
        sorted_keys.index_select(0, match_idx[in_range]) == selected_keys[in_range]
    )
    if not bool(exact_match.all().item()):
        invalid_graphs = flat_graph_ids[~exact_match].tolist()
        invalid_start_abs_nodes = flat_start_nodes[~exact_match].tolist()
        raise ValueError(
            "Sampled start node is not a valid target-policy start candidate. "
            f"graph_idx={invalid_graphs} start_abs_nodes={invalid_start_abs_nodes}."
        )
    selected_positions = order.index_select(0, match_idx)
    start_log_probs = (
        start_distribution.log_probs.to(dtype=torch.float32)
        .index_select(0, selected_positions)
        .view_as(start_nodes)
    )
    start_log_flows = (
        start_distribution.log_flows.to(dtype=torch.float32)
        .index_select(0, selected_positions)
        .view_as(start_nodes)
    )
    if start_distribution.start_log_rewards is not None:
        start_log_rewards = (
            start_distribution.start_log_rewards.to(dtype=torch.float32)
            .index_select(0, selected_positions)
            .view_as(start_nodes)
        )
    return (
        start_log_probs,
        start_log_flows,
        start_log_rewards,
        start_distribution.graph_log_z.to(dtype=torch.float32),
        start_distribution,
    )


def _compute_root_proposal_target_kl(
    *,
    proposal_distribution: RootActionDistribution,
    target_distribution: RootActionDistribution,
) -> torch.Tensor:
    if tuple(proposal_distribution.log_probs.shape) != tuple(
        target_distribution.log_probs.shape
    ):
        raise ValueError(
            "proposal and target root distributions must align to compute KL. "
            f"proposal={tuple(proposal_distribution.log_probs.shape)} "
            f"target={tuple(target_distribution.log_probs.shape)}."
        )
    if not torch.equal(
        proposal_distribution.candidate_nodes_abs,
        target_distribution.candidate_nodes_abs,
    ) or not torch.equal(
        proposal_distribution.candidate_graph_ids,
        target_distribution.candidate_graph_ids,
    ):
        raise ValueError(
            "proposal and target root distributions must expose the same candidates to compute KL."
        )
    num_graphs = int(target_distribution.graph_log_z.numel())
    if num_graphs == 0:
        return target_distribution.graph_log_z.new_empty((0,), dtype=torch.float32)
    proposal_log_probs = proposal_distribution.log_probs.to(dtype=torch.float32)
    target_log_probs = target_distribution.log_probs.to(dtype=torch.float32)
    segment_ids = target_distribution.candidate_graph_ids.to(dtype=torch.long)
    proposal_probs = proposal_log_probs.exp()
    contribution = proposal_probs * (proposal_log_probs - target_log_probs)
    kl = target_distribution.graph_log_z.new_zeros((num_graphs,), dtype=torch.float32)
    kl.scatter_add_(0, segment_ids, contribution)
    return kl


def _compute_start_distribution_entropy(
    *,
    log_probs: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_graphs < 1:
        empty = log_probs.new_empty((0,), dtype=torch.float32)
        return empty, empty
    log_probs = log_probs.to(dtype=torch.float32)
    probs = torch.exp(log_probs)
    finite_mask = torch.isfinite(log_probs)
    safe_probs = torch.where(finite_mask, probs, torch.zeros_like(probs))
    entropy_terms = torch.where(
        finite_mask,
        (-safe_probs) * log_probs,
        torch.zeros_like(safe_probs),
    )
    segment_ids = candidate_graph_ids.to(dtype=torch.long)
    entropy = torch.zeros((num_graphs,), device=log_probs.device, dtype=torch.float32)
    entropy.scatter_add_(0, segment_ids, entropy_terms)
    candidate_counts = torch.zeros_like(entropy)
    candidate_counts.scatter_add_(
        0,
        segment_ids,
        torch.ones_like(safe_probs, dtype=torch.float32),
    )
    max_entropy = torch.log(candidate_counts.clamp_min(1.0))
    normalized_entropy = torch.where(
        candidate_counts > 1.0,
        entropy / max_entropy.clamp_min(1.0e-6),
        torch.ones_like(entropy),
    )
    return entropy, normalized_entropy.clamp(0.0, 1.0)


def _rebuild_target_sample_batch(
    *,
    batch: TrajectoryBatch,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    trajectory_supervisor: TrajectoryRolloutSupervisorProtocol,
    start_nodes: torch.Tensor,
    planned_edge_ids: torch.Tensor,
    planned_stop_mask: torch.Tensor,
    path_lengths: torch.Tensor,
    termination_action_steps: torch.Tensor,
    trace_nodes: torch.Tensor,
    trace_edge_ids: torch.Tensor,
    trace_num_steps: torch.Tensor,
    trace_mask: torch.Tensor,
    trace_stop_mask: torch.Tensor,
    max_steps: int,
    step_log_penalty: float,
    answer_stop_log_reward_bonus: float,
) -> TrajectoryGFNSampleBatch:
    (
        start_log_probs,
        start_log_flows,
        start_log_rewards,
        graph_log_z,
        _,
    ) = _resolve_selected_start_values(
        prepared_batch=prepared_batch,
        policy=policy,
        start_nodes=start_nodes,
    )
    terminal_target_mask = trajectory_supervisor.build_terminal_target_mask(batch=batch)
    max_actions = max_steps + 1

    log_pf_steps = torch.zeros(
        (batch.num_graphs, int(start_nodes.size(1)), max_actions),
        device=batch.node_ptr.device,
        dtype=torch.float32,
    )
    # Move-step backward logits are no longer reconstructed on the training hot
    # path because the current SubTB objective only consumes forward prefixes.
    # Terminal stop-action backward scores are still written after rollout assembly.
    log_pb_steps = torch.zeros_like(log_pf_steps)
    log_reward_steps = torch.zeros_like(log_pf_steps)
    next_state_log_f_steps = torch.zeros_like(log_pf_steps)
    move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)
    gold_stop_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)

    current_nodes = start_nodes.clone()
    absorbing_mask = torch.zeros_like(start_nodes, dtype=torch.bool)
    num_steps = torch.zeros_like(start_nodes)
    current_path_token_ids = initialize_path_token_ids(
        start_nodes=start_nodes,
        max_steps=max_steps,
    )
    current_control_states = policy.build_start_control_states(
        prepared_batch,
        start_nodes,
    )
    explicit_stop_log_rewards = torch.zeros_like(start_nodes, dtype=torch.float32)
    total_agents = int(batch.num_graphs * int(start_nodes.size(1)))
    total_active_agent_count = 0
    total_unique_active_state_count = 0
    total_raw_graph_candidate_count = 0
    total_scored_graph_candidate_count = 0

    for step_idx in range(max_actions):
        active_mask = termination_action_steps > step_idx
        if not bool(active_mask.any().item()):
            break

        search_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=~active_mask,
            num_steps=num_steps,
            path_token_ids=current_path_token_ids,
            control_state=current_control_states,
            absorbing_mask=absorbing_mask,
        )
        distribution = apply_forward_constraints(
            policy.compute_forward_distribution(prepared_batch, search_state),
            state=search_state,
            max_steps=max_steps,
        )
        total_active_agent_count += int(distribution.active_agent_count)
        total_unique_active_state_count += int(distribution.unique_active_state_count)
        total_raw_graph_candidate_count += int(distribution.raw_graph_candidate_count)
        total_scored_graph_candidate_count += int(
            distribution.scored_graph_candidate_count
        )
        chosen_edge_ids = planned_edge_ids[:, :, step_idx].reshape(-1)
        chosen_is_stop_action = planned_stop_mask[:, :, step_idx].reshape(-1)
        flat_active = active_mask.reshape(-1)
        flat_current_nodes = current_nodes.reshape(-1)
        flat_num_steps = num_steps.reshape(-1)
        chosen_target_nodes = flat_current_nodes.clone()
        chosen_log_probs = torch.zeros(
            (total_agents,),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        selected_nodes, selected_log_probs = _select_edge_log_probs(
            distribution=distribution,
            selected_edge_ids=chosen_edge_ids,
            selected_is_stop_action=chosen_is_stop_action,
            active_mask=flat_active,
            policy=policy,
            error_prefix=(f"Sampled trajectory step={step_idx}"),
        )
        current_on_target = terminal_target_mask.index_select(
            0, flat_current_nodes
        ).view_as(current_nodes)
        flat_stop_base_log_rewards = _resolve_stop_branch_base_log_rewards(
            policy=policy,
            prepared_batch=prepared_batch,
            search_state=search_state,
            distribution=distribution,
            total_agents=total_agents,
            on_target=current_on_target.view(-1),
            answer_stop_log_reward_bonus=answer_stop_log_reward_bonus,
        )
        chosen_target_nodes[flat_active] = selected_nodes[flat_active]
        chosen_log_probs[flat_active] = selected_log_probs[flat_active]

        flat_graph_move = flat_active & (~chosen_is_stop_action)
        next_nodes = flat_current_nodes.clone()
        next_nodes[flat_graph_move] = chosen_target_nodes[flat_graph_move]
        next_num_steps = flat_num_steps.clone()
        next_num_steps[flat_graph_move] = next_num_steps[flat_graph_move] + 1
        explicit_stop_log_rewards.view(-1)[flat_active & chosen_is_stop_action] = (
            flat_stop_base_log_rewards[flat_active & chosen_is_stop_action]
        )
        chosen_relation_ids = _resolve_chosen_relation_ids(
            edge_type=prepared_batch.topology.edge_type,
            chosen_edge_ids=chosen_edge_ids,
            view_shape=current_nodes.shape,
        )
        next_path_token_ids = append_relation_and_node_tokens_inplace(
            path_token_ids=current_path_token_ids,
            num_steps=num_steps,
            relation_ids=chosen_relation_ids,
            target_nodes=next_nodes.view_as(current_nodes),
            active_mask=flat_graph_move.view_as(active_mask),
        )
        next_path_token_ids = append_stop_token_inplace(
            path_token_ids=next_path_token_ids,
            num_steps=num_steps,
            active_mask=chosen_is_stop_action.view_as(active_mask),
        )
        next_control_states = current_control_states.clone()
        if bool(flat_graph_move.any().item()):
            flat_next_control_states = next_control_states.view(
                -1, int(next_control_states.size(-1))
            )
            flat_current_control_states = current_control_states.view(
                -1, int(current_control_states.size(-1))
            )
            flat_relation_ids = chosen_relation_ids.view(-1)
            flat_next_control_states[flat_graph_move] = (
                policy.compute_next_control_states(
                    prepared_batch,
                    control_states=flat_current_control_states[flat_graph_move],
                    next_nodes=next_nodes[flat_graph_move],
                    relation_ids=flat_relation_ids[flat_graph_move],
                )
            )
        next_log_f = torch.zeros_like(current_nodes, dtype=torch.float32)
        if bool(flat_graph_move.any().item()):
            next_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=next_nodes.view_as(current_nodes),
                done_mask=torch.zeros_like(active_mask),
                num_steps=next_num_steps.view_as(num_steps),
                path_token_ids=next_path_token_ids,
                control_state=next_control_states,
            )
            next_log_f = policy.compute_log_state_scores(prepared_batch, next_state)

        log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
        next_state_log_f_steps[:, :, step_idx] = next_log_f
        move_mask[:, :, step_idx] = flat_graph_move.view_as(current_nodes)
        gold_stop_mask[:, :, step_idx] = chosen_is_stop_action.view_as(
            current_nodes
        ) & (current_on_target)
        log_reward_steps[:, :, step_idx] = _build_step_log_rewards(
            policy=policy,
            prepared_batch=prepared_batch,
            current_nodes=current_nodes,
            next_nodes=next_nodes.view_as(current_nodes),
            move_mask=move_mask[:, :, step_idx],
            gold_stop_mask=gold_stop_mask[:, :, step_idx],
            step_log_penalty=step_log_penalty,
            answer_stop_log_reward_bonus=answer_stop_log_reward_bonus,
        )
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)
        current_path_token_ids = next_path_token_ids
        current_control_states = next_control_states
        absorbing_mask = absorbing_mask | chosen_is_stop_action.view_as(active_mask)

    success_mask = terminal_target_mask.index_select(0, current_nodes.view(-1)).view_as(
        current_nodes
    )
    terminal_transition = trajectory_supervisor.resolve_terminal_transitions(
        batch=batch,
        terminal_nodes=current_nodes,
    )
    explicit_stop_mask = trace_stop_mask.any(dim=-1)
    terminal_log_rewards, terminal_rewards = _apply_terminal_log_reward_overrides(
        policy=policy,
        terminal_transition=terminal_transition,
        explicit_stop_mask=explicit_stop_mask,
        explicit_stop_log_rewards=explicit_stop_log_rewards,
    )
    masked_terminal_backward_log_probs = _mask_terminal_stop_action_backward_log_probs(
        termination_action_steps=termination_action_steps,
        terminal_num_steps=path_lengths,
        terminal_backward_log_probs=terminal_transition.terminal_backward_log_probs,
    )
    log_pb_steps = _apply_terminal_stop_action_backward_log_probs(
        log_pb_steps=log_pb_steps,
        termination_action_steps=termination_action_steps,
        terminal_num_steps=path_lengths,
        terminal_backward_log_probs=masked_terminal_backward_log_probs,
    )
    return TrajectoryGFNSampleBatch(
        graph_log_z=graph_log_z,
        start_nodes=start_nodes,
        start_log_probs=start_log_probs,
        start_state_log_f=start_log_flows.to(dtype=torch.float32),
        start_log_rewards=start_log_rewards,
        log_pf_steps=log_pf_steps,
        log_pb_steps=log_pb_steps,
        log_reward_steps=log_reward_steps,
        state_log_f_steps=None,
        next_state_log_f_steps=next_state_log_f_steps,
        move_mask=move_mask,
        trace_nodes=trace_nodes,
        trace_edge_ids=trace_edge_ids,
        trace_num_steps=trace_num_steps,
        trace_mask=trace_mask,
        trace_stop_mask=trace_stop_mask,
        terminal_nodes=current_nodes,
        terminal_entity_ids=terminal_transition.terminal_entity_ids,
        terminal_is_gold=terminal_transition.terminal_is_gold,
        terminal_sink_ids=terminal_transition.terminal_sink_ids,
        terminal_sink_log_rewards=terminal_transition.terminal_sink_log_rewards,
        gold_answer_counts=terminal_transition.gold_answer_counts,
        terminal_num_steps=path_lengths,
        termination_action_steps=termination_action_steps,
        terminal_state_log_f=None,
        terminal_rewards=terminal_rewards,
        terminal_log_rewards=terminal_log_rewards,
        terminal_backward_log_probs=masked_terminal_backward_log_probs,
        success_mask=success_mask,
        total_active_agent_count=total_active_agent_count,
        total_unique_active_state_count=total_unique_active_state_count,
        total_raw_graph_candidate_count=total_raw_graph_candidate_count,
        total_scored_graph_candidate_count=total_scored_graph_candidate_count,
    )


@dataclass(frozen=True)
class TrajectoryGFNSampleBatch:
    """Trajectory samples with explicit STOP-marked sequence prefixes.

    ``trace_edge_ids`` keeps only graph expansion moves, while the exact state
    sequence itself is stored in ``path_token_ids`` with the terminal STOP token
    appended for absorbing states. ``termination_action_steps`` is the required
    explicit STOP-action index for each rollout and must equal
    ``terminal_num_steps + 1`` because ``terminal_num_steps`` counts only graph
    moves.
    Terminal metadata includes both the entity-level reward anchor and the
    answer-sink ids/log-rewards used by the quotient-style terminal loss.
    """

    graph_log_z: torch.Tensor
    start_nodes: torch.Tensor
    start_log_probs: torch.Tensor
    start_state_log_f: torch.Tensor
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    next_state_log_f_steps: torch.Tensor
    move_mask: torch.Tensor
    trace_nodes: torch.Tensor
    trace_edge_ids: torch.Tensor
    trace_num_steps: torch.Tensor
    trace_mask: torch.Tensor
    terminal_nodes: torch.Tensor
    terminal_num_steps: torch.Tensor
    termination_action_steps: torch.Tensor
    terminal_rewards: torch.Tensor
    terminal_log_rewards: torch.Tensor
    success_mask: torch.Tensor
    start_log_rewards: torch.Tensor | None = None
    log_reward_steps: torch.Tensor | None = None
    state_log_f_steps: torch.Tensor | None = None
    trace_stop_mask: torch.Tensor | None = None
    terminal_entity_ids: torch.Tensor | None = None
    terminal_is_gold: torch.Tensor | None = None
    terminal_sink_ids: torch.Tensor | None = None
    terminal_sink_log_rewards: torch.Tensor | None = None
    gold_answer_counts: torch.Tensor | None = None
    terminal_state_log_f: torch.Tensor | None = None
    terminal_backward_log_probs: torch.Tensor | None = None
    start_entropy: torch.Tensor | None = None
    start_entropy_normalized: torch.Tensor | None = None
    proposal_start_target_kl: torch.Tensor | None = None
    total_active_agent_count: int = 0
    total_unique_active_state_count: int = 0
    total_raw_graph_candidate_count: int = 0
    total_scored_graph_candidate_count: int = 0

    def __post_init__(self) -> None:
        if tuple(self.termination_action_steps.shape) != tuple(
            self.terminal_num_steps.shape
        ):
            raise ValueError(
                "termination_action_steps must match terminal_num_steps shape. "
                f"termination_action_steps={tuple(self.termination_action_steps.shape)} "
                f"terminal_num_steps={tuple(self.terminal_num_steps.shape)}."
            )
        termination_action_steps = self.termination_action_steps.to(dtype=torch.long)
        terminal_num_steps = self.terminal_num_steps.to(dtype=torch.long)
        if bool((termination_action_steps <= 0).any().item()):
            raise ValueError(
                "termination_action_steps must be positive explicit STOP indices."
            )
        expected_termination_steps = terminal_num_steps + 1
        if not bool(
            (termination_action_steps == expected_termination_steps).all().item()
        ):
            raise ValueError(
                "termination_action_steps must equal terminal_num_steps + 1 so the "
                "terminal STOP action is represented explicitly in SubTB."
            )

    @property
    def proposal_start_entropy(self) -> torch.Tensor | None:
        return self.start_entropy

    @property
    def proposal_start_entropy_normalized(self) -> torch.Tensor | None:
        return self.start_entropy_normalized


class TrajectorySamplerProtocol(Protocol):
    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        rollouts_per_graph: int,
        temperature: float,
        action_prior_scale: float = 1.0,
        transition_bias_scale: float = 1.0,
    ) -> TrajectoryGFNSampleBatch: ...

    def rebuild_sample_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        start_nodes: torch.Tensor,
        planned_edge_ids: torch.Tensor,
        planned_stop_mask: torch.Tensor,
        path_lengths: torch.Tensor,
        termination_action_steps: torch.Tensor,
        trace_nodes: torch.Tensor,
        trace_edge_ids: torch.Tensor,
        trace_num_steps: torch.Tensor,
        trace_mask: torch.Tensor,
        trace_stop_mask: torch.Tensor,
    ) -> TrajectoryGFNSampleBatch: ...


class ForwardTrajectoryGFNSampler:
    def __init__(
        self,
        *,
        max_steps: int,
        trajectory_supervisor: TrajectoryRolloutSupervisorProtocol,
        force_stop_on_answer_hit: bool = False,
        step_log_penalty: float = 0.0,
        answer_stop_log_reward_bonus: float = 0.0,
    ) -> None:
        self.max_steps = int(max_steps)
        self.trajectory_supervisor = trajectory_supervisor
        self.force_stop_on_answer_hit = bool(force_stop_on_answer_hit)
        if float(step_log_penalty) > 0.0:
            raise ValueError(
                "step_log_penalty must be <= 0 for ForwardTrajectoryGFNSampler."
            )
        if float(answer_stop_log_reward_bonus) < 0.0:
            raise ValueError(
                "answer_stop_log_reward_bonus must be >= 0 for ForwardTrajectoryGFNSampler."
            )
        self.step_log_penalty = float(step_log_penalty)
        self.answer_stop_log_reward_bonus = float(answer_stop_log_reward_bonus)

    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        rollouts_per_graph: int,
        temperature: float,
        action_prior_scale: float = 1.0,
        transition_bias_scale: float = 1.0,
    ) -> TrajectoryGFNSampleBatch:
        with torch.no_grad():
            proposal_start_dist = policy.compute_proposal_start_distribution(
                prepared_batch,
                action_prior_scale=action_prior_scale,
            )
            sampling_start_log_probs = _temper_segmented_log_probs(
                log_probs=proposal_start_dist.log_probs,
                segment_ids=proposal_start_dist.candidate_graph_ids,
                num_segments=int(proposal_start_dist.graph_log_z.numel()),
                temperature=float(temperature),
            )
            (
                start_entropy,
                start_entropy_normalized,
            ) = _compute_start_distribution_entropy(
                log_probs=sampling_start_log_probs,
                candidate_graph_ids=proposal_start_dist.candidate_graph_ids,
                num_graphs=int(proposal_start_dist.graph_log_z.numel()),
            )
            start_nodes, _, _ = policy.sample_start_nodes(
                proposal_start_dist,
                num_rollouts=int(rollouts_per_graph),
                deterministic=False,
                temperature=float(temperature),
            )
        (
            start_log_probs,
            start_log_flows,
            start_log_rewards,
            graph_log_z,
            target_start_distribution,
        ) = _resolve_selected_start_values(
            prepared_batch=prepared_batch,
            policy=policy,
            start_nodes=start_nodes,
        )
        proposal_start_target_kl = _compute_root_proposal_target_kl(
            proposal_distribution=proposal_start_dist,
            target_distribution=target_start_distribution,
        )
        num_graphs, num_rollouts = start_nodes.shape
        terminal_target_mask = self.trajectory_supervisor.build_terminal_target_mask(
            batch=batch
        )
        max_actions = self.max_steps + 1
        trace_nodes = torch.zeros(
            (num_graphs, num_rollouts, max_actions),
            device=batch.node_ptr.device,
            dtype=torch.long,
        )
        trace_edge_ids = torch.full_like(trace_nodes, fill_value=-1)
        trace_num_steps = torch.zeros_like(trace_nodes)
        trace_mask = torch.zeros(
            (num_graphs, num_rollouts, max_actions),
            device=batch.node_ptr.device,
            dtype=torch.bool,
        )
        trace_stop_mask = torch.zeros_like(trace_mask)
        log_pf_steps = torch.zeros(
            (num_graphs, num_rollouts, max_actions),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        # Move-step backward logits are no longer reconstructed on the training
        # hot path because the current SubTB objective only consumes forward
        # prefixes. Terminal stop-action backward scores are still written below.
        log_pb_steps = torch.zeros_like(log_pf_steps)
        log_reward_steps = torch.zeros_like(log_pf_steps)
        next_state_log_f_steps = torch.zeros_like(log_pf_steps)
        move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)
        gold_stop_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)

        current_nodes = start_nodes.clone()
        done_mask = torch.zeros_like(start_nodes, dtype=torch.bool)
        absorbing_mask = torch.zeros_like(start_nodes, dtype=torch.bool)
        num_steps = torch.zeros_like(start_nodes)
        current_path_token_ids = initialize_path_token_ids(
            start_nodes=start_nodes,
            max_steps=self.max_steps,
        )
        current_control_states = policy.build_start_control_states(
            prepared_batch,
            start_nodes,
        )
        termination_action_steps = torch.zeros_like(start_nodes)
        explicit_stop_log_rewards = torch.zeros_like(start_nodes, dtype=torch.float32)
        total_agents = int(num_graphs * num_rollouts)
        total_active_agent_count = 0
        total_unique_active_state_count = 0
        total_raw_graph_candidate_count = 0
        total_scored_graph_candidate_count = 0

        for step_idx in range(max_actions):
            active_mask = ~done_mask
            trace_nodes[:, :, step_idx] = current_nodes
            trace_num_steps[:, :, step_idx] = num_steps
            trace_mask[:, :, step_idx] = active_mask

            if not bool(active_mask.any().item()):
                break

            on_target = terminal_target_mask.index_select(
                0, current_nodes.view(-1)
            ).view_as(current_nodes)
            forced_stop_mask = (
                active_mask & on_target
                if self.force_stop_on_answer_hit
                else torch.zeros_like(active_mask)
            )
            search_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=current_nodes,
                done_mask=done_mask,
                num_steps=num_steps,
                path_token_ids=current_path_token_ids,
                control_state=current_control_states,
                absorbing_mask=absorbing_mask,
            )
            target_distribution = apply_forward_constraints(
                policy.compute_forward_distribution(prepared_batch, search_state),
                state=search_state,
                max_steps=self.max_steps,
            )
            total_active_agent_count += int(target_distribution.active_agent_count)
            total_unique_active_state_count += int(
                target_distribution.unique_active_state_count
            )
            total_raw_graph_candidate_count += int(
                target_distribution.raw_graph_candidate_count
            )
            total_scored_graph_candidate_count += int(
                target_distribution.scored_graph_candidate_count
            )
            flat_stop_base_log_rewards = _resolve_stop_branch_base_log_rewards(
                policy=policy,
                prepared_batch=prepared_batch,
                search_state=search_state,
                distribution=target_distribution,
                total_agents=total_agents,
                on_target=on_target.view(-1),
                answer_stop_log_reward_bonus=self.answer_stop_log_reward_bonus,
            )
            _, _, has_values = policy.compute_move_log_probs(target_distribution)
            has_values = has_values.view_as(current_nodes)
            policy_active_mask = active_mask & (~forced_stop_mask)
            dead_end = policy_active_mask & (~has_values)
            flat_active = active_mask.view(-1)
            flat_policy_active = policy_active_mask.view(-1)
            flat_forced_stop = forced_stop_mask.view(-1)

            chosen_edge_ids = torch.full(
                (total_agents,),
                fill_value=-1,
                device=batch.node_ptr.device,
                dtype=torch.long,
            )
            chosen_target_nodes = current_nodes.view(-1).clone()
            chosen_is_stop_action = torch.zeros(
                (total_agents,), device=batch.node_ptr.device, dtype=torch.bool
            )
            chosen_log_probs = torch.zeros(
                (total_agents,), device=batch.node_ptr.device, dtype=torch.float32
            )
            chosen_is_stop_action[flat_forced_stop] = True
            selected_mask = flat_forced_stop.clone()
            if bool(flat_policy_active.any().item()):
                with torch.no_grad():
                    proposal_distribution = ForwardActionDistribution(
                        edge_logits=policy.compute_proposal_edge_logits(
                            prepared_batch,
                            search_state,
                            target_distribution,
                            action_prior_scale=action_prior_scale,
                            transition_bias_scale=transition_bias_scale,
                        ),
                        edge_agent_batch=target_distribution.edge_agent_batch,
                        edge_ids=target_distribution.edge_ids,
                        target_nodes=target_distribution.target_nodes,
                        out_degrees=target_distribution.out_degrees,
                        is_stop_action=target_distribution.is_stop_action,
                        is_root_action=target_distribution.is_root_action,
                        current_log_f=target_distribution.current_log_f,
                        active_agent_count=target_distribution.active_agent_count,
                        unique_active_state_count=target_distribution.unique_active_state_count,
                        raw_graph_candidate_count=target_distribution.raw_graph_candidate_count,
                        scored_graph_candidate_count=target_distribution.scored_graph_candidate_count,
                    )
                    (
                        sampled_positions,
                        sampled_edge_ids,
                        _,
                        _,
                        sampled_is_stop_action,
                    ) = _sample_edges(
                        distribution=proposal_distribution,
                        temperature=float(temperature),
                    )
                sampled_selected_mask = sampled_positions >= 0
                selected_mask = selected_mask | sampled_selected_mask
                chosen_edge_ids[sampled_selected_mask] = sampled_edge_ids[
                    sampled_selected_mask
                ]
                chosen_is_stop_action[sampled_selected_mask] = sampled_is_stop_action[
                    sampled_selected_mask
                ]
            if bool(selected_mask.any().item()):
                selected_nodes, selected_log_probs = _select_edge_log_probs(
                    distribution=target_distribution,
                    selected_edge_ids=chosen_edge_ids,
                    selected_is_stop_action=chosen_is_stop_action,
                    active_mask=selected_mask,
                    policy=policy,
                    error_prefix=f"Sampled trajectory step={step_idx}",
                )
                chosen_target_nodes[selected_mask] = selected_nodes[selected_mask]
                chosen_log_probs[selected_mask] = selected_log_probs[selected_mask]

            flat_current = current_nodes.view(-1)
            flat_next_nodes = flat_current.clone()
            flat_num_steps = num_steps.view(-1)
            next_num_steps = flat_num_steps.clone()
            flat_stop_action = selected_mask & chosen_is_stop_action
            flat_graph_move = selected_mask & (~chosen_is_stop_action)
            flat_next_nodes[flat_graph_move] = chosen_target_nodes[flat_graph_move]
            next_num_steps[flat_graph_move] = next_num_steps[flat_graph_move] + 1
            trace_edge_ids[:, :, step_idx] = chosen_edge_ids.view_as(current_nodes)
            trace_stop_mask[:, :, step_idx] = chosen_is_stop_action.view_as(
                current_nodes
            )
            termination_action_steps[selected_mask.view_as(current_nodes)] = (
                step_idx + 1
            )
            explicit_stop_log_rewards.view(-1)[flat_stop_action] = (
                flat_stop_base_log_rewards[flat_stop_action]
            )

            chosen_relation_ids = _resolve_chosen_relation_ids(
                edge_type=prepared_batch.topology.edge_type,
                chosen_edge_ids=chosen_edge_ids,
                view_shape=current_nodes.shape,
            )
            next_path_token_ids = append_relation_and_node_tokens_inplace(
                path_token_ids=current_path_token_ids,
                num_steps=num_steps,
                relation_ids=chosen_relation_ids,
                target_nodes=flat_next_nodes.view_as(current_nodes),
                active_mask=flat_graph_move.view_as(active_mask),
            )
            next_path_token_ids = append_stop_token_inplace(
                path_token_ids=next_path_token_ids,
                num_steps=num_steps,
                active_mask=flat_stop_action.view_as(active_mask),
            )
            next_control_states = current_control_states.clone()
            if bool(flat_graph_move.any().item()):
                flat_next_control_states = next_control_states.view(
                    -1, int(next_control_states.size(-1))
                )
                flat_current_control_states = current_control_states.view(
                    -1, int(current_control_states.size(-1))
                )
                flat_relation_ids = chosen_relation_ids.view(-1)
                flat_next_control_states[flat_graph_move] = (
                    policy.compute_next_control_states(
                        prepared_batch,
                        control_states=flat_current_control_states[flat_graph_move],
                        next_nodes=flat_next_nodes[flat_graph_move],
                        relation_ids=flat_relation_ids[flat_graph_move],
                    )
                )
            next_log_f = torch.zeros_like(current_nodes, dtype=torch.float32)
            if bool(flat_graph_move.any().item()):
                next_state = SearchState(
                    topology=prepared_batch.topology,
                    observation=prepared_batch.observation,
                    current_nodes=flat_next_nodes.view_as(current_nodes),
                    done_mask=torch.zeros_like(active_mask),
                    num_steps=next_num_steps.view_as(num_steps),
                    path_token_ids=next_path_token_ids,
                    control_state=next_control_states,
                )
                next_log_f = policy.compute_log_state_scores(prepared_batch, next_state)

            log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
            next_state_log_f_steps[:, :, step_idx] = next_log_f
            move_mask[:, :, step_idx] = flat_graph_move.view_as(current_nodes)
            gold_stop_mask[:, :, step_idx] = flat_stop_action.view_as(current_nodes) & (
                on_target
            )
            log_reward_steps[:, :, step_idx] = _build_step_log_rewards(
                policy=policy,
                prepared_batch=prepared_batch,
                current_nodes=current_nodes,
                next_nodes=flat_next_nodes.view_as(current_nodes),
                move_mask=move_mask[:, :, step_idx],
                gold_stop_mask=gold_stop_mask[:, :, step_idx],
                step_log_penalty=self.step_log_penalty,
                answer_stop_log_reward_bonus=self.answer_stop_log_reward_bonus,
            )

            current_nodes = flat_next_nodes.view_as(current_nodes)
            num_steps = next_num_steps.view_as(num_steps)
            current_path_token_ids = next_path_token_ids
            current_control_states = next_control_states
            absorbing_mask = absorbing_mask | flat_stop_action.view_as(absorbing_mask)
            done_mask = done_mask | dead_end | flat_stop_action.view_as(done_mask)

        success_mask = terminal_target_mask.index_select(
            0, current_nodes.view(-1)
        ).view_as(current_nodes)
        terminal_transition = self.trajectory_supervisor.resolve_terminal_transitions(
            batch=batch,
            terminal_nodes=current_nodes,
        )
        explicit_stop_mask = trace_stop_mask.any(dim=-1)
        terminal_log_rewards, terminal_rewards = _apply_terminal_log_reward_overrides(
            policy=policy,
            terminal_transition=terminal_transition,
            explicit_stop_mask=explicit_stop_mask,
            explicit_stop_log_rewards=explicit_stop_log_rewards,
        )
        masked_terminal_backward_log_probs = _mask_terminal_stop_action_backward_log_probs(
            termination_action_steps=termination_action_steps,
            terminal_num_steps=num_steps,
            terminal_backward_log_probs=terminal_transition.terminal_backward_log_probs,
        )
        log_pb_steps = _apply_terminal_stop_action_backward_log_probs(
            log_pb_steps=log_pb_steps,
            termination_action_steps=termination_action_steps,
            terminal_num_steps=num_steps,
            terminal_backward_log_probs=masked_terminal_backward_log_probs,
        )
        return TrajectoryGFNSampleBatch(
            graph_log_z=graph_log_z,
            start_nodes=start_nodes,
            start_log_probs=start_log_probs,
            start_state_log_f=start_log_flows.to(dtype=torch.float32),
            start_log_rewards=start_log_rewards,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_reward_steps=log_reward_steps,
            state_log_f_steps=None,
            next_state_log_f_steps=next_state_log_f_steps,
            move_mask=move_mask,
            trace_nodes=trace_nodes,
            trace_edge_ids=trace_edge_ids,
            trace_num_steps=trace_num_steps,
            trace_mask=trace_mask,
            trace_stop_mask=trace_stop_mask,
            terminal_nodes=current_nodes,
            terminal_entity_ids=terminal_transition.terminal_entity_ids,
            terminal_is_gold=terminal_transition.terminal_is_gold,
            terminal_sink_ids=terminal_transition.terminal_sink_ids,
            terminal_sink_log_rewards=terminal_transition.terminal_sink_log_rewards,
            gold_answer_counts=terminal_transition.gold_answer_counts,
            terminal_num_steps=num_steps,
            termination_action_steps=termination_action_steps,
            terminal_state_log_f=None,
            terminal_rewards=terminal_rewards,
            terminal_log_rewards=terminal_log_rewards,
            terminal_backward_log_probs=masked_terminal_backward_log_probs,
            success_mask=success_mask,
            start_entropy=start_entropy,
            start_entropy_normalized=start_entropy_normalized,
            proposal_start_target_kl=proposal_start_target_kl,
            total_active_agent_count=total_active_agent_count,
            total_unique_active_state_count=total_unique_active_state_count,
            total_raw_graph_candidate_count=total_raw_graph_candidate_count,
            total_scored_graph_candidate_count=total_scored_graph_candidate_count,
        )

    def rebuild_sample_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        start_nodes: torch.Tensor,
        planned_edge_ids: torch.Tensor,
        planned_stop_mask: torch.Tensor,
        path_lengths: torch.Tensor,
        termination_action_steps: torch.Tensor,
        trace_nodes: torch.Tensor,
        trace_edge_ids: torch.Tensor,
        trace_num_steps: torch.Tensor,
        trace_mask: torch.Tensor,
        trace_stop_mask: torch.Tensor,
    ) -> TrajectoryGFNSampleBatch:
        return _rebuild_target_sample_batch(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            trajectory_supervisor=self.trajectory_supervisor,
            start_nodes=start_nodes,
            planned_edge_ids=planned_edge_ids,
            planned_stop_mask=planned_stop_mask,
            path_lengths=path_lengths,
            termination_action_steps=termination_action_steps,
            trace_nodes=trace_nodes,
            trace_edge_ids=trace_edge_ids,
            trace_num_steps=trace_num_steps,
            trace_mask=trace_mask,
            trace_stop_mask=trace_stop_mask,
            max_steps=self.max_steps,
            step_log_penalty=self.step_log_penalty,
            answer_stop_log_reward_bonus=self.answer_stop_log_reward_bonus,
        )


__all__ = [
    "AnswerReachabilityTrajectorySupervisor",
    "ForwardTrajectoryGFNSampler",
    "TerminalTransitionBatch",
    "TrajectoryGFNSampleBatch",
    "TrajectoryRolloutSupervisorProtocol",
    "TrajectorySamplerProtocol",
]
