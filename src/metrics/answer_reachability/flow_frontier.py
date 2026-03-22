from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.configs import HorizonConfig, SearchEvalConfig
from src.models.gflownet import (
    PreparedSearchBatch,
    RootActionDistribution,
    SearchPolicyProtocol,
    SearchState,
    compute_constrained_policy_step,
)
from src.models.gflownet.path import (
    append_relation_and_node_tokens_inplace,
    initialize_path_token_ids,
)

from .analysis import ReachabilityAnalysis
from .posterior import DiscoveredTrajectory, build_window_result
from .schema import SupportWindowResult

_LOG_ZERO = float("-inf")


@dataclass(frozen=True)
class FlowFrontierSearchSummary:
    analysis: ReachabilityAnalysis
    discovered_paths: list[DiscoveredTrajectory]
    expanded_state_count: int
    remaining_mass_upper: float
    stop_reason: str
    coverage_certified: bool


@dataclass(frozen=True)
class _GraphSearchContext:
    graph_idx: int
    graph_batch: TrajectoryBatch
    node_offset: int
    edge_offset: int
    start_nodes_abs: torch.Tensor
    start_log_probs: torch.Tensor
    start_log_flows: torch.Tensor
    graph_log_z: float


@dataclass(frozen=True)
class _FrontierBatch:
    current_nodes_abs: torch.Tensor
    start_nodes_local: torch.Tensor
    num_steps: torch.Tensor
    path_token_ids: torch.Tensor
    control_states: torch.Tensor | None
    log_prefix_probs: torch.Tensor
    log_state_flows: torch.Tensor
    edge_traces_local: tuple[tuple[int, ...], ...]

    def size(self) -> int:
        return int(self.current_nodes_abs.numel())


def _compute_edge_offsets(batch: TrajectoryBatch) -> torch.Tensor:
    edge_counts = torch.bincount(batch.edge_batch, minlength=batch.num_graphs)
    return edge_counts.cumsum(0) - edge_counts


def _select_graph_search_context(
    *,
    batch: TrajectoryBatch,
    start_distribution: RootActionDistribution,
    graph_idx: int,
) -> _GraphSearchContext:
    graph_batch = batch.select_graph(graph_idx, validate=False)
    graph_mask = start_distribution.candidate_graph_ids == int(graph_idx)
    start_nodes_abs = start_distribution.candidate_nodes_abs[graph_mask]
    start_log_probs = start_distribution.log_probs[graph_mask].to(dtype=torch.float64)
    start_log_flows = start_distribution.log_flows[graph_mask].to(dtype=torch.float32)
    edge_offsets = _compute_edge_offsets(batch)
    return _GraphSearchContext(
        graph_idx=int(graph_idx),
        graph_batch=graph_batch,
        node_offset=int(batch.node_ptr[graph_idx].item()),
        edge_offset=int(edge_offsets[graph_idx].item()),
        start_nodes_abs=start_nodes_abs,
        start_log_probs=start_log_probs,
        start_log_flows=start_log_flows,
        graph_log_z=float(start_distribution.graph_log_z[graph_idx].item()),
    )


def _build_search_state(
    frontier: _FrontierBatch,
    *,
    prepared_batch: PreparedSearchBatch,
) -> SearchState:
    control_state = None
    if frontier.control_states is not None:
        control_state = frontier.control_states.unsqueeze(0)
    return SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=frontier.current_nodes_abs.unsqueeze(0),
        done_mask=torch.zeros(
            (1, frontier.size()),
            device=frontier.current_nodes_abs.device,
            dtype=torch.bool,
        ),
        num_steps=frontier.num_steps.unsqueeze(0),
        path_token_ids=frontier.path_token_ids.unsqueeze(0),
        control_state=control_state,
    )


def _sum_normalized_flow_mass(
    *,
    log_state_flows: torch.Tensor,
    graph_log_z: float,
) -> float:
    if int(log_state_flows.numel()) == 0:
        return 0.0
    probabilities = torch.exp(
        log_state_flows.to(dtype=torch.float64) - float(graph_log_z)
    )
    return float(probabilities.sum().item())


def _compute_log_probs(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.full_like(probabilities, fill_value=_LOG_ZERO),
    )


def _membership_mask(*, values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) == 0 or int(candidates.numel()) == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    sorted_candidates = torch.unique(
        candidates.to(device=values.device, dtype=torch.long), sorted=True
    )
    positions = torch.searchsorted(sorted_candidates, values.to(dtype=torch.long))
    in_range = positions < int(sorted_candidates.numel())
    if not bool(in_range.any().item()):
        return torch.zeros_like(values, dtype=torch.bool)
    safe_positions = positions.clamp_max(max(int(sorted_candidates.numel()) - 1, 0))
    matched = sorted_candidates.index_select(0, safe_positions) == values.to(
        dtype=torch.long
    )
    return in_range & matched


def _build_exact_analysis(
    *,
    batch: TrajectoryBatch,
    terminal_mass: torch.Tensor,
    expanded_state_count: int,
    remaining_mass_upper: float,
    stop_reason: str,
    coverage_certified: bool,
) -> ReachabilityAnalysis:
    terminal_mass = terminal_mass.to(device=batch.node_ptr.device, dtype=torch.float32)
    nonzero_mask = terminal_mass > 0
    if bool(nonzero_mask.any().item()):
        terminal_entities = batch.node_global_ids.index_select(
            0, torch.nonzero(nonzero_mask, as_tuple=False).view(-1)
        )
        terminal_probs = terminal_mass[nonzero_mask]
        answer_entity_ids, inverse = torch.unique(
            terminal_entities,
            sorted=True,
            return_inverse=True,
        )
        answer_probs = torch.zeros(
            (int(answer_entity_ids.numel()),),
            device=terminal_mass.device,
            dtype=torch.float32,
        )
        answer_probs.scatter_add_(0, inverse, terminal_probs)
    else:
        answer_entity_ids = torch.empty(
            (0,), device=terminal_mass.device, dtype=torch.long
        )
        answer_probs = torch.empty(
            (0,), device=terminal_mass.device, dtype=torch.float32
        )
    gold_mask = _membership_mask(
        values=answer_entity_ids,
        candidates=batch.answer_entity_ids,
    )
    gold_total_mass = (
        float(answer_probs[gold_mask].sum().item())
        if int(answer_probs.numel()) > 0
        else 0.0
    )
    log_terminal_mass = _compute_log_probs(terminal_mass)
    log_answer_probs = _compute_log_probs(answer_probs)
    log_gold_total_mass = (
        _LOG_ZERO if gold_total_mass <= 0.0 else float(math.log(gold_total_mass))
    )
    return ReachabilityAnalysis(
        terminal_mass=terminal_mass,
        answer_entity_ids=answer_entity_ids,
        answer_probs=answer_probs,
        gold_total_mass=gold_total_mass,
        answer_prob_ci_low=answer_probs.clone(),
        answer_prob_ci_high=answer_probs.clone(),
        gold_total_mass_ci_low=gold_total_mass,
        gold_total_mass_ci_high=gold_total_mass,
        ci_confidence_level=1.0,
        retrieval_answer_entity_ids=answer_entity_ids,
        retrieval_answer_probs=answer_probs,
        log_terminal_mass=log_terminal_mass,
        log_answer_probs=log_answer_probs,
        log_gold_total_mass=log_gold_total_mass,
        log_retrieval_answer_probs=log_answer_probs,
        inference_mode="flow_frontier",
        probe_count=int(expanded_state_count),
        remaining_mass_upper=min(max(float(remaining_mass_upper), 0.0), 1.0),
        stop_reason=str(stop_reason),
        coverage_certified=bool(coverage_certified),
    )


def _aggregate_discovered_paths(
    *,
    discovered_paths: list[DiscoveredTrajectory],
) -> list[DiscoveredTrajectory]:
    if not discovered_paths:
        return []
    probability_by_path: dict[tuple[int, int, tuple[int, ...]], float] = {}
    is_gold_by_path: dict[tuple[int, int, tuple[int, ...]], bool] = {}
    for path in discovered_paths:
        key = (int(path.start_node), int(path.terminal_node), tuple(path.edge_ids))
        probability_by_path[key] = probability_by_path.get(key, 0.0) + float(
            math.exp(path.log_prob)
        )
        is_gold_by_path[key] = bool(path.is_gold)
    aggregated: list[DiscoveredTrajectory] = []
    for (
        start_node,
        terminal_node,
        edge_ids,
    ), probability in probability_by_path.items():
        answer_entity_id = int(discovered_paths[0].answer_entity_id)
        for path in discovered_paths:
            if (
                path.start_node == start_node
                and path.terminal_node == terminal_node
                and path.edge_ids == edge_ids
            ):
                answer_entity_id = int(path.answer_entity_id)
                break
        aggregated.append(
            DiscoveredTrajectory(
                start_node=int(start_node),
                terminal_node=int(terminal_node),
                answer_entity_id=int(answer_entity_id),
                edge_ids=tuple(int(edge_id) for edge_id in edge_ids),
                log_prob=(
                    _LOG_ZERO if probability <= 0.0 else float(math.log(probability))
                ),
                is_gold=bool(is_gold_by_path[(start_node, terminal_node, edge_ids)]),
            )
        )
    aggregated.sort(
        key=lambda item: (
            -item.prob,
            item.answer_entity_id,
            item.edge_ids,
            item.start_node,
            item.terminal_node,
        )
    )
    return aggregated


def _add_terminal_path(
    *,
    context: _GraphSearchContext,
    current_node_abs: int,
    start_node_local: int,
    edge_trace_local: tuple[int, ...],
    log_prob: float,
    terminal_mass: torch.Tensor,
    discovered_paths: list[DiscoveredTrajectory],
) -> None:
    terminal_node_local = int(current_node_abs) - int(context.node_offset)
    if (
        terminal_node_local < 0
        or terminal_node_local >= context.graph_batch.num_nodes_total
    ):
        raise ValueError(
            "Terminal node escaped graph-local range during flow-frontier search. "
            f"terminal_node_abs={current_node_abs} node_offset={context.node_offset}."
        )
    probability = 0.0 if not math.isfinite(log_prob) else float(math.exp(log_prob))
    terminal_mass[terminal_node_local] = (
        terminal_mass[terminal_node_local] + probability
    )
    answer_entity_id = int(
        context.graph_batch.node_global_ids[terminal_node_local].item()
    )
    gold_answers = {
        int(value) for value in context.graph_batch.answer_entity_ids.tolist()
    }
    discovered_paths.append(
        DiscoveredTrajectory(
            start_node=int(start_node_local),
            terminal_node=int(terminal_node_local),
            answer_entity_id=answer_entity_id,
            edge_ids=tuple(int(edge_id) for edge_id in edge_trace_local),
            log_prob=float(log_prob),
            is_gold=answer_entity_id in gold_answers,
        )
    )


def _build_start_frontier(
    *,
    context: _GraphSearchContext,
    eval_cfg: SearchEvalConfig,
    prepared_batch: PreparedSearchBatch,
    policy: SearchPolicyProtocol,
    max_steps: int,
) -> tuple[_FrontierBatch | None, float]:
    if int(context.start_nodes_abs.numel()) == 0:
        return None, 0.0
    start_control_states = policy.build_start_control_states(
        prepared_batch,
        context.start_nodes_abs.view(1, -1),
    ).view(int(context.start_nodes_abs.numel()), -1)
    normalized_mass = torch.exp(
        context.start_log_flows.to(dtype=torch.float64) - float(context.graph_log_z)
    )
    prune_threshold = float(eval_cfg.flow_prune_epsilon)
    keep_mask = normalized_mass > 0.0
    if prune_threshold > 0.0:
        keep_mask = keep_mask & (normalized_mass >= prune_threshold)
    pruned_mass = float(normalized_mass[~keep_mask].sum().item())
    if not bool(keep_mask.any().item()):
        return None, pruned_mass
    kept_start_nodes = context.start_nodes_abs[keep_mask]
    path_token_ids = initialize_path_token_ids(
        start_nodes=kept_start_nodes.view(1, -1),
        max_steps=int(max_steps),
    ).view(int(kept_start_nodes.numel()), -1)
    return (
        _FrontierBatch(
            current_nodes_abs=kept_start_nodes,
            start_nodes_local=(kept_start_nodes - int(context.node_offset)).to(
                dtype=torch.long
            ),
            num_steps=torch.zeros_like(kept_start_nodes, dtype=torch.long),
            path_token_ids=path_token_ids,
            control_states=start_control_states[keep_mask],
            log_prefix_probs=context.start_log_probs[keep_mask],
            log_state_flows=context.start_log_flows[keep_mask],
            edge_traces_local=tuple(() for _ in range(int(kept_start_nodes.numel()))),
        ),
        pruned_mass,
    )


def _build_child_frontier(
    *,
    frontier: _FrontierBatch,
    prepared_batch: PreparedSearchBatch,
    policy: SearchPolicyProtocol,
    context: _GraphSearchContext,
    max_steps: int,
    child_parent_index: torch.Tensor,
    child_edge_ids_abs: torch.Tensor,
    child_target_nodes_abs: torch.Tensor,
    child_log_prefix_probs: torch.Tensor,
    child_edge_traces_local: tuple[tuple[int, ...], ...],
    eval_cfg: SearchEvalConfig,
) -> tuple[_FrontierBatch | None, float]:
    if int(child_edge_ids_abs.numel()) == 0:
        return None, 0.0
    relation_ids = prepared_batch.topology.edge_type.index_select(0, child_edge_ids_abs)
    child_num_steps = frontier.num_steps.index_select(0, child_parent_index) + 1
    child_path_token_ids = frontier.path_token_ids.index_select(
        0, child_parent_index
    ).clone()
    append_relation_and_node_tokens_inplace(
        path_token_ids=child_path_token_ids,
        num_steps=frontier.num_steps.index_select(0, child_parent_index),
        relation_ids=relation_ids,
        target_nodes=child_target_nodes_abs,
    )
    control_states = frontier.control_states
    child_control_states = None
    if control_states is not None:
        child_control_states = policy.compute_next_control_states(
            prepared_batch,
            control_states=control_states.index_select(0, child_parent_index),
            next_nodes=child_target_nodes_abs,
            relation_ids=relation_ids,
        )
    child_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=child_target_nodes_abs.unsqueeze(0),
        done_mask=torch.zeros(
            (1, int(child_target_nodes_abs.numel())),
            device=child_target_nodes_abs.device,
            dtype=torch.bool,
        ),
        num_steps=child_num_steps.unsqueeze(0),
        path_token_ids=child_path_token_ids.unsqueeze(0),
        control_state=(
            child_control_states.unsqueeze(0)
            if child_control_states is not None
            else None
        ),
    )
    child_log_state_flows = policy.compute_log_state_scores(
        prepared_batch,
        child_state,
    ).view(-1)
    normalized_mass = torch.exp(
        child_log_state_flows.to(dtype=torch.float64) - float(context.graph_log_z)
    )
    keep_mask = normalized_mass > 0.0
    if float(eval_cfg.flow_prune_epsilon) > 0.0:
        keep_mask = keep_mask & (normalized_mass >= float(eval_cfg.flow_prune_epsilon))
    pruned_mass = float(normalized_mass[~keep_mask].sum().item())
    if not bool(keep_mask.any().item()):
        return None, pruned_mass
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    kept_edge_traces = tuple(
        child_edge_traces_local[int(index)] for index in kept_indices.tolist()
    )
    kept_control_states = None
    if child_control_states is not None:
        kept_control_states = child_control_states.index_select(0, kept_indices)
    return (
        _FrontierBatch(
            current_nodes_abs=child_target_nodes_abs.index_select(0, kept_indices),
            start_nodes_local=frontier.start_nodes_local.index_select(
                0, child_parent_index.index_select(0, kept_indices)
            ),
            num_steps=child_num_steps.index_select(0, kept_indices),
            path_token_ids=child_path_token_ids.index_select(0, kept_indices),
            control_states=kept_control_states,
            log_prefix_probs=child_log_prefix_probs.index_select(0, kept_indices),
            log_state_flows=child_log_state_flows.index_select(0, kept_indices),
            edge_traces_local=kept_edge_traces,
        ),
        pruned_mass,
    )


def run_flow_frontier_search(
    *,
    batch: TrajectoryBatch,
    policy: SearchPolicyProtocol,
    prepared_batch: PreparedSearchBatch,
    max_steps: int,
    eval_cfg: SearchEvalConfig,
    start_distribution: RootActionDistribution,
    graph_idx: int = 0,
) -> FlowFrontierSearchSummary:
    context = _select_graph_search_context(
        batch=batch,
        start_distribution=start_distribution,
        graph_idx=graph_idx,
    )
    terminal_mass = torch.zeros(
        (context.graph_batch.num_nodes_total,),
        device=batch.node_ptr.device,
        dtype=torch.float64,
    )
    discovered_paths: list[DiscoveredTrajectory] = []
    frontier, remaining_mass_upper = _build_start_frontier(
        context=context,
        eval_cfg=eval_cfg,
        prepared_batch=prepared_batch,
        policy=policy,
        max_steps=max_steps,
    )
    expanded_state_count = 0
    coverage_certified = True
    stop_reason = "flow_frontier_exhausted"

    while frontier is not None and frontier.size() > 0:
        frontier_size = frontier.size()
        if frontier_size > int(eval_cfg.max_frontier_size):
            remaining_mass_upper += _sum_normalized_flow_mass(
                log_state_flows=frontier.log_state_flows,
                graph_log_z=context.graph_log_z,
            )
            coverage_certified = False
            stop_reason = "flow_frontier_frontier_budget_exhausted"
            break
        if expanded_state_count + frontier_size > int(eval_cfg.max_expansions):
            remaining_mass_upper += _sum_normalized_flow_mass(
                log_state_flows=frontier.log_state_flows,
                graph_log_z=context.graph_log_z,
            )
            coverage_certified = False
            stop_reason = "flow_frontier_expansion_budget_exhausted"
            break
        expanded_state_count += frontier_size

        state = _build_search_state(frontier, prepared_batch=prepared_batch)
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
            state=state,
            max_steps=int(max_steps),
        )
        has_values = step.has_values.to(dtype=torch.bool)
        dead_mask = ~has_values
        if bool(dead_mask.any().item()):
            for dead_index in (
                torch.nonzero(dead_mask, as_tuple=False).view(-1).tolist()
            ):
                _add_terminal_path(
                    context=context,
                    current_node_abs=int(frontier.current_nodes_abs[dead_index].item()),
                    start_node_local=int(frontier.start_nodes_local[dead_index].item()),
                    edge_trace_local=frontier.edge_traces_local[dead_index],
                    log_prob=float(frontier.log_prefix_probs[dead_index].item()),
                    terminal_mass=terminal_mass,
                    discovered_paths=discovered_paths,
                )

        distribution = step.distribution
        edge_agent_batch = distribution.edge_agent_batch
        action_counts = torch.zeros(
            (frontier_size,), device=edge_agent_batch.device, dtype=torch.long
        )
        if int(edge_agent_batch.numel()) > 0:
            action_counts.scatter_add_(
                0, edge_agent_batch, torch.ones_like(edge_agent_batch)
            )
        action_offsets = action_counts.cumsum(0) - action_counts
        stop_action_mask = (
            distribution.is_stop_action.to(dtype=torch.bool)
            if distribution.is_stop_action is not None
            else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
        )

        child_parent_indices: list[int] = []
        child_edge_ids_abs: list[int] = []
        child_target_nodes_abs: list[int] = []
        child_log_prefix_probs: list[float] = []
        child_edge_traces_local: list[tuple[int, ...]] = []

        for state_index in range(frontier_size):
            count = int(action_counts[state_index].item())
            if count == 0:
                continue
            start = int(action_offsets[state_index].item())
            end = start + count
            prefix_log_prob = float(frontier.log_prefix_probs[state_index].item())
            current_node_abs = int(frontier.current_nodes_abs[state_index].item())
            start_node_local = int(frontier.start_nodes_local[state_index].item())
            edge_trace_local = frontier.edge_traces_local[state_index]
            for action_index in range(start, end):
                action_log_prob = float(
                    step.move_log_probs[action_index].to(dtype=torch.float64).item()
                )
                trajectory_log_prob = prefix_log_prob + action_log_prob
                if bool(stop_action_mask[action_index].item()):
                    _add_terminal_path(
                        context=context,
                        current_node_abs=current_node_abs,
                        start_node_local=start_node_local,
                        edge_trace_local=edge_trace_local,
                        log_prob=trajectory_log_prob,
                        terminal_mass=terminal_mass,
                        discovered_paths=discovered_paths,
                    )
                    continue
                edge_id_abs = int(distribution.edge_ids[action_index].item())
                child_parent_indices.append(state_index)
                child_edge_ids_abs.append(edge_id_abs)
                child_target_nodes_abs.append(
                    int(distribution.target_nodes[action_index].item())
                )
                child_log_prefix_probs.append(trajectory_log_prob)
                child_edge_traces_local.append(
                    edge_trace_local + (edge_id_abs - int(context.edge_offset),)
                )

        if not child_edge_ids_abs:
            frontier = None
            continue

        frontier, pruned_mass = _build_child_frontier(
            frontier=frontier,
            prepared_batch=prepared_batch,
            policy=policy,
            context=context,
            max_steps=max_steps,
            child_parent_index=torch.tensor(
                child_parent_indices,
                device=batch.node_ptr.device,
                dtype=torch.long,
            ),
            child_edge_ids_abs=torch.tensor(
                child_edge_ids_abs,
                device=batch.node_ptr.device,
                dtype=torch.long,
            ),
            child_target_nodes_abs=torch.tensor(
                child_target_nodes_abs,
                device=batch.node_ptr.device,
                dtype=torch.long,
            ),
            child_log_prefix_probs=torch.tensor(
                child_log_prefix_probs,
                device=batch.node_ptr.device,
                dtype=torch.float64,
            ),
            child_edge_traces_local=tuple(child_edge_traces_local),
            eval_cfg=eval_cfg,
        )
        remaining_mass_upper += float(pruned_mass)

    aggregated_paths = _aggregate_discovered_paths(discovered_paths=discovered_paths)
    analysis = _build_exact_analysis(
        batch=context.graph_batch,
        terminal_mass=terminal_mass,
        expanded_state_count=int(expanded_state_count),
        remaining_mass_upper=float(remaining_mass_upper),
        stop_reason=stop_reason,
        coverage_certified=bool(coverage_certified),
    )
    return FlowFrontierSearchSummary(
        analysis=analysis,
        discovered_paths=aggregated_paths,
        expanded_state_count=int(expanded_state_count),
        remaining_mass_upper=min(max(float(remaining_mass_upper), 0.0), 1.0),
        stop_reason=stop_reason,
        coverage_certified=bool(coverage_certified),
    )


class FlowFrontierReachabilityAnalyzer:
    def __init__(self, *, max_steps: int, eval_cfg: SearchEvalConfig) -> None:
        self.max_steps = int(max_steps)
        self.eval_cfg = eval_cfg

    def analyze(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ReachabilityAnalysis:
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
        summary = run_flow_frontier_search(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            eval_cfg=self.eval_cfg,
            start_distribution=start_distribution,
            graph_idx=0,
        )
        return summary.analysis

    def analyze_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> list[ReachabilityAnalysis]:
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
        return [
            run_flow_frontier_search(
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
                max_steps=self.max_steps,
                eval_cfg=self.eval_cfg,
                start_distribution=start_distribution,
                graph_idx=graph_idx,
            ).analysis
            for graph_idx in range(batch.num_graphs)
        ]


class FlowFrontierSupportSearch:
    requires_analysis = False

    def __init__(
        self, *, horizon_cfg: HorizonConfig, eval_cfg: SearchEvalConfig
    ) -> None:
        self.horizon_cfg = horizon_cfg
        self.eval_cfg = eval_cfg

    def _build_window_result(
        self,
        *,
        batch: TrajectoryBatch,
        summary: FlowFrontierSearchSummary,
        include_answer_support: bool,
    ) -> SupportWindowResult:
        discovered_total_mass = float(summary.analysis.answer_probs.sum().item())
        support_answer_upper_bounds = None
        if summary.remaining_mass_upper <= 0.0:
            support_answer_upper_bounds = {
                int(answer_id): float(prob)
                for answer_id, prob in zip(
                    summary.analysis.answer_entity_ids.tolist(),
                    summary.analysis.answer_probs.tolist(),
                )
            }
        result = build_window_result(
            batch=batch,
            discovered_paths=summary.discovered_paths,
            analysis=summary.analysis,
            inference_mode="flow_frontier",
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            probe_count=int(summary.expanded_state_count),
            remaining_mass_upper=float(summary.remaining_mass_upper),
            stop_reason=str(summary.stop_reason),
            coverage_certified=bool(summary.coverage_certified),
            answer_mass_reference="flow_frontier",
            support_mass_reference="flow_frontier",
            answer_mass_reference_total=1.0,
            support_answer_upper_bounds=support_answer_upper_bounds,
            include_answer_support=include_answer_support,
            ci_confidence_level=1.0,
            gold_total_mass_ci_low=float(summary.analysis.gold_total_mass),
            gold_total_mass_ci_high=float(summary.analysis.gold_total_mass),
        )
        uncovered_discovered_mass = max(
            discovered_total_mass - float(result.covered_mass), 0.0
        )
        return replace(
            result,
            remaining_mass_upper=min(
                max(
                    uncovered_discovered_mass + float(summary.remaining_mass_upper), 0.0
                ),
                1.0,
            ),
            covered_mass_ci_low=float(result.covered_mass),
            covered_mass_ci_high=float(result.covered_mass),
        )

    def generate_window(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: ReachabilityAnalysis | None = None,
        include_answer_support: bool = True,
    ) -> SupportWindowResult:
        del analysis
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
        summary = run_flow_frontier_search(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=int(self.horizon_cfg.max_steps),
            eval_cfg=self.eval_cfg,
            start_distribution=start_distribution,
            graph_idx=0,
        )
        return self._build_window_result(
            batch=batch,
            summary=summary,
            include_answer_support=include_answer_support,
        )

    def generate_windows_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: list[ReachabilityAnalysis] | None = None,
        include_answer_support: bool = True,
    ) -> list[SupportWindowResult]:
        del analysis
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
        results: list[SupportWindowResult] = []
        for graph_idx in range(batch.num_graphs):
            graph_batch = batch.select_graph(graph_idx, validate=False)
            summary = run_flow_frontier_search(
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
                max_steps=int(self.horizon_cfg.max_steps),
                eval_cfg=self.eval_cfg,
                start_distribution=start_distribution,
                graph_idx=graph_idx,
            )
            results.append(
                self._build_window_result(
                    batch=graph_batch,
                    summary=summary,
                    include_answer_support=include_answer_support,
                )
            )
        return results


__all__ = [
    "FlowFrontierReachabilityAnalyzer",
    "FlowFrontierSearchSummary",
    "FlowFrontierSupportSearch",
    "run_flow_frontier_search",
]
