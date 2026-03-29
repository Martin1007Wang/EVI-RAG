from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from src.graph import TrajectoryBatch
from src.models.configs import SearchEvalConfig
from src.models.gflownet.prefix_state import (
    PreparedSearchBatch,
    RootActionDistribution,
    SearchPolicyProtocol,
    SearchState,
)
from src.models.gflownet.transitions import compute_constrained_policy_step
from src.models.gflownet.path import (
    append_relation_and_node_tokens_inplace,
    append_stop_token_inplace,
    initialize_path_token_ids,
)
from src.utils.segment_ops import sample_segmented_one_1d

from .answer_metrics import (
    DiscoveredTrajectory,
    EdgeSupportAnalysis,
    ReachabilityAnalysis,
    ReachabilityBackendProtocol,
    SearchDiagnostics,
    SupportWindowResult,
    build_rank_only_result,
    build_window_result,
    graph_gold_answers,
    ranking_from_analysis,
)

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
        terminal_entities = batch.node_entity_ids.index_select(
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
    gold_answer_mass = (
        float(answer_probs[gold_mask].sum().item())
        if int(answer_probs.numel()) > 0
        else 0.0
    )
    return ReachabilityAnalysis(
        terminal_mass=terminal_mass,
        answer_entity_ids=answer_entity_ids,
        answer_probs=answer_probs,
        gold_answer_mass=gold_answer_mass,
        answer_prob_ci_low=answer_probs.clone(),
        answer_prob_ci_high=answer_probs.clone(),
        gold_answer_mass_ci_low=gold_answer_mass,
        gold_answer_mass_ci_high=gold_answer_mass,
        ci_confidence_level=1.0,
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
        context.graph_batch.node_entity_ids[terminal_node_local].item()
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
    prune_threshold = float(eval_cfg.flow_frontier.prune_epsilon)
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
    if float(eval_cfg.flow_frontier.prune_epsilon) > 0.0:
        keep_mask = keep_mask & (
            normalized_mass >= float(eval_cfg.flow_frontier.prune_epsilon)
        )
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
        if frontier_size > int(eval_cfg.flow_frontier.max_frontier_size):
            remaining_mass_upper += _sum_normalized_flow_mass(
                log_state_flows=frontier.log_state_flows,
                graph_log_z=context.graph_log_z,
            )
            coverage_certified = False
            stop_reason = "flow_frontier_frontier_budget_exhausted"
            break
        if expanded_state_count + frontier_size > int(
            eval_cfg.flow_frontier.max_expansions
        ):
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


class FlowFrontierBackend(ReachabilityBackendProtocol):
    inference_mode = "flow_frontier"

    def __init__(self, *, max_steps: int, eval_cfg: SearchEvalConfig) -> None:
        self.max_steps = int(max_steps)
        self.eval_cfg = eval_cfg

    def _build_graph_result(
        self,
        *,
        batch: TrajectoryBatch,
        summary: FlowFrontierSearchSummary,
        report_profile: str,
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
        diagnostics = SearchDiagnostics(
            inference_mode=self.inference_mode,
            probe_count=int(summary.expanded_state_count),
            remaining_mass_upper=float(summary.remaining_mass_upper),
            stop_reason=str(summary.stop_reason),
            coverage_certified=bool(summary.coverage_certified),
            covered_mass_ci_low=0.0 if report_profile == "rank_only" else None,
            covered_mass_ci_high=0.0 if report_profile == "rank_only" else None,
            ci_confidence_level=1.0,
        )
        if report_profile == "rank_only":
            return build_rank_only_result(
                batch=batch,
                analysis=summary.analysis,
                ranking=ranking_from_analysis(summary.analysis),
                diagnostics=diagnostics,
                answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
                support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
                answer_mass_reference=self.inference_mode,
                answer_mass_reference_total=1.0,
            )
        result = build_window_result(
            batch=batch,
            analysis=summary.analysis,
            diagnostics=diagnostics,
            discovered_paths=summary.discovered_paths,
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            answer_mass_reference=self.inference_mode,
            support_mass_reference=self.inference_mode,
            answer_mass_reference_total=1.0,
            support_answer_upper_bounds=support_answer_upper_bounds,
            include_answer_support=include_answer_support,
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

    def evaluate_graph(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        report_profile: str,
        include_answer_support: bool = True,
    ) -> SupportWindowResult:
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
        return self._build_graph_result(
            batch=batch,
            summary=summary,
            report_profile=report_profile,
            include_answer_support=include_answer_support,
        )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        report_profile: str,
        include_answer_support: bool = True,
    ) -> list[SupportWindowResult]:
        start_distribution = policy.compute_root_action_distribution(prepared_batch)
        results: list[SupportWindowResult] = []
        for graph_idx in range(batch.num_graphs):
            graph_batch = batch.select_graph(graph_idx, validate=False)
            summary = run_flow_frontier_search(
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
                max_steps=self.max_steps,
                eval_cfg=self.eval_cfg,
                start_distribution=start_distribution,
                graph_idx=graph_idx,
            )
            results.append(
                self._build_graph_result(
                    batch=graph_batch,
                    summary=summary,
                    report_profile=report_profile,
                    include_answer_support=include_answer_support,
                )
            )
        return results


_LOG_ZERO = float("-inf")


@dataclass(frozen=True)
class MonteCarloRolloutSummary:
    """Monte Carlo rollouts grouped by graph and rollout index.

    `total_rollouts` is the rollout budget per graph. Legacy single-graph callers may
    still provide 1D/2D tensors; the helpers below normalize those shapes.
    """

    start_nodes: torch.Tensor
    terminal_nodes: torch.Tensor
    trace_edge_ids: torch.Tensor
    terminal_num_steps: torch.Tensor
    total_rollouts: int


def _resolve_rollout_summary_shapes(
    rollout_summary: MonteCarloRolloutSummary,
) -> tuple[int, int]:
    if rollout_summary.start_nodes.dim() == 1:
        num_graphs = 1
        num_rollouts = int(rollout_summary.start_nodes.numel())
    elif rollout_summary.start_nodes.dim() == 2:
        num_graphs = int(rollout_summary.start_nodes.size(0))
        num_rollouts = int(rollout_summary.start_nodes.size(1))
    else:
        raise ValueError(
            "MonteCarloRolloutSummary.start_nodes must be 1D or 2D. "
            f"Got shape={tuple(rollout_summary.start_nodes.shape)}."
        )
    if rollout_summary.terminal_nodes.shape != rollout_summary.start_nodes.shape:
        raise ValueError(
            "MonteCarloRolloutSummary.terminal_nodes must match start_nodes shape. "
            f"start_nodes={tuple(rollout_summary.start_nodes.shape)} "
            f"terminal_nodes={tuple(rollout_summary.terminal_nodes.shape)}."
        )
    if rollout_summary.terminal_num_steps.shape != rollout_summary.start_nodes.shape:
        raise ValueError(
            "MonteCarloRolloutSummary.terminal_num_steps must match start_nodes shape. "
            f"start_nodes={tuple(rollout_summary.start_nodes.shape)} "
            f"terminal_num_steps={tuple(rollout_summary.terminal_num_steps.shape)}."
        )
    if rollout_summary.trace_edge_ids.dim() == 2:
        if num_graphs != 1:
            raise ValueError(
                "2D trace_edge_ids are only valid for single-graph rollout summaries."
            )
        if int(rollout_summary.trace_edge_ids.size(0)) != num_rollouts:
            raise ValueError(
                "MonteCarloRolloutSummary.trace_edge_ids must align with rollouts. "
                f"num_rollouts={num_rollouts} "
                f"trace_edge_ids={tuple(rollout_summary.trace_edge_ids.shape)}."
            )
    elif rollout_summary.trace_edge_ids.dim() == 3:
        if tuple(rollout_summary.trace_edge_ids.shape[:2]) != (
            num_graphs,
            num_rollouts,
        ):
            raise ValueError(
                "MonteCarloRolloutSummary.trace_edge_ids leading dims must align with "
                "start_nodes. "
                f"start_nodes={tuple(rollout_summary.start_nodes.shape)} "
                f"trace_edge_ids={tuple(rollout_summary.trace_edge_ids.shape)}."
            )
    else:
        raise ValueError(
            "MonteCarloRolloutSummary.trace_edge_ids must be 2D or 3D. "
            f"Got shape={tuple(rollout_summary.trace_edge_ids.shape)}."
        )
    return num_graphs, num_rollouts


def _normalize_single_graph_rollout_summary(
    rollout_summary: MonteCarloRolloutSummary,
) -> MonteCarloRolloutSummary:
    num_graphs, _ = _resolve_rollout_summary_shapes(rollout_summary)
    if num_graphs != 1:
        raise ValueError(
            "Single-graph Monte Carlo helpers require exactly one graph in the rollout "
            f"summary. Got num_graphs={num_graphs}."
        )
    if rollout_summary.start_nodes.dim() == 1:
        return rollout_summary
    return MonteCarloRolloutSummary(
        start_nodes=rollout_summary.start_nodes[0],
        terminal_nodes=rollout_summary.terminal_nodes[0],
        trace_edge_ids=(
            rollout_summary.trace_edge_ids[0]
            if rollout_summary.trace_edge_ids.dim() == 3
            else rollout_summary.trace_edge_ids
        ),
        terminal_num_steps=rollout_summary.terminal_num_steps[0],
        total_rollouts=int(rollout_summary.total_rollouts),
    )


def _select_graph_rollout_summary(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
    graph_idx: int,
) -> MonteCarloRolloutSummary:
    num_graphs, _ = _resolve_rollout_summary_shapes(rollout_summary)
    if graph_idx < 0 or graph_idx >= num_graphs:
        raise IndexError(f"graph_idx out of range for rollout summary: {graph_idx}.")
    if num_graphs == 1:
        normalized = _normalize_single_graph_rollout_summary(rollout_summary)
        if batch.num_graphs == 1:
            return normalized
    edge_counts = torch.bincount(batch.edge_batch, minlength=batch.num_graphs)
    edge_offsets = edge_counts.cumsum(0) - edge_counts
    node_offset = batch.node_ptr[graph_idx]
    edge_offset = edge_offsets[graph_idx]
    graph_trace_edge_ids = (
        rollout_summary.trace_edge_ids[graph_idx]
        if rollout_summary.trace_edge_ids.dim() == 3
        else rollout_summary.trace_edge_ids
    )
    local_trace_edge_ids = torch.where(
        graph_trace_edge_ids >= 0,
        graph_trace_edge_ids - edge_offset,
        torch.full_like(graph_trace_edge_ids, fill_value=-1),
    )
    return MonteCarloRolloutSummary(
        start_nodes=rollout_summary.start_nodes[graph_idx] - node_offset,
        terminal_nodes=rollout_summary.terminal_nodes[graph_idx] - node_offset,
        trace_edge_ids=local_trace_edge_ids,
        terminal_num_steps=rollout_summary.terminal_num_steps[graph_idx],
        total_rollouts=int(rollout_summary.total_rollouts),
    )


def _split_batched_rollout_summary(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
) -> list[MonteCarloRolloutSummary]:
    num_graphs, _ = _resolve_rollout_summary_shapes(rollout_summary)
    if num_graphs != batch.num_graphs:
        raise ValueError(
            "Monte Carlo rollout summary must align with TrajectoryBatch graph count. "
            f"summary_graphs={num_graphs} batch_graphs={batch.num_graphs}."
        )
    return [
        _select_graph_rollout_summary(
            batch=batch,
            rollout_summary=rollout_summary,
            graph_idx=graph_idx,
        )
        for graph_idx in range(batch.num_graphs)
    ]


def _normal_quantile(confidence: float) -> float:
    probability = 0.5 + (float(confidence) / 2.0)
    erf_arg = torch.tensor((2.0 * probability) - 1.0, dtype=torch.float64)
    return float(math.sqrt(2.0) * torch.erfinv(erf_arg).item())


def _wilson_interval_tensor(
    *, counts: torch.Tensor, total: int, confidence: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if total < 1:
        zeros = torch.zeros_like(counts, dtype=torch.float32)
        ones = torch.ones_like(counts, dtype=torch.float32)
        return zeros, ones
    z_score = _normal_quantile(confidence)
    total_tensor = torch.tensor(float(total), device=counts.device, dtype=torch.float32)
    counts_float = counts.to(device=counts.device, dtype=torch.float32)
    phat = counts_float / total_tensor
    z_squared = float(z_score * z_score)
    denom = 1.0 + (z_squared / total_tensor)
    center = (phat + (z_squared / (2.0 * total_tensor))) / denom
    margin = (
        z_score
        * torch.sqrt(
            ((phat * (1.0 - phat)) + (z_squared / (4.0 * total_tensor))) / total_tensor
        )
        / denom
    )
    return center.sub(margin).clamp_(0.0, 1.0), center.add(margin).clamp_(0.0, 1.0)


def _wilson_interval_scalar(
    *, count: int, total: int, confidence: float
) -> tuple[float, float]:
    low, high = _wilson_interval_tensor(
        counts=torch.tensor([int(count)], dtype=torch.float32),
        total=int(total),
        confidence=float(confidence),
    )
    return float(low.item()), float(high.item())


def _sample_start_nodes_from_distribution(
    *, distribution, num_rollouts: int
) -> torch.Tensor:
    num_graphs = int(distribution.graph_log_z.numel())
    sampled_nodes: list[torch.Tensor] = []
    for _ in range(int(num_rollouts)):
        selected_positions, _, has_values = sample_segmented_one_1d(
            logits=distribution.log_probs,
            segment_ids=distribution.candidate_graph_ids,
            num_segments=num_graphs,
            temperature=1.0,
        )
        if not bool(has_values.all().item()):
            raise ValueError("Each graph must expose at least one start candidate.")
        sampled_nodes.append(
            distribution.candidate_nodes_abs.index_select(0, selected_positions)
        )
    return torch.stack(sampled_nodes, dim=1)


def rollout_search_policy(
    *,
    batch: TrajectoryBatch,
    policy: SearchPolicyProtocol,
    prepared_batch: PreparedSearchBatch,
    max_steps: int,
    num_rollouts: int,
) -> MonteCarloRolloutSummary:
    start_distribution = policy.compute_root_action_distribution(prepared_batch)
    sampled_start_nodes = _sample_start_nodes_from_distribution(
        distribution=start_distribution,
        num_rollouts=int(num_rollouts),
    )
    current_nodes = sampled_start_nodes.clone()
    done_mask = torch.zeros_like(current_nodes, dtype=torch.bool)
    absorbing_mask = torch.zeros_like(current_nodes, dtype=torch.bool)
    num_steps = torch.zeros_like(current_nodes, dtype=torch.long)
    path_token_ids = initialize_path_token_ids(
        start_nodes=sampled_start_nodes,
        max_steps=int(max_steps),
    )
    control_states = policy.build_start_control_states(
        prepared_batch,
        sampled_start_nodes,
    )
    trace_edge_ids = torch.full(
        (batch.num_graphs, int(num_rollouts), int(max_steps)),
        fill_value=-1,
        device=batch.node_ptr.device,
        dtype=torch.long,
    )
    max_actions = int(max_steps) + 1

    for step_idx in range(max_actions):
        active_mask = ~done_mask
        if not bool(active_mask.any().item()):
            break
        search_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=done_mask,
            num_steps=num_steps,
            path_token_ids=path_token_ids,
            control_state=control_states,
            absorbing_mask=absorbing_mask,
        )
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
            state=search_state,
            max_steps=int(max_steps),
        )
        has_values = step.has_values.view_as(current_nodes)
        dead_end = active_mask & (~has_values)
        chosen_positions, _, sampled_has_values = sample_segmented_one_1d(
            logits=step.move_log_probs,
            segment_ids=step.distribution.edge_agent_batch,
            num_segments=int(current_nodes.numel()),
            temperature=1.0,
        )
        flat_active = active_mask.view(-1)
        selected_mask = flat_active & sampled_has_values
        flat_current_nodes = current_nodes.view(-1)
        next_nodes = flat_current_nodes.clone()
        next_num_steps = num_steps.view(-1).clone()
        chosen_edge_ids = torch.full_like(flat_current_nodes, fill_value=-1)
        chosen_is_stop_action = torch.zeros_like(flat_current_nodes, dtype=torch.bool)
        if bool(selected_mask.any().item()):
            selected_positions = chosen_positions[selected_mask]
            chosen_edge_ids[selected_mask] = step.distribution.edge_ids.index_select(
                0, selected_positions
            )
            next_nodes[selected_mask] = step.distribution.target_nodes.index_select(
                0, selected_positions
            )
            stop_action_mask = (
                step.distribution.is_stop_action.to(dtype=torch.bool)
                if step.distribution.is_stop_action is not None
                else torch.zeros_like(step.distribution.edge_ids, dtype=torch.bool)
            )
            chosen_is_stop_action[selected_mask] = stop_action_mask.index_select(
                0, selected_positions
            )
        graph_move_mask = selected_mask & (~chosen_is_stop_action)
        stop_action_mask = selected_mask & chosen_is_stop_action
        if bool(graph_move_mask.any().item()):
            trace_edge_ids[:, :, step_idx].view(-1)[graph_move_mask.view(-1)] = (
                chosen_edge_ids[graph_move_mask]
            )
            next_num_steps[graph_move_mask] = next_num_steps[graph_move_mask] + 1
        relation_ids = torch.zeros_like(current_nodes, dtype=torch.long)
        if bool(graph_move_mask.any().item()):
            relation_ids.view(-1)[graph_move_mask] = (
                prepared_batch.topology.edge_type.index_select(
                    0,
                    chosen_edge_ids[graph_move_mask],
                )
            )
        path_token_ids = append_relation_and_node_tokens_inplace(
            path_token_ids=path_token_ids,
            num_steps=num_steps,
            relation_ids=relation_ids,
            target_nodes=next_nodes.view_as(current_nodes),
            active_mask=graph_move_mask.view_as(current_nodes),
        )
        path_token_ids = append_stop_token_inplace(
            path_token_ids=path_token_ids,
            num_steps=num_steps,
            active_mask=stop_action_mask.view_as(current_nodes),
        )
        next_control_states = control_states
        if control_states is not None and bool(graph_move_mask.any().item()):
            next_control_states = control_states.clone()
            flat_next_control_states = next_control_states.view(
                -1, int(next_control_states.size(-1))
            )
            flat_control_states = control_states.view(-1, int(control_states.size(-1)))
            flat_relation_ids = relation_ids.view(-1)
            flat_next_control_states[graph_move_mask] = (
                policy.compute_next_control_states(
                    prepared_batch,
                    control_states=flat_control_states[graph_move_mask],
                    next_nodes=next_nodes[graph_move_mask],
                    relation_ids=flat_relation_ids[graph_move_mask],
                )
            )
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)
        control_states = next_control_states
        absorbing_mask = absorbing_mask | stop_action_mask.view_as(absorbing_mask)
        done_mask = done_mask | dead_end | stop_action_mask.view_as(done_mask)

    return MonteCarloRolloutSummary(
        start_nodes=sampled_start_nodes,
        terminal_nodes=current_nodes,
        trace_edge_ids=trace_edge_ids,
        terminal_num_steps=num_steps,
        total_rollouts=int(num_rollouts),
    )


def build_monte_carlo_analysis(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
    confidence: float,
) -> ReachabilityAnalysis:
    rollout_summary = _normalize_single_graph_rollout_summary(rollout_summary)
    total_rollouts = int(rollout_summary.total_rollouts)
    terminal_counts = torch.bincount(
        rollout_summary.terminal_nodes,
        minlength=int(batch.num_nodes_total),
    ).to(device=batch.node_ptr.device, dtype=torch.float32)
    terminal_mass = terminal_counts / float(max(total_rollouts, 1))
    terminal_entities = batch.node_entity_ids.index_select(
        0, rollout_summary.terminal_nodes
    )
    answer_entity_ids, answer_counts = torch.unique(
        terminal_entities,
        sorted=True,
        return_counts=True,
    )
    answer_probs = answer_counts.to(dtype=torch.float32) / float(max(total_rollouts, 1))
    answer_ci_low, answer_ci_high = _wilson_interval_tensor(
        counts=answer_counts.to(dtype=torch.float32),
        total=total_rollouts,
        confidence=float(confidence),
    )
    gold_hits = _membership_mask(
        values=terminal_entities, candidates=batch.answer_entity_ids
    )
    gold_count = int(gold_hits.to(dtype=torch.int64).sum().item())
    gold_answer_mass = float(gold_count) / float(max(total_rollouts, 1))
    gold_ci_low, gold_ci_high = _wilson_interval_scalar(
        count=gold_count,
        total=total_rollouts,
        confidence=float(confidence),
    )
    return ReachabilityAnalysis(
        terminal_mass=terminal_mass,
        answer_entity_ids=answer_entity_ids,
        answer_probs=answer_probs,
        gold_answer_mass=gold_answer_mass,
        answer_prob_ci_low=answer_ci_low,
        answer_prob_ci_high=answer_ci_high,
        gold_answer_mass_ci_low=gold_ci_low,
        gold_answer_mass_ci_high=gold_ci_high,
        ci_confidence_level=float(confidence),
    )


def build_discovered_trajectories(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
) -> tuple[list[DiscoveredTrajectory], dict[tuple[int, int, tuple[int, ...]], int]]:
    rollout_summary = _normalize_single_graph_rollout_summary(rollout_summary)
    if rollout_summary.total_rollouts < 1:
        return [], {}
    path_matrix = torch.cat(
        [
            rollout_summary.start_nodes.unsqueeze(1),
            rollout_summary.trace_edge_ids,
            rollout_summary.terminal_nodes.unsqueeze(1),
        ],
        dim=1,
    )
    unique_paths, counts = torch.unique(path_matrix, return_counts=True, dim=0)
    gold_answers = graph_gold_answers(batch=batch)
    discovered_paths: list[DiscoveredTrajectory] = []
    path_counts: dict[tuple[int, int, tuple[int, ...]], int] = {}
    for row, count in zip(unique_paths, counts):
        start_node = int(row[0].item())
        terminal_node = int(row[-1].item())
        edge_ids = tuple(
            int(edge_id) for edge_id in row[1:-1].tolist() if int(edge_id) >= 0
        )
        answer_id = int(batch.node_entity_ids[terminal_node].item())
        probability = float(int(count.item())) / float(rollout_summary.total_rollouts)
        discovered_paths.append(
            DiscoveredTrajectory(
                start_node=start_node,
                terminal_node=terminal_node,
                answer_entity_id=answer_id,
                edge_ids=edge_ids,
                log_prob=(
                    _LOG_ZERO if probability <= 0.0 else float(math.log(probability))
                ),
                is_gold=answer_id in gold_answers,
            )
        )
        path_counts[
            (int(batch.node_entity_ids[start_node].item()), answer_id, edge_ids)
        ] = int(count.item())
    discovered_paths.sort(
        key=lambda item: (
            -item.prob,
            item.answer_entity_id,
            item.edge_ids,
            item.start_node,
        )
    )
    return discovered_paths, path_counts


def build_monte_carlo_edge_support_analysis(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
) -> EdgeSupportAnalysis:
    rollout_summary = _normalize_single_graph_rollout_summary(rollout_summary)
    num_edges = int(batch.edge_index.size(1))
    if rollout_summary.total_rollouts < 1:
        zeros = torch.zeros(
            (num_edges,), device=batch.node_ptr.device, dtype=torch.float32
        )
        return EdgeSupportAnalysis(
            edge_success_mass=zeros,
            edge_conditional_success_prob=zeros,
            success_rollout_mass=0.0,
        )
    success_mask = _membership_mask(
        values=rollout_summary.terminal_nodes,
        candidates=batch.a_local_indices.to(
            device=rollout_summary.terminal_nodes.device, dtype=torch.long
        ),
    )
    gold_count = int(success_mask.to(dtype=torch.int64).sum().item())
    if gold_count < 1:
        zeros = torch.zeros(
            (num_edges,), device=batch.node_ptr.device, dtype=torch.float32
        )
        return EdgeSupportAnalysis(
            edge_success_mass=zeros,
            edge_conditional_success_prob=zeros,
            success_rollout_mass=0.0,
        )
    success_trace_edge_ids = rollout_summary.trace_edge_ids[success_mask]
    valid_edge_mask = success_trace_edge_ids >= 0
    if not bool(valid_edge_mask.any().item()):
        zeros = torch.zeros(
            (num_edges,), device=batch.node_ptr.device, dtype=torch.float32
        )
        return EdgeSupportAnalysis(
            edge_success_mass=zeros,
            edge_conditional_success_prob=zeros,
            success_rollout_mass=float(gold_count)
            / float(rollout_summary.total_rollouts),
        )
    success_rollout_ids = (
        torch.arange(
            int(success_trace_edge_ids.size(0)),
            device=success_trace_edge_ids.device,
            dtype=torch.long,
        )
        .unsqueeze(1)
        .expand_as(success_trace_edge_ids)
    )
    pair_keys = (
        success_rollout_ids[valid_edge_mask] * max(num_edges, 1)
        + success_trace_edge_ids[valid_edge_mask]
    )
    unique_pair_keys = torch.unique(pair_keys, sorted=False)
    edge_counts = torch.bincount(
        torch.remainder(unique_pair_keys, max(num_edges, 1)),
        minlength=max(num_edges, 1),
    )[:num_edges].to(device=batch.node_ptr.device, dtype=torch.float32)
    edge_success_mass = edge_counts / float(rollout_summary.total_rollouts)
    edge_conditional_success_prob = edge_counts / float(gold_count)
    return EdgeSupportAnalysis(
        edge_success_mass=edge_success_mass,
        edge_conditional_success_prob=edge_conditional_success_prob,
        success_rollout_mass=float(gold_count) / float(rollout_summary.total_rollouts),
    )


def build_batched_monte_carlo_edge_support_analyses(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
) -> list[EdgeSupportAnalysis]:
    graph_rollout_summaries = _split_batched_rollout_summary(
        batch=batch,
        rollout_summary=rollout_summary,
    )
    analyses: list[EdgeSupportAnalysis] = []
    for graph_idx, graph_rollout_summary in enumerate(graph_rollout_summaries):
        graph_batch = batch.select_graph(graph_idx, validate=False)
        analyses.append(
            build_monte_carlo_edge_support_analysis(
                batch=graph_batch,
                rollout_summary=graph_rollout_summary,
            )
        )
    return analyses


def build_batched_monte_carlo_analyses(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
    confidence: float,
) -> list[ReachabilityAnalysis]:
    graph_rollout_summaries = _split_batched_rollout_summary(
        batch=batch,
        rollout_summary=rollout_summary,
    )
    analyses: list[ReachabilityAnalysis] = []
    for graph_idx, graph_rollout_summary in enumerate(graph_rollout_summaries):
        graph_batch = batch.select_graph(graph_idx, validate=False)
        analyses.append(
            build_monte_carlo_analysis(
                batch=graph_batch,
                rollout_summary=graph_rollout_summary,
                confidence=confidence,
            )
        )
    return analyses


def _answer_upper_bounds(
    analysis: ReachabilityAnalysis,
) -> dict[int, float]:
    upper_bounds = (
        analysis.answer_prob_ci_high
        if analysis.answer_prob_ci_high is not None
        else analysis.answer_probs
    )
    return {
        int(answer_id): float(upper)
        for answer_id, upper in zip(
            analysis.answer_entity_ids.tolist(),
            upper_bounds.tolist(),
        )
    }


def _build_support_window_result_from_summary(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
    analysis: ReachabilityAnalysis,
    eval_cfg: SearchEvalConfig,
    include_answer_support: bool,
) -> SupportWindowResult:
    discovered_paths, path_counts = build_discovered_trajectories(
        batch=batch,
        rollout_summary=rollout_summary,
    )
    diagnostics = SearchDiagnostics(
        inference_mode="monte_carlo",
        probe_count=int(rollout_summary.total_rollouts),
        remaining_mass_upper=1.0,
        stop_reason="monte_carlo_budget_exhausted",
        coverage_certified=False,
        ci_confidence_level=float(eval_cfg.monte_carlo.confidence),
    )
    result = build_window_result(
        batch=batch,
        analysis=analysis,
        diagnostics=diagnostics,
        discovered_paths=discovered_paths,
        answer_mass_threshold=float(eval_cfg.answer_mass_threshold),
        support_mass_threshold=float(eval_cfg.support_mass_threshold),
        support_path_overlap_penalty=float(eval_cfg.support_path_overlap_penalty),
        answer_mass_reference="monte_carlo",
        support_mass_reference="monte_carlo",
        answer_mass_reference_total=1.0,
        support_answer_upper_bounds=_answer_upper_bounds(analysis),
        include_answer_support=include_answer_support,
    )
    selected_count = 0
    for trajectory in result.trajectories:
        edge_ids = tuple(int(edge.edge_id) for edge in trajectory.edges)
        if trajectory.start_entity_id is None:
            continue
        selected_count += path_counts.get(
            (
                int(trajectory.start_entity_id),
                int(trajectory.terminal_entity_id),
                edge_ids,
            ),
            0,
        )
    covered_ci_low, covered_ci_high = _wilson_interval_scalar(
        count=selected_count,
        total=rollout_summary.total_rollouts,
        confidence=float(eval_cfg.monte_carlo.confidence),
    )
    return replace(
        result,
        covered_mass_ci_low=covered_ci_low,
        covered_mass_ci_high=covered_ci_high,
        gold_answer_mass_ci_low=analysis.gold_answer_mass_ci_low,
        gold_answer_mass_ci_high=analysis.gold_answer_mass_ci_high,
        ci_confidence_level=float(eval_cfg.monte_carlo.confidence),
        remaining_mass_upper=max(1.0 - covered_ci_low, 0.0),
    )


def build_batched_monte_carlo_window_results(
    *,
    batch: TrajectoryBatch,
    rollout_summary: MonteCarloRolloutSummary,
    eval_cfg: SearchEvalConfig,
    include_answer_support: bool,
) -> list[SupportWindowResult]:
    graph_rollout_summaries = _split_batched_rollout_summary(
        batch=batch,
        rollout_summary=rollout_summary,
    )
    analyses = build_batched_monte_carlo_analyses(
        batch=batch,
        rollout_summary=rollout_summary,
        confidence=float(eval_cfg.monte_carlo.confidence),
    )
    return [
        _build_support_window_result_from_summary(
            batch=batch.select_graph(graph_idx, validate=False),
            rollout_summary=graph_rollout_summary,
            analysis=analysis,
            eval_cfg=eval_cfg,
            include_answer_support=include_answer_support,
        )
        for graph_idx, (graph_rollout_summary, analysis) in enumerate(
            zip(graph_rollout_summaries, analyses)
        )
    ]


class MonteCarloBackend(ReachabilityBackendProtocol):
    inference_mode = "monte_carlo"

    def __init__(self, *, max_steps: int, eval_cfg: SearchEvalConfig) -> None:
        self.max_steps = int(max_steps)
        self.eval_cfg = eval_cfg

    def analyze_edge_support(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> EdgeSupportAnalysis:
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo.rollouts),
        )
        return build_monte_carlo_edge_support_analysis(
            batch=batch,
            rollout_summary=rollout_summary,
        )

    def analyze_edge_support_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> list[EdgeSupportAnalysis]:
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo.rollouts),
        )
        return build_batched_monte_carlo_edge_support_analyses(
            batch=batch,
            rollout_summary=rollout_summary,
        )

    def evaluate_graph(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        report_profile: str,
        include_answer_support: bool,
    ) -> SupportWindowResult:
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo.rollouts),
        )
        analysis = build_monte_carlo_analysis(
            batch=batch,
            rollout_summary=rollout_summary,
            confidence=float(self.eval_cfg.monte_carlo.confidence),
        )
        if report_profile == "rank_only":
            return build_rank_only_result(
                batch=batch,
                analysis=analysis,
                ranking=ranking_from_analysis(analysis),
                diagnostics=SearchDiagnostics(
                    inference_mode=self.inference_mode,
                    probe_count=int(rollout_summary.total_rollouts),
                    remaining_mass_upper=1.0,
                    stop_reason="rank_only_monte_carlo",
                    coverage_certified=False,
                    ci_confidence_level=float(self.eval_cfg.monte_carlo.confidence),
                ),
                answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
                support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
                answer_mass_reference=self.inference_mode,
                answer_mass_reference_total=1.0,
            )
        return _build_support_window_result_from_summary(
            batch=batch,
            analysis=analysis,
            rollout_summary=rollout_summary,
            eval_cfg=self.eval_cfg,
            include_answer_support=include_answer_support,
        )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        report_profile: str,
        include_answer_support: bool,
    ) -> list[SupportWindowResult]:
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo.rollouts),
        )
        if report_profile == "rank_only":
            analyses = build_batched_monte_carlo_analyses(
                batch=batch,
                rollout_summary=rollout_summary,
                confidence=float(self.eval_cfg.monte_carlo.confidence),
            )
            return [
                build_rank_only_result(
                    batch=batch.select_graph(graph_idx, validate=False),
                    analysis=analysis,
                    ranking=ranking_from_analysis(analysis),
                    diagnostics=SearchDiagnostics(
                        inference_mode=self.inference_mode,
                        probe_count=int(rollout_summary.total_rollouts),
                        remaining_mass_upper=1.0,
                        stop_reason="rank_only_monte_carlo",
                        coverage_certified=False,
                        ci_confidence_level=float(self.eval_cfg.monte_carlo.confidence),
                    ),
                    answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
                    support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
                    answer_mass_reference=self.inference_mode,
                    answer_mass_reference_total=1.0,
                )
                for graph_idx, analysis in enumerate(analyses)
            ]
        return build_batched_monte_carlo_window_results(
            batch=batch,
            rollout_summary=rollout_summary,
            eval_cfg=self.eval_cfg,
            include_answer_support=include_answer_support,
        )
