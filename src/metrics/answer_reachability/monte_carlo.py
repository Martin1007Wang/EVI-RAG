from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.configs import HorizonConfig, SearchEvalConfig
from src.models.gflownet import (
    PreparedSearchBatch,
    SearchPolicyProtocol,
    SearchState,
    compute_constrained_policy_step,
)
from src.models.gflownet.path import (
    append_relation_and_node_tokens_inplace,
    initialize_path_token_ids,
)
from src.utils.segment_ops import sample_segmented_one_1d

from .analysis import EdgeSupportAnalysis, ReachabilityAnalysis
from .posterior import DiscoveredTrajectory, build_window_result, graph_gold_answers
from .schema import SupportWindowResult


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
    start_distribution = policy.compute_start_distribution(prepared_batch)
    sampled_start_nodes = _sample_start_nodes_from_distribution(
        distribution=start_distribution,
        num_rollouts=int(num_rollouts),
    )
    current_nodes = sampled_start_nodes.clone()
    done_mask = torch.zeros_like(current_nodes, dtype=torch.bool)
    num_steps = torch.zeros_like(current_nodes, dtype=torch.long)
    path_token_ids = initialize_path_token_ids(
        start_nodes=sampled_start_nodes,
        max_steps=int(max_steps),
    )
    build_start_control_states = getattr(policy, "build_start_control_states", None)
    control_states = None
    if callable(build_start_control_states):
        control_states = build_start_control_states(
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
        chosen_is_submit = torch.zeros_like(flat_current_nodes, dtype=torch.bool)
        if bool(selected_mask.any().item()):
            selected_positions = chosen_positions[selected_mask]
            chosen_edge_ids[selected_mask] = step.distribution.edge_ids.index_select(
                0, selected_positions
            )
            next_nodes[selected_mask] = step.distribution.target_nodes.index_select(
                0, selected_positions
            )
            submit_mask = (
                step.distribution.is_submit.to(dtype=torch.bool)
                if step.distribution.is_submit is not None
                else torch.zeros_like(step.distribution.edge_ids, dtype=torch.bool)
            )
            chosen_is_submit[selected_mask] = submit_mask.index_select(
                0, selected_positions
            )
        graph_move_mask = selected_mask & (~chosen_is_submit)
        submit_mask = selected_mask & chosen_is_submit
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
        next_control_states = control_states
        compute_next_control_states = getattr(
            policy, "compute_next_control_states", None
        )
        if (
            control_states is not None
            and callable(compute_next_control_states)
            and bool(graph_move_mask.any().item())
        ):
            next_control_states = control_states.clone()
            flat_next_control_states = next_control_states.view(
                -1, int(next_control_states.size(-1))
            )
            flat_control_states = control_states.view(-1, int(control_states.size(-1)))
            flat_relation_ids = relation_ids.view(-1)
            flat_next_control_states[graph_move_mask] = compute_next_control_states(
                prepared_batch,
                control_states=flat_control_states[graph_move_mask],
                next_nodes=next_nodes[graph_move_mask],
                relation_ids=flat_relation_ids[graph_move_mask],
            )
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)
        control_states = next_control_states
        done_mask = done_mask | dead_end | submit_mask.view_as(done_mask)

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
    terminal_entities = batch.node_global_ids.index_select(
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
    gold_total_mass = float(gold_count) / float(max(total_rollouts, 1))
    gold_ci_low, gold_ci_high = _wilson_interval_scalar(
        count=gold_count,
        total=total_rollouts,
        confidence=float(confidence),
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
        answer_prob_ci_low=answer_ci_low,
        answer_prob_ci_high=answer_ci_high,
        gold_total_mass_ci_low=gold_ci_low,
        gold_total_mass_ci_high=gold_ci_high,
        ci_confidence_level=float(confidence),
        retrieval_answer_entity_ids=answer_entity_ids,
        retrieval_answer_probs=answer_probs,
        log_terminal_mass=log_terminal_mass,
        log_answer_probs=log_answer_probs,
        log_gold_total_mass=log_gold_total_mass,
        log_retrieval_answer_probs=log_answer_probs,
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
        answer_id = int(batch.node_global_ids[terminal_node].item())
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
            (int(batch.node_global_ids[start_node].item()), answer_id, edge_ids)
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
            gold_mass=0.0,
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
            gold_mass=0.0,
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
            gold_mass=float(gold_count) / float(rollout_summary.total_rollouts),
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
        gold_mass=float(gold_count) / float(rollout_summary.total_rollouts),
    )


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
    result = build_window_result(
        batch=batch,
        discovered_paths=discovered_paths,
        analysis=analysis,
        inference_mode="monte_carlo",
        answer_mass_threshold=float(eval_cfg.answer_mass_threshold),
        support_mass_threshold=float(eval_cfg.support_mass_threshold),
        support_path_overlap_penalty=float(eval_cfg.support_path_overlap_penalty),
        probe_count=int(rollout_summary.total_rollouts),
        remaining_mass_upper=1.0,
        stop_reason="monte_carlo_budget_exhausted",
        coverage_certified=False,
        answer_mass_reference="monte_carlo",
        support_mass_reference="monte_carlo",
        answer_mass_reference_total=1.0,
        support_answer_upper_bounds=_answer_upper_bounds(analysis),
        include_answer_support=include_answer_support,
        ci_confidence_level=float(eval_cfg.monte_carlo_confidence),
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
        confidence=float(eval_cfg.monte_carlo_confidence),
    )
    return replace(
        result,
        covered_mass_ci_low=covered_ci_low,
        covered_mass_ci_high=covered_ci_high,
        gold_total_mass_ci_low=analysis.gold_total_mass_ci_low,
        gold_total_mass_ci_high=analysis.gold_total_mass_ci_high,
        ci_confidence_level=float(eval_cfg.monte_carlo_confidence),
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
        confidence=float(eval_cfg.monte_carlo_confidence),
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


class MonteCarloReachabilityAnalyzer:
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
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo_rollouts),
        )
        return build_monte_carlo_analysis(
            batch=batch,
            rollout_summary=rollout_summary,
            confidence=float(self.eval_cfg.monte_carlo_confidence),
        )

    def analyze_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> list[ReachabilityAnalysis]:
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=self.max_steps,
            num_rollouts=int(self.eval_cfg.monte_carlo_rollouts),
        )
        return build_batched_monte_carlo_analyses(
            batch=batch,
            rollout_summary=rollout_summary,
            confidence=float(self.eval_cfg.monte_carlo_confidence),
        )

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
            num_rollouts=int(self.eval_cfg.monte_carlo_rollouts),
        )
        return build_monte_carlo_edge_support_analysis(
            batch=batch,
            rollout_summary=rollout_summary,
        )


class MonteCarloSupportSearch:
    requires_analysis = False

    def __init__(
        self, *, horizon_cfg: HorizonConfig, eval_cfg: SearchEvalConfig
    ) -> None:
        self.horizon_cfg = horizon_cfg
        self.eval_cfg = eval_cfg

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
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=int(self.horizon_cfg.max_steps),
            num_rollouts=int(self.eval_cfg.monte_carlo_rollouts),
        )
        mc_analysis = build_monte_carlo_analysis(
            batch=batch,
            rollout_summary=rollout_summary,
            confidence=float(self.eval_cfg.monte_carlo_confidence),
        )
        return _build_support_window_result_from_summary(
            batch=batch,
            analysis=mc_analysis,
            rollout_summary=rollout_summary,
            eval_cfg=self.eval_cfg,
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
        rollout_summary = rollout_search_policy(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            max_steps=int(self.horizon_cfg.max_steps),
            num_rollouts=int(self.eval_cfg.monte_carlo_rollouts),
        )
        return build_batched_monte_carlo_window_results(
            batch=batch,
            rollout_summary=rollout_summary,
            eval_cfg=self.eval_cfg,
            include_answer_support=include_answer_support,
        )


__all__ = [
    "build_batched_monte_carlo_analyses",
    "build_batched_monte_carlo_window_results",
    "MonteCarloReachabilityAnalyzer",
    "MonteCarloRolloutSummary",
    "MonteCarloSupportSearch",
    "build_monte_carlo_edge_support_analysis",
    "build_discovered_trajectories",
    "build_monte_carlo_analysis",
    "rollout_search_policy",
]
