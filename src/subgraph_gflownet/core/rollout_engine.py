from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .cuda_memory import profile_cuda_memory
from .policy import SubgraphPolicy
from .reward import SubgraphTerminalReward
from .rollout_actions import sample_state_action, teacher_forced_state_action
from .rollout_context import (
    build_active_state_context,
    resolve_rollout_terminal_analyses,
)
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch, SubgraphState
from .subgraph_batch import SubgraphBatch


@dataclass(frozen=True)
class SubgraphTrajectorySampleBatch:
    state_log_flows: torch.Tensor
    log_pf_actions: torch.Tensor
    log_pb_actions: torch.Tensor
    log_reward_actions: torch.Tensor
    action_mask: torch.Tensor
    termination_action_steps: torch.Tensor
    chosen_edge_ids: torch.Tensor
    stop_actions: torch.Tensor
    terminal_answer_candidate_counts: torch.Tensor
    terminal_gold_answer_counts: torch.Tensor
    terminal_hit_mask: torch.Tensor
    terminal_component_counts: torch.Tensor
    terminal_edge_ids: tuple[tuple[int, ...], ...]
    terminal_node_ids: tuple[tuple[int, ...], ...]
    terminal_reachability_bits: tuple[dict[int, int], ...]
    sample_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    num_graphs: int
    num_rollouts: int
    terminal_answer_set_entity_ids: tuple[tuple[int, ...], ...] = field(
        default_factory=tuple
    )
    chosen_source_graph_nodes: torch.Tensor | None = None
    state_component_counts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected_shape = (self.num_graphs, self.num_rollouts)
        if tuple(self.termination_action_steps.shape) != expected_shape:
            raise ValueError("termination_action_steps shape mismatch.")
        if tuple(self.terminal_answer_candidate_counts.shape) != expected_shape:
            raise ValueError("terminal_answer_candidate_counts shape mismatch.")
        if tuple(self.terminal_gold_answer_counts.shape) != expected_shape:
            raise ValueError("terminal_gold_answer_counts shape mismatch.")
        if tuple(self.terminal_hit_mask.shape) != expected_shape:
            raise ValueError("terminal_hit_mask shape mismatch.")
        if tuple(self.terminal_component_counts.shape) != expected_shape:
            raise ValueError("terminal_component_counts shape mismatch.")
        if self.chosen_source_graph_nodes is not None and tuple(
            self.chosen_source_graph_nodes.shape
        ) != tuple(self.chosen_edge_ids.shape):
            raise ValueError("chosen_source_graph_nodes shape mismatch.")
        if self.state_component_counts is not None and tuple(
            self.state_component_counts.shape
        ) != tuple(self.state_log_flows.shape):
            raise ValueError("state_component_counts shape mismatch.")
        expected_flat = int(self.num_graphs) * int(self.num_rollouts)
        if len(self.terminal_edge_ids) != expected_flat:
            raise ValueError("terminal_edge_ids length mismatch.")
        if len(self.terminal_node_ids) != expected_flat:
            raise ValueError("terminal_node_ids length mismatch.")
        if len(self.terminal_reachability_bits) != expected_flat:
            raise ValueError("terminal_reachability_bits length mismatch.")
        if len(self.terminal_answer_set_entity_ids) != expected_flat:
            raise ValueError("terminal_answer_set_entity_ids length mismatch.")

    @property
    def success_rate(self) -> torch.Tensor:
        return self.terminal_hit_mask.to(dtype=torch.float32).mean()


def _lookup_backward_log_prob(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    graph_idx: int,
    state: SubgraphState,
    backward_log_prob_cache: dict[tuple[int, tuple[Any, ...]], float],
) -> float:
    cache_key = (int(graph_idx), state.key())
    cached = backward_log_prob_cache.get(cache_key)
    if cached is None:
        cached = float(
            policy.compute_backward_log_prob(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                state=state,
            )
        )
        backward_log_prob_cache[cache_key] = cached
    return float(cached)


def _lookup_terminal_reward(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis: SubgraphAnalysis,
    terminal_reward_cache: dict[tuple[int, tuple[Any, ...]], SubgraphTerminalReward],
) -> SubgraphTerminalReward:
    cache_key = (int(graph_idx), state.key())
    cached = terminal_reward_cache.get(cache_key)
    if cached is None:
        cached = policy.compute_terminal_reward(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            analysis=analysis,
        )
        terminal_reward_cache[cache_key] = cached
    return cached


class HierarchicalRolloutEngine:
    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = int(max_steps)

    def sample(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphBatch,
        rollouts_per_graph: int,
        temperature: float,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphTrajectorySampleBatch:
        return self._run_rollout(
            policy=policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=int(rollouts_per_graph),
            temperature=float(temperature),
            action_pruning=action_pruning,
            edge_sequences=None,
            profile_prefix="sampler.sample",
        )

    def teacher_force(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphBatch,
        edge_sequences: tuple[tuple[int, ...], ...],
    ) -> SubgraphTrajectorySampleBatch:
        return self._run_rollout(
            policy=policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=1,
            temperature=1.0,
            action_pruning=None,
            edge_sequences=edge_sequences,
            profile_prefix="sampler.teacher_force",
        )

    def _run_rollout(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphBatch,
        rollouts_per_graph: int,
        temperature: float,
        action_pruning: Mapping[str, Any] | None,
        edge_sequences: tuple[tuple[int, ...], ...] | None,
        profile_prefix: str,
    ) -> SubgraphTrajectorySampleBatch:
        num_graphs = int(prepared_batch.num_graphs)
        num_rollouts = int(rollouts_per_graph)
        if edge_sequences is not None and len(edge_sequences) != int(num_graphs):
            raise ValueError(
                "edge_sequences must align with prepared_batch.num_graphs."
            )
        flat_states = num_graphs * num_rollouts
        max_actions = self.max_steps + 1
        analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis] = {}
        backward_log_prob_cache: dict[tuple[int, tuple[Any, ...]], float] = {}
        terminal_reward_cache: dict[
            tuple[int, tuple[Any, ...]], SubgraphTerminalReward
        ] = {}

        init_context = profile_cuda_memory(
            f"{profile_prefix}.initialize_rollout_batch",
            device=prepared_batch.device,
            extra=(
                f"num_graphs={num_graphs} num_rollouts={num_rollouts} "
                f"max_steps={self.max_steps}"
            ),
        )
        with init_context:
            rollout_batch = policy.initialize_rollout_batch(
                prepared_batch=prepared_batch,
                num_rollouts=num_rollouts,
            )

        with profile_cuda_memory(
            f"{profile_prefix}.allocate_buffers",
            device=prepared_batch.device,
            extra=(
                f"flat_states={flat_states} max_actions={max_actions} "
                f"max_steps={self.max_steps}"
            ),
        ):
            state_log_flows = torch.zeros(
                (flat_states, max_actions),
                device=prepared_batch.device,
                dtype=torch.float32,
            )
            state_component_counts = torch.zeros_like(state_log_flows)
            log_pf_actions = torch.zeros_like(state_log_flows)
            log_pb_actions = torch.zeros_like(state_log_flows)
            log_reward_actions = torch.zeros_like(state_log_flows)
            action_mask = torch.zeros_like(state_log_flows, dtype=torch.bool)
            chosen_edge_ids = torch.full(
                (flat_states, self.max_steps),
                fill_value=-1,
                device=prepared_batch.device,
                dtype=torch.long,
            )
            chosen_source_graph_nodes = torch.full(
                (flat_states, self.max_steps),
                fill_value=-1,
                device=prepared_batch.device,
                dtype=torch.long,
            )
            stop_actions = torch.zeros(
                (flat_states, max_actions),
                device=prepared_batch.device,
                dtype=torch.bool,
            )
            termination_action_steps = torch.zeros(
                (flat_states,), device=prepared_batch.device, dtype=torch.long
            )
            terminal_answer_candidate_counts = torch.zeros_like(
                termination_action_steps
            )
            terminal_gold_answer_counts = torch.zeros_like(termination_action_steps)
            terminal_hit_mask = torch.zeros_like(
                termination_action_steps, dtype=torch.bool
            )
            terminal_component_counts = torch.zeros_like(termination_action_steps)
            terminal_answer_set_entity_ids: list[tuple[int, ...]] = [
                () for _ in range(flat_states)
            ]

        for action_step in range(max_actions):
            active_state_indices = rollout_batch.active_state_indices()
            active_states = int(len(active_state_indices))
            if active_states <= 0:
                break
            stage_extra = (
                f"action_step={action_step} active_states={active_states} "
                f"num_graphs={num_graphs} num_rollouts={num_rollouts}"
            )
            active_context = build_active_state_context(
                policy=policy,
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                active_state_indices=active_state_indices,
                analysis_cache=analysis_cache,
                action_pruning=action_pruning,
                profile_prefix=profile_prefix,
                profile_extra=stage_extra,
            )
            current_log_flow = state_log_flows.new_zeros(
                (flat_states,), dtype=torch.float32
            )
            current_log_flow.index_copy_(
                0,
                active_context.unique_layout.active_state_tensor,
                active_context.active_log_flows,
            )
            state_log_flows[:, action_step] = current_log_flow
            state_component_counts[
                active_context.unique_layout.active_state_tensor, action_step
            ] = active_context.active_component_counts
            chosen_actions: list[SubgraphAction] = [SubgraphAction.stop()] * flat_states
            for state_idx in range(flat_states):
                if bool(rollout_batch.done_mask[state_idx].item()):
                    chosen_actions[state_idx] = SubgraphAction.stop()
            for local_state_idx, flat_state_idx in enumerate(
                active_context.unique_layout.active_state_indices
            ):
                state_distribution = active_context.active_state_distributions[
                    local_state_idx
                ]
                current_analysis = active_context.active_analyses[local_state_idx]
                current_state = rollout_batch.states[int(flat_state_idx)]
                graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
                if edge_sequences is None:
                    action, log_pf, _next_components = sample_state_action(
                        state_distribution=state_distribution,
                        temperature=float(temperature),
                    )
                else:
                    sequence_graph_idx = int(
                        flat_state_idx // max(int(num_rollouts), 1)
                    )
                    planned_edges = edge_sequences[sequence_graph_idx]
                    planned_edge_id = (
                        int(planned_edges[action_step])
                        if action_step < int(len(planned_edges))
                        else None
                    )
                    action, log_pf, _next_components = teacher_forced_state_action(
                        state_distribution=state_distribution,
                        planned_edge_id=planned_edge_id,
                    )
                action_mask[flat_state_idx, action_step] = True
                chosen_actions[flat_state_idx] = action
                stop_actions[flat_state_idx, action_step] = bool(action.is_stop)
                log_pf_actions[flat_state_idx, action_step] = log_pf.to(
                    dtype=torch.float32
                )
                if action.is_stop:
                    terminal_reward = _lookup_terminal_reward(
                        policy=policy,
                        prepared_batch=prepared_batch,
                        graph_idx=graph_idx,
                        state=current_state,
                        analysis=current_analysis,
                        terminal_reward_cache=terminal_reward_cache,
                    )
                    log_reward_actions[flat_state_idx, action_step] = float(
                        terminal_reward.log_reward
                    )
                    termination_action_steps[flat_state_idx] = int(action_step + 1)
                    terminal_answer_candidate_counts[flat_state_idx] = int(
                        terminal_reward.answer_set.count
                    )
                    terminal_gold_answer_counts[flat_state_idx] = int(
                        terminal_reward.gold_answer_count
                    )
                    terminal_hit_mask[flat_state_idx] = bool(terminal_reward.hit)
                    terminal_component_counts[flat_state_idx] = int(
                        current_analysis.anchor_component_count
                    )
                    terminal_answer_set_entity_ids[flat_state_idx] = tuple(
                        int(entity_id) for entity_id in terminal_reward.answer_entities
                    )
                    continue
                if action.edge_id is None:
                    raise RuntimeError("Expand actions must carry edge_id.")
                next_state = current_state.with_edge(int(action.edge_id))
                chosen_edge_ids[flat_state_idx, action_step] = int(action.edge_id)
                if action.source_graph_node is not None:
                    chosen_source_graph_nodes[flat_state_idx, action_step] = int(
                        action.source_graph_node
                    )
                log_pb_actions[flat_state_idx, action_step] = _lookup_backward_log_prob(
                    policy=policy,
                    prepared_batch=prepared_batch,
                    graph_idx=graph_idx,
                    state=next_state,
                    backward_log_prob_cache=backward_log_prob_cache,
                )
            with profile_cuda_memory(
                f"{profile_prefix}.transition",
                device=prepared_batch.device,
                extra=stage_extra,
            ):
                rollout_batch = policy.transition(
                    rollout_batch=rollout_batch,
                    chosen_actions=tuple(chosen_actions),
                )
        if bool((termination_action_steps <= 0).any().item()):
            missing = torch.nonzero(termination_action_steps <= 0, as_tuple=False).view(
                -1
            )
            raise RuntimeError(
                "Subgraph rollout ended without explicit STOP for some trajectories. "
                f"missing={missing.detach().cpu().tolist()}"
            )
        with profile_cuda_memory(
            f"{profile_prefix}.terminal_analyses",
            device=prepared_batch.device,
            extra=f"num_graphs={num_graphs} num_rollouts={num_rollouts}",
        ):
            terminal_analyses = resolve_rollout_terminal_analyses(
                policy=policy,
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                analysis_cache=analysis_cache,
            )
        terminal_edge_ids = tuple(state.edge_ids for state in rollout_batch.states)
        terminal_node_ids = tuple(
            analysis.selected_node_ids for analysis in terminal_analyses
        )
        terminal_reachability_bits = tuple(
            dict(analysis.reachability_bits) for analysis in terminal_analyses
        )
        return SubgraphTrajectorySampleBatch(
            state_log_flows=state_log_flows.view(num_graphs, num_rollouts, max_actions),
            state_component_counts=state_component_counts.view(
                num_graphs, num_rollouts, max_actions
            ),
            log_pf_actions=log_pf_actions.view(num_graphs, num_rollouts, max_actions),
            log_pb_actions=log_pb_actions.view(num_graphs, num_rollouts, max_actions),
            log_reward_actions=log_reward_actions.view(
                num_graphs, num_rollouts, max_actions
            ),
            action_mask=action_mask.view(num_graphs, num_rollouts, max_actions),
            termination_action_steps=termination_action_steps.view(
                num_graphs, num_rollouts
            ),
            chosen_edge_ids=chosen_edge_ids.view(
                num_graphs, num_rollouts, self.max_steps
            ),
            chosen_source_graph_nodes=chosen_source_graph_nodes.view(
                num_graphs, num_rollouts, self.max_steps
            ),
            stop_actions=stop_actions.view(num_graphs, num_rollouts, max_actions),
            terminal_answer_candidate_counts=terminal_answer_candidate_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_gold_answer_counts=terminal_gold_answer_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_hit_mask=terminal_hit_mask.view(num_graphs, num_rollouts),
            terminal_component_counts=terminal_component_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_edge_ids=terminal_edge_ids,
            terminal_node_ids=terminal_node_ids,
            terminal_reachability_bits=terminal_reachability_bits,
            terminal_answer_set_entity_ids=tuple(terminal_answer_set_entity_ids),
            sample_ids=prepared_batch.sample_ids,
            question_ids=prepared_batch.sample_ids,
            num_graphs=num_graphs,
            num_rollouts=num_rollouts,
        )


__all__ = ["HierarchicalRolloutEngine", "SubgraphTrajectorySampleBatch"]
