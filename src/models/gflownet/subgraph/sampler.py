from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.graph import TrajectoryBatch
from src.utils.segment_ops import sample_segmented_one_1d

from .policy import SubgraphPolicy
from .prepared_batch import SubgraphPreparedBatch
from .state import SubgraphAction


@dataclass(frozen=True)
class SubgraphTrajectorySampleBatch:
    state_log_flows: torch.Tensor
    log_pf_actions: torch.Tensor
    log_reward_actions: torch.Tensor
    action_mask: torch.Tensor
    termination_action_steps: torch.Tensor
    chosen_edge_ids: torch.Tensor
    stop_actions: torch.Tensor
    terminal_answer_counts: torch.Tensor
    terminal_hit_mask: torch.Tensor
    terminal_component_counts: torch.Tensor
    terminal_edge_ids: tuple[tuple[int, ...], ...]
    terminal_node_ids: tuple[tuple[int, ...], ...]
    terminal_reachability_bits: tuple[dict[int, int], ...]
    sample_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    num_graphs: int
    num_rollouts: int

    def __post_init__(self) -> None:
        expected_shape = (self.num_graphs, self.num_rollouts)
        if tuple(self.termination_action_steps.shape) != expected_shape:
            raise ValueError("termination_action_steps shape mismatch.")
        if tuple(self.terminal_answer_counts.shape) != expected_shape:
            raise ValueError("terminal_answer_counts shape mismatch.")
        if tuple(self.terminal_hit_mask.shape) != expected_shape:
            raise ValueError("terminal_hit_mask shape mismatch.")
        if tuple(self.terminal_component_counts.shape) != expected_shape:
            raise ValueError("terminal_component_counts shape mismatch.")

    @property
    def success_rate(self) -> torch.Tensor:
        return self.terminal_hit_mask.to(dtype=torch.float32).mean()


class SubgraphSampler:
    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = int(max_steps)

    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: Any,
        prepared_batch: Any,
        rollouts_per_graph: int,
        temperature: float,
        action_prior_scale: float = 1.0,
        transition_bias_scale: float = 1.0,
    ) -> Any:
        if not isinstance(policy, SubgraphPolicy):
            raise TypeError("SubgraphSampler requires SubgraphPolicy.")
        if not isinstance(prepared_batch, SubgraphPreparedBatch):
            raise TypeError("SubgraphSampler requires SubgraphPreparedBatch inputs.")
        del batch, transition_bias_scale
        num_graphs = int(prepared_batch.num_graphs)
        num_rollouts = int(rollouts_per_graph)
        flat_states = num_graphs * num_rollouts
        max_actions = self.max_steps + 1
        rollout_batch = policy.initialize_rollout_batch(
            prepared_batch=prepared_batch,
            num_rollouts=num_rollouts,
        )
        state_log_flows = torch.zeros(
            (flat_states, max_actions),
            device=prepared_batch.device,
            dtype=torch.float32,
        )
        log_pf_actions = torch.zeros_like(state_log_flows)
        log_reward_actions = torch.zeros_like(state_log_flows)
        action_mask = torch.zeros_like(state_log_flows, dtype=torch.bool)
        chosen_edge_ids = torch.full(
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
        terminal_answer_counts = torch.zeros_like(termination_action_steps)
        terminal_hit_mask = torch.zeros_like(termination_action_steps, dtype=torch.bool)
        terminal_component_counts = torch.zeros_like(termination_action_steps)
        for action_step in range(max_actions):
            if not bool((~rollout_batch.done_mask).any().item()):
                break
            analyses = policy.analyze_rollout_batch(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
            )
            current_log_flow = policy.compute_log_flows(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                analyses=analyses,
            )
            state_log_flows[:, action_step] = current_log_flow
            distribution = policy.compute_action_distribution(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                analyses=analyses,
            )
            if int(distribution.logits.numel()) == 0:
                break
            target_log_probs = policy.compute_target_log_probs(distribution)
            proposal_logits = distribution.logits + policy.compute_proposal_bias(
                prepared_batch=prepared_batch,
                distribution=distribution,
                action_prior_scale=action_prior_scale,
            )
            chosen_positions, _, has_values = sample_segmented_one_1d(
                logits=proposal_logits,
                segment_ids=distribution.segment_ids,
                num_segments=int(distribution.flat_state_indices.numel()),
                temperature=float(temperature),
            )
            if not bool(has_values.all().item()):
                raise RuntimeError(
                    "Subgraph proposal sampling could not find a legal action for every active state."
                )
            chosen_actions: list[SubgraphAction] = [SubgraphAction.stop()] * flat_states
            for state_idx in range(flat_states):
                if bool(rollout_batch.done_mask[state_idx].item()):
                    chosen_actions[state_idx] = SubgraphAction.stop()
            for local_state_idx, flat_state_idx in enumerate(
                distribution.flat_state_indices.detach().cpu().tolist()
            ):
                action_pos = int(chosen_positions[local_state_idx].item())
                action = distribution.actions[action_pos]
                action_mask[int(flat_state_idx), action_step] = True
                chosen_actions[int(flat_state_idx)] = action
                chosen_is_stop = bool(action.is_stop)
                stop_actions[int(flat_state_idx), action_step] = chosen_is_stop
                log_pf_actions[int(flat_state_idx), action_step] = target_log_probs[
                    action_pos
                ]
                current_analysis = analyses[int(flat_state_idx)]
                current_components = int(
                    distribution.current_component_counts[action_pos].item()
                )
                if chosen_is_stop:
                    reward_value, answer_count, hit = (
                        policy.env.compute_stop_log_reward(
                            prepared_batch=prepared_batch,
                            graph_idx=int(
                                rollout_batch.graph_ids[flat_state_idx].item()
                            ),
                            analysis=current_analysis,
                        )
                    )
                    log_reward_actions[int(flat_state_idx), action_step] = float(
                        reward_value
                    )
                    termination_action_steps[int(flat_state_idx)] = int(action_step + 1)
                    terminal_answer_counts[int(flat_state_idx)] = int(answer_count)
                    terminal_hit_mask[int(flat_state_idx)] = bool(hit)
                    terminal_component_counts[int(flat_state_idx)] = int(
                        current_components
                    )
                    continue
                if action_step >= self.max_steps:
                    raise RuntimeError(
                        "Expand chosen beyond max_steps in subgraph sampler."
                    )
                if action.edge_id is None:
                    raise RuntimeError("Expand actions must carry an edge_id.")
                edge_id = int(action.edge_id)
                next_state = rollout_batch.states[int(flat_state_idx)].with_edge(
                    edge_id
                )
                next_analysis = policy.env.analyze_state(
                    prepared_batch=prepared_batch,
                    graph_idx=int(rollout_batch.graph_ids[flat_state_idx].item()),
                    state=next_state,
                )
                chosen_edge_ids[int(flat_state_idx), action_step] = edge_id
                log_reward_actions[int(flat_state_idx), action_step] = float(
                    policy.env.compute_expand_log_reward(
                        current_analysis=current_analysis,
                        next_analysis=next_analysis,
                    )
                )
            rollout_batch = policy.env.transition(
                rollout_batch=rollout_batch,
                chosen_actions=tuple(chosen_actions),
            )
        if bool((termination_action_steps <= 0).any().item()):
            missing = torch.nonzero(termination_action_steps <= 0, as_tuple=False).view(
                -1
            )
            raise RuntimeError(
                "Subgraph sampler ended without explicit STOP for some rollouts. "
                f"missing={missing.detach().cpu().tolist()}"
            )
        terminal_analyses = policy.analyze_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
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
            log_pf_actions=log_pf_actions.view(num_graphs, num_rollouts, max_actions),
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
            stop_actions=stop_actions.view(num_graphs, num_rollouts, max_actions),
            terminal_answer_counts=terminal_answer_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_hit_mask=terminal_hit_mask.view(num_graphs, num_rollouts),
            terminal_component_counts=terminal_component_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_edge_ids=terminal_edge_ids,
            terminal_node_ids=terminal_node_ids,
            terminal_reachability_bits=terminal_reachability_bits,
            sample_ids=prepared_batch.sample_ids,
            question_ids=prepared_batch.questions,
            num_graphs=num_graphs,
            num_rollouts=num_rollouts,
        )

    def rebuild_sample_batch(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError("subgraph replay rebuilding is not implemented yet.")


__all__ = [
    "SubgraphSampler",
    "SubgraphTrajectorySampleBatch",
]
