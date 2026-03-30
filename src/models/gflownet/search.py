from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.graph import TrajectoryBatch

from .policy import SubgraphPolicy
from .prepared_batch import SubgraphPreparedBatch
from .state import SubgraphRolloutBatch, SubgraphState


@dataclass(frozen=True)
class SubgraphTerminalSubgraph:
    edge_ids: tuple[int, ...]
    log_mass: float
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    answer_count: int


@dataclass(frozen=True)
class SubgraphBeamSearchResult:
    terminal_subgraphs: tuple[SubgraphTerminalSubgraph, ...]
    frontier_state_count: int
    frontier_answering_state_count: int


def _single_state_rollout_batch(
    *,
    prepared_batch: SubgraphPreparedBatch,
    state: SubgraphState,
) -> SubgraphRolloutBatch:
    return SubgraphRolloutBatch(
        graph_ids=torch.zeros((1,), device=prepared_batch.device, dtype=torch.long),
        states=(state,),
        done_mask=torch.zeros((1,), device=prepared_batch.device, dtype=torch.bool),
        view_shape=(1, 1),
    )


def beam_search_subgraphs(
    *,
    batch: TrajectoryBatch | None = None,
    policy: SubgraphPolicy,
    eval_cfg: dict[str, Any],
    prepared_batch: SubgraphPreparedBatch | None = None,
) -> SubgraphBeamSearchResult:
    if prepared_batch is None:
        if batch is None:
            raise ValueError("beam_search_subgraphs requires batch or prepared_batch.")
        prepared_batch = policy.prepare_batch(batch)
    if prepared_batch.num_graphs != 1:
        raise ValueError(
            "beam_search_subgraphs expects a single-graph TrajectoryBatch."
        )
    max_frontier = int(eval_cfg["flow_frontier"]["max_frontier_size"])
    max_expansions = int(eval_cfg["flow_frontier"]["max_expansions"])
    frontier: dict[tuple[int, ...], tuple[SubgraphState, float]] = {
        (): (policy.initial_state(), 0.0)
    }
    terminal_states: dict[tuple[int, ...], SubgraphTerminalSubgraph] = {}
    expansions = 0
    while frontier and expansions < max_expansions:
        next_frontier: dict[tuple[int, ...], tuple[SubgraphState, float]] = {}
        ordered_frontier = sorted(
            frontier.items(), key=lambda item: -float(item[1][1])
        )[:max_frontier]
        frontier.clear()
        for _, (state, log_mass) in ordered_frontier:
            rollout_batch = _single_state_rollout_batch(
                prepared_batch=prepared_batch, state=state
            )
            analyses = policy.analyze_rollout_batch(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
            )
            distribution = policy.compute_action_distribution(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                analyses=analyses,
            )
            target_log_probs = policy.compute_target_log_probs(distribution)
            if int(distribution.logits.numel()) == 0:
                continue
            valid_positions = torch.nonzero(
                distribution.segment_ids == 0, as_tuple=False
            ).view(-1)
            top_positions = sorted(
                [int(pos) for pos in valid_positions.detach().cpu().tolist()],
                key=lambda pos: float(target_log_probs[pos].item()),
                reverse=True,
            )[:max_frontier]
            analysis = analyses[0]
            for action_pos in top_positions:
                action_log_prob = float(target_log_probs[action_pos].item())
                action = distribution.actions[action_pos]
                if action.is_stop:
                    key = state.key()
                    best = terminal_states.get(key)
                    total_log_mass = float(log_mass + action_log_prob)
                    answer_count, _ = policy.count_gold_answers(
                        prepared_batch=prepared_batch,
                        graph_idx=0,
                        analysis=analysis,
                    )
                    terminal_payload = SubgraphTerminalSubgraph(
                        edge_ids=key,
                        log_mass=total_log_mass,
                        selected_node_ids=analysis.selected_node_ids,
                        reachability_bits=dict(analysis.reachability_bits),
                        answer_count=answer_count,
                    )
                    if best is None:
                        terminal_states[key] = terminal_payload
                    else:
                        terminal_states[key] = SubgraphTerminalSubgraph(
                            edge_ids=best.edge_ids,
                            log_mass=float(
                                torch.logaddexp(
                                    torch.tensor(best.log_mass),
                                    torch.tensor(total_log_mass),
                                ).item()
                            ),
                            selected_node_ids=best.selected_node_ids,
                            reachability_bits=best.reachability_bits,
                            answer_count=max(int(best.answer_count), int(answer_count)),
                        )
                    continue
                if action.edge_id is None:
                    raise RuntimeError("Expand actions must carry an edge_id.")
                edge_id = int(action.edge_id)
                next_state = state.with_edge(edge_id)
                key = next_state.key()
                total_log_mass = float(log_mass + action_log_prob)
                existing = next_frontier.get(key)
                if existing is None:
                    next_frontier[key] = (next_state, total_log_mass)
                else:
                    next_frontier[key] = (
                        existing[0],
                        float(
                            torch.logaddexp(
                                torch.tensor(existing[1]),
                                torch.tensor(total_log_mass),
                            ).item()
                        ),
                    )
                expansions += 1
                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break
        frontier = dict(
            sorted(next_frontier.items(), key=lambda item: -float(item[1][1]))[
                :max_frontier
            ]
        )
    frontier_answering_state_count = 0
    for _, (state, _) in frontier.items():
        analysis = policy.analyze_state(
            prepared_batch=prepared_batch,
            graph_idx=0,
            state=state,
        )
        answer_count, _ = policy.count_gold_answers(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=analysis,
        )
        if int(answer_count) > 0:
            frontier_answering_state_count += 1
    terminal_subgraphs = tuple(
        sorted(terminal_states.values(), key=lambda item: -float(item.log_mass))
    )
    return SubgraphBeamSearchResult(
        terminal_subgraphs=terminal_subgraphs,
        frontier_state_count=int(len(frontier)),
        frontier_answering_state_count=int(frontier_answering_state_count),
    )


__all__ = [
    "SubgraphBeamSearchResult",
    "SubgraphTerminalSubgraph",
    "beam_search_subgraphs",
]
