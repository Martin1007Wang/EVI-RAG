from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.policy.output import ForwardPolicyOutput
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.result import RolloutResult


class EdgeFirstPolicy:
    def __call__(self, *, features, state, context, frontier):
        del features, context
        terminal_log_flow = torch.full((state.num_rows,), -1.0e9, dtype=torch.float32, device=state.device)
        continue_log_flow = torch.full((state.num_rows,), -torch.inf, dtype=torch.float32, device=state.device)
        if frontier.row_ids.numel() > 0:
            continue_log_flow.index_fill_(0, frontier.row_ids, 0.0)
        terminal_log_flow = torch.where(torch.isfinite(continue_log_flow), terminal_log_flow, torch.zeros_like(terminal_log_flow))
        state_log_flow = torch.logaddexp(terminal_log_flow, continue_log_flow)
        edge_logit = torch.zeros(frontier.edge_ids.numel(), dtype=torch.float32, device=state.device)
        edge_log_prob = torch.zeros_like(edge_logit)
        edge_log_flow = continue_log_flow.index_select(0, frontier.row_ids) + edge_log_prob
        return ForwardPolicyOutput(
            frontier_row_ids=frontier.row_ids,
            frontier_edge_ids=frontier.edge_ids,
            terminal_log_flow=terminal_log_flow,
            continue_log_flow=continue_log_flow,
            state_log_flow=state_log_flow,
            edge_logit=edge_logit,
            edge_log_prob=edge_log_prob,
            edge_log_flow=edge_log_flow,
            stop_log_prob=terminal_log_flow - state_log_flow,
            expand_log_prob=continue_log_flow - state_log_flow,
            edge_action_log_prob=edge_log_flow - state_log_flow.index_select(0, frontier.row_ids),
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )


def test_budget_truncated_state_enters_terminal_batch() -> None:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    context = GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(edge_index=edge_index, num_nodes=2),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )

    _, training = RolloutEngine(expand_budget=1).sample_fused_rollouts(
        policy=EdgeFirstPolicy(),
        context=context,
        features=object(),
        rollouts_per_graph=1,
    )

    assert training is not None
    assert training.expansions.num_items == 1
    assert training.terminals.num_items == 1
    assert training.terminals.stop_reason.tolist() == [RolloutResult.BUDGET_TRUNCATED]
