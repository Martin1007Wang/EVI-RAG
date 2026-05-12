from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.graph.segments import segment_logsumexp
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.nn.frontier_builder import build_frontier
from src.weaver.nn.frontier_context import FrontierContext, frontier_semantic_scores
from src.weaver.reward import RewardModel
from src.weaver.state import RolloutState, State


@dataclass(frozen=True, slots=True)
class BudgetedOracleOutput:
    terminal_J: torch.Tensor
    V_star: torch.Tensor
    stop_prob: torch.Tensor
    edge_probs: torch.Tensor
    edge_branch_logits: torch.Tensor
    frontier_edge_ids: torch.Tensor
    frontier_batch_ids: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class BudgetedLexicographicOracle:
    """
    Exact budgeted flow teacher for small budgets.

    F*(s,0)=exp(J(s))
    F*(s,b)=exp(J(s)) + sum_e F*(s+e,b-1) P_B(s|s+e)
    V*(s,b)=log F*(s,b)
    """

    def __init__(
        self,
        *,
        reward_model: RewardModel,
        exact_budget: int = 2,
        top_m_for_budget3: int = 64,
        frontier_mode: str = "boundary",
    ) -> None:
        self.reward_model = reward_model
        self.exact_budget = int(exact_budget)
        self.top_m_for_budget3 = int(top_m_for_budget3)
        self.frontier_mode = str(frontier_mode)
        if self.exact_budget < 0:
            raise ValueError("exact_budget must be non-negative.")
        if self.top_m_for_budget3 < 1:
            raise ValueError("top_m_for_budget3 must be >= 1.")
        self._memo: dict[tuple[int, tuple[int, ...], int], tuple[float, float]] = {}
        self._coverage_seen = 0
        self._coverage_kept = 0
        self._pruned_rows = 0

    @torch.no_grad()
    def evaluate(
        self,
        *,
        batch: RetrievalBatch,
        fb: FeatureBank,
        state: State | RolloutState,
        remaining_budget: torch.Tensor,
        frontier: FrontierContext | None = None,
    ) -> BudgetedOracleOutput:
        rollout_state = _as_rollout_state(batch=batch, state=state)
        device = batch.edge_index.device
        dtype = torch.float32
        rows = int(rollout_state.num_rollouts)
        budget = remaining_budget.to(device=device, dtype=torch.long).view(-1)
        if budget.shape != (rows,):
            raise ValueError(
                f"remaining_budget must have shape [{rows}], got {tuple(budget.shape)}."
            )
        if frontier is None:
            frontier = build_frontier(
                fb=fb,
                batch=batch,
                state=rollout_state,
                frontier_mode=self.frontier_mode,
            )
        edge_ids = frontier.edge_ids.to(device=device, dtype=torch.long).view(-1)
        row_ids = frontier.graph_id.to(device=device, dtype=torch.long).view(-1)
        has_budget = budget.gt(0).index_select(0, row_ids) if row_ids.numel() else torch.zeros((0,), dtype=torch.bool, device=device)
        edge_ids = edge_ids[has_budget]
        row_ids = row_ids[has_budget]
        frontier = _filter_frontier(frontier, has_budget, device=device)

        terminal = torch.empty((rows,), dtype=dtype, device=device)
        value = torch.empty_like(terminal)
        valid = torch.ones((rows,), dtype=torch.bool, device=device)
        for row in range(rows):
            j, v = self._value_for_row(
                batch=batch,
                fb=fb,
                state=rollout_state.select_rollouts(torch.tensor([row], device=device)),
                budget=int(budget[row].item()),
            )
            terminal[row] = j
            value[row] = v

        branch = torch.full((int(edge_ids.numel()),), -torch.inf, dtype=dtype, device=device)
        if edge_ids.numel() > 0:
            for pos in range(int(edge_ids.numel())):
                edge = edge_ids[pos : pos + 1]
                row = row_ids[pos : pos + 1]
                child = _successor_state(batch=batch, state=rollout_state, edge_ids=edge, row_ids=row)
                _, child_v = self._value_for_row(
                    batch=batch,
                    fb=fb,
                    state=child,
                    budget=int(budget[int(row.item())].item()) - 1,
                )
                log_pb = _log_pb(batch=batch, successor=child, edge_ids=edge)[0]
                branch[pos] = float(child_v) + log_pb.to(dtype=dtype)

        stop_prob = torch.exp(terminal - value)
        edge_probs = torch.exp(branch - value.index_select(0, row_ids)) if branch.numel() else branch
        edge_entropy = _edge_entropy(edge_probs=edge_probs, row_ids=row_ids, rows=rows)
        diagnostics = {
            "oracle/V_star_mean": value.mean().detach(),
            "oracle/terminal_J_mean": terminal.mean().detach(),
            "oracle/oracle_stop_prob_mean": stop_prob.mean().detach(),
            "oracle/oracle_edge_entropy": edge_entropy.mean().detach(),
            "oracle/topm_coverage": torch.tensor(
                float(self._coverage_kept) / float(max(self._coverage_seen, 1)),
                dtype=dtype,
                device=device,
            ),
            "oracle/topm_pruned_row_rate": torch.tensor(
                float(self._pruned_rows) / float(max(rows, 1)),
                dtype=dtype,
                device=device,
            ),
        }
        return BudgetedOracleOutput(
            terminal_J=terminal,
            V_star=value,
            stop_prob=stop_prob,
            edge_probs=edge_probs,
            edge_branch_logits=branch,
            frontier_edge_ids=edge_ids,
            frontier_batch_ids=row_ids,
            valid_mask=valid,
            diagnostics=diagnostics,
        )

    def _value_for_row(
        self,
        *,
        batch: RetrievalBatch,
        fb: FeatureBank,
        state: RolloutState,
        budget: int,
    ) -> tuple[float, float]:
        budget = max(int(budget), 0)
        key = _memo_key(state=state, budget=budget)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        reward = self.reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            state=state,
            diagnostics="basic",
        )
        j = float(reward.log_reward.detach().view(-1)[0].item())
        if budget <= 0:
            out = (j, j)
            self._memo[key] = out
            return out
        frontier = build_frontier(
            fb=fb,
            batch=batch,
            state=state,
            frontier_mode=self.frontier_mode,
        )
        edge_ids = frontier.edge_ids.to(device=state.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == 0:
            out = (j, j)
            self._memo[key] = out
            return out
        edge_ids = self._prune_edges_for_budget3(fb=fb, frontier=frontier, budget=budget)
        terms = [j]
        for edge_id in edge_ids.tolist():
            edge = torch.tensor([int(edge_id)], dtype=torch.long, device=state.device)
            row = torch.zeros((1,), dtype=torch.long, device=state.device)
            child = _successor_state(batch=batch, state=state, edge_ids=edge, row_ids=row)
            _, child_v = self._value_for_row(
                batch=batch,
                fb=fb,
                state=child,
                budget=budget - 1,
            )
            log_pb = float(_log_pb(batch=batch, successor=child, edge_ids=edge)[0].item())
            terms.append(child_v + log_pb)
        v = float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), dim=0).item())
        out = (j, v)
        self._memo[key] = out
        return out

    def _prune_edges_for_budget3(
        self,
        *,
        fb: FeatureBank,
        frontier: FrontierContext,
        budget: int,
    ) -> torch.Tensor:
        edge_ids = frontier.edge_ids.to(device=fb.node_h.device, dtype=torch.long).view(-1)
        seen = int(edge_ids.numel())
        if budget <= self.exact_budget or seen <= self.top_m_for_budget3:
            self._coverage_seen += seen
            self._coverage_kept += seen
            return edge_ids
        scores = frontier_semantic_scores(fb=fb, frontier=frontier)
        score = scores.query_relation_score + scores.query_new_node_score
        top = torch.topk(score, k=min(self.top_m_for_budget3, int(score.numel()))).indices
        self._coverage_seen += seen
        self._coverage_kept += int(top.numel())
        self._pruned_rows += 1
        return edge_ids.index_select(0, top)


def _as_rollout_state(*, batch: RetrievalBatch, state: State | RolloutState) -> RolloutState:
    if isinstance(state, RolloutState):
        return state
    device = batch.edge_index.device
    graph_ids = torch.arange(int(batch.num_graphs), dtype=torch.long, device=device)
    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
    node_belongs = node_batch.view(1, -1).eq(graph_ids.view(-1, 1))
    edge_belongs = edge_batch.view(1, -1).eq(graph_ids.view(-1, 1))
    return RolloutState(
        rollout_to_graph=graph_ids,
        expand_budget=int(state.expand_budget),
        edge_index=batch.edge_index,
        num_nodes=int(batch.num_nodes_total),
        num_edges=int(batch.edge_index.size(1)),
        active_nodes=state.active_nodes.to(device=device, dtype=torch.bool).view(1, -1) & node_belongs,
        active_edges=state.active_edges.to(device=device, dtype=torch.bool).view(1, -1) & edge_belongs,
        root_edges=state.root_edges.to(device=device, dtype=torch.bool).view(1, -1) & edge_belongs,
        boundary_nodes=(
            state.boundary_nodes.to(device=device, dtype=torch.bool).view(1, -1) & node_belongs
            if state.boundary_nodes is not None
            else None
        ),
    )


def _successor_state(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    edge_ids: torch.Tensor,
    row_ids: torch.Tensor,
) -> RolloutState:
    from src.weaver.policy import _successor_state_for_frontier

    return _successor_state_for_frontier(
        batch=batch,
        state=state,
        frontier_edge_ids=edge_ids,
        frontier_batch_ids=row_ids,
    )


def _log_pb(
    *,
    batch: RetrievalBatch,
    successor: RolloutState,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    from src.weaver.policy import _uniform_log_pb_for_successor_edges

    return _uniform_log_pb_for_successor_edges(
        batch=batch,
        successor=successor,
        frontier_edge_ids=edge_ids,
    )


def _memo_key(*, state: RolloutState, budget: int) -> tuple[int, tuple[int, ...], int]:
    graph_id = int(state.rollout_to_graph.view(-1)[0].item())
    expanded = state.expanded_edge_ids_for_rollout(0).detach().cpu().to(dtype=torch.long)
    return graph_id, tuple(sorted(int(x) for x in expanded.tolist())), int(budget)


def _filter_frontier(frontier: FrontierContext, mask: torch.Tensor, *, device: torch.device) -> FrontierContext:
    mask = mask.to(device=device, dtype=torch.bool).view(-1)
    return FrontierContext(
        edge_ids=frontier.edge_ids.to(device=device)[mask],
        src=frontier.src.to(device=device)[mask],
        dst=frontier.dst.to(device=device)[mask],
        graph_id=frontier.graph_id.to(device=device)[mask],
        src_active=frontier.src_active.to(device=device)[mask],
        dst_active=frontier.dst_active.to(device=device)[mask],
        static_graph_id=(
            None
            if frontier.static_graph_id is None
            else frontier.static_graph_id.to(device=device)[mask]
        ),
    )


def _edge_entropy(*, edge_probs: torch.Tensor, row_ids: torch.Tensor, rows: int) -> torch.Tensor:
    out = torch.zeros((int(rows),), dtype=torch.float32, device=edge_probs.device)
    if edge_probs.numel() == 0:
        return out
    entropy_terms = -edge_probs.clamp_min(1.0e-12).log() * edge_probs
    out.scatter_add_(0, row_ids.to(device=edge_probs.device, dtype=torch.long), entropy_terms)
    return out


__all__ = ["BudgetedLexicographicOracle", "BudgetedOracleOutput"]
