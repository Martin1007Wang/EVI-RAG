from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_log_softmax
from src.utils.nn_utils import init_xavier

from ..context import GraphContext
from ..nn.feature_encoder import EncodedFeatures
from ..nn.state_encoder import StateEncoder
from ..state import Frontier, State


@dataclass(frozen=True, slots=True)
class EdgeOnlyPolicyOutput:
    frontier_row_ids: torch.Tensor
    frontier_edge_ids: torch.Tensor
    edge_logits: torch.Tensor
    edge_log_prob: torch.Tensor
    num_rows: int
    num_edges: int
    action_key_order: torch.Tensor | None = None
    sorted_action_keys: torch.Tensor | None = None

    def __post_init__(self) -> None:
        device = self.edge_logits.device
        for name, value in {
            "frontier_row_ids": self.frontier_row_ids,
            "frontier_edge_ids": self.frontier_edge_ids,
        }.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as edge_logits.")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long.")
        for name, value in {
            "edge_logits": self.edge_logits,
            "edge_log_prob": self.edge_log_prob,
        }.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as edge_logits.")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32.")

        if self.action_key_order is None or self.sorted_action_keys is None:
            edge_keys = self.frontier_row_ids * int(self.num_edges) + self.frontier_edge_ids
            order = torch.argsort(edge_keys)
            object.__setattr__(self, "action_key_order", order)
            object.__setattr__(self, "sorted_action_keys", edge_keys.index_select(0, order))

    def has_frontier(self) -> torch.Tensor:
        out = torch.zeros(int(self.num_rows), dtype=torch.bool, device=self.edge_logits.device)
        if self.frontier_row_ids.numel() > 0:
            out.index_fill_(0, self.frontier_row_ids, True)
        return out

    def gather_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.edge_logits.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.edge_logits.device, dtype=torch.long).view(-1)
        if int(row_ids.numel()) != int(edge_ids.numel()):
            raise ValueError("row_ids and edge_ids must have the same length.")
        if row_ids.numel() == 0:
            return self.edge_logits.new_empty((0,))
        if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(self.num_rows)).any()):
            raise IndexError("row_ids must be in [0, EdgeOnlyPolicyOutput.num_rows).")
        if bool(edge_ids.lt(0).any()) or bool(edge_ids.ge(int(self.num_edges)).any()):
            raise IndexError("edge_ids must be in [0, EdgeOnlyPolicyOutput.num_edges).")

        target_keys = row_ids * int(self.num_edges) + edge_ids
        sorted_keys = self.sorted_action_keys
        order = self.action_key_order
        positions = torch.searchsorted(sorted_keys, target_keys)
        if bool(positions.ge(sorted_keys.numel()).any()):
            raise KeyError("Requested edge action is not present in EdgeOnlyPolicyOutput.")
        matched = sorted_keys.index_select(0, positions)
        if not torch.equal(matched, target_keys):
            raise KeyError("Requested edge action is not present in EdgeOnlyPolicyOutput.")
        return self.edge_log_prob.index_select(0, order.index_select(0, positions))

    def sample(self, *, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(device=self.edge_logits.device, dtype=torch.long).view(-1)
        if rows.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=rows.device)
        has_frontier = self.has_frontier().index_select(0, rows)
        if not bool(has_frontier.all()):
            raise ValueError("Edge-only policy cannot sample rows with empty frontier.")

        row_pos = torch.full((int(self.num_rows),), -1, dtype=torch.long, device=rows.device)
        row_pos[rows] = torch.arange(rows.numel(), dtype=torch.long, device=rows.device)
        keep = row_pos.index_select(0, self.frontier_row_ids).ge(0)
        edge_positions = keep.nonzero(as_tuple=False).flatten()
        segment_ids = row_pos.index_select(0, self.frontier_row_ids.index_select(0, edge_positions))
        logits = self.edge_log_prob.index_select(0, edge_positions)
        gumbel = -torch.empty_like(logits).exponential_().log()
        winner = _segment_argmax(
            values=logits + gumbel,
            segment_ids=segment_ids,
            num_segments=int(rows.numel()),
        )
        picked_positions = edge_positions.index_select(0, winner)
        return self.frontier_edge_ids.index_select(0, picked_positions)


class EdgeOnlyProposalPolicy(nn.Module):
    """
    Edge-only fixed-horizon proposal policy.

    This policy deliberately has no STOP action and no state-flow head.
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        max_expand_budget: int = 3,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.max_expand_budget = int(max_expand_budget)
        hidden_dim = state_encoder.hidden_dim
        edge_dim = state_encoder.edge_encoder.output_dim
        self.budget_embedding = nn.Embedding(self.max_expand_budget + 1, hidden_dim)
        self.edge_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    @property
    def hidden_dim(self) -> int:
        return self.state_encoder.hidden_dim

    def forward(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        frontier: Frontier,
    ) -> EdgeOnlyPolicyOutput:
        encoding = self.state_encoder(features=features, state=state, context=context)
        if frontier.edge_ids.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=state.device)
            empty_float = torch.empty(0, dtype=torch.float32, device=state.device)
            return EdgeOnlyPolicyOutput(
                frontier_row_ids=empty_long,
                frontier_edge_ids=empty_long,
                edge_logits=empty_float,
                edge_log_prob=empty_float,
                num_rows=state.num_rows,
                num_edges=state.num_edges,
            )

        row_ids = frontier.row_ids.to(device=state.device, dtype=torch.long)
        budget_h = self.encode_budget(state)
        row_frontier_size = torch.bincount(row_ids, minlength=int(state.num_rows)).to(
            dtype=torch.float32,
            device=state.device,
        )
        edge_log_reference = -torch.log(row_frontier_size.index_select(0, row_ids).clamp_min(1.0))
        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=torch.empty(0, dtype=torch.long, device=state.device),
            edge_ids=frontier.edge_ids,
            dst_node_ids=torch.empty(0, dtype=torch.long, device=state.device),
        )
        edge_advantage = self.edge_score_head(
            torch.cat(
                [
                    encoding.query_h.index_select(0, row_ids),
                    encoding.row_state_h.index_select(0, row_ids),
                    budget_h.index_select(0, row_ids),
                    edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1).float()
        edge_logits = edge_log_reference + edge_advantage
        edge_log_prob = segment_log_softmax(
            edge_logits,
            row_ids,
            num_segments=int(state.num_rows),
        ).float()
        return EdgeOnlyPolicyOutput(
            frontier_row_ids=frontier.row_ids,
            frontier_edge_ids=frontier.edge_ids,
            edge_logits=edge_logits.float(),
            edge_log_prob=edge_log_prob,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )

    def encode_budget(self, state: State) -> torch.Tensor:
        remaining = torch.clamp(
            state.remaining_budget.to(dtype=torch.long),
            min=0,
            max=self.max_expand_budget,
        )
        return self.budget_embedding(remaining)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.budget_embedding.weight, mean=0.0, std=0.02)
        for module in self.edge_score_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.edge_score_head[-1])


class PrefixSelector(nn.Module):
    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        hidden_dim = state_encoder.hidden_dim
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def forward(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        trajectory_ids: torch.Tensor,
        num_trajectories: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.state_encoder(features=features, state=state, context=context)
        scores = self.score_head(torch.cat([encoding.query_h, encoding.row_state_h], dim=-1)).squeeze(-1).float()
        log_prob = segment_log_softmax(
            scores,
            trajectory_ids.to(device=scores.device, dtype=torch.long),
            num_segments=int(num_trajectories),
        ).float()
        return scores, log_prob

    def reset_parameters(self) -> None:
        for module in self.score_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.score_head[-1])


def _segment_argmax(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    best = values.new_full((int(num_segments),), -torch.inf)
    best.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    matches = values.eq(best.index_select(0, segment_ids))
    candidate_pos = torch.arange(values.numel(), device=values.device, dtype=torch.long)
    fallback = torch.full_like(candidate_pos, values.numel())
    winner = torch.full((int(num_segments),), values.numel(), dtype=torch.long, device=values.device)
    winner.scatter_reduce_(
        0,
        segment_ids,
        torch.where(matches, candidate_pos, fallback),
        reduce="amin",
        include_self=True,
    )
    if bool(winner.eq(values.numel()).any()):
        raise RuntimeError("segment argmax failed to pick an action for at least one segment.")
    return winner


def _zero_linear(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


__all__ = [
    "EdgeOnlyPolicyOutput",
    "EdgeOnlyProposalPolicy",
    "PrefixSelector",
]
