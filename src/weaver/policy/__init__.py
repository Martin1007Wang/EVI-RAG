from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.graph.segments import segment_log_softmax
from src.weaver.context import GraphContext

from .nn.edge_flow_scorer import EdgeActionScorer
from .nn.feature_encoder import FeatureBank
from .nn.frontier_encoder import FrontierEncoder
from .nn.state_encoder import StateEncoding, StateEncoder
from .state import Frontier, FrontierBuilder, State


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    frontier: Frontier
    stop_logit: torch.Tensor
    stop_log_prob: torch.Tensor
    continue_log_prob: torch.Tensor
    edge_logits: torch.Tensor
    edge_log_prob: torch.Tensor
    transition_log_prob: torch.Tensor


class Policy(nn.Module):
    """
    Forward construction policy.
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        frontier_encoder: FrontierEncoder,
        edge_scorer: EdgeActionScorer,
        frontier_score_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.frontier_encoder = frontier_encoder
        self.edge_scorer = edge_scorer
        self.frontier_score_chunk_size = int(frontier_score_chunk_size)
        if self.frontier_score_chunk_size <= 0:
            raise ValueError(
                "frontier_score_chunk_size must be positive, "
                f"got {frontier_score_chunk_size}."
            )

        hidden_dim = int(self.state_encoder.hidden_dim)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.state_encoder.hidden_dim)

    @property
    def max_budget(self) -> int:
        return int(self.state_encoder.max_budget)

    def forward(
        self,
        *,
        context: GraphContext,
        state: State,
        features: FeatureBank,
        frontier_builder: FrontierBuilder,
    ) -> PolicyOutput:
        state_encoding = self.state_encoder(
            features=features,
            context=context,
            state=state,
        )
        num_states = int(state_encoding.state_h.size(0))
        frontier = frontier_builder.build(state)
        edge_logits = self.score_edges(
            state_encoding=state_encoding,
            context=context,
            features=features,
            state=state,
            frontier=frontier,
        )
        stop_logit = self.stop_logits(state_encoding).view(num_states)
        stop_log_prob = F.logsigmoid(stop_logit)
        continue_log_prob = F.logsigmoid(-stop_logit)
        edge_log_prob = self.normalize_edge_logits(
            edge_logits=edge_logits,
            frontier=frontier,
            num_states=num_states,
        )
        if edge_log_prob.numel() == 0:
            transition_log_prob = edge_log_prob
        else:
            rows = frontier.row_ids.to(device=edge_log_prob.device, dtype=torch.long).view(-1)
            transition_log_prob = continue_log_prob.index_select(0, rows) + edge_log_prob
        return PolicyOutput(
            frontier=frontier,
            stop_logit=stop_logit.view(num_states),
            stop_log_prob=stop_log_prob.view(num_states),
            continue_log_prob=continue_log_prob.view(num_states),
            edge_logits=edge_logits.view(-1),
            edge_log_prob=edge_log_prob.view(-1),
            transition_log_prob=transition_log_prob.view(-1),
        )

    def stop_logits(
        self,
        state_encoding: StateEncoding,
    ) -> torch.Tensor:
        return self.stop_head(
            torch.cat([state_encoding.query_h, state_encoding.state_h], dim=-1)
        ).squeeze(-1)

    def score_edges(
        self,
        *,
        state_encoding: StateEncoding,
        context: GraphContext,
        features: FeatureBank,
        state: State,
        frontier: Frontier,
    ) -> torch.Tensor:
        num_frontier = int(frontier.edge_ids.numel())
        if num_frontier == 0:
            return state_encoding.state_h.new_empty((0,))

        frontier_encoding = self.frontier_encoder(
            context=context,
            features=features,
            state=state,
            frontier=frontier,
        )

        device = features.edge_h.device
        row_ids = frontier_encoding.row_ids.to(device=device, dtype=torch.long).view(-1)
        edge_h = frontier_encoding.edge_h.to(device=device)
        query_h = frontier_encoding.query_h.to(device=device)
        rel_sem_h = frontier_encoding.rel_sem_h.to(device=device)
        query_sem_h = frontier_encoding.query_sem_h.to(device=device)

        def score_chunk(
            row_chunk: torch.Tensor,
            edge_chunk: torch.Tensor,
            query_chunk: torch.Tensor,
            rel_sem_chunk: torch.Tensor,
            query_sem_chunk: torch.Tensor,
        ) -> torch.Tensor:
            return self.edge_scorer.score_tensors(
                state_h=state_encoding.state_h,
                row_ids=row_chunk,
                edge_h=edge_chunk,
                query_h=query_chunk,
                rel_sem_h=rel_sem_chunk,
                query_sem_h=query_sem_chunk,
            )

        use_checkpoint = torch.is_grad_enabled()
        parts: list[torch.Tensor] = []
        chunk_size = int(self.frontier_score_chunk_size)
        for start in range(0, num_frontier, chunk_size):
            stop = min(start + chunk_size, num_frontier)
            row_chunk = row_ids[start:stop]
            edge_chunk = edge_h[start:stop]
            query_chunk = query_h[start:stop]
            rel_sem_chunk = rel_sem_h[start:stop]
            query_sem_chunk = query_sem_h[start:stop]
            if use_checkpoint:
                part = checkpoint(
                    score_chunk,
                    row_chunk,
                    edge_chunk,
                    query_chunk,
                    rel_sem_chunk,
                    query_sem_chunk,
                    use_reentrant=False,
                )
            else:
                part = score_chunk(
                    row_chunk,
                    edge_chunk,
                    query_chunk,
                    rel_sem_chunk,
                    query_sem_chunk,
                )
            parts.append(part.view(-1))
        return torch.cat(parts, dim=0).view(num_frontier)

    @staticmethod
    def normalize_edge_logits(
        *,
        edge_logits: torch.Tensor,
        frontier: Frontier,
        num_states: int,
    ) -> torch.Tensor:
        edge_logits = edge_logits.view(-1)
        if edge_logits.numel() == 0:
            return edge_logits
        row_ids = frontier.row_ids.to(device=edge_logits.device, dtype=torch.long).view(-1)
        return segment_log_softmax(
            edge_logits,
            row_ids,
            num_segments=int(num_states),
        )


__all__ = [
    "Policy",
    "PolicyOutput",
]
