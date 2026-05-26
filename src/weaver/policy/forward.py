from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from src.graph.segments import segment_log_softmax, segment_logsumexp
from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import (
    FeatureBank,
    select_node_embedding,
    select_node_text_mask,
    select_relation_embedding,
)
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.state import ActionSpace, StateBatch

from .output import PolicyOutput

Tensor = torch.Tensor


class ForwardPolicy(nn.Module):
    """
    Unified action-flow policy over STOP plus legal expansion edges.

    Contract:
    - StateBatch rows are ordered-prefix trajectory states.
    - ActionSpace contains only legal EXPAND actions.
    - STOP is not stored as a fake edge.
    - Policy does not enumerate, sort, filter, or repair ActionSpace.
    - Policy scores actions; StateBatch owns state semantics.
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        budget: int = 3,
        semantic_relation_weight: float = 1.0,
        semantic_dst_weight: float = 0.5,
        edge_residual_weight: float = 1.0,
        frontier_size_correction: float = 1.0,
        action_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.state_encoder = state_encoder
        self.budget = int(budget)

        if self.budget < 0:
            raise ValueError("budget must be nonnegative.")

        hidden_dim = int(state_encoder.hidden_dim)
        edge_dim = int(state_encoder.edge_output_dim)

        self.action_hidden_dim = int(action_hidden_dim or hidden_dim)
        self.semantic_relation_weight = float(semantic_relation_weight)
        self.semantic_dst_weight = float(semantic_dst_weight)
        self.edge_residual_weight = float(edge_residual_weight)
        self.frontier_size_correction = float(frontier_size_correction)

        if self.edge_residual_weight < 0.0:
            raise ValueError("edge_residual_weight must be nonnegative.")

        self.budget_embedding = nn.Embedding(
            self.budget + 1,
            hidden_dim,
        )

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, self.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.action_hidden_dim, 1),
        )

        self.edge_residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_dim, self.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.action_hidden_dim, 1),
        )

        self.reset_parameters()

    @property
    def hidden_dim(self) -> int:
        return int(self.state_encoder.hidden_dim)

    def forward(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
        action_space: ActionSpace,
    ) -> PolicyOutput:
        """
        Score STOP and all legal EXPAND actions.

        For each state z:

            state_log_flow(z)
              = logsumexp(
                    stop_log_flow(z),
                    {edge_log_flow(z, e): e in legal_expand(z)}
                )

        STOP and EXPAND edges compete in one action normalizer.
        This is not a separate Bernoulli stop model.
        """

        if int(action_space.num_states) != int(state.num_states):
            raise ValueError("action_space.num_states must match state.num_states.")
        if int(state.budget) != int(self.budget):
            raise ValueError(
                "state.budget must match policy budget, got "
                f"{int(state.budget)} and {int(self.budget)}."
            )

        query_h = self.state_encoder.query_embeddings(
            features=features,
            state=state,
        )

        selected_h = self.state_encoder.selected_edge_summary(
            features=features,
            state=state,
            context=context,
            query_h=query_h,
        )

        covered_h = self.state_encoder.covered_node_summary(
            features=features,
            state=state,
            context=context,
            query_h=query_h,
        )

        budget_h = self.encode_budget(state)

        stop_log_flow = self.score_stop_flow(
            query_h=query_h,
            selected_h=selected_h,
            covered_h=covered_h,
            budget_h=budget_h,
        )

        edge_log_flow, edge_raw_score, conditional_edge_log_prob = self.score_edge_flows(
            features=features,
            context=context,
            action_space=action_space,
            query_h=query_h,
            selected_h=selected_h,
            covered_h=covered_h,
            budget_h=budget_h,
        )

        continue_log_flow = segment_logsumexp(
            values=edge_log_flow,
            segment_ids=action_space.expand_state_ids,
            num_segments=int(state.num_states),
        ).float()

        state_log_flow = torch.logaddexp(
            stop_log_flow,
            continue_log_flow,
        ).float()

        stop_log_prob = (stop_log_flow - state_log_flow).float()

        continue_log_prob = torch.where(
            torch.isfinite(continue_log_flow),
            continue_log_flow - state_log_flow,
            state_log_flow.new_full(
                (int(state.num_states),),
                float("-inf"),
            ),
        ).float()

        if int(action_space.num_expansions) == 0:
            edge_log_prob = edge_log_flow
        else:
            edge_log_prob = (edge_log_flow - state_log_flow.index_select(0, action_space.expand_state_ids)).float()

        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=stop_log_prob,
            continue_log_prob=continue_log_prob,
            edge_log_flow=edge_log_flow,
            edge_log_prob=edge_log_prob,
            conditional_edge_log_prob=conditional_edge_log_prob,
            edge_raw_score=edge_raw_score,
        )

    def encode_budget(self, state: StateBatch) -> Tensor:
        remaining = torch.clamp(
            state.budget_left.to(dtype=torch.long),
            min=0,
            max=self.budget,
        )

        return self.budget_embedding(remaining)

    def score_stop_flow(
        self,
        *,
        query_h: Tensor,
        selected_h: Tensor,
        covered_h: Tensor,
        budget_h: Tensor,
    ) -> Tensor:
        return (
            self.stop_head(
                torch.cat(
                    (
                        query_h,
                        selected_h,
                        covered_h,
                        budget_h,
                    ),
                    dim=-1,
                )
            )
            .squeeze(-1)
            .float()
        )

    def score_edge_flows(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        action_space: ActionSpace,
        query_h: Tensor,
        selected_h: Tensor,
        covered_h: Tensor,
        budget_h: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if int(action_space.num_expansions) == 0:
            empty = query_h.new_empty((0,), dtype=torch.float32)
            return empty, empty, empty

        row_ids = action_space.expand_state_ids
        edge_ids = action_space.expand_edge_ids

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            context=context,
            edge_ids=edge_ids,
        )

        semantic_prior = self.score_edge_semantic_prior(
            features=features,
            context=context,
            query_h=query_h,
            action_space=action_space,
        )

        residual = self.score_edge_residual(
            query_h=query_h,
            selected_h=selected_h,
            covered_h=covered_h,
            budget_h=budget_h,
            edge_h=edge_h,
            row_ids=row_ids,
        )

        edge_raw_score = (semantic_prior + float(self.edge_residual_weight) * residual).float()

        edge_log_flow = self.size_normalized_edge_flow(
            edge_raw_score=edge_raw_score,
            action_space=action_space,
            frontier_size_correction=self.frontier_size_correction,
        )

        conditional_edge_log_prob = segment_log_softmax(
            edge_log_flow,
            row_ids,
            num_segments=int(action_space.num_states),
        ).float()

        return edge_log_flow, edge_raw_score, conditional_edge_log_prob

    def score_edge_residual(
        self,
        *,
        query_h: Tensor,
        selected_h: Tensor,
        covered_h: Tensor,
        budget_h: Tensor,
        edge_h: Tensor,
        row_ids: Tensor,
    ) -> Tensor:
        return (
            self.edge_residual_head(
                torch.cat(
                    (
                        query_h.index_select(0, row_ids),
                        selected_h.index_select(0, row_ids),
                        covered_h.index_select(0, row_ids),
                        budget_h.index_select(0, row_ids),
                        edge_h,
                    ),
                    dim=-1,
                )
            )
            .squeeze(-1)
            .float()
        )

    def score_edge_semantic_prior(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        query_h: Tensor,
        action_space: ActionSpace,
    ) -> Tensor:
        edge_ids = action_space.expand_edge_ids
        row_ids = action_space.expand_state_ids

        dst_node_ids = context.edge_dst.index_select(0, edge_ids)

        query_edge_h = query_h.index_select(0, row_ids)
        relation_h = select_relation_embedding(features, edge_ids)
        dst_h = select_node_embedding(features, dst_node_ids)

        dst_text_mask = select_node_text_mask(
            features,
            dst_node_ids,
        ).to(dtype=query_edge_h.dtype)

        relation_prior = (query_edge_h * relation_h).sum(dim=-1)
        dst_prior = dst_text_mask * (query_edge_h * dst_h).sum(dim=-1)

        return (self.semantic_relation_weight * relation_prior + self.semantic_dst_weight * dst_prior).float()

    @staticmethod
    def size_normalized_edge_flow(
        *,
        edge_raw_score: Tensor,
        action_space: ActionSpace,
        frontier_size_correction: float = 1.0,
    ) -> Tensor:
        """
        Convert per-edge raw scores into edge log-flow.

        The correction subtracts beta * log |frontier(z)|. beta=1 makes the
        continue flow a log-mean-exp over edge scores; beta=0 makes it a
        log-sum-exp over edge scores.
        """

        row_ids = action_space.expand_state_ids

        expand_count = action_space.expand_count.index_select(0, row_ids)
        log_expand_count = expand_count.to(dtype=edge_raw_score.dtype).log()

        return (edge_raw_score - float(frontier_size_correction) * log_expand_count).float()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.budget_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        for module in self.stop_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)

        for module in self.edge_residual_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)

        _zero_linear(self.stop_head[-1])
        _zero_linear(self.edge_residual_head[-1])


def _zero_linear(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")

    init.zeros_(module.weight)
    init.zeros_(module.bias)


__all__ = [
    "ForwardPolicy",
]
