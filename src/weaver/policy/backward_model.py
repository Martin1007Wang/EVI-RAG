from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.policy.backward import BackwardPolicy, BackwardPolicyOutput, removable_edges
from src.weaver.state import StateBatch


@dataclass(frozen=True, slots=True)
class BackwardPolicyInput:
    question_h_by_graph: torch.Tensor
    edge_h: torch.Tensor


class BackwardScoringModel(nn.Module):
    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        backward_policy: BackwardPolicy,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.backward_policy = backward_policy

    def build_policy_input(
        self,
        features: FeaturePack,
    ) -> BackwardPolicyInput:
        return BackwardPolicyInput(
            question_h_by_graph=features.question_h.float(),
            edge_h=features.edge_h.float(),
        )

    def forward(
        self,
        *,
        child_state: StateBatch,
        graph_context: GraphContext,
        policy_input: BackwardPolicyInput,
    ) -> BackwardPolicyOutput:
        removable = removable_edges(
            child_state=child_state,
            graph_context=graph_context,
        )
        counts = torch.bincount(
            removable.row_ids,
            minlength=child_state.num_states,
        )
        non_root = child_state.edge_count.gt(0)
        if bool(counts[non_root].le(0).any()):
            raise ValueError("Every non-root child state must have a removable predecessor.")
        child_state_h = _build_state_h_batched(
            state_encoder=self.state_encoder,
            state=child_state,
            graph_context=graph_context,
            policy_input=policy_input,
        )
        return self.backward_policy(
            child_state_h=child_state_h,
            question_h_by_graph=policy_input.question_h_by_graph,
            edge_h=policy_input.edge_h,
            removable=removable,
        )


def _build_state_h_batched(
    *,
    state_encoder: StateEncoder,
    state: StateBatch,
    graph_context: GraphContext,
    policy_input: BackwardPolicyInput,
) -> torch.Tensor:
    selected = state.selected_edge_index()
    question_h_per_state = policy_input.question_h_by_graph.index_select(
        0,
        state.graph_ids,
    )
    return _encode_state_batch(
        state_encoder=state_encoder,
        num_states=int(state.num_states),
        question_h_per_state=question_h_per_state,
        selected_row_ids=selected.row_ids,
        selected_edge_ids=selected.edge_ids,
        edge_h=policy_input.edge_h,
        device=state.device,
        graph_context=graph_context,
    )


def _encode_state_batch(
    *,
    state_encoder: StateEncoder,
    num_states: int,
    question_h_per_state: torch.Tensor,
    selected_row_ids: torch.Tensor,
    selected_edge_ids: torch.Tensor,
    edge_h: torch.Tensor,
    device: torch.device,
    graph_context: GraphContext,
) -> torch.Tensor:
    del graph_context
    hidden_dim = int(question_h_per_state.shape[-1])

    if int(selected_row_ids.numel()) == 0:
        is_empty = torch.ones(num_states, dtype=torch.bool, device=device)
        dummy_kv = torch.zeros(num_states, 1, hidden_dim, device=device)
        full_mask = torch.ones(num_states, 1, dtype=torch.bool, device=device)
        return state_encoder(
            question_h=question_h_per_state,
            selected_edge_h=dummy_kv,
            key_padding_mask=full_mask,
            is_empty=is_empty,
        )

    order = torch.argsort(selected_row_ids, stable=True)
    selected_row_ids = selected_row_ids.index_select(0, order)
    selected_edge_ids = selected_edge_ids.index_select(0, order)

    edge_counts = torch.bincount(selected_row_ids, minlength=num_states)
    max_edges = int(edge_counts.max().item())
    is_empty = edge_counts.eq(0)

    state_offsets = torch.zeros(num_states + 1, dtype=torch.long, device=device)
    state_offsets[1:] = edge_counts.cumsum(0)
    local_pos = torch.arange(selected_row_ids.numel(), device=device) - state_offsets.index_select(0, selected_row_ids)

    padded = torch.zeros(num_states, max_edges, hidden_dim, device=device)
    key_padding_mask = torch.ones(num_states, max_edges, dtype=torch.bool, device=device)
    padded[selected_row_ids, local_pos] = edge_h.index_select(0, selected_edge_ids)
    key_padding_mask[selected_row_ids, local_pos] = False

    return state_encoder(
        question_h=question_h_per_state,
        selected_edge_h=padded,
        key_padding_mask=key_padding_mask,
        is_empty=is_empty,
    )


__all__ = [
    "BackwardPolicyInput",
    "BackwardScoringModel",
]
