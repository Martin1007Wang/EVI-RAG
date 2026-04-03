from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.components import ActionScoringHead

from .actor_distribution import (
    _collect_state_distribution_context as _collect_state_distribution_context_impl,
)
from .actor_distribution import (
    build_action_distribution as _build_action_distribution_impl,
)
from .actor_distribution import (
    build_state_distribution as _build_state_distribution_impl,
)
from .actor_scoring import _build_edge_logits
from .actor_scoring import _build_edge_logits_batch
from .actor_scoring import _build_failure_stop_logit
from .actor_scoring import _build_failure_stop_logits
from .actor_scoring import _build_mlp
from .actor_scoring import _build_node_logits
from .actor_scoring import _build_node_logits_batch
from .actor_scoring import _build_relation_logits
from .actor_scoring import _build_relation_logits_batch
from .actor_scoring import _build_stop_choice_logits
from .actor_scoring import _build_stop_choice_logits_batch
from .actor_scoring import _log_action_distribution_stats
from .actor_types import AnswerStopChoice
from .actor_types import HierarchicalEdgeChoice
from .actor_types import HierarchicalNodeChoice
from .actor_types import HierarchicalRelationChoice
from .actor_types import HierarchicalStateActionDistribution
from .actor_types import SubgraphActionDistribution


class SubgraphActor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        actor: dict[str, Any],
        proposal_prior: dict[str, Any],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.proposal_prior = dict(proposal_prior)
        node_struct_dim = 4
        relation_struct_dim = 4
        candidate_struct_dim = 6
        self.node_focus_norm = nn.LayerNorm((2 * self.hidden_dim) + node_struct_dim)
        self.node_focus_head = _build_mlp(
            input_dim=(2 * self.hidden_dim) + node_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        self.continue_head = nn.Linear(self.hidden_dim, 1)
        stop_struct_dim = 3
        self.stop_choice_norm = nn.LayerNorm((2 * self.hidden_dim) + stop_struct_dim)
        self.stop_choice_head = _build_mlp(
            input_dim=(2 * self.hidden_dim) + stop_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.failure_stop_norm = nn.LayerNorm(self.hidden_dim + stop_struct_dim)
        self.failure_stop_head = _build_mlp(
            input_dim=self.hidden_dim + stop_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.relation_norm = nn.LayerNorm((3 * self.hidden_dim) + relation_struct_dim)
        self.relation_head = _build_mlp(
            input_dim=(3 * self.hidden_dim) + relation_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.candidate_encoder_norm = nn.LayerNorm(
            (3 * self.hidden_dim) + candidate_struct_dim
        )
        self.candidate_encoder = _build_mlp(
            input_dim=(3 * self.hidden_dim) + candidate_struct_dim,
            output_dim=self.hidden_dim,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.action_head = ActionScoringHead(
            state_dim=self.hidden_dim,
            relation_dim=self.hidden_dim,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
            detach_input_features=False,
        )
        # A moderate edge batch keeps the deeper-step actor kernels large enough
        # to amortize launch overhead without recreating the giant full-support
        # candidate surface in one allocation.
        self.edge_logit_chunk_size = 4096

    _log_action_distribution_stats = staticmethod(_log_action_distribution_stats)
    _build_node_logits = _build_node_logits
    _build_node_logits_batch = _build_node_logits_batch
    _build_relation_logits = _build_relation_logits
    _build_relation_logits_batch = _build_relation_logits_batch
    _build_edge_logits = _build_edge_logits
    _build_edge_logits_batch = _build_edge_logits_batch
    _build_stop_choice_logits = _build_stop_choice_logits
    _build_stop_choice_logits_batch = _build_stop_choice_logits_batch
    _build_failure_stop_logits = _build_failure_stop_logits
    _build_failure_stop_logit = _build_failure_stop_logit
    _collect_state_distribution_context = _collect_state_distribution_context_impl
    build_state_distribution = _build_state_distribution_impl
    build_action_distribution = _build_action_distribution_impl


__all__ = [
    "AnswerStopChoice",
    "HierarchicalEdgeChoice",
    "HierarchicalNodeChoice",
    "HierarchicalRelationChoice",
    "HierarchicalStateActionDistribution",
    "SubgraphActionDistribution",
    "SubgraphActor",
]
