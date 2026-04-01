from __future__ import annotations

import math

import torch
from torch import nn

from src.utils.precision_utils import align_float_input_dtype


def _build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(max(num_layers - 1, 0)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class StateFlowHead(nn.Module):
    """Question-conditioned log-flow head over state features."""

    def __init__(
        self,
        *,
        node_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        conditioning: str = "concat",
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.question_dim = int(question_dim)
        self.conditioning = str(conditioning)
        if self.conditioning not in {"concat", "none"}:
            raise ValueError(
                "StateFlowHead conditioning must be one of {'concat', 'none'}."
            )
        input_dim = self.node_dim
        if self.conditioning == "concat":
            input_dim += self.question_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.critic = _build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    def _build_inputs(
        self,
        *,
        node_features: torch.Tensor,
        question_features: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(node_features.shape[:-1]) != tuple(question_features.shape[:-1]):
            raise ValueError(
                "question_features must match node_features batch shape in StateFlowHead."
            )
        if int(node_features.size(-1)) != self.node_dim:
            raise ValueError(
                "node_features last dimension mismatch in StateFlowHead. "
                f"Expected {self.node_dim}, got {int(node_features.size(-1))}."
            )
        if int(question_features.size(-1)) != self.question_dim:
            raise ValueError(
                "question_features last dimension mismatch in StateFlowHead. "
                f"Expected {self.question_dim}, got {int(question_features.size(-1))}."
            )
        if self.conditioning == "none":
            return node_features
        return torch.cat((node_features, question_features), dim=-1)

    def forward(
        self, node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        critic_inputs = self._build_inputs(
            node_features=node_features,
            question_features=question_features,
        )
        critic_inputs = align_float_input_dtype(critic_inputs, module=self.input_norm)
        critic_inputs = self.input_norm(critic_inputs)
        critic_inputs = align_float_input_dtype(critic_inputs, module=self.critic[0])
        return self.critic(critic_inputs).squeeze(-1)


class ActionScoringHead(nn.Module):
    """Score candidate actions from a state query and candidate edge features."""

    def __init__(
        self,
        *,
        state_dim: int,
        relation_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        detach_input_features: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("action_scoring_head.num_layers must be >= 1.")
        self.detach_input_features = bool(detach_input_features)
        self.actor_dim = int(state_dim)
        self.query_norm = nn.LayerNorm(self.actor_dim)
        self.edge_norm = nn.LayerNorm(int(state_dim + relation_dim))
        self.query_mlp = _build_mlp(
            input_dim=self.actor_dim,
            output_dim=self.actor_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.edge_mlp = _build_mlp(
            input_dim=int(state_dim + relation_dim),
            output_dim=self.actor_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    @staticmethod
    def _validate_inputs(
        *,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> None:
        if candidate_state_features.shape != current_state_features.shape:
            raise ValueError(
                "candidate_state_features must match current_state_features shape in ActionScoringHead."
            )
        if tuple(relation_features.shape[:-1]) != tuple(
            current_state_features.shape[:-1]
        ):
            raise ValueError(
                "relation_features batch shape must match current_state_features in ActionScoringHead."
            )

    def encode_query(self, current_state_features: torch.Tensor) -> torch.Tensor:
        actor_query_inputs = align_float_input_dtype(
            current_state_features, module=self.query_norm
        )
        actor_query_inputs = self.query_norm(actor_query_inputs)
        actor_query_inputs = align_float_input_dtype(
            actor_query_inputs, module=self.query_mlp[0]
        )
        return self.query_mlp(actor_query_inputs)

    def encode_edge_keys(
        self,
        *,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        edge_inputs = torch.cat((relation_features, candidate_state_features), dim=-1)
        edge_inputs = align_float_input_dtype(edge_inputs, module=self.edge_norm)
        edge_inputs = self.edge_norm(edge_inputs)
        edge_inputs = align_float_input_dtype(edge_inputs, module=self.edge_mlp[0])
        return self.edge_mlp(edge_inputs)

    def score_from_encoded(
        self,
        *,
        actor_query: torch.Tensor,
        edge_key: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(actor_query.shape[:-1]) != tuple(edge_key.shape[:-1]):
            if int(actor_query.size(0)) == 1 and actor_query.dim() == edge_key.dim():
                actor_query = actor_query.expand_as(edge_key)
            else:
                raise ValueError(
                    "actor_query batch shape must match edge_key in ActionScoringHead."
                )
        logits = (actor_query * edge_key).sum(dim=-1)
        return logits / math.sqrt(float(max(self.actor_dim, 1)))

    def score_from_query(
        self,
        *,
        actor_query: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        edge_key = self.encode_edge_keys(
            candidate_state_features=candidate_state_features,
            relation_features=relation_features,
        )
        return self.score_from_encoded(actor_query=actor_query, edge_key=edge_key)

    def forward(
        self,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            current_state_features=current_state_features,
            candidate_state_features=candidate_state_features,
            relation_features=relation_features,
        )
        if self.detach_input_features:
            current_state_features = current_state_features.detach()
            relation_features = relation_features.detach()
            candidate_state_features = candidate_state_features.detach()
        actor_query = self.encode_query(current_state_features)
        return self.score_from_query(
            actor_query=actor_query,
            candidate_state_features=candidate_state_features,
            relation_features=relation_features,
        )


NodeFlowHead = StateFlowHead
TransitionPolicyHead = ActionScoringHead


__all__ = [
    "ActionScoringHead",
    "NodeFlowHead",
    "StateFlowHead",
    "TransitionPolicyHead",
]
