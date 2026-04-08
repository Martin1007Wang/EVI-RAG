from __future__ import annotations

import torch
from torch import nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(max(num_layers - 1, 0)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _zero_last_linear(module: nn.Sequential) -> None:
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return
    raise TypeError("Expected nn.Sequential to contain at least one nn.Linear layer.")


class ZHead(nn.Module):
    """Predict log Z(q, G_0) from the query and root-state summary."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.mlp = _build_mlp(
            input_dim=hidden_dim * 2,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if zero_init_output:
            _zero_last_linear(self.mlp)

    def forward(
        self, question_h: torch.Tensor, root_subgraph_h: torch.Tensor
    ) -> torch.Tensor:
        z_ctx = torch.cat([question_h, root_subgraph_h], dim=-1)
        return self.mlp(z_ctx).squeeze(-1)


class ActionHead(nn.Module):
    """
    动作策略头 (Forward Policy Head)
    将动作空间解耦为：
    1. 宏观决策：Expand vs Stop
    2. 若 Expand，对候选有向边 (u, r, v) 做联合打分
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        type_feature_dim: int = 6,
        edge_discrimination_feature_dim: int = 6,
        prior_lambda_init: float = 10.0,
        learnable_prior_scale: bool = True,
        positive_prior_scale: bool = True,
        zero_init_type_output: bool = True,
        zero_init_expand_edge_output: bool = True,
    ):
        super().__init__()
        if positive_prior_scale and prior_lambda_init <= 0.0:
            raise ValueError(
                "prior_lambda_init must be > 0 when positive_prior_scale is enabled."
            )
        if type_feature_dim < 1:
            raise ValueError(f"type_feature_dim must be >= 1, got {type_feature_dim}.")
        if edge_discrimination_feature_dim < 1:
            raise ValueError(
                "edge_discrimination_feature_dim must be >= 1, got "
                f"{edge_discrimination_feature_dim}."
            )

        self.type_feature_dim = int(type_feature_dim)
        self.edge_discrimination_feature_dim = int(edge_discrimination_feature_dim)

        # 1. 宏观决策: P_type (Expand, Stop)
        self.type_scorer = _build_mlp(
            input_dim=hidden_dim + self.type_feature_dim,
            output_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if zero_init_type_output:
            _zero_last_linear(self.type_scorer)

        # 2. 扩张边打分: dynamic edge state plus global state and structure bits.
        self.expand_edge_scorer = _build_mlp(
            input_dim=hidden_dim * 2 + 5,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.edge_retrieval_scorer = _build_mlp(
            input_dim=self.edge_discrimination_feature_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if zero_init_expand_edge_output:
            _zero_last_linear(self.expand_edge_scorer)
            _zero_last_linear(self.edge_retrieval_scorer)

        self.learnable_prior_scale = bool(learnable_prior_scale)
        self.positive_prior_scale = bool(positive_prior_scale)
        prior_scale_param = torch.tensor(float(prior_lambda_init), dtype=torch.float32)
        if self.positive_prior_scale:
            prior_scale_param = prior_scale_param.log()

        if self.learnable_prior_scale:
            self._prior_scale = nn.Parameter(prior_scale_param)
        else:
            self.register_buffer("_prior_scale", prior_scale_param)

    @property
    def prior_scale(self) -> torch.Tensor:
        if self.positive_prior_scale:
            return self._prior_scale.exp()
        return self._prior_scale

    def forward(
        self,
        edge_state_h: torch.Tensor,
        subgraph_h: torch.Tensor,
        edge_batch_index: torch.Tensor,
        edge_struct_features: torch.Tensor,
        expand_edge_prior_logits: torch.Tensor | None = None,
        edge_discrimination_features: torch.Tensor | None = None,
        type_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        safe_subgraph_h = torch.nan_to_num(subgraph_h)
        safe_edge_state_h = torch.nan_to_num(edge_state_h)
        safe_edge_struct_features = torch.nan_to_num(edge_struct_features)

        # 在候选边级别联合评分，显式暴露 incoming/outgoing/closure 结构位。
        edge_ctx = torch.cat(
            [
                safe_edge_state_h,
                safe_subgraph_h[edge_batch_index],
                safe_edge_struct_features,
            ],
            dim=-1,
        )

        # 返回所有未归一化的原始潜力分数 (Logits)
        expand_edge_logits = self.expand_edge_scorer(edge_ctx).squeeze(-1)
        if edge_discrimination_features is None:
            edge_discrimination_features = safe_edge_state_h.new_zeros(
                (safe_edge_state_h.size(0), self.edge_discrimination_feature_dim)
            )
        else:
            edge_discrimination_features = torch.nan_to_num(
                edge_discrimination_features.to(safe_edge_state_h.dtype)
            )
            if (
                edge_discrimination_features.dim() != 2
                or edge_discrimination_features.size(1)
                != self.edge_discrimination_feature_dim
            ):
                raise ValueError(
                    "edge_discrimination_features must have shape "
                    f"(num_edges, {self.edge_discrimination_feature_dim}), got "
                    f"{tuple(edge_discrimination_features.shape)}."
                )
        expand_edge_logits = expand_edge_logits + self.edge_retrieval_scorer(
            edge_discrimination_features
        ).squeeze(-1)
        if expand_edge_prior_logits is not None:
            expand_edge_prior_logits = torch.nan_to_num(
                expand_edge_prior_logits.to(expand_edge_logits.dtype)
            )
            expand_edge_logits = (
                expand_edge_logits
                + self.prior_scale.to(expand_edge_logits.dtype)
                * expand_edge_prior_logits
            )

        if type_features is None:
            type_features = safe_subgraph_h.new_zeros(
                (safe_subgraph_h.size(0), self.type_feature_dim)
            )
        else:
            type_features = torch.nan_to_num(type_features.to(safe_subgraph_h.dtype))
            if (
                type_features.dim() != 2
                or type_features.size(1) != self.type_feature_dim
            ):
                raise ValueError(
                    "type_features must have shape "
                    f"(num_graphs, {self.type_feature_dim}), got {tuple(type_features.shape)}."
                )
        type_ctx = torch.cat([safe_subgraph_h, type_features], dim=-1)
        type_logits = self.type_scorer(type_ctx)

        return {
            "type_logits": type_logits,  # [Num_Graphs, 2]
            "expand_edge_logits": expand_edge_logits,  # [Total_Edges]
        }


__all__ = ["ActionHead", "ZHead"]
