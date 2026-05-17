from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import init_xavier
from src.weaver.context import FlowContext
from src.weaver.state import State

from .feature_encoder import FeatureBank
from .state_encoder import StateEncoding


class TerminalUtilityEstimator(nn.Module):
    """
    Learned terminal log-flow estimator s_hat_psi(z, q).

    It receives only label-free flow inputs. Gold utility is supplied outside
    this module by the training objective.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        adapter_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        adapter_hidden_dim = self.hidden_dim if adapter_hidden_dim is None else int(adapter_hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, 1),
        )
        self._reset_parameters()

    def forward(
        self,
        *,
        context: FlowContext,
        features: FeatureBank,
        state: State,
        state_encoding: StateEncoding,
    ) -> torch.Tensor:
        del context, features, state
        x = torch.cat(
            [state_encoding.query_h, state_encoding.state_h],
            dim=-1,
        )
        return self.head(x).squeeze(-1)

    def _reset_parameters(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                init_xavier(module)


__all__ = ["TerminalUtilityEstimator"]
