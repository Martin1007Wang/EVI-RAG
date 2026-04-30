from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import zero_last_linear


class StopExpandGate(nn.Module):
    """
    Stop/Expand option gate.

    This module only scores the option-level decision:

        P(Stop | s), P(Expand | s)
            = softmax([z_stop(s), z_expand(s)])

    The edge scorer separately provides conditional edge-ranking logits:

        P(e | s, Expand) = softmax_{e in frontier(s)} z_e

    When enabled, edge logits are used here only through size-normalized
    frontier summaries, not through raw joint Stop-vs-edge competition.

    Inputs:
        state_h:
            Query-conditioned evidence-state representation h_s.
        query_h:
            Question representation h_q.
        progress_ratio:
            |E_s \\ E_0| / expand_budget.
        edge_logmeanexp:
            log mean exp of conditional edge logits within each frontier.
            This removes raw frontier-size multiplicity.
        edge_max:
            max conditional edge logit within each frontier.

    Frontier feature:
        sharpness = edge_max - edge_logmeanexp

    The gate deliberately does not consume:
        - target labels;
        - target distances;
        - shortest-path masks;
        - stop-now reward;
        - reward advantage;
        - raw frontier size.

    Those quantities may be used by reward, diagnostics, proposal, or auxiliary
    losses, but should not be policy inputs at inference time.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        stop_bias_init: float = 0.0,
        expand_bias_init: float = 0.0,
        dropout: float = 0.0,
        trainable_bias: bool = True,
        use_frontier_summary: bool = True,
        detach_frontier_summary: bool = False,
        **_: object,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.use_frontier_summary = bool(use_frontier_summary)
        self.detach_frontier_summary = bool(detach_frontier_summary)
        self._stop_bias_init = float(stop_bias_init)
        self._expand_bias_init = float(expand_bias_init)

        self.stop_bias = nn.Parameter(torch.tensor(float(stop_bias_init)))
        self.expand_bias = nn.Parameter(torch.tensor(float(expand_bias_init)))

        self.stop_bias.requires_grad_(bool(trainable_bias))
        self.expand_bias.requires_grad_(bool(trainable_bias))

        scalar_dim = 3 if self.use_frontier_summary else 1
        input_dim = self.hidden_dim * 2 + scalar_dim
        self.scalar_norm: nn.Module
        self.scalar_norm = (
            nn.LayerNorm(3) if self.use_frontier_summary else nn.Identity()
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )

        # Initial option logits are controlled only by stop/expand bias.
        # This avoids hard-coded depth or frontier-energy priors.
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Reset the trainable Stop/Expand readout to configured neutral logits."""
        zero_last_linear(self.gate)
        with torch.no_grad():
            self.stop_bias.fill_(self._stop_bias_init)
            self.expand_bias.fill_(self._expand_bias_init)

    def forward(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        progress_ratio: torch.Tensor | None = None,
        edge_logmeanexp: torch.Tensor | None = None,
        edge_max: torch.Tensor | None = None,
        has_candidate_edge: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(query_h=query_h, state_h=state_h)

        batch_size = query_h.size(0)
        dtype = query_h.dtype
        device = query_h.device

        progress = _optional_vector(
            progress_ratio,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)

        if self.use_frontier_summary:
            frontier_value = _optional_vector(
                edge_logmeanexp,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

            frontier_max = _optional_vector(
                edge_max,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

            frontier_sharpness = frontier_max - frontier_value

            if has_candidate_edge is not None:
                has_edge = has_candidate_edge.to(device=device, dtype=torch.bool).view(
                    batch_size
                )
                frontier_value = torch.where(
                    has_edge, frontier_value, torch.zeros_like(frontier_value)
                )
                frontier_sharpness = torch.where(
                    has_edge,
                    frontier_sharpness,
                    torch.zeros_like(frontier_sharpness),
                )

            scalar_inputs = torch.stack(
                [progress, frontier_value, frontier_sharpness], dim=-1
            )
        else:
            scalar_inputs = progress.unsqueeze(-1)

        scalars = self.scalar_norm(scalar_inputs)

        features = torch.cat(
            [
                state_h,
                query_h,
                scalars,
            ],
            dim=-1,
        )

        logits = self.gate(features)

        stop_logit = logits[:, 0] + self.stop_bias.to(device=device, dtype=dtype)
        expand_logit = logits[:, 1] + self.expand_bias.to(device=device, dtype=dtype)

        return stop_logit, expand_logit

    def _validate_inputs(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> None:
        if query_h.ndim != 2:
            raise ValueError(
                f"query_h must have shape [B, H], got {tuple(query_h.shape)}."
            )
        if state_h.ndim != 2:
            raise ValueError(
                f"state_h must have shape [B, H], got {tuple(state_h.shape)}."
            )
        if query_h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected query_h hidden_dim={self.hidden_dim}, "
                f"got {query_h.size(-1)}."
            )
        if state_h.shape != query_h.shape:
            raise ValueError(
                "state_h shape must match query_h shape: "
                f"{tuple(state_h.shape)} != {tuple(query_h.shape)}."
            )


def _optional_vector(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)

    return value.to(device=device, dtype=dtype).view(batch_size)


__all__ = ["StopExpandGate"]
