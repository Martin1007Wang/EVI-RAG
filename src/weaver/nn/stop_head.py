from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import zero_last_linear


class LearnedStopHead(nn.Module):
    """
    Learned Stop action logit head.

    Stop is a regular action in the flat target policy. Its logit is learned
    from state, progress, frontier semantic-base summary, and scalar state
    stats; it never receives stop-now reward as an input.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        stop_bias_init: float = 0.0,
        dropout: float = 0.0,
        trainable_bias: bool = True,
        use_progress: bool = False,
        use_frontier_summary: bool = False,
        state_stat_dim: int = 0,
        **removed_kwargs: object,
    ) -> None:
        super().__init__()

        if removed_kwargs:
            raise ValueError(
                "LearnedStopHead no longer accepts Stop/Expand gate keys: "
                f"{sorted(removed_kwargs)}."
            )

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.use_progress = bool(use_progress)
        self.use_frontier_summary = bool(use_frontier_summary)
        self.state_stat_dim = int(state_stat_dim)
        if self.state_stat_dim < 0:
            raise ValueError(
                f"state_stat_dim must be >= 0, got {state_stat_dim}."
            )

        self._stop_bias_init = float(stop_bias_init)

        self.stop_bias = nn.Parameter(torch.tensor(self._stop_bias_init))

        self.stop_bias.requires_grad_(bool(trainable_bias))

        scalar_dim = (
            int(self.use_progress)
            + 3 * int(self.use_frontier_summary)
            + self.state_stat_dim
        )
        input_dim = self.hidden_dim + scalar_dim

        self.scalar_norm = nn.LayerNorm(scalar_dim) if scalar_dim > 0 else None

        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        zero_last_linear(self.net)
        with torch.no_grad():
            self.stop_bias.fill_(self._stop_bias_init)

    def forward(
        self,
        *,
        state_h: torch.Tensor,
        edge_logits: torch.Tensor | None = None,
        edge_batch_index: torch.Tensor | None = None,
        num_graphs: int | None = None,
        progress_ratio: torch.Tensor | None = None,
        frontier_summary: torch.Tensor | None = None,
        edge_logmeanexp: torch.Tensor | None = None,
        edge_max: torch.Tensor | None = None,
        edge_log_size: torch.Tensor | None = None,
        state_stats: torch.Tensor | None = None,
        has_candidate_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_logits, edge_batch_index

        if state_h.ndim != 2:
            raise ValueError(
                f"state_h must have shape [B, H], got {tuple(state_h.shape)}."
            )
        if state_h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected state_h hidden_dim={self.hidden_dim}, got {state_h.size(-1)}."
            )

        batch_size = int(state_h.size(0))
        if num_graphs is not None and int(num_graphs) != batch_size:
            raise ValueError(
                f"num_graphs must match state_h batch size {batch_size}, got {num_graphs}."
            )

        device = state_h.device
        dtype = state_h.dtype

        progress = _vector_or_zeros(
            progress_ratio,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)

        scalars: list[torch.Tensor] = []
        if self.use_progress:
            scalars.append(progress)

        if self.use_frontier_summary:
            max_value, logmeanexp, log_size = _resolve_frontier_summary(
                frontier_summary=frontier_summary,
                edge_logmeanexp=edge_logmeanexp,
                edge_max=edge_max,
                edge_log_size=edge_log_size,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

            if has_candidate_edge is not None:
                has_edge = has_candidate_edge.to(device=device, dtype=torch.bool).view(
                    batch_size
                )
                max_value = torch.where(
                    has_edge, max_value, torch.zeros_like(max_value)
                )
                logmeanexp = torch.where(
                    has_edge,
                    logmeanexp,
                    torch.zeros_like(logmeanexp),
                )
                log_size = torch.where(has_edge, log_size, torch.zeros_like(log_size))

            scalars.extend([max_value, logmeanexp, log_size])

        stop_input = state_h
        if self.state_stat_dim > 0:
            stats = _matrix_or_zeros(
                state_stats,
                batch_size=batch_size,
                width=self.state_stat_dim,
                device=device,
                dtype=dtype,
            )
            scalars.extend([stats[:, idx] for idx in range(self.state_stat_dim)])

        if scalars:
            scalar_h = torch.stack(scalars, dim=-1)
            if self.scalar_norm is not None:
                scalar_h = self.scalar_norm(scalar_h)
            stop_input = torch.cat([state_h, scalar_h], dim=-1)

        return self.net(stop_input).squeeze(-1) + self.stop_bias.to(
            device=device,
            dtype=dtype,
        )


def _resolve_frontier_summary(
    *,
    frontier_summary: torch.Tensor | None,
    edge_logmeanexp: torch.Tensor | None,
    edge_max: torch.Tensor | None,
    edge_log_size: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frontier_summary is not None:
        summary = frontier_summary.to(device=device, dtype=dtype)
        if summary.shape != (batch_size, 3):
            raise ValueError(
                f"frontier_summary must have shape [{batch_size}, 3], "
                f"got {tuple(summary.shape)}."
            )
        return summary[:, 0], summary[:, 1], summary[:, 2]

    return (
        _vector_or_zeros(edge_max, batch_size=batch_size, device=device, dtype=dtype),
        _vector_or_zeros(
            edge_logmeanexp,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
        _vector_or_zeros(
            edge_log_size,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
    )


def _vector_or_zeros(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)

    return value.to(device=device, dtype=dtype).view(batch_size)


def _matrix_or_zeros(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, width, device=device, dtype=dtype)

    value = value.to(device=device, dtype=dtype)
    if value.shape != (batch_size, width):
        raise ValueError(
            f"state_stats must have shape [{batch_size}, {width}], "
            f"got {tuple(value.shape)}."
        )
    return value


__all__ = ["LearnedStopHead"]
