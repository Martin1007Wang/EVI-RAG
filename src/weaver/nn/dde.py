from __future__ import annotations

import torch
from torch import nn


class DirectionalDDE(nn.Module):
    """
    Anchor-conditioned directional diffusion encoding.

    For each node, the output contains the anchor indicator followed by forward
    propagation coordinates and backward propagation coordinates. Propagation is
    answer-agnostic and uses only graph structure plus anchor nodes.
    """

    def __init__(
        self,
        *,
        num_forward_rounds: int = 2,
        num_backward_rounds: int = 2,
        include_anchor_indicator: bool = True,
    ) -> None:
        super().__init__()

        self.num_forward_rounds = _non_negative_int(
            num_forward_rounds,
            "num_forward_rounds",
        )
        self.num_backward_rounds = _non_negative_int(
            num_backward_rounds,
            "num_backward_rounds",
        )
        self.include_anchor_indicator = bool(include_anchor_indicator)

        if self.output_dim <= 0:
            raise ValueError(
                "DirectionalDDE must emit at least one coordinate; enable the "
                "anchor indicator or use a positive number of propagation rounds."
            )

    @property
    def output_dim(self) -> int:
        return (
            int(self.include_anchor_indicator)
            + self.num_forward_rounds
            + self.num_backward_rounds
        )

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        anchor_node_ids: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        num_nodes = int(num_nodes)
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")

        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
            )

        device = edge_index.device
        anchor = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        anchor_node_ids = anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchor_node_ids.numel() > 0:
            valid = anchor_node_ids.ge(0) & anchor_node_ids.lt(num_nodes)
            if bool(valid.any()):
                anchor[anchor_node_ids[valid]] = 1.0

        columns: list[torch.Tensor] = []
        if self.include_anchor_indicator:
            columns.append(anchor)

        src = edge_index[0]
        dst = edge_index[1]

        forward_h = anchor
        for _ in range(self.num_forward_rounds):
            forward_h = _mean_messages(
                values=forward_h,
                source_index=src,
                target_index=dst,
                num_nodes=num_nodes,
            )
            columns.append(forward_h)

        backward_h = anchor
        for _ in range(self.num_backward_rounds):
            backward_h = _mean_messages(
                values=backward_h,
                source_index=dst,
                target_index=src,
                num_nodes=num_nodes,
            )
            columns.append(backward_h)

        if not columns:
            return anchor.new_zeros((num_nodes, 0))

        return torch.stack(columns, dim=-1)


def _mean_messages(
    *,
    values: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    out = values.new_zeros((num_nodes,))
    if source_index.numel() == 0:
        return out

    out.index_add_(0, target_index, values.index_select(0, source_index))

    counts = values.new_zeros((num_nodes,))
    counts.index_add_(
        0,
        target_index,
        torch.ones_like(target_index, dtype=values.dtype),
    )

    return out / counts.clamp_min(1.0)


def _non_negative_int(value: object, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}.")
    return out


__all__ = ["DirectionalDDE"]
