from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch


def node_mask_from_ids(
    ids: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = ids.to(device=device, dtype=torch.long).view(-1)

    mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    if ids.numel() == 0:
        return mask

    _check_id_range(ids, upper=int(num_nodes), name=name)
    mask[ids] = True
    return mask


def anchor_node_mask(
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    return node_mask_from_ids(
        batch.anchor_node_ids,
        num_nodes=int(batch.num_nodes_total),
        device=device,
        name="anchor_node_ids",
    )


def _check_id_range(
    ids: torch.Tensor,
    *,
    upper: int,
    name: str,
) -> None:
    if ids.numel() == 0:
        return

    min_id = int(ids.min())
    max_id = int(ids.max())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(
            f"{name} contains ids outside range [0, {upper}): "
            f"min={min_id}, max={max_id}."
        )


__all__ = [
    "anchor_node_mask",
    "node_mask_from_ids",
]
