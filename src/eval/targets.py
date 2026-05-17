from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.graph.masks import node_mask_from_ids


def eval_target_node_mask(
    batch: RetrievalBatch,
    *,
    device: torch.device,
    use_reachable_targets: bool = True,
) -> torch.Tensor:
    """
    Build target-node mask for retrieval evaluation.

    If use_reachable_targets=True and reachable_target_node_ids exists, metrics
    are computed over reachable / teachable targets even when that tensor is
    empty. Otherwise metrics use all target_node_ids present in the graph.
    """
    if use_reachable_targets:
        reachable = getattr(batch, "reachable_target_node_ids", None)
        if isinstance(reachable, torch.Tensor):
            return node_mask_from_ids(
                reachable,
                num_nodes=int(batch.num_nodes_total),
                device=device,
                name="reachable_target_node_ids",
            )

    return node_mask_from_ids(
        batch.target_node_ids,
        num_nodes=int(batch.num_nodes_total),
        device=device,
        name="target_node_ids",
    )


__all__ = ["eval_target_node_mask"]
