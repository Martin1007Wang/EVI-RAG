from __future__ import annotations

import torch


def node_log_degree(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Undirected log-degree over the local retrieval graph.

    This is static for one RetrievalBatch, so callers should prefer caching it
    in FeatureBank and only use this helper as a fallback.
    """
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
        )

    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, device=edge_index.device, dtype=dtype)

    node_ids = torch.cat([edge_index[0], edge_index[1]], dim=0)
    degree = torch.bincount(node_ids, minlength=num_nodes).to(dtype=dtype)
    return torch.log1p(degree)


def edge_relation_log_frequency(
    *,
    relation_ids: torch.Tensor,
    edge_batch: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Per-edge log frequency of its relation inside the same graph.
    """
    relation_ids = relation_ids.to(dtype=torch.long).view(-1)
    edge_batch = edge_batch.to(device=relation_ids.device, dtype=torch.long).view(-1)

    if relation_ids.numel() != edge_batch.numel():
        raise ValueError(
            "relation_ids and edge_batch length mismatch: "
            f"{relation_ids.numel()} != {edge_batch.numel()}."
        )

    if relation_ids.numel() == 0:
        return torch.zeros((0,), device=relation_ids.device, dtype=dtype)

    pairs = torch.stack([edge_batch, relation_ids], dim=-1)
    _, inverse, counts = torch.unique(
        pairs,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    return torch.log1p(counts.to(device=relation_ids.device, dtype=dtype)).index_select(
        0,
        inverse,
    )


__all__ = ["edge_relation_log_frequency", "node_log_degree"]
