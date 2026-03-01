from __future__ import annotations

import torch

from src.models.components.gnn import RelationalGNNLayer


def test_pna_aggregate_matches_dense_reference() -> None:
    torch.manual_seed(11)
    hidden_dim = 8
    num_nodes = 5
    num_edges = 12
    layer = RelationalGNNLayer(hidden_dim=hidden_dim, dropout=0.0)
    messages = torch.randn((num_edges, hidden_dim), dtype=torch.float32)
    tails = torch.randint(low=0, high=num_nodes, size=(num_edges,), dtype=torch.long)

    optimized = layer._pna_aggregate(messages=messages, tails=tails, num_nodes=num_nodes)

    stats, deg, has_in = layer._safe_pna_stats(messages=messages, tails=tails, num_nodes=num_nodes)
    log_deg = torch.log(deg + 1.0).clamp(min=1.0e-6)
    scales = torch.stack(
        (
            torch.ones_like(log_deg),
            log_deg / layer.delta,
            layer.delta / log_deg,
        ),
        dim=-1,
    )
    scaled = stats.unsqueeze(1) * scales.unsqueeze(-1)
    features = scaled.reshape(num_nodes, -1)
    features = torch.where(has_in.unsqueeze(-1), features, torch.zeros_like(features))
    reference = layer.agg_proj(features)

    assert torch.allclose(optimized, reference, atol=1.0e-6, rtol=1.0e-6)
