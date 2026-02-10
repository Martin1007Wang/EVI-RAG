from __future__ import annotations

import torch

from src.models.components.backbone import CvtNodeInitializer


def test_cvt_initializer_aggregates_incident_edges() -> None:
    initializer = CvtNodeInitializer()
    node_embeddings = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32)
    relation_embeddings = torch.tensor([[0.5, -1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # 0 -> 1 (CVT node has only outgoing edges)
    node_is_cvt = torch.tensor([True, False])

    out = initializer(
        node_embeddings=node_embeddings,
        relation_embeddings=relation_embeddings,
        edge_index=edge_index,
        node_is_cvt=node_is_cvt,
    )

    expected_cvt = node_embeddings[1] + relation_embeddings[0]
    assert torch.allclose(out[0], expected_cvt)
    assert torch.allclose(out[1], node_embeddings[1])

