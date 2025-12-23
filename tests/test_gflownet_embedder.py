from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.components.gflownet_embedder import GraphEmbedder


def test_graph_embedder_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires retriever ckpt"):
        GraphEmbedder(hidden_dim=4, proj_dropout=0.0, projector_checkpoint=None)


def test_graph_embedder_edge_batch_bucketize_respects_ptr_boundaries() -> None:
    g0 = Data(num_nodes=2, edge_index=torch.tensor([[1], [0]], dtype=torch.long))
    # Head index 0 in the second graph becomes exactly ptr[1] after batching (a boundary case).
    g1 = Data(num_nodes=2, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    batch = Batch.from_data_list([g0, g1])

    device = torch.device("cpu")
    edge_index = batch.edge_index.to(device)
    node_ptr = batch.ptr.to(device)
    num_graphs = int(node_ptr.numel() - 1)

    edge_batch, edge_ptr = GraphEmbedder._compute_edge_batch(
        edge_index,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
        device=device,
    )
    expected_edge_batch = batch.batch.to(device)[edge_index[0]]
    assert torch.equal(edge_batch, expected_edge_batch)
    assert edge_ptr.detach().cpu().tolist() == [0, 1, 2]
