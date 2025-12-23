import torch

from src.losses.retriever_loss import RetrieverReachabilityLoss
from src.models.components.retriever import RetrieverOutput


def test_retriever_reachability_loss_listwise_only():
    logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 0], dtype=torch.long)
    output = RetrieverOutput(
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverReachabilityLoss(
        temperature=1.0,
        listwise_weight=1.0,
        bce_weight=0.0,
        soft_label_power=1.0,
    )
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=1)

    expected = -(
        torch.logsumexp(logits[targets > 0.5], dim=0)
        - torch.logsumexp(logits, dim=0)
    )
    assert torch.allclose(out.loss, expected)


def test_retriever_reachability_loss_bce_only_matches_graph_mean():
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    output = RetrieverOutput(
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverReachabilityLoss(
        temperature=1.0,
        listwise_weight=0.0,
        bce_weight=1.0,
        soft_label_power=1.0,
    )
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=2)

    per_edge = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = torch.stack([per_edge[:2].mean(), per_edge[2:].mean()]).mean()
    assert torch.allclose(out.loss, expected)
