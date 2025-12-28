import torch
import torch.nn.functional as F
from src.losses.retriever_loss import RetrieverLoss
from src.models.components.retriever import RetrieverOutput


def test_retriever_composite_loss_infonce_only():
    logits = torch.tensor([0.2, 0.4, -0.3, 0.1], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    output = RetrieverOutput(
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverLoss(infonce_temperature=1.0)
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=1)

    scores = logits
    expected = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[targets > 0.5], dim=0)
    assert torch.allclose(out.loss, expected)


def test_retriever_loss_supports_bce_only():
    logits = torch.tensor([0.2, -0.1, 0.5, -0.2], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    output = RetrieverOutput(
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverLoss(infonce_weight=0.0, bce_weight=1.0)
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=2)

    per_edge = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = torch.stack([per_edge[:2].mean(), per_edge[2:].mean()]).mean()
    assert torch.allclose(out.loss, expected)
