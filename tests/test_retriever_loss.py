import torch

from src.losses.retriever_loss import RetrieverBCELoss, RetrieverListwiseHardNegLoss
from src.models.components.retriever import RetrieverOutput


def test_retriever_bce_loss_supports_graph_normalize():
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    output = RetrieverOutput(
        scores=torch.sigmoid(logits),
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverBCELoss(pos_weight=1.0)
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=2)

    per_edge = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = torch.stack([per_edge[:2].mean(), per_edge[2:].mean()]).mean()
    assert torch.allclose(out.loss, expected)


def test_retriever_listwise_hardneg_loss_matches_manual():
    logits = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    output = RetrieverOutput(
        scores=torch.sigmoid(logits),
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverListwiseHardNegLoss(
        temperature=1.0,
        listwise_weight=1.0,
        hard_neg_k=1,
        pairwise_margin=0.0,
        pairwise_weight=1.0,
    )
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=2)

    logits_g0 = logits[:3]
    logits_g1 = logits[3:]
    listwise_g0 = -(torch.logsumexp(logits_g0[targets[:3] > 0.5], dim=0) - torch.logsumexp(logits_g0, dim=0))
    listwise_g1 = -(torch.logsumexp(logits_g1[targets[3:] > 0.5], dim=0) - torch.logsumexp(logits_g1, dim=0))
    expected_listwise = (listwise_g0 + listwise_g1) / 2.0

    pairwise_g0 = torch.nn.functional.softplus(logits_g0[1] - logits_g0[0])
    pairwise_g1 = torch.nn.functional.softplus(logits_g1[0] - logits_g1[1])
    expected_pairwise = (pairwise_g0 + pairwise_g1) / 2.0

    expected_total = expected_listwise + expected_pairwise
    assert torch.allclose(out.loss, expected_total, atol=1e-6)


def test_retriever_listwise_hardneg_loss_handles_no_positive_graphs():
    logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 0.0], dtype=torch.float32)
    edge_batch = torch.tensor([0, 0], dtype=torch.long)
    output = RetrieverOutput(
        scores=torch.sigmoid(logits),
        logits=logits,
        query_ids=edge_batch,
        relation_ids=None,
    )

    loss_fn = RetrieverListwiseHardNegLoss(
        temperature=1.0,
        listwise_weight=1.0,
        hard_neg_k=1,
        pairwise_margin=1.0,
        pairwise_weight=1.0,
    )
    out = loss_fn(output, targets, edge_batch=edge_batch, num_graphs=1)
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, torch.tensor(0.0, dtype=out.loss.dtype))
