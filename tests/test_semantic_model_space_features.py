from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] = torch.logaddexp(out[dest], src[row])
        return out

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_sum = _scatter_sum
    sys.modules["torch_scatter"] = torch_scatter_stub


from src.weaver.nn.action_head import StopExpandGate
from src.weaver.nn.backbone import FeatureBank, SemanticFeatureEncoder, l2_normalize
from src.weaver.nn.edge_scorer import EdgeScoreBreakdown, ExpandEdgeScorer


def test_feature_encoder_splits_semantic_and_model_space() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        question_emb=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        non_text_node_mask=torch.tensor([False, True], dtype=torch.bool),
    )
    encoder = SemanticFeatureEncoder(
        entity_text_embeddings=torch.tensor([[3.0, 4.0], [5.0, 12.0]]),
        entity_embedding_map=torch.tensor([0, -1], dtype=torch.long),
        relation_embeddings=torch.tensor([[8.0, 15.0]]),
        embedding_dim=2,
        hidden_dim=2,
        normalize=True,
    )

    fb = encoder(batch)

    assert torch.allclose(
        fb.node_sem_h[0], l2_normalize(torch.tensor([[3.0, 4.0]])).squeeze(0)
    )
    assert torch.allclose(fb.rel_sem_h.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(fb.query_sem_h.norm(dim=-1), torch.ones(1), atol=1e-6)

    assert torch.allclose(fb.node_h, encoder.node_projection(fb.node_sem_h))
    assert torch.allclose(fb.rel_h, encoder.rel_projection(fb.rel_sem_h))
    assert torch.allclose(fb.query_h, encoder.query_projection(fb.query_sem_h))
    assert not torch.allclose(fb.query_h, fb.query_sem_h)


def test_expand_edge_prior_uses_semantic_space_not_model_space() -> None:
    fb = FeatureBank(
        node_h=torch.zeros((2, 2), dtype=torch.float32),
        rel_h=torch.zeros((1, 2), dtype=torch.float32),
        query_h=torch.zeros((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        anchor_forward_bucket=torch.tensor([1, 0], dtype=torch.long),
        anchor_backward_bucket=torch.tensor([1, 0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False], dtype=torch.bool),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    scorer = ExpandEdgeScorer(
        hidden_dim=2,
        entity_weight_init=2.0,
        logit_scale_init=3.0,
        residual_scale_init=1.0,
    )

    output = scorer(
        fb=fb,
        state_h=torch.zeros((1, 2), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        node_is_non_text=fb.node_is_non_text,
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert torch.allclose(output.q_rel, torch.tensor([0.5]))
    assert torch.allclose(output.q_new, torch.tensor([0.25]))
    assert torch.allclose(output.semantic_logits, torch.tensor([3.0]))
    assert torch.allclose(output.final_logits, output.semantic_logits)


def test_expand_edge_can_disable_residual_online_logits() -> None:
    fb = FeatureBank(
        node_h=torch.ones((2, 2), dtype=torch.float32),
        rel_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        anchor_forward_bucket=torch.tensor([1, 0], dtype=torch.long),
        anchor_backward_bucket=torch.tensor([1, 0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False], dtype=torch.bool),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    scorer = ExpandEdgeScorer(
        hidden_dim=2,
        entity_weight_init=2.0,
        logit_scale_init=3.0,
        residual_scale_init=1.0,
        use_residual=False,
    )
    with torch.no_grad():
        scorer.residual_head[-1].bias.fill_(7.0)

    output = scorer(
        fb=fb,
        state_h=torch.ones((1, 2), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        node_is_non_text=fb.node_is_non_text,
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert torch.allclose(output.semantic_logits, torch.tensor([3.0]))
    assert torch.allclose(output.residual_logits, torch.zeros(1))
    assert torch.allclose(output.final_logits, output.semantic_logits)


def test_stop_gate_without_frontier_summary_uses_progress_only() -> None:
    gate = StopExpandGate(hidden_dim=2, use_frontier_summary=False)
    with torch.no_grad():
        first = gate.gate[0]
        last = gate.gate[-1]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        first.weight[0, 4] = 1.0
        last.weight[0, 0] = 1.0

    kwargs = {
        "query_h": torch.zeros((1, 2), dtype=torch.float32),
        "state_h": torch.zeros((1, 2), dtype=torch.float32),
        "progress_ratio": torch.tensor([0.5], dtype=torch.float32),
        "has_candidate_edge": torch.tensor([True]),
    }

    stop_a, expand_a = gate(
        **kwargs,
        edge_logmeanexp=torch.tensor([100.0], dtype=torch.float32),
        edge_max=torch.tensor([120.0], dtype=torch.float32),
    )
    stop_b, expand_b = gate(
        **kwargs,
        edge_logmeanexp=torch.tensor([-100.0], dtype=torch.float32),
        edge_max=torch.tensor([-80.0], dtype=torch.float32),
    )

    assert torch.allclose(stop_a, stop_b)
    assert torch.allclose(expand_a, expand_b)
    assert torch.allclose(stop_a, torch.nn.functional.gelu(torch.tensor([0.5])))
    assert torch.allclose(expand_a, torch.zeros(1))
