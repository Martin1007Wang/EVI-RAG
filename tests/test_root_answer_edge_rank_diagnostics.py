from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

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
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
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
        out = torch.full((size,), -float("inf"), dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

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
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.training.diagnostics import TrainingDiagnosticsCollector
from src.training.rollout_diagnostics import compute_root_answer_edge_ranking_diagnostics
from src.weaver.nn.edge_scorer import EdgeScoreBreakdown
from src.weaver.policy import PolicyOutput


class _Batch:
    num_graphs = 2

    def __init__(self) -> None:
        self.edge_index = torch.tensor(
            [
                [0, 0, 3],
                [2, 1, 5],
            ],
            dtype=torch.long,
        )
        self.anchor_node_ids = torch.tensor([0, 3], dtype=torch.long)
        self.target_node_ids = torch.tensor([1, 4], dtype=torch.long)
        self.reachable_target_node_ids = torch.tensor([1, 4], dtype=torch.long)

    @property
    def num_nodes_total(self) -> int:
        return 6


class _Policy:
    def prepare_rollout_context(self, batch: _Batch) -> object:
        del batch
        return object()

    def __call__(
        self,
        batch: _Batch,
        state: object,
        *,
        rollout_context: object,
        return_edge_breakdown: bool = False,
    ) -> PolicyOutput:
        del batch, state, rollout_context, return_edge_breakdown
        return PolicyOutput(
            state_log_flow=torch.zeros(2),
            stop_logits=torch.zeros(2),
            expand_logits=torch.zeros(2),
            edge_logits=torch.tensor([4.0, 3.0, 8.0], dtype=torch.float32),
            candidate_batch_ids=torch.tensor([0, 0, 1], dtype=torch.long),
            candidate_edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            edge_score_breakdown=EdgeScoreBreakdown(
                query_relation_score=torch.tensor([0.1, 0.9, 0.3], dtype=torch.float32),
                query_new_node_score=torch.tensor([0.2, 0.8, 0.4], dtype=torch.float32),
                semantic_score=torch.tensor([0.3, 0.7, 0.5], dtype=torch.float32),
                new_text_mask=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
                semantic_logits=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                residual_logits=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                final_logits=torch.tensor([4.0, 3.0, 8.0], dtype=torch.float32),
                residual_scale=torch.tensor(0.0, dtype=torch.float32),
            ),
        )


def test_root_answer_edge_rank_excludes_missing_graphs() -> None:
    metrics = compute_root_answer_edge_ranking_diagnostics(
        _Policy(),
        batch=_Batch(),  # type: ignore[arg-type]
        expand_budget=3,
    )

    assert metrics["root/frontier_answer_edge_rate"] == pytest.approx(0.5)
    assert metrics["root/frontier_answer_edge_count_mean"] == pytest.approx(0.5)
    assert metrics["root/frontier_candidate_count_mean"] == pytest.approx(1.5)
    assert metrics["root/prior_answer_edge_best_rank_mean"] == pytest.approx(1.0)
    assert metrics["root/prior_answer_edge_best_rank_median"] == pytest.approx(1.0)
    assert metrics["root/prior_answer_edge_top1_rate"] == pytest.approx(1.0)
    assert metrics["root/prior_answer_edge_top5_rate"] == pytest.approx(1.0)
    assert metrics["root/prior_answer_edge_mrr"] == pytest.approx(1.0)
    assert metrics["root/policy_answer_edge_best_rank_mean"] == pytest.approx(2.0)
    assert metrics["root/policy_answer_edge_best_rank_median"] == pytest.approx(2.0)
    assert metrics["root/policy_answer_edge_top1_rate"] == pytest.approx(0.0)
    assert metrics["root/policy_answer_edge_top5_rate"] == pytest.approx(1.0)
    assert metrics["root/policy_answer_edge_mrr"] == pytest.approx(0.5)
    assert metrics["root/answer_edge_rank_delta_mean"] == pytest.approx(1.0)
    assert metrics["root/final_worse_than_prior_rate"] == pytest.approx(1.0)
    assert metrics["edge/base_logit_std"] == pytest.approx(
        float(torch.tensor([1.0, 2.0, 3.0]).std(unbiased=False).item())
    )
    assert metrics["edge/residual_logit_std"] == pytest.approx(0.0)
    assert metrics["edge/residual_to_base_std_ratio"] == pytest.approx(0.0)
    assert metrics["edge/prior_rank_vs_final_rank_kendall"] == pytest.approx(-1.0)
    assert metrics["edge/answer_edge_prior_rank"] == pytest.approx(1.0)
    assert metrics["edge/answer_edge_final_rank"] == pytest.approx(2.0)

    assert metrics["root/answer_edge_q_rel_mean"] == pytest.approx(0.9)
    assert metrics["root/answer_edge_q_new_mean"] == pytest.approx(0.8)
    assert metrics["root/answer_edge_q_candidate_mean"] == pytest.approx(0.7)
    assert metrics["root/answer_edge_new_text_rate"] == pytest.approx(1.0)
    assert metrics["root/answer_edge_logit_mean"] == pytest.approx(3.0)
    assert metrics["root/nonanswer_edge_q_rel_mean"] == pytest.approx(0.2)
    assert metrics["root/nonanswer_edge_q_new_mean"] == pytest.approx(0.3)
    assert metrics["root/nonanswer_edge_q_candidate_mean"] == pytest.approx(0.4)
    assert metrics["root/nonanswer_edge_new_text_rate"] == pytest.approx(0.5)
    assert metrics["root/nonanswer_edge_logit_mean"] == pytest.approx(6.0)


def test_training_policy_diagnostics_emit_root_edge_prior_metrics() -> None:
    collector = TrainingDiagnosticsCollector(
        debug=False,
        rollout_diagnostics=False,
        policy_diagnostics=True,
    )

    metrics = collector.collect(
        loss_output=types.SimpleNamespace(
            loss=torch.tensor(1.0),
            metrics={"loss/total": torch.tensor(1.0)},
        ),
        batch=_Batch(),  # type: ignore[arg-type]
        policy=_Policy(),
    )

    assert metrics["train/loss/total"] == pytest.approx(1.0)
    assert metrics["train/edge/base_logit_std"] > 0.0
    assert metrics["train/edge/residual_logit_std"] == pytest.approx(0.0)
    assert metrics["train/edge/answer_edge_prior_rank"] == pytest.approx(1.0)
    assert metrics["train/edge/answer_edge_final_rank"] == pytest.approx(2.0)
