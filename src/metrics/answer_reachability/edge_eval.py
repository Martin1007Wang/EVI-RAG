from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Callable, Iterable

import torch

from src.data.preprocess.labels.edge_retrieval import compute_shortest_path_labels
from src.graph_runtime import TrajectoryBatch
from src.models.configs import SearchEvalConfig
from src.models.gflownet import StartDistributionError
from src.models.gflownet import SearchPolicyProtocol
from src.metrics.protocol import MetricEvaluationOutput
from src.utils.metrics_io import to_serializable

from .exact_analysis import ExactEdgeSupportAnalysis, ExactReachabilityAnalyzer


@dataclass(frozen=True)
class EdgePredictionRecord:
    edge_id: int
    src_entity_id: int
    relation_id: int
    dst_entity_id: int
    score: float
    conditional_score: float
    is_positive: bool


@dataclass(frozen=True)
class EdgeRetrievalResult:
    sample_id: str
    dataset_scope: str
    num_edges: int
    num_positive_edges: int
    max_path_length: int | None
    gold_total_mass: float
    first_positive_rank: int | None
    positive_edge_ids: list[int] = field(default_factory=list)
    ranked_edge_ids: list[int] = field(default_factory=list)
    ranked_edges: list[EdgePredictionRecord] = field(default_factory=list)


@dataclass(frozen=True)
class EdgeRetrievalLabelRecord:
    sample_id: str
    question: str
    num_edges: int
    positive_edge_ids: list[int]
    max_path_length: int | None


@dataclass(frozen=True)
class EdgeRetrievalLabels:
    num_edges: int
    positive_edge_ids: torch.Tensor
    max_path_length: int | None


@dataclass(frozen=True)
class PreparedEdgeRetrievalGraph:
    batch: TrajectoryBatch
    edge_support: ExactEdgeSupportAnalysis
    labels: EdgeRetrievalLabels
    invalid_start: bool = False


def compute_edge_retrieval_labels(*, batch: TrajectoryBatch) -> EdgeRetrievalLabels:
    labels = compute_shortest_path_labels(
        edge_index=torch.as_tensor(batch.edge_index, dtype=torch.long),
        q_local_indices=torch.as_tensor(batch.q_local_indices, dtype=torch.long),
        a_local_indices=torch.as_tensor(batch.a_local_indices, dtype=torch.long),
        num_nodes=int(batch.num_nodes_total),
    )
    return EdgeRetrievalLabels(
        num_edges=int(labels.num_edges),
        positive_edge_ids=labels.positive_edge_ids.to(dtype=torch.long),
        max_path_length=labels.max_path_length,
    )


def compute_edge_metrics(
    *,
    results: Iterable[EdgeRetrievalResult],
    edge_top_ks: tuple[int, ...],
) -> dict[str, float]:
    result_list = list(results)
    if not result_list:
        return {}
    metrics: dict[str, float] = {
        "edge/mrr": 0.0,
        "edge/positive_count": 0.0,
        "edge/no_path_rate": 0.0,
        "edge/zero_hop_rate": 0.0,
        "edge/gold_mass": 0.0,
    }
    for k in edge_top_ks:
        metrics[f"edge/hit@{k}"] = 0.0
        metrics[f"edge/precision@{k}"] = 0.0
        metrics[f"edge/recall@{k}"] = 0.0
    for result in result_list:
        positive_ids = set(result.positive_edge_ids)
        ranked_ids = result.ranked_edge_ids
        num_positive = len(positive_ids)
        metrics["edge/positive_count"] += float(num_positive)
        metrics["edge/no_path_rate"] += float(result.max_path_length is None)
        metrics["edge/zero_hop_rate"] += float(result.max_path_length == 0)
        metrics["edge/gold_mass"] += float(result.gold_total_mass)
        if result.first_positive_rank is not None:
            metrics["edge/mrr"] += 1.0 / float(result.first_positive_rank)
        for k in edge_top_ks:
            top_ids = ranked_ids[: int(k)]
            hits = sum(1 for edge_id in top_ids if edge_id in positive_ids)
            metrics[f"edge/hit@{k}"] += float(hits > 0)
            denom = max(1, len(top_ids))
            metrics[f"edge/precision@{k}"] += float(hits) / float(denom)
            recall = 0.0 if num_positive == 0 else float(hits) / float(num_positive)
            metrics[f"edge/recall@{k}"] += recall
    total = float(len(result_list))
    return {name: value / total for name, value in metrics.items()}


class EdgeRetrievalEvaluator:
    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
        analyzer: ExactReachabilityAnalyzer,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.analyzer = analyzer

    @staticmethod
    def _empty_edge_support(*, batch: TrajectoryBatch) -> ExactEdgeSupportAnalysis:
        num_edges = int(batch.edge_index.size(1))
        zeros = torch.zeros(
            (num_edges,), device=batch.edge_index.device, dtype=torch.float32
        )
        return ExactEdgeSupportAnalysis(
            edge_success_mass=zeros,
            edge_conditional_success_prob=zeros,
            gold_mass=0.0,
        )

    def _prepare_graph(
        self,
        *,
        batch: TrajectoryBatch,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> PreparedEdgeRetrievalGraph:
        prepared_batch = self.policy.prepare_batch(batch)
        labels = compute_edge_retrieval_labels(batch=batch)
        try:
            edge_support = self.analyzer.analyze_edge_support(
                batch=batch,
                policy=self.policy,
                prepared_batch=prepared_batch,
            )
        except StartDistributionError:
            if on_invalid_start is not None:
                on_invalid_start(batch)
            return PreparedEdgeRetrievalGraph(
                batch=batch,
                edge_support=self._empty_edge_support(batch=batch),
                labels=labels,
                invalid_start=True,
            )
        return PreparedEdgeRetrievalGraph(
            batch=batch,
            edge_support=edge_support,
            labels=labels,
        )

    @staticmethod
    def _result_sample_id(batch: TrajectoryBatch) -> str:
        sample_ids = getattr(batch, "sample_ids", None)
        if sample_ids:
            return str(sample_ids[0])
        sample_id = getattr(batch, "sample_id", "")
        return str(sample_id or "")

    def _build_result(self, graph: PreparedEdgeRetrievalGraph) -> EdgeRetrievalResult:
        batch = graph.batch
        labels = graph.labels
        edge_scores = graph.edge_support.edge_success_mass
        conditional_scores = graph.edge_support.edge_conditional_success_prob
        edge_ids = list(range(int(batch.edge_index.size(1))))
        edge_type_values = getattr(batch, "edge_rel_global", None)
        if edge_type_values is None:
            edge_type_values = getattr(batch, "edge_attr", None)
        if edge_type_values is None:
            raise AttributeError(
                "TrajectoryBatch must define edge_rel_global for edge retrieval ranking."
            )
        edge_types = torch.as_tensor(edge_type_values, dtype=torch.long)
        positive_ids = set(
            int(edge_id) for edge_id in labels.positive_edge_ids.tolist()
        )
        ranked_ids = sorted(
            edge_ids,
            key=lambda edge_id: (
                -float(conditional_scores[edge_id].item()),
                -float(edge_scores[edge_id].item()),
                edge_id,
            ),
        )
        first_positive_rank = next(
            (
                rank
                for rank, edge_id in enumerate(ranked_ids, start=1)
                if edge_id in positive_ids
            ),
            None,
        )
        emit_top_k = min(int(self.eval_cfg.edge_emit_top_k), len(ranked_ids))
        ranked_edges = [
            EdgePredictionRecord(
                edge_id=edge_id,
                src_entity_id=int(
                    batch.node_global_ids[
                        int(batch.edge_index[0, edge_id].item())
                    ].item()
                ),
                relation_id=int(edge_types[edge_id].item()),
                dst_entity_id=int(
                    batch.node_global_ids[
                        int(batch.edge_index[1, edge_id].item())
                    ].item()
                ),
                score=float(edge_scores[edge_id].item()),
                conditional_score=float(conditional_scores[edge_id].item()),
                is_positive=edge_id in positive_ids,
            )
            for edge_id in ranked_ids[:emit_top_k]
        ]
        return EdgeRetrievalResult(
            sample_id=self._result_sample_id(batch),
            dataset_scope=str(batch.dataset_scope),
            num_edges=int(labels.num_edges),
            num_positive_edges=int(labels.positive_edge_ids.numel()),
            max_path_length=labels.max_path_length,
            gold_total_mass=float(graph.edge_support.gold_mass),
            first_positive_rank=first_positive_rank,
            positive_edge_ids=[
                int(edge_id) for edge_id in labels.positive_edge_ids.tolist()
            ],
            ranked_edge_ids=[int(edge_id) for edge_id in ranked_ids],
            ranked_edges=ranked_edges,
        )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        del metrics_profile, include_answer_support
        results: list[EdgeRetrievalResult] = []
        invalid_start = 0
        for graph_idx in range(batch.num_graphs):
            graph = self._prepare_graph(
                batch=batch.select_graph(graph_idx),
                on_invalid_start=on_invalid_start,
            )
            invalid_start += int(graph.invalid_start)
            results.append(self._build_result(graph))
        return MetricEvaluationOutput(
            model_metrics={
                "invalid_start_rate": float(invalid_start)
                / float(max(1, len(results))),
                "graph_count": float(len(results)),
            },
            primary_metrics=compute_edge_metrics(
                results=results,
                edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
            ),
            secondary_metrics={},
            results=results,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[EdgeRetrievalResult]:
        return self.evaluate_batch(
            batch=batch,
            metrics_profile=metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=on_invalid_start,
        ).results

    @staticmethod
    def build_predict_labels(
        batch: TrajectoryBatch,
        outputs: list[EdgeRetrievalResult],
    ) -> list[EdgeRetrievalLabelRecord]:
        if len(outputs) != batch.num_graphs:
            raise ValueError(
                "Predict outputs must align with TrajectoryBatch graph count. "
                f"outputs={len(outputs)} num_graphs={batch.num_graphs}."
            )
        labels: list[EdgeRetrievalLabelRecord] = []
        for graph_idx, result in enumerate(outputs):
            graph_batch = batch.select_graph(graph_idx)
            graph_labels = compute_edge_retrieval_labels(batch=graph_batch)
            labels.append(
                EdgeRetrievalLabelRecord(
                    sample_id=result.sample_id,
                    question=str(graph_batch.questions[0]),
                    num_edges=int(graph_labels.num_edges),
                    positive_edge_ids=[
                        int(edge_id)
                        for edge_id in graph_labels.positive_edge_ids.tolist()
                    ],
                    max_path_length=graph_labels.max_path_length,
                )
            )
        return labels

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[EdgeRetrievalResult],
        metrics_profile: str,
    ) -> dict[str, float]:
        del metrics_profile
        return compute_edge_metrics(
            results=predict_results,
            edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
        )

    def write_prediction_artifacts(
        self,
        *,
        results: list[EdgeRetrievalResult],
        labels: list[EdgeRetrievalLabelRecord],
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        del schema_version, entity_vocab_path, relation_vocab_path, questions_path
        if not results:
            return None
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_path = output_path / f"{artifact_name}.{split}.results.jsonl"
        labels_path = output_path / f"{artifact_name}.{split}.labels.jsonl"
        if not overwrite:
            for path in (results_path, labels_path):
                if path.exists():
                    raise FileExistsError(f"Artifact already exists: {path}")
        self._write_jsonl(results_path, [asdict(result) for result in results])
        self._write_jsonl(labels_path, [asdict(label) for label in labels])
        return {"results_path": results_path, "labels_path": labels_path}

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(to_serializable(record), ensure_ascii=True) + "\n"
                )


__all__ = [
    "EdgePredictionRecord",
    "EdgeRetrievalEvaluator",
    "EdgeRetrievalLabelRecord",
    "EdgeRetrievalLabels",
    "EdgeRetrievalResult",
    "PreparedEdgeRetrievalGraph",
    "compute_edge_metrics",
    "compute_edge_retrieval_labels",
]
