from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, cast

import torch
from torchmetrics import MeanMetric

from src.data.preprocess.labels.edge_retrieval import compute_shortest_path_labels
from src.graph import TrajectoryBatch
from src.models.configs import SearchEvalConfig
from src.models.gflownet import (
    RootActionDistributionError,
    SearchPolicyProtocol,
    TrajectorySamplerProtocol,
)
from src.utils.metrics_io import to_serializable

from .base import BaseMetricRuntime
from .protocol import MetricEvaluationOutput
from .prediction_io import PredictionCodecProtocol
from .ranking_metrics import (
    compute_topk_set_metrics,
    mean_metric_dicts,
    reciprocal_rank,
)
from .search_backends import MonteCarloBackend


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
    success_rollout_mass: float
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
    num_edges: int
    positive_edge_ids: torch.Tensor
    max_path_length: int | None
    invalid_start: bool
    batch: TrajectoryBatch
    edge_success_mass: torch.Tensor
    edge_conditional_success_prob: torch.Tensor
    success_rollout_mass: float


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_edge_prediction_record(record: dict[str, Any]) -> EdgePredictionRecord:
    return EdgePredictionRecord(
        edge_id=int(record.get("edge_id", 0)),
        src_entity_id=int(record.get("src_entity_id", 0)),
        relation_id=int(record.get("relation_id", 0)),
        dst_entity_id=int(record.get("dst_entity_id", 0)),
        score=float(record.get("score", 0.0)),
        conditional_score=float(record.get("conditional_score", 0.0)),
        is_positive=bool(record.get("is_positive", False)),
    )


def load_edge_retrieval_result(record: dict[str, Any]) -> EdgeRetrievalResult:
    return EdgeRetrievalResult(
        sample_id=str(record.get("sample_id", "")),
        dataset_scope=str(record.get("dataset_scope", "")),
        num_edges=int(record.get("num_edges", 0)),
        num_positive_edges=int(record.get("num_positive_edges", 0)),
        max_path_length=_optional_int(record.get("max_path_length")),
        success_rollout_mass=float(
            record.get("success_rollout_mass", record.get("gold_total_mass", 0.0))
        ),
        first_positive_rank=_optional_int(record.get("first_positive_rank")),
        positive_edge_ids=[
            int(value) for value in record.get("positive_edge_ids") or []
        ],
        ranked_edge_ids=[int(value) for value in record.get("ranked_edge_ids") or []],
        ranked_edges=[
            load_edge_prediction_record(edge)
            for edge in record.get("ranked_edges") or []
        ],
    )


def load_edge_retrieval_label(record: dict[str, Any]) -> EdgeRetrievalLabelRecord:
    return EdgeRetrievalLabelRecord(
        sample_id=str(record.get("sample_id", "")),
        question=str(record.get("question", "")),
        num_edges=int(record.get("num_edges", 0)),
        positive_edge_ids=[
            int(value) for value in record.get("positive_edge_ids") or []
        ],
        max_path_length=_optional_int(record.get("max_path_length")),
    )


class EdgeRetrievalPredictionCodec(PredictionCodecProtocol):
    kind = "edge_retrieval"

    def serialize_result(self, result: EdgeRetrievalResult) -> dict[str, Any]:
        return asdict(cast(Any, result))

    def serialize_label(self, label: EdgeRetrievalLabelRecord) -> dict[str, Any]:
        return asdict(cast(Any, label))

    def deserialize_result(self, record: dict[str, Any]) -> EdgeRetrievalResult:
        return load_edge_retrieval_result(record)

    def deserialize_label(self, record: dict[str, Any]) -> EdgeRetrievalLabelRecord:
        return load_edge_retrieval_label(record)


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
    per_graph_metrics: list[dict[str, float]] = []
    for result in result_list:
        positive_ids = set(result.positive_edge_ids)
        metrics = {
            "edge/mrr": reciprocal_rank(result.first_positive_rank),
            "edge/success_rollout_mass": float(result.success_rollout_mass),
        }
        metrics.update(
            compute_topk_set_metrics(
                ranked_ids=result.ranked_edge_ids,
                relevant_ids=positive_ids,
                top_ks=edge_top_ks,
                prefix="edge",
                include_precision=False,
                include_f1=False,
            )
        )
        per_graph_metrics.append(metrics)
    return mean_metric_dicts(per_graph_metrics)


@dataclass
class EdgeMetricsAccumulator:
    metrics: dict[str, MeanMetric] = field(default_factory=dict)


class EdgeRetrievalRuntime(BaseMetricRuntime):
    sampler: TrajectorySamplerProtocol | None

    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
        backend: MonteCarloBackend,
        sampler: TrajectorySamplerProtocol,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.backend = backend
        self._prediction_codec = EdgeRetrievalPredictionCodec()
        self.sampler = sampler
        self.search = backend

    @staticmethod
    def _empty_graph(batch: TrajectoryBatch) -> PreparedEdgeRetrievalGraph:
        num_edges = int(batch.edge_index.size(1))
        zeros = torch.zeros(
            (num_edges,), device=batch.edge_index.device, dtype=torch.float32
        )
        labels = compute_edge_retrieval_labels(batch=batch)
        return PreparedEdgeRetrievalGraph(
            num_edges=int(labels.num_edges),
            positive_edge_ids=labels.positive_edge_ids,
            max_path_length=labels.max_path_length,
            invalid_start=True,
            batch=batch,
            edge_success_mass=zeros,
            edge_conditional_success_prob=zeros,
            success_rollout_mass=0.0,
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
            edge_support = self.backend.analyze_edge_support(
                batch=batch,
                policy=self.policy,
                prepared_batch=prepared_batch,
            )
        except RootActionDistributionError:
            if on_invalid_start is not None:
                on_invalid_start(batch)
            return self._empty_graph(batch)
        return PreparedEdgeRetrievalGraph(
            num_edges=int(labels.num_edges),
            positive_edge_ids=labels.positive_edge_ids,
            max_path_length=labels.max_path_length,
            invalid_start=False,
            batch=batch,
            edge_success_mass=edge_support.edge_success_mass,
            edge_conditional_success_prob=edge_support.edge_conditional_success_prob,
            success_rollout_mass=float(edge_support.success_rollout_mass),
        )

    def _prepare_batch_graphs(
        self,
        *,
        batch: TrajectoryBatch,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[PreparedEdgeRetrievalGraph]:
        prepared_batch = self.policy.prepare_batch(batch)
        try:
            edge_support_by_graph = self.backend.analyze_edge_support_batch(
                batch=batch,
                policy=self.policy,
                prepared_batch=prepared_batch,
            )
        except RootActionDistributionError:
            return [
                self._prepare_graph(
                    batch=batch.select_graph(graph_idx, validate=False),
                    on_invalid_start=on_invalid_start,
                )
                for graph_idx in range(batch.num_graphs)
            ]

        if len(edge_support_by_graph) != batch.num_graphs:
            raise ValueError(
                "Batched edge support analysis must align with TrajectoryBatch graph count. "
                f"analyses={len(edge_support_by_graph)} num_graphs={batch.num_graphs}."
            )

        graphs: list[PreparedEdgeRetrievalGraph] = []
        for graph_idx, edge_support in enumerate(edge_support_by_graph):
            graph_batch = batch.select_graph(graph_idx, validate=False)
            labels = compute_edge_retrieval_labels(batch=graph_batch)
            graphs.append(
                PreparedEdgeRetrievalGraph(
                    num_edges=int(labels.num_edges),
                    positive_edge_ids=labels.positive_edge_ids,
                    max_path_length=labels.max_path_length,
                    invalid_start=False,
                    batch=graph_batch,
                    edge_success_mass=edge_support.edge_success_mass,
                    edge_conditional_success_prob=(
                        edge_support.edge_conditional_success_prob
                    ),
                    success_rollout_mass=float(edge_support.success_rollout_mass),
                )
            )
        return graphs

    @staticmethod
    def _result_sample_id(batch: TrajectoryBatch) -> str:
        return str(batch.sample_ids[0]) if batch.sample_ids else ""

    def _build_result(self, graph: PreparedEdgeRetrievalGraph) -> EdgeRetrievalResult:
        batch = graph.batch
        edge_scores = graph.edge_success_mass
        conditional_scores = graph.edge_conditional_success_prob
        edge_ids = list(range(int(batch.edge_index.size(1))))
        edge_types = batch.edge_rel_global.to(dtype=torch.long)
        positive_ids = set(int(edge_id) for edge_id in graph.positive_edge_ids.tolist())
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
                    batch.node_entity_ids[
                        int(batch.edge_index[0, edge_id].item())
                    ].item()
                ),
                relation_id=int(edge_types[edge_id].item()),
                dst_entity_id=int(
                    batch.node_entity_ids[
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
            num_edges=int(graph.num_edges),
            num_positive_edges=int(graph.positive_edge_ids.numel()),
            max_path_length=graph.max_path_length,
            success_rollout_mass=float(graph.success_rollout_mass),
            first_positive_rank=first_positive_rank,
            positive_edge_ids=[
                int(edge_id) for edge_id in graph.positive_edge_ids.tolist()
            ],
            ranked_edge_ids=[int(edge_id) for edge_id in ranked_ids],
            ranked_edges=ranked_edges,
        )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        del report_profile, include_answer_support
        graphs = self._prepare_batch_graphs(
            batch=batch,
            on_invalid_start=on_invalid_start,
        )
        invalid_start = sum(int(graph.invalid_start) for graph in graphs)
        results = [self._build_result(graph) for graph in graphs]
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
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[EdgeRetrievalResult]:
        del report_profile, include_answer_support
        graphs = self._prepare_batch_graphs(
            batch=batch,
            on_invalid_start=on_invalid_start,
        )
        return [self._build_result(graph) for graph in graphs]

    def build_predict_labels(
        self,
        batch: TrajectoryBatch,
        outputs: list[EdgeRetrievalResult],
    ) -> list[EdgeRetrievalLabelRecord]:
        del self
        if len(outputs) != batch.num_graphs:
            raise ValueError(
                "Predict outputs must align with TrajectoryBatch graph count. "
                f"outputs={len(outputs)} num_graphs={batch.num_graphs}."
            )
        labels: list[EdgeRetrievalLabelRecord] = []
        for graph_idx, result in enumerate(outputs):
            graph_batch = batch.select_graph(graph_idx, validate=False)
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
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        return compute_edge_metrics(
            results=predict_results,
            edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
        )

    def initialize_predict_metrics_accumulator(self, *, report_profile: str) -> Any:
        del report_profile
        return EdgeMetricsAccumulator()

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: EdgeMetricsAccumulator,
        predict_results: list[Any],
        report_profile: str,
    ) -> None:
        del report_profile
        if not predict_results:
            return
        metrics = compute_edge_metrics(
            results=cast(list[EdgeRetrievalResult], predict_results),
            edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
        )
        batch_weight = torch.tensor(float(len(predict_results)), dtype=torch.float32)
        for name, value in metrics.items():
            metric = accumulator.metrics.get(name)
            if metric is None:
                metric = MeanMetric()
                accumulator.metrics[name] = metric
            metric.update(
                torch.tensor(float(value), dtype=torch.float32),
                weight=batch_weight,
            )

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: EdgeMetricsAccumulator,
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        return {
            name: float(metric.compute().item())
            for name, metric in accumulator.metrics.items()
        }

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
        del (
            artifact_name,
            schema_version,
            entity_vocab_path,
            relation_vocab_path,
            questions_path,
        )
        if not results:
            return None
        results_path, labels_path = self._resolve_artifact_paths(
            output_dir=output_dir,
            split=split,
        )
        if not overwrite:
            for path in (results_path, labels_path):
                if path.exists():
                    raise FileExistsError(f"Artifact already exists: {path}")
        self._write_jsonl(results_path, (asdict(result) for result in results))
        self._write_jsonl(labels_path, (asdict(label) for label in labels))
        return {"results_path": results_path, "labels_path": labels_path}

    def write_prediction_artifacts_from_jsonl(
        self,
        *,
        results_path: str | Path,
        labels_path: str | Path,
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        del (
            artifact_name,
            schema_version,
            entity_vocab_path,
            relation_vocab_path,
            questions_path,
        )
        source_results_path = Path(results_path)
        source_labels_path = Path(labels_path)
        if not source_results_path.exists() or source_results_path.stat().st_size == 0:
            return None
        target_results_path, target_labels_path = self._resolve_artifact_paths(
            output_dir=output_dir,
            split=split,
        )
        if not overwrite:
            for path in (target_results_path, target_labels_path):
                if path.exists():
                    raise FileExistsError(f"Artifact already exists: {path}")
        shutil.copyfile(source_results_path, target_results_path)
        shutil.copyfile(source_labels_path, target_labels_path)
        return {"results_path": target_results_path, "labels_path": target_labels_path}

    @staticmethod
    def _resolve_artifact_paths(
        *, output_dir: str | Path, split: str
    ) -> tuple[Path, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return (
            output_path / f"{split}.results.jsonl",
            output_path / f"{split}.labels.jsonl",
        )

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(to_serializable(record), ensure_ascii=True) + "\n"
                )
