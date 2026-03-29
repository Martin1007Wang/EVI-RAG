from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import torch

from src.graph import TrajectoryBatch
from src.models.configs import SearchEvalConfig
from src.models.gflownet.subgraph.answers import resolve_subgraph_answer_entities
from src.models.gflownet.subgraph.policy import SubgraphPolicy
from src.models.gflownet.subgraph.sampler import SubgraphSampler
from src.models.gflownet.subgraph.search import beam_search_subgraphs
from src.models.gflownet.subgraph.state import SubgraphAnalysis

from .base import BaseMetricRuntime
from .protocol import MetricEvaluationOutput


class _SubgraphPredictionCodec:
    kind = "subgraph_answer_search"

    @staticmethod
    def serialize_result(result: Any) -> dict[str, Any]:
        return dict(result)

    @staticmethod
    def serialize_label(label: Any) -> dict[str, Any]:
        return dict(label)

    @staticmethod
    def deserialize_result(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    @staticmethod
    def deserialize_label(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)


def _logaddexp_pair(lhs: float | None, rhs: float) -> float:
    if lhs is None:
        return float(rhs)
    return float(
        torch.logaddexp(
            torch.tensor(float(lhs), dtype=torch.float32),
            torch.tensor(float(rhs), dtype=torch.float32),
        ).item()
    )


def _topk_metrics(
    *, predicted_entities: list[int], gold_entities: list[int], top_ks: tuple[int, ...]
) -> dict[str, float]:
    gold_set = {int(entity_id) for entity_id in gold_entities}
    if not gold_set:
        return {
            **{f"answer/hit@{int(k)}": 0.0 for k in top_ks},
            **{f"answer/recall@{int(k)}": 0.0 for k in top_ks},
        }
    metrics: dict[str, float] = {}
    for k in top_ks:
        top_predicted = predicted_entities[: int(k)]
        hits = gold_set.intersection(top_predicted)
        metrics[f"answer/hit@{int(k)}"] = 1.0 if hits else 0.0
        metrics[f"answer/recall@{int(k)}"] = float(len(hits)) / float(len(gold_set))
    return metrics


def _mean_metric_dict(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in metric_rows))
        / float(len(metric_rows))
        for key in keys
    }


def _split_terminal_answer_log_mass(
    *,
    log_mass: float,
    answer_entities: tuple[int, ...],
) -> float | None:
    if not answer_entities:
        return None
    return float(log_mass - math.log(float(len(answer_entities))))


class SubgraphAnswerSearchRuntime(BaseMetricRuntime):
    sampler: Any

    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        policy: SubgraphPolicy,
        sampler: SubgraphSampler,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.sampler = sampler
        self.search = self
        self._prediction_codec = _SubgraphPredictionCodec()

    def _predict_single_graph(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> dict[str, Any]:
        prepared_batch = self.policy.prepare_batch(batch)
        search_result = beam_search_subgraphs(
            policy=self.policy,
            eval_cfg=self.eval_cfg,
            prepared_batch=prepared_batch,
        )
        terminal_subgraphs = search_result.terminal_subgraphs
        answer_log_masses: dict[int, float] = {}
        for terminal_subgraph in terminal_subgraphs:
            answer_entities = resolve_subgraph_answer_entities(
                prepared_batch=prepared_batch,
                graph_idx=0,
                analysis=SubgraphAnalysis(
                    selected_node_ids=terminal_subgraph.selected_node_ids,
                    reachability_bits=dict(terminal_subgraph.reachability_bits),
                    component_labels={},
                    anchor_component_count=0,
                    num_selected_edges=len(terminal_subgraph.edge_ids),
                ),
            )
            per_answer_log_mass = _split_terminal_answer_log_mass(
                log_mass=float(terminal_subgraph.log_mass),
                answer_entities=answer_entities,
            )
            if per_answer_log_mass is None:
                continue
            for entity_id in answer_entities:
                answer_log_masses[int(entity_id)] = _logaddexp_pair(
                    answer_log_masses.get(int(entity_id)),
                    float(per_answer_log_mass),
                )
        ranked_answers = sorted(
            answer_log_masses.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        gold_answers = [
            int(value) for value in batch.answer_entity_ids.detach().cpu().tolist()
        ]
        top_subgraph = terminal_subgraphs[0] if terminal_subgraphs else None
        result: dict[str, Any] = {
            "sample_id": str(batch.sample_ids[0]),
            "question": str(batch.questions[0]),
            "gold_answer_entity_ids": gold_answers,
            "predicted_answer_entity_ids": [
                int(entity_id) for entity_id, _ in ranked_answers
            ],
            "answer_log_masses": [float(score) for _, score in ranked_answers],
            "terminal_subgraph_count": int(len(terminal_subgraphs)),
            "frontier_state_count": int(search_result.frontier_state_count),
            "frontier_answering_state_count": int(
                search_result.frontier_answering_state_count
            ),
        }
        if top_subgraph is not None:
            result["top_subgraph_edge_ids"] = [
                int(edge_id) for edge_id in top_subgraph.edge_ids
            ]
            result["top_subgraph_node_ids"] = [
                int(node_id) for node_id in top_subgraph.selected_node_ids
            ]
        if include_answer_support:
            result["terminal_subgraphs"] = [
                {
                    "edge_ids": [
                        int(edge_id) for edge_id in terminal_subgraph.edge_ids
                    ],
                    "log_mass": float(terminal_subgraph.log_mass),
                    "selected_node_ids": [
                        int(node_id) for node_id in terminal_subgraph.selected_node_ids
                    ],
                    "answer_entities": list(answer_entities),
                    "per_answer_log_mass": _split_terminal_answer_log_mass(
                        log_mass=float(terminal_subgraph.log_mass),
                        answer_entities=answer_entities,
                    ),
                }
                for terminal_subgraph in terminal_subgraphs[
                    : int(self.eval_cfg.edge_emit_top_k)
                ]
                for answer_entities in [
                    resolve_subgraph_answer_entities(
                        prepared_batch=prepared_batch,
                        graph_idx=0,
                        analysis=SubgraphAnalysis(
                            selected_node_ids=terminal_subgraph.selected_node_ids,
                            reachability_bits=dict(terminal_subgraph.reachability_bits),
                            component_labels={},
                            anchor_component_count=0,
                            num_selected_edges=len(terminal_subgraph.edge_ids),
                        ),
                    )
                ]
            ]
        return result

    def _predict_batch_results(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> list[dict[str, Any]]:
        return [
            self._predict_single_graph(
                batch=batch.select_graph(graph_idx, validate=False),
                include_answer_support=include_answer_support,
            )
            for graph_idx in range(int(batch.num_graphs))
        ]

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        del report_profile, on_invalid_start
        results = self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )
        metric_rows = [
            _topk_metrics(
                predicted_entities=list(result["predicted_answer_entity_ids"]),
                gold_entities=list(result["gold_answer_entity_ids"]),
                top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
            )
            for result in results
        ]
        primary_metrics = _mean_metric_dict(metric_rows)
        secondary_metrics = {
            "subgraph/terminal_count": float(
                sum(int(result["terminal_subgraph_count"]) for result in results)
            )
            / float(max(len(results), 1)),
            "subgraph/frontier_state_count": float(
                sum(int(result["frontier_state_count"]) for result in results)
            )
            / float(max(len(results), 1)),
            "subgraph/frontier_answering_state_count": float(
                sum(int(result["frontier_answering_state_count"]) for result in results)
            )
            / float(max(len(results), 1)),
            "subgraph/predicted_answer_count": float(
                sum(len(result["predicted_answer_entity_ids"]) for result in results)
            )
            / float(max(len(results), 1)),
        }
        return MetricEvaluationOutput(
            model_metrics={},
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
            results=results,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]:
        del report_profile, on_invalid_start
        return self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )

    def build_predict_labels(
        self, batch: TrajectoryBatch, outputs: list[Any]
    ) -> list[Any]:
        del batch
        return [
            {
                "sample_id": str(output["sample_id"]),
                "gold_answer_entity_ids": list(output["gold_answer_entity_ids"]),
            }
            for output in outputs
        ]

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        metric_rows = [
            _topk_metrics(
                predicted_entities=list(result.get("predicted_answer_entity_ids", [])),
                gold_entities=list(result.get("gold_answer_entity_ids", [])),
                top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
            )
            for result in predict_results
        ]
        return _mean_metric_dict(metric_rows)

    def write_prediction_artifacts(
        self,
        *,
        results: list[Any],
        labels: list[Any],
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
            results,
            labels,
            output_dir,
            split,
            artifact_name,
            schema_version,
            entity_vocab_path,
            relation_vocab_path,
            questions_path,
            overwrite,
        )
        return None


__all__ = ["SubgraphAnswerSearchRuntime"]
