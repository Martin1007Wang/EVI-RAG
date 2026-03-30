from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from src.graph import TrajectoryBatch
from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.models.gflownet.policy import SubgraphPolicy
from src.models.gflownet.reward import resolve_subgraph_answer_entities
from src.models.gflownet.sampler import SubgraphSampler
from src.models.gflownet.state import SubgraphAnalysis

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


@dataclass
class _TerminalSampleAggregate:
    edge_ids: tuple[int, ...]
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    answer_entities: tuple[int, ...]
    sample_count: int = 0


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
    *, probability_mass: float, answer_entities: tuple[int, ...]
) -> float | None:
    if probability_mass <= 0.0 or not answer_entities:
        return None
    return float(math.log(float(probability_mass)))


def _graph_candidate_answer_upper_bound(*, prepared_batch: Any, graph_idx: int) -> int:
    node_start = int(prepared_batch.node_ptr[graph_idx].item())
    node_end = int(prepared_batch.node_ptr[graph_idx + 1].item())
    return max(int(node_end - node_start), 1)


def _topk_stability_margin(
    *,
    answer_vote_counts: dict[int, int],
    executed_rollouts: int,
    candidate_answer_upper_bound: int,
    confidence: float,
    stability_top_k: int,
) -> float | None:
    if executed_rollouts < 1 or stability_top_k < 1:
        return None
    ranked_counts = sorted(
        answer_vote_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    if len(ranked_counts) < int(stability_top_k):
        return None
    delta = max(1.0 - float(confidence), 1.0e-12)
    support_size = max(int(candidate_answer_upper_bound), 1)
    radius = math.sqrt(
        math.log((4.0 * float(support_size)) / float(delta))
        / (2.0 * float(executed_rollouts))
    )
    kth_probability = float(ranked_counts[int(stability_top_k) - 1][1]) / float(
        executed_rollouts
    )
    next_probability = 0.0
    if len(ranked_counts) > int(stability_top_k):
        next_probability = float(ranked_counts[int(stability_top_k)][1]) / float(
            executed_rollouts
        )
    lower_bound = kth_probability - radius
    unseen_upper_bound = radius
    next_upper_bound = next_probability + radius
    return float(lower_bound - max(next_upper_bound, unseen_upper_bound))


def _topk_metrics_from_result(
    *, result: dict[str, Any], top_ks: tuple[int, ...]
) -> dict[str, float]:
    return _topk_metrics(
        predicted_entities=list(result["predicted_answer_entity_ids"]),
        gold_entities=list(result["gold_answer_entity_ids"]),
        top_ks=top_ks,
    )


def _summarize_result_rows(
    *, results: list[dict[str, Any]], answer_top_ks: tuple[int, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    metric_rows = [
        _topk_metrics_from_result(result=result, top_ks=answer_top_ks)
        for result in results
    ]
    primary_metrics = _mean_metric_dict(metric_rows)
    secondary_metrics = {
        "subgraph/requested_rollout_count": float(
            sum(
                int(result.get("requested_rollout_count", result["rollout_count"]))
                for result in results
            )
        )
        / float(max(len(results), 1)),
        "subgraph/rollout_count": float(
            sum(int(result["rollout_count"]) for result in results)
        )
        / float(max(len(results), 1)),
        "subgraph/early_stop_rate": float(
            sum(1.0 for result in results if bool(result.get("stopped_early", False)))
        )
        / float(max(len(results), 1)),
        "subgraph/terminal_count": float(
            sum(int(result["terminal_subgraph_count"]) for result in results)
        )
        / float(max(len(results), 1)),
        "subgraph/answering_rollout_rate": float(
            sum(
                float(result["answering_rollout_count"])
                / float(max(int(result["rollout_count"]), 1))
                for result in results
            )
        )
        / float(max(len(results), 1)),
        "subgraph/hit_rollout_rate": float(
            sum(
                float(result["hit_rollout_count"])
                / float(max(int(result["rollout_count"]), 1))
                for result in results
            )
        )
        / float(max(len(results), 1)),
        "subgraph/mean_stop_step": float(
            sum(float(result["mean_stop_step"]) for result in results)
        )
        / float(max(len(results), 1)),
        "subgraph/mean_terminal_component_count": float(
            sum(float(result["mean_terminal_component_count"]) for result in results)
        )
        / float(max(len(results), 1)),
        "subgraph/predicted_answer_count": float(
            sum(len(result["predicted_answer_entity_ids"]) for result in results)
        )
        / float(max(len(results), 1)),
    }
    return primary_metrics, secondary_metrics


def _build_analysis(
    *,
    selected_node_ids: tuple[int, ...],
    reachability_bits: dict[int, int],
    anchor_component_count: int,
    num_selected_edges: int,
) -> SubgraphAnalysis:
    return SubgraphAnalysis(
        selected_node_ids=selected_node_ids,
        reachability_bits=dict(reachability_bits),
        component_labels={},
        anchor_component_count=int(anchor_component_count),
        num_selected_edges=int(num_selected_edges),
    )


class SubgraphAnswerSearchRuntime(BaseMetricRuntime):
    sampler: Any

    def __init__(
        self,
        *,
        eval_cfg: dict[str, Any],
        policy: SubgraphPolicy,
        sampler: SubgraphSampler,
    ) -> None:
        self.eval_cfg = normalize_search_eval_cfg(eval_cfg)
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
        monte_carlo_cfg = self.eval_cfg["monte_carlo"]
        requested_rollouts = int(monte_carlo_cfg["rollouts"])
        batch_rollouts = min(
            requested_rollouts,
            int(monte_carlo_cfg.get("batch_rollouts", requested_rollouts)),
        )
        confidence = float(monte_carlo_cfg["confidence"])
        temperature = float(monte_carlo_cfg["temperature"])
        early_stop_cfg = monte_carlo_cfg["early_stop"]
        early_stop_enabled = bool(early_stop_cfg["enabled"])
        early_stop_min_rollouts = min(
            requested_rollouts,
            int(early_stop_cfg["min_rollouts"]),
        )
        stability_top_k = int(early_stop_cfg["stability_top_k"])
        action_pruning_cfg = monte_carlo_cfg["action_pruning"]
        candidate_answer_upper_bound = _graph_candidate_answer_upper_bound(
            prepared_batch=prepared_batch,
            graph_idx=0,
        )

        answer_vote_counts: dict[int, int] = {}
        terminal_subgraphs: dict[tuple[int, ...], _TerminalSampleAggregate] = {}
        answering_rollout_count = 0
        hit_rollout_count = 0
        total_stop_steps = 0.0
        total_terminal_component_count = 0.0
        early_stop_margin: float | None = None

        processed_rollouts = 0
        while processed_rollouts < requested_rollouts:
            current_rollouts = min(
                batch_rollouts, requested_rollouts - processed_rollouts
            )
            sample_batch = self.sampler.sample(
                policy=self.policy,
                prepared_batch=prepared_batch,
                rollouts_per_graph=current_rollouts,
                temperature=temperature,
                proposal_bias_scale=0.0,
                action_pruning=action_pruning_cfg,
            )
            stop_steps = (
                sample_batch.termination_action_steps[0].detach().cpu().tolist()
            )
            terminal_component_counts = (
                sample_batch.terminal_component_counts[0].detach().cpu().tolist()
            )
            hit_mask = sample_batch.terminal_hit_mask[0].detach().cpu().tolist()
            total_stop_steps += float(sum(int(step) for step in stop_steps))
            total_terminal_component_count += float(
                sum(int(count) for count in terminal_component_counts)
            )
            hit_rollout_count += int(sum(bool(value) for value in hit_mask))

            for rollout_idx in range(current_rollouts):
                edge_ids = tuple(
                    int(edge_id)
                    for edge_id in sample_batch.terminal_edge_ids[rollout_idx]
                )
                selected_node_ids = tuple(
                    int(node_id)
                    for node_id in sample_batch.terminal_node_ids[rollout_idx]
                )
                reachability_bits = {
                    int(node_id): int(bits)
                    for node_id, bits in sample_batch.terminal_reachability_bits[
                        rollout_idx
                    ].items()
                }
                analysis = _build_analysis(
                    selected_node_ids=selected_node_ids,
                    reachability_bits=reachability_bits,
                    anchor_component_count=int(terminal_component_counts[rollout_idx]),
                    num_selected_edges=len(edge_ids),
                )
                answer_entities = tuple(
                    int(entity_id)
                    for entity_id in resolve_subgraph_answer_entities(
                        prepared_batch=prepared_batch,
                        graph_idx=0,
                        analysis=analysis,
                    )
                )
                if answer_entities:
                    answering_rollout_count += 1
                    for entity_id in dict.fromkeys(answer_entities):
                        answer_vote_counts[int(entity_id)] = (
                            int(answer_vote_counts.get(int(entity_id), 0)) + 1
                        )
                payload = terminal_subgraphs.get(edge_ids)
                if payload is None:
                    payload = _TerminalSampleAggregate(
                        edge_ids=edge_ids,
                        selected_node_ids=selected_node_ids,
                        reachability_bits=reachability_bits,
                        answer_entities=answer_entities,
                    )
                    terminal_subgraphs[edge_ids] = payload
                payload.sample_count += 1
            processed_rollouts += current_rollouts

            if early_stop_enabled and processed_rollouts >= early_stop_min_rollouts:
                early_stop_margin = _topk_stability_margin(
                    answer_vote_counts=answer_vote_counts,
                    executed_rollouts=processed_rollouts,
                    candidate_answer_upper_bound=candidate_answer_upper_bound,
                    confidence=confidence,
                    stability_top_k=stability_top_k,
                )
                if early_stop_margin is not None and early_stop_margin > 0.0:
                    break

        executed_rollouts = max(processed_rollouts, 1)
        stopped_early = processed_rollouts < requested_rollouts

        ranked_answers = sorted(
            answer_vote_counts.items(),
            key=lambda item: (-int(item[1]), int(item[0])),
        )
        ranked_terminals = sorted(
            terminal_subgraphs.values(),
            key=lambda item: (-int(item.sample_count), item.edge_ids),
        )
        gold_answers = [
            int(value) for value in batch.answer_entity_ids.detach().cpu().tolist()
        ]
        top_subgraph = ranked_terminals[0] if ranked_terminals else None
        result: dict[str, Any] = {
            "sample_id": str(batch.sample_ids[0]),
            "question": str(batch.questions[0]),
            "gold_answer_entity_ids": gold_answers,
            "predicted_answer_entity_ids": [
                int(entity_id) for entity_id, _ in ranked_answers
            ],
            "answer_log_masses": [
                float(math.log(float(votes) / float(executed_rollouts)))
                for _, votes in ranked_answers
            ],
            "requested_rollout_count": int(requested_rollouts),
            "rollout_count": int(processed_rollouts),
            "answering_rollout_count": int(answering_rollout_count),
            "hit_rollout_count": int(hit_rollout_count),
            "terminal_subgraph_count": int(len(ranked_terminals)),
            "mean_stop_step": float(total_stop_steps) / float(executed_rollouts),
            "mean_terminal_component_count": float(total_terminal_component_count)
            / float(executed_rollouts),
            "stopped_early": bool(stopped_early),
            "early_stop_margin": early_stop_margin,
        }
        if top_subgraph is not None:
            top_probability = float(top_subgraph.sample_count) / float(
                executed_rollouts
            )
            result["top_subgraph_edge_ids"] = [
                int(edge_id) for edge_id in top_subgraph.edge_ids
            ]
            result["top_subgraph_node_ids"] = [
                int(node_id) for node_id in top_subgraph.selected_node_ids
            ]
            result["top_subgraph_probability"] = float(top_probability)
            result["top_subgraph_sample_count"] = int(top_subgraph.sample_count)
        if include_answer_support:
            result["terminal_subgraphs"] = [
                {
                    "edge_ids": [int(edge_id) for edge_id in payload.edge_ids],
                    "selected_node_ids": [
                        int(node_id) for node_id in payload.selected_node_ids
                    ],
                    "answer_entities": [
                        int(entity_id) for entity_id in payload.answer_entities
                    ],
                    "sample_count": int(payload.sample_count),
                    "probability": float(payload.sample_count)
                    / float(executed_rollouts),
                    "per_answer_log_mass": _split_terminal_answer_log_mass(
                        probability_mass=float(payload.sample_count)
                        / float(executed_rollouts),
                        answer_entities=payload.answer_entities,
                    ),
                }
                for payload in ranked_terminals[: int(self.eval_cfg["edge_emit_top_k"])]
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
        primary_metrics, secondary_metrics = _summarize_result_rows(
            results=results,
            answer_top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
        )
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
        primary_metrics, _ = _summarize_result_rows(
            results=[dict(result) for result in predict_results],
            answer_top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
        )
        return primary_metrics

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
