from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.data.schema.constants import EntityVocabFields, RelationVocabFields
from src.graph import TrajectoryBatch

from src.metrics.base import BaseMetricRuntime
from src.metrics.prediction_io import append_jsonl_records
from src.metrics.protocol import MetricEvaluationOutput
from src.metrics.search_eval_utils import normalize_search_eval_cfg

from ...core.cuda_memory import profile_cuda_memory
from ...core.policy import SubgraphPolicy
from ...core.sampler import SubgraphSampler
from ...core.subgraph_batch import SubgraphBatchBuildOptions
from .answer_search_metrics import _accumulate_metric_sums
from .answer_search_metrics import _average_metric_sums
from .answer_search_metrics import _clamped_positive_weight
from .answer_search_metrics import _graph_candidate_answer_upper_bound
from .answer_search_metrics import _mean_metric_dict
from .answer_search_metrics import _rollout_support_weight
from .answer_search_metrics import _secondary_metrics_from_result
from .answer_search_metrics import _split_terminal_answer_log_mass
from .answer_search_metrics import _summarize_result_rows
from .answer_search_metrics import _support_mass_metrics_from_result
from .answer_search_metrics import _terminal_action_index
from .answer_search_metrics import _topk_metrics
from .answer_search_metrics import _topk_metrics_from_result
from .answer_search_metrics import _topk_stability_margin
from .answer_search_metrics import _trajectory_log_prob
from .answer_search_support import _build_graph_prediction_accumulator
from .answer_search_support import _build_support_records
from .answer_search_support import _decorate_trajectory_records
from .answer_search_support import _edge_overlap_ratio
from .answer_search_support import _edge_records_from_terminal
from .answer_search_support import _finalize_graph_result
from .answer_search_support import _load_vocab_label_map
from .answer_search_support import _select_terminal_support
from .answer_search_support import _trajectory_text_from_edge_records
from .answer_search_types import _GraphPredictionAccumulator
from .answer_search_types import _PredictMetricsAccumulator
from .answer_search_types import _SubgraphPredictionCodec
from .answer_search_types import _TerminalSampleAggregate


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

    @staticmethod
    def _action_pruning_enabled(action_pruning_cfg: dict[str, Any]) -> bool:
        return (
            int(action_pruning_cfg.get("per_node_top_k", 0)) > 0
            or int(action_pruning_cfg.get("per_state_top_k", 0)) > 0
        )

    def _eval_batch_build_options(self) -> SubgraphBatchBuildOptions:
        action_pruning_cfg = self.eval_cfg["monte_carlo"]["action_pruning"]
        return SubgraphBatchBuildOptions(
            include_edge_question_similarity=self._action_pruning_enabled(
                action_pruning_cfg
            ),
            include_oracle_distance=False,
            include_teacher_banks=False,
        )

    def _predict_single_graph(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> dict[str, Any]:
        return self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )[0]

    def _predict_batch_results(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> list[dict[str, Any]]:
        monte_carlo_cfg = self.eval_cfg["monte_carlo"]
        requested_rollouts = int(monte_carlo_cfg["rollouts"])
        batch_rollouts = min(
            requested_rollouts,
            int(monte_carlo_cfg.get("batch_rollouts", requested_rollouts)),
        )
        confidence = float(monte_carlo_cfg["confidence"])
        temperature = float(monte_carlo_cfg["temperature"])
        aggregation_backend = str(monte_carlo_cfg["answer_aggregation"]["backend"])
        early_stop_cfg = monte_carlo_cfg["early_stop"]
        early_stop_enabled = bool(early_stop_cfg["enabled"])
        early_stop_min_rollouts = min(
            requested_rollouts,
            int(early_stop_cfg["min_rollouts"]),
        )
        stability_top_k = int(early_stop_cfg["stability_top_k"])
        action_pruning_cfg = monte_carlo_cfg["action_pruning"]

        active_graph_indices = list(range(int(batch.num_graphs)))
        with profile_cuda_memory(
            "eval.prepare_batch.initial",
            device=batch.edge_index.device,
            extra=(
                f"num_graphs={int(batch.num_graphs)} requested_rollouts={requested_rollouts} "
                f"batch_rollouts={batch_rollouts}"
            ),
        ):
            full_prepared_batch = self.policy.prepare_batch(
                batch,
                build_options=self._eval_batch_build_options(),
            )
        active_prepared_batch = full_prepared_batch
        accumulators = {
            int(graph_idx): _build_graph_prediction_accumulator(
                batch=batch,
                prepared_batch=full_prepared_batch,
                graph_idx=int(graph_idx),
                original_graph_idx=int(graph_idx),
            )
            for graph_idx in active_graph_indices
        }

        while active_graph_indices:
            processed_rollouts = accumulators[active_graph_indices[0]].rollout_count
            remaining_rollouts = int(requested_rollouts) - int(processed_rollouts)
            if remaining_rollouts <= 0:
                break
            current_rollouts = min(int(batch_rollouts), int(remaining_rollouts))
            chunk_extra = (
                f"active_graphs={len(active_graph_indices)} current_rollouts={current_rollouts} "
                f"processed_rollouts={processed_rollouts}"
            )
            with profile_cuda_memory(
                "eval.sampler.sample",
                device=active_prepared_batch.device,
                extra=chunk_extra,
            ):
                sample_batch = self.sampler.sample(
                    policy=self.policy,
                    prepared_batch=active_prepared_batch,
                    rollouts_per_graph=current_rollouts,
                    temperature=temperature,
                    action_pruning=action_pruning_cfg,
                )
            next_active_graph_indices: list[int] = []
            for local_graph_idx, original_graph_idx in enumerate(active_graph_indices):
                accumulator = accumulators[original_graph_idx]
                original_node_start = int(batch.node_ptr[original_graph_idx].item())
                original_edge_start = int(batch.edge_ptr[original_graph_idx].item())
                stop_steps = (
                    sample_batch.termination_action_steps[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                terminal_component_counts = (
                    sample_batch.terminal_component_counts[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                hit_mask = (
                    sample_batch.terminal_hit_mask[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                accumulator.total_stop_steps += float(
                    sum(int(step) for step in stop_steps)
                )
                accumulator.total_terminal_component_count += float(
                    sum(int(count) for count in terminal_component_counts)
                )
                accumulator.gold_answer_in_state_rollout_count += int(
                    sum(bool(value) for value in hit_mask)
                )

                for rollout_idx in range(current_rollouts):
                    flat_rollout_idx = (
                        local_graph_idx * current_rollouts
                    ) + rollout_idx
                    global_edge_ids = tuple(
                        int(edge_id)
                        for edge_id in sample_batch.terminal_edge_ids[flat_rollout_idx]
                    )
                    edge_ids = tuple(
                        int(edge_id) - int(original_edge_start)
                        for edge_id in global_edge_ids
                    )
                    selected_node_ids = tuple(
                        int(node_id) - int(original_node_start)
                        for node_id in sample_batch.terminal_node_ids[flat_rollout_idx]
                    )
                    reachability_bits = {
                        int(node_id) - int(original_node_start): int(bits)
                        for node_id, bits in sample_batch.terminal_reachability_bits[
                            flat_rollout_idx
                        ].items()
                    }
                    terminal_answer_set_entity_ids = tuple(
                        int(entity_id)
                        for entity_id in sample_batch.terminal_answer_set_entity_ids[
                            flat_rollout_idx
                        ]
                    )
                    support_weight = _rollout_support_weight(
                        sample_batch=sample_batch,
                        graph_idx=local_graph_idx,
                        rollout_idx=rollout_idx,
                        backend=aggregation_backend,
                    )
                    if terminal_answer_set_entity_ids:
                        accumulator.nonempty_terminal_answer_set_rollout_count += 1
                        per_answer_mass = float(support_weight) / float(
                            len(terminal_answer_set_entity_ids)
                        )
                        for answer_entity_id in terminal_answer_set_entity_ids:
                            accumulator.answer_vote_counts[int(answer_entity_id)] = (
                                float(
                                    accumulator.answer_vote_counts.get(
                                        int(answer_entity_id), 0.0
                                    )
                                )
                                + float(per_answer_mass)
                            )
                    payload = accumulator.terminal_witnesses.get(edge_ids)
                    if payload is None:
                        payload = _TerminalSampleAggregate(
                            edge_ids=edge_ids,
                            selected_node_ids=selected_node_ids,
                            reachability_bits=reachability_bits,
                            terminal_answer_set_entity_ids=(
                                terminal_answer_set_entity_ids
                            ),
                        )
                        accumulator.terminal_witnesses[edge_ids] = payload
                    payload.sample_count += 1
                    payload.score_sum += float(support_weight)

                accumulator.rollout_count += int(current_rollouts)
                if accumulator.rollout_count >= int(requested_rollouts):
                    continue
                if not early_stop_enabled or accumulator.rollout_count < int(
                    early_stop_min_rollouts
                ):
                    next_active_graph_indices.append(int(original_graph_idx))
                    continue
                accumulator.early_stop_margin = _topk_stability_margin(
                    answer_vote_counts=accumulator.answer_vote_counts,
                    executed_rollouts=accumulator.rollout_count,
                    candidate_answer_upper_bound=accumulator.candidate_answer_upper_bound,
                    confidence=confidence,
                    stability_top_k=stability_top_k,
                )
                if (
                    accumulator.early_stop_margin is not None
                    and accumulator.early_stop_margin > 0.0
                ):
                    accumulator.stopped_early = True
                    continue
                next_active_graph_indices.append(int(original_graph_idx))

            if next_active_graph_indices == active_graph_indices:
                continue
            active_graph_indices = next_active_graph_indices
            if not active_graph_indices:
                break
            with profile_cuda_memory(
                "eval.prepared_batch.select_graphs",
                device=full_prepared_batch.device,
                extra=f"active_graphs={len(active_graph_indices)}",
            ):
                active_prepared_batch = full_prepared_batch.select_graphs(
                    active_graph_indices
                )

        return [
            _finalize_graph_result(
                accumulator=accumulators[graph_idx],
                batch=batch,
                include_answer_support=include_answer_support,
                aggregation_backend=aggregation_backend,
                edge_emit_top_k=int(self.eval_cfg["edge_emit_top_k"]),
                support_mass_threshold=float(self.eval_cfg["support_mass_threshold"]),
                support_path_overlap_penalty=float(
                    self.eval_cfg["support_path_overlap_penalty"]
                ),
                requested_rollouts=int(requested_rollouts),
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
            edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
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
                "question": str(output["question"]),
                "gold_answer_entity_ids": list(output["gold_answer_entity_ids"]),
                "gold_answer_in_graph": bool(output.get("gold_answer_in_graph", False)),
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
        primary_metrics, secondary_metrics = _summarize_result_rows(
            results=[dict(result) for result in predict_results],
            answer_top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
            edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
        )
        return {**primary_metrics, **secondary_metrics}

    def initialize_predict_metrics_accumulator(
        self,
        *,
        report_profile: str,
    ) -> _PredictMetricsAccumulator:
        del report_profile
        return _PredictMetricsAccumulator()

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: _PredictMetricsAccumulator,
        predict_results: list[Any],
        report_profile: str,
    ) -> None:
        del report_profile
        for result in [dict(item) for item in predict_results]:
            accumulator.count += 1
            _accumulate_metric_sums(
                accumulator.primary_sums,
                _topk_metrics_from_result(
                    result=result,
                    top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
                ),
            )
            _accumulate_metric_sums(
                accumulator.secondary_sums,
                _secondary_metrics_from_result(
                    result=result,
                    edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
                ),
            )

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: _PredictMetricsAccumulator,
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        primary_metrics = _average_metric_sums(
            accumulator.primary_sums,
            count=accumulator.count,
        )
        secondary_metrics = _average_metric_sums(
            accumulator.secondary_sums,
            count=accumulator.count,
        )
        return {**primary_metrics, **secondary_metrics}

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
        del artifact_name, schema_version, questions_path
        if not results:
            return None
        output_root = Path(str(output_dir))
        output_root.mkdir(parents=True, exist_ok=True)
        results_path = output_root / f"{split}.jsonl"
        labels_path = output_root / f"{split}.labels.jsonl"
        if overwrite:
            for path in (results_path, labels_path):
                if path.exists():
                    path.unlink()

        entity_labels = _load_vocab_label_map(
            path=entity_vocab_path,
            id_field=EntityVocabFields.ENTITY_ID,
            label_field=EntityVocabFields.LABEL,
        )
        relation_labels = _load_vocab_label_map(
            path=relation_vocab_path,
            id_field=RelationVocabFields.RELATION_ID,
            label_field=RelationVocabFields.LABEL,
        )

        serialized_results: list[dict[str, Any]] = []
        serialized_labels: list[dict[str, Any]] = []
        label_by_sample_id = {
            str(label["sample_id"]): dict(label)
            for label in [dict(item) for item in labels]
        }
        for result in [dict(item) for item in results]:
            sample_id = str(result["sample_id"])
            trajectories = _decorate_trajectory_records(
                trajectories=[
                    dict(item) for item in result.get("witness_supports", [])
                ],
                entity_labels=entity_labels,
                relation_labels=relation_labels,
            )
            serialized_results.append(
                {
                    "sample_id": sample_id,
                    "question": str(result["question"]),
                    "predicted_answer_entity_ids": list(
                        result.get("predicted_answer_entity_ids", [])
                    ),
                    "answer_log_posterior_surrogate_masses": list(
                        result.get("answer_log_posterior_surrogate_masses", [])
                    ),
                    "requested_rollout_count": int(
                        result.get("requested_rollout_count", result["rollout_count"])
                    ),
                    "rollout_count": int(result["rollout_count"]),
                    "nonempty_terminal_answer_set_rollout_count": int(
                        result["nonempty_terminal_answer_set_rollout_count"]
                    ),
                    "gold_answer_in_state_rollout_count": int(
                        result["gold_answer_in_state_rollout_count"]
                    ),
                    "stopped_early": bool(result.get("stopped_early", False)),
                    "witness_support_probability_mass": float(
                        result.get(
                            "witness_support_probability_mass",
                            sum(
                                float(item.get("probability", 0.0))
                                for item in trajectories
                            ),
                        )
                    ),
                    "trajectories": trajectories,
                }
            )
            label_record = label_by_sample_id.get(
                sample_id,
                {
                    "sample_id": sample_id,
                    "question": str(result["question"]),
                    "gold_answer_entity_ids": list(result["gold_answer_entity_ids"]),
                    "gold_answer_in_graph": bool(
                        result.get("gold_answer_in_graph", False)
                    ),
                },
            )
            gold_answer_entity_ids = [
                int(entity_id)
                for entity_id in label_record.get(
                    "gold_answer_entity_ids", result["gold_answer_entity_ids"]
                )
            ]
            serialized_labels.append(
                {
                    "sample_id": sample_id,
                    "question": str(label_record.get("question", result["question"])),
                    "answer_entity_ids": gold_answer_entity_ids,
                    "answer_texts": [
                        str(entity_labels.get(int(entity_id), str(entity_id)))
                        for entity_id in gold_answer_entity_ids
                    ],
                    "gold_answer_in_graph": bool(
                        label_record.get(
                            "gold_answer_in_graph",
                            result.get("gold_answer_in_graph", False),
                        )
                    ),
                }
            )

        append_jsonl_records(results_path, records=serialized_results)
        append_jsonl_records(labels_path, records=serialized_labels)
        return {
            "prompt_path": results_path,
            "results_path": results_path,
            "labels_path": labels_path,
        }


__all__ = ["SubgraphAnswerSearchRuntime"]
