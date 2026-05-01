from __future__ import annotations

from collections.abc import Sequence

from src.data.schema import RetrievalBatch
from src.eval.compactness import compute_compactness_expectations
from src.eval.diversity import compute_exploration_diversity_at_ks
from src.eval.groups import MetricGroups
from src.eval.retrieval import (
    compute_best_of_k_node_retrieval_quality,
    compute_expected_node_retrieval_quality,
)
from src.training.rollout_diagnostics import (
    compute_policy_behavior_diagnostics,
    compute_stop_behavior_diagnostics,
    compute_stop_counterfactual_diagnostics,
    compute_teacher_edge_diagnostics,
)
from src.weaver.rollout.schema import RolloutBatch
from src.weaver.rollout.terminal_subgraph import (
    UnionSubgraphMasks,
    compute_union_subgraph_masks,
)


def union_subgraph_masks_from_rollouts(
    *,
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
) -> UnionSubgraphMasks:
    return compute_union_subgraph_masks(rollouts, batch)


def evaluate_rollouts(
    *,
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    eval_budgets: Sequence[int],
    debug_metrics: bool,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    stage: str,
) -> MetricGroups:
    sample_metrics = compute_expected_node_retrieval_quality(
        rollouts,
        batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )
    best_of_k_metrics = compute_best_of_k_node_retrieval_quality(
        rollouts,
        batch,
        ks=eval_budgets,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )
    best_of_k_metrics.update(
        _compute_best_of_k_gain_metrics(
            best_of_k_metrics=best_of_k_metrics,
            expected_target_f1=sample_metrics["expected_target_f1"],
        )
    )
    all_answer_sample_metrics: dict[str, float] | None = None
    all_answer_best_of_k_metrics: dict[str, float] | None = None
    if use_reachable_targets:
        all_answer_sample_metrics = compute_expected_node_retrieval_quality(
            rollouts,
            batch,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=False,
        )
        all_answer_best_of_k_metrics = compute_best_of_k_node_retrieval_quality(
            rollouts,
            batch,
            ks=eval_budgets,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=False,
        )
        all_answer_best_of_k_metrics.update(
            _compute_best_of_k_gain_metrics(
                best_of_k_metrics=all_answer_best_of_k_metrics,
                expected_target_f1=all_answer_sample_metrics["expected_target_f1"],
            )
        )

    include_dangling = debug_metrics or stage == "val"
    metrics: MetricGroups = {
        "sample": sample_metrics,
        "best_of_k": best_of_k_metrics,
        "compactness": compute_compactness_expectations(
            rollouts,
            batch,
            include_dangling=include_dangling,
        ),
    }
    if (
        all_answer_sample_metrics is not None
        and all_answer_best_of_k_metrics is not None
    ):
        metrics["sample_all_answers"] = all_answer_sample_metrics
        metrics["best_of_k_all_answers"] = all_answer_best_of_k_metrics

    diversity_ks = _diversity_ks(eval_budgets=eval_budgets, stage=stage)
    if diversity_ks:
        metrics["diversity"] = compute_exploration_diversity_at_ks(
            rollouts,
            batch,
            ks=diversity_ks,
        )

    if stage != "test":
        rollout_tuple = tuple(rollouts)
        metrics["rollout"] = _strip_metric_prefix(
            compute_stop_behavior_diagnostics(rollout_tuple),
            prefix="rollout/",
        )
        metrics["policy"] = _strip_metric_prefix(
            compute_policy_behavior_diagnostics(rollout_tuple),
            prefix="policy/",
        )
        if debug_metrics:
            metrics["stop_counterfactual"] = _strip_metric_prefix(
                compute_stop_counterfactual_diagnostics(rollout_tuple),
                prefix="stop_counterfactual/",
            )
            metrics["teacher_edge"] = _strip_metric_prefix(
                compute_teacher_edge_diagnostics(rollouts=rollout_tuple, batch=batch),
                prefix="teacher_edge/",
            )

    return metrics


def _compute_best_of_k_gain_metrics(
    *,
    best_of_k_metrics: dict[str, float],
    expected_target_f1: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    prefix = "max_target_f1_at_"
    for name, value in best_of_k_metrics.items():
        if name.startswith(prefix):
            k = name[len(prefix) :]
            metrics[f"f1_gain_at_{k}"] = float(value) - float(expected_target_f1)
    return metrics


def _diversity_ks(*, eval_budgets: Sequence[int], stage: str) -> tuple[int, ...]:
    if stage not in {"val", "test"}:
        return ()
    positive = tuple(sorted({int(k) for k in eval_budgets if int(k) >= 1}))
    if not positive:
        return ()
    return (positive[-1],)


def _strip_metric_prefix(metrics: dict[str, float], *, prefix: str) -> dict[str, float]:
    return {
        name[len(prefix) :] if name.startswith(prefix) else name: value
        for name, value in metrics.items()
    }
