from __future__ import annotations

from typing import Any

from src.eval.groups import MetricDict, MetricGroups, flatten_metric_groups

_LAZY_EXPORTS = {
    "compute_best_of_k_node_retrieval_quality": (
        "src.eval.retrieval",
        "compute_best_of_k_node_retrieval_quality",
    ),
    "compute_compactness_expectations": (
        "src.eval.compactness",
        "compute_compactness_expectations",
    ),
    "compute_expected_node_retrieval_quality": (
        "src.eval.retrieval",
        "compute_expected_node_retrieval_quality",
    ),
    "compute_exploration_diversity": (
        "src.eval.diversity",
        "compute_exploration_diversity",
    ),
    "compute_exploration_diversity_at_ks": (
        "src.eval.diversity",
        "compute_exploration_diversity_at_ks",
    ),
    "compute_node_retrieval_matrix": (
        "src.eval.retrieval",
        "compute_node_retrieval_matrix",
    ),
    "compute_sample_retrieval_metrics": (
        "src.eval.retrieval",
        "compute_sample_retrieval_metrics",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'src.eval' has no attribute {name!r}")

    module_name, object_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), object_name)
    globals()[name] = value
    return value


__all__ = [
    "MetricDict",
    "MetricGroups",
    "compute_best_of_k_node_retrieval_quality",
    "compute_compactness_expectations",
    "compute_expected_node_retrieval_quality",
    "compute_exploration_diversity",
    "compute_exploration_diversity_at_ks",
    "compute_node_retrieval_matrix",
    "compute_sample_retrieval_metrics",
    "flatten_metric_groups",
]
