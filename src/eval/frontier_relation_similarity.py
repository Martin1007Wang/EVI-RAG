from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.artifacts import load_materialization_artifact_from_path
from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.data.tensor_table import read_table
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.objectives.subtb.batch import prepare_subtb_batch
from src.weaver.rollout.trajectory import EXTERNAL_TERMINAL, TrajectoryBatch


@dataclass(frozen=True, slots=True)
class FrontierStateRecord:
    positive_scores: tuple[float, ...]
    negative_scores: tuple[float, ...]
    frontier_size: int


@dataclass(frozen=True, slots=True)
class DatasetScoreCollection:
    dataset: str
    state_records: tuple[FrontierStateRecord, ...]
    skipped_empty_frontier: int
    skipped_no_gold_frontier: int
    replay_trajectory_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class DistributionStats:
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class SweepRow:
    dataset: str
    threshold: float
    eligible_state_recall: float
    all_positive_state_recall: float
    states_with_some_gold_dropped_rate: float
    states_with_no_edges_left_rate: float
    positive_edge_recall: float
    frontier_edge_prune_rate: float
    negative_edge_prune_rate: float
    mean_kept_edges_per_eligible_state: float


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    dataset: str
    eligible_state_count: int
    positive_edge_count: int
    negative_edge_count: int
    skipped_empty_frontier: int
    skipped_no_gold_frontier: int
    replay_trajectory_count: int
    sample_count: int
    positive_score_stats: DistributionStats
    negative_score_stats: DistributionStats
    recommended_thresholds: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    collection: DatasetScoreCollection
    summary: DatasetSummary
    sweep_rows: list[SweepRow]
    histogram_rows: list[dict[str, Any]]


def run_analysis(
    *,
    dataset: str,
    data_root: Path,
    split: str,
    max_samples: int | None = None,
    sweep_step: float = 0.01,
) -> AnalysisResult:
    manifest_path = data_root / dataset / "metadata" / "materialization_manifest.json"
    materialization = load_materialization_artifact_from_path(manifest_path)
    if materialization is None:
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    relation_semantic_table = torch.nn.functional.normalize(
        read_table(materialization.relation_semantic_table).float(),
        p=2,
        dim=1,
    )
    ds = RetrievalDataset(materialization=materialization, split=split)
    collator = RetrievalCollator()
    try:
        collection = _collect_dataset_scores(
            dataset=dataset,
            ds=ds,
            collator=collator,
            relation_semantic_table=relation_semantic_table,
            max_samples=max_samples,
        )
    finally:
        ds.close()

    summary = summarize_collection(
        collection,
        sweep_step=sweep_step,
    )
    sweep_rows = sweep_thresholds(
        dataset=dataset,
        state_records=collection.state_records,
        step=sweep_step,
    )
    histogram_rows = build_histogram_rows(
        dataset=dataset,
        state_records=collection.state_records,
    )
    return AnalysisResult(
        collection=collection,
        summary=summary,
        sweep_rows=sweep_rows,
        histogram_rows=histogram_rows,
    )


def summarize_collection(
    collection: DatasetScoreCollection,
    *,
    sweep_step: float = 0.01,
) -> DatasetSummary:
    positive_scores, negative_scores = flatten_scores(collection.state_records)
    sweep_rows = sweep_thresholds(
        dataset=collection.dataset,
        state_records=collection.state_records,
        step=sweep_step,
    )
    return DatasetSummary(
        dataset=collection.dataset,
        eligible_state_count=len(collection.state_records),
        positive_edge_count=len(positive_scores),
        negative_edge_count=len(negative_scores),
        skipped_empty_frontier=collection.skipped_empty_frontier,
        skipped_no_gold_frontier=collection.skipped_no_gold_frontier,
        replay_trajectory_count=collection.replay_trajectory_count,
        sample_count=collection.sample_count,
        positive_score_stats=_distribution_stats(positive_scores),
        negative_score_stats=_distribution_stats(negative_scores),
        recommended_thresholds=recommend_thresholds(
            sweep_rows=sweep_rows,
            recall_targets=(0.99, 0.95, 0.90),
        ),
    )


def flatten_scores(
    state_records: tuple[FrontierStateRecord, ...],
) -> tuple[list[float], list[float]]:
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for record in state_records:
        positive_scores.extend(record.positive_scores)
        negative_scores.extend(record.negative_scores)
    return positive_scores, negative_scores


def collect_frontier_state_records(
    *,
    batch,
    relation_semantic_table: torch.Tensor,
) -> DatasetScoreCollection:
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay = ReplayContext.from_batch(
        batch=batch,
        graph_context=graph,
        target_context=target,
    )
    trajectories = replay_prefix_trajectories(replay_context=replay, device=graph.device)
    if trajectories.num_trajectories == 0:
        return DatasetScoreCollection(
            dataset=str(batch.sample_id[0]).split("/", 1)[0] if hasattr(batch, "sample_id") else "unknown",
            state_records=(),
            skipped_empty_frontier=0,
            skipped_no_gold_frontier=0,
            replay_trajectory_count=0,
            sample_count=int(len(getattr(batch, "sample_id", [])) or 1),
        )

    prepared = prepare_subtb_batch(
        trajectories=trajectories,
        graph_context=graph,
    )
    state_ids = torch.unique(prepared.prefix_state_ids[prepared.prefix_state_ids.ge(0)])
    relation_ids = batch.edge_relation_catalog_ids.to(dtype=torch.long, device=graph.device)
    question_h = torch.nn.functional.normalize(
        batch.question_emb.to(dtype=torch.float32, device=graph.device),
        p=2,
        dim=1,
    )

    state_records: list[FrontierStateRecord] = []
    skipped_empty_frontier = 0
    skipped_no_gold_frontier = 0
    for state_id in state_ids.tolist():
        state = prepared.states.take(torch.tensor([state_id], dtype=torch.long, device=graph.device))
        frontier = state.frontier(graph_context=graph)
        edge_ids = frontier.edge_ids
        if int(edge_ids.numel()) == 0:
            skipped_empty_frontier += 1
            continue
        gold_mask = target.edge_on_shortest_path.index_select(0, edge_ids)
        if not bool(gold_mask.any()):
            skipped_no_gold_frontier += 1
            continue
        graph_id = int(state.graph_ids[0].item())
        rel_ids = relation_ids.index_select(0, edge_ids)
        scores = relation_semantic_table.index_select(0, rel_ids).matmul(question_h[graph_id])
        positive = tuple(float(x) for x in scores[gold_mask].detach().cpu().tolist())
        negative = tuple(float(x) for x in scores[~gold_mask].detach().cpu().tolist())
        state_records.append(
            FrontierStateRecord(
                positive_scores=positive,
                negative_scores=negative,
                frontier_size=int(edge_ids.numel()),
            )
        )

    return DatasetScoreCollection(
        dataset=str(batch.sample_id[0]).split("/", 1)[0] if hasattr(batch, "sample_id") else "unknown",
        state_records=tuple(state_records),
        skipped_empty_frontier=skipped_empty_frontier,
        skipped_no_gold_frontier=skipped_no_gold_frontier,
        replay_trajectory_count=int(trajectories.num_trajectories),
        sample_count=int(len(getattr(batch, "sample_id", [])) or 1),
    )


def replay_prefix_trajectories(
    *,
    replay_context: ReplayContext,
    device: torch.device,
) -> TrajectoryBatch:
    edge_ids = replay_context.edge_ids
    edge_count = replay_context.edge_count
    if int(edge_count.numel()) == 0:
        return TrajectoryBatch.empty(device=device, budget=int(edge_ids.size(-1)))

    valid = edge_count.ge(0)
    if not bool(valid.any()):
        return TrajectoryBatch.empty(device=device, budget=int(edge_ids.size(-1)))

    graph_ids = (
        torch.arange(int(edge_count.size(0)), device=device, dtype=torch.long)
        .view(-1, 1, 1)
        .expand_as(edge_count)[valid]
    )
    selected_edge_ids = edge_ids[valid].contiguous()
    selected_edge_count = edge_count[valid].contiguous()
    budget = int(selected_edge_ids.size(1))
    num = int(selected_edge_count.numel())
    return TrajectoryBatch(
        graph_ids=graph_ids,
        edge_ids=selected_edge_ids,
        edge_logp=torch.zeros((num, budget), dtype=torch.float32, device=device),
        edge_count=selected_edge_count,
        stop_reason=torch.full((num,), int(EXTERNAL_TERMINAL), dtype=torch.uint8, device=device),
        stop_logp=torch.zeros((num,), dtype=torch.float32, device=device),
        source=torch.ones((num,), dtype=torch.bool, device=device),
    )


def sweep_thresholds(
    *,
    dataset: str,
    state_records: tuple[FrontierStateRecord, ...],
    step: float,
) -> list[SweepRow]:
    thresholds = threshold_grid(step=step)
    total_states = len(state_records)
    total_pos = sum(len(record.positive_scores) for record in state_records)
    total_neg = sum(len(record.negative_scores) for record in state_records)
    total_frontier = sum(record.frontier_size for record in state_records)
    rows: list[SweepRow] = []
    for threshold in thresholds:
        kept_states = 0
        all_positive_states = 0
        states_with_some_gold_dropped = 0
        states_with_no_edges_left = 0
        kept_pos = 0
        kept_neg = 0
        kept_frontier = 0
        for record in state_records:
            pos_kept = sum(score >= threshold for score in record.positive_scores)
            neg_kept = sum(score >= threshold for score in record.negative_scores)
            if pos_kept > 0:
                kept_states += 1
            if pos_kept == len(record.positive_scores):
                all_positive_states += 1
            if pos_kept < len(record.positive_scores):
                states_with_some_gold_dropped += 1
            if pos_kept + neg_kept == 0:
                states_with_no_edges_left += 1
            kept_pos += pos_kept
            kept_neg += neg_kept
            kept_frontier += pos_kept + neg_kept
        rows.append(
            SweepRow(
                dataset=dataset,
                threshold=round(threshold, 6),
                eligible_state_recall=_safe_rate(kept_states, total_states),
                all_positive_state_recall=_safe_rate(all_positive_states, total_states),
                states_with_some_gold_dropped_rate=_safe_rate(states_with_some_gold_dropped, total_states),
                states_with_no_edges_left_rate=_safe_rate(states_with_no_edges_left, total_states),
                positive_edge_recall=_safe_rate(kept_pos, total_pos),
                frontier_edge_prune_rate=1.0 - _safe_rate(kept_frontier, total_frontier),
                negative_edge_prune_rate=1.0 - _safe_rate(kept_neg, total_neg),
                mean_kept_edges_per_eligible_state=_safe_rate(kept_frontier, total_states),
            )
        )
    return rows


def recommend_thresholds(
    *,
    sweep_rows: list[SweepRow],
    recall_targets: tuple[float, ...],
) -> dict[str, float | None]:
    recommended: dict[str, float | None] = {}
    for target in recall_targets:
        valid_rows = [row for row in sweep_rows if row.eligible_state_recall >= target]
        if not valid_rows:
            recommended[f"{target:.2f}"] = None
            continue
        best = max(
            valid_rows,
            key=lambda row: (
                row.frontier_edge_prune_rate,
                row.threshold,
            ),
        )
        recommended[f"{target:.2f}"] = best.threshold
    return recommended


def build_histogram_rows(
    *,
    dataset: str,
    state_records: tuple[FrontierStateRecord, ...],
    bin_width: float = 0.05,
) -> list[dict[str, Any]]:
    positive_scores, negative_scores = flatten_scores(state_records)
    rows: list[dict[str, Any]] = []
    for label, scores in (("positive", positive_scores), ("negative", negative_scores)):
        rows.extend(
            _histogram_rows_for_scores(
                dataset=dataset,
                label=label,
                scores=scores,
                bin_width=bin_width,
            )
        )
    return rows


def write_outputs(
    *,
    output_dir: Path,
    summaries: list[DatasetSummary],
    sweep_rows: list[SweepRow],
    histogram_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        summary.dataset: {
            **asdict(summary),
        }
        for summary in summaries
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "threshold_sweep.csv",
        rows=[asdict(row) for row in sweep_rows],
        fieldnames=[
            "dataset",
            "threshold",
            "eligible_state_recall",
            "all_positive_state_recall",
            "states_with_some_gold_dropped_rate",
            "states_with_no_edges_left_rate",
            "positive_edge_recall",
            "frontier_edge_prune_rate",
            "negative_edge_prune_rate",
            "mean_kept_edges_per_eligible_state",
        ],
    )
    _write_csv(
        output_dir / "score_histogram.csv",
        rows=histogram_rows,
        fieldnames=[
            "dataset",
            "label",
            "bin_left",
            "bin_right",
            "count",
            "density",
        ],
    )


def threshold_grid(*, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be positive.")
    count = int(round(2.0 / step))
    return [(-1.0 + idx * step) for idx in range(count + 1)]


def _collect_dataset_scores(
    *,
    dataset: str,
    ds: RetrievalDataset,
    collator: RetrievalCollator,
    relation_semantic_table: torch.Tensor,
    max_samples: int | None,
) -> DatasetScoreCollection:
    state_records: list[FrontierStateRecord] = []
    skipped_empty_frontier = 0
    skipped_no_gold_frontier = 0
    replay_trajectory_count = 0
    limit = len(ds) if max_samples is None else min(len(ds), max_samples)
    for idx in range(limit):
        sample = ds.get(idx)
        batch = collator([sample])
        sample_collection = collect_frontier_state_records(
            batch=batch,
            relation_semantic_table=relation_semantic_table,
        )
        state_records.extend(sample_collection.state_records)
        skipped_empty_frontier += sample_collection.skipped_empty_frontier
        skipped_no_gold_frontier += sample_collection.skipped_no_gold_frontier
        replay_trajectory_count += sample_collection.replay_trajectory_count
    return DatasetScoreCollection(
        dataset=dataset,
        state_records=tuple(state_records),
        skipped_empty_frontier=skipped_empty_frontier,
        skipped_no_gold_frontier=skipped_no_gold_frontier,
        replay_trajectory_count=replay_trajectory_count,
        sample_count=limit,
    )


def _distribution_stats(scores: list[float]) -> DistributionStats:
    if not scores:
        nan = float("nan")
        return DistributionStats(
            count=0,
            mean=0.0,
            std=0.0,
            min=nan,
            max=nan,
            median=nan,
            p05=nan,
            p25=nan,
            p50=nan,
            p75=nan,
            p95=nan,
        )
    values = torch.tensor(scores, dtype=torch.float64)
    return DistributionStats(
        count=int(values.numel()),
        mean=float(values.mean().item()),
        std=float(values.std(unbiased=False).item()),
        min=float(values.min().item()),
        max=float(values.max().item()),
        median=float(values.median().item()),
        p05=float(torch.quantile(values, 0.05).item()),
        p25=float(torch.quantile(values, 0.25).item()),
        p50=float(torch.quantile(values, 0.50).item()),
        p75=float(torch.quantile(values, 0.75).item()),
        p95=float(torch.quantile(values, 0.95).item()),
    )


def _histogram_rows_for_scores(
    *,
    dataset: str,
    label: str,
    scores: list[float],
    bin_width: float,
) -> list[dict[str, Any]]:
    edges = threshold_grid(step=bin_width)
    counts = [0 for _ in range(len(edges) - 1)]
    for score in scores:
        clipped = min(max(score, -1.0), 1.0)
        if math.isclose(clipped, 1.0):
            index = len(counts) - 1
        else:
            index = int(math.floor((clipped + 1.0) / bin_width))
            index = max(0, min(index, len(counts) - 1))
        counts[index] += 1
    total = sum(counts)
    rows: list[dict[str, Any]] = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "bin_left": round(edges[idx], 6),
                "bin_right": round(edges[idx + 1], 6),
                "count": count,
                "density": float(count / total) if total > 0 else 0.0,
            }
        )
    return rows


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _write_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "AnalysisResult",
    "DatasetScoreCollection",
    "DatasetSummary",
    "FrontierStateRecord",
    "SweepRow",
    "build_histogram_rows",
    "collect_frontier_state_records",
    "flatten_scores",
    "recommend_thresholds",
    "replay_prefix_trajectories",
    "run_analysis",
    "summarize_collection",
    "sweep_thresholds",
    "write_outputs",
]
