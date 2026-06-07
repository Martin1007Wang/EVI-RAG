from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.frontier_relation_similarity import (
    DatasetScoreCollection,
    DatasetSummary,
    FrontierStateRecord,
    build_histogram_rows,
    run_analysis,
    summarize_collection,
    sweep_thresholds,
    write_outputs,
)


def main() -> None:
    args = _parse_args()
    summaries: list[DatasetSummary] = []
    sweep_rows = []
    histogram_rows = []
    combined_state_records: list[FrontierStateRecord] = []

    for dataset in args.datasets:
        result = run_analysis(
            dataset=dataset,
            data_root=args.data_root,
            split=args.split,
            max_samples=args.max_samples,
            sweep_step=args.sweep_step,
        )
        summary = result.summary
        summaries.append(summary)
        sweep_rows.extend(result.sweep_rows)
        histogram_rows.extend(result.histogram_rows)
        combined_state_records.extend(result.collection.state_records)
        _print_summary(summary)

    combined_collection = DatasetScoreCollection(
        dataset="combined",
        state_records=tuple(combined_state_records),
        skipped_empty_frontier=sum(summary.skipped_empty_frontier for summary in summaries),
        skipped_no_gold_frontier=sum(summary.skipped_no_gold_frontier for summary in summaries),
        replay_trajectory_count=sum(summary.replay_trajectory_count for summary in summaries),
        sample_count=sum(summary.sample_count for summary in summaries),
    )
    combined_summary = summarize_collection(
        combined_collection,
        sweep_step=args.sweep_step,
    )
    summaries.append(combined_summary)
    sweep_rows.extend(
        sweep_thresholds(
            dataset="combined",
            state_records=tuple(combined_state_records),
            step=args.sweep_step,
        )
    )
    histogram_rows.extend(
        build_histogram_rows(
            dataset="combined",
            state_records=tuple(combined_state_records),
        )
    )
    _print_summary(combined_summary)
    write_outputs(
        output_dir=args.output_dir,
        summaries=summaries,
        sweep_rows=sweep_rows,
        histogram_rows=histogram_rows,
    )


def _print_summary(summary: DatasetSummary) -> None:
    print(
        (
            f"{summary.dataset}: "
            f"states={summary.eligible_state_count}, "
            f"pos_edges={summary.positive_edge_count}, "
            f"neg_edges={summary.negative_edge_count}, "
            f"pos_mean={summary.positive_score_stats.mean:.4f}, "
            f"neg_mean={summary.negative_score_stats.mean:.4f}, "
            f"thr@0.95={summary.recommended_thresholds.get('0.95')}"
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze relation-query similarity on replay frontier edges.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["webqsp", "cwq"],
    )
    parser.add_argument(
        "--split",
        default="validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/frontier_relation_similarity"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
