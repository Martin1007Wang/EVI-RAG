from __future__ import annotations

# %% [markdown]
# # WebQSP / CWQ Answer Diversity and Path Diversity
#
# This notebook-style script measures two forms of diversity:
# 1. Ground-truth answer diversity: how many gold answers each question has.
# 2. Shortest-path diversity: how many distinct shortest directed paths in the
#    provided RoG Freebase subgraph connect any topic entity to any gold answer.
#
# Assumptions used in this analysis:
# - Answer diversity is counted from the unique non-empty strings in `answer`.
# - Path diversity is counted inside each question's provided `graph`, not the
#   full global Freebase. This matches the data shipped with the repo and keeps
#   the notebook fully offline.
# - A path is a shortest simple directed path of 1-3 triples.
# - Duplicate triples inside one question graph are removed before counting
#   paths, so repeated retrieval copies do not inflate diversity.
# - Different paths are distinguished by exact triple sequence
#   `(head, relation, tail)`.
#
# If you want a quick dry run first, set `MAX_QUESTIONS_PER_SPLIT` to a small
# number like 100 in the execution cells near the bottom.

# %%
import os
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from datasets import Dataset
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:  # pragma: no cover - display fallback for plain Python

    def display(obj: object) -> None:
        print(obj)


def _project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return Path.cwd()


PROJECT_ROOT = _project_root()
CONFIG_DIR = PROJECT_ROOT / "configs" / "dataset"
HF_DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", "/mnt/data/huggingface/datasets"))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "answer_path_diversity"
DEFAULT_DATASETS = ("webqsp", "cwq")
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_MAX_HOPS = 3
PLOT_COLORS = {
    "webqsp": "#2E5AAC",
    "cwq": "#C24E33",
}

pd.options.display.max_colwidth = 120
sns.set_theme(style="whitegrid", context="talk")


# %% [markdown]
# ## Dataset loading helpers


# %%
@lru_cache(maxsize=None)
def load_dataset_config(dataset_name: str) -> dict[str, object]:
    config_path = CONFIG_DIR / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def resolve_dataset_cache_dir(dataset_name: str) -> Path:
    cfg = load_dataset_config(dataset_name)
    revision = str(cfg["hf_revision"]).strip()
    matches = sorted(HF_DATASETS_CACHE.glob(f"*/default/0.0.0/{revision}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a local HuggingFace cache directory for {dataset_name!r} " f"under {HF_DATASETS_CACHE}. Expected revision={revision}."
        )
    return matches[0]


@lru_cache(maxsize=None)
def list_split_arrow_files(dataset_name: str, split: str) -> tuple[Path, ...]:
    cache_dir = resolve_dataset_cache_dir(dataset_name)
    files = tuple(sorted(path for path in cache_dir.glob(f"*{split}*.arrow") if path.is_file() and not path.name.startswith("cache-")))
    if not files:
        raise FileNotFoundError(f"No materialized arrow files found for dataset={dataset_name!r}, split={split!r} " f"in {cache_dir}.")
    return files


@lru_cache(maxsize=None)
def count_split_rows(dataset_name: str, split: str) -> int:
    return sum(len(Dataset.from_file(str(path))) for path in list_split_arrow_files(dataset_name, split))


def iter_split_rows(
    dataset_name: str,
    split: str,
    *,
    max_questions: int | None = None,
) -> Iterator[dict[str, object]]:
    yielded = 0
    for arrow_path in list_split_arrow_files(dataset_name, split):
        dataset = Dataset.from_file(str(arrow_path))
        for row in dataset:
            yield row
            yielded += 1
            if max_questions is not None and yielded >= max_questions:
                return


def total_rows_for_run(
    dataset_name: str,
    split: str,
    *,
    max_questions: int | None = None,
) -> int:
    if max_questions is None:
        return count_split_rows(dataset_name, split)
    return min(count_split_rows(dataset_name, split), int(max_questions))


# %% [markdown]
# ## Core parsing helpers

# %%
Triple = tuple[str, str, str]


def unique_nonempty(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        values = [values]
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def normalize_triple(raw: object) -> Triple | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    head = str(raw[0]).strip()
    relation = str(raw[1]).strip()
    tail = str(raw[2]).strip()
    if not head or not relation or not tail:
        return None
    return (head, relation, tail)


def deduplicate_triples(graph: object) -> list[Triple]:
    if not isinstance(graph, (list, tuple)):
        return []
    seen: set[Triple] = set()
    triples: list[Triple] = []
    for raw in graph:
        triple = normalize_triple(raw)
        if triple is None or triple in seen:
            continue
        seen.add(triple)
        triples.append(triple)
    return triples


def build_adjacency(triples: Sequence[Triple]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    forward: dict[str, list[int]] = defaultdict(list)
    reverse: dict[str, list[int]] = defaultdict(list)
    for edge_id, (head, _relation, tail) in enumerate(triples):
        forward[head].append(edge_id)
        reverse[tail].append(edge_id)
    return forward, reverse


# %% [markdown]
# ## Dimension 1: answer diversity


# %%
def compute_answer_diversity(
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_questions_per_split: int | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name in datasets:
        for split in splits:
            total = total_rows_for_run(dataset_name, split, max_questions=max_questions_per_split)
            iterator = iter_split_rows(dataset_name, split, max_questions=max_questions_per_split)
            for row in tqdm(iterator, total=total, desc=f"answers::{dataset_name}/{split}"):
                answers = unique_nonempty(row.get("answer"))
                answer_entities = unique_nonempty(row.get("a_entity"))
                records.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "question_id": str(row.get("id", "")).strip(),
                        "question": str(row.get("question", "")).strip(),
                        "answer_count": len(answers),
                        "answer_entity_count": len(answer_entities),
                        "is_multi_answer": len(answers) > 1,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_answer_diversity(
    answer_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("dataset",),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in answer_df.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group["answer_count"].astype(int)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "questions": int(len(group)),
                "mean_answers": float(counts.mean()),
                "median_answers": float(counts.median()),
                "p90_answers": float(counts.quantile(0.90)),
                "max_answers": int(counts.max()),
                "multi_answer_rate": float((counts > 1).mean()),
                "answer_count_gt_1": int((counts > 1).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# %% [markdown]
# ## Dimension 2: shortest 1-3 hop path diversity


# %%
def bounded_reverse_distances(
    triples: Sequence[Triple],
    reverse_adjacency: dict[str, list[int]],
    answers: Iterable[str],
    *,
    max_hops: int,
) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: deque[str] = deque()
    for answer in answers:
        if answer in distances:
            continue
        distances[answer] = 0
        queue.append(answer)
    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        if current_distance >= max_hops:
            continue
        for edge_id in reverse_adjacency.get(node, ()):
            predecessor = triples[edge_id][0]
            next_distance = current_distance + 1
            if predecessor in distances and distances[predecessor] <= next_distance:
                continue
            distances[predecessor] = next_distance
            if next_distance < max_hops:
                queue.append(predecessor)
    return distances


def _collect_shortest_paths_for_pair(
    *,
    anchor: str,
    answer: str,
    triples: Sequence[Triple],
    forward_adjacency: dict[str, list[int]],
    reverse_adjacency: dict[str, list[int]],
    max_hops: int,
) -> set[tuple[int, ...]]:
    reverse_distances = bounded_reverse_distances(
        triples,
        reverse_adjacency,
        (answer,),
        max_hops=max_hops,
    )
    shortest_length = reverse_distances.get(anchor)
    if shortest_length is None or shortest_length <= 0 or shortest_length > max_hops:
        return set()

    shortest_paths: set[tuple[int, ...]] = set()
    stack: list[tuple[str, tuple[int, ...], frozenset[str]]] = [(anchor, tuple(), frozenset((anchor,)))]
    while stack:
        node, edge_path, visited_nodes = stack.pop()
        depth = len(edge_path)
        distance_to_answer = reverse_distances.get(node)
        if distance_to_answer is None:
            continue
        if node == answer and depth == shortest_length:
            shortest_paths.add(edge_path)
            continue
        if depth >= shortest_length:
            continue

        for edge_id in forward_adjacency.get(node, ()):
            _head, _relation, tail = triples[edge_id]
            if tail in visited_nodes:
                continue
            tail_distance = reverse_distances.get(tail)
            if tail_distance is None or tail_distance != distance_to_answer - 1:
                continue
            stack.append(
                (
                    tail,
                    edge_path + (edge_id,),
                    visited_nodes | {tail},
                )
            )
    return shortest_paths


def count_shortest_answer_paths(
    row: dict[str, object],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> int:
    anchors = unique_nonempty(row.get("q_entity"))
    answers = unique_nonempty(row.get("a_entity"))
    triples = deduplicate_triples(row.get("graph"))
    if not anchors or not answers or not triples:
        return 0

    forward_adjacency, reverse_adjacency = build_adjacency(triples)
    shortest_paths: set[tuple[int, ...]] = set()
    for anchor in anchors:
        for answer in answers:
            shortest_paths.update(
                _collect_shortest_paths_for_pair(
                    anchor=anchor,
                    answer=answer,
                    triples=triples,
                    forward_adjacency=forward_adjacency,
                    reverse_adjacency=reverse_adjacency,
                    max_hops=max_hops,
                )
            )
    return len(shortest_paths)


def compute_path_diversity(
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_questions_per_split: int | None = None,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name in datasets:
        for split in splits:
            total = total_rows_for_run(dataset_name, split, max_questions=max_questions_per_split)
            iterator = iter_split_rows(dataset_name, split, max_questions=max_questions_per_split)
            for row in tqdm(iterator, total=total, desc=f"paths::{dataset_name}/{split}"):
                path_count = count_shortest_answer_paths(row, max_hops=max_hops)
                records.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "question_id": str(row.get("id", "")).strip(),
                        "question": str(row.get("question", "")).strip(),
                        "anchor_count": len(unique_nonempty(row.get("q_entity"))),
                        "answer_count": len(unique_nonempty(row.get("answer"))),
                        "answer_entity_count": len(unique_nonempty(row.get("a_entity"))),
                        "graph_triple_count": len(deduplicate_triples(row.get("graph"))),
                        "shortest_path_count": int(path_count),
                        "has_shortest_path": bool(path_count > 0),
                        "has_multiple_shortest_paths": bool(path_count > 1),
                        "point_recall_upper_bound": (1.0 / path_count) if path_count > 0 else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_path_diversity(
    path_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("dataset",),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in path_df.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group["shortest_path_count"].astype(int)
        positive = counts[counts > 0]
        upper_bounds = group["point_recall_upper_bound"].dropna().astype(float)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "questions": int(len(group)),
                "shortest_path_positive_rate": float((counts > 0).mean()),
                "multi_shortest_path_rate_all": float((counts > 1).mean()),
                "multi_shortest_path_rate_given_positive": float((positive > 1).mean()) if len(positive) else np.nan,
                "mean_shortest_paths_given_positive": float(positive.mean()) if len(positive) else np.nan,
                "median_shortest_paths_given_positive": float(positive.median()) if len(positive) else np.nan,
                "p90_shortest_paths_given_positive": float(positive.quantile(0.90)) if len(positive) else np.nan,
                "max_shortest_paths": int(counts.max()),
                "mean_point_recall_upper_bound": float(upper_bounds.mean()) if len(upper_bounds) else np.nan,
                "median_point_recall_upper_bound": float(upper_bounds.median()) if len(upper_bounds) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# %% [markdown]
# ## Visualization helpers


# %%
def _dataset_color(dataset_name: str) -> str:
    return PLOT_COLORS.get(dataset_name, "#4C4C4C")


def _maybe_2d_axes(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[None, :]
    return axes


def plot_bucketed_distribution(
    ax: plt.Axes,
    series: pd.Series,
    *,
    max_exact_count: int,
    color: str,
    title: str,
    xlabel: str,
) -> None:
    values = series.dropna().astype(int)
    if values.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [str(i) for i in range(0, max_exact_count + 1)] + [f">{max_exact_count}"]
    fractions = [float((values == i).mean()) for i in range(0, max_exact_count + 1)]
    fractions.append(float((values > max_exact_count).mean()))

    ax.bar(labels, fractions, color=color, alpha=0.9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of questions")
    ax.tick_params(axis="x", rotation=0)


def plot_ecdf(
    ax: plt.Axes,
    series: pd.Series,
    *,
    color: str,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    values = np.sort(series.dropna().astype(float).to_numpy())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative share")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    y = np.arange(1, values.size + 1) / values.size
    ax.step(values, y, where="post", color=color, linewidth=2.5)
    if log_x:
        ax.set_xscale("log")


def plot_binary_ratio(
    ax: plt.Axes,
    bool_series: pd.Series,
    *,
    color: str,
    title: str,
    true_label: str,
    false_label: str,
) -> None:
    values = bool_series.dropna().astype(bool)
    fractions = {
        false_label: float((~values).mean()) if len(values) else 0.0,
        true_label: float(values.mean()) if len(values) else 0.0,
    }
    ax.bar(list(fractions.keys()), list(fractions.values()), color=[color, color], alpha=0.85)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Share of questions")


def plot_upper_bound_histogram(
    ax: plt.Axes,
    upper_bound_series: pd.Series,
    *,
    color: str,
    title: str,
) -> None:
    upper_bounds = upper_bound_series.dropna().astype(float).to_numpy()
    ax.set_title(title)
    ax.set_xlabel("Shortest-path recall upper bound (1 / k)")
    ax.set_ylabel("Share of shortest-path-positive questions")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if upper_bounds.size == 0:
        ax.text(0.5, 0.5, "No shortest-path-positive questions", ha="center", va="center", transform=ax.transAxes)
        return
    weights = np.full(upper_bounds.shape[0], 1.0 / upper_bounds.shape[0])
    ax.hist(
        upper_bounds,
        bins=np.linspace(0.0, 1.0, 11),
        weights=weights,
        color=color,
        alpha=0.9,
        edgecolor="white",
    )
    ax.axvline(upper_bounds.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={upper_bounds.mean():.3f}")
    ax.axvline(np.median(upper_bounds), color="black", linestyle=":", linewidth=1.5, label=f"median={np.median(upper_bounds):.3f}")
    ax.legend(frameon=False, fontsize=11)


def plot_answer_diversity(answer_df: pd.DataFrame, *, scope_label: str = "all splits") -> plt.Figure:
    datasets = list(dict.fromkeys(answer_df["dataset"].tolist()))
    fig, axes = plt.subplots(len(datasets), 3, figsize=(18, 4.6 * len(datasets)), constrained_layout=True)
    axes = _maybe_2d_axes(axes)

    for row_idx, dataset_name in enumerate(datasets):
        subset = answer_df.loc[answer_df["dataset"] == dataset_name].copy()
        color = _dataset_color(dataset_name)

        plot_bucketed_distribution(
            axes[row_idx, 0],
            subset["answer_count"],
            max_exact_count=8,
            color=color,
            title=f"{dataset_name.upper()} answer-count distribution",
            xlabel="Unique gold answers per question",
        )
        plot_ecdf(
            axes[row_idx, 1],
            subset["answer_count"],
            color=color,
            title=f"{dataset_name.upper()} answer-count ECDF",
            xlabel="Unique gold answers per question",
        )
        plot_binary_ratio(
            axes[row_idx, 2],
            subset["is_multi_answer"],
            color=color,
            title=f"{dataset_name.upper()} single vs multi-answer",
            true_label="> 1 answer",
            false_label="1 answer",
        )

        counts = subset["answer_count"].astype(int)
        summary_text = (
            f"n = {len(subset):,}\n" f"mean = {counts.mean():.2f}\n" f"median = {counts.median():.0f}\n" f"multi-answer = {(counts > 1).mean():.1%}"
        )
        axes[row_idx, 0].text(
            0.98,
            0.97,
            summary_text,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    fig.suptitle(f"Answer diversity in WebQSP / CWQ ({scope_label})", y=1.02, fontsize=22)
    return fig


def plot_path_diversity(path_df: pd.DataFrame, *, scope_label: str = "all splits") -> plt.Figure:
    datasets = list(dict.fromkeys(path_df["dataset"].tolist()))
    fig, axes = plt.subplots(len(datasets), 3, figsize=(18, 4.8 * len(datasets)), constrained_layout=True)
    axes = _maybe_2d_axes(axes)

    for row_idx, dataset_name in enumerate(datasets):
        subset = path_df.loc[path_df["dataset"] == dataset_name].copy()
        color = _dataset_color(dataset_name)
        positive_paths = subset.loc[subset["shortest_path_count"] > 0, "shortest_path_count"]

        plot_bucketed_distribution(
            axes[row_idx, 0],
            subset["shortest_path_count"],
            max_exact_count=10,
            color=color,
            title=f"{dataset_name.upper()} shortest-path distribution",
            xlabel="Distinct shortest 1-3 hop paths per question",
        )
        plot_ecdf(
            axes[row_idx, 1],
            positive_paths,
            color=color,
            title=f"{dataset_name.upper()} shortest-path ECDF",
            xlabel="Distinct shortest 1-3 hop paths per question",
            log_x=True,
        )
        plot_upper_bound_histogram(
            axes[row_idx, 2],
            subset["point_recall_upper_bound"],
            color=color,
            title=f"{dataset_name.upper()} shortest-path upper bound",
        )

        counts = subset["shortest_path_count"].astype(int)
        positive = counts[counts > 0]
        upper = subset["point_recall_upper_bound"].dropna()
        summary_text = (
            f"n = {len(subset):,}\n"
            f"shortest-path-positive = {(counts > 0).mean():.1%}\n"
            f"multi-shortest-path = {(counts > 1).mean():.1%}\n"
            f"mean k | k>0 = {positive.mean():.2f}"
            if len(positive)
            else f"n = {len(subset):,}\nshortest-path-positive = 0.0%"
        )
        if len(upper):
            summary_text = summary_text + f"\nmean 1/k = {upper.mean():.3f}"
        axes[row_idx, 0].text(
            0.98,
            0.97,
            summary_text,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    fig.suptitle(f"Shortest-path diversity in WebQSP / CWQ ({scope_label})", y=1.02, fontsize=22)
    return fig


def save_figure(fig: plt.Figure, filename: str, *, dpi: int = 220) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# %% [markdown]
# ## Execution cells
#
# Full-dataset run:
# - `RUN_SPLITS = ("train", "validation", "test")`
# - `MAX_QUESTIONS_PER_SPLIT = None`
#
# Quick smoke test:
# - `RUN_SPLITS = ("validation",)`
# - `MAX_QUESTIONS_PER_SPLIT = 100`

# %%
RUN_DATASETS = DEFAULT_DATASETS
RUN_SPLITS = DEFAULT_SPLITS
MAX_QUESTIONS_PER_SPLIT = None
MAX_HOPS = DEFAULT_MAX_HOPS


# %%
def run_default_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    answer_df = compute_answer_diversity(
        datasets=RUN_DATASETS,
        splits=RUN_SPLITS,
        max_questions_per_split=MAX_QUESTIONS_PER_SPLIT,
    )

    answer_dataset_summary = summarize_answer_diversity(answer_df, group_cols=("dataset",))
    answer_split_summary = summarize_answer_diversity(answer_df, group_cols=("dataset", "split"))

    display(answer_dataset_summary.sort_values(["dataset"]).reset_index(drop=True))
    display(answer_split_summary.sort_values(["dataset", "split"]).reset_index(drop=True))

    answer_fig = plot_answer_diversity(answer_df, scope_label=", ".join(RUN_SPLITS))
    display(answer_fig)
    save_figure(answer_fig, "answer_diversity.png")

    path_df = compute_path_diversity(
        datasets=RUN_DATASETS,
        splits=RUN_SPLITS,
        max_questions_per_split=MAX_QUESTIONS_PER_SPLIT,
        max_hops=MAX_HOPS,
    )

    path_dataset_summary = summarize_path_diversity(path_df, group_cols=("dataset",))
    path_split_summary = summarize_path_diversity(path_df, group_cols=("dataset", "split"))

    display(path_dataset_summary.sort_values(["dataset"]).reset_index(drop=True))
    display(path_split_summary.sort_values(["dataset", "split"]).reset_index(drop=True))

    path_fig = plot_path_diversity(path_df, scope_label=", ".join(RUN_SPLITS))
    display(path_fig)
    save_figure(path_fig, "path_diversity.png")

    top_ambiguous_questions = (
        path_df.sort_values(["shortest_path_count", "graph_triple_count"], ascending=[False, False])
        .loc[:, ["dataset", "split", "question_id", "question", "shortest_path_count", "point_recall_upper_bound"]]
        .head(20)
        .reset_index(drop=True)
    )
    display(top_ambiguous_questions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    answer_df.to_csv(OUTPUT_DIR / "answer_diversity_records.csv", index=False)
    path_df.to_csv(OUTPUT_DIR / "path_diversity_records.csv", index=False)
    answer_dataset_summary.to_csv(OUTPUT_DIR / "summary_answer_by_dataset.csv", index=False)
    answer_split_summary.to_csv(OUTPUT_DIR / "summary_answer_by_split.csv", index=False)
    path_dataset_summary.to_csv(OUTPUT_DIR / "summary_path_by_dataset.csv", index=False)
    path_split_summary.to_csv(OUTPUT_DIR / "summary_path_by_split.csv", index=False)
    return answer_df, path_df


# %%
if __name__ == "__main__":
    run_default_analysis()
