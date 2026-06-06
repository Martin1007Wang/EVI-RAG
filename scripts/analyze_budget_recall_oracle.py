from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.artifacts import load_materialization_artifact
from src.data.dataset import RetrievalDataset


DEFAULT_METADATA_DIR = Path("/mnt/data/retrieval/webqsp/metadata")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/budget_recall_oracle")
UNREACHABLE = -1


@dataclass(frozen=True, slots=True)
class PathCandidate:
    edge_ids: frozenset[int]
    edge_bits: int
    covered_target_bits: int
    target_pos: int


@dataclass(frozen=True, slots=True)
class SampleOracle:
    sample_id: str
    target_count: int
    candidate_count: int
    path_truncated: bool
    exact: bool
    dp_fallback: bool
    recall_by_budget: dict[int, float]
    covered_by_budget: dict[int, int]
    used_edges_by_budget: dict[int, int]
    b_hit: float
    b_cover50: float
    b_cover80: float
    b_cover100: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute shortest-path budget-recall oracle curves for materialized retrieval splits.",
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--budgets", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split.")
    parser.add_argument("--max-paths-per-target", type=int, default=64)
    parser.add_argument("--max-dp-states", type=int, default=200_000)
    parser.add_argument("--answer-weight", type=float, default=4.0)
    parser.add_argument("--edge-cost", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = parse_int_list(args.budgets, name="budgets")
    splits = [split.strip() for split in str(args.splits).split(",") if split.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split.")

    artifact = load_materialization_artifact(args.metadata_dir)
    if artifact is None:
        raise FileNotFoundError(f"Materialization manifest not found under {args.metadata_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, object]] = []
    cover_rows: list[dict[str, object]] = []
    per_sample_curve_rows: list[dict[str, object]] = []
    per_sample_cover_rows: list[dict[str, object]] = []

    for split in splits:
        dataset = RetrievalDataset(
            materialization=artifact,
            split=split,
            lmdb_readahead=False,
            max_readers=64,
        )
        try:
            end = len(dataset)
            if int(args.max_samples) > 0:
                end = min(end, int(args.start_idx) + int(args.max_samples))
            indices = range(int(args.start_idx), end)
            split_results: list[SampleOracle] = []
            for idx in indices:
                sample = dataset[int(idx)]
                result = analyze_sample(
                    sample=sample,
                    budgets=budgets,
                    max_paths_per_target=int(args.max_paths_per_target),
                    max_dp_states=int(args.max_dp_states),
                )
                split_results.append(result)
                per_sample_curve_rows.extend(
                    per_sample_curve_records(
                        dataset=artifact.provenance.get("dataset", {}).get("name", "unknown")
                        if artifact.provenance is not None
                        else "unknown",
                        split=split,
                        row_idx=int(idx),
                        result=result,
                        budgets=budgets,
                    )
                )
                per_sample_cover_rows.append(
                    per_sample_cover_record(
                        dataset=artifact.provenance.get("dataset", {}).get("name", "unknown")
                        if artifact.provenance is not None
                        else "unknown",
                        split=split,
                        row_idx=int(idx),
                        result=result,
                    )
                )

            dataset_name = (
                artifact.provenance.get("dataset", {}).get("name", "unknown")
                if artifact.provenance is not None
                else "unknown"
            )
            curve_rows.extend(
                summarize_curve(
                    dataset=dataset_name,
                    split=split,
                    results=split_results,
                    budgets=budgets,
                    answer_weight=float(args.answer_weight),
                    edge_cost=float(args.edge_cost),
                )
            )
            cover_rows.append(
                summarize_cover(dataset=dataset_name, split=split, results=split_results)
            )
        finally:
            dataset.close()

    write_csv(output_dir / "budget_curve_summary.csv", curve_rows)
    write_csv(output_dir / "budget_cover_summary.csv", cover_rows)
    write_csv(output_dir / "per_sample_budget_curve.csv", per_sample_curve_rows)
    write_csv(output_dir / "per_sample_cover_stats.csv", per_sample_cover_rows)
    write_json(
        output_dir / "run_config.json",
        {
            "metadata_dir": str(args.metadata_dir),
            "splits": splits,
            "budgets": budgets,
            "start_idx": int(args.start_idx),
            "max_samples": int(args.max_samples),
            "max_paths_per_target": int(args.max_paths_per_target),
            "max_dp_states": int(args.max_dp_states),
            "answer_weight": float(args.answer_weight),
            "edge_cost": float(args.edge_cost),
        },
    )

    print(f"wrote {output_dir / 'budget_curve_summary.csv'}")
    print(f"wrote {output_dir / 'budget_cover_summary.csv'}")
    print(f"wrote {output_dir / 'per_sample_budget_curve.csv'}")
    print(f"wrote {output_dir / 'per_sample_cover_stats.csv'}")


def analyze_sample(
    *,
    sample,
    budgets: list[int],
    max_paths_per_target: int,
    max_dp_states: int,
) -> SampleOracle:
    target_count = int(sample.reachable_target_node_ids.numel())
    max_budget = max(budgets) if budgets else 0
    if target_count == 0:
        return SampleOracle(
            sample_id=str(sample.sample_id),
            target_count=0,
            candidate_count=0,
            path_truncated=False,
            exact=True,
            dp_fallback=False,
            recall_by_budget={budget: 0.0 for budget in budgets},
            covered_by_budget={budget: 0 for budget in budgets},
            used_edges_by_budget={budget: 0 for budget in budgets},
            b_hit=math.nan,
            b_cover50=math.nan,
            b_cover80=math.nan,
            b_cover100=math.nan,
        )

    node_target_distances_flat = getattr(
        sample,
        "node_target_distances_flat",
        getattr(sample, "node_target_distance", None),
    )
    if node_target_distances_flat is None:
        raise AttributeError("sample must provide node_target_distances_flat or node_target_distance.")

    candidates, initial_target_bits, path_truncated = build_path_candidates(
        edge_index=sample.edge_index,
        anchor_node_ids=sample.anchor_node_ids,
        reachable_target_node_ids=sample.reachable_target_node_ids,
        node_target_distances_flat=node_target_distances_flat,
        num_nodes=int(sample.num_nodes),
        max_paths_per_target=int(max_paths_per_target),
    )

    solve_budgets = list(range(max_budget + 1))
    exact_result = exact_budget_curve(
        candidates=candidates,
        initial_target_bits=initial_target_bits,
        target_count=target_count,
        budgets=solve_budgets,
        max_dp_states=int(max_dp_states),
    )
    dp_fallback = exact_result is None
    if exact_result is None:
        recall_by_budget, covered_by_budget, used_edges_by_budget = greedy_budget_curve(
            candidates=candidates,
            initial_target_bits=initial_target_bits,
            target_count=target_count,
            budgets=solve_budgets,
        )
    else:
        recall_by_budget, covered_by_budget, used_edges_by_budget = exact_result

    return SampleOracle(
        sample_id=str(sample.sample_id),
        target_count=target_count,
        candidate_count=len(candidates),
        path_truncated=path_truncated,
        exact=not dp_fallback,
        dp_fallback=dp_fallback,
        recall_by_budget=recall_by_budget,
        covered_by_budget=covered_by_budget,
        used_edges_by_budget=used_edges_by_budget,
        b_hit=first_budget_at_recall(
            recall_by_budget=recall_by_budget,
            target=1.0 / float(target_count),
            budgets=solve_budgets,
        ),
        b_cover50=first_budget_at_recall(
            recall_by_budget=recall_by_budget,
            target=0.5,
            budgets=solve_budgets,
        ),
        b_cover80=first_budget_at_recall(
            recall_by_budget=recall_by_budget,
            target=0.8,
            budgets=solve_budgets,
        ),
        b_cover100=first_budget_at_recall(
            recall_by_budget=recall_by_budget,
            target=1.0,
            budgets=solve_budgets,
        ),
    )


def build_path_candidates(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    reachable_target_node_ids: torch.Tensor,
    node_target_distances_flat: torch.Tensor,
    num_nodes: int,
    max_paths_per_target: int,
) -> tuple[list[PathCandidate], int, bool]:
    targets = [int(x) for x in reachable_target_node_ids.view(-1).tolist()]
    target_pos_by_node = {node: pos for pos, node in enumerate(targets)}
    anchors = [int(x) for x in anchor_node_ids.view(-1).tolist()]
    outgoing = outgoing_edges_by_src(edge_index=edge_index, num_nodes=int(num_nodes))
    distances = target_distance_matrix(
        edge_index=edge_index,
        targets=targets,
        num_nodes=int(num_nodes),
        node_target_distances_flat=node_target_distances_flat,
    )

    initial_bits = 0
    for anchor in anchors:
        pos = target_pos_by_node.get(anchor)
        if pos is not None:
            initial_bits |= 1 << pos

    candidates: list[PathCandidate] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    truncated = False
    for target_pos in range(len(targets)):
        target_candidates = 0
        anchor_order = sorted(
            (
                int(distances[target_pos, anchor].item()),
                anchor,
            )
            for anchor in anchors
            if 0 <= anchor < int(num_nodes)
            and int(distances[target_pos, anchor].item()) != UNREACHABLE
        )
        for distance, anchor in anchor_order:
            if distance == 0:
                continue
            remaining = int(max_paths_per_target) - target_candidates
            if remaining <= 0:
                truncated = True
                break
            paths, hit_limit = enumerate_shortest_paths(
                outgoing=outgoing,
                distances=distances[target_pos],
                anchor=anchor,
                limit=remaining,
            )
            truncated = truncated or hit_limit
            target_candidates += len(paths)
            for edge_tuple, node_tuple in paths:
                key = (edge_tuple, target_pos)
                if key in seen:
                    continue
                seen.add(key)
                covered_bits = target_bits_for_nodes(
                    nodes=node_tuple,
                    target_pos_by_node=target_pos_by_node,
                )
                candidates.append(
                    PathCandidate(
                        edge_ids=frozenset(edge_tuple),
                        edge_bits=edge_bits(edge_tuple),
                        covered_target_bits=covered_bits,
                        target_pos=target_pos,
                    )
                )
    return candidates, initial_bits, truncated


def enumerate_shortest_paths(
    *,
    outgoing: list[list[tuple[int, int]]],
    distances: torch.Tensor,
    anchor: int,
    limit: int,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], bool]:
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    hit_limit = False

    def dfs(node: int, edge_path: tuple[int, ...], node_path: tuple[int, ...]) -> None:
        nonlocal hit_limit
        if len(out) >= int(limit):
            hit_limit = True
            return
        distance = int(distances[node].item())
        if distance == 0:
            out.append((edge_path, node_path))
            return
        if distance == UNREACHABLE:
            return
        for edge_id, dst in outgoing[node]:
            dst_distance = int(distances[dst].item())
            if dst_distance != distance - 1:
                continue
            dfs(
                dst,
                (*edge_path, int(edge_id)),
                (*node_path, int(dst)),
            )
            if hit_limit:
                return

    dfs(int(anchor), (), (int(anchor),))
    return out, hit_limit


def exact_budget_curve(
    *,
    candidates: list[PathCandidate],
    initial_target_bits: int,
    target_count: int,
    budgets: list[int],
    max_dp_states: int,
) -> tuple[dict[int, float], dict[int, int], dict[int, int]] | None:
    max_budget = max(budgets) if budgets else 0
    states: set[tuple[int, int]] = {(0, int(initial_target_bits))}
    for candidate in candidates:
        next_states = set(states)
        for edge_mask, covered_bits in states:
            new_edge_mask = edge_mask | candidate.edge_bits
            if new_edge_mask.bit_count() > max_budget:
                continue
            new_covered = covered_bits | candidate.covered_target_bits
            next_states.add((new_edge_mask, new_covered))
        if len(next_states) > int(max_dp_states):
            return None
        states = next_states

    best = best_by_budget(
        states=states,
        target_count=target_count,
        budgets=budgets,
    )
    return best


def greedy_budget_curve(
    *,
    candidates: list[PathCandidate],
    initial_target_bits: int,
    target_count: int,
    budgets: list[int],
) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    recalls: dict[int, float] = {}
    covered: dict[int, int] = {}
    used: dict[int, int] = {}
    for budget in budgets:
        edge_mask = 0
        covered_bits = int(initial_target_bits)
        remaining = list(candidates)
        while True:
            best_idx = -1
            best_score = 0.0
            best_new_edge_bits = 0
            best_new_covered = 0
            for idx, candidate in enumerate(remaining):
                candidate_edge_bits = edge_mask | candidate.edge_bits
                if candidate_edge_bits.bit_count() > int(budget):
                    continue
                new_covered = candidate.covered_target_bits & ~covered_bits
                if new_covered == 0:
                    continue
                new_edge_bits = candidate.edge_bits & ~edge_mask
                score = new_covered.bit_count() / max(1, new_edge_bits.bit_count())
                if score > best_score:
                    best_idx = idx
                    best_score = score
                    best_new_edge_bits = new_edge_bits
                    best_new_covered = new_covered
            if best_idx < 0:
                break
            edge_mask |= best_new_edge_bits
            covered_bits |= best_new_covered
            remaining.pop(best_idx)
        covered_count = covered_bits.bit_count()
        recalls[int(budget)] = covered_count / float(target_count)
        covered[int(budget)] = covered_count
        used[int(budget)] = edge_mask.bit_count()
    return recalls, covered, used


def best_by_budget(
    *,
    states: set[tuple[int, int]],
    target_count: int,
    budgets: list[int],
) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    recalls: dict[int, float] = {}
    covered: dict[int, int] = {}
    used: dict[int, int] = {}
    for budget in budgets:
        best_cover = -1
        best_used = 0
        for edge_mask, covered_bits in states:
            edge_count = edge_mask.bit_count()
            if edge_count > int(budget):
                continue
            cover_count = covered_bits.bit_count()
            if cover_count > best_cover or (
                cover_count == best_cover and edge_count < best_used
            ):
                best_cover = cover_count
                best_used = edge_count
        best_cover = max(0, best_cover)
        recalls[int(budget)] = best_cover / float(target_count)
        covered[int(budget)] = best_cover
        used[int(budget)] = best_used
    return recalls, covered, used


def summarize_curve(
    *,
    dataset: str,
    split: str,
    results: list[SampleOracle],
    budgets: list[int],
    answer_weight: float,
    edge_cost: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_mean = 0.0
    for budget in budgets:
        recalls = np.array([r.recall_by_budget[int(budget)] for r in results], dtype=np.float64)
        hits = np.array([float(r.covered_by_budget[int(budget)] > 0) for r in results], dtype=np.float64)
        full = np.array(
            [float(r.target_count > 0 and r.covered_by_budget[int(budget)] == r.target_count) for r in results],
            dtype=np.float64,
        )
        used = np.array([r.used_edges_by_budget[int(budget)] for r in results], dtype=np.float64)
        mean_recall = float(recalls.mean()) if recalls.size else math.nan
        marginal = mean_recall - previous_mean
        previous_mean = mean_recall
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "budget": int(budget),
                "oracle_hit_rate": safe_mean(hits),
                "oracle_recall_mean": mean_recall,
                "oracle_recall_median": safe_quantile(recalls, 0.5),
                "oracle_recall_p90": safe_quantile(recalls, 0.9),
                "oracle_full_cover_rate": safe_mean(full),
                "mean_used_edges": safe_mean(used),
                "marginal_recall_gain": marginal,
                "reward_marginal_gain": float(answer_weight) * marginal - float(edge_cost),
                "sample_count": len(results),
                "exact_sample_rate": mean_bool(not r.dp_fallback for r in results),
                "path_truncated_sample_rate": mean_bool(r.path_truncated for r in results),
                "dp_fallback_sample_rate": mean_bool(r.dp_fallback for r in results),
            }
        )
    return rows


def summarize_cover(
    *,
    dataset: str,
    split: str,
    results: list[SampleOracle],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "sample_count": len(results),
        "b_hit_p50": finite_quantile((r.b_hit for r in results), 0.5),
        "b_hit_p90": finite_quantile((r.b_hit for r in results), 0.9),
        "b_hit_p95": finite_quantile((r.b_hit for r in results), 0.95),
        "b_cover50_p50": finite_quantile((r.b_cover50 for r in results), 0.5),
        "b_cover50_p90": finite_quantile((r.b_cover50 for r in results), 0.9),
        "b_cover80_p50": finite_quantile((r.b_cover80 for r in results), 0.5),
        "b_cover80_p90": finite_quantile((r.b_cover80 for r in results), 0.9),
        "b_cover100_p50": finite_quantile((r.b_cover100 for r in results), 0.5),
        "b_cover100_p90": finite_quantile((r.b_cover100 for r in results), 0.9),
        "b_hit_observed_rate": observed_rate(r.b_hit for r in results),
        "b_cover50_observed_rate": observed_rate(r.b_cover50 for r in results),
        "b_cover80_observed_rate": observed_rate(r.b_cover80 for r in results),
        "b_cover100_observed_rate": observed_rate(r.b_cover100 for r in results),
        "mean_target_count": safe_mean(np.array([r.target_count for r in results], dtype=np.float64)),
        "mean_candidate_count": safe_mean(np.array([r.candidate_count for r in results], dtype=np.float64)),
        "exact_sample_rate": mean_bool(not r.dp_fallback for r in results),
        "path_truncated_sample_rate": mean_bool(r.path_truncated for r in results),
        "dp_fallback_sample_rate": mean_bool(r.dp_fallback for r in results),
    }


def per_sample_curve_records(
    *,
    dataset: str,
    split: str,
    row_idx: int,
    result: SampleOracle,
    budgets: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for budget in budgets:
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "row_idx": int(row_idx),
                "sample_id": result.sample_id,
                "budget": int(budget),
                "recall": result.recall_by_budget[int(budget)],
                "covered": result.covered_by_budget[int(budget)],
                "used_edges": result.used_edges_by_budget[int(budget)],
                "target_count": result.target_count,
                "candidate_count": result.candidate_count,
                "exact": result.exact,
                "path_truncated": result.path_truncated,
                "dp_fallback": result.dp_fallback,
            }
        )
    return rows


def per_sample_cover_record(
    *,
    dataset: str,
    split: str,
    row_idx: int,
    result: SampleOracle,
) -> dict[str, object]:
    row = asdict(result)
    del row["recall_by_budget"]
    del row["covered_by_budget"]
    del row["used_edges_by_budget"]
    row.update({"dataset": dataset, "split": split, "row_idx": int(row_idx)})
    return row


def first_budget_at_recall(
    *,
    recall_by_budget: dict[int, float],
    target: float,
    budgets: Iterable[int],
) -> float:
    for budget in budgets:
        recall = recall_by_budget.get(int(budget))
        if recall is not None and recall + 1.0e-12 >= float(target):
            return float(budget)
    return math.nan


def outgoing_edges_by_src(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[list[tuple[int, int]]]:
    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    for edge_id in range(int(edge_index.size(1))):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        outgoing[src].append((int(edge_id), dst))
    return outgoing


def target_distance_matrix(
    *,
    edge_index: torch.Tensor,
    targets: list[int],
    num_nodes: int,
    node_target_distances_flat: torch.Tensor,
) -> torch.Tensor:
    distances_flat = node_target_distances_flat.to(dtype=torch.long, device="cpu")
    if int(distances_flat.numel()) == len(targets) * int(num_nodes):
        return distances_flat.view(len(targets), int(num_nodes))

    incoming: list[list[int]] = [[] for _ in range(int(num_nodes))]
    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    for edge_id in range(int(edge_index.size(1))):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        incoming[dst].append(src)

    rows: list[torch.Tensor] = []
    for target in targets:
        rows.append(single_target_distances(incoming=incoming, target=int(target)))
    return torch.stack(rows, dim=0) if rows else torch.empty((0, int(num_nodes)), dtype=torch.long)


def single_target_distances(*, incoming: list[list[int]], target: int) -> torch.Tensor:
    distances = [UNREACHABLE] * len(incoming)
    if not (0 <= int(target) < len(incoming)):
        return torch.tensor(distances, dtype=torch.long)

    queue = [int(target)]
    distances[int(target)] = 0
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        next_distance = distances[node] + 1
        for src in incoming[node]:
            if distances[src] != UNREACHABLE:
                continue
            distances[src] = next_distance
            queue.append(src)
    return torch.tensor(distances, dtype=torch.long)


def target_bits_for_nodes(
    *,
    nodes: Iterable[int],
    target_pos_by_node: dict[int, int],
) -> int:
    bits = 0
    for node in nodes:
        pos = target_pos_by_node.get(int(node))
        if pos is not None:
            bits |= 1 << pos
    return bits


def edge_bits(edge_ids: Iterable[int]) -> int:
    bits = 0
    for edge_id in edge_ids:
        bits |= 1 << int(edge_id)
    return bits


def parse_int_list(value: str, *, name: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"{name} must not be empty.")
    if any(x < 0 for x in out):
        raise ValueError(f"{name} must contain non-negative integers.")
    return sorted(dict.fromkeys(out))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(values.mean())


def safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def finite_quantile(values: Iterable[float], q: float) -> float:
    finite = np.array([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return safe_quantile(finite, q)


def observed_rate(values: Iterable[float]) -> float:
    xs = [float(value) for value in values]
    if not xs:
        return math.nan
    return float(sum(math.isfinite(value) for value in xs) / len(xs))


def mean_bool(values: Iterable[bool]) -> float:
    xs = [bool(value) for value in values]
    if not xs:
        return math.nan
    return float(sum(xs) / len(xs))


if __name__ == "__main__":
    main()
