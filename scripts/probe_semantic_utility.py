from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.artifacts import load_materialization_artifact_from_path
from src.data.dataset import RetrievalDataset
from src.data.tensor_table import read_table


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    units: int
    samples_seen: int
    skipped_no_path: int
    skipped_no_random: int
    gt_mean: float
    random_mean: float
    diff_mean: float
    diff_ci95_low: float
    diff_ci95_high: float
    paired_t: float
    normal_approx_p: float
    cohen_dz: float
    gt_greater_rate: float
    gt_median: float
    random_median: float


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    for dataset in args.datasets:
        result = run_dataset(
            dataset=dataset,
            data_root=args.data_root,
            split=args.split,
            max_samples=args.max_samples,
            random_attempts=args.random_attempts,
            rng=rng,
        )
        print(_format_result(result), flush=True)


def run_dataset(
    *,
    dataset: str,
    data_root: Path,
    split: str,
    max_samples: int | None,
    random_attempts: int,
    rng: random.Random,
) -> DatasetResult:
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

    gt_scores: list[float] = []
    random_scores: list[float] = []
    skipped_no_path = 0
    skipped_no_random = 0
    limit = len(ds) if max_samples is None else min(len(ds), max_samples)
    try:
        for idx in range(limit):
            sample = ds.get(idx)
            question = torch.nn.functional.normalize(
                sample.question_emb.float().view(1, -1),
                p=2,
                dim=1,
            ).view(-1)
            target_count = int(sample.reachable_target_node_ids.numel())
            for target_idx in range(target_count):
                gt_path = _ground_truth_path_edges(sample, target_idx=target_idx)
                if not gt_path:
                    skipped_no_path += 1
                    continue
                random_path = _random_non_answer_walk(
                    sample,
                    length=len(gt_path),
                    attempts=random_attempts,
                    rng=rng,
                )
                if not random_path:
                    skipped_no_random += 1
                    continue
                gt_scores.append(
                    _relation_utility(
                        edge_ids=gt_path,
                        edge_relation_ids=sample.edge_relation_catalog_ids,
                        relation_semantic_table=relation_semantic_table,
                        question=question,
                    )
                )
                random_scores.append(
                    _relation_utility(
                        edge_ids=random_path,
                        edge_relation_ids=sample.edge_relation_catalog_ids,
                        relation_semantic_table=relation_semantic_table,
                        question=question,
                    )
                )
    finally:
        ds.close()

    if not gt_scores:
        raise RuntimeError(f"{dataset}: no comparable path pairs were collected")
    return _summarize(
        dataset=dataset,
        samples_seen=limit,
        skipped_no_path=skipped_no_path,
        skipped_no_random=skipped_no_random,
        gt_scores=gt_scores,
        random_scores=random_scores,
    )


def _ground_truth_path_edges(sample: object, *, target_idx: int) -> list[int]:
    num_nodes = int(sample.num_nodes)
    num_edges = int(sample.num_edges)
    target_nodes = sample.reachable_target_node_ids.long()
    target = int(target_nodes[target_idx].item())
    distances = sample.node_target_distances_flat.view(-1, num_nodes)[target_idx].long()
    edge_counts = sample.node_target_shortest_path_edge_count_flat.view(-1, num_edges)[target_idx].float()

    anchors = [int(value) for value in sample.anchor_node_ids.view(-1).tolist()]
    anchors = [anchor for anchor in anchors if 0 <= anchor < num_nodes and int(distances[anchor].item()) >= 0]
    if not anchors:
        return []
    current = min(anchors, key=lambda node: int(distances[node].item()))
    if current == target:
        return []

    src = sample.edge_index[0].long()
    dst = sample.edge_index[1].long()
    path: list[int] = []
    max_steps = int(distances[current].item())
    for _ in range(max_steps):
        current_dist = int(distances[current].item())
        if current_dist <= 0:
            break
        candidate_mask = src.eq(current) & distances.index_select(0, dst).eq(current_dist - 1)
        candidate_ids = candidate_mask.nonzero(as_tuple=False).view(-1)
        if candidate_ids.numel() == 0:
            return []
        counts = edge_counts.index_select(0, candidate_ids)
        best_offset = int(torch.argmax(counts).item())
        edge_id = int(candidate_ids[best_offset].item())
        path.append(edge_id)
        current = int(dst[edge_id].item())
        if current == target:
            return path
    return path if current == target else []


def _random_non_answer_walk(
    sample: object,
    *,
    length: int,
    attempts: int,
    rng: random.Random,
) -> list[int]:
    if length <= 0:
        return []
    src = sample.edge_index[0].long().tolist()
    dst = sample.edge_index[1].long().tolist()
    adjacency: dict[int, list[int]] = {}
    for edge_id, node in enumerate(src):
        adjacency.setdefault(int(node), []).append(edge_id)

    anchors = [int(value) for value in sample.anchor_node_ids.view(-1).tolist()]
    targets = {int(value) for value in sample.reachable_target_node_ids.view(-1).tolist()}
    if not anchors or not targets:
        return []

    for _ in range(attempts):
        current = rng.choice(anchors)
        visited = {current}
        edges: list[int] = []
        for _step in range(length):
            choices = adjacency.get(current)
            if not choices:
                break
            edge_id = rng.choice(choices)
            edges.append(edge_id)
            current = int(dst[edge_id])
            visited.add(current)
        if len(edges) == length and visited.isdisjoint(targets):
            return edges
    return []


def _relation_utility(
    *,
    edge_ids: Iterable[int],
    edge_relation_ids: torch.Tensor,
    relation_semantic_table: torch.Tensor,
    question: torch.Tensor,
) -> float:
    ids = torch.tensor(list(edge_ids), dtype=torch.long)
    rel_ids = edge_relation_ids.long().index_select(0, ids)
    scores = relation_semantic_table.index_select(0, rel_ids).matmul(question)
    return float(scores.mean().item())


def _summarize(
    *,
    dataset: str,
    samples_seen: int,
    skipped_no_path: int,
    skipped_no_random: int,
    gt_scores: list[float],
    random_scores: list[float],
) -> DatasetResult:
    gt = torch.tensor(gt_scores, dtype=torch.float64)
    random_scores_t = torch.tensor(random_scores, dtype=torch.float64)
    diff = gt - random_scores_t
    units = int(diff.numel())
    diff_mean = float(diff.mean().item())
    diff_std = float(diff.std(unbiased=True).item()) if units > 1 else 0.0
    se = diff_std / math.sqrt(units) if units > 1 else 0.0
    t_value = diff_mean / se if se > 0 else math.inf
    p_value = math.erfc(abs(t_value) / math.sqrt(2.0)) if math.isfinite(t_value) else 0.0
    ci_low = diff_mean - 1.96 * se
    ci_high = diff_mean + 1.96 * se
    cohen_dz = diff_mean / diff_std if diff_std > 0 else math.inf
    return DatasetResult(
        dataset=dataset,
        units=units,
        samples_seen=int(samples_seen),
        skipped_no_path=int(skipped_no_path),
        skipped_no_random=int(skipped_no_random),
        gt_mean=float(gt.mean().item()),
        random_mean=float(random_scores_t.mean().item()),
        diff_mean=diff_mean,
        diff_ci95_low=float(ci_low),
        diff_ci95_high=float(ci_high),
        paired_t=float(t_value),
        normal_approx_p=float(p_value),
        cohen_dz=float(cohen_dz),
        gt_greater_rate=float(diff.gt(0).double().mean().item()),
        gt_median=float(gt.median().item()),
        random_median=float(random_scores_t.median().item()),
    )


def _format_result(result: DatasetResult) -> str:
    return "\n".join(
        [
            f"[{result.dataset}] relation SemanticUtility probe",
            f"  samples_seen={result.samples_seen} comparable_pairs={result.units} "
            f"skipped_no_gt_path={result.skipped_no_path} "
            f"skipped_no_random_path={result.skipped_no_random}",
            f"  gt_mean={result.gt_mean:.6f} random_mean={result.random_mean:.6f} "
            f"diff={result.diff_mean:.6f} "
            f"ci95=[{result.diff_ci95_low:.6f}, {result.diff_ci95_high:.6f}]",
            f"  gt_median={result.gt_median:.6f} random_median={result.random_median:.6f} " f"gt>random={result.gt_greater_rate:.3f}",
            f"  paired_t={result.paired_t:.3f} normal_approx_p={result.normal_approx_p:.3e} " f"cohen_dz={result.cohen_dz:.3f}",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare relation SemanticUtility on answer shortest paths vs random non-answer walks.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/retrieval"))
    parser.add_argument("--datasets", nargs="+", default=["webqsp", "cwq"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-attempts", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main()
