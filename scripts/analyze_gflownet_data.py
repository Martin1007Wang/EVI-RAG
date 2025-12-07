"""Quick health-check for g_agent caches.

Computes retriever vs GT alignment, branching factor, and path depth statistics
for train/val/test splits without running training.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys
from typing import Iterable, Tuple

import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
import rootutils  # type: ignore

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# expose helper before use
def _greedy_retriever_metrics(
    data: Data,
    k_list: list[int],
    use_top_mask: bool,
) -> dict[int, dict[str, float]]:
    """Simulate greedy policy over edge_scores with start-constraint; length=1 rollout."""
    edge_scores = data.edge_scores
    edge_index = data.edge_index
    num_edges = edge_scores.numel()
    start_nodes = data.start_node_locals
    answer_nodes = data.answer_node_locals
    gt_edges = data.gt_path_edge_local_ids
    edge_mask = data.top_edge_mask if use_top_mask and hasattr(data, "top_edge_mask") else None

    # mask to edges touching any start node
    if start_nodes.numel() > 0:
        touch_start = torch.zeros(num_edges, dtype=torch.bool)
        for s in start_nodes.tolist():
            touch_start |= (edge_index[0] == s) | (edge_index[1] == s)
    else:
        touch_start = torch.ones(num_edges, dtype=torch.bool)

    valid_mask = touch_start if edge_mask is None else (touch_start & edge_mask)
    if not valid_mask.any():
        return {k: {"success": 0.0, "answer_hit_any": 0.0, "answer_recall_union": 0.0, "path_hit_any": 0.0,
                    "path_hit_precision": 0.0, "path_hit_recall": 0.0, "path_hit_f1": 0.0} for k in k_list}

    scores = edge_scores.clone()
    scores[~valid_mask] = -1e9
    order = torch.argsort(scores, descending=True)

    results: dict[int, dict[str, float]] = {}
    for k in k_list:
        topk = order[: min(k, num_edges)]
        if edge_mask is not None:
            topk = topk[valid_mask[topk]]
        if topk.numel() == 0:
            results[k] = {
                "success": 0.0,
                "answer_hit_any": 0.0,
                "answer_recall_union": 0.0,
                "path_hit_any": 0.0,
                "path_hit_precision": 0.0,
                "path_hit_recall": 0.0,
                "path_hit_f1": 0.0,
            }
            continue
        # answer coverage
        hit_answer_nodes = set()
        for e in topk.tolist():
            h = int(edge_index[0, e].item())
            t = int(edge_index[1, e].item())
            if answer_nodes.numel() > 0:
                if (answer_nodes == h).any():
                    hit_answer_nodes.add(h)
                if (answer_nodes == t).any():
                    hit_answer_nodes.add(t)
        ans_total = int(answer_nodes.numel())
        answer_recall = (len(hit_answer_nodes) / ans_total) if ans_total > 0 else 0.0
        answer_hit_any = 1.0 if len(hit_answer_nodes) > 0 else 0.0
        success = answer_hit_any

        # GT path hit
        hit_path = 0.0
        if gt_edges.numel() > 0:
            gt_set = set(gt_edges.tolist())
            hit = any(e in gt_set for e in topk.tolist())
            hit_path = 1.0 if hit else 0.0
            tp = sum(1 for e in topk.tolist() if e in gt_set)
            precision = tp / max(1, topk.numel())
            recall = tp / max(1, len(gt_set))
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        else:
            precision = recall = f1 = 0.0

        results[k] = {
            "success": success,
            "answer_hit_any": answer_hit_any,
            "answer_recall_union": answer_recall,
            "path_hit_any": hit_path,
            "path_hit_precision": precision,
            "path_hit_recall": recall,
            "path_hit_f1": f1,
        }
    return results


def _beam_greedy_metrics(
    data: Data,
    k_list: list[int],
    max_steps: int,
    use_top_mask: bool,
) -> dict[int, dict[str, float]]:
    """Greedy beam search on edge_scores with env-like constraints (start-only step0, no revisit/backtrack)."""
    edge_scores = data.edge_scores
    edge_index = data.edge_index
    num_edges = edge_scores.numel()
    start_nodes = data.start_node_locals
    answer_nodes = data.answer_node_locals
    gt_edges = data.gt_path_edge_local_ids
    edge_mask = data.top_edge_mask if use_top_mask and hasattr(data, "top_edge_mask") else None

    # build adjacency lists per node for fast expansion
    num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for eid in range(num_edges):
        h = int(edge_index[0, eid].item())
        t = int(edge_index[1, eid].item())
        adj[h].append(eid)
        adj[t].append(eid)

    # step0: edges touching start
    start_set = set(start_nodes.tolist())
    if start_set:
        touch_start = torch.zeros(num_edges, dtype=torch.bool)
        for s in start_set:
            for eid in adj[s]:
                touch_start[eid] = True
    else:
        touch_start = torch.ones(num_edges, dtype=torch.bool)
    valid_mask = touch_start if edge_mask is None else (touch_start & edge_mask)
    if not valid_mask.any():
        empty = {k: {"success": 0.0, "answer_hit_any": 0.0, "answer_recall_union": 0.0, "path_hit_any": 0.0,
                     "path_hit_precision": 0.0, "path_hit_recall": 0.0, "path_hit_f1": 0.0} for k in k_list}
        return empty

    scores = edge_scores.clone()
    scores[~valid_mask] = -1e9
    order = torch.argsort(scores, descending=True)

    def _other_node(edge_id: int, current: int | None) -> tuple[int, int]:
        h = int(edge_index[0, edge_id].item())
        t = int(edge_index[1, edge_id].item())
        if current is None:
            if h in start_nodes.tolist():
                return h, t
            if t in start_nodes.tolist():
                return t, h
        if current == h:
            return h, t
        return t, h

    Beam = dict[str, object]
    beams: list[Beam] = []
    for e in order.tolist():
        cur, nxt = _other_node(e, None)
        beams.append(
            {
                "path": [e],
                "score": float(scores[e].item()),
                "current": nxt,
                "prev": cur,
                "visited": set([cur, nxt]),
                "hit_ans": set([nxt]) if (answer_nodes == nxt).any() else set(),
            }
        )

    # expand beams up to max_steps
    for _ in range(max_steps - 1):
        new_beams: list[Beam] = []
        for beam in beams:
            cur = beam["current"]  # type: ignore[assignment]
            prev = beam["prev"]  # type: ignore[assignment]
            visited = beam["visited"]  # type: ignore[assignment]
            valid = []
            for eid in adj[cur]:
                if edge_mask is not None and not bool(edge_mask[eid]):
                    continue
                h = int(edge_index[0, eid].item())
                t = int(edge_index[1, eid].item())
                nxt = t if cur == h else h
                if nxt in visited:
                    continue
                if nxt == prev:
                    continue
                valid.append((eid, nxt))
            if not valid:
                new_beams.append(beam)
                continue
            valid = sorted(valid, key=lambda x: float(edge_scores[x[0]].item()), reverse=True)
            for eid, nxt in valid[: max(k_list)]:
                new_visited = set(visited)
                new_visited.add(nxt)
                new_hit = set(beam["hit_ans"])  # type: ignore[assignment]
                if (answer_nodes == nxt).any():
                    new_hit.add(nxt)
                new_beams.append(
                    {
                        "path": beam["path"] + [eid],  # type: ignore[operator]
                        "score": beam["score"] + float(edge_scores[eid].item()),  # type: ignore[operator]
                        "current": nxt,
                        "prev": cur,
                        "visited": new_visited,
                        "hit_ans": new_hit,
                    }
                )
        beams = sorted(new_beams, key=lambda b: b["score"], reverse=True)
        beams = beams[: max(k_list)]

    # aggregate per K using top-k beams
    results: dict[int, dict[str, float]] = {}
    gt_set = set(gt_edges.tolist())
    ans_total = int(answer_nodes.numel())
    for k in k_list:
        topk = beams[:k]
        union_ans = set()
        hit_any_ans = False
        hit_any_path = False
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        for beam in topk:
            union_ans |= beam["hit_ans"]  # type: ignore[arg-type]
            hit_any_ans = hit_any_ans or (len(beam["hit_ans"]) > 0)  # type: ignore[arg-type]
            if gt_set:
                tp = sum(1 for e in beam["path"] if e in gt_set)  # type: ignore[arg-type]
                prec = tp / max(1, len(beam["path"]))  # type: ignore[arg-type]
                rec = tp / max(1, len(gt_set))
                f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
                if f1 > best_f1:
                    best_f1, best_precision, best_recall = f1, prec, rec
                hit_any_path = hit_any_path or (tp > 0)
        answer_recall = (len(union_ans) / ans_total) if ans_total > 0 else 0.0
        results[k] = {
            "success": 1.0 if hit_any_ans else 0.0,
            "answer_hit_any": 1.0 if hit_any_ans else 0.0,
            "answer_recall_union": answer_recall,
            "path_hit_any": 1.0 if hit_any_path else 0.0,
            "path_hit_precision": best_precision,
            "path_hit_recall": best_recall,
            "path_hit_f1": best_f1,
        }
    return results
import importlib.util

_g_agent_path = REPO_ROOT / "src" / "data" / "g_agent_dataset.py"
spec = importlib.util.spec_from_file_location("local_g_agent_dataset", _g_agent_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load g_agent_dataset from {_g_agent_path}")
g_agent_module = importlib.util.module_from_spec(spec)
sys.modules["local_g_agent_dataset"] = g_agent_module
spec.loader.exec_module(g_agent_module)
GAgentPyGDataset = getattr(g_agent_module, "GAgentPyGDataset")


def _shortest_path_length(edge_index: torch.Tensor, num_nodes: int, starts: torch.Tensor, targets: torch.Tensor) -> float:
    """Undirected multi-source shortest path length; returns inf if unreachable."""
    if starts.numel() == 0 or targets.numel() == 0:
        return math.inf
    # build adjacency list
    adj = [[] for _ in range(num_nodes)]
    heads = edge_index[0].tolist()
    tails = edge_index[1].tolist()
    for h, t in zip(heads, tails):
        adj[h].append(t)
        adj[t].append(h)

    dist = [-1] * num_nodes
    queue = []
    for s in starts.tolist():
        dist[s] = 0
        queue.append(s)

    target_set = set(targets.tolist())
    qi = 0
    while qi < len(queue):
        node = queue[qi]
        qi += 1
        if node in target_set:
            return float(dist[node])
        for nxt in adj[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return math.inf


def _gt_ranks(edge_scores: torch.Tensor, gt_ids: torch.Tensor, mask: torch.Tensor | None) -> Tuple[list[int], list[float]]:
    """Ranks of GT edges among candidates (lower is better). Returns ranks and score gaps."""
    if gt_ids.numel() == 0:
        return [], []
    if mask is not None:
        edge_scores = edge_scores[mask]
        gt_local = torch.nonzero(mask, as_tuple=False).view(-1)
        mapping = {int(i): idx for idx, i in enumerate(gt_local.tolist())}
        gt_ids = torch.as_tensor([mapping[i] for i in gt_ids.tolist()], device=edge_scores.device)
    scores = edge_scores
    order = torch.argsort(scores, descending=True)
    ranks: list[int] = []
    gaps: list[float] = []
    for gt in gt_ids.tolist():
        gt_score = float(scores[gt].item())
        higher = (scores > gt_score).sum().item()
        rank = higher + 1
        ranks.append(int(rank))
        if scores.numel() > 1:
            # max other score (exclude this gt)
            mask_other = torch.ones_like(scores, dtype=torch.bool)
            mask_other[gt] = False
            max_other = float(scores[mask_other].max().item())
            gaps.append(gt_score - max_other)
        else:
            gaps.append(0.0)
    return ranks, gaps


def analyze_dataset(dataset: GAgentPyGDataset, name: str, max_samples: int, max_steps: int | None, use_top_mask: bool) -> None:
    total = len(dataset)
    n = min(max_samples, total)
    indices = random.sample(range(total), n)

    rank_list: list[int] = []
    gap_list: list[float] = []
    out_deg_list: list[int] = []
    path_len_list: list[float] = []
    reachable_within_max: int = 0

    answer_present = 0
    answer_reachable = 0
    gt_path_present = 0
    greedy_k_list = [1, 5, 10, 20]
    greedy_sums = {k: {"success": 0.0, "answer_hit_any": 0.0, "answer_recall_union": 0.0, "path_hit_any": 0.0,
                       "path_hit_precision": 0.0, "path_hit_recall": 0.0, "path_hit_f1": 0.0} for k in greedy_k_list}
    greedy_counts = {k: {"answer": 0, "path": 0, "all": 0} for k in greedy_k_list}
    beam_sums = {k: {"success": 0.0, "answer_hit_any": 0.0, "answer_recall_union": 0.0, "path_hit_any": 0.0,
                     "path_hit_precision": 0.0, "path_hit_recall": 0.0, "path_hit_f1": 0.0} for k in greedy_k_list}
    beam_counts = {k: {"answer": 0, "path": 0, "all": 0} for k in greedy_k_list}

    for idx in indices:
        data: Data = dataset[idx]
        edge_index = data.edge_index
        num_nodes = int(data.num_nodes)
        # candidate mask
        edge_mask = data.top_edge_mask if use_top_mask and hasattr(data, "top_edge_mask") else None
        if hasattr(data, "answer_node_locals") and data.answer_node_locals.numel() > 0:
            answer_present += 1
        if hasattr(data, "is_answer_reachable") and bool(data.is_answer_reachable.item()):
            answer_reachable += 1
        if hasattr(data, "gt_path_exists") and bool(data.gt_path_exists.item()):
            gt_path_present += 1

        # branching factor: edges per start node (after mask)
        if edge_mask is not None:
            masked_edges = edge_index[:, edge_mask]
        else:
            masked_edges = edge_index
        starts = data.start_node_locals
        out_deg = 0
        if starts.numel() > 0:
            for s in starts.tolist():
                out_deg += int(((masked_edges[0] == s) | (masked_edges[1] == s)).sum().item())
            out_deg = out_deg // max(1, starts.numel())
        out_deg_list.append(out_deg)

        # GT ranks
        gt_edges = data.gt_path_edge_local_ids
        ranks, gaps = _gt_ranks(data.edge_scores, gt_edges, edge_mask)
        rank_list.extend(ranks)
        gap_list.extend(gaps)

        # shortest path length
        answers = data.answer_node_locals
        d = _shortest_path_length(edge_index=edge_index, num_nodes=num_nodes, starts=starts, targets=answers)
        path_len_list.append(d if math.isfinite(d) else float("inf"))
        if max_steps is not None and d <= max_steps:
            reachable_within_max += 1

    def _mean(xs: Iterable[float]) -> float:
        xs_list = list(xs)
        return float(sum(xs_list) / max(1, len(xs_list)))

    rank_tensor = torch.tensor(rank_list, dtype=torch.float32)
    recall_at = {k: float((rank_tensor <= k).float().mean().item()) if rank_tensor.numel() > 0 else 0.0 for k in (1, 5, 10)}
    mrr = float((1.0 / rank_tensor).mean().item()) if rank_tensor.numel() > 0 else 0.0

    print(f"\n=== {name} ({n}/{total} samples) ===")
    print(
        f"GT rank count: {len(rank_list)} | MRR: {mrr:.4f} | Recall@1/5/10: {recall_at[1]:.4f}/{recall_at[5]:.4f}/{recall_at[10]:.4f}"
    )
    print(f"Avg score gap (gt - best_other): {_mean(gap_list):.4f}")
    print(
        f"Avg out-degree (per start, masked={use_top_mask}): {_mean(out_deg_list):.2f} | max: {max(out_deg_list) if out_deg_list else 0}"
    )
    finite_paths = [p for p in path_len_list if math.isfinite(p)]
    print(
        f"Shortest path len (finite only) mean: {_mean(finite_paths):.2f} | median: {float(torch.median(torch.tensor(finite_paths)).item()) if finite_paths else float('nan')}"
    )
    # Greedy (retriever-score, 单步) 上界
    for idx in indices:
        data: Data = dataset[idx]
        greedy = _greedy_retriever_metrics(data, greedy_k_list, use_top_mask)
        for k in greedy_k_list:
            greedy_counts[k]["all"] += 1
            if data.answer_node_locals.numel() > 0 and bool(getattr(data, "is_answer_reachable", True)):
                greedy_counts[k]["answer"] += 1
                greedy_sums[k]["answer_hit_any"] += greedy[k]["answer_hit_any"]
                greedy_sums[k]["answer_recall_union"] += greedy[k]["answer_recall_union"]
                greedy_sums[k]["success"] += greedy[k]["success"]
            if bool(getattr(data, "gt_path_exists", True)):
                greedy_counts[k]["path"] += 1
                greedy_sums[k]["path_hit_any"] += greedy[k]["path_hit_any"]
                greedy_sums[k]["path_hit_precision"] += greedy[k]["path_hit_precision"]
                greedy_sums[k]["path_hit_recall"] += greedy[k]["path_hit_recall"]
                greedy_sums[k]["path_hit_f1"] += greedy[k]["path_hit_f1"]
    # Beam 贪心（多步 env 约束）上界
    for idx in indices:
        data: Data = dataset[idx]
        beam = _beam_greedy_metrics(data, greedy_k_list, max_steps if max_steps is not None else 6, use_top_mask)
        for k in greedy_k_list:
            beam_counts[k]["all"] += 1
            if data.answer_node_locals.numel() > 0 and bool(getattr(data, "is_answer_reachable", True)):
                beam_counts[k]["answer"] += 1
                beam_sums[k]["answer_hit_any"] += beam[k]["answer_hit_any"]
                beam_sums[k]["answer_recall_union"] += beam[k]["answer_recall_union"]
                beam_sums[k]["success"] += beam[k]["success"]
            if bool(getattr(data, "gt_path_exists", True)):
                beam_counts[k]["path"] += 1
                beam_sums[k]["path_hit_any"] += beam[k]["path_hit_any"]
                beam_sums[k]["path_hit_precision"] += beam[k]["path_hit_precision"]
                beam_sums[k]["path_hit_recall"] += beam[k]["path_hit_recall"]
                beam_sums[k]["path_hit_f1"] += beam[k]["path_hit_f1"]

    if max_steps is not None:
        print(f"Reachable within max_steps={max_steps}: {reachable_within_max}/{n} = {reachable_within_max / max(1, n):.4f}")
    print("\nGreedy retriever metrics (start-constrained, top edge per step, length=1):")
    for k in greedy_k_list:
        ans_denom = max(1, greedy_counts[k]["answer"])
        path_denom = max(1, greedy_counts[k]["path"])
        all_denom = max(1, greedy_counts[k]["all"])
        print(
            f"K={k} | "
            f"success@K<= {greedy_sums[k]['success'] / ans_denom:.4f} (answerable={greedy_counts[k]['answer']}) | "
            f"answer_hit_any@K<= {greedy_sums[k]['answer_hit_any'] / ans_denom:.4f} | "
            f"answer_recall_union@K<= {greedy_sums[k]['answer_recall_union'] / ans_denom:.4f} | "
            f"path_hit_any@K<= {greedy_sums[k]['path_hit_any'] / path_denom:.4f} | "
            f"path_hit_precision@K<= {greedy_sums[k]['path_hit_precision'] / path_denom:.4f} | "
            f"path_hit_recall@K<= {greedy_sums[k]['path_hit_recall'] / path_denom:.4f} | "
            f"path_hit_f1@K<= {greedy_sums[k]['path_hit_f1'] / path_denom:.4f} | "
            f"coverage_base={all_denom}"
        )
    print("\nBeam-greedy metrics (env-like constraints, max_steps, best K beams by score):")
    for k in greedy_k_list:
        ans_denom = max(1, beam_counts[k]["answer"])
        path_denom = max(1, beam_counts[k]["path"])
        all_denom = max(1, beam_counts[k]["all"])
        print(
            f"K={k} | "
            f"success@K<= {beam_sums[k]['success'] / ans_denom:.4f} (answerable={beam_counts[k]['answer']}) | "
            f"answer_hit_any@K<= {beam_sums[k]['answer_hit_any'] / ans_denom:.4f} | "
            f"answer_recall_union@K<= {beam_sums[k]['answer_recall_union'] / ans_denom:.4f} | "
            f"path_hit_any@K<= {beam_sums[k]['path_hit_any'] / path_denom:.4f} | "
            f"path_hit_precision@K<= {beam_sums[k]['path_hit_precision'] / path_denom:.4f} | "
            f"path_hit_recall@K<= {beam_sums[k]['path_hit_recall'] / path_denom:.4f} | "
            f"path_hit_f1@K<= {beam_sums[k]['path_hit_f1'] / path_denom:.4f} | "
            f"coverage_base={all_denom}"
        )


DEFAULT_BASE = Path("/mnt/data/retrieval_dataset/webqsp/materialized/g_agent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze g_agent caches for retriever vs GT alignment and search difficulty.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_BASE / "train_g_agent.pt",
        help="Path to train_g_agent.pt (default: /mnt/data/retrieval_dataset/webqsp/materialized/g_agent/train_g_agent.pt)",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=DEFAULT_BASE / "validation_g_agent.pt",
        help="Path to validation_g_agent.pt (default: /mnt/data/retrieval_dataset/webqsp/materialized/g_agent/validation_g_agent.pt)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_BASE / "test_g_agent.pt",
        help="Path to test_g_agent.pt (default: /mnt/data/retrieval_dataset/webqsp/materialized/g_agent/test_g_agent.pt)",
    )
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per split to analyze")
    parser.add_argument(
        "--max-steps", type=int, default=6, help="Max steps for reachability stats (None to skip)", nargs="?", const=None
    )
    parser.add_argument("--use-top-mask", action="store_true", help="Use top_edge_mask to filter edges")
    args = parser.parse_args()

    if not args.train_path.exists() or not args.val_path.exists():
        raise FileNotFoundError("train/val g_agent cache not found; please check the path arguments.")

    train_ds = GAgentPyGDataset(args.train_path, drop_unreachable=False)
    val_ds = GAgentPyGDataset(args.val_path, drop_unreachable=False)

    analyze_dataset(train_ds, "train", args.max_samples, args.max_steps, args.use_top_mask)
    analyze_dataset(val_ds, "val", args.max_samples, args.max_steps, args.use_top_mask)

    if args.test_path.exists():
        test_ds = GAgentPyGDataset(args.test_path, drop_unreachable=False)
        analyze_dataset(test_ds, "test", args.max_samples, args.max_steps, args.use_top_mask)
    else:
        print(f"Test path {args.test_path} not found; skipping test split.")


if __name__ == "__main__":
    main()
def _greedy_retriever_metrics(
    data: Data,
    k_list: list[int],
    use_top_mask: bool,
) -> dict[int, dict[str, float]]:
    """Simulate greedy policy over edge_scores with start-constraint; length=1 rollout."""
    edge_scores = data.edge_scores
    edge_index = data.edge_index
    num_edges = edge_scores.numel()
    start_nodes = data.start_node_locals
    answer_nodes = data.answer_node_locals
    gt_edges = data.gt_path_edge_local_ids
    edge_mask = data.top_edge_mask if use_top_mask and hasattr(data, "top_edge_mask") else None

    # mask to edges touching any start node
    if start_nodes.numel() > 0:
        touch_start = torch.zeros(num_edges, dtype=torch.bool)
        for s in start_nodes.tolist():
            touch_start |= (edge_index[0] == s) | (edge_index[1] == s)
    else:
        touch_start = torch.ones(num_edges, dtype=torch.bool)

    valid_mask = touch_start if edge_mask is None else (touch_start & edge_mask)
    if not valid_mask.any():
        return {k: {"success": 0.0, "answer_hit_any": 0.0, "answer_recall_union": 0.0, "path_hit_any": 0.0,
                    "path_hit_precision": 0.0, "path_hit_recall": 0.0, "path_hit_f1": 0.0} for k in k_list}

    scores = edge_scores.clone()
    scores[~valid_mask] = -1e9
    order = torch.argsort(scores, descending=True)

    results: dict[int, dict[str, float]] = {}
    for k in k_list:
        topk = order[: min(k, num_edges)]
        if edge_mask is not None:
            topk = topk[valid_mask[topk]]
        if topk.numel() == 0:
            results[k] = {
                "success": 0.0,
                "answer_hit_any": 0.0,
                "answer_recall_union": 0.0,
                "path_hit_any": 0.0,
                "path_hit_precision": 0.0,
                "path_hit_recall": 0.0,
                "path_hit_f1": 0.0,
            }
            continue
        # answer coverage
        hit_answer_nodes = set()
        for e in topk.tolist():
            h = int(edge_index[0, e].item())
            t = int(edge_index[1, e].item())
            if answer_nodes.numel() > 0:
                if (answer_nodes == h).any():
                    hit_answer_nodes.add(h)
                if (answer_nodes == t).any():
                    hit_answer_nodes.add(t)
        ans_total = int(answer_nodes.numel())
        answer_recall = (len(hit_answer_nodes) / ans_total) if ans_total > 0 else 0.0
        answer_hit_any = 1.0 if len(hit_answer_nodes) > 0 else 0.0
        success = answer_hit_any

        # GT path hit
        hit_path = 0.0
        if gt_edges.numel() > 0:
            gt_set = set(gt_edges.tolist())
            hit = any(e in gt_set for e in topk.tolist())
            hit_path = 1.0 if hit else 0.0
            tp = sum(1 for e in topk.tolist() if e in gt_set)
            precision = tp / max(1, topk.numel())
            recall = tp / max(1, len(gt_set))
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        else:
            precision = recall = f1 = 0.0

        results[k] = {
            "success": success,
            "answer_hit_any": answer_hit_any,
            "answer_recall_union": answer_recall,
            "path_hit_any": hit_path,
            "path_hit_precision": precision,
            "path_hit_recall": recall,
            "path_hit_f1": f1,
        }
    return results
