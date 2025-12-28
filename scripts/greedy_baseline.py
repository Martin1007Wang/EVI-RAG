from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import rootutils
from tqdm import tqdm

# Ensure repository root on sys.path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _default_cache_path(data_dir: Path, dataset: str, split: str) -> Path:
    return data_dir / dataset / "materialized" / "g_agent" / f"{split}_g_agent.pt"


def _load_samples(path: Path) -> List[Dict]:
    payload = torch.load(path, map_location="cpu")
    if "samples" not in payload:
        raise RuntimeError(f"Missing 'samples' in {path}")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"Unexpected g_agent payload structure in {path}")
    return samples


def _build_node_maps(sample: Dict) -> Tuple[Dict[int, int], Dict[int, int], Set[int]]:
    """Return global->local, local->global, and set of locals."""
    g2l: Dict[int, int] = {}
    l2g: Dict[int, int] = {}
    locals_set: Set[int] = set()
    if "selected_nodes" not in sample:
        raise KeyError("selected_nodes missing from sample")
    nodes = sample["selected_nodes"]
    if not isinstance(nodes, list):
        raise TypeError("selected_nodes must be a list")
    for node in nodes:
        if "local_index" not in node or "global_id" not in node:
            raise KeyError("selected_nodes entries must include local_index and global_id")
        local = int(node["local_index"])
        gid = int(node["global_id"])
        g2l[gid] = local
        l2g[local] = gid
        locals_set.add(local)
    return g2l, l2g, locals_set


def _build_edges(sample: Dict) -> List[Tuple[int, int, int, float]]:
    edges = []
    if "selected_edges" not in sample:
        raise KeyError("selected_edges missing from sample")
    raw_edges = sample["selected_edges"]
    if not isinstance(raw_edges, list):
        raise TypeError("selected_edges must be a list")
    for edge in raw_edges:
        if "head" not in edge or "tail" not in edge or "local_index" not in edge or "score" not in edge:
            raise KeyError("selected_edges entries must include head/tail/local_index/score")
        h = int(edge["head"])
        t = int(edge["tail"])
        idx = int(edge["local_index"])
        score = float(edge["score"])
        rel = edge["relation"]
        edges.append((h, t, idx, score, rel))
    return edges


def _answer_hit(node_map: Dict[int, int], answers: Sequence[int]) -> bool:
    if not answers:
        return False
    return any(int(a) in node_map for a in answers)


def _path_reachability(edges: List[Tuple[int, int, int, float]], starts: Set[int], answers: Set[int]) -> bool:
    if not starts or not answers or not edges:
        return False
    adj: Dict[int, List[int]] = {}
    for h, t, _idx, _score, _rel in edges:
        adj.setdefault(h, []).append(t)
        adj.setdefault(t, []).append(h)
    frontier = list(starts)
    visited = set(frontier)
    while frontier:
        cur = frontier.pop()
        if cur in answers:
            return True
        for nb in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                frontier.append(nb)
    return False


def _gt_recall(gt_edges: Set[int], selected_edges: Set[int]) -> Tuple[float, bool, int]:
    if not gt_edges:
        return 0.0, False, 0
    hits = sum(1 for idx in gt_edges if idx in selected_edges)
    recall = hits / max(len(gt_edges), 1)
    perfect = hits == len(gt_edges)
    return recall, perfect, hits


def _component_count(selected_edges: List[Tuple[int, int, int, float, object]], node_count: int) -> int:
    if not selected_edges:
        return 0
    adj: Dict[int, List[int]] = {}
    for h, t, _idx, _score, _rel in selected_edges:
        adj.setdefault(h, []).append(t)
        adj.setdefault(t, []).append(h)
    visited: Set[int] = set()
    comps = 0
    for node in list(adj.keys()):
        if node in visited:
            continue
        comps += 1
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    stack.append(nb)
    return comps


def _truncate_sample(sample: Dict, edges_sorted: List[Tuple[int, int, int, float, object]], k: int) -> Dict:
    """Build a minimal g_agent-like sample with top-k edges."""
    top_edges = edges_sorted[:k]
    edge_idx_set = {idx for _, _, idx, _ in top_edges}
    selected_edges = []
    nodes_needed: Set[int] = set()
    for h, t, idx, score in top_edges:
        selected_edges.append({"local_index": idx, "head": h, "tail": t, "relation": None, "label": 0.0, "score": score})
        nodes_needed.add(h)
        nodes_needed.add(t)
    node_lookup = {int(n["local_index"]): n for n in sample["selected_nodes"]}
    selected_nodes = []
    for loc in sorted(nodes_needed):
        node = node_lookup.get(loc)
        if node is not None:
            selected_nodes.append({"local_index": int(node["local_index"]), "global_id": int(node["global_id"])})
        else:
            selected_nodes.append({"local_index": int(loc), "global_id": -1})
    cc = _component_count(top_edges, len(selected_nodes))
    new_sample = dict(sample)
    new_sample["selected_edges"] = selected_edges
    new_sample["selected_nodes"] = selected_nodes
    new_sample["selected_edge_local_indices"] = sorted(edge_idx_set)
    new_sample["selected_node_local_indices"] = sorted(nodes_needed)
    new_sample["top_edge_local_indices"] = sorted(edge_idx_set)
    new_sample["core_node_local_indices"] = sorted(nodes_needed)
    new_sample["core_component_count"] = cc
    return new_sample


def evaluate_greedy(
    cache_path: Path,
    *,
    topk_list: List[int],
    save_dir: Optional[Path] = None,
    limit: int | None = None,
) -> None:
    samples = _load_samples(cache_path)
    total = 0
    answer_present = 0
    full_edge_counts: List[int] = []
    stats = {
        k: {
            "path_reach": 0,
            "gt_samples": 0,
            "gt_recall_sum": 0.0,
            "gt_perfect": 0,
            "snr_num": 0.0,
            "snr_den": 0.0,
            "edge_counts": [],
            "rel_div_counts": [],
            "answer_reach_total": 0,
            "answer_reach_hits": 0,
        }
        for k in topk_list
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        truncated = {k: [] for k in topk_list}

    for sample in tqdm(samples, desc=f"GreedyEval-{cache_path.name}"):
        if limit is not None and limit > 0 and total >= limit:
            break
        total += 1
        g2l, l2g, _ = _build_node_maps(sample)
        if "answer_entity_ids" not in sample:
            raise KeyError("answer_entity_ids missing from sample")
        if "start_entity_ids" not in sample:
            raise KeyError("start_entity_ids missing from sample")
        answers_global = [int(a) for a in sample["answer_entity_ids"] if int(a) >= 0]
        starts_global = [int(s) for s in sample["start_entity_ids"] if int(s) >= 0]
        answers_local = {g2l[a] for a in answers_global if a in g2l}
        starts_local = {g2l[s] for s in starts_global if s in g2l}
        edges = _build_edges(sample)
        edges_sorted = sorted(edges, key=lambda x: x[3], reverse=True)
        full_edge_counts.append(len(edges_sorted))

        has_answer = _answer_hit(g2l, answers_global)
        if has_answer:
            answer_present += 1
        if "gt_path_edge_local_indices" not in sample:
            raise KeyError("gt_path_edge_local_indices missing from sample")
        gt_edge_locals = set(int(x) for x in sample["gt_path_edge_local_indices"])

        for k in topk_list:
            top_edges = edges_sorted[:k]
            edge_set = {idx for _, _, idx, _score, _ in top_edges}
            stats[k]["edge_counts"].append(len(edge_set))
            if has_answer and _path_reachability(top_edges, starts_local, answers_local):
                stats[k]["path_reach"] += 1
            if gt_edge_locals:
                stats[k]["gt_samples"] += 1
                recall, perfect, hits = _gt_recall(gt_edge_locals, edge_set)
                stats[k]["gt_recall_sum"] += recall
                if perfect:
                    stats[k]["gt_perfect"] += 1
                stats[k]["snr_num"] += hits
                stats[k]["snr_den"] += len(edge_set)
            # Relation diversity: unique relations / k
            rels = [r for *_rest, r in top_edges if r is not None]
            unique_rels = len(set(rels)) if rels else 0
            stats[k]["rel_div_counts"].append(unique_rels / max(len(top_edges), 1))
            # Answer coverage: reachable distinct answers within top-k edges
            if answers_local:
                stats[k]["answer_reach_total"] += len(answers_local)
                # build adjacency limited to top-k
                limited_adj: Dict[int, List[int]] = {}
                for h, t, _idx, _score, _r in top_edges:
                    limited_adj.setdefault(h, []).append(t)
                    limited_adj.setdefault(t, []).append(h)
                found: Set[int] = set()
                frontier = list(starts_local)
                visited = set(frontier)
                while frontier:
                    cur = frontier.pop()
                    if cur in answers_local:
                        found.add(cur)
                    for nb in limited_adj.get(cur, []):
                        if nb not in visited:
                            visited.add(nb)
                            frontier.append(nb)
                stats[k]["answer_reach_hits"] += len(found)
            if save_dir is not None:
                truncated[k].append(_truncate_sample(sample, edges_sorted, k))

    if total == 0:
        print("No samples found.")
        return

    def _mean(values: List[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _median(values: List[int]) -> float:
        return float(np.median(values)) if values else 0.0

    answer_hit_rate = answer_present / total
    print(f"Cache: {cache_path}")
    print(f"Total samples: {total}")
    print(f"Answer present in graph (hit rate): {answer_hit_rate:.4f} ({answer_present}/{total})")
    print(f"Full selected edges: mean={_mean(full_edge_counts):.2f} | median={_median(full_edge_counts):.2f}")
    for k in topk_list:
        reach_rate = stats[k]["path_reach"] / total
        gt_samples = stats[k]["gt_samples"]
        gt_recall_mean = stats[k]["gt_recall_sum"] / gt_samples if gt_samples else 0.0
        snr = (stats[k]["snr_num"] / stats[k]["snr_den"]) if stats[k]["snr_den"] > 0 else 0.0
        rel_div_mean = _mean(stats[k]["rel_div_counts"])
        rel_div_med = _median(stats[k]["rel_div_counts"])
        ans_total = stats[k]["answer_reach_total"]
        ans_hits = stats[k]["answer_reach_hits"]
        ans_cover = (ans_hits / ans_total) if ans_total > 0 else 0.0
        print(f"[Top-{k}] Q→A reachable: {reach_rate:.4f} ({stats[k]['path_reach']}/{total})")
        print(f"[Top-{k}] GT edge recall (mean over {gt_samples} with GT): {gt_recall_mean:.4f}")
        print(f"[Top-{k}] GT perfect recovery: {stats[k]['gt_perfect']}/{gt_samples}")
        print(f"[Top-{k}] SNR (GT/edges): {snr:.4f}")
        print(f"[Top-{k}] Avg edges: {_mean(stats[k]['edge_counts']):.2f} | Median edges: {_median(stats[k]['edge_counts']):.2f}")
        print(f"[Top-{k}] Relation diversity (unique_relations/k): mean={rel_div_mean:.4f}, median={rel_div_med:.4f}")
        print(f"[Top-{k}] Answer coverage: {ans_cover:.4f} ({ans_hits}/{ans_total})")

    if save_dir is not None:
        for k in topk_list:
            out = {"settings": {"top_k": k, "source_cache": str(cache_path)}, "samples": truncated[k]}
            out_path = save_dir / f"greedy_top{k}.pt"
            torch.save(out, out_path)
            print(f"Saved Top-{k} greedy artifact to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate greedy g_agent snapshot quality (no GFlowNet).")
    parser.add_argument("--data_dir", type=Path, default=Path("/mnt/data/retrieval_dataset"), help="Root data dir.")
    parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name (e.g., webqsp).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--cache_path", type=Path, default=None, help="Override g_agent cache path.")
    parser.add_argument("--lmdb_path", type=Path, default=None, help="(deprecated) LMDB backfill is unsupported.")
    parser.add_argument("--topk", type=str, default="5,10,15,20,50", help="Comma-separated list of K for eval and export.")
    parser.add_argument("--save_dir", type=Path, default=None, help="Optional dir to save greedy_topK.pt artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick stats.")
    args = parser.parse_args()

    cache_path = args.cache_path or _default_cache_path(args.data_dir, args.dataset, args.split)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    if args.lmdb_path is not None:
        raise ValueError("lmdb_path backfill is no longer supported; provide start/answer ids in the cache.")
    topk_list = [int(x) for x in args.topk.split(",") if str(x).strip()]
    evaluate_greedy(
        cache_path,
        topk_list=topk_list,
        save_dir=args.save_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
