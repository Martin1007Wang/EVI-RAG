#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("KMP_SHM_DISABLE", "1")

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
try:
    import rootutils  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    rootutils = None  # type: ignore[assignment]
if rootutils is not None:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.g_retrieval_dataset import GRetrievalDataset

_POSITIVE_THRESHOLD = 0.5
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_NUM_WORKERS = 0
_DEFAULT_DATA_DIR = "/mnt/data/retrieval_dataset"
_DEFAULT_TOP_K = 5
_SUMMARY_QUANTILES = (0.1, 0.5, 0.9)
_MIN_COUNT = 1
_MIN_COUNT_FLOAT = 1.0


@dataclass
class GraphStats:
    edge_counts: torch.Tensor
    pos_counts: torch.Tensor
    neg_counts: torch.Tensor
    node_counts: torch.Tensor
    non_text_counts: torch.Tensor
    pos_ratios: torch.Tensor
    non_text_ratios: torch.Tensor
    sample_ids: Optional[List[str]] = None


@dataclass
class StatAccumulator:
    total_graphs: int = 0
    total_edges: int = 0
    total_pos: int = 0
    total_nodes: int = 0
    total_non_text: int = 0
    zero_pos_graphs: int = 0
    zero_neg_graphs: int = 0
    edge_counts: List[int] = None
    pos_counts: List[int] = None
    pos_ratios: List[float] = None
    non_text_ratios: List[float] = None
    top_pos_ratio: List[Tuple[float, str]] = None
    top_non_text_ratio: List[Tuple[float, str]] = None

    def __post_init__(self) -> None:
        self.edge_counts = []
        self.pos_counts = []
        self.pos_ratios = []
        self.non_text_ratios = []
        self.top_pos_ratio = []
        self.top_non_text_ratio = []

    def update(self, stats: GraphStats, *, top_k: int) -> None:
        edge_counts = stats.edge_counts.detach().cpu().tolist()
        pos_counts = stats.pos_counts.detach().cpu().tolist()
        pos_ratios = stats.pos_ratios.detach().cpu().tolist()
        non_text_ratios = stats.non_text_ratios.detach().cpu().tolist()

        self.edge_counts.extend(edge_counts)
        self.pos_counts.extend(pos_counts)
        self.pos_ratios.extend(pos_ratios)
        self.non_text_ratios.extend(non_text_ratios)

        self.total_graphs += len(edge_counts)
        self.total_edges += int(stats.edge_counts.sum().item())
        self.total_pos += int(stats.pos_counts.sum().item())
        self.total_nodes += int(stats.node_counts.sum().item())
        self.total_non_text += int(stats.non_text_counts.sum().item())
        self.zero_pos_graphs += int((stats.pos_counts == 0).sum().item())
        self.zero_neg_graphs += int((stats.neg_counts == 0).sum().item())

        if stats.sample_ids:
            for ratio, sample_id in zip(pos_ratios, stats.sample_ids):
                self._push_top(self.top_pos_ratio, ratio, sample_id, top_k)
            for ratio, sample_id in zip(non_text_ratios, stats.sample_ids):
                self._push_top(self.top_non_text_ratio, ratio, sample_id, top_k)

    @staticmethod
    def _push_top(store: List[Tuple[float, str]], value: float, sample_id: str, top_k: int) -> None:
        store.append((float(value), str(sample_id)))
        store.sort(key=lambda item: item[0], reverse=True)
        if len(store) > top_k:
            del store[top_k:]


def _resolve_materialized_dir(args: argparse.Namespace) -> Path:
    if args.materialized_dir:
        return Path(args.materialized_dir)
    data_dir = Path(args.data_dir or _DEFAULT_DATA_DIR)
    return data_dir / args.dataset / "materialized"


def _build_dataset(materialized_dir: Path, split: str, *, validate: bool) -> GRetrievalDataset:
    embeddings_dir = materialized_dir / "embeddings"
    vocab_path = materialized_dir / "vocabulary" / "vocabulary.lmdb"
    split_path = embeddings_dir / f"{split}.lmdb"
    return GRetrievalDataset(
        split_path=split_path,
        vocabulary_path=vocab_path,
        embeddings_dir=embeddings_dir,
        dataset_name=materialized_dir.name,
        split_name=split,
        validate_on_init=validate,
    )


def _infer_num_graphs(batch: object) -> int:
    ptr = getattr(batch, "ptr", None)
    if torch.is_tensor(ptr):
        return int(ptr.numel() - 1)
    num_graphs = getattr(batch, "num_graphs", None)
    if num_graphs is None:
        raise ValueError("Batch missing ptr/num_graphs; cannot infer graph count.")
    return int(num_graphs)


def _compute_graph_stats(batch: object, *, threshold: float) -> GraphStats:
    labels = torch.as_tensor(getattr(batch, "labels")).view(-1)
    if labels.numel() == 0:
        raise ValueError("Batch labels are empty.")

    edge_index = torch.as_tensor(getattr(batch, "edge_index"))
    node_batch = torch.as_tensor(getattr(batch, "batch"))
    head_idx = edge_index[0]
    edge_batch = node_batch[head_idx]

    num_graphs = _infer_num_graphs(batch)
    edge_counts = torch.zeros(num_graphs, dtype=torch.long)
    edge_counts.scatter_add_(0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long))

    pos_mask = labels > threshold
    pos_counts = torch.zeros(num_graphs, dtype=torch.long)
    pos_counts.scatter_add_(0, edge_batch, pos_mask.to(dtype=torch.long))
    neg_counts = edge_counts - pos_counts
    pos_ratios = pos_counts.to(dtype=torch.float32) / edge_counts.clamp_min(_MIN_COUNT).to(dtype=torch.float32)

    node_embedding_ids = torch.as_tensor(getattr(batch, "node_embedding_ids")).view(-1)
    non_text_mask = node_embedding_ids == 0
    non_text_counts = torch.zeros(num_graphs, dtype=torch.long)
    non_text_counts.scatter_add_(0, node_batch, non_text_mask.to(dtype=torch.long))
    node_ptr = getattr(batch, "ptr", None)
    if not torch.is_tensor(node_ptr):
        raise ValueError("Batch missing ptr required for node counts.")
    node_counts = (node_ptr[1:] - node_ptr[:-1]).to(dtype=torch.float32)
    non_text_ratios = non_text_counts.to(dtype=torch.float32) / node_counts.clamp_min(_MIN_COUNT_FLOAT)

    sample_ids = getattr(batch, "sample_id", None)
    if sample_ids is not None:
        sample_ids = [str(sid) for sid in sample_ids]

    return GraphStats(
        edge_counts=edge_counts,
        pos_counts=pos_counts,
        neg_counts=neg_counts,
        node_counts=node_counts.to(dtype=torch.long),
        non_text_counts=non_text_counts,
        pos_ratios=pos_ratios,
        non_text_ratios=non_text_ratios,
        sample_ids=sample_ids,
    )


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    values_list = list(values)
    if not values_list:
        return {"count": 0.0}
    tensor = torch.as_tensor(values_list, dtype=torch.float32)
    summary: Dict[str, float] = {
        "count": float(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }
    for q in _SUMMARY_QUANTILES:
        q_val = float(torch.quantile(tensor, torch.tensor(q)).item())
        summary[f"p{int(q * 100)}"] = q_val
    return summary


def _log_summary(title: str, summary: Dict[str, float]) -> None:
    if not summary or summary.get("count", 0.0) <= 0:
        logging.info("%s: no data", title)
        return
    logging.info(
        "%s: count=%d mean=%.4f min=%.4f p10=%.4f p50=%.4f p90=%.4f max=%.4f",
        title,
        int(summary["count"]),
        summary["mean"],
        summary["min"],
        summary.get("p10", 0.0),
        summary.get("p50", 0.0),
        summary.get("p90", 0.0),
        summary["max"],
    )


def _log_top(title: str, items: List[Tuple[float, str]]) -> None:
    if not items:
        return
    logging.info("%s (top):", title)
    for value, sample_id in items:
        logging.info("  %.4f  %s", value, sample_id)


def _iter_batches(
    loader: PyGDataLoader,
    *,
    max_batches: Optional[int],
    max_graphs: Optional[int],
) -> Iterable[object]:
    seen = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch
        if max_graphs is not None:
            seen += _infer_num_graphs(batch)
            if seen >= max_graphs:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-graph retriever label and non-text stats.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., webqsp, cwq).")
    parser.add_argument("--split", default="validation", help="Split name (train/validation/test).")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="Root data dir containing dataset folders.")
    parser.add_argument("--materialized-dir", default=None, help="Override materialized dir (uses data-dir/dataset/materialized).")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Batch size for scanning.")
    parser.add_argument("--num-workers", type=int, default=_DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit on number of batches.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional limit on number of graphs.")
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help="Show top-k sample_ids by ratio.")
    parser.add_argument("--validate", action="store_true", help="Enable dataset validation on init.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    materialized_dir = _resolve_materialized_dir(args)
    dataset = _build_dataset(materialized_dir, args.split, validate=args.validate)
    loader = PyGDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    accumulator = StatAccumulator()
    for batch in _iter_batches(loader, max_batches=args.max_batches, max_graphs=args.max_graphs):
        stats = _compute_graph_stats(batch, threshold=_POSITIVE_THRESHOLD)
        accumulator.update(stats, top_k=args.top_k)

    total_pos_ratio = (
        float(accumulator.total_pos) / float(accumulator.total_edges)
        if accumulator.total_edges > 0
        else 0.0
    )
    logging.info("Graphs=%d edges=%d pos_edges=%d pos_ratio=%.6f", accumulator.total_graphs, accumulator.total_edges, accumulator.total_pos, total_pos_ratio)
    logging.info("Graphs with zero positives: %d", accumulator.zero_pos_graphs)
    logging.info("Graphs with zero negatives: %d", accumulator.zero_neg_graphs)

    overall_non_text_ratio = (
        float(accumulator.total_non_text) / float(accumulator.total_nodes)
        if accumulator.total_nodes > 0
        else 0.0
    )
    logging.info(
        "Nodes=%d non_text_nodes=%d non_text_ratio=%.6f",
        accumulator.total_nodes,
        accumulator.total_non_text,
        overall_non_text_ratio,
    )

    _log_summary("edge_counts", _summarize(accumulator.edge_counts))
    _log_summary("pos_counts", _summarize(accumulator.pos_counts))
    _log_summary("pos_ratios", _summarize(accumulator.pos_ratios))
    _log_summary("non_text_ratios", _summarize(accumulator.non_text_ratios))

    _log_top("pos_ratio", accumulator.top_pos_ratio)
    _log_top("non_text_ratio", accumulator.top_non_text_ratio)


if __name__ == "__main__":
    main()
