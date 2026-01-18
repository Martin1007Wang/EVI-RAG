from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

from src.utils.logging_utils import get_logger, log_event

_ZERO = 0
_ONE = 1
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_PROGRESS_INTERVAL = 0
_EMPTY_PATTERN = ()

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CvtPatternStats:
    num_entities: int
    num_cvt: int
    num_patterns: int
    max_rel_count: int
    rel_count_p50: int
    rel_count_p90: int
    rel_count_p99: int
    rel_count_max: int
    avg_rel_count: float


def _load_entity_vocab(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["entity_id", "is_cvt"])
    entity_ids = np.asarray(table.column("entity_id").to_numpy(), dtype=np.int64)
    is_cvt = np.asarray(table.column("is_cvt").to_numpy(), dtype=bool)
    if entity_ids.size == 0:
        raise ValueError("entity_vocab is empty.")
    if entity_ids.size != is_cvt.size:
        raise ValueError("entity_vocab entity_id/is_cvt length mismatch.")
    return entity_ids, is_cvt


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except RuntimeError:
        return torch.load(path, map_location="cpu")


def _iter_graph_batches(graphs_path: Path, batch_size: int) -> Iterable[dict]:
    parquet = pq.ParquetFile(graphs_path)
    columns = ["node_entity_ids", "edge_src", "edge_dst", "edge_relation_ids"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pydict()


def _update_rel_sets_for_graph(
    *,
    node_entity_ids: Sequence[int],
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    edge_rel: Sequence[int],
    is_cvt: np.ndarray,
    rel_sets: Dict[int, Set[int]],
) -> None:
    if not edge_src or not edge_dst or not edge_rel:
        return
    nodes = np.asarray(node_entity_ids, dtype=np.int64)
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)
    rel = np.asarray(edge_rel, dtype=np.int64)
    if src.size == 0 or dst.size == 0 or rel.size == 0:
        return
    src_global = nodes[src]
    dst_global = nodes[dst]
    endpoints = np.concatenate([src_global, dst_global], axis=0)
    rel_ids = np.concatenate([rel, rel], axis=0)
    cvt_mask = is_cvt[endpoints]
    if not np.any(cvt_mask):
        return
    cvt_entities = endpoints[cvt_mask]
    cvt_rels = rel_ids[cvt_mask]
    pairs = np.stack([cvt_entities, cvt_rels], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    if unique_pairs.size == 0:
        return
    entity_ids = unique_pairs[:, 0]
    rel_ids_unique = unique_pairs[:, 1]
    order = np.argsort(entity_ids, kind="mergesort")
    entity_ids = entity_ids[order]
    rel_ids_unique = rel_ids_unique[order]
    unique_entities, idx_start = np.unique(entity_ids, return_index=True)
    for idx, ent_id in enumerate(unique_entities):
        start = int(idx_start[idx])
        end = int(idx_start[idx + 1]) if idx + 1 < len(idx_start) else len(entity_ids)
        rel_slice = rel_ids_unique[start:end]
        rel_set = rel_sets.get(int(ent_id))
        if rel_set is None:
            rel_set = set()
            rel_sets[int(ent_id)] = rel_set
        rel_set.update(rel_slice.tolist())


def _collect_cvt_relation_sets(
    *,
    graphs_path: Path,
    is_cvt: np.ndarray,
    batch_size: int,
    progress_interval: int,
) -> Dict[int, Set[int]]:
    rel_sets: Dict[int, Set[int]] = {}
    processed = _ZERO
    for batch_dict in _iter_graph_batches(graphs_path, batch_size):
        node_lists = batch_dict["node_entity_ids"]
        src_lists = batch_dict["edge_src"]
        dst_lists = batch_dict["edge_dst"]
        rel_lists = batch_dict["edge_relation_ids"]
        for node_entity_ids, edge_src, edge_dst, edge_rel in zip(node_lists, src_lists, dst_lists, rel_lists):
            _update_rel_sets_for_graph(
                node_entity_ids=node_entity_ids,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_rel=edge_rel,
                is_cvt=is_cvt,
                rel_sets=rel_sets,
            )
            processed += _ONE
            if progress_interval > _ZERO and processed % progress_interval == 0:
                log_event(LOGGER, "cvt_patterns_progress", graphs=processed, cvt_nodes=len(rel_sets))
    return rel_sets


def _build_patterns(
    *,
    cvt_entity_ids: np.ndarray,
    rel_sets: Dict[int, Set[int]],
) -> Tuple[np.ndarray, List[Tuple[int, ...]], np.ndarray]:
    pattern_to_id: Dict[Tuple[int, ...], int] = {_EMPTY_PATTERN: _ZERO}
    patterns: List[Tuple[int, ...]] = [_EMPTY_PATTERN]
    node2pattern = np.empty(cvt_entity_ids.shape[0], dtype=np.int32)
    rel_counts = np.empty(cvt_entity_ids.shape[0], dtype=np.int32)
    for idx, ent_id in enumerate(cvt_entity_ids.tolist()):
        rel_set = rel_sets.get(int(ent_id))
        if not rel_set:
            rel_tuple = _EMPTY_PATTERN
        else:
            rel_tuple = tuple(sorted(rel_set))
        rel_counts[idx] = len(rel_tuple)
        pattern_id = pattern_to_id.get(rel_tuple)
        if pattern_id is None:
            pattern_id = len(patterns)
            pattern_to_id[rel_tuple] = pattern_id
            patterns.append(rel_tuple)
        node2pattern[idx] = pattern_id
    return node2pattern, patterns, rel_counts


def _patterns_to_tensor(patterns: List[Tuple[int, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(pattern) for pattern in patterns), default=_ZERO)
    if max_len <= _ZERO:
        comp = torch.zeros((len(patterns), _ONE), dtype=torch.long)
        mask = torch.zeros((len(patterns), _ONE), dtype=torch.bool)
        return comp, mask
    comp = torch.zeros((len(patterns), max_len), dtype=torch.long)
    mask = torch.zeros((len(patterns), max_len), dtype=torch.bool)
    for idx, pattern in enumerate(patterns):
        if not pattern:
            continue
        rel_ids = torch.as_tensor(pattern, dtype=torch.long)
        length = int(rel_ids.numel())
        comp[idx, :length] = rel_ids
        mask[idx, :length] = True
    return comp, mask


def _compute_stats(num_entities: int, cvt_entity_ids: np.ndarray, rel_counts: np.ndarray, num_patterns: int) -> CvtPatternStats:
    rel_counts_np = np.asarray(rel_counts, dtype=np.int64)
    if rel_counts_np.size == 0:
        return CvtPatternStats(
            num_entities=num_entities,
            num_cvt=int(cvt_entity_ids.size),
            num_patterns=int(num_patterns),
            max_rel_count=_ZERO,
            rel_count_p50=_ZERO,
            rel_count_p90=_ZERO,
            rel_count_p99=_ZERO,
            rel_count_max=_ZERO,
            avg_rel_count=0.0,
        )
    rel_count_p50 = int(np.percentile(rel_counts_np, 50))
    rel_count_p90 = int(np.percentile(rel_counts_np, 90))
    rel_count_p99 = int(np.percentile(rel_counts_np, 99))
    rel_count_max = int(rel_counts_np.max())
    avg_rel_count = float(rel_counts_np.mean())
    return CvtPatternStats(
        num_entities=num_entities,
        num_cvt=int(cvt_entity_ids.size),
        num_patterns=int(num_patterns),
        max_rel_count=rel_count_max,
        rel_count_p50=rel_count_p50,
        rel_count_p90=rel_count_p90,
        rel_count_p99=rel_count_p99,
        rel_count_max=rel_count_max,
        avg_rel_count=avg_rel_count,
    )


def precompute_cvt_patterns(
    *,
    graphs_path: Path,
    entity_vocab_path: Path,
    output_dir: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    progress_interval: int = _DEFAULT_PROGRESS_INTERVAL,
    overwrite: bool = False,
) -> None:
    graphs_path = Path(graphs_path)
    entity_vocab_path = Path(entity_vocab_path)
    output_dir = Path(output_dir)
    if not graphs_path.exists():
        raise FileNotFoundError(f"graphs.parquet not found: {graphs_path}")
    if not entity_vocab_path.exists():
        raise FileNotFoundError(f"entity_vocab.parquet not found: {entity_vocab_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"CVT pattern dir already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_ids, is_cvt_flags = _load_entity_vocab(entity_vocab_path)
    num_entities = int(entity_ids.max()) + _ONE
    is_cvt = np.zeros((num_entities,), dtype=bool)
    is_cvt[entity_ids] = is_cvt_flags
    cvt_entity_ids = entity_ids[is_cvt_flags]
    cvt_entity_ids = np.asarray(np.sort(cvt_entity_ids), dtype=np.int64)
    log_event(
        LOGGER,
        "cvt_patterns_start",
        graphs_path=str(graphs_path),
        entity_vocab_path=str(entity_vocab_path),
        output_dir=str(output_dir),
        num_entities=num_entities,
        num_cvt=int(cvt_entity_ids.size),
        batch_size=batch_size,
        progress_interval=progress_interval,
    )
    rel_sets = _collect_cvt_relation_sets(
        graphs_path=graphs_path,
        is_cvt=is_cvt,
        batch_size=batch_size,
        progress_interval=progress_interval,
    )
    node2pattern, patterns, rel_counts = _build_patterns(
        cvt_entity_ids=cvt_entity_ids,
        rel_sets=rel_sets,
    )
    pattern_comp, pattern_mask = _patterns_to_tensor(patterns)
    stats = _compute_stats(num_entities, cvt_entity_ids, rel_counts, len(patterns))
    torch.save(torch.as_tensor(cvt_entity_ids, dtype=torch.long), output_dir / "cvt_entity_ids.pt")
    torch.save(torch.as_tensor(node2pattern, dtype=torch.int32), output_dir / "node2pattern.pt")
    torch.save(pattern_comp.to(dtype=torch.int32), output_dir / "pattern_composition.pt")
    torch.save(pattern_mask, output_dir / "pattern_mask.pt")
    log_event(
        LOGGER,
        "cvt_patterns_done",
        num_entities=stats.num_entities,
        num_cvt=stats.num_cvt,
        num_patterns=stats.num_patterns,
        max_rel_count=stats.max_rel_count,
        rel_count_p50=stats.rel_count_p50,
        rel_count_p90=stats.rel_count_p90,
        rel_count_p99=stats.rel_count_p99,
        rel_count_max=stats.rel_count_max,
        avg_rel_count=round(stats.avg_rel_count, 4),
        output_dir=str(output_dir),
    )
