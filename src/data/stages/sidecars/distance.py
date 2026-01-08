from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import pyarrow.parquet as pq
import torch

from src.data.context import StageContext
from src.data.io.lmdb_utils import (
    _build_core_envs,
    _deserialize_sample,
    _require_ascii_key,
    _resolve_core_lmdb_paths,
    _select_core_env,
    ensure_dir,
)
from src.data.schema.constants import (
    _DISTANCE_DEFAULT_CHUNK_SIZE,
    _DISTANCE_DEFAULT_NUM_WORKERS,
    _DISTANCE_DEFAULT_PROGRESS_INTERVAL,
    _DISTANCE_EDGE_WEIGHT,
    _DISTANCE_INT16_MAX,
    _DISTANCE_INT8_MAX,
    _DISTANCE_LMDB_MAX_READERS,
    _DISTANCE_MIN_CHUNK_SIZE,
    _DISTANCE_MIN_WORKERS,
    _DISTANCE_PROGRESS_DISABLED,
    _DIST_UNREACHABLE,
    _ONE,
    _ZERO,
)
from src.data.utils.validation import _validate_split_names
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)

_DISTANCE_ENV_CACHE: Dict[str, lmdb.Environment] = {}
_DISTANCE_PT_SUFFIX = ".pt"


def _require_scipy():
    try:
        import scipy.sparse  # type: ignore[import-not-found]
        import scipy.sparse.csgraph  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Distance precompute requires scipy. Install scipy or disable precompute_distances.") from exc
    return scipy.sparse, scipy.sparse.csgraph


def _resolve_distance_num_workers(cfg) -> int:
    num_workers_cfg = cfg.get("distance_num_workers")
    num_workers = _DISTANCE_DEFAULT_NUM_WORKERS if num_workers_cfg is None else int(num_workers_cfg)
    if num_workers < _DISTANCE_MIN_WORKERS:
        raise ValueError(f"distance_num_workers must be >= {_DISTANCE_MIN_WORKERS}, got {num_workers}")
    return num_workers


def _resolve_distance_chunk_size(cfg) -> int:
    chunk_cfg = cfg.get("distance_chunk_size")
    chunk_size = _DISTANCE_DEFAULT_CHUNK_SIZE if chunk_cfg is None else int(chunk_cfg)
    if chunk_size < _DISTANCE_MIN_CHUNK_SIZE:
        raise ValueError(f"distance_chunk_size must be >= {_DISTANCE_MIN_CHUNK_SIZE}, got {chunk_size}")
    return chunk_size


def _resolve_distance_progress_interval(cfg) -> int:
    progress_cfg = cfg.get("distance_progress_interval")
    progress_interval = _DISTANCE_DEFAULT_PROGRESS_INTERVAL if progress_cfg is None else int(progress_cfg)
    if progress_interval < _DISTANCE_PROGRESS_DISABLED:
        raise ValueError(f"distance_progress_interval must be >= {_DISTANCE_PROGRESS_DISABLED}, got {progress_interval}")
    return progress_interval


def _load_question_splits(parquet_dir: Path) -> List[str]:
    questions_path = parquet_dir / "questions.parquet"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.parquet not found at {questions_path}")
    table = pq.read_table(questions_path, columns=["split"])
    splits = table.column("split").unique().to_pylist()
    return _validate_split_names(splits, context="questions.parquet")


def _resolve_distance_splits(cfg, parquet_dir: Path) -> List[str]:
    splits_cfg = cfg.get("distance_splits")
    if splits_cfg is None:
        return _load_question_splits(parquet_dir)
    if isinstance(splits_cfg, str):
        raise TypeError("distance_splits must be a sequence of split names, not a string.")
    splits = [str(item) for item in list(splits_cfg)]
    return _validate_split_names(splits, context="distance_splits")


def _iter_lmdb_keys(path: Path) -> Iterable[bytes]:
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=_DISTANCE_LMDB_MAX_READERS,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(values=False):
                yield key
    finally:
        env.close()


def _get_lmdb_env(path: str) -> lmdb.Environment:
    env = _DISTANCE_ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=_DISTANCE_LMDB_MAX_READERS,
        )
        _DISTANCE_ENV_CACHE[path] = env
    return env


def _to_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return array


def _build_directed_csr(edge_index: np.ndarray, num_nodes: int, *, reverse: bool) -> Any:
    scipy_sparse, _ = _require_scipy()
    src = edge_index[0]
    dst = edge_index[1]
    if reverse:
        row = dst
        col = src
    else:
        row = src
        col = dst
    data = np.full(row.shape[0], _DISTANCE_EDGE_WEIGHT, dtype=np.int8)
    return scipy_sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))


def _compute_distances(edge_index: np.ndarray, num_nodes: int, answer_nodes: np.ndarray) -> np.ndarray:
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return np.zeros((0,), dtype=np.int8)
    answer_nodes = np.asarray(answer_nodes, dtype=np.int64).reshape(-1)
    if answer_nodes.size == 0:
        return np.full((num_nodes,), _DIST_UNREACHABLE, dtype=np.int8)
    if answer_nodes.min() < 0 or answer_nodes.max() >= num_nodes:
        raise ValueError("answer_nodes indices out of range for distance computation.")
    answer_nodes = np.unique(answer_nodes)
    if edge_index.size == 0:
        dist = np.full((num_nodes,), _DIST_UNREACHABLE, dtype=np.int8)
        dist[answer_nodes] = 0
        return dist
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError("edge_index contains out-of-range node ids.")

    _, csgraph = _require_scipy()
    graph = _build_directed_csr(edge_index, num_nodes, reverse=True)
    dist = csgraph.shortest_path(
        graph,
        directed=True,
        unweighted=True,
        indices=answer_nodes,
    )
    if dist.ndim == 2:
        dist = dist.min(axis=0)
    dist = np.asarray(dist)
    dist = np.where(np.isfinite(dist), dist, _DIST_UNREACHABLE)
    dist = np.rint(dist).astype(np.int64, copy=False)
    min_val = int(dist.min()) if dist.size > 0 else _DIST_UNREACHABLE
    if min_val < _DIST_UNREACHABLE:
        raise ValueError(f"Distances contain values below {_DIST_UNREACHABLE}: {min_val}")
    max_val = int(dist.max()) if dist.size > 0 else _DIST_UNREACHABLE
    if max_val <= _DISTANCE_INT8_MAX:
        return dist.astype(np.int8, copy=False)
    if max_val <= _DISTANCE_INT16_MAX:
        return dist.astype(np.int16, copy=False)
    raise ValueError(f"Max distance {max_val} exceeds int16 range.")


def _compute_distance_from_raw(raw: Dict[str, Any]) -> np.ndarray:
    edge_index = _to_numpy(raw["edge_index"], dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    num_nodes = _extract_num_nodes(raw)
    answer_nodes = _to_numpy(raw["a_local_indices"], dtype=np.int64)
    return _compute_distances(edge_index, num_nodes, answer_nodes)


def _process_distance_task_pt(task: Tuple[int, str, str, bytes]) -> Tuple[int, np.ndarray]:
    idx, sample_id, shard_path, sample_key = task
    env = _get_lmdb_env(shard_path)
    with env.begin(write=False) as txn:
        payload = txn.get(sample_key)
    if payload is None:
        raise KeyError(f"Sample {sample_id} not found in {shard_path}.")
    raw = _deserialize_sample(payload)
    dist = _compute_distance_from_raw(raw)
    dist = dist.astype(np.int16, copy=False)
    return idx, dist


def _collect_distance_samples(core_paths: List[Path]) -> List[Tuple[str, str, bytes]]:
    samples: List[Tuple[str, str, bytes]] = []
    for path in core_paths:
        for key in _iter_lmdb_keys(path):
            _require_ascii_key(key, context=str(path))
            samples.append((key.decode("ascii"), str(path), key))
    if not samples:
        raise ValueError("No samples found in core LMDB shards.")
    samples.sort(key=lambda item: item[0])
    sample_ids = [item[0] for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample ids detected in core LMDB shards.")
    return samples


def _prepare_distance_pt_samples(
    *,
    split: str,
    core_paths: List[Path],
    output_path: Path,
    skip_existing: bool,
    overwrite: bool,
) -> List[Tuple[str, str, bytes]]:
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        elif skip_existing:
            log_event(LOGGER, "distance_pt_skip", split=split, output=str(output_path))
            return []
        else:
            raise FileExistsError(f"Distance PT already exists at {output_path}; set distance_pt_overwrite=true.")
    return _collect_distance_samples(core_paths)


def _init_distance_pt_buffers(num_nodes_list: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(list(num_nodes_list), dtype=torch.long)
    ptr = torch.zeros(len(num_nodes_list) + _ONE, dtype=torch.long)
    if lengths.numel() > _ZERO:
        ptr[_ONE:] = torch.cumsum(lengths, dim=0)
    total = int(ptr[-1].item())
    values = torch.empty((total,), dtype=torch.int16)
    return ptr, values


def _fill_distance_pt_values(
    *,
    split: str,
    tasks: List[Tuple[int, str, str, bytes]],
    sample_ids: Sequence[str],
    ptr: torch.Tensor,
    values: torch.Tensor,
    num_workers: int,
    chunk_size: int,
    progress_interval: int,
) -> int:
    ctx = mp.get_context()
    processed = 0
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_process_distance_task_pt, tasks, chunksize=chunk_size)
        for idx, dist in results:
            start = int(ptr[idx].item())
            end = int(ptr[idx + _ONE].item())
            expected = end - start
            if dist.shape[0] != expected:
                sample_id = sample_ids[idx]
                raise ValueError(
                    f"Distance length mismatch for sample_id={sample_id}: expected {expected} got {dist.shape[0]}."
                )
            values[start:end] = torch.from_numpy(dist)
            processed += 1
            if progress_interval > _DISTANCE_PROGRESS_DISABLED and processed % progress_interval == _ZERO:
                log_event(LOGGER, "distance_pt_progress", split=split, processed=processed)
    return processed


def _write_distance_pt_payload(
    *,
    split: str,
    samples: Sequence[Tuple[str, str, bytes]],
    num_nodes_list: Sequence[int],
    output_path: Path,
    num_workers: int,
    chunk_size: int,
    progress_interval: int,
) -> int:
    sample_ids = [item[0] for item in samples]
    tasks = [(idx, sample_ids[idx], samples[idx][1], samples[idx][2]) for idx in range(len(samples))]
    ptr, values = _init_distance_pt_buffers(num_nodes_list)
    processed = _fill_distance_pt_values(
        split=split,
        tasks=tasks,
        sample_ids=sample_ids,
        ptr=ptr,
        values=values,
        num_workers=num_workers,
        chunk_size=chunk_size,
        progress_interval=progress_interval,
    )
    payload = {"sample_ids": sample_ids, "ptr": ptr, "values": values}
    torch.save(payload, output_path)
    log_event(
        LOGGER,
        "distance_pt_written",
        split=split,
        samples=len(sample_ids),
        values=int(values.numel()),
        output=str(output_path),
    )
    return processed


def _precompute_distance_pt_split(
    *,
    split: str,
    core_paths: List[Path],
    output_path: Path,
    num_workers: int,
    chunk_size: int,
    progress_interval: int,
    skip_existing: bool,
    overwrite: bool,
) -> None:
    samples = _prepare_distance_pt_samples(
        split=split,
        core_paths=core_paths,
        output_path=output_path,
        skip_existing=skip_existing,
        overwrite=overwrite,
    )
    if not samples:
        return
    log_event(
        LOGGER,
        "distance_pt_start",
        split=split,
        samples=len(samples),
        workers=num_workers,
        output=str(output_path),
    )
    envs = _build_core_envs(core_paths, max_readers=_DISTANCE_LMDB_MAX_READERS)
    try:
        sample_ids = [item[0] for item in samples]
        num_nodes_list = _load_num_nodes_list(sample_ids, envs)
    finally:
        for env in envs.values():
            env.close()
    processed = _write_distance_pt_payload(
        split=split,
        samples=samples,
        num_nodes_list=num_nodes_list,
        output_path=output_path,
        num_workers=num_workers,
        chunk_size=chunk_size,
        progress_interval=progress_interval,
    )
    log_event(LOGGER, "distance_pt_done", split=split, samples=processed, output=str(output_path))


def run_distance_stage(ctx: StageContext) -> None:
    run_distance_pt_stage(ctx)


def _extract_num_nodes(raw: Dict[str, Any]) -> int:
    num_nodes = raw.get("num_nodes")
    if num_nodes is None:
        node_ids = raw.get("node_global_ids")
        if node_ids is None:
            raise KeyError("Sample missing num_nodes and node_global_ids.")
        if torch.is_tensor(node_ids):
            num_nodes = int(node_ids.numel())
        else:
            num_nodes = int(len(node_ids))
    elif torch.is_tensor(num_nodes):
        num_nodes = int(num_nodes.item())
    else:
        num_nodes = int(num_nodes)
    if num_nodes < _ZERO:
        raise ValueError(f"num_nodes must be >= {_ZERO}, got {num_nodes}.")
    return num_nodes


def _load_core_num_nodes(sample_id: str, envs: Dict[int, lmdb.Environment]) -> int:
    sample_key = sample_id.encode("ascii")
    env = _select_core_env(sample_key, envs)
    with env.begin(write=False) as txn:
        payload = txn.get(sample_key)
    if payload is None:
        raise KeyError(f"Sample {sample_id} not found in core LMDB.")
    raw = _deserialize_sample(payload)
    return _extract_num_nodes(raw)


def _load_num_nodes_list(sample_ids: Sequence[str], envs: Dict[int, lmdb.Environment]) -> List[int]:
    return [_load_core_num_nodes(sample_id, envs) for sample_id in sample_ids]


def run_distance_pt_stage(ctx: StageContext) -> None:
    cfg = ctx.cfg
    if not bool(cfg.get("precompute_distances", False)):
        return
    parquet_dir = ctx.parquet_dir
    embeddings_dir = ctx.embeddings_dir
    distances_dir = ctx.output_dir / "distances"
    splits = _resolve_distance_splits(cfg, parquet_dir)
    num_workers = _resolve_distance_num_workers(cfg)
    chunk_size = _resolve_distance_chunk_size(cfg)
    progress_interval = _resolve_distance_progress_interval(cfg)
    skip_existing = bool(cfg.get("distance_skip_existing", True))
    overwrite_cfg = cfg.get("distance_pt_overwrite")
    if overwrite_cfg is None:
        raise ValueError("distance_pt_overwrite must be set explicitly; no defaults are allowed.")
    overwrite = bool(overwrite_cfg)
    ensure_dir(distances_dir)
    log_event(
        LOGGER,
        "distance_pt_stage_start",
        splits=splits,
        num_workers=num_workers,
        chunk_size=chunk_size,
        progress_interval=progress_interval,
        skip_existing=skip_existing,
        overwrite=overwrite,
    )
    for split in splits:
        core_paths = _resolve_core_lmdb_paths(embeddings_dir, split)
        output_path = distances_dir / f"{split}{_DISTANCE_PT_SUFFIX}"
        _precompute_distance_pt_split(
            split=split,
            core_paths=core_paths,
            output_path=output_path,
            num_workers=num_workers,
            chunk_size=chunk_size,
            progress_interval=progress_interval,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )
    log_event(LOGGER, "distance_pt_stage_done", splits=splits, output_dir=str(distances_dir))
