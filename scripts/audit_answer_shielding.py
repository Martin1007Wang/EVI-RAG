from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import lmdb
import torch
from safetensors.torch import load as safe_load


_LMDB_MAX_READERS = 256


@dataclass(frozen=True)
class _QueryStats:
    sample_id: str
    num_nodes: int
    num_edges: int
    num_answers: int
    num_starts: int
    shielding_exists: bool
    reachable_answers_any: int
    reachable_answers: int


def _resolve_lmdb_paths(embeddings_dir: Path, split: str) -> list[Path]:
    split = str(split)
    base = embeddings_dir / f"{split}.lmdb"
    shards = sorted(embeddings_dir.glob(f"{split}.shard*.lmdb"))
    if base.exists():
        if shards:
            raise ValueError(f"Both sharded and unsharded LMDBs exist under {embeddings_dir} for split={split}.")
        return [base]
    if not shards:
        raise FileNotFoundError(f"LMDB not found under {embeddings_dir} for split={split}.")
    shard_map: dict[int, Path] = {}
    token = f"{split}.shard"
    for path in shards:
        stem = path.stem
        if not stem.startswith(token):
            raise ValueError(f"Unexpected LMDB shard name: {path.name}")
        shard_text = stem[len(token) :]
        if not shard_text.isdigit():
            raise ValueError(f"Invalid shard id in LMDB shard: {path.name}")
        shard_id = int(shard_text)
        if shard_id in shard_map:
            raise ValueError(f"Duplicate LMDB shard id={shard_id} for split={split}.")
        shard_map[shard_id] = path
    shard_ids = sorted(shard_map)
    expected = list(range(shard_ids[-1] + 1))
    if shard_ids != expected:
        raise ValueError(f"LMDB shards must be contiguous from 0; found {shard_ids} for split={split}.")
    return [shard_map[shard_id] for shard_id in expected]


def _load_filter_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    data = json.loads(text)
    if isinstance(data, list):
        return set(map(str, data))
    if isinstance(data, dict):
        sample_ids = data.get("sample_ids")
        if sample_ids is None:
            raise ValueError(f"Filter JSON dict missing 'sample_ids': {path}")
        return set(map(str, sample_ids))
    raise ValueError(f"Filter JSON must be list or dict: {path}")


def _iter_lmdb_entries(
    path: Path,
    *,
    keep_ids: Optional[set[str]] = None,
    limit: Optional[int] = None,
    readahead: bool = False,
) -> Iterable[tuple[str, bytes]]:
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=bool(readahead),
        meminit=False,
        max_readers=_LMDB_MAX_READERS,
    )
    emitted = 0
    try:
        with env.begin(write=False) as txn:
            cur = txn.cursor()
            for key, value in cur.iternext(keys=True, values=True):
                sample_id = key.decode("utf-8")
                if keep_ids is not None and sample_id not in keep_ids:
                    continue
                yield sample_id, value
                emitted += 1
                if limit is not None and emitted >= int(limit):
                    break
    finally:
        env.close()


def _to_unique_int_list(x: torch.Tensor) -> list[int]:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor, got {type(x)}")
    values = x.to(dtype=torch.long).view(-1).tolist()
    if not values:
        return []
    return sorted(set(int(v) for v in values))


def _build_adj(num_nodes: int, heads: list[int], tails: list[int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(heads, tails):
        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            raise ValueError(f"edge_index out of range: u={u} v={v} num_nodes={num_nodes}")
        adj[u].append(v)
    return adj


def _has_answer_to_answer_path(adj: list[list[int]], *, is_answer: list[bool], answers: list[int]) -> bool:
    num_nodes = len(adj)
    for src in answers:
        visited = [False] * num_nodes
        dq: deque[int] = deque([src])
        visited[src] = True
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if visited[v]:
                    continue
                if is_answer[v] and v != src:
                    return True
                visited[v] = True
                dq.append(v)
    return False


def _reachable_answers_any(
    *,
    adj: list[list[int]],
    starts: list[int],
    answers: list[int],
    max_steps: Optional[int],
) -> int:
    if max_steps is not None and int(max_steps) < 0:
        raise ValueError(f"max_steps must be >= 0 or None, got {max_steps}")

    num_nodes = len(adj)
    is_answer = [False] * num_nodes
    for a in answers:
        is_answer[a] = True

    reachable_answer = [False] * num_nodes
    dist = [-1] * num_nodes
    dq: deque[int] = deque()
    for s in starts:
        if dist[s] != -1:
            continue
        dist[s] = 0
        dq.append(s)
        if is_answer[s]:
            reachable_answer[s] = True

    while dq:
        u = dq.popleft()
        if max_steps is not None and dist[u] >= int(max_steps):
            continue
        for v in adj[u]:
            if dist[v] != -1:
                continue
            dist[v] = dist[u] + 1
            dq.append(v)
            if is_answer[v]:
                reachable_answer[v] = True

    return sum(1 for a in answers if reachable_answer[a])


def _reachable_answers_hit_and_stop(
    *,
    adj: list[list[int]],
    heads: list[int],
    tails: list[int],
    starts: list[int],
    answers: list[int],
    max_steps: Optional[int],
) -> int:
    if max_steps is not None and int(max_steps) < 0:
        raise ValueError(f"max_steps must be >= 0 or None, got {max_steps}")

    num_nodes = len(adj)
    is_answer = [False] * num_nodes
    for a in answers:
        is_answer[a] = True

    reachable_answer = [False] * num_nodes
    start_non_answer: list[int] = []
    for s in starts:
        if is_answer[s]:
            reachable_answer[s] = True
        else:
            start_non_answer.append(s)

    reachable_non_answer = [False] * num_nodes
    dist_non_answer = [-1] * num_nodes
    dq: deque[int] = deque()
    for s in start_non_answer:
        if not reachable_non_answer[s]:
            reachable_non_answer[s] = True
            dist_non_answer[s] = 0
            dq.append(s)

    while dq:
        u = dq.popleft()
        if max_steps is not None and dist_non_answer[u] >= int(max_steps) - 1:
            continue
        for v in adj[u]:
            if is_answer[v] or reachable_non_answer[v]:
                continue
            reachable_non_answer[v] = True
            dist_non_answer[v] = dist_non_answer[u] + 1
            dq.append(v)

    for u, v in zip(heads, tails):
        if not reachable_non_answer[u] or not is_answer[v]:
            continue
        if max_steps is not None and dist_non_answer[u] + 1 > int(max_steps):
            continue
        if reachable_non_answer[u] and is_answer[v]:
            reachable_answer[v] = True

    return sum(1 for a in answers if reachable_answer[a])


def _analyze_sample(sample_id: str, payload: bytes, *, max_steps: Optional[int]) -> Optional[_QueryStats]:
    sample = safe_load(payload)
    edge_index = sample.get("edge_index")
    if edge_index is None:
        raise KeyError(f"{sample_id}: missing edge_index")
    num_nodes_raw = sample.get("num_nodes")
    if num_nodes_raw is None:
        node_global_ids = sample.get("node_global_ids")
        if node_global_ids is None:
            raise KeyError(f"{sample_id}: missing num_nodes and node_global_ids")
        num_nodes = int(torch.as_tensor(node_global_ids).numel())
    else:
        num_nodes = int(torch.as_tensor(num_nodes_raw).view(-1)[0].item())
    if num_nodes <= 0:
        return None

    edge_index = torch.as_tensor(edge_index).to(dtype=torch.long)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{sample_id}: edge_index must be [2, E], got {tuple(edge_index.shape)}")
    num_edges = int(edge_index.size(1))
    heads = edge_index[0].tolist()
    tails = edge_index[1].tolist()
    adj = _build_adj(num_nodes, heads, tails)

    answers = _to_unique_int_list(torch.as_tensor(sample.get("a_local_indices", torch.empty((0,), dtype=torch.long))))
    starts = _to_unique_int_list(torch.as_tensor(sample.get("q_local_indices", torch.empty((0,), dtype=torch.long))))
    if len(answers) < 2:
        reachable_any = _reachable_answers_any(adj=adj, starts=starts, answers=answers, max_steps=max_steps)
        return _QueryStats(
            sample_id=sample_id,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_answers=len(answers),
            num_starts=len(starts),
            shielding_exists=False,
            reachable_answers_any=int(reachable_any),
            reachable_answers=len(answers),
        )

    is_answer = [False] * num_nodes
    for a in answers:
        is_answer[a] = True
    shielding_exists = _has_answer_to_answer_path(adj, is_answer=is_answer, answers=answers)
    reachable_any = _reachable_answers_any(adj=adj, starts=starts, answers=answers, max_steps=max_steps)
    reachable_answers = _reachable_answers_hit_and_stop(
        adj=adj,
        heads=heads,
        tails=tails,
        starts=starts,
        answers=answers,
        max_steps=max_steps,
    )
    return _QueryStats(
        sample_id=sample_id,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_answers=len(answers),
        num_starts=len(starts),
        shielding_exists=shielding_exists,
        reachable_answers_any=int(reachable_any),
        reachable_answers=int(reachable_answers),
    )


def _format_pct(num: float) -> str:
    return f"{100.0 * float(num):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Answer Shielding (Hit-and-Stop upper bound).")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data/retrieval_dataset"))
    parser.add_argument("--dataset", type=str, default="webqsp", help="e.g. webqsp, webqsp-sub, cwq, cwq-sub")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max path length (edges) from a start to an answer when computing bounds.",
    )
    parser.add_argument("--readahead", action="store_true", help="Enable LMDB readahead (may help on HDD).")
    parser.add_argument("--show-examples", type=int, default=0, help="Print first N shielding/unreachable sample_ids.")
    args = parser.parse_args()

    dataset = str(args.dataset).strip()
    dataset_family = dataset.removesuffix("-sub")
    is_sub = dataset.endswith("-sub")
    base = Path(args.data_dir) / dataset_family
    embeddings_dir = base / "materialized" / "embeddings"
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"embeddings_dir not found: {embeddings_dir}")
    keep_ids = None
    if is_sub:
        filter_path = base / "normalized" / "sub_filter.json"
        if not filter_path.exists():
            raise FileNotFoundError(f"sub_filter.json not found: {filter_path}")
        keep_ids = _load_filter_ids(filter_path)

    paths = _resolve_lmdb_paths(embeddings_dir, args.split)

    t0 = time.time()
    total = 0
    multi_answer = 0
    shielding = 0
    any_disconnected_q = 0
    any_shielded_q = 0
    total_answers = 0
    total_answers_any = 0
    total_reachable_answers = 0
    total_shielded_answers = 0
    k_hist: Counter[int] = Counter()
    recall_macro_sum = 0.0
    recall_any_macro_sum = 0.0
    examples: list[str] = []

    remaining = args.limit
    for path in paths:
        local_limit = None
        if remaining is not None:
            local_limit = max(int(remaining), 0)
            if local_limit == 0:
                break
        for sample_id, payload in _iter_lmdb_entries(
            path,
            keep_ids=keep_ids,
            limit=local_limit,
            readahead=bool(args.readahead),
        ):
            stats = _analyze_sample(sample_id, payload, max_steps=args.max_steps)
            if stats is None:
                continue
            total += 1
            if stats.num_answers >= 2:
                multi_answer += 1
                k_hist.update([stats.num_answers])
                total_answers += stats.num_answers
                total_answers_any += stats.reachable_answers_any
                total_reachable_answers += stats.reachable_answers
                if stats.shielding_exists:
                    shielding += 1
                if stats.reachable_answers_any < stats.num_answers:
                    any_disconnected_q += 1
                if stats.reachable_answers < stats.reachable_answers_any:
                    any_shielded_q += 1
                total_shielded_answers += max(stats.reachable_answers_any - stats.reachable_answers, 0)

                recall_macro_sum += stats.reachable_answers / max(stats.num_answers, 1)
                recall_any_macro_sum += stats.reachable_answers_any / max(stats.num_answers, 1)

                if stats.reachable_answers < stats.num_answers:
                    if len(examples) < int(args.show_examples):
                        examples.append(
                            f"{stats.sample_id} k={stats.num_answers} reachable={stats.reachable_answers}/{stats.reachable_answers_any}"
                        )
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    elapsed = time.time() - t0
    print(f"dataset={dataset} split={args.split} samples={total} elapsed={elapsed:.1f}s")
    if is_sub and keep_ids is not None:
        print(f"scope=sub filter_size={len(keep_ids)}")
    if args.max_steps is not None:
        print(f"max_steps={int(args.max_steps)}")

    if multi_answer == 0:
        print("multi_answer_queries=0 (no stats)")
        return

    shielding_rate = shielding / multi_answer
    disconnected_rate = any_disconnected_q / multi_answer
    shielded_rate = any_shielded_q / multi_answer

    recall_upper_micro = total_reachable_answers / max(total_answers, 1)
    recall_upper_any_micro = total_answers_any / max(total_answers, 1)
    recall_upper_macro = recall_macro_sum / max(multi_answer, 1)
    recall_upper_any_macro = recall_any_macro_sum / max(multi_answer, 1)

    shielded_answers_micro = total_shielded_answers / max(total_answers, 1)
    shielded_answers_cond_micro = total_shielded_answers / max(total_answers_any, 1)
    print(f"multi_answer_queries={multi_answer}")
    print(f"shielding_rate={_format_pct(shielding_rate)} ({shielding}/{multi_answer})")
    print(f"queries_with_disconnected_answers={_format_pct(disconnected_rate)} ({any_disconnected_q}/{multi_answer})")
    print(f"queries_with_shielded_answers={_format_pct(shielded_rate)} ({any_shielded_q}/{multi_answer})")
    print(f"recall_upper_bound_micro={_format_pct(recall_upper_micro)} ({total_reachable_answers}/{total_answers})")
    print(f"recall_upper_bound_macro={_format_pct(recall_upper_macro)}")
    print(f"recall_upper_bound_graph_micro={_format_pct(recall_upper_any_micro)} ({total_answers_any}/{total_answers})")
    print(f"recall_upper_bound_graph_macro={_format_pct(recall_upper_any_macro)}")
    print(f"shielded_answer_fraction_micro={_format_pct(shielded_answers_micro)} ({total_shielded_answers}/{total_answers})")
    print(
        f"shielded_answer_fraction_given_reachable_micro={_format_pct(shielded_answers_cond_micro)}"
        f" ({total_shielded_answers}/{total_answers_any})"
    )
    print(f"answer_count_hist={dict(sorted(k_hist.items()))}")
    if examples:
        print("examples_unreachable:")
        for ex in examples:
            print(f"  {ex}")

if __name__ == "__main__":
    main()
