#!/usr/bin/env python3
"""Analyze answer reachability upper bounds in eval_dual_flow artifacts.

This script quantifies, for each requested top-k rollout cutoff:
1) Stop-node oracle: answer entity appears in any rollout stop node.
2) Any-node oracle: answer entity appears anywhere in rollout nodes/edges.
3) Endpoint-only loss: answer is reachable in path nodes but not in stop nodes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

_FIELD_SAMPLE_ID = "sample_id"


def _parse_topk_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"top-k must be > 0, got {value}")
        out.append(value)
    if not out:
        raise ValueError("top-k list is empty")
    return sorted(set(out))


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _to_nonneg_int(value) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    if out < 0:
        return None
    return out


def _collect_stop_ids(rollouts: List[Dict]) -> Set[int]:
    out: Set[int] = set()
    for ro in rollouts:
        node = _to_nonneg_int(ro.get("stop_node_entity_id"))
        if node is not None:
            out.add(node)
    return out


def _collect_any_ids(rollouts: List[Dict]) -> Set[int]:
    out: Set[int] = set()
    for ro in rollouts:
        node = _to_nonneg_int(ro.get("stop_node_entity_id"))
        if node is not None:
            out.add(node)
        edges = ro.get("edges")
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            for key in ("src_entity_id", "dst_entity_id", "head_entity_id", "tail_entity_id"):
                value = _to_nonneg_int(edge.get(key))
                if value is not None:
                    out.add(value)
    return out


def _load_indexed_jsonl(path: Path) -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for row in _iter_jsonl(path):
        sample_id = str(row.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id:
            raise ValueError(f"Missing sample_id in {path}")
        if sample_id in data:
            raise ValueError(f"Duplicate sample_id in {path}: {sample_id!r}")
        data[sample_id] = row
    return data


def _pct(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(num) / float(denom)


def _analyze(
    *,
    input_rows: Dict[str, Dict],
    label_rows: Dict[str, Dict],
    topk_values: List[int],
) -> Dict[int, Counter]:
    if set(input_rows.keys()) != set(label_rows.keys()):
        missing_input = sorted(set(label_rows.keys()) - set(input_rows.keys()))
        missing_labels = sorted(set(input_rows.keys()) - set(label_rows.keys()))
        raise ValueError(
            "sample_id mismatch between input and labels. "
            f"missing_input={missing_input[:5]} missing_labels={missing_labels[:5]}"
        )

    out: Dict[int, Counter] = {k: Counter() for k in topk_values}
    for sample_id, row in input_rows.items():
        labels = label_rows[sample_id]
        if not bool(labels.get("a_entity_in_graph")):
            continue
        answer_ids = {_to_nonneg_int(x) for x in (labels.get("answer_entity_ids") or [])}
        answer_ids = {x for x in answer_ids if x is not None}
        if not answer_ids:
            continue
        rollouts = row.get("rollouts")
        if not isinstance(rollouts, list):
            rollouts = []
        for k in topk_values:
            stats = out[k]
            stats["good_total"] += 1
            selected = rollouts[:k]
            stop_ids = _collect_stop_ids(selected)
            any_ids = _collect_any_ids(selected)
            has_stop_oracle = bool(answer_ids & stop_ids)
            has_any_oracle = bool(answer_ids & any_ids)
            if has_stop_oracle:
                stats["stop_oracle"] += 1
            if has_any_oracle:
                stats["any_oracle"] += 1
            if (not has_stop_oracle) and has_any_oracle:
                stats["endpoint_only_loss"] += 1
            if not has_any_oracle:
                stats["no_any_oracle"] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="eval_dual_flow input JSONL (with rollouts)")
    parser.add_argument("--labels", type=Path, required=True, help="labels sidecar JSONL")
    parser.add_argument(
        "--topk",
        type=str,
        default="1,10,25,50",
        help="Comma-separated rollout cutoffs, e.g. 1,10,25,50",
    )
    args = parser.parse_args()

    topk_values = _parse_topk_list(args.topk)
    input_rows = _load_indexed_jsonl(args.input)
    label_rows = _load_indexed_jsonl(args.labels)
    stats_by_k = _analyze(input_rows=input_rows, label_rows=label_rows, topk_values=topk_values)

    print("k\tgood_total\tstop_oracle%\tany_oracle%\tendpoint_loss%\tno_any_oracle%")
    for k in topk_values:
        stats = stats_by_k[k]
        total = int(stats.get("good_total", 0))
        stop_oracle = int(stats.get("stop_oracle", 0))
        any_oracle = int(stats.get("any_oracle", 0))
        endpoint_loss = int(stats.get("endpoint_only_loss", 0))
        no_any_oracle = int(stats.get("no_any_oracle", 0))
        print(
            f"{k}\t{total}\t"
            f"{_pct(stop_oracle, total):.2f}\t"
            f"{_pct(any_oracle, total):.2f}\t"
            f"{_pct(endpoint_loss, total):.2f}\t"
            f"{_pct(no_any_oracle, total):.2f}"
        )


if __name__ == "__main__":
    main()
