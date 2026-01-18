#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required to read relation_vocab.parquet.") from exc

_ZERO = 0
_ONE = 1
_FIELD_LABEL = "label"
_FIELD_KG_ID = "kg_id"
_KEY_FORWARD = "forward"
_KEY_FORWARD_TEXT = "forward_text"
_KEY_INVERSE = "inverse"


def _load_relation_records(path: Path, *, inverse_suffix: str | None) -> List[Tuple[str, str]]:
    columns = [_FIELD_KG_ID, _FIELD_LABEL]
    table = pq.read_table(path, columns=columns, use_threads=False)
    kg_ids = table.column(_FIELD_KG_ID).to_pylist()
    labels = table.column(_FIELD_LABEL).to_pylist()
    out: List[Tuple[str, str]] = []
    for label, kg_id in zip(labels, kg_ids):
        kg_id = "" if kg_id is None else str(kg_id)
        if inverse_suffix and kg_id.endswith(inverse_suffix):
            continue
        out.append((kg_id, str(label)))
    return out


def _build_stub(relations: Sequence[Tuple[str, str]], *, include_forward_text: bool) -> dict:
    entries = []
    for kg_id, label in relations:
        entry = {_KEY_FORWARD: kg_id, _KEY_INVERSE: ""}
        if include_forward_text:
            entry[_KEY_FORWARD_TEXT] = label
        entries.append(entry)
    return {"inverse_relations": entries}


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export inverse relation stub JSON.")
    parser.add_argument(
        "--relation-vocab",
        type=Path,
        nargs="+",
        required=True,
        help="One or more relation_vocab.parquet paths",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to write inverse_relations stub JSON")
    parser.add_argument("--inverse-suffix", type=str, default=None, help="Filter out relations ending with suffix")
    parser.add_argument(
        "--include-forward-text",
        action="store_true",
        help="Include forward_text entries using the relation label column",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    vocab_paths = args.relation_vocab
    for vocab_path in vocab_paths:
        if not vocab_path.exists():
            raise FileNotFoundError(f"relation_vocab.parquet not found: {vocab_path}")
    out_path = args.output
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
    relation_records: List[Tuple[str, str]] = []
    seen = set()
    for vocab_path in vocab_paths:
        for kg_id, label in _load_relation_records(vocab_path, inverse_suffix=args.inverse_suffix):
            if kg_id in seen:
                continue
            seen.add(kg_id)
            relation_records.append((kg_id, label))
    relation_records.sort(key=lambda rec: rec[0])
    payload = _build_stub(relation_records, include_forward_text=args.include_forward_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote inverse relation stub with {len(relation_records)} relations to {out_path}")
    return _ZERO


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[_ONE:]))
