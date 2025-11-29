#!/usr/bin/env python3
"""Convert a g_agent_samples.pt file into a JSON dump for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_g_agent(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return torch.load(path, map_location="cpu")


def _serialize(payload: Dict[str, Any], max_samples: int | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "settings": payload.get("settings"),
        "num_samples": payload.get("num_samples"),
        "samples": payload.get("samples", []),
    }
    samples: List[Dict[str, Any]] = result["samples"] or []
    if max_samples is not None:
        result["samples"] = samples[: max(0, max_samples)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to g_agent_samples.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("g_agent_samples.json"),
        help="Destination JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate to the first N samples (default: entire file)",
    )
    args = parser.parse_args()

    payload = _load_g_agent(args.input)
    serialized = _serialize(payload, args.max_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(serialized, fp, ensure_ascii=False, indent=2)
    print(f"Wrote JSON preview to {args.output} (samples={len(serialized['samples'])})")


if __name__ == "__main__":
    main()
