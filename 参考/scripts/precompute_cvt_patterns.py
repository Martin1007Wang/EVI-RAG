#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.stages.sidecars.cvt_patterns import precompute_cvt_patterns
from src.utils.logging_utils import init_logging, log_event, get_logger

LOGGER = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute CVT relation patterns.")
    parser.add_argument("--graphs-path", type=str, required=True, help="Path to normalized graphs.parquet.")
    parser.add_argument("--entity-vocab-path", type=str, required=True, help="Path to entity_vocab.parquet.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for CVT pattern files.")
    parser.add_argument("--batch-size", type=int, default=256, help="Graphs per batch.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Log progress every N graphs (0 disables).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output_dir.")
    parser.add_argument("--log-path", type=str, default=None, help="Optional log file path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    init_logging(log_path=log_path)
    log_event(
        LOGGER,
        "cvt_patterns_cli_start",
        graphs_path=args.graphs_path,
        entity_vocab_path=args.entity_vocab_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        overwrite=args.overwrite,
    )
    precompute_cvt_patterns(
        graphs_path=Path(args.graphs_path),
        entity_vocab_path=Path(args.entity_vocab_path),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        overwrite=args.overwrite,
    )
    log_event(LOGGER, "cvt_patterns_cli_done", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
