#!/usr/bin/env python3
"""Thin CLI wrapper for the retrieval preprocessing pipeline."""

from __future__ import annotations

import os

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]

from src.data.pipeline_main import build_pipeline
from src.utils.logging_utils import get_logger, init_logging, log_event

LOGGER = get_logger(__name__)


if hydra is not None:

    @hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_pipeline")
    def main(cfg):
        log_path_cfg = cfg.get("pipeline_log_path")
        log_path = None
        if log_path_cfg:
            from pathlib import Path

            log_path = Path(hydra.utils.to_absolute_path(str(log_path_cfg)))
        init_logging(log_path=log_path)
        log_event(
            LOGGER,
            "pipeline_start",
            dataset=str(cfg.get("dataset_name") or cfg.get("dataset") or "dataset"),
            out_dir=str(cfg.get("out_dir")),
            output_dir=str(cfg.get("output_dir")),
            parquet_dir=str(cfg.get("parquet_dir")),
        )
        build_pipeline(cfg)
        log_event(LOGGER, "pipeline_done")

else:  # pragma: no cover

    def main(cfg):
        raise ModuleNotFoundError("hydra-core is required to run scripts/build_retrieval_pipeline.py")


if __name__ == "__main__":
    main()
