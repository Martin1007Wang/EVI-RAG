from __future__ import annotations

import sys
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR]

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _require_dataset_config(cfg: DictConfig) -> None:
    if cfg.get("dataset") is not None:
        return
    raise ValueError(
        "Missing required config group: `dataset`. "
        "Fix: run `python src/preprocess.py dataset=<name>` such as `dataset=webqsp`."
    )


def _get_preprocess_runner():
    from src.data.preprocess.main import run_preprocess_pipeline

    return run_preprocess_pipeline


def _run_preprocess(cfg: DictConfig) -> None:
    _require_dataset_config(cfg)
    dataset_cfg = cfg.get("dataset") or {}
    log.info(
        "Starting preprocess pipeline: dataset=%s stage=%s",
        dataset_cfg.get("name"),
        cfg.get("pipeline_stage", "all"),
    )
    _get_preprocess_runner()(cfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig) -> None:
    _run_preprocess(cfg)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
