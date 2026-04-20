from __future__ import annotations
import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _require_dataset_config(cfg: DictConfig) -> None:
    if "dataset" not in cfg or cfg.dataset is None:
        raise ValueError(
            "Missing required config group: `dataset`. "
            "Fix: run `python src/preprocess.py dataset=<name>` such as `dataset=webqsp`."
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig) -> None:
    _require_dataset_config(cfg)
    dataset_name = cfg.dataset.get("name", "Unknown")
    stage = cfg.get("pipeline_stage", "all")
    log.info(f"Starting preprocess pipeline: dataset={dataset_name} stage={stage}")
    from src.data.preprocess import run_preprocess_pipeline

    run_preprocess_pipeline(cfg)


if __name__ == "__main__":
    main()
