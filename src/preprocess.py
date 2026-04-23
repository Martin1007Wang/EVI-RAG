from __future__ import annotations
import hydra
import rootutils
from omegaconf import DictConfig
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging_utils import get_logger
log = get_logger(__name__)
@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig) -> None:
    if "dataset" not in cfg or cfg.dataset is None:
        raise ValueError(
            "Missing required config group: `dataset`. "
            "Fix: run `python src/preprocess.py dataset=<name>` such as `dataset=webqsp`."
        )
    dataset_name = cfg.dataset.get("name", "Unknown")
    log.info(
        "Starting full preprocess pipeline: "
        f"dataset={dataset_name} "
        "(graph_collect -> text_encode -> materialize)"
    )
    from src.data.preprocess import run_preprocess_pipeline
    run_preprocess_pipeline(cfg)
if __name__ == "__main__":
    main()