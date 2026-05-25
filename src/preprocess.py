from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.runtime import load_project_env
from src.utils.logging_utils import get_logger

PROJECT_ROOT = load_project_env(__file__)

log = get_logger(__name__)


def maybe_print_config(cfg: DictConfig) -> None:
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="preprocess",
)
def main(cfg: DictConfig) -> None:
    # maybe_print_config(cfg)

    if "dataset" not in cfg or cfg.dataset is None:
        raise ValueError(
            "Missing required config group: dataset. "
            "Example: preprocess_command experiment=preprocess/webqsp"
        )

    if "preprocess" not in cfg or cfg.preprocess is None:
        raise ValueError("Missing required config group: preprocess.")

    dataset_name = cfg.dataset.get("name", "unknown")
    log.info(
        "Starting preprocess pipeline: "
        f"dataset={dataset_name} "
        "(scan -> encode globals -> stream materialize)"
    )

    from src.data.preprocess import run_preprocess_pipeline

    result = run_preprocess_pipeline(cfg)
    log.info(
        "Preprocess summary: "
        f"dataset={result.dataset_name} "
        f"samples={result.num_samples} "
        f"entities={result.num_entities} "
        f"relations={result.num_relations} "
        f"splits={result.split_counts}"
    )


if __name__ == "__main__":
    main()
