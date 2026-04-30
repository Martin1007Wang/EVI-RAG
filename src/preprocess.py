from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import rootutils
except ModuleNotFoundError:
    rootutils = None
else:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)


def maybe_print_config(cfg: DictConfig) -> None:
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))


@hydra.main(version_base=None, config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    maybe_print_config(cfg)

    if "dataset" not in cfg or cfg.dataset is None:
        raise ValueError("Missing required config group: dataset. " "Example: python src/preprocess.py dataset=webqsp")

    if "preprocess" not in cfg or cfg.preprocess is None:
        raise ValueError("Missing required config group: preprocess.")

    dataset_name = cfg.dataset.get("name", "unknown")
    log.info("Starting preprocess pipeline: " f"dataset={dataset_name} " "(graph_collect -> text_encode -> materialize)")

    from src.data.preprocess import run_preprocess_pipeline

    run_preprocess_pipeline(cfg)


if __name__ == "__main__":
    main()
