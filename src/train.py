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
from lightning import seed_everything  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.training.checkpoint import load_pretrained_if_requested  # noqa: E402
from src.training.factory import build_datamodule, build_model, build_trainer  # noqa: E402
from src.training.resources import setup_datamodule  # noqa: E402


def fit_ckpt_path(cfg: DictConfig) -> str | None:
    value = cfg.get("fit_ckpt_path", None)
    if value in (None, ""):
        return None
    return str(value)


def maybe_print_config(cfg: DictConfig) -> None:
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(f"Starting training run: {cfg.get('task_name', 'train')}")

    maybe_print_config(cfg)

    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed), workers=True)

    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(datamodule)

    model = build_model(cfg, resources)
    load_pretrained_if_requested(cfg, model)

    trainer = build_trainer(cfg)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=fit_ckpt_path(cfg),
    )

    if bool(cfg.get("test_after_fit", False)):
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
        )


if __name__ == "__main__":
    main()
