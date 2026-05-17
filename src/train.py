from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from src.runtime import load_project_env
from src.training.checkpoint import load_pretrained_if_requested
from src.training.factory import (
    build_datamodule,
    build_model,
    build_trainer,
    setup_datamodule,
)


PROJECT_ROOT: Path = load_project_env(__file__)
console = Console()


def print_config(cfg: DictConfig) -> None:
    yaml = OmegaConf.to_yaml(
        cfg,
        resolve=True,
        sort_keys=False,
    )

    console.print(
        Panel(
            Syntax(
                yaml,
                "yaml",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=False,
            ),
            title=f"Config: {cfg.task_name}",
            border_style="cyan",
        )
    )


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    console.print(f"[bold]Starting training run:[/bold] {cfg.task_name}")
    print_config(cfg)

    seed_everything(int(cfg.seed), workers=True)

    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(datamodule)

    model = build_model(cfg, resources)
    load_pretrained_if_requested(cfg, model)

    trainer = build_trainer(cfg)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.fit_ckpt_path,
    )

    if cfg.test_after_fit:
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
        )


if __name__ == "__main__":
    main()
