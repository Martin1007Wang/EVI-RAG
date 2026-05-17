from __future__ import annotations

from typing import Any

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from src.runtime import load_project_env
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_datamodule, build_model, build_trainer, setup_datamodule

PROJECT_ROOT = load_project_env(__file__)


def required_ckpt_path(cfg: DictConfig) -> str:
    value = cfg.get("ckpt_path", None)
    if value in (None, ""):
        raise ValueError(
            "ckpt_path must be provided for evaluation. "
            "Example: evaluate_command experiment=eval/webqsp "
            "ckpt_path=/path/to/model.ckpt"
        )
    return str(value)


def maybe_print_config(cfg: DictConfig) -> None:
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))


def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}

    for key, value in metrics.items():
        if not isinstance(key, str):
            continue

        if hasattr(value, "item"):
            scalars[key] = float(value.item())
        elif isinstance(value, (int, float)):
            scalars[key] = float(value)

    return scalars


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="evaluate",
)
def main(cfg: DictConfig) -> None:
    print(f"Starting evaluation run: {cfg.get('task_name', 'evaluate')}")

    maybe_print_config(cfg)

    run_validate = bool(cfg.get("validate", True))
    run_test = bool(cfg.get("test", True))

    if not run_validate and not run_test:
        raise ValueError("At least one of validate or test must be true for evaluation.")

    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed), workers=True)

    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(
        datamodule,
        stage="validate" if run_validate else "test",
    )

    model = build_model(cfg, resources)

    ckpt_path = required_ckpt_path(cfg)
    missing, unexpected = load_checkpoint_weights(model, ckpt_path, strict=False)
    print(f"Loaded evaluation checkpoint from {ckpt_path!r}; " f"missing={missing}, unexpected={unexpected}")

    trainer = build_trainer(cfg)
    summaries: dict[str, float] = {}

    if run_validate:
        trainer.validate(model=model, datamodule=datamodule)
        summaries.update(scalarize_metrics(dict(trainer.callback_metrics)))

    if run_test:
        trainer.test(model=model, datamodule=datamodule)
        summaries.update(scalarize_metrics(dict(trainer.callback_metrics)))

    for key, value in sorted(summaries.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
