"""Simplified evaluation script following PyTorch Lightning best practices."""

from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.training import (
    setup_training_extras,
    setup_hf_cache,
    print_config_summary,
)
from src.utils.logging_utils import get_logger
from src.utils.run_name import set_run_name_in_config

log = get_logger(__name__)


def evaluate_model(cfg: DictConfig) -> dict[str, Any]:
    """Core evaluation logic - simplified and direct."""

    # 1. Basic setup
    L.seed_everything(cfg.get("seed", 42), workers=True)
    setup_training_extras(cfg)
    setup_hf_cache(cfg)

    # 2. Print config summary if requested
    if cfg.get("extras", {}).get("print_config", False):
        print_config_summary(cfg, resolve=True)

    # 3. Generate and set run name
    run_name = set_run_name_in_config(cfg)
    log.info("Starting evaluation run: %s", run_name)

    # 3. Check for required checkpoint
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Checkpoint path is required for evaluation.")

    # 4. Instantiate datamodule
    log.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data)

    # 5. Load model from checkpoint
    log.info("Loading model from checkpoint: %s", ckpt_path)

    # Try to load using Lightning's load_from_checkpoint first
    model_target = cfg.model.get("_target_", "")
    if model_target:
        try:
            # Extract class from target string
            module_name, class_name = model_target.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            # Load model with weights
            model = model_class.load_from_checkpoint(
                ckpt_path,
                **OmegaConf.to_container(cfg.model, resolve=True),
            )
        except Exception as e:
            log.warning("Failed to load with load_from_checkpoint: %s", e)
            log.info("Falling back to direct instantiation...")
            model = hydra.utils.instantiate(cfg.model)

            # Load weights manually
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
    else:
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    # 6. Instantiate callbacks and logger
    callbacks = []
    if cfg.get("callbacks"):
        callbacks = hydra.utils.instantiate(cfg.callbacks)
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

    experiment_logger = None
    if cfg.get("logger"):
        experiment_logger = hydra.utils.instantiate(cfg.logger)

    # 7. Create trainer for evaluation
    log.info("Creating trainer for evaluation...")
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Override trainer config for evaluation
    trainer_cfg.update(
        {
            "max_epochs": 1,  # Single evaluation pass
            "limit_val_batches": 1.0,  # Evaluate all validation data
            "limit_test_batches": 1.0,  # Evaluate all test data
        }
    )

    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=experiment_logger,
    )

    # 8. Run evaluation
    log.info("Starting evaluation...")

    # Determine which splits to evaluate
    eval_splits = []
    if cfg.get("validate", True):
        eval_splits.append("val")
    if cfg.get("test", True):
        eval_splits.append("test")

    metrics = {}
    for split in eval_splits:
        log.info("Evaluating %s split...", split)

        if split == "val":
            trainer.validate(model=model, datamodule=datamodule)
        elif split == "test":
            trainer.test(model=model, datamodule=datamodule)

        # Collect metrics
        split_metrics = dict(trainer.callback_metrics)
        for key, value in split_metrics.items():
            metrics[f"{split}/{key}"] = value

    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""

    # Validate required config
    if cfg.get("run") is None:
        raise ValueError("Missing required config group: `run`.")

    # Run evaluation
    metrics = evaluate_model(cfg)

    # Print summary
    log.info("%s", "=" * 80)
    log.info("EVALUATION SUMMARY")
    log.info("%s", "=" * 80)
    for key, value in sorted(metrics.items()):
        log.info("%s: %.4f", key, value)
    log.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
