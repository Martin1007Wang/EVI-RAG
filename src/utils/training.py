"""Training utilities following PyTorch Lightning best practices.

This module contains essential training utilities without unnecessary abstractions.
"""

from __future__ import annotations

import warnings
from typing import Any
import os
import torch
from omegaconf import DictConfig, OmegaConf


def get_simple_run_name(cfg: DictConfig) -> str:
    """Get a simple run name for logging.

    Args:
        cfg: Hydra configuration dict

    Returns:
        Run name string
    """
    # Try to get from Hydra first
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        if hasattr(hydra_cfg, "job") and hasattr(hydra_cfg.job, "name"):
            job_name = hydra_cfg.job.name
            if job_name and job_name != "<unnamed>":
                return job_name
    except:
        pass

    # Fallback: use experiment and dataset
    experiment = cfg.get("experiment", "train")
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "") if isinstance(dataset_cfg, dict) else ""

    if dataset_name:
        # Clean up dataset name (remove -sub suffix for brevity)
        dataset_name = dataset_name.replace("-sub", "")
        return f"{experiment}_{dataset_name}"

    return experiment


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load weights from checkpoint into model.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly match state dict keys

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    return missing, unexpected


def print_config_summary(cfg: DictConfig, resolve: bool = True) -> None:
    print("=" * 80)
    print("CONFIG SUMMARY")
    print("=" * 80)
    key_groups = ["data", "model", "trainer", "dataset", "experiment"]
    for group in key_groups:
        if group in cfg:
            group_cfg = cfg[group]
            if resolve:
                group_cfg = OmegaConf.to_container(group_cfg, resolve=True)
            print(f"\n[{group.upper()}]")
            if isinstance(group_cfg, dict):
                for key, value in list(group_cfg.items())[:5]:  # First 5 items
                    print(f"  {key}: {value}")
                if len(group_cfg) > 5:
                    print(f"  ... and {len(group_cfg) - 5} more items")
            else:
                print(f"  {group_cfg}")
    print("=" * 80)


__all__ = [
    "get_simple_run_name",
    "load_model_weights",
    "print_config_summary",
]
