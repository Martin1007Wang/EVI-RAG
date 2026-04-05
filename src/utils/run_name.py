"""Run name generation system.

This module provides a standardized way to generate run names for experiments.
Run names follow the format: {task_type}_{model}_{dataset}_{variant}

Design principles:
1. Human readable and informative
2. Filesystem friendly (no special characters)
3. Consistent across all experiments
4. Configurable through Hydra
"""

from __future__ import annotations

import re
from typing import Any

from omegaconf import DictConfig, open_dict


def clean_for_filesystem(name: str) -> str:
    """Clean string to be filesystem friendly.

    Args:
        name: Input string

    Returns:
        Cleaned string with only alphanumeric, dash, and underscore
    """
    # Replace spaces and special characters with underscore
    name = re.sub(r"[^\w\-]", "_", name)
    # Replace multiple underscores with single
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name


def extract_task_type(cfg: DictConfig) -> str:
    """Extract task type from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        'train' or 'eval'
    """
    run_name = str(cfg.get("run", {}).get("name", "")).lower()

    # Check if it's a training run
    if run_name.startswith("train"):
        return "train"

    # Check experiment name
    experiment = str(cfg.get("experiment", "")).lower()
    if experiment.startswith("train"):
        return "train"

    # Default to eval
    return "eval"


def extract_model_name(cfg: DictConfig) -> str:
    """Extract model name from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Model name (e.g., 'rankflow', 'gflownet', 'llm')
    """
    # Try to get from experiment first
    experiment = str(cfg.get("experiment", "")).lower()

    # Extract model from experiment name
    if experiment:
        # Remove 'train_' or 'eval_' prefix
        model = re.sub(r"^(train_|eval_)", "", experiment)
        if model and model != "null":
            return model

    # Try to get from model config
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        target = str(model_cfg.get("_target_", "")).lower()
        if target:
            # Extract class name from target
            parts = target.split(".")
            if parts:
                class_name = parts[-1].lower()
                # Remove common suffixes
                class_name = re.sub(r"(module|model|net)$", "", class_name)
                if class_name:
                    return class_name

    # Fallback
    return "unknown"


def extract_dataset_name(cfg: DictConfig) -> str:
    """Extract dataset name from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Dataset name (e.g., 'webqsp', 'webqsp-sub', 'cwq')
    """
    dataset_cfg = cfg.get("dataset")

    if dataset_cfg is None:
        return "unknown"

    # Handle DictConfig
    if hasattr(dataset_cfg, "get"):
        name = dataset_cfg.get("name")
        if name:
            return str(name)

    # Handle string
    if isinstance(dataset_cfg, str):
        return dataset_cfg

    # Try to get from Hydra runtime choices
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        runtime = getattr(hydra_cfg, "runtime", None)
        if runtime is not None:
            dataset_choice = runtime.choices.get("dataset")
            if dataset_choice:
                return str(dataset_choice)
    except:
        pass

    return "unknown"


def extract_variant(cfg: DictConfig) -> str:
    """Extract experiment variant from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Variant name (e.g., 'baseline', 'ablation-attention', 'v2')
    """
    # First try to get from run config
    run_cfg = cfg.get("run", {})
    variant = run_cfg.get("variant")
    if variant:
        return str(variant)

    # Check tags for variant hints
    tags = run_cfg.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            tag_str = str(tag).lower()
            if tag_str.startswith("ablation"):
                return "ablation"
            elif tag_str.startswith("v") and tag_str[1:].isdigit():
                return tag_str
            elif tag_str in ["dev", "debug", "test"]:
                return tag_str

    # Default variant
    return "baseline"


def generate_run_name(cfg: DictConfig) -> str:
    """Generate standardized run name.

    Format: {task_type}_{model}_{dataset}_{variant}

    Args:
        cfg: Hydra configuration

    Returns:
        Standardized run name
    """
    # Extract components
    task_type = extract_task_type(cfg)
    model = extract_model_name(cfg)
    dataset = extract_dataset_name(cfg)
    variant = extract_variant(cfg)

    # Build run name
    run_name = f"{task_type}_{model}_{dataset}_{variant}"

    # Clean for filesystem
    run_name = clean_for_filesystem(run_name)

    return run_name


def set_run_name_in_config(cfg: DictConfig) -> str:
    run_name = f"{cfg.experiment.name}-{cfg.dataset.name}-{cfg.experiment.variant or 'default'}"
    with open_dict(cfg):
        cfg.run_name = run_name
    return run_name


def parse_run_name(run_name: str) -> dict[str, str]:
    """Parse a run name into its components.

    Args:
        run_name: Run name string

    Returns:
        Dictionary with components: task_type, model, dataset, variant
    """
    parts = run_name.split("_")

    if len(parts) < 4:
        # Not enough parts, return what we have
        return {
            "task_type": parts[0] if len(parts) > 0 else "",
            "model": parts[1] if len(parts) > 1 else "",
            "dataset": parts[2] if len(parts) > 2 else "",
            "variant": parts[3] if len(parts) > 3 else "",
        }

    # Standard format: task_type_model_dataset_variant
    # But dataset name might contain underscores, so we need to handle that
    task_type = parts[0]
    model = parts[1]

    # Dataset is everything between model and last part (variant)
    dataset_parts = parts[2:-1]
    dataset = "_".join(dataset_parts)

    variant = parts[-1]

    return {
        "task_type": task_type,
        "model": model,
        "dataset": dataset,
        "variant": variant,
    }


__all__ = [
    "generate_run_name",
    "set_run_name_in_config",
    "parse_run_name",
    "clean_for_filesystem",
]
