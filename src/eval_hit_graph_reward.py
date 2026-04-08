from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.collate import RetrievalCollator
from src.eval.hit_graph_reward import (
    evaluate_hit_graph_reward,
    summarize_hit_graph_rewards,
)
from src.models.reward import RewardModel
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _select_dataset(datamodule: Any, split_name: str) -> Any:
    split_name = str(split_name)
    split_to_dataset = {
        str(
            datamodule.dataset_cfg.get("train_split", "train")
        ): datamodule.train_dataset,
        str(
            datamodule.dataset_cfg.get("val_split", "validation")
        ): datamodule.val_dataset,
        str(datamodule.dataset_cfg.get("eval_split", "test")): datamodule.test_dataset,
        str(
            datamodule.dataset_cfg.get(
                "predict_split", datamodule.dataset_cfg.get("eval_split", "test")
            )
        ): datamodule.predict_dataset,
    }
    dataset = split_to_dataset.get(split_name)
    if dataset is None:
        available = sorted(
            name for name, value in split_to_dataset.items() if value is not None
        )
        raise ValueError(
            f"Split {split_name!r} is unavailable. Available loaded splits: {available}."
        )
    return dataset


def _sample_dataset_indices(
    dataset_size: int, *, max_graphs: int | None, seed: int
) -> list[int]:
    all_indices = list(range(dataset_size))
    if max_graphs is None or max_graphs <= 0 or max_graphs >= dataset_size:
        return all_indices
    rng = random.Random(seed)
    return sorted(rng.sample(all_indices, k=max_graphs))


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage=None)

    if datamodule._shared_resources is None:
        raise RuntimeError("Data resources were not initialized by datamodule.setup().")

    dataset = _select_dataset(datamodule, str(cfg.split))
    collator = RetrievalCollator(datamodule._shared_resources)
    reward_model = RewardModel(
        **OmegaConf.to_container(cfg.answer_reward, resolve=True)
    )
    chosen_indices = _sample_dataset_indices(
        len(dataset), max_graphs=cfg.max_graphs, seed=seed
    )

    log.info(
        "Estimating hit-graph log reward on split=%s with %d graphs.",
        cfg.split,
        len(chosen_indices),
    )

    results = []
    try:
        for dataset_idx in chosen_indices:
            batch = collator([dataset[dataset_idx]])
            results.append(
                evaluate_hit_graph_reward(
                    batch,
                    reward_model=reward_model,
                    path_mode=str(cfg.path_mode),
                    stop_on_first_hit=bool(cfg.stop_on_first_hit),
                )
            )
    finally:
        datamodule.teardown(stage=None)

    summary = summarize_hit_graph_rewards(results)
    summary.update(
        {
            "split": str(cfg.split),
            "seed": seed,
            "path_mode": str(cfg.path_mode),
            "stop_on_first_hit": bool(cfg.stop_on_first_hit),
            "requested_max_graphs": None
            if cfg.max_graphs is None
            else int(cfg.max_graphs),
        }
    )
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="eval_hit_graph_reward.yaml",
)
def main(cfg: DictConfig) -> None:
    metrics = run_experiment(cfg)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hit_graph_reward_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Hit-graph reward metrics written to %s", output_path)
    log.info("Summary: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
