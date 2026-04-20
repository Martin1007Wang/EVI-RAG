from __future__ import annotations

import argparse
import os

import hydra
import lightning as L
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single-sample RankFlow debug run and print periodic root-gold-rank snapshots."
        )
    )
    parser.add_argument(
        "--experiment",
        default="train_rankflow_single_sample_debug",
        help="Hydra experiment config to compose.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Number of optimizer steps to run.",
    )
    parser.add_argument(
        "--default-root-dir",
        default=os.path.abspath("logs/tmp_root_gold_rank_probe"),
        help="Explicit trainer.default_root_dir to avoid Hydra runtime interpolation.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable Lightning progress bar.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. trainer=gpu model.num_rollout=4.",
    )
    return parser.parse_args()


def _build_cfg(args: argparse.Namespace) -> DictConfig:
    GlobalHydra.instance().clear()
    overrides = [
        f"experiment={args.experiment}",
        f"trainer.max_steps={args.max_steps}",
        "trainer.limit_val_batches=0",
        "trainer.limit_test_batches=0",
        "+test=false",
        *args.overrides,
    ]
    with initialize_config_dir(
        config_dir=os.path.abspath("configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    with open_dict(cfg):
        cfg.trainer.default_root_dir = args.default_root_dir
    return cfg


def _summarize_rank_history(history: list[dict[str, float]]) -> None:
    print("=== rank_history ===")
    if not history:
        print("<empty>")
        return
    for item in history:
        print(item)

    first = history[0]
    last = history[-1]
    delta_rank = last["gold_rank_mean"] - first["gold_rank_mean"]
    delta_ratio = last["gold_rank_ratio_mean"] - first["gold_rank_ratio_mean"]
    print("=== rank_summary ===")
    print(
        "first_step={first_step:.0f} last_step={last_step:.0f} first_mean_rank={first_rank:.1f} "
        "last_mean_rank={last_rank:.1f} delta_rank={delta_rank:.1f} first_ratio={first_ratio:.4f} "
        "last_ratio={last_ratio:.4f} delta_ratio={delta_ratio:.4f}".format(
            first_step=first["step"],
            last_step=last["step"],
            first_rank=first["gold_rank_mean"],
            last_rank=last["gold_rank_mean"],
            delta_rank=delta_rank,
            first_ratio=first["gold_rank_ratio_mean"],
            last_ratio=last["gold_rank_ratio_mean"],
            delta_ratio=delta_ratio,
        )
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg(args)

    L.seed_everything(cfg.get("seed", 42), workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=None,
        enable_progress_bar=args.progress_bar,
        enable_model_summary=False,
    )
    trainer.fit(model=model, datamodule=datamodule)
    rank_history = getattr(model, "root_gold_rank_history", [])
    _summarize_rank_history(rank_history)


if __name__ == "__main__":
    main()
