from __future__ import annotations

from typing import Any, Dict

import torch

from lightning import Callback, Trainer
from lightning.pytorch.core.module import LightningModule

from src.utils.logging_utils import infer_batch_size, log_metric


class RewardProbeCallback(Callback):
    """Debug callback to inspect reward magnitude for successful trajectories."""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        debug = getattr(pl_module, "_last_debug", None)
        if not debug:
            return
        log_reward = debug.get("log_reward")
        answer_hit = debug.get("answer_hit")
        prefix_ratio = debug.get("path_prefix_ratio")
        full_hit = debug.get("path_full_hit")
        gt_exists = debug.get("gt_path_exists_ratio")
        if log_reward is None or answer_hit is None:
            return
        # Only inspect samples that reached answers (success=True).
        mask = answer_hit.bool()
        if mask.numel() == 0 or not mask.any():
            return
        log_reward_hit = log_reward[mask]
        prefix_hit = prefix_ratio[mask] if prefix_ratio is not None else None
        full_hit_mask = full_hit[mask] if full_hit is not None else None
        gt_hit = gt_exists[mask] if isinstance(gt_exists, torch.Tensor) and gt_exists.numel() == mask.numel() else None
        batch_size = infer_batch_size(batch)
        log_metric(
            pl_module,
            "debug/log_reward_success_mean",
            log_reward_hit.mean(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        if prefix_hit is not None and prefix_hit.numel() > 0:
            log_metric(
                pl_module,
                "debug/path_prefix_ratio_success_mean",
                prefix_hit.mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        if full_hit_mask is not None and full_hit_mask.numel() > 0:
            log_metric(
                pl_module,
                "debug/path_full_hit_success_mean",
                full_hit_mask.float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        if gt_hit is not None and gt_hit.numel() > 0:
            log_metric(
                pl_module,
                "debug/gt_path_exists_success_mean",
                gt_hit.float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )


__all__ = ["RewardProbeCallback"]
