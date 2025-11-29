from __future__ import annotations

from typing import Any, Dict

from lightning import Callback, Trainer
from lightning.pytorch.core.module import LightningModule


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
        reward = debug.get("reward")
        success = debug.get("success")
        gt_exists = debug.get("gt_path_exists")
        path_f1 = debug.get("gt_path_f1")
        reach = debug.get("answer_reach_frac")
        if reward is None or success is None:
            return
        # Only inspect samples that reached answers (success=True).
        mask = success.bool()
        if mask.numel() == 0 or not mask.any():
            return
        reward_hit = reward[mask]
        reach_hit = reach[mask] if reach is not None else None
        path_hit = path_f1[mask] if path_f1 is not None else None
        gt_hit = gt_exists[mask] if gt_exists is not None else None
        pl_module.log(
            "debug/reward_success_mean",
            reward_hit.mean(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=mask.numel(),
        )
        if reach_hit is not None and reach_hit.numel() > 0:
            pl_module.log(
                "debug/answer_reach_success_mean",
                reach_hit.mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=mask.numel(),
            )
        if path_hit is not None and path_hit.numel() > 0:
            pl_module.log(
                "debug/path_f1_success_mean",
                path_hit.mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=mask.numel(),
            )
        if gt_hit is not None and gt_hit.numel() > 0:
            pl_module.log(
                "debug/gt_path_exists_success_mean",
                gt_hit.float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=mask.numel(),
            )


__all__ = ["RewardProbeCallback"]
