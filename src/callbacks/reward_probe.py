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
        answer_reach_frac = debug.get("answer_reach_frac")
        if log_reward is None or answer_hit is None:
            return
        # Only inspect samples that reached answers (success=True).
        mask = answer_hit.bool()
        if mask.numel() == 0 or not mask.any():
            return
        log_reward_hit = log_reward[mask]
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
        if isinstance(answer_reach_frac, torch.Tensor) and answer_reach_frac.numel() == mask.numel():
            log_metric(
                pl_module,
                "debug/answer_reach_frac_success_mean",
                answer_reach_frac[mask].mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )


__all__ = ["RewardProbeCallback"]
