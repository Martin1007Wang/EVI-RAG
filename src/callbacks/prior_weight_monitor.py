from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback

from src.utils.logging_utils import log_metric

if TYPE_CHECKING:
    from src.weaver.nn.edge_scorer import ExpandEdgeScorer


class PriorWeightMonitor(Callback):
    """Log semantic-prior fusion weights without coupling monitoring to model code."""

    PRIOR_NAMES = ("relation",)

    def __init__(
        self,
        *,
        log_every_n_steps: int = 100,
        metric_prefix: str = "prior_w",
    ) -> None:
        super().__init__()
        if log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be >= 1, got {log_every_n_steps}."
            )
        self.log_every_n_steps = int(log_every_n_steps)
        self.metric_prefix = metric_prefix.rstrip("/")

    @classmethod
    @torch.no_grad()
    def get_weights(cls, scorer: Any) -> dict[str, float]:
        prior_scale = getattr(scorer, "prior_scale", None)
        if not isinstance(prior_scale, torch.Tensor) or prior_scale.shape != ():
            raise ValueError(
                "prior_scale must be a scalar tensor, "
                f"got {None if prior_scale is None else tuple(prior_scale.shape)}."
            )
        return {f"prior_w/{cls.PRIOR_NAMES[0]}": prior_scale.detach().item()}

    @classmethod
    @torch.no_grad()
    def get_effective_contribution(
        cls,
        scorer: Any,
        prior_bank: torch.Tensor,
    ) -> dict[str, float]:
        if prior_bank.ndim != 2 or prior_bank.shape[1] != len(cls.PRIOR_NAMES):
            raise ValueError(
                "prior_bank must have shape "
                f"(E, {len(cls.PRIOR_NAMES)}), got {tuple(prior_bank.shape)}."
            )
        prior_scale = getattr(scorer, "prior_scale", None)
        if not isinstance(prior_scale, torch.Tensor) or prior_scale.shape != ():
            raise ValueError(
                "prior_scale must be a scalar tensor, "
                f"got {None if prior_scale is None else tuple(prior_scale.shape)}."
            )
        scale = prior_scale.detach().to(
            device=prior_bank.device,
            dtype=prior_bank.dtype,
        )
        contributions = (prior_bank * scale).abs().mean(dim=0)
        return {
            f"prior_contrib/{name}": contributions[i].item()
            for i, name in enumerate(cls.PRIOR_NAMES)
        }

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        step = int(getattr(trainer, "global_step", 0))
        if step < 1 or step % self.log_every_n_steps != 0:
            return
        scorer = self._resolve_scorer(pl_module)
        for name, value in self.get_weights(scorer).items():
            log_metric(
                pl_module,
                self._prefix_metric_name(name),
                value,
                batch_size=1,
                on_step=True,
                on_epoch=False,
            )

    def _prefix_metric_name(self, name: str) -> str:
        if self.metric_prefix == "":
            return name
        suffix = name.split("/", 1)[-1]
        return f"{self.metric_prefix}/{suffix}"

    @staticmethod
    def _resolve_scorer(pl_module: Any) -> Any:
        policy = getattr(pl_module, "policy", None)
        scorer = getattr(policy, "expand_edge_scorer", None)
        prior_scale = getattr(scorer, "prior_scale", None)
        if not isinstance(prior_scale, torch.Tensor):
            raise TypeError(
                "PriorWeightMonitor expects pl_module.policy.expand_edge_scorer "
                "to expose a scalar prior_scale tensor. "
                f"Got {type(scorer).__name__}."
            )
        return scorer


__all__ = ["PriorWeightMonitor"]
