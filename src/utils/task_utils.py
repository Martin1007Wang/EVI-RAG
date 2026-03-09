from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Callable, Optional, TYPE_CHECKING

from .logging_utils import RankedLogger

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig

log = RankedLogger(__name__, rank_zero_only=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function."""

    def wrap(cfg: "DictConfig") -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    from lightning_utilities.core.rank_zero import rank_zero_only

                    @rank_zero_only
                    def _finish() -> None:
                        log.info("Closing wandb!")
                        wandb.finish()

                    _finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].detach().tolist()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


__all__ = ["task_wrapper", "get_metric_value"]
