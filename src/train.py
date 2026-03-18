from typing import Any, Callable, Dict, Optional, Protocol, Tuple, cast

import lightning as L
import hydra
import rootutils
import torch

from lightning import LightningModule
from omegaconf import DictConfig


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils.entrypoint_utils import (
    instantiate_lightning_task_objects,
    instantiate_task_runner,
    require_run_target_config,
)
from src.utils.entrypoint_contracts import validate_train_entry_contract
from src.utils.fit_schedule import (
    ResolvedPassFitSchedule,
    apply_resolved_pass_fit_schedule,
    resolve_pass_fit_schedule,
)
from src.utils.hydra_utils import apply_run_name, extras
from src.utils.logging_utils import RankedLogger
from src.utils.task_utils import get_metric_value, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


TrainModelFn = Callable[[DictConfig], Tuple[Dict[str, Any], Dict[str, Any]]]


class TrainRunnerProtocol(Protocol):
    def validate(self, cfg: DictConfig) -> None: ...

    def run(
        self,
        *,
        cfg: DictConfig,
        train_model: TrainModelFn,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...


def _maybe_load_model_weights(model: LightningModule, cfg: DictConfig) -> None:
    init_ckpt_path = cfg.get("init_ckpt_path")
    if init_ckpt_path in (None, ""):
        return
    checkpoint = torch.load(str(init_ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "init_ckpt_path must point to a checkpoint containing a `state_dict`."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    log.info(
        "Loaded initial model weights from checkpoint: %s (missing=%d, unexpected=%d)",
        init_ckpt_path,
        len(missing),
        len(unexpected),
    )
    if missing:
        log.warning("Missing keys when loading init_ckpt_path: %s", missing)
    if unexpected:
        log.warning("Unexpected keys when loading init_ckpt_path: %s", unexpected)


def _configure_pass_fit_schedule(
    cfg: DictConfig,
    *,
    datamodule: Any,
) -> ResolvedPassFitSchedule:
    dataset_cfg = cfg.get("dataset") or {}
    if dataset_cfg.get("val_check_interval") not in (None, ""):
        raise ValueError(
            "dataset.val_check_interval has been removed. "
            "Use `fit_schedule.val_every_passes` instead so validation cadence scales with train-set size."
        )

    datamodule.setup(stage="fit")
    train_dataset = getattr(datamodule, "train_dataset", None)
    if train_dataset is None:
        raise RuntimeError(
            "Datamodule did not materialize `train_dataset` during setup('fit'); "
            "cannot derive pass-based schedule."
        )
    per_device_batch_size = int(
        getattr(
            datamodule, "batch_size_per_device", getattr(datamodule, "batch_size", 0)
        )
    )
    resolved = resolve_pass_fit_schedule(
        fit_schedule_cfg=cfg.get("fit_schedule"),
        trainer_cfg=cfg.trainer,
        train_size=len(train_dataset),
        per_device_batch_size=per_device_batch_size,
    )
    apply_resolved_pass_fit_schedule(cfg, resolved)
    log.info(
        "Resolved pass-based fit schedule: train_size=%d max_steps=%d val_check_interval=%d "
        "patience_checks=%d max_passes=%.2f val_every_passes=%.2f",
        resolved.train_size,
        resolved.max_steps,
        resolved.val_check_interval_batches,
        resolved.early_stopping_patience_checks,
        resolved.max_passes,
        resolved.val_every_passes,
    )
    return resolved


@task_wrapper
def train_model(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    resolved_run_name = apply_run_name(cfg)
    log.info(f"Resolved run name: {resolved_run_name}")

    run_cfg = cfg.get("run") or {}
    resolved_fit_schedule: ResolvedPassFitSchedule | None = None

    def _on_datamodule_instantiated(datamodule: Any) -> None:
        nonlocal resolved_fit_schedule
        if bool(run_cfg.get("train", True)):
            resolved_fit_schedule = _configure_pass_fit_schedule(
                cfg,
                datamodule=datamodule,
            )

    def _on_model_instantiated(model: Any) -> None:
        _maybe_load_model_weights(model=model, cfg=cfg)
        if resolved_fit_schedule is None:
            return
        setter = getattr(model, "set_fit_schedule", None)
        if callable(setter):
            setter(resolved_fit_schedule)

    objects = instantiate_lightning_task_objects(
        cfg,
        log=log,
        on_datamodule_instantiated=_on_datamodule_instantiated,
        on_model_instantiated=_on_model_instantiated,
    )
    datamodule = objects.datamodule
    model = cast(LightningModule, objects.model)
    trainer = objects.trainer
    object_dict = objects.as_dict()

    if bool(run_cfg.get("train", True)):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if bool(run_cfg.get("test", False)):
        log.info("Starting testing!")
        test_ckpt_path: Optional[str] = run_cfg.get("test_ckpt_path")
        if test_ckpt_path not in (None, ""):
            ckpt_path = test_ckpt_path
        else:
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback is None:
                raise RuntimeError(
                    "Testing requested but no checkpoint callback is configured. "
                    "Provide `test_ckpt_path` or enable a checkpoint callback."
                )
            ckpt_path = checkpoint_callback.best_model_path
            if ckpt_path == "":
                if bool(run_cfg.get("allow_test_without_checkpoint", False)):
                    log.warning(
                        "Best ckpt not found! Using current weights for testing..."
                    )
                    ckpt_path = None
                else:
                    raise RuntimeError(
                        "Best checkpoint path is empty. Set `allow_test_without_checkpoint=True` "
                        "or provide `test_ckpt_path` to proceed explicitly."
                    )
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    require_run_target_config(
        cfg,
        missing_run_message=(
            "Missing required config group: `run`. "
            "Fix: use a train config that sets `/run`, for example `experiment=train_rankflow`."
        ),
        missing_target_message=(
            "Missing required run target: `run._target_`. "
            "Fix: use a concrete run config such as `run=train_rankflow`."
        ),
    )
    validate_train_entry_contract(cfg)
    extras(cfg)
    runner = cast(
        TrainRunnerProtocol,
        instantiate_task_runner(cfg.run, run_signature="run(cfg=..., train_model=...)"),
    )
    runner.validate(cfg)

    metric_dict, _ = runner.run(cfg=cfg, train_model=train_model)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    return get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
