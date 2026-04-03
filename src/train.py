from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import hydra
import rootutils
import torch

from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf, open_dict


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

from src.metrics.search_eval_utils import (
    format_search_eval_answer_posterior,
    normalize_search_eval_cfg,
    search_eval_answer_posterior_signature,
    search_eval_is_answer_task,
)
from src.runs.common import (
    normalize_dataset_scope,
)
from src.runs.entrypoints import validate_train_entrypoint
from src.runs.fit_schedule import (
    ResolvedPassFitSchedule,
    apply_resolved_pass_fit_schedule,
    resolve_pass_fit_schedule,
)
from src.runs.hydra import apply_run_name, compose_config, extras
from src.runs.lightning import (
    finalize_task,
    instantiate_lightning_task_objects,
    seed_everything_if_needed,
)
from src.runs.rankflow import (
    RANKFLOW_TRAIN_RUN_PREFIX,
    is_rankflow_train_run,
    run_eval as run_rankflow_eval,
    validate_train_config,
)
from src.utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _get_metric_value(
    metric_dict: dict[str, Any], metric_name: str | None
) -> float | None:
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None
    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
    value = metric_dict[metric_name]
    metric_value = value.detach().tolist() if torch.is_tensor(value) else float(value)
    log.info("Retrieved metric value! <%s=%s>", metric_name, metric_value)
    return metric_value


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


def _clone_cfg_node(node: Any) -> Any:
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.create(OmegaConf.to_container(node, resolve=False))
    return OmegaConf.create(node)


def _coerce_search_eval_cfg(eval_cfg: Any) -> dict[str, Any]:
    return normalize_search_eval_cfg(eval_cfg)


def _compose_final_eval_template(
    cfg: DictConfig,
    *,
    ckpt_path: str,
    output_dir: str,
) -> DictConfig:
    run_cfg = cfg.get("run") or {}
    dataset_cfg = cfg.get("dataset") or {}
    dataset_name = str(dataset_cfg.get("name") or "").strip()
    if not dataset_name:
        raise ValueError("Final eval requires `dataset.name` to be populated.")

    final_eval_experiment = str(
        run_cfg.get("final_eval_experiment") or "eval_rankflow"
    ).strip()
    return compose_config(
        config_name="eval.yaml",
        overrides=[
            f"experiment={final_eval_experiment}",
            f"dataset={dataset_name}",
            f"ckpt.gflownet={ckpt_path}",
            f"paths.output_dir={output_dir}",
            "extras.enforce_tags=false",
            "extras.print_config=false",
        ],
    )


def _validate_answer_posterior_alignment(
    cfg: DictConfig,
    *,
    eval_template: DictConfig,
) -> None:
    train_eval_cfg = _coerce_search_eval_cfg(cfg.model.eval_cfg)
    final_eval_cfg = _coerce_search_eval_cfg(eval_template.model.eval_cfg)
    if not search_eval_is_answer_task(train_eval_cfg) or not search_eval_is_answer_task(
        final_eval_cfg
    ):
        return

    train_signature = search_eval_answer_posterior_signature(train_eval_cfg)
    final_signature = search_eval_answer_posterior_signature(final_eval_cfg)
    if train_signature == final_signature:
        return

    final_eval_experiment = (
        str(
            (cfg.get("run") or {}).get("final_eval_experiment") or "eval_rankflow"
        ).strip()
        or "eval_rankflow"
    )
    raise ValueError(
        "Training-time answer-ranking checkpoint selection and post-fit final eval "
        "must use the same answer-posterior estimator. "
        f"Got train eval_cfg={format_search_eval_answer_posterior(train_eval_cfg)} but "
        f"final eval experiment={final_eval_experiment!r} uses "
        f"{format_search_eval_answer_posterior(final_eval_cfg)}. "
        "This would pick checkpoints under one answer posterior and score final "
        "eval under another. Keep the answer-posterior config aligned between "
        "train validation and final eval (the canonical RankFlow pair uses the "
        "same Monte Carlo rollout budget in both places), or disable post-fit "
        "final eval and run the alternate configuration explicitly via "
        "`python src/eval.py ...`."
    )


def _resolve_post_fit_ckpt_path(
    *,
    run_cfg: DictConfig | dict[str, Any],
    trainer: Any,
) -> Optional[str]:
    test_ckpt_path: Optional[str] = run_cfg.get("test_ckpt_path")
    if test_ckpt_path not in (None, ""):
        return test_ckpt_path

    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback is None:
        raise RuntimeError(
            "Testing requested but no checkpoint callback is configured. "
            "Provide `test_ckpt_path` or enable a checkpoint callback."
        )

    ckpt_path = checkpoint_callback.best_model_path
    if ckpt_path != "":
        return ckpt_path

    if bool(run_cfg.get("allow_test_without_checkpoint", False)):
        log.warning("Best ckpt not found! Using current weights for testing...")
        return None

    raise RuntimeError(
        "Best checkpoint path is empty. Set `allow_test_without_checkpoint=True` "
        "or provide `test_ckpt_path` to proceed explicitly."
    )


def _resolve_train_output_dir(cfg: DictConfig) -> Path:
    try:
        return Path(str(cfg.paths.output_dir))
    except Exception:
        paths_cfg = cfg.get("paths") or {}
        root_dir = paths_cfg.get("root_dir") if hasattr(paths_cfg, "get") else None
        if root_dir not in (None, ""):
            return Path(str(root_dir))
        return Path.cwd()


def _build_final_eval_cfg(
    cfg: DictConfig,
    *,
    ckpt_path: str,
) -> DictConfig:
    run_cfg = cfg.get("run") or {}
    final_eval_split = str(run_cfg.get("final_eval_split") or "test").strip() or "test"
    output_subdir = str(run_cfg.get("final_eval_output_subdir") or "final_eval").strip()
    output_dir = str(_resolve_train_output_dir(cfg) / output_subdir)

    eval_template = _compose_final_eval_template(
        cfg,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
    )
    _validate_answer_posterior_alignment(cfg, eval_template=eval_template)

    eval_cfg = _clone_cfg_node(eval_template)
    merged_eval_cfg = OmegaConf.create(
        _coerce_search_eval_cfg(
            OmegaConf.merge(
                _clone_cfg_node(cfg.model.eval_cfg),
                _clone_cfg_node(eval_template.model.eval_cfg),
            )
        )
    )

    model_cfg = _clone_cfg_node(cfg.model)
    with open_dict(model_cfg):
        model_cfg.eval_cfg = merged_eval_cfg

    with open_dict(eval_cfg):
        eval_cfg.seed = cfg.get("seed")
        eval_cfg.ckpt_path = ckpt_path
        eval_cfg.paths = OmegaConf.merge(
            _clone_cfg_node(cfg.paths),
            OmegaConf.create({"output_dir": output_dir}),
        )
        eval_cfg.dataset = _clone_cfg_node(cfg.dataset)
        eval_cfg.data = _clone_cfg_node(cfg.data)
        eval_cfg.model = model_cfg
        eval_cfg.callbacks = OmegaConf.create({})
        eval_cfg.logger = OmegaConf.create({})
        eval_cfg.trainer = _clone_cfg_node(eval_template.trainer)
        eval_cfg.run = _clone_cfg_node(eval_template.run)
        eval_cfg.run.split = final_eval_split
        eval_cfg.run.ckpt_path = ckpt_path

    return eval_cfg


def _release_post_fit_runtime_state(
    *,
    model: LightningModule | None,
    datamodule: Any | None,
) -> None:
    if datamodule is not None:
        teardown = getattr(datamodule, "teardown", None)
        if callable(teardown):
            teardown()


def _run_final_eval_suite(cfg: DictConfig) -> Dict[str, Any]:
    from src.eval import evaluate_model as evaluate_model_fn

    report_profile = str(cfg.model.eval_cfg.get("report_profile") or "")
    answer_posterior_cfg = format_search_eval_answer_posterior(
        _coerce_search_eval_cfg(cfg.model.eval_cfg)
    )

    def _evaluate_once(current_cfg: DictConfig):
        current_run = current_cfg.get("run") or {}
        current_dataset = current_cfg.get("dataset") or {}
        dataset_variant = str(
            current_run.get("dataset_variant")
            or current_dataset.get("name")
            or normalize_dataset_scope(current_dataset)
        )
        split = str(current_run.get("split") or "test")
        log.info(
            "Final evaluation: dataset_variant=%s split=%s report_profile=%s answer_posterior_surrogate=%s",
            dataset_variant,
            split,
            report_profile,
            answer_posterior_cfg,
        )
        return evaluate_model_fn(current_cfg)

    return run_rankflow_eval(
        cfg,
        evaluate_model=_evaluate_once,
        allow_default_dataset_variant=True,
        metric_namespace_prefix="final_eval",
    )


def _run_inprocess_test(
    *,
    trainer: Any,
    model: LightningModule,
    datamodule: Any,
    ckpt_path: Optional[str],
) -> Dict[str, Any]:
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Best ckpt path: %s", ckpt_path)
    return dict(trainer.callback_metrics)


def _run_post_fit_evaluation(
    *,
    cfg: DictConfig,
    trainer: Any,
    model: LightningModule,
    datamodule: Any,
) -> Dict[str, Any]:
    run_cfg = cfg.get("run") or {}
    if not bool(run_cfg.get("test", False)):
        return {}

    log.info("Starting post-fit evaluation!")
    try:
        ckpt_path = _resolve_post_fit_ckpt_path(run_cfg=run_cfg, trainer=trainer)
        final_eval_experiment = str(run_cfg.get("final_eval_experiment") or "").strip()
        if final_eval_experiment:
            if ckpt_path is None:
                raise RuntimeError(
                    "Final eval experiment requires a resolved checkpoint path. "
                    "Set `test_ckpt_path`, enable checkpointing, or disable `run.final_eval_experiment`."
                )
            final_eval_cfg = _build_final_eval_cfg(cfg, ckpt_path=ckpt_path)
            return _run_final_eval_suite(final_eval_cfg)

        return _run_inprocess_test(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
    finally:
        _release_post_fit_runtime_state(model=model, datamodule=datamodule)


def train_model(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    try:

        def _log_post_fit_eval_plan(run_cfg: Any) -> None:
            log.info(
                "Training-time validation report_profile=%s answer_posterior_surrogate=%s",
                cfg.model.eval_cfg.get("report_profile"),
                format_search_eval_answer_posterior(
                    _coerce_search_eval_cfg(cfg.model.eval_cfg)
                ),
            )
            if not bool(run_cfg.get("test", False)):
                return
            final_eval_experiment = str(
                run_cfg.get("final_eval_experiment") or ""
            ).strip()
            if not final_eval_experiment:
                log.info(
                    "Post-fit evaluation is enabled via in-process trainer.test()."
                )
                return
            output_subdir = str(
                run_cfg.get("final_eval_output_subdir") or "final_eval"
            ).strip()
            eval_template = _compose_final_eval_template(
                cfg,
                ckpt_path=str(
                    run_cfg.get("test_ckpt_path")
                    or cfg.get("ckpt_path")
                    or "__rankflow_backend_guard__.ckpt"
                ),
                output_dir=str(_resolve_train_output_dir(cfg) / output_subdir),
            )
            _validate_answer_posterior_alignment(cfg, eval_template=eval_template)
            log.info(
                "Post-fit final evaluation is enabled: experiment=%s split=%s output_subdir=%s",
                final_eval_experiment,
                run_cfg.get("final_eval_split") or "test",
                run_cfg.get("final_eval_output_subdir") or "final_eval",
            )

        seed_everything_if_needed(cfg)

        resolved_run_name = apply_run_name(cfg)
        log.info("Resolved run name: %s", resolved_run_name)

        run_cfg = cfg.get("run") or {}
        _log_post_fit_eval_plan(run_cfg)

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
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=cfg.get("ckpt_path"),
            )

        train_metrics = dict(trainer.callback_metrics)
        test_metrics = _run_post_fit_evaluation(
            cfg=cfg,
            trainer=trainer,
            model=model,
            datamodule=datamodule,
        )
        return {**train_metrics, **test_metrics}, object_dict
    except Exception:
        log.exception("")
        raise
    finally:
        finalize_task(cfg=cfg, log=log)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(
            "Missing required config group: `run`. "
            "Fix: use a train config that sets `/run`, for example `experiment=train_rankflow`."
        )
    validate_train_entrypoint(cfg)
    extras(cfg)
    run_name = str(run_cfg.get("name") or "").strip()
    if not is_rankflow_train_run(run_name):
        raise ValueError(
            f"Unsupported train run.name={run_name!r}. Expected a `{RANKFLOW_TRAIN_RUN_PREFIX}*` run."
        )
    validate_train_config(cfg)

    metric_dict, _ = train_model(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    return _get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
