from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.metrics_writer import append_stage_metrics
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components

DEFAULT_MONITOR = "val/rollout_union@8/recall"
CURRENT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "weaver.yaml"


@dataclass(frozen=True, slots=True)
class BestCheckpoint:
    ckpt_path: Path
    epoch: int
    step: int
    monitor_name: str
    monitor_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing test metrics for a completed training run by "
            "re-evaluating its best checkpoint and writing test.jsonl into the run's artifacts."
        ),
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more historical training run directories, e.g. outputs/train_webqsp/2026-06-07/18-41-00",
    )
    parser.add_argument(
        "--metric-name",
        default=DEFAULT_MONITOR,
        help="Validation metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. If set, val.jsonl is not used to choose the checkpoint.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an existing test.jsonl instead of skipping the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, checkpoint, and output path without running evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for deterministic evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        backfill_run(
            run_dir=run_dir.resolve(),
            metric_name=str(args.metric_name),
            explicit_ckpt_path=args.ckpt_path.resolve() if args.ckpt_path is not None else None,
            replace_existing=bool(args.replace_existing),
            dry_run=bool(args.dry_run),
            seed=args.seed,
        )


def backfill_run(
    *,
    run_dir: Path,
    metric_name: str,
    explicit_ckpt_path: Path | None,
    replace_existing: bool,
    dry_run: bool,
    seed: int | None,
) -> None:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")

    metrics_dir = run_dir / "artifacts" / "metrics"
    output_path = metrics_dir / "test.jsonl"
    if output_path.exists():
        if not replace_existing:
            print(f"skip {run_dir}: {output_path} already exists (use --replace-existing)")
            return
        output_path.unlink()

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    _prepare_eval_cfg(cfg, seed=seed)

    best = (
        resolve_explicit_checkpoint(explicit_ckpt_path)
        if explicit_ckpt_path is not None
        else resolve_best_checkpoint(run_dir=run_dir, metric_name=metric_name)
    )

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {best.ckpt_path}")
    print(f"monitor: {best.monitor_name}={best.monitor_value:.6f} at epoch={best.epoch} step={best.step}")
    print(f"output: {output_path}")
    if dry_run:
        return

    metrics = evaluate_checkpoint(cfg=cfg, ckpt_path=best.ckpt_path)
    append_stage_metrics(
        output_dir=metrics_dir,
        stage="test",
        step=best.step,
        epoch=best.epoch,
        metrics=metrics,
        metadata={
            "checkpoint": str(best.ckpt_path),
            "monitor_name": best.monitor_name,
            "monitor_value": best.monitor_value,
            "source_run_dir": str(run_dir),
            "backfilled": True,
        },
    )
    print(f"wrote {output_path}")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


def _prepare_eval_cfg(cfg: DictConfig, *, seed: int | None) -> None:
    if seed is not None:
        cfg.seed = int(seed)
    _normalize_legacy_model_cfg(cfg)
    _materialize_model_references(cfg)
    cfg.logger = None
    cfg.callbacks = None
    cfg.profiler = None
    cfg.test_after_fit = False
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.enable_progress_bar = True
    cfg.trainer.accelerator = "auto"
    cfg.trainer.devices = 1


def _normalize_legacy_model_cfg(cfg: DictConfig) -> None:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        return
    if "forward_feature_encoder" in model_cfg:
        return

    current_model_cfg = OmegaConf.load(CURRENT_MODEL_CONFIG)
    normalized = OmegaConf.create(OmegaConf.to_container(current_model_cfg, resolve=False))

    for key in ("_target_", "budget", "sem_dim", "hidden_dim", "validate_batch_coordinates"):
        if key in model_cfg:
            normalized[key] = copy.deepcopy(model_cfg[key])

    _copy_allowed_fields(
        target=normalized.reward_model,
        source=model_cfg.get("reward_model"),
        keys=(
            "_target_",
            "reward_beta",
            "edge_cost_lambda",
            "reward_epsilon",
            "center_log_reward",
            "budget",
        ),
    )
    _copy_allowed_fields(
        target=normalized.objective,
        source=model_cfg.get("objective"),
        keys=(
            "_target_",
            "subtb_lambda",
            "terminal_loss_weight",
            "replay_loss_weight",
            "path_nce_weight",
            "path_nce_temperature",
        ),
    )
    _copy_allowed_fields(
        target=normalized.runner,
        source=model_cfg.get("runner"),
        keys=(
            "_target_",
            "engine",
            "replay_source",
            "replay_keep_ratio",
            "min_replay_per_graph",
            "train_policy_rollouts",
            "eval_rollouts",
        ),
    )
    _copy_allowed_fields(
        target=normalized.evaluation,
        source=model_cfg.get("evaluation"),
        keys=(
            "k_windows",
            "exclude_anchors_from_retrieved",
            "use_reachable_targets",
            "enable_terminal_diagnostics",
            "diversity_edge_penalty",
        ),
    )

    if "feature_encoder" in model_cfg:
        normalized.forward_feature_encoder = copy.deepcopy(model_cfg.feature_encoder)
        normalized.backward_feature_encoder = copy.deepcopy(model_cfg.feature_encoder)

    legacy_policy = model_cfg.get("policy")
    if legacy_policy is not None:
        for key in ("frontier_pruning", "state_encoder", "flow_estimator", "state_flow_head"):
            if key in legacy_policy:
                normalized.forward_policy[key] = copy.deepcopy(legacy_policy[key])
        flow_estimator = normalized.forward_policy.get("flow_estimator")
        if flow_estimator is not None and "align_scale_init" in flow_estimator:
            del flow_estimator["align_scale_init"]
        legacy_backward = legacy_policy.get("backward_policy")
        if legacy_backward is not None:
            normalized.backward_policy.backward_policy = copy.deepcopy(legacy_backward)
            if "state_encoder" in legacy_policy:
                normalized.backward_policy.state_encoder = copy.deepcopy(legacy_policy.state_encoder)

    legacy_optimization = model_cfg.get("optimization")
    if legacy_optimization is not None:
        optimizer = legacy_optimization.get("optimizer")
        scheduler = legacy_optimization.get("scheduler")
        if optimizer is not None:
            normalized.optimization.forward.optimizer = copy.deepcopy(optimizer)
            normalized.optimization.backward.optimizer = copy.deepcopy(optimizer)
        if scheduler is not None:
            normalized.optimization.forward.scheduler = copy.deepcopy(scheduler)
            normalized.optimization.backward.scheduler = copy.deepcopy(scheduler)
        if "target_ema_decay" in legacy_optimization:
            normalized.optimization.target_ema_decay = copy.deepcopy(legacy_optimization.target_ema_decay)

    cfg.model = normalized


def _copy_allowed_fields(*, target: Any, source: Any, keys: tuple[str, ...]) -> None:
    if source is None or target is None:
        return
    for key in keys:
        if key not in source:
            continue
        target[key] = copy.deepcopy(source[key])


def _materialize_model_references(cfg: DictConfig) -> None:
    model_cfg = cfg.model
    budget = int(model_cfg.budget)
    hidden_dim = int(model_cfg.hidden_dim)
    sem_dim = int(model_cfg.sem_dim)

    for encoder_key in ("forward_feature_encoder", "backward_feature_encoder"):
        encoder_cfg = model_cfg.get(encoder_key)
        if encoder_cfg is None:
            continue
        encoder_cfg.sem_dim = sem_dim
        encoder_cfg.hidden_dim = hidden_dim

    model_cfg.forward_policy.state_encoder.hidden_dim = hidden_dim
    model_cfg.forward_policy.flow_estimator.hidden_dim = hidden_dim
    model_cfg.forward_policy.state_flow_head.state_dim = hidden_dim
    model_cfg.backward_policy.state_encoder.hidden_dim = hidden_dim
    model_cfg.backward_policy.backward_policy.hidden_dim = hidden_dim
    model_cfg.reward_model.budget = budget


def resolve_explicit_checkpoint(ckpt_path: Path) -> BestCheckpoint:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return BestCheckpoint(
        ckpt_path=ckpt_path,
        epoch=-1,
        step=-1,
        monitor_name="explicit_ckpt_path",
        monitor_value=float("nan"),
    )


def resolve_best_checkpoint(*, run_dir: Path, metric_name: str) -> BestCheckpoint:
    val_metrics_path = run_dir / "artifacts" / "metrics" / "val.jsonl"
    if not val_metrics_path.is_file():
        raise FileNotFoundError(f"Validation metrics file not found: {val_metrics_path}")

    best_record: tuple[float, int, int] | None = None
    with val_metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics", {})
            value = metrics.get(metric_name)
            if value is None:
                continue
            current = (float(value), int(payload["epoch"]), int(payload["step"]))
            if best_record is None or current[0] > best_record[0]:
                best_record = current

    if best_record is None:
        raise ValueError(f"No metric named {metric_name!r} found in {val_metrics_path}")

    metric_value, epoch, step = best_record
    ckpt_path = run_dir / "checkpoints" / f"epoch_epoch={epoch:03d}-step_step={step:07d}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            "Resolved best checkpoint does not exist. "
            f"Expected {ckpt_path} from metric {metric_name}={metric_value:.6f}."
        )

    return BestCheckpoint(
        ckpt_path=ckpt_path,
        epoch=epoch,
        step=step,
        monitor_name=metric_name,
        monitor_value=metric_value,
    )


def evaluate_checkpoint(*, cfg: DictConfig, ckpt_path: Path) -> dict[str, float]:
    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed), workers=True)

    datamodule, resources = prepare_training_components(cfg, stage="test")
    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(ckpt_path), strict=False)
    if missing or unexpected:
        print(f"checkpoint load: missing={missing}, unexpected={unexpected}")

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=False,
        profiler=None,
    )
    if not isinstance(trainer, Trainer):
        raise TypeError(f"Expected Trainer, got {type(trainer).__name__}.")

    trainer.test(model=model, datamodule=datamodule)
    return scalarize_metrics(dict(trainer.callback_metrics), prefix="test/")


def scalarize_metrics(metrics: dict[str, Any], *, prefix: str) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if hasattr(value, "item"):
            scalars[key] = float(value.item())
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
    return scalars


if __name__ == "__main__":
    main()
