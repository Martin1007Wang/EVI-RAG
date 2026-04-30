from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.schema import RetrievalBatch
from src.eval.groups import flatten_metric_groups
from src.training.diagnostics import TrainingDiagnosticsCollector
from src.training.optimization import build_optimizer_and_scheduler
from src.training.rollout_eval import evaluate_rollouts
from src.training.schedule import TemperatureSchedule
from src.weaver.losses import SubTrajectoryBalanceLoss
from src.weaver.policy import Policy
from src.weaver.reward import RewardModel
from src.weaver.rollout import RolloutRunner
from src.weaver.rollout.terminal_subgraph import compute_union_subgraph_masks


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    hidden_dim: int
    feature_encoder_cfg: dict[str, Any]
    state_readout_dropout: float
    stop_scorer_cfg: dict[str, Any]
    edge_scorer_cfg: dict[str, Any]
    flow_head_cfg: dict[str, Any]


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    expand_budget: int
    train_num_rollout: int
    eval_num_rollout: int
    train_chunk_size: int
    eval_chunk_size: int


@dataclass(frozen=True)
class EvalRuntimeConfig:
    budgets: tuple[int, ...]
    debug_metrics: bool
    exclude_anchors_from_retrieved: bool
    use_reachable_targets: bool


@dataclass(frozen=True)
class ScheduleRuntimeConfig:
    temperature: float
    eval_temperature: float
    temperature_cfg: dict[str, Any] | None


@dataclass(frozen=True)
class DiagnosticsRuntimeConfig:
    train_rollout_diagnostics: bool
    train_rollout_diagnostics_interval: int
    train_stop_counterfactual: bool
    train_policy_diagnostics: bool
    train_validate_rollout_depth: bool
    eval_stop_counterfactual: bool
    eval_validate_rollout_depth: bool
    grad_norm_interval: int


class WeaverModule(LightningModule):
    """
    Lightning module for VIGOR-style subgraph-state GFlowNet training.

    Main path:
        policy rollout
        -> terminal RewardModel
        -> SubTB + StopTB + VIGOR auxiliary loss
        -> manual backward
        -> manual optimizer step

    This module intentionally does not own coverage guides, rollout proposals,
    or external teachers. VIGOR is a loss-level reward-improvement objective,
    not a behavior-policy guide.
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        relation_embeddings: torch.Tensor,
        policy_cfg: dict[str, Any] | None = None,
        rollout_cfg: dict[str, Any] | None = None,
        eval_cfg: dict[str, Any] | None = None,
        schedule_cfg: dict[str, Any] | None = None,
        reward_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        diagnostic_cfg: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "entity_text_embeddings",
                "entity_embedding_map",
                "relation_embeddings",
            ]
        )

        policy_runtime = build_policy_runtime_config(
            policy_cfg=policy_cfg,
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            relation_embeddings=relation_embeddings,
        )
        rollout_runtime = build_rollout_runtime_config(rollout_cfg)
        eval_runtime = build_eval_runtime_config(
            eval_cfg=eval_cfg,
            eval_num_rollout=rollout_runtime.eval_num_rollout,
        )
        schedule_runtime = build_schedule_runtime_config(schedule_cfg)
        diagnostics_runtime = build_diagnostics_runtime_config(diagnostic_cfg)

        self.optimizer_cfg = dict(optimizer_cfg or {})
        self.scheduler_cfg = dict(scheduler_cfg or {})

        self.gradient_clip_val = (
            None if gradient_clip_val is None else float(gradient_clip_val)
        )
        self.gradient_clip_algorithm = str(gradient_clip_algorithm)

        self.eval_budgets = eval_runtime.budgets
        self.debug_metrics = eval_runtime.debug_metrics
        self.exclude_anchors_from_retrieved = (
            eval_runtime.exclude_anchors_from_retrieved
        )
        self.use_reachable_targets = eval_runtime.use_reachable_targets

        self.train_stop_counterfactual = diagnostics_runtime.train_stop_counterfactual
        self.train_policy_diagnostics = diagnostics_runtime.train_policy_diagnostics
        self.train_validate_rollout_depth = (
            diagnostics_runtime.train_validate_rollout_depth
        )
        self.eval_stop_counterfactual = diagnostics_runtime.eval_stop_counterfactual
        self.eval_validate_rollout_depth = (
            diagnostics_runtime.eval_validate_rollout_depth
        )
        self.grad_norm_interval = diagnostics_runtime.grad_norm_interval

        self.policy = Policy(
            feature_encoder_cfg=policy_runtime.feature_encoder_cfg,
            hidden_dim=policy_runtime.hidden_dim,
            state_readout_dropout=policy_runtime.state_readout_dropout,
            stop_scorer_cfg=policy_runtime.stop_scorer_cfg,
            edge_scorer_cfg=policy_runtime.edge_scorer_cfg,
            flow_head_cfg=policy_runtime.flow_head_cfg,
        )

        self.reward_model = RewardModel(**dict(reward_cfg or {}))

        loss_kwargs = dict(loss_cfg or {})
        loss_kwargs.setdefault("max_trajectory_len", rollout_runtime.expand_budget + 1)
        self.loss_fn = SubTrajectoryBalanceLoss(**loss_kwargs)

        self.temperature_schedule = TemperatureSchedule(
            temperature=schedule_runtime.temperature,
            eval_temperature=schedule_runtime.eval_temperature,
            cfg=schedule_runtime.temperature_cfg,
        )

        self.rollout_runner = RolloutRunner(
            expand_budget=rollout_runtime.expand_budget,
            train_num_rollout=rollout_runtime.train_num_rollout,
            eval_num_rollout=rollout_runtime.eval_num_rollout,
            train_chunk_size=rollout_runtime.train_chunk_size,
            eval_chunk_size=rollout_runtime.eval_chunk_size,
        )

        self.expand_budget = rollout_runtime.expand_budget
        self.train_num_rollout = rollout_runtime.train_num_rollout

        self.train_metrics = TrainingDiagnosticsCollector(
            debug=self.debug_metrics,
            rollout_diagnostics=diagnostics_runtime.train_rollout_diagnostics,
            rollout_diagnostics_interval=diagnostics_runtime.train_rollout_diagnostics_interval,
        )

        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        logger = self.logger
        if logger is None:
            return

        experiment = getattr(logger, "experiment", None)
        define_metric = getattr(experiment, "define_metric", None)
        if not callable(define_metric):
            return

        define_metric("trainer/global_step")
        for prefix in ("train/*", "val/*", "test/*"):
            define_metric(prefix, step_metric="trainer/global_step")

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        optimizer = self.optimizer()
        accumulation_batches = self.accumulation_batches()

        temperature = self.temperature_schedule.current(self.global_step)
        self.loss_fn.set_global_step(int(self.global_step))

        result = self.rollout_runner.run_training_rollouts_and_backward(
            policy=self.policy,
            reward_model=self.reward_model,
            loss_fn=self.loss_fn,
            backward_fn=self.manual_backward,
            batch=batch,
            rollout_temperature=temperature,
            accumulation_batches=accumulation_batches,
            collect_stop_counterfactual=self.train_stop_counterfactual,
            collect_policy_diagnostics=self.train_policy_diagnostics,
            validate_synchronous_depth=self.train_validate_rollout_depth,
        )

        grad_norm = self.compute_grad_norm_if_due()

        if self.optimizer_step_due(
            batch_idx=batch_idx,
            accumulation_batches=accumulation_batches,
        ):
            self.step_optimizer(optimizer)

        self.log_training_step(
            result=result,
            batch=batch,
            optimizer=optimizer,
            temperature=temperature,
            grad_norm=grad_norm,
        )

        return {"loss": result.loss_output.loss.detach()}

    @torch.no_grad()
    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> dict[str, Any]:
        del batch_idx
        return self.eval_step(batch=batch, prefix="val")

    @torch.no_grad()
    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> dict[str, Any]:
        del batch_idx
        return self.eval_step(batch=batch, prefix="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return build_optimizer_and_scheduler(
            module=self,
            optimizer_cfg=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
        )

    @torch.no_grad()
    def forward(
        self,
        batch: RetrievalBatch,
        num_rollouts: int = 1,
        temperature: float | None = None,
    ) -> Any:
        return self.generate_subgraph_masks(
            batch=batch,
            num_rollouts=num_rollouts,
            temperature=temperature,
        )

    @torch.no_grad()
    def generate_subgraph_masks(
        self,
        *,
        batch: RetrievalBatch,
        num_rollouts: int = 1,
        temperature: float | None = None,
    ) -> Any:
        num_rollouts = int(num_rollouts)
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}.")

        rollout_temperature = (
            float(temperature)
            if temperature is not None
            else self.temperature_schedule.eval_temperature
        )

        rollouts = self.rollout_runner.generate_online_rollouts(
            policy=self.policy,
            reward_model=self.reward_model,
            batch=batch,
            num_rollouts=num_rollouts,
            temperature=rollout_temperature,
        )

        return compute_union_subgraph_masks(
            rollouts=rollouts,
            batch=batch,
        )

    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = self._filter_pretrained_state_dict(state_dict, strict=strict)
        incompatible = self.load_state_dict(state_dict, strict=strict)

        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def _filter_pretrained_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        strict: bool,
    ) -> dict[str, Any]:
        if strict:
            return state_dict

        current_state = self.state_dict()
        filtered: dict[str, Any] = {}
        reset_stop_gate = False

        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(current_value, torch.Tensor)
                and value.shape != current_value.shape
            ):
                if key.startswith("policy.action_scorer.gate.0."):
                    reset_stop_gate = True
                continue
            filtered[key] = value

        if reset_stop_gate:
            stop_gate_prefixes = (
                "policy.action_scorer.gate.",
                "policy.action_scorer.stop_bias",
                "policy.action_scorer.expand_bias",
            )
            filtered = {
                key: value
                for key, value in filtered.items()
                if not key.startswith(stop_gate_prefixes)
            }

        return filtered

    def eval_step(
        self,
        *,
        batch: RetrievalBatch,
        prefix: str,
    ) -> dict[str, Any]:
        rollouts = self.rollout_runner.generate_eval_rollouts(
            policy=self.policy,
            reward_model=self.reward_model,
            batch=batch,
            temperature=self.temperature_schedule.eval_temperature,
            collect_stop_counterfactual=self.eval_stop_counterfactual
            or self.debug_metrics,
            collect_policy_diagnostics=(prefix != "test"),
            validate_synchronous_depth=self.eval_validate_rollout_depth,
        )

        metrics = evaluate_rollouts(
            rollouts=rollouts,
            batch=batch,
            eval_budgets=self.eval_budgets,
            debug_metrics=self.debug_metrics,
            exclude_anchors_from_retrieved=self.exclude_anchors_from_retrieved,
            use_reachable_targets=self.use_reachable_targets,
            stage=prefix,
        )

        self.log_dict(
            flatten_metric_groups(metrics, prefix=prefix),
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix == "val"),
            sync_dist=True,
            batch_size=int(batch.num_graphs),
        )

        return metrics

    def log_training_step(
        self,
        *,
        result: Any,
        batch: RetrievalBatch,
        optimizer: torch.optim.Optimizer,
        temperature: float,
        grad_norm: float | None,
    ) -> None:
        metrics = self.train_metrics.collect(
            loss_output=result.loss_output,
            batch=batch,
            online_rollouts=tuple(result.rollouts.online),
            coverage_rollouts=(),
            policy=self.policy,
            root_expand_budget=self.expand_budget,
            global_step=int(self.global_step),
        )

        metrics.update(
            {
                "train/optim/lr": float(optimizer.param_groups[0]["lr"]),
                "train/optim/temperature": float(temperature),
            }
        )
        if grad_norm is not None:
            metrics["train/optim/grad_norm"] = float(grad_norm)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=int(batch.num_graphs),
        )

    def optimizer(self) -> torch.optim.Optimizer:
        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        return cast(torch.optim.Optimizer, optimizer)

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.gradient_clip_val),
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.step_schedulers()

    def compute_grad_norm(self) -> float:
        total = torch.zeros((), device=self.device)
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                total = total + grad.float().pow(2).sum()
        return float(total.sqrt().item())

    def compute_grad_norm_if_due(self) -> float | None:
        if self.grad_norm_interval <= 0:
            return None
        if int(self.global_step) % int(self.grad_norm_interval) != 0:
            return None
        return self.compute_grad_norm()

    def step_schedulers(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return

        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for scheduler in schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                raise RuntimeError(
                    "ReduceLROnPlateau requires a monitored validation metric and "
                    "is not compatible with the current manual step-based scheduler path."
                )
            scheduler.step()

    def accumulation_batches(self) -> int:
        accumulation_batches = getattr(self.trainer, "accumulate_grad_batches", 1)

        if not isinstance(accumulation_batches, int):
            raise TypeError(
                "Manual optimization expects trainer.accumulate_grad_batches "
                f"to be an int, got {type(accumulation_batches)!r}."
            )

        if accumulation_batches < 1:
            raise ValueError(
                "trainer.accumulate_grad_batches must be >= 1, "
                f"got {accumulation_batches}."
            )

        return accumulation_batches

    def optimizer_step_due(
        self,
        *,
        batch_idx: int,
        accumulation_batches: int,
    ) -> bool:
        if (int(batch_idx) + 1) % int(accumulation_batches) == 0:
            return True

        num_batches = getattr(self.trainer, "num_training_batches", None)
        return (
            isinstance(num_batches, int)
            and num_batches > 0
            and (int(batch_idx) + 1) == num_batches
        )


def build_policy_runtime_config(
    *,
    policy_cfg: dict[str, Any] | None,
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
) -> PolicyRuntimeConfig:
    cfg = dict(policy_cfg or {})

    hidden_dim = int(cfg.pop("hidden_dim", 1024))
    state_readout_dropout = float(cfg.pop("state_readout_dropout", 0.0))

    feature_encoder_cfg = dict(cfg.pop("feature_encoder", {}))
    stop_scorer_cfg = dict(cfg.pop("stop_scorer", {}))
    edge_scorer_cfg = dict(cfg.pop("edge_scorer", {}))
    flow_head_cfg = dict(cfg.pop("flow_head", {}))

    if cfg:
        raise ValueError(f"Unused policy_cfg keys: {sorted(cfg)}.")

    feature_encoder_cfg = build_feature_encoder_config(
        cfg=feature_encoder_cfg,
        entity_text_embeddings=entity_text_embeddings,
        entity_embedding_map=entity_embedding_map,
        relation_embeddings=relation_embeddings,
        hidden_dim=hidden_dim,
    )

    return PolicyRuntimeConfig(
        hidden_dim=hidden_dim,
        feature_encoder_cfg=feature_encoder_cfg,
        state_readout_dropout=state_readout_dropout,
        stop_scorer_cfg=stop_scorer_cfg,
        edge_scorer_cfg=edge_scorer_cfg,
        flow_head_cfg=flow_head_cfg,
    )


def build_rollout_runtime_config(
    rollout_cfg: dict[str, Any] | None,
) -> RolloutRuntimeConfig:
    cfg = dict(rollout_cfg or {})

    expand_budget = int(cfg.pop("expand_budget", 3))
    train_num_rollout = int(cfg.pop("train_num_rollout", 8))
    eval_num_rollout = int(cfg.pop("eval_num_rollout", 8))

    train_chunk_size = cfg.pop("train_chunk_size", train_num_rollout)
    eval_chunk_size = cfg.pop("eval_chunk_size", eval_num_rollout)

    if cfg:
        raise ValueError(f"Unused rollout_cfg keys: {sorted(cfg)}.")

    validate_rollout_counts(
        expand_budget=expand_budget,
        train_num_rollout=train_num_rollout,
        eval_num_rollout=eval_num_rollout,
    )

    return RolloutRuntimeConfig(
        expand_budget=expand_budget,
        train_num_rollout=train_num_rollout,
        eval_num_rollout=eval_num_rollout,
        train_chunk_size=normalize_chunk_size(
            train_chunk_size,
            fallback=train_num_rollout,
            name="train_chunk_size",
        ),
        eval_chunk_size=normalize_chunk_size(
            eval_chunk_size,
            fallback=eval_num_rollout,
            name="eval_chunk_size",
        ),
    )


def build_eval_runtime_config(
    *,
    eval_cfg: dict[str, Any] | None,
    eval_num_rollout: int,
) -> EvalRuntimeConfig:
    cfg = dict(eval_cfg or {})

    raw_budgets = cfg.pop("budgets", (1, 2, 4, 8))
    debug_metrics = bool(cfg.pop("debug_metrics", False))
    exclude_anchors_from_retrieved = bool(
        cfg.pop("exclude_anchors_from_retrieved", True)
    )
    use_reachable_targets = bool(cfg.pop("use_reachable_targets", True))

    if cfg:
        raise ValueError(f"Unused eval_cfg keys: {sorted(cfg)}.")

    budgets = tuple(sorted({int(k) for k in raw_budgets}))

    if not budgets:
        raise ValueError("eval_cfg.budgets must be non-empty.")
    if any(k < 1 for k in budgets):
        raise ValueError(f"eval_cfg.budgets must all be >= 1, got {budgets}.")
    if max(budgets) > int(eval_num_rollout):
        raise ValueError(
            f"max(eval_cfg.budgets)={max(budgets)} cannot exceed "
            f"eval_num_rollout={eval_num_rollout}."
        )

    return EvalRuntimeConfig(
        budgets=budgets,
        debug_metrics=debug_metrics,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )


def build_schedule_runtime_config(
    schedule_cfg: dict[str, Any] | None,
) -> ScheduleRuntimeConfig:
    cfg = dict(schedule_cfg or {})

    temperature = float(cfg.pop("temperature", 1.0))
    eval_temperature = float(cfg.pop("eval_temperature", temperature))
    temperature_cfg = cfg.pop("temperature_cfg", None)

    if cfg:
        raise ValueError(f"Unused schedule_cfg keys: {sorted(cfg)}.")

    return ScheduleRuntimeConfig(
        temperature=temperature,
        eval_temperature=eval_temperature,
        temperature_cfg=dict(temperature_cfg) if temperature_cfg is not None else None,
    )


def build_diagnostics_runtime_config(
    diagnostic_cfg: dict[str, Any] | None,
) -> DiagnosticsRuntimeConfig:
    cfg = dict(diagnostic_cfg or {})

    train_rollout_diagnostics = bool(cfg.pop("train_rollout_diagnostics", False))
    train_rollout_diagnostics_interval = int(
        cfg.pop("train_rollout_diagnostics_interval", 0)
    )
    train_stop_counterfactual = bool(cfg.pop("train_stop_counterfactual", True))
    train_policy_diagnostics = bool(cfg.pop("train_policy_diagnostics", False))
    train_validate_rollout_depth = bool(cfg.pop("train_validate_rollout_depth", False))
    eval_stop_counterfactual = bool(cfg.pop("eval_stop_counterfactual", True))
    eval_validate_rollout_depth = bool(cfg.pop("eval_validate_rollout_depth", False))
    grad_norm_interval = int(cfg.pop("grad_norm_interval", 0))

    if cfg:
        raise ValueError(f"Unused diagnostic_cfg keys: {sorted(cfg)}.")
    if train_rollout_diagnostics_interval < 0:
        raise ValueError(
            "diagnostic_cfg.train_rollout_diagnostics_interval must be >= 0, "
            f"got {train_rollout_diagnostics_interval}."
        )
    if grad_norm_interval < 0:
        raise ValueError(
            f"diagnostic_cfg.grad_norm_interval must be >= 0, got {grad_norm_interval}."
        )

    return DiagnosticsRuntimeConfig(
        train_rollout_diagnostics=train_rollout_diagnostics,
        train_rollout_diagnostics_interval=train_rollout_diagnostics_interval,
        train_stop_counterfactual=train_stop_counterfactual,
        train_policy_diagnostics=train_policy_diagnostics,
        train_validate_rollout_depth=train_validate_rollout_depth,
        eval_stop_counterfactual=eval_stop_counterfactual,
        eval_validate_rollout_depth=eval_validate_rollout_depth,
        grad_norm_interval=grad_norm_interval,
    )


def build_feature_encoder_config(
    *,
    cfg: dict[str, Any],
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
    hidden_dim: int,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("hidden_dim", int(hidden_dim))

    forbidden = {
        "entity_text_embeddings",
        "entity_embedding_map",
        "relation_embeddings",
    }

    overlap = forbidden.intersection(cfg)
    if overlap:
        raise ValueError(
            "policy_cfg.feature_encoder must not contain runtime embedding tensors: "
            f"{sorted(overlap)}."
        )

    cfg.update(
        {
            "entity_text_embeddings": entity_text_embeddings,
            "entity_embedding_map": entity_embedding_map,
            "relation_embeddings": relation_embeddings,
        }
    )

    return cfg


def validate_rollout_counts(
    *,
    expand_budget: int,
    train_num_rollout: int,
    eval_num_rollout: int,
) -> None:
    if expand_budget < 0:
        raise ValueError(f"expand_budget must be >= 0, got {expand_budget}.")
    if train_num_rollout < 1:
        raise ValueError(f"train_num_rollout must be >= 1, got {train_num_rollout}.")
    if eval_num_rollout < 1:
        raise ValueError(f"eval_num_rollout must be >= 1, got {eval_num_rollout}.")


def normalize_chunk_size(
    value: Any,
    *,
    fallback: int,
    name: str,
) -> int:
    if value is None:
        return int(fallback)

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1 or None, got {value}.")

    return value


__all__ = ["WeaverModule"]
