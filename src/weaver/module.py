from __future__ import annotations

import os
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.schema import RetrievalBatch
from src.eval.groups import flatten_metric_groups
from src.training.checkpoint import filter_compatible_state_dict
from src.training.diagnostics import TrainingDiagnosticsCollector
from src.training.optimization import build_optimizer_and_scheduler
from src.training.rollout_eval import evaluate_rollouts
from src.training.schedule import TemperatureSchedule
from src.weaver.config import (
    build_diagnostics_runtime_config,
    build_eval_runtime_config,
    build_policy_runtime_config,
    build_rollout_runtime_config,
    build_schedule_runtime_config,
)
from src.weaver.loss import SubTrajectoryBalanceLoss
from src.weaver.policy import Policy
from src.weaver.reward import RewardModel
from src.weaver.rollout import RewardMode, RolloutRunner
from src.weaver.rollout.local_improvement import LocalImprovementAuxiliary
from src.weaver.rollout.stop_advantage import StopAdvantageAuxiliary
from src.weaver.rollout.terminal_subgraph import compute_union_subgraph_masks


class WeaverModule(LightningModule):
    """
    Lightning module for VIGOR-style subgraph-state GFlowNet training.

    Main path:
        policy rollout
        -> terminal RewardModel
        -> SubTB + StopTB + potential-guided loss
        -> manual backward
        -> manual optimizer step

    This module intentionally does not own coverage guides, rollout proposals,
    or external teachers.
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
            state_readout_cfg=policy_runtime.state_readout_cfg,
            stop_scorer_cfg=policy_runtime.stop_scorer_cfg,
            edge_scorer_cfg=policy_runtime.edge_scorer_cfg,
            edge_residual_cfg=policy_runtime.edge_residual_cfg,
            flow_head_cfg=policy_runtime.flow_head_cfg,
            action_parameterization=policy_runtime.action_parameterization,
        )

        self.reward_model = RewardModel(**dict(reward_cfg or {}))

        loss_kwargs = normalize_loss_config(
            loss_cfg=loss_cfg,
            max_trajectory_len=rollout_runtime.expand_budget + 1,
        )
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
        self.stop_adv_auxiliary = (
            StopAdvantageAuxiliary(rollout_runtime.stop_advantage_cfg)
            if rollout_runtime.stop_advantage_cfg.enabled
            else None
        )
        self.local_improvement_auxiliary = (
            LocalImprovementAuxiliary(rollout_runtime.local_improvement_cfg)
            if rollout_runtime.local_improvement_cfg.enabled
            else None
        )
        if (
            self.stop_adv_auxiliary is not None
            and self.local_improvement_auxiliary is not None
        ):
            raise ValueError("Only one rollout auxiliary can be enabled at a time.")
        self.rollout_auxiliary = self.stop_adv_auxiliary or self.local_improvement_auxiliary

        self.train_metrics = TrainingDiagnosticsCollector(
            debug=self.debug_metrics,
            rollout_diagnostics=diagnostics_runtime.train_rollout_diagnostics,
            rollout_diagnostics_interval=diagnostics_runtime.train_rollout_diagnostics_interval,
            policy_diagnostics=diagnostics_runtime.train_policy_diagnostics,
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

        result = self.rollout_runner.run_training_rollouts_and_backward(
            policy=self.policy,
            reward_model=self.reward_model,
            loss_fn=self.loss_fn,
            backward_fn=self.manual_backward,
            batch=batch,
            rollout_temperature=temperature,
            accumulation_batches=accumulation_batches,
            auxiliary=self.rollout_auxiliary,
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

        rollouts = self.rollout_runner.generate_rollouts(
            policy=self.policy,
            reward_model=self.reward_model,
            batch=batch,
            num_rollouts=num_rollouts,
            temperature=rollout_temperature,
            collect_stop_counterfactual=False,
            collect_policy_diagnostics=False,
            reward_mode=RewardMode.EAGER_STOP_NOW,
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
        state_dict = filter_compatible_state_dict(
            state_dict=state_dict,
            current_state=self.state_dict(),
            strict=strict,
        )
        incompatible = self.load_state_dict(state_dict, strict=strict)

        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

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
            online_rollouts=tuple(result.rollouts),
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


def normalize_loss_config(
    *,
    loss_cfg: dict[str, Any] | None,
    max_trajectory_len: int,
) -> dict[str, Any]:
    cfg = dict(loss_cfg or {})
    if cfg.get("max_trajectory_len") is None:
        cfg["max_trajectory_len"] = int(max_trajectory_len)
    return cfg


__all__ = ["WeaverModule"]
