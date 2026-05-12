from __future__ import annotations

import os
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.schema import RetrievalBatch
from src.training.checkpoint import filter_compatible_state_dict, load_checkpoint_payload
from src.training.metrics import WeaverMetricSuite
from src.training.optimization import build_optimizer_and_scheduler
from src.training.schedule import TemperatureSchedule
from src.weaver.config import (
    build_diagnostics_runtime_config,
    build_eval_runtime_config,
    build_loss_config,
    build_policy_runtime_config,
    build_reward_config,
    build_rollout_runtime_config,
    build_sampling_runtime_config,
    validate_algorithm_coupling,
)
from src.weaver.loss import (
    BudgetedDAGDetailedBalanceLoss,
    LossOutput,
)
from src.weaver.policy import Policy
from src.weaver.reward import RewardModel
from src.weaver.rollout import RolloutRunner
from src.weaver.rollout.runner import (
    TrainingRolloutResult,
    concat_rollout_batches,
    detach_rollout_for_metrics,
)
from src.weaver.rollout.terminal_subgraph import compute_union_subgraph_masks


class WeaverModule(LightningModule):
    """
    Lightning module for Weaver subgraph-state GFlowNet training.

    Main BDB path:
        rollout collection
        -> terminal evidence reward evaluation
        -> budgeted DAG detailed-balance traces
        -> BDB loss
        -> manual backward
        -> manual optimizer step

    # REMOVED: TE-BFM/SubTB training paths — see methodology.md §3.9
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        relation_embeddings: torch.Tensor,
        hidden_dim: int = 1024,
        rollout: dict[str, Any] | None = None,
        sampling: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        eval: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
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

        reward_kwargs = build_reward_config(reward)
        rollout_runtime = build_rollout_runtime_config(
            rollout=rollout,
            runtime=runtime,
        )
        eval_runtime = build_eval_runtime_config(
            eval_cfg=eval,
            eval_num_rollout=rollout_runtime.eval_num_rollout,
        )
        sampling_runtime = build_sampling_runtime_config(sampling)
        diagnostics_runtime = build_diagnostics_runtime_config(diagnostics)
        policy_runtime = build_policy_runtime_config(
            hidden_dim=int(hidden_dim),
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            relation_embeddings=relation_embeddings,
            policy=policy,
        )

        loss_kwargs = build_loss_config(
            loss,
            max_trajectory_len=rollout_runtime.expand_budget + 1,
        )
        validate_algorithm_coupling(
            policy=policy_runtime,
            loss=loss_kwargs,
            rollout=rollout_runtime,
            reward=reward_kwargs,
        )

        self.optimizer_cfg = dict(optimizer_cfg or {})
        self.scheduler_cfg = dict(scheduler_cfg or {})

        self.gradient_clip_val = (
            None if gradient_clip_val is None else float(gradient_clip_val)
        )
        self.gradient_clip_algorithm = str(gradient_clip_algorithm)

        self.policy = Policy(
            feature_encoder_cfg=policy_runtime.feature_encoder_cfg,
            hidden_dim=policy_runtime.hidden_dim,
            mode=policy_runtime.mode,
            max_budget=rollout_runtime.expand_budget,
            flow_budget_conditioning=policy_runtime.flow_budget_conditioning,
            bdb_child_chunk_size=int(loss_kwargs.get("child_chunk_size", 2048)),
            edge_scorer=policy_runtime.edge_scorer,
            continuation_logit_bias_init=(
                policy_runtime.continuation_logit_bias_init
            ),
            continuation_mass_reduction=policy_runtime.continuation_mass_reduction,
            evidence_state_encoder_dropout=(
                policy_runtime.evidence_state_encoder_dropout
            ),
            evidence_state_encoder_cfg=policy_runtime.evidence_state_encoder_cfg,
            flow_head_cfg=policy_runtime.flow_head_cfg,
            frontier_pointer_cfg=policy_runtime.frontier_pointer_cfg,
            stop_head_cfg=policy_runtime.stop_head_cfg,
        )

        self.reward_model = RewardModel(
            relation_embeddings=relation_embeddings,
            **reward_kwargs,
        )

        self.loss_fn = build_loss(loss_kwargs)

        self.temperature_schedule = TemperatureSchedule(
            temperature=sampling_runtime.train_temperature,
            eval_temperature=sampling_runtime.eval_temperature,
        )

        self.rollout_runner = RolloutRunner(
            expand_budget=rollout_runtime.expand_budget,
            train_num_rollout=rollout_runtime.train_num_rollout,
            eval_num_rollout=rollout_runtime.eval_num_rollout,
            train_chunk_size=rollout_runtime.train_chunk_size,
            eval_chunk_size=rollout_runtime.eval_chunk_size,
        )
        self.metric_suite = WeaverMetricSuite.from_runtime_configs(
            rollout_runtime=rollout_runtime,
            eval_runtime=eval_runtime,
            diagnostics_runtime=diagnostics_runtime,
        )
        self.compute_eval_loss = bool(eval_runtime.compute_loss)

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
        result = self.run_training_rollouts_and_backward(
            batch=batch,
            rollout_temperature=temperature,
            accumulation_batches=accumulation_batches,
        )

        grad_norm = self.compute_grad_norm_if_due()

        if self.optimizer_step_due(
            batch_idx=batch_idx,
            accumulation_batches=accumulation_batches,
        ):
            self.step_optimizer(optimizer)

        metrics = self.metric_suite.train_metrics(
            result=result,
            batch=batch,
            policy=self.policy,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            temperature=temperature,
            grad_norm=grad_norm,
            global_step=int(self.global_step),
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=int(batch.num_graphs),
        )

        return {"loss": result.loss_output.loss.detach()}

    def run_training_rollouts_and_backward(
        self,
        *,
        batch: RetrievalBatch,
        rollout_temperature: float,
        accumulation_batches: int,
    ) -> TrainingRolloutResult:
        accumulation_batches = self._positive_int(
            accumulation_batches,
            "accumulation_batches",
        )
        normalize_by = self.rollout_runner.train_num_rollout * accumulation_batches

        loss_outputs: list[LossOutput] = []
        metric_rollouts = []
        diagnostic_flags = self.metric_suite.diagnostic_flags(stage="train")

        for chunk in self.rollout_runner.iter_training_rollout_chunks(
            policy=self.policy,
            reward_model=self.reward_model,
            batch=batch,
            rollout_temperature=rollout_temperature,
            loss_fn=self.loss_fn,
            **diagnostic_flags.as_runner_kwargs(),
        ):
            rollout_batch = concat_rollout_batches(chunk.rollouts)
            loss_output = self.loss_fn(rollout_batch)

            scale = float(chunk.num_rollouts) / float(normalize_by)
            self.manual_backward(loss_output.loss * scale)

            loss_outputs.append(loss_output)
            metric_rollouts.extend(
                detach_rollout_for_metrics(rollout)
                for rollout in chunk.rollouts
            )

        return TrainingRolloutResult(
            loss_output=LossOutput.aggregate(loss_outputs),
            rollouts=tuple(metric_rollouts),
        )

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
        """Inference helper: sample rollout subgraphs and return union masks."""
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
            collect_policy_diagnostics=False,
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

        checkpoint = load_checkpoint_payload(checkpoint_path)

        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = self._drop_target_policy_state(state_dict)
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
        diagnostic_flags = self.metric_suite.diagnostic_flags(stage=prefix)
        metrics: dict[str, float] = {}

        rollouts = self.rollout_runner.generate_eval_rollouts(
            policy=self.policy,
            reward_model=self.reward_model,
            batch=batch,
            temperature=self.temperature_schedule.eval_temperature,
            **diagnostic_flags.as_runner_kwargs(),
            loss_fn=self.loss_fn if self.compute_eval_loss else None,
        )

        if self.compute_eval_loss and rollouts:
            loss_output = self.loss_fn(concat_rollout_batches(rollouts))
            metrics.update(
                self.metric_suite.eval_loss_metrics(
                    loss_output=loss_output,
                    stage=prefix,
                )
            )

        metrics.update(
            self.metric_suite.eval_metrics(
                rollouts=rollouts,
                batch=batch,
                stage=prefix,
            )
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=int(batch.num_graphs),
        )
        progress_metrics = self._progress_bar_metrics(metrics, prefix=prefix)
        if progress_metrics:
            self.log_dict(
                progress_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=False,
                sync_dist=True,
                batch_size=int(batch.num_graphs),
            )

        return metrics

    @staticmethod
    def _progress_bar_metrics(
        metrics: dict[str, float],
        *,
        prefix: str,
    ) -> dict[str, float]:
        if prefix != "val":
            return {}
        aliases = {
            "val/loss/total": "val_loss",
            "val/best_of_k/target_f1": "val_f1",
            "val/sample/mean_stop_depth": "val_depth",
            "val/sample/forced_stop_rate": "val_forced",
        }
        return {
            alias: metrics[key]
            for key, alias in aliases.items()
            if key in metrics
        }

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

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get("state_dict", None)
        if isinstance(state_dict, dict):
            checkpoint["state_dict"] = self._drop_target_policy_state(state_dict)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint.pop("target_policy_optimizer_steps", None)

    @staticmethod
    def _drop_target_policy_state(
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        # Backward compatibility for older checkpoints. "target_policy" here is
        # legacy naming, not a target network or EMA teacher in the current
        # training path.
        return {
            key: value
            for key, value in state_dict.items()
            if not str(key).startswith("target_policy.")
        }

    def compute_grad_norm(self) -> float:
        total = torch.zeros((), device=self.device)
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                total = total + grad.float().pow(2).sum()
        return float(total.sqrt().item())

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value

    def compute_grad_norm_if_due(self) -> float | None:
        if not self.metric_suite.should_log_grad_norm(
            global_step=int(self.global_step),
        ):
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


def build_loss(
    loss_config: dict[str, Any],
) -> BudgetedDAGDetailedBalanceLoss:
    cfg = dict(loss_config)
    loss_type = str(cfg.pop("type", "bdb")).lower()
    if loss_type == "bdb":
        return BudgetedDAGDetailedBalanceLoss(**cfg)
    raise ValueError(f"loss.type must be 'bdb', got {loss_type!r}.")


__all__ = ["WeaverModule", "build_loss"]
