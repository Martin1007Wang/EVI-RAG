from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)

from src.data.schema import RetrievalBatch, validate_retrieval_batch
from src.eval.rollout import evaluate_rollout_samples
from src.training.config import EvalRuntimeConfig, OptimizationRuntimeConfig
from src.training.optimization import (
    build_lightning_scheduler_config,
    build_optimizer,
    build_scheduler,
)
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureBank, FeatureEncoder
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.runner import RolloutBatch, RolloutRunner
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.reward import TerminalRecallReward

Scalar = torch.Tensor | float | int


@dataclass(frozen=True, slots=True)
class BatchContext:
    graph: GraphContext
    target: TargetContext
    features: FeatureBank


@dataclass(frozen=True, slots=True)
class StepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, Scalar]


class WeaverModule(LightningModule):
    """
    Minimal Lightning wrapper for Weaver.

    Responsibilities:
    - build GraphContext / TargetContext / FeatureBank;
    - sample rollouts through RolloutRunner;
    - call the training loss;
    - call rollout evaluation;
    - configure optimizer / scheduler;
    - log a small number of scalar metrics.

    Non-responsibilities:
    - no State/Frontier manipulation;
    - no PolicyOutput slicing;
    - no SubTB input construction;
    - no backward-policy logic;
    - no oracle diagnostics;
    - no metric-suite indirection;
    - no debug text lookup;
    - no gradient norm logging.
    """

    def __init__(
        self,
        *,
        budget: int,
        policy_feature_encoder: FeatureEncoder,
        policy: ForwardPolicy,
        reward_model: TerminalRecallReward,
        loss_fn: torch.nn.Module,
        weak_replay_loss: torch.nn.Module | None = None,
        runner: RolloutRunner,
        optimization: OptimizationRuntimeConfig,
        evaluation: EvalRuntimeConfig,
        validate_batch_coordinates: bool = False,
    ) -> None:
        super().__init__()

        self.budget = int(budget)
        self.policy_feature_encoder = policy_feature_encoder
        self.policy = policy
        self.reward_model = reward_model
        self.loss_fn = loss_fn
        self.weak_replay_loss = weak_replay_loss
        self.runner = runner
        self.optimization = optimization
        self.evaluation = evaluation
        self.validate_batch_coordinates = bool(validate_batch_coordinates)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            modules=self._optimized_modules(),
            cfg=self.optimization.optimizer,
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            cfg=self.optimization.scheduler,
            trainer=self.trainer,
            base_lr=self.optimization.optimizer.lr,
        )

        if scheduler is None:
            return optimizer

        config: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": build_lightning_scheduler_config(
                scheduler=scheduler,
                interval=self.optimization.scheduler.interval,
            ),
        }
        return config

    def _optimized_modules(self) -> tuple[torch.nn.Module, ...]:
        """
        Train only modules that can own learnable parameters.

        Reward is included deliberately: current TerminalRecallReward has no
        parameters, but this keeps the module correct if reward later becomes
        learnable.
        """

        return (
            self.policy_feature_encoder,
            self.policy,
            self.reward_model,
            *(tuple() if self.weak_replay_loss is None else (self.weak_replay_loss,)),
        )

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx

        output = self.compute_train_step(batch)

        batch_size = graph_batch_size(batch)

        self.log_scalar(
            "train/loss",
            output.loss,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log_scalars(
            prefix="train",
            values=output.metrics,
            batch_size=batch_size,
        )

        return output.loss

    def compute_train_step(
        self,
        batch: RetrievalBatch,
    ) -> StepOutput:
        ctx = self.batch_context(batch)

        rollout = self.train_rollout(
            batch=batch,
            ctx=ctx,
        )

        loss_out = self.loss_fn(
            policy=self.policy,
            reward_model=self.reward_model,
            features=ctx.features,
            graph_context=ctx.graph,
            target_context=ctx.target,
            trajectories=rollout.trajectories,
        )

        weak_out = self._compute_weak_replay_loss(
            ctx=ctx,
            rollout=rollout,
        )

        loss = self._extract_loss(loss_out) + self._extract_loss(weak_out)
        metrics = dict(self._extract_metrics(loss_out))
        metrics.update(self._extract_metrics(weak_out))
        metrics.update(rollout.metrics)

        return StepOutput(
            loss=loss,
            metrics=metrics,
        )

    def _compute_weak_replay_loss(
        self,
        *,
        ctx: BatchContext,
        rollout: RolloutBatch,
    ) -> Any:
        if self.weak_replay_loss is None:
            zero = torch.zeros((), dtype=torch.float32, device=ctx.graph.device)
            return StepOutput(
                loss=zero,
                metrics={
                    "weak_replay/loss": zero,
                    "weak_replay/active_state_count": zero,
                },
            )
        return self.weak_replay_loss(
            policy=self.policy,
            features=ctx.features,
            graph_context=ctx.graph,
            target_context=ctx.target,
            weak_replay=rollout.weak_replay,
        )

    def train_rollout(
        self,
        *,
        batch: RetrievalBatch,
        ctx: BatchContext,
    ) -> RolloutBatch:
        with torch.no_grad():
            return self.runner.train_rollouts(
                policy=self.policy,
                batch=batch,
                context=ctx.graph,
                features=ctx.features,
                target_context=ctx.target,
            )

    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self.eval_step(
            split="val",
            batch=batch,
        )

    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self.eval_step(
            split="test",
            batch=batch,
        )

    def predict_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> TrajectoryBatch:
        del batch_idx, dataloader_idx

        with torch.no_grad():
            ctx = self.batch_context(batch)
            return self.runner.eval_rollouts(
                policy=self.policy,
                context=ctx.graph,
                features=ctx.features,
            )

    def eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        with torch.no_grad():
            ctx = self.batch_context(batch)

            trajectories = self.runner.eval_rollouts(
                policy=self.policy,
                context=ctx.graph,
                features=ctx.features,
            )

            metrics = evaluate_rollout_samples(
                trajectories=trajectories,
                batch=batch,
                context=ctx.graph,
                exclude_anchors_from_retrieved=self.evaluation.exclude_anchors_from_retrieved,
                use_reachable_targets=self.evaluation.use_reachable_targets,
                k_windows=self.evaluation.k_windows,
                enable_terminal_diagnostics=self.evaluation.enable_terminal_diagnostics,
            )

        batch_size = graph_batch_size(batch)

        self.log_scalar(
            f"{split}/num_trajectories",
            float(trajectories.num_trajectories),
            batch_size=batch_size,
        )
        self.log_scalars(
            prefix=split,
            values=metrics,
            batch_size=batch_size,
        )

    def batch_context(
        self,
        batch: RetrievalBatch,
    ) -> BatchContext:
        if self.validate_batch_coordinates:
            validate_retrieval_batch(batch)

        graph = GraphContext.from_batch(
            batch,
            validate=self.validate_batch_coordinates,
        )
        target = TargetContext.from_batch(
            batch=batch,
            graph_context=graph,
        )
        features = self.policy_feature_encoder(batch)

        return BatchContext(
            graph=graph,
            target=target,
            features=features,
        )

    def log_scalar(
        self,
        name: str,
        value: Scalar,
        *,
        batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        self.log(
            name,
            detach_scalar(value),
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=int(batch_size),
            sync_dist=True,
        )

    def log_scalars(
        self,
        *,
        prefix: str,
        values: Mapping[str, Scalar],
        batch_size: int,
    ) -> None:
        for name, value in values.items():
            self.log_scalar(
                f"{prefix}/{name}",
                value,
                batch_size=batch_size,
            )

    @staticmethod
    def _extract_loss(loss_out: Any) -> torch.Tensor:
        loss = getattr(loss_out, "loss", None)

        if loss is None:
            raise TypeError("loss_fn output must expose a `.loss` tensor.")

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"loss_fn output `.loss` must be a Tensor, got {type(loss)!r}.")

        if loss.ndim != 0:
            raise ValueError(f"loss_fn output `.loss` must be scalar, got shape {tuple(loss.shape)}.")

        return loss

    @staticmethod
    def _extract_metrics(loss_out: Any) -> Mapping[str, Scalar]:
        metrics = getattr(loss_out, "metrics", None)

        if metrics is None:
            return {}

        if not isinstance(metrics, Mapping):
            raise TypeError(f"loss_fn output `.metrics` must be a Mapping, got {type(metrics)!r}.")

        return metrics


def detach_scalar(value: Scalar) -> Scalar:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            value = value.float().mean()
        return value.detach()
    return float(value)


def graph_batch_size(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs_total)


__all__ = [
    "BatchContext",
    "StepOutput",
    "WeaverModule",
    "graph_batch_size",
]
