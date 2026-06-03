from __future__ import annotations

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from dataclasses import dataclass
from collections.abc import Mapping

from src.data.schema import RetrievalBatch
from src.eval.rollout import evaluate_rollout_samples
from src.training.logging import Scalar
from src.training.optimization import configure_optimization
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.feature import FeatureEncoder, FeaturePack
from src.weaver.policy import ForwardPolicy, PolicyCache
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.objectives import (
    ObjectiveOutput,
)
from src.weaver.objectives.subtb import prepare_subtb_batch, score_subtb_batch


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    graph: GraphContext
    features: FeaturePack
    cache: PolicyCache


@dataclass(frozen=True, slots=True)
class StepInputs(PolicyInputs):
    target: TargetContext
    replay: ReplayContext


class WeaverModule(LightningModule):
    def __init__(
        self,
        *,
        budget: int,
        hidden_dim: int = 1024,
        feature_encoder: FeatureEncoder,
        policy: ForwardPolicy,
        terminal_reward_model: torch.nn.Module,
        objective: torch.nn.Module,
        runner: RolloutRunner,
        optimization: DictConfig,
        evaluation: DictConfig,
        validate_batch_coordinates: bool = False,
    ) -> None:
        super().__init__()
        self.budget = budget
        self.hidden_dim = hidden_dim
        self.feature_encoder = feature_encoder
        self.policy = policy
        self.terminal_reward_model = terminal_reward_model
        self.objective = objective
        self.runner = runner
        self.optimization = optimization
        self.evaluation = evaluation
        self.validate_batch_coordinates = bool(validate_batch_coordinates)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return configure_optimization(
            modules=(
                self.feature_encoder,
                self.policy,
                self.objective,
            ),
            cfg=self.optimization,
            trainer=self.trainer,
        )

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        inputs = self._build_inputs(batch)
        with torch.no_grad():
            rollout = self.runner.train_rollouts(
                policy=self.policy,
                context=inputs.graph,
                target_context=inputs.target,
                replay_context=inputs.replay,
                features=inputs.features,
                cache=inputs.cache,
                budget=self.budget,
                global_step=int(self.global_step),
                replay_round=int(self.global_step),
            )
        output = self.objective(
            **self._build_objective_inputs(
                trajectories=rollout.trajectories,
                graph=inputs.graph,
                target=inputs.target,
                features=inputs.features,
                cache=inputs.cache,
            )
        )
        loss = output.require_loss()
        self._log_train(
            batch=batch,
            output=output,
            rollout=rollout,
        )
        return loss

    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._eval_step(
            split="val",
            batch=batch,
        )

    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._eval_step(
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
        inputs = self._build_policy_inputs(batch)
        with torch.no_grad():
            return self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
                cache=inputs.cache,
                budget=self.budget,
            )

    def _eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        inputs = self._build_policy_inputs(batch)
        with torch.no_grad():
            trajectories = self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
                cache=inputs.cache,
                budget=self.budget,
            )
            metrics = evaluate_rollout_samples(
                trajectories=trajectories,
                batch=batch,
                context=inputs.graph,
                exclude_anchors_from_retrieved=bool(self.evaluation.exclude_anchors_from_retrieved),
                use_reachable_targets=bool(self.evaluation.use_reachable_targets),
                k_windows=tuple(self.evaluation.k_windows),
                enable_terminal_diagnostics=bool(self.evaluation.enable_terminal_diagnostics),
            )
        self._log_eval(
            split=split,
            batch=batch,
            trajectories=trajectories,
            metrics=metrics,
        )

    def _build_inputs(self, batch: RetrievalBatch) -> StepInputs:
        policy_inputs = self._build_policy_inputs(batch)
        target = TargetContext.from_batch(
            batch=batch,
            graph_context=policy_inputs.graph,
            validate=self.validate_batch_coordinates,
        )
        replay = ReplayContext.from_batch(
            batch=batch,
            graph_context=policy_inputs.graph,
            target_context=target,
            validate=self.validate_batch_coordinates,
        )
        return StepInputs(
            graph=policy_inputs.graph,
            features=policy_inputs.features,
            cache=policy_inputs.cache,
            target=target,
            replay=replay,
        )

    def _build_policy_inputs(self, batch: RetrievalBatch) -> PolicyInputs:
        graph = GraphContext.from_batch(
            batch,
            validate=self.validate_batch_coordinates,
        )
        features = self.feature_encoder(batch)
        cache = self.policy.build_cache(features)
        return PolicyInputs(
            graph=graph,
            features=features,
            cache=cache,
        )

    def _log_train(
        self,
        *,
        batch: RetrievalBatch,
        output: ObjectiveOutput,
        rollout: TrainRolloutBatch,
    ) -> None:
        self.log(
            "train/loss",
            output.loss,
            batch_size=batch.num_graphs_total,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        objective_metrics = output.detached_metrics()
        step_residual_metrics = {
            f"train/{k}": float(v)
            for k, v in objective_metrics.items()
            if "residual_mean" in k
        }
        if step_residual_metrics:
            self.log_dict(
                step_residual_metrics,
                batch_size=batch.num_graphs_total,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        scalar_metrics = {
            f"train/{k}": float(v)
            for k, v in objective_metrics.items()
            if f"train/{k}" not in step_residual_metrics
        }
        scalar_metrics.update({f"train/rollout/{k}": float(v.detach()) for k, v in rollout.metrics.items()})
        self.log_dict(
            scalar_metrics,
            batch_size=batch.num_graphs_total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _build_objective_inputs(
        self,
        *,
        trajectories: TrajectoryBatch,
        graph: GraphContext,
        target: TargetContext,
        features: FeaturePack,
        cache: PolicyCache,
    ) -> dict[str, object]:
        prepared = prepare_subtb_batch(
            trajectories=trajectories,
            graph_context=graph,
            max_subtrajectory_length=getattr(self.objective, "max_subtrajectory_length", None),
        )
        scores = score_subtb_batch(
            batch=prepared,
            policy=self.policy,
            features=features,
            cache=cache,
            graph_context=graph,
        )
        reward = self.terminal_reward_model(
            state=prepared.states,
            target_context=target,
            graph_context=graph,
            active=scores.action_space.active,
        )
        return {"batch": prepared, "scores": scores, "reward": reward}

    def _log_eval(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
        trajectories: TrajectoryBatch,
        metrics: Mapping[str, Scalar],
    ) -> None:
        scalar_metrics: dict[str, float] = {
            f"{split}/num_trajectories": float(trajectories.num_trajectories),
        }
        scalar_metrics.update({f"{split}/{k}": float(v) for k, v in metrics.items()})
        self.log_dict(
            scalar_metrics,
            batch_size=batch.num_graphs_total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


__all__ = [
    "WeaverModule",
]
