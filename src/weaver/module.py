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
from src.weaver.feature import FeatureEncoder, FeatureBank
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.objectives import (
    ObjectiveOutput,
    build_edge_flow_matching_batch,
    transition_source_counts,
)


@dataclass(frozen=True, slots=True)
class StepInputs:
    graph: GraphContext
    target: TargetContext
    replay: ReplayContext
    features: FeatureBank


class WeaverModule(LightningModule):
    def __init__(
        self,
        *,
        budget: int,
        hidden_dim: int = 1024,
        feature_encoder: FeatureEncoder,
        policy: ForwardPolicy,
        reward_model: torch.nn.Module,
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
        self.reward_model = reward_model
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
                self.reward_model,
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
                budget=self.budget,
            )
        efm_batch = build_edge_flow_matching_batch(
            policy_trajectories=rollout.trajectories,
            replay_transitions=rollout.replay_transitions,
            graph_context=inputs.graph,
        )

        output = self.objective(
            policy=self.policy,
            reward_model=self.reward_model,
            features=inputs.features,
            graph_context=inputs.graph,
            target_context=inputs.target,
            nonterminal=efm_batch.nonterminal,
            terminal=efm_batch.terminal,
        )
        reward_metrics = self._train_reward_metrics(
            target=inputs.target,
            terminal=efm_batch.terminal,
        )
        loss = output.require_loss()
        self._log_train(
            batch=batch,
            output=output,
            rollout=rollout,
            efm_metrics=transition_source_counts(batch=efm_batch),
            reward_metrics=reward_metrics,
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
        inputs = self._build_inputs(batch)
        with torch.no_grad():
            return self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
                budget=self.budget,
            )

    def _eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        inputs = self._build_inputs(batch)
        with torch.no_grad():
            trajectories = self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
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
        graph = GraphContext.from_batch(
            batch,
            validate=self.validate_batch_coordinates,
        )
        target = TargetContext.from_batch(
            batch=batch,
            graph_context=graph,
            validate=self.validate_batch_coordinates,
        )
        replay = ReplayContext.from_batch(
            batch=batch,
            graph_context=graph,
            target_context=target,
            validate=self.validate_batch_coordinates,
        )
        features = self.feature_encoder(batch)
        return StepInputs(
            graph=graph,
            target=target,
            replay=replay,
            features=features,
        )

    def _log_train(
        self,
        *,
        batch: RetrievalBatch,
        output: ObjectiveOutput,
        rollout: TrainRolloutBatch,
        efm_metrics: Mapping[str, float],
        reward_metrics: Mapping[str, float],
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
        scalar_metrics: dict[str, float] = {f"train/{k}": float(v) for k, v in output.detached_metrics().items()}
        scalar_metrics.update({f"train/rollout/{k}": float(v.detach()) for k, v in rollout.metrics.items()})
        scalar_metrics.update({f"train/{k}": float(v) for k, v in efm_metrics.items()})
        scalar_metrics.update({f"train/{k}": float(v) for k, v in reward_metrics.items()})
        self.log_dict(
            scalar_metrics,
            batch_size=batch.num_graphs_total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _train_reward_metrics(
        self,
        *,
        target: TargetContext,
        terminal,
    ) -> Mapping[str, float]:
        if terminal is None or terminal.num_transitions == 0:
            return {}
        with torch.no_grad():
            reward_output = self.reward_model(
                state=terminal.state,
                target_context=target,
            )
        return {
            str(name): float(value.detach())
            for name, value in reward_output.metrics.items()
        }

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
