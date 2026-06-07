from __future__ import annotations

import inspect
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
from src.weaver.policy import ForwardPolicy, PolicyInput
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.objectives import (
    ObjectiveOutput,
)
from src.weaver.objectives.subtb import prepare_subtb_batch, score_subtb_batch
from src.weaver.reward import EvidenceStateScoreOutput


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    graph: GraphContext
    features: FeaturePack
    policy_input: PolicyInput


@dataclass(frozen=True, slots=True)
class StepInputs(PolicyInputs):
    target: TargetContext
    replay: ReplayContext


@dataclass(frozen=True, slots=True)
class StepContexts:
    graph: GraphContext
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
        contexts = self._build_step_contexts(batch)
        with torch.no_grad():
            rollout_inputs = self._build_policy_inputs_from_graph(
                batch=batch,
                graph=contexts.graph,
            )
            rollout = self.runner.train_rollouts(
                policy=self.policy,
                context=contexts.graph,
                target_context=contexts.target,
                replay_context=contexts.replay,
                features=rollout_inputs.features,
                policy_input=rollout_inputs.policy_input,
                budget=self.budget,
                global_step=int(self.global_step),
                replay_round=int(self.global_step),
            )
        score_inputs = self._build_policy_inputs_from_graph(
            batch=batch,
            graph=contexts.graph,
        )
        output = self.objective(
            **self._build_objective_inputs(
                trajectories=rollout.trajectories,
                graph=contexts.graph,
                target=contexts.target,
                features=score_inputs.features,
                policy_input=score_inputs.policy_input,
            ),
            global_step=int(self.global_step),
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
        with torch.no_grad():
            inputs = self._build_policy_inputs(batch)
            return self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
                policy_input=inputs.policy_input,
                budget=self.budget,
            )

    def _eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        with torch.no_grad():
            inputs = self._build_policy_inputs(batch)
            trajectories = self.runner.eval_rollouts(
                policy=self.policy,
                context=inputs.graph,
                features=inputs.features,
                policy_input=inputs.policy_input,
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
            diversity_edge_penalty = float(getattr(self.evaluation, "diversity_edge_penalty", 0.0))
            if diversity_edge_penalty > 0.0:
                diverse_trajectories = self.runner.eval_rollouts(
                    policy=self.policy,
                    context=inputs.graph,
                    features=inputs.features,
                    policy_input=inputs.policy_input,
                    budget=self.budget,
                    diversity_edge_penalty=diversity_edge_penalty,
                )
                diverse_metrics = evaluate_rollout_samples(
                    trajectories=diverse_trajectories,
                    batch=batch,
                    context=inputs.graph,
                    exclude_anchors_from_retrieved=bool(self.evaluation.exclude_anchors_from_retrieved),
                    use_reachable_targets=bool(self.evaluation.use_reachable_targets),
                    k_windows=tuple(self.evaluation.k_windows),
                    enable_terminal_diagnostics=bool(self.evaluation.enable_terminal_diagnostics),
                )
                metrics.update({f"diverse_{key}": value for key, value in diverse_metrics.items()})
        self._log_eval(
            split=split,
            batch=batch,
            trajectories=trajectories,
            metrics=metrics,
        )

    def _build_inputs(self, batch: RetrievalBatch) -> StepInputs:
        contexts = self._build_step_contexts(batch)
        policy_inputs = self._build_policy_inputs_from_graph(
            batch=batch,
            graph=contexts.graph,
        )
        return StepInputs(
            graph=policy_inputs.graph,
            features=policy_inputs.features,
            policy_input=policy_inputs.policy_input,
            target=contexts.target,
            replay=contexts.replay,
        )

    def _build_step_contexts(self, batch: RetrievalBatch) -> StepContexts:
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
        return StepContexts(
            graph=graph,
            target=target,
            replay=replay,
        )

    def _build_policy_inputs(self, batch: RetrievalBatch) -> PolicyInputs:
        graph = GraphContext.from_batch(
            batch,
            validate=self.validate_batch_coordinates,
        )
        return self._build_policy_inputs_from_graph(
            batch=batch,
            graph=graph,
        )

    def _build_policy_inputs_from_graph(
        self,
        *,
        batch: RetrievalBatch,
        graph: GraphContext,
    ) -> PolicyInputs:
        features = self.feature_encoder(batch)
        policy_input = self.policy.build_policy_input(features, graph_context=graph)
        return PolicyInputs(
            graph=graph,
            features=features,
            policy_input=policy_input,
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
        step_residual_metrics = {f"train/{k}": float(v) for k, v in objective_metrics.items() if "residual_" in k}
        if step_residual_metrics:
            self.log_dict(
                step_residual_metrics,
                batch_size=batch.num_graphs_total,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        scalar_metrics = {f"train/{k}": float(v) for k, v in objective_metrics.items() if f"train/{k}" not in step_residual_metrics}
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
        policy_input: PolicyInput,
    ) -> dict[str, object]:
        prepared = prepare_subtb_batch(
            trajectories=trajectories,
            graph_context=graph,
        )
        action_space = self.policy.prepare_action_space(
            state=prepared.states,
            graph_context=graph,
            policy_input=policy_input,
            training=True,
        )
        reward = self.reward_model(
            state=prepared.states,
            target_context=target,
            graph_context=graph,
            active=action_space.active,
        )
        scores = score_subtb_batch(
            batch=prepared,
            policy=self.policy,
            features=features,
            policy_input=policy_input,
            graph_context=graph,
            reward=reward,
        )
        path_gold_mask = _build_path_gold_mask(
            scores=scores,
            reward=reward,
            target=target,
        )
        return {
            "batch": prepared,
            "scores": scores,
            "reward": reward,
            "path_gold_mask": path_gold_mask,
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
        if split == "val" and self.runner.replay_source is not None:
            self.log(
                "val/replay_fraction",
                float(self.runner.replay_source.current_fraction()),
                batch_size=batch.num_graphs_total,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_validation_end(self) -> None:
        replay_source = self.runner.replay_source
        if replay_source is None:
            return
        trainer = _attached_trainer(self)
        if trainer is None or bool(getattr(trainer, "sanity_checking", False)):
            return
        metric_name = str(replay_source.metric_name)
        metric_value = _lookup_metric(trainer, metric_name)
        if metric_value is None:
            return
        replay_source.update_from_validation(metric_value=metric_value)


__all__ = [
    "WeaverModule",
]


def _attached_trainer(module: WeaverModule):
    try:
        return module.trainer
    except RuntimeError:
        return None


def _lookup_metric(trainer, name: str) -> float | None:
    for source_name in ("callback_metrics", "logged_metrics"):
        metrics = getattr(trainer, source_name, None)
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if int(value.numel()) != 1:
                continue
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_path_gold_mask(
    *,
    scores,
    reward: EvidenceStateScoreOutput,
    target: TargetContext,
) -> torch.Tensor | None:
    if scores.frontier_row_ids is None or scores.frontier_edge_ids is None:
        return None
    if int(scores.frontier_edge_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.bool, device=target.device)
    state_unhit = ~reward.success_mask.detach().index_select(0, scores.frontier_row_ids)
    edge_gold = target.edge_on_shortest_path.index_select(0, scores.frontier_edge_ids)
    return state_unhit & edge_gold


def _build_policy_input(
    *,
    policy: torch.nn.Module,
    features: FeaturePack,
    graph: GraphContext,
) -> PolicyInput:
    build_policy_input = getattr(policy, "build_policy_input")
    signature = inspect.signature(build_policy_input)
    if "graph_context" in signature.parameters:
        return build_policy_input(features, graph_context=graph)
    return build_policy_input(features)
