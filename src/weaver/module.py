from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torch.optim import Optimizer

from src.data.schema import RetrievalBatch
from src.eval.rollout import evaluate_rollout_samples
from src.training.logging import Scalar
from src.training.optimization import (
    build_lightning_scheduler_config,
    build_optimizer,
    build_scheduler,
    parse_optimization_config,
)
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.feature import FeatureEncoder, FeaturePack
from src.weaver.objectives import ObjectiveOutput
from src.weaver.objectives.subtb import (
    combine_subtb_scores,
    prepare_subtb_batch,
    score_backward_step_log_probs,
    score_forward_subtb_batch,
)
from src.weaver.policy import BackwardScoringModel, ForwardPolicy, PolicyInput
from src.weaver.reward import EvidenceStateScoreOutput
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch
from src.weaver.rollout.trajectory import TrajectoryBatch


@dataclass(frozen=True, slots=True)
class ForwardPolicyInputs:
    graph: GraphContext
    features: FeaturePack
    policy_input: PolicyInput


@dataclass(frozen=True, slots=True)
class StepContexts:
    graph: GraphContext
    target: TargetContext
    replay: ReplayContext


class WeaverModule(LightningModule):
    automatic_optimization = False

    def __init__(
        self,
        *,
        budget: int,
        hidden_dim: int = 1024,
        forward_feature_encoder: FeatureEncoder,
        backward_feature_encoder: FeatureEncoder,
        forward_policy: ForwardPolicy,
        backward_policy: BackwardScoringModel,
        reward_model: torch.nn.Module,
        objective: torch.nn.Module,
        runner: RolloutRunner,
        optimization: DictConfig,
        evaluation: DictConfig,
        validate_batch_coordinates: bool = False,
    ) -> None:
        super().__init__()
        self.budget = int(budget)
        self.hidden_dim = int(hidden_dim)
        self.forward_feature_encoder = forward_feature_encoder
        self.backward_feature_encoder = backward_feature_encoder
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.backward_target = copy.deepcopy(backward_policy)
        self.backward_target.requires_grad_(False)
        self.backward_target.eval()
        self.reward_model = reward_model
        self.objective = objective
        self.runner = runner
        self.optimization = optimization
        self.evaluation = evaluation
        self.validate_batch_coordinates = bool(validate_batch_coordinates)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        forward_spec = parse_optimization_config(self.optimization.forward)
        backward_spec = parse_optimization_config(self.optimization.backward)

        forward_optimizer = build_optimizer(
            modules=(
                self.forward_feature_encoder,
                self.forward_policy,
                self.reward_model,
                self.objective,
            ),
            cfg=forward_spec.optimizer,
        )
        backward_optimizer = build_optimizer(
            modules=(
                self.backward_feature_encoder,
                self.backward_policy,
            ),
            cfg=backward_spec.optimizer,
        )

        forward_scheduler = build_scheduler(
            optimizer=forward_optimizer,
            cfg=forward_spec.scheduler,
            trainer=self.trainer,
            base_lr=forward_spec.optimizer.lr,
        )
        backward_scheduler = build_scheduler(
            optimizer=backward_optimizer,
            cfg=backward_spec.scheduler,
            trainer=self.trainer,
            base_lr=backward_spec.optimizer.lr,
        )

        if forward_scheduler is None and backward_scheduler is None:
            return [forward_optimizer, backward_optimizer]

        schedulers: list[dict[str, object]] = []
        if forward_scheduler is not None and forward_spec.scheduler is not None:
            schedulers.append(
                build_lightning_scheduler_config(
                    scheduler=forward_scheduler,
                    interval=forward_spec.scheduler.interval,
                )
            )
        if backward_scheduler is not None and backward_spec.scheduler is not None:
            schedulers.append(
                build_lightning_scheduler_config(
                    scheduler=backward_scheduler,
                    interval=backward_spec.scheduler.interval,
                )
            )
        return [forward_optimizer, backward_optimizer], schedulers

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        contexts = self._build_step_contexts(batch)
        with torch.no_grad():
            rollout_inputs = self._build_forward_policy_inputs_from_graph(
                batch=batch,
                graph=contexts.graph,
            )
            rollout = self.runner.train_rollouts(
                policy=self.forward_policy,
                context=contexts.graph,
                target_context=contexts.target,
                replay_context=contexts.replay,
                features=rollout_inputs.features,
                policy_input=rollout_inputs.policy_input,
                budget=self.budget,
                global_step=_safe_global_step(self),
                replay_round=_safe_global_step(self),
            )

        prepared = prepare_subtb_batch(
            trajectories=rollout.trajectories,
            graph_context=contexts.graph,
        )

        forward_optimizer, backward_optimizer = self.optimizers()
        forward_scheduler, backward_scheduler = self._split_schedulers()

        backward_features = self.backward_feature_encoder(batch)
        tlm_loss, tlm_metrics = self._backward_tlm_step(
            prepared=prepared,
            graph=contexts.graph,
            features=backward_features,
            optimizer=backward_optimizer,
        )
        if backward_scheduler is not None:
            backward_scheduler.step()
        self._ema_update()

        forward_inputs = self._build_forward_policy_inputs_from_graph(
            batch=batch,
            graph=contexts.graph,
        )
        objective_output = self._forward_pf_step(
            prepared=prepared,
            graph=contexts.graph,
            target=contexts.target,
            batch=batch,
            features=forward_inputs.features,
            policy_input=forward_inputs.policy_input,
            optimizer=forward_optimizer,
        )
        if forward_scheduler is not None:
            forward_scheduler.step()

        self._log_train(
            batch=batch,
            output=objective_output,
            rollout=rollout,
            tlm_loss=tlm_loss,
            tlm_metrics=tlm_metrics,
        )

    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._eval_step(split="val", batch=batch)

    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._eval_step(split="test", batch=batch)

    def predict_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> TrajectoryBatch:
        del batch_idx, dataloader_idx
        with torch.no_grad():
            inputs = self._build_forward_policy_inputs(batch)
            return self.runner.eval_rollouts(
                policy=self.forward_policy,
                context=inputs.graph,
                features=inputs.features,
                policy_input=inputs.policy_input,
                budget=self.budget,
            )

    def _backward_tlm_step(
        self,
        *,
        prepared,
        graph: GraphContext,
        features: FeaturePack,
        optimizer: Optimizer,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        policy_mask = prepared.trajectories.is_policy.index_select(0, prepared.step_traj_ids)
        optimizer.zero_grad()
        if not bool(policy_mask.any()):
            zero = features.question_h.sum() * 0.0
            return zero.detach(), {"train/tlm_step_count": 0.0}
        batch_step_logp = score_backward_step_log_probs(
            batch=prepared,
            model=self.backward_policy,
            features=features,
            graph_context=graph,
        )
        selected = batch_step_logp[prepared.step_traj_ids[policy_mask], prepared.step_ids[policy_mask]]
        tlm_loss = -selected.mean()
        self.manual_backward(tlm_loss)
        optimizer.step()
        return tlm_loss.detach(), {
            "train/tlm_loss": float(tlm_loss.detach()),
            "train/tlm_step_count": float(selected.numel()),
            "train/tlm_log_prob_mean": float(selected.detach().mean()),
        }

    def _forward_pf_step(
        self,
        *,
        prepared,
        graph: GraphContext,
        target: TargetContext,
        batch: RetrievalBatch,
        features: FeaturePack,
        policy_input: PolicyInput,
        optimizer: Optimizer,
    ) -> ObjectiveOutput:
        reward_action_space = self.forward_policy.prepare_action_space(
            state=prepared.states,
            graph_context=graph,
            policy_input=policy_input,
            training=True,
        )
        reward = self.reward_model(
            state=prepared.states,
            target_context=target,
            graph_context=graph,
            active=reward_action_space.active,
        )
        forward_scores = score_forward_subtb_batch(
            batch=prepared,
            policy=self.forward_policy,
            features=features,
            policy_input=policy_input,
            graph_context=graph,
            reward=reward,
        )
        with torch.no_grad():
            target_features = self.backward_feature_encoder(batch)
            backward_step_logp = score_backward_step_log_probs(
                batch=prepared,
                model=self.backward_target,
                features=target_features,
                graph_context=graph,
            )
        scores = combine_subtb_scores(
            forward_scores=forward_scores,
            backward_step_log_prob=backward_step_logp,
        )
        path_gold_mask = _build_path_gold_mask(
            scores=scores,
            reward=reward,
            target=target,
            graph=graph,
        )
        output = self.objective(
            batch=prepared,
            scores=scores,
            reward=reward,
            global_step=_safe_global_step(self),
            path_gold_mask=path_gold_mask,
        )
        optimizer.zero_grad()
        self.manual_backward(output.require_loss())
        optimizer.step()
        return output

    def _split_schedulers(self) -> tuple[object | None, object | None]:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return None, None
        if isinstance(schedulers, list):
            forward_scheduler = schedulers[0] if len(schedulers) > 0 else None
            backward_scheduler = schedulers[1] if len(schedulers) > 1 else None
            return forward_scheduler, backward_scheduler
        return schedulers, None

    def _ema_update(self) -> None:
        decay = float(self.optimization.target_ema_decay)
        with torch.no_grad():
            for target_param, online_param in zip(self.backward_target.parameters(), self.backward_policy.parameters(), strict=True):
                target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
            for target_buffer, online_buffer in zip(self.backward_target.buffers(), self.backward_policy.buffers(), strict=True):
                target_buffer.copy_(online_buffer)

    def _eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        with torch.no_grad():
            inputs = self._build_forward_policy_inputs(batch)
            trajectories = self.runner.eval_rollouts(
                policy=self.forward_policy,
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
                    policy=self.forward_policy,
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
        self._log_eval(split=split, batch=batch, trajectories=trajectories, metrics=metrics)

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
        return StepContexts(graph=graph, target=target, replay=replay)

    def _build_forward_policy_inputs(self, batch: RetrievalBatch) -> ForwardPolicyInputs:
        graph = GraphContext.from_batch(
            batch,
            validate=self.validate_batch_coordinates,
        )
        return self._build_forward_policy_inputs_from_graph(batch=batch, graph=graph)

    def _build_forward_policy_inputs_from_graph(
        self,
        *,
        batch: RetrievalBatch,
        graph: GraphContext,
    ) -> ForwardPolicyInputs:
        features = self.forward_feature_encoder(batch)
        policy_input = self.forward_policy.build_policy_input(features, graph_context=graph)
        return ForwardPolicyInputs(graph=graph, features=features, policy_input=policy_input)

    def _log_train(
        self,
        *,
        batch: RetrievalBatch,
        output: ObjectiveOutput,
        rollout: TrainRolloutBatch,
        tlm_loss: torch.Tensor,
        tlm_metrics: Mapping[str, float],
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
        self.log(
            "train/tlm_loss",
            tlm_loss,
            batch_size=batch.num_graphs_total,
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
        scalar_metrics = {
            f"train/{k}": float(v)
            for k, v in objective_metrics.items()
            if f"train/{k}" not in step_residual_metrics
        }
        scalar_metrics.update({f"train/rollout/{k}": float(v.detach()) for k, v in rollout.metrics.items()})
        scalar_metrics.update({key: value for key, value in tlm_metrics.items() if key != "train/tlm_loss"})
        scalar_metrics["train/target_ema_decay"] = float(self.optimization.target_ema_decay)
        self.log_dict(
            scalar_metrics,
            batch_size=batch.num_graphs_total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

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
            return float(value.detach().item())
        if isinstance(value, (float, int)):
            return float(value)
    return None


def _safe_global_step(module: WeaverModule) -> int:
    try:
        return int(module.global_step)
    except Exception:
        return 0


def _build_path_gold_mask(
    *,
    scores,
    reward: EvidenceStateScoreOutput,
    target: TargetContext,
    graph: GraphContext,
) -> torch.Tensor | None:
    if scores.frontier_row_ids is None or scores.frontier_edge_ids is None:
        return None
    if int(scores.frontier_edge_ids.numel()) == 0:
        return torch.zeros((0,), dtype=torch.bool, device=reward.log_reward.device)

    edge_dst = graph.edge_index[1].index_select(0, scores.frontier_edge_ids)
    target_hits = target.target_mask.index_select(0, edge_dst)
    row_valid = reward.valid_target_mask.index_select(0, scores.frontier_row_ids)
    return target_hits & row_valid
