from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.schema import RetrievalBatch
from src.training.config import EvalRuntimeConfig, OptimizationRuntimeConfig
from src.training.metrics import WeaverMetricSuite
from src.weaver.loss import LossOutput, ProbabilityDBLoss
from src.weaver.context import RewardContext
from src.weaver.nn.feature_encoder import FeatureBank, FeatureEncoder
from src.weaver.policy import Policy
from src.weaver.reward import EvidenceLogReward
from src.weaver.rollout.engine import RolloutContext
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.runner import RolloutChunk, RolloutRunner
from src.weaver.transitions import TransitionBatch

Scalar = torch.Tensor | float | int


class WeaverModule(LightningModule):
    def __init__(
        self,
        *,
        feature_encoder: FeatureEncoder,
        policy: Policy,
        reward_model: EvidenceLogReward,
        runner: RolloutRunner,
        optimization: OptimizationRuntimeConfig,
        evaluation: EvalRuntimeConfig,
        train_temperature: float = 1.0,
        eval_temperature: float = 1.0,
        gradient_clip_val: float | None = None,
    ) -> None:
        super().__init__()

        self.feature_encoder = feature_encoder
        self.policy = policy
        self.reward_model = reward_model
        self.db_loss = ProbabilityDBLoss()

        self.runner = runner
        self.optimization = optimization
        self.evaluation = evaluation
        self.metric_suite = WeaverMetricSuite(
            best_of_k=evaluation.best_of_k,
            exclude_anchors_from_retrieved=evaluation.exclude_anchors_from_retrieved,
            use_reachable_targets=evaluation.use_reachable_targets,
        )

        self.train_temperature = float(train_temperature)
        self.eval_temperature = float(eval_temperature)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)

        if self.train_temperature <= 0.0:
            raise ValueError(f"train_temperature must be positive, got {self.train_temperature}.")
        if self.eval_temperature <= 0.0:
            raise ValueError(f"eval_temperature must be positive, got {self.eval_temperature}.")
        if self.gradient_clip_val is not None and self.gradient_clip_val < 0.0:
            raise ValueError(f"gradient_clip_val must be non-negative, got {self.gradient_clip_val}.")

        self.runner.progress_fn = self._training_progress
        self.automatic_optimization = False
        self.save_hyperparameters(
            {
                "train_temperature": self.train_temperature,
                "eval_temperature": self.eval_temperature,
                "gradient_clip_val": self.gradient_clip_val,
            }
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.optimization.scheduler.interval,
            },
        }

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        optimizer = self._single_optimizer()
        optimizer.zero_grad(set_to_none=True)

        batch_size = _batch_size(batch)
        total_weight = 0
        num_chunks = 0
        weighted_loss_sum = torch.zeros((), device=self.device)
        metric_sums: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            rollout_features = self.feature_encoder(batch)
            rollout_context = self.runner.engine.prepare_context(
                batch=batch,
                features=rollout_features,
            )
            reward_context = self.reward_model.prepare_context(
                batch,
                expand_budget=self.runner.engine.expand_budget,
            )

        chunks = self.runner.train_chunks(
            policy=self.policy,
            batch=batch,
            context=rollout_context,
            temperature=self.train_temperature,
        )

        for chunk in chunks:
            if not _chunk_has_signal(chunk):
                continue
            output = self._forward_chunk(
                chunk=chunk,
                features=self.feature_encoder(batch),
                rollout_context=rollout_context,
                reward_context=reward_context,
            )
            weight = int(output.num_states)
            if weight <= 0:
                continue

            loss = _require_scalar_loss(output.loss)
            weighted_loss = loss * float(weight)
            self.manual_backward(weighted_loss)

            weighted_loss_sum = weighted_loss_sum + weighted_loss.detach()
            total_weight += weight
            num_chunks += 1
            _accumulate_metrics(
                metric_sums=metric_sums,
                metrics=output.metrics,
                weight=weight,
                device=self.device,
            )

        if total_weight <= 0 or num_chunks <= 0:
            raise RuntimeError(
                "No usable training signal was produced. "
                "Check rollout sampling, replay schedule, and objective construction."
            )

        _normalize_gradients(
            parameters=self.parameters(),
            denominator=total_weight,
        )
        if self.gradient_clip_val is not None and self.gradient_clip_val > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()
        self._step_scheduler(interval="step")

        mean_loss = weighted_loss_sum / float(total_weight)
        self.log(
            "train/loss",
            mean_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/signal/transitions_per_graph",
            torch.tensor(float(total_weight) / float(batch_size), device=self.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_averaged_scalars(
            prefix="train",
            metric_sums=metric_sums,
            total_weight=total_weight,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": mean_loss}

    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._rollout_eval_step(split="val", batch=batch)

    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._rollout_eval_step(split="test", batch=batch)

    def predict_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[RolloutResult, ...]:
        del batch_idx, dataloader_idx
        with torch.no_grad():
            features = self.feature_encoder(batch)
            rollout_context = self.runner.engine.prepare_context(
                batch=batch,
                features=features,
            )
            return self.runner.eval_rollouts(
                policy=self.policy,
                batch=batch,
                context=rollout_context,
                temperature=self.eval_temperature,
            )

    def on_train_epoch_end(self) -> None:
        self._step_scheduler(interval="epoch")

    def _rollout_eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        batch_size = _batch_size(batch)

        with torch.no_grad():
            features = self.feature_encoder(batch)
            rollout_context = self.runner.engine.prepare_context(
                batch=batch,
                features=features,
            )
            rollouts = self.runner.eval_rollouts(
                policy=self.policy,
                batch=batch,
                context=rollout_context,
                temperature=self.eval_temperature,
            )
            output_metrics = self.metric_suite.eval_metrics(
                rollout_samples=rollouts,
                batch=batch,
                stage="",
            )

        self.log(
            f"{split}/num_rollouts",
            float(len(rollouts)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_scalars(
            prefix=split,
            values=output_metrics,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

    def _single_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self.optimizers(use_pl_optimizer=False)
        if isinstance(optimizer, list):
            if len(optimizer) != 1:
                raise RuntimeError(f"WeaverModule expects exactly one optimizer, got {len(optimizer)}.")
            optimizer = optimizer[0]
        return optimizer

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cfg = self.optimization.optimizer
        if optimizer_cfg.type != "adamw":
            raise ValueError(f"Unsupported optimizer type {optimizer_cfg.type!r}.")
        if optimizer_cfg.no_decay_on_bias_and_norm:
            parameters = _parameter_groups_without_decay(
                self,
                weight_decay=optimizer_cfg.weight_decay,
            )
        else:
            parameters = self.parameters()
        return AdamW(
            parameters,
            lr=optimizer_cfg.lr,
            betas=optimizer_cfg.betas,
            weight_decay=optimizer_cfg.weight_decay,
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        scheduler_cfg = self.optimization.scheduler
        if scheduler_cfg is None:
            return None
        if scheduler_cfg.type != "cosine":
            raise ValueError(f"Unsupported scheduler type {scheduler_cfg.type!r}.")

        horizon = _resolve_scheduler_horizon(
            trainer=self.trainer,
            interval=scheduler_cfg.interval,
        )
        if horizon <= 0:
            raise RuntimeError(
                "Could not resolve a positive scheduler horizon. "
                "Set Trainer.max_steps or Trainer.max_epochs correctly."
            )

        warmup_steps = int(float(horizon) * scheduler_cfg.warmup_ratio)
        cosine_steps = max(1, horizon - warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=scheduler_cfg.eta_min,
        )
        if warmup_steps <= 0:
            return cosine
        warmup = LinearLR(
            optimizer,
            start_factor=1.0e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _step_scheduler(
        self,
        *,
        interval: str,
    ) -> None:
        scheduler_cfg = self.optimization.scheduler
        if scheduler_cfg is None or scheduler_cfg.interval != interval:
            return
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, list):
            if len(scheduler) != 1:
                raise RuntimeError(f"WeaverModule expects exactly one scheduler, got {len(scheduler)}.")
            scheduler = scheduler[0]
        scheduler.step()

    def _training_progress(self) -> float:
        try:
            trainer = self.trainer
        except RuntimeError:
            return 0.0
        horizon = _resolve_scheduler_horizon(trainer=trainer, interval="step")
        if horizon <= 0:
            return 0.0
        step = int(getattr(trainer, "global_step", 0))
        return min(1.0, max(0.0, float(step) / float(horizon)))

    def _log_scalars(
        self,
        *,
        prefix: str,
        values: Mapping[str, Scalar],
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in values.items():
            scalar = _to_scalar_tensor(name=name, value=value, device=self.device)
            self.log(
                f"{prefix}/{name}",
                scalar,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _log_averaged_scalars(
        self,
        *,
        prefix: str,
        metric_sums: Mapping[str, torch.Tensor],
        total_weight: int,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        if total_weight <= 0:
            return
        values = {name: value / float(total_weight) for name, value in metric_sums.items()}
        self._log_scalars(
            prefix=prefix,
            values=values,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    def _forward_chunk(
        self,
        *,
        chunk: RolloutChunk,
        features: FeatureBank,
        rollout_context: RolloutContext,
        reward_context: RewardContext,
    ) -> LossOutput:
        transitions = chunk.transitions
        if transitions is None or transitions.num_transitions <= 0:
            zero = torch.zeros((), device=features.edge_h.device)
            return LossOutput(
                loss=zero,
                metrics={},
                num_states=0,
                per_unit_loss=None,
            )
        return self._forward_transitions(
            transitions=transitions,
            features=features,
            rollout_context=rollout_context,
            reward_context=reward_context,
        )

    def _forward_transitions(
        self,
        *,
        transitions: TransitionBatch,
        features: FeatureBank,
        rollout_context: RolloutContext,
        reward_context: RewardContext,
    ) -> LossOutput:
        parent_out = self.policy(
            context=rollout_context.graph_context,
            state=transitions.parent_state,
            features=features,
            frontier_builder=rollout_context.frontier_builder,
        )
        child_out = self.policy(
            context=rollout_context.graph_context,
            state=transitions.child_state,
            features=features,
            frontier_builder=rollout_context.frontier_builder,
        )

        selected_positions = _match_transition_actions(
            row_ids=parent_out.frontier.row_ids,
            edge_ids=parent_out.frontier.edge_ids,
            action_edge_ids=transitions.action_edge_ids,
            device=parent_out.stop_log_prob.device,
        )
        parent_edge_log_prob = parent_out.edge_log_prob.index_select(0, selected_positions)

        parent_reward = self.reward_model(
            state=transitions.parent_state,
            context=reward_context,
        )
        child_reward = self.reward_model(
            state=transitions.child_state,
            context=reward_context,
        )

        output = self.db_loss(
            parent_log_reward=parent_reward.log_reward,
            child_log_reward=child_reward.log_reward,
            log_backward_prob=transitions.log_backward_prob,
            parent_stop_log_prob=parent_out.stop_log_prob,
            parent_continue_log_prob=parent_out.continue_log_prob,
            parent_edge_log_prob=parent_edge_log_prob,
            child_stop_log_prob=child_out.stop_log_prob,
        )
        metrics = dict(output.metrics)
        before_hit_rate = (~parent_reward.fail_penalty).float().mean().detach()
        after_hit_rate = (~child_reward.fail_penalty).float().mean().detach()
        metrics.update(
            {
                "reward/hit_rate_after_action": after_hit_rate,
                "reward/hit_rate_delta": after_hit_rate - before_hit_rate,
                "policy/stop_prob_before_action": parent_out.stop_log_prob.exp().float().mean().detach(),
                "policy/selected_edge_prob": parent_edge_log_prob.exp().float().mean().detach(),
            }
        )
        return LossOutput(
            loss=output.loss,
            metrics=metrics,
            num_states=output.num_states,
            per_unit_loss=output.per_unit_loss,
        )


def _chunk_has_signal(chunk: RolloutChunk) -> bool:
    return bool(chunk.has_rollouts or chunk.has_replay)


def _require_scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"objective.loss must be a Tensor, got {type(loss).__name__}.")
    if loss.ndim != 0:
        raise ValueError(f"objective.loss must be scalar, got shape {tuple(loss.shape)}.")
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(f"objective.loss is not finite: {float(loss.detach())}.")
    if not loss.requires_grad:
        raise RuntimeError(
            "objective.loss does not require grad. "
            "The objective must recompute policy scores from sampled states."
        )
    return loss


def _accumulate_metrics(
    *,
    metric_sums: dict[str, torch.Tensor],
    metrics: Mapping[str, Scalar],
    weight: int,
    device: torch.device,
) -> None:
    for name, value in metrics.items():
        scalar = _to_scalar_tensor(name=name, value=value, device=device)
        if name not in metric_sums:
            metric_sums[name] = torch.zeros((), device=device)
        metric_sums[name] = metric_sums[name] + scalar.detach() * float(weight)


def _to_scalar_tensor(
    *,
    name: str,
    value: Scalar,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim != 0:
            raise ValueError(f"Logged value {name!r} must be scalar, got shape {tuple(tensor.shape)}.")
        return tensor.to(device=device)
    if isinstance(value, (float, int)):
        return torch.tensor(float(value), device=device)
    raise TypeError(
        f"Logged value {name!r} must be Tensor, float, or int, "
        f"got {type(value).__name__}."
    )


def _normalize_gradients(
    *,
    parameters: Iterator[torch.nn.Parameter],
    denominator: int,
) -> None:
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}.")
    scale = float(denominator)
    for param in parameters:
        if param.grad is not None:
            param.grad.div_(scale)


def _batch_size(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs_total)


def _parameter_groups_without_decay(
    module: torch.nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict[str, object]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def _resolve_scheduler_horizon(
    *,
    trainer: object,
    interval: str,
) -> int:
    if interval == "step":
        max_steps = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return max_steps
        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, int) and estimated > 0:
            return estimated
        return 0
    if interval == "epoch":
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(max_epochs, int) and max_epochs > 0:
            return max_epochs
        return 0
    raise ValueError(f"Unsupported scheduler interval {interval!r}.")


__all__ = [
    "WeaverModule",
]


def _match_transition_actions(
    *,
    row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    action_edge_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    row_ids = row_ids.to(device=device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=device, dtype=torch.long).view(-1)
    target_rows = torch.arange(
        action_edge_ids.numel(),
        device=device,
        dtype=torch.long,
    )
    target_edges = action_edge_ids.to(device=device, dtype=torch.long).view(-1)
    if row_ids.numel() == 0:
        raise RuntimeError("Parent policy frontier is empty for transition batch.")
    selected = row_ids.eq(target_rows.unsqueeze(1)) & edge_ids.eq(target_edges.unsqueeze(1))
    if not bool(selected.any(dim=1).all()):
        raise RuntimeError("Transition action missing from parent frontier.")
    return selected.float().argmax(dim=1).to(dtype=torch.long)
