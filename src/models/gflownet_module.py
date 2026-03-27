from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from src.models.configs import (
    ActionPriorConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SearchEvalConfig,
    SchedulerConfig,
)
from src.graph import TrajectoryBatch
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeFactoryProtocol
from src.utils.fit_schedule import ResolvedPassFitSchedule
from src.utils.logging_utils import get_logger, log_event, log_metric

from .evaluation_controller import (
    MetricRuntimeController,
    PredictionArtifactWriteConfig,
    PredictionLabel,
    PredictionResult,
)
from .gflownet.answer_supervision import compute_gold_entity_ranking_loss
from .gflownet import (
    ActionPriorScheduler,
    ForwardTrajectoryGFNSampler,
    SamplingTemperatureScheduler,
    SubTrajectoryBalanceLoss,
    SubTrajectoryBalanceLossOutput,
    SuccessReplayBuffer,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
)
from .gflownet.module_factory import GFlowNetPolicyFactory
from .gflownet.module_metrics import (
    build_train_metrics_payload,
    build_training_metrics,
    compute_root_diagnostics,
    compute_training_rollout_metrics,
    safe_batch_correlation,
)
from .gflownet.module_optim import (
    build_optimizer_and_scheduler,
    build_optimizer_param_groups,
    cfg_to_dict,
    use_zero_weight_decay,
)
from .gflownet.module_types import GFlowNetConfig, TrainingRolloutMetrics


logger = get_logger(__name__)


class GFlowNetModule(LightningModule):
    @staticmethod
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        return cfg_to_dict(cfg)

    @classmethod
    def _use_zero_weight_decay(
        cls, *, name: str, parameter: torch.nn.Parameter
    ) -> bool:
        del cls
        return use_zero_weight_decay(name=name, parameter=parameter)

    @classmethod
    def _build_optimizer_param_groups(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del cls
        return build_optimizer_param_groups(
            model_parameters=model_parameters,
            optimizer_cfg=optimizer_cfg,
        )

    @classmethod
    def _build_optimizer_and_scheduler(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        schedule_context: TrainingScheduleContext,
    ) -> dict[str, Any]:
        del cls
        return build_optimizer_and_scheduler(
            model_parameters=model_parameters,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            schedule_context=schedule_context,
        )

    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        policy_cfg: PolicyConfig,
        eval_cfg: SearchEvalConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        metric_runtime_factory: MetricRuntimeFactoryProtocol,
        action_prior_cfg: ActionPriorConfig | None = None,
    ) -> None:
        super().__init__()
        if action_prior_cfg is None:
            action_prior_cfg = ActionPriorConfig()
        self.cfg = GFlowNetConfig(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            action_prior_cfg=action_prior_cfg,
            policy_cfg=policy_cfg,
            eval_cfg=eval_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
        self.save_hyperparameters({"config": asdict(self.cfg)}, logger=False)
        self.policy = GFlowNetPolicyFactory.build_policy(
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            action_prior_cfg=action_prior_cfg,
            max_steps=horizon_cfg.max_steps,
        )
        self.metric_runtime_factory = metric_runtime_factory
        self.metric_runtime = metric_runtime_factory.build_runtime(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            eval_cfg=eval_cfg,
            policy=self.policy,
        )
        self.runtime_controller = MetricRuntimeController(
            metric_runtime=self.metric_runtime,
            metrics_profile=str(self.cfg.eval_cfg.metrics_profile),
            on_invalid_start=self._log_invalid_start,
        )
        self.sampler = self.runtime_controller.sampler
        self.loss_fn = SubTrajectoryBalanceLoss(
            config=training_cfg.subtb,
            answer_quotient_config=training_cfg.answer_quotient,
        )
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=training_cfg.sampling_temperature,
            config=training_cfg.sampling_temperature_schedule,
        )
        self.action_prior_scheduler = ActionPriorScheduler(
            base_scale=1.0,
            config=training_cfg.action_prior_schedule,
        )
        self.success_replay_buffer = SuccessReplayBuffer(
            config=training_cfg.success_replay
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self._schedule_context_override: TrainingScheduleContext | None = None
        self._invalid_start_count = 0
        self._latest_train_metrics: dict[str, float] | None = None

    @property
    def metrics_profile(self) -> str:
        return str(self.runtime_controller.metrics_profile)

    @property
    def evaluation_task(self) -> str:
        return str(self.cfg.eval_cfg.runtime_task)

    @property
    def predict_results(self) -> list[PredictionResult]:
        return self.runtime_controller.get_predict_results()

    @property
    def predict_labels(self) -> list[PredictionLabel]:
        return self.runtime_controller.get_predict_labels()

    @property
    def predict_metrics(self) -> dict[str, float]:
        return self.runtime_controller.get_predict_metrics()

    def reset_prediction_state(self) -> None:
        self.runtime_controller.reset_prediction_state()

    def reconfigure_evaluation(self, *, eval_cfg: SearchEvalConfig) -> None:
        self.cfg = GFlowNetConfig(
            horizon_cfg=self.cfg.horizon_cfg,
            training_cfg=self.cfg.training_cfg,
            action_prior_cfg=self.cfg.action_prior_cfg,
            policy_cfg=self.cfg.policy_cfg,
            eval_cfg=eval_cfg,
            optimizer_cfg=self.cfg.optimizer_cfg,
            scheduler_cfg=self.cfg.scheduler_cfg,
        )
        self.metric_runtime = self.metric_runtime_factory.build_runtime(
            horizon_cfg=self.cfg.horizon_cfg,
            training_cfg=self.cfg.training_cfg,
            eval_cfg=eval_cfg,
            policy=self.policy,
        )
        self.runtime_controller = MetricRuntimeController(
            metric_runtime=self.metric_runtime,
            metrics_profile=str(eval_cfg.metrics_profile),
            on_invalid_start=self._log_invalid_start,
        )
        self.sampler = self.runtime_controller.sampler
        self.search = self.runtime_controller.search
        self.reset_prediction_state()

    def replace_prediction_state(
        self,
        *,
        results: list[PredictionResult] | None = None,
        labels: list[PredictionLabel] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.runtime_controller.replace_prediction_state(
            results=results,
            labels=labels,
            metrics=metrics,
        )

    @staticmethod
    def _require_trajectory_batch(batch: object) -> TrajectoryBatch:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError(
                "GFlowNetModule expects TrajectoryBatch inputs from the datamodule."
            )
        return batch

    def set_fit_schedule(self, schedule: ResolvedPassFitSchedule) -> None:
        self._fit_schedule = schedule

    def pop_latest_train_metrics(self) -> dict[str, float] | None:
        metrics = self._latest_train_metrics
        self._latest_train_metrics = None
        return metrics

    def set_training_schedule_context(
        self, schedule_context: TrainingScheduleContext | None
    ) -> None:
        self._schedule_context_override = schedule_context

    def _resolve_effective_pass(self, *, after_current_step: bool) -> float | None:
        if self._fit_schedule is None:
            return None
        current_step = int(self.global_step)
        if after_current_step:
            current_step += 1
        return self._fit_schedule.effective_pass(global_step=current_step)

    def _compute_training_rollout_metrics(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> TrainingRolloutMetrics:
        return compute_training_rollout_metrics(
            batch=batch,
            sample_batch=sample_batch,
        )

    @staticmethod
    def _safe_batch_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        return safe_batch_correlation(x, y)

    def _compute_root_diagnostics(
        self,
        *,
        prepared_batch: Any,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> dict[str, float]:
        return compute_root_diagnostics(
            prepared_batch=prepared_batch,
            sample_batch=sample_batch,
        )

    def _build_training_metrics(
        self,
        *,
        online_loss_output: SubTrajectoryBalanceLossOutput,
        total_loss: torch.Tensor,
        online_direct_entity_ranking_loss: torch.Tensor,
        online_direct_gold_entity_mass: torch.Tensor,
        online_direct_entity_count: torch.Tensor,
        rollouts_per_graph: int,
        sampling_temperature: float,
        action_prior_scale: float,
        rollout_metrics: TrainingRolloutMetrics,
        root_diagnostics: dict[str, float],
        success_replay_effective_mix_alpha: float,
        success_replay_buffer_size: int,
        success_replay_ready: bool,
        success_replay_added: int,
        success_replay_sampled: int,
        replay_subtb_loss: torch.Tensor,
        replay_direct_entity_ranking_loss: torch.Tensor,
    ) -> dict[str, Any]:
        return build_training_metrics(
            cfg=self.cfg,
            online_loss_output=online_loss_output,
            total_loss=total_loss,
            online_direct_entity_ranking_loss=online_direct_entity_ranking_loss,
            online_direct_gold_entity_mass=online_direct_gold_entity_mass,
            online_direct_entity_count=online_direct_entity_count,
            rollouts_per_graph=rollouts_per_graph,
            sampling_temperature=sampling_temperature,
            action_prior_scale=action_prior_scale,
            rollout_metrics=rollout_metrics,
            root_diagnostics=root_diagnostics,
            success_replay_effective_mix_alpha=success_replay_effective_mix_alpha,
            success_replay_buffer_size=success_replay_buffer_size,
            success_replay_ready=success_replay_ready,
            success_replay_added=success_replay_added,
            success_replay_sampled=success_replay_sampled,
            replay_subtb_loss=replay_subtb_loss,
            replay_direct_entity_ranking_loss=replay_direct_entity_ranking_loss,
            resolve_effective_pass=lambda after_current_step: self._resolve_effective_pass(
                after_current_step=after_current_step
            ),
        )

    @staticmethod
    def _build_train_metrics_payload(metrics: dict[str, Any]) -> dict[str, float]:
        return build_train_metrics_payload(metrics)

    def _raise_on_nonfinite_training_loss(
        self,
        *,
        total_loss: torch.Tensor,
        batch: TrajectoryBatch,
    ) -> None:
        if torch.isfinite(total_loss).item():
            return
        sample_ids = [str(sample_id) for sample_id in batch.sample_ids]
        log_event(
            logger,
            "gflownet_non_finite_loss",
            level=logging.ERROR,
            dataset_scope=batch.dataset_scope,
            loss_value=float(total_loss.detach().item()),
            num_graphs=batch.num_graphs,
            sample_ids=sample_ids,
        )
        raise RuntimeError(
            "Non-finite training loss detected. Check SubTB and reward inputs."
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, TrajectoryBatch):
            return batch.to(device)
        raise TypeError(
            "GFlowNetModule expects TrajectoryBatch inputs from the datamodule during device transfer."
        )

    def _trainer_schedule_context(self) -> TrainingScheduleContext:
        if self._schedule_context_override is not None:
            return self._schedule_context_override
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return TrainingScheduleContext(estimated_stepping_batches=None)
        estimated_stepping_batches = None
        if trainer.estimated_stepping_batches is not None:
            estimated_stepping_batches = int(trainer.estimated_stepping_batches)
        trainer_max_steps = (
            int(trainer.max_steps) if int(trainer.max_steps) > 0 else None
        )
        trainer_max_epochs = (
            int(trainer.max_epochs) if int(trainer.max_epochs) > 0 else None
        )
        return TrainingScheduleContext(
            estimated_stepping_batches=estimated_stepping_batches,
            trainer_max_steps=trainer_max_steps,
            trainer_max_epochs=trainer_max_epochs,
        )

    def _resolve_sampling_temperature(self, *, global_step: int | None = None) -> float:
        trainer = getattr(self, "_trainer", None)
        current_step = 0 if trainer is None else int(trainer.global_step)
        if global_step is not None:
            current_step = int(global_step)
        return self.sampling_temperature_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_action_prior_scale(self, *, global_step: int | None = None) -> float:
        trainer = getattr(self, "_trainer", None)
        current_step = 0 if trainer is None else int(trainer.global_step)
        if global_step is not None:
            current_step = int(global_step)
        return self.action_prior_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _require_forward_sampler(self) -> ForwardTrajectoryGFNSampler:
        if not isinstance(self.sampler, ForwardTrajectoryGFNSampler):
            raise TypeError(
                "Success replay currently requires ForwardTrajectoryGFNSampler."
            )
        return self.sampler

    def _resolve_replay_trajectories_per_step(
        self,
        *,
        num_graphs: int,
        rollouts_per_graph: int,
    ) -> int:
        replay_cfg = self.cfg.training_cfg.success_replay
        if replay_cfg.replay_trajectories_per_step is not None:
            return int(replay_cfg.replay_trajectories_per_step)
        return int(num_graphs) * int(rollouts_per_graph)

    def _compute_direct_gold_entity_ranking_loss(
        self,
        *,
        batch: TrajectoryBatch,
        prepared_batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
        answer_cfg = self.cfg.training_cfg.answer_quotient
        if not answer_cfg.direct_entity_ranking_active:
            return zero, zero, zero

        num_nodes = int(prepared_batch.topology.num_nodes)
        if num_nodes == 0 or int(batch.answer_entity_ids.numel()) == 0:
            return zero, zero, zero

        base_policy = getattr(self.policy, "base_policy", None)
        if base_policy is None:
            base_policy = self.policy
        if not hasattr(base_policy, "build_local_state_features") or not hasattr(
            base_policy, "_compute_log_state_scores_from_flat_features"
        ):
            raise TypeError(
                "Direct gold-entity ranking supervision requires a base policy with local state scoring helpers."
            )
        flat_nodes = torch.arange(
            num_nodes, device=batch.node_ptr.device, dtype=torch.long
        )
        flat_num_steps = torch.zeros_like(flat_nodes, dtype=torch.long)
        flat_done_mask = torch.zeros_like(flat_nodes, dtype=torch.bool)
        flat_state_features = base_policy.build_local_state_features(
            prepared_batch,
            flat_nodes=flat_nodes,
            flat_num_steps=flat_num_steps,
            flat_done_mask=flat_done_mask,
        )
        graph_ids = prepared_batch.topology.graph_index_from_nodes(flat_nodes)
        entity_scores = base_policy._compute_log_state_scores_from_flat_features(
            prepared_batch=prepared_batch,
            flat_state_features=flat_state_features,
            graph_ids=graph_ids,
        )
        return compute_gold_entity_ranking_loss(
            graph_ids=graph_ids,
            entity_ids=batch.node_entity_ids.to(device=batch.node_ptr.device),
            entity_scores=entity_scores,
            answer_entity_ids=batch.answer_entity_ids.to(device=batch.node_ptr.device),
            answer_ptr=batch.answer_ptr.to(device=batch.node_ptr.device),
        )

    def _compute_replay_loss(
        self,
        *,
        num_graphs: int,
        rollouts_per_graph: int,
        device: torch.device,
    ) -> tuple[SubTrajectoryBalanceLossOutput | None, int, torch.Tensor]:
        replay_cfg = self.cfg.training_cfg.success_replay
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not replay_cfg.enabled or not self.success_replay_buffer.ready:
            return None, 0, zero
        replay_trajectories_per_step = self._resolve_replay_trajectories_per_step(
            num_graphs=num_graphs,
            rollouts_per_graph=rollouts_per_graph,
        )
        replay_batch = self.success_replay_buffer.sample_replay_batch(
            device=device,
            replay_trajectories_per_step=replay_trajectories_per_step,
        )
        if replay_batch is None:
            return None, 0, zero

        replay_prepared_batch = self.policy.prepare_batch(replay_batch.batch)
        replay_sampler = self._require_forward_sampler()
        replay_sample_batch = replay_sampler.rebuild_sample_batch(
            batch=replay_batch.batch,
            policy=self.policy,
            prepared_batch=replay_prepared_batch,
            start_nodes=replay_batch.start_nodes,
            planned_edge_ids=replay_batch.planned_edge_ids,
            planned_stop_mask=replay_batch.planned_stop_mask,
            path_lengths=replay_batch.path_lengths,
            termination_action_steps=replay_batch.termination_action_steps,
            trace_nodes=replay_batch.trace_nodes,
            trace_edge_ids=replay_batch.trace_edge_ids,
            trace_num_steps=replay_batch.trace_num_steps,
            trace_mask=replay_batch.trace_mask,
            trace_stop_mask=replay_batch.trace_stop_mask,
        )
        replay_loss_output = self.loss_fn.compute(replay_sample_batch)
        replay_direct_entity_ranking_loss, _, _ = (
            self._compute_direct_gold_entity_ranking_loss(
                batch=replay_batch.batch,
                prepared_batch=replay_prepared_batch,
            )
        )
        return (
            replay_loss_output,
            int(replay_batch.start_nodes.numel()),
            replay_direct_entity_ranking_loss,
        )

    @staticmethod
    def _mix_coverage_losses(
        *,
        online_loss: torch.Tensor,
        replay_loss: torch.Tensor | None,
        replay_mix_alpha: float,
    ) -> torch.Tensor:
        """Mix online proposal coverage with replay coverage under one objective."""

        if replay_loss is None:
            return online_loss
        alpha = float(replay_mix_alpha)
        return (1.0 - alpha) * online_loss + alpha * replay_loss

    def configure_optimizers(self) -> dict[str, Any]:
        schedule_context = self._trainer_schedule_context()
        return self._build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=self._cfg_to_dict(self.cfg.optimizer_cfg),
            scheduler_cfg=self._cfg_to_dict(self.cfg.scheduler_cfg),
            schedule_context=schedule_context,
        )

    def _log_metric_bundle(
        self,
        *,
        metrics: dict[str, Any],
        prefix: str,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
        prog_bar_key: str | None = None,
    ) -> None:
        for name, value in metrics.items():
            metric_value = (
                value.detach()
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=self.device)
            )
            key = f"{prefix}/{name}"
            log_metric(
                self,
                key,
                metric_value,
                batch_size=batch_size,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(key == prog_bar_key),
                sync_dist=on_epoch,
            )

    def _log_invalid_start(self, batch: TrajectoryBatch) -> None:
        self._invalid_start_count += 1
        log_event(
            logger,
            "gflownet_invalid_start_skipped",
            level=logging.WARNING,
            dataset_scope=batch.dataset_scope,
            invalid_start_count=self._invalid_start_count,
            num_graphs=batch.num_graphs,
            sample_ids=[str(sample_id) for sample_id in batch.sample_ids],
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        replay_source_batch = self._require_trajectory_batch(batch)
        if self.sampler is None:
            raise RuntimeError(
                "Current metric runtime does not define a training sampler; this model cannot train with the configured metric_runtime_factory."
            )
        prepared_batch = self.policy.prepare_batch(replay_source_batch)
        rollouts_per_graph = int(self.cfg.training_cfg.rollouts_per_graph)
        sampling_temperature = self._resolve_sampling_temperature()
        action_prior_scale = self._resolve_action_prior_scale()
        trajectory_batch = replay_source_batch.without_raw_features()
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=rollouts_per_graph,
            temperature=sampling_temperature,
            action_prior_scale=action_prior_scale,
        )
        online_loss_output = self.loss_fn.compute(sample_batch)
        (
            online_direct_entity_ranking_loss,
            online_direct_gold_entity_mass,
            online_direct_entity_count,
        ) = self._compute_direct_gold_entity_ranking_loss(
            batch=replay_source_batch,
            prepared_batch=prepared_batch,
        )
        direct_entity_weight = float(
            self.cfg.training_cfg.answer_quotient.direct_entity_ranking_weight
        )
        replay_cfg = self.cfg.training_cfg.success_replay
        replay_added = self.success_replay_buffer.add_successes(
            batch=replay_source_batch,
            sample_batch=sample_batch,
        )
        replay_ready = bool(replay_cfg.enabled and self.success_replay_buffer.ready)
        replay_loss_output = None
        replay_sampled = 0
        effective_replay_mix_alpha = 0.0
        replay_direct_entity_ranking_loss = torch.zeros(
            (), device=online_loss_output.loss.device, dtype=torch.float32
        )
        online_total_loss = online_loss_output.loss + (
            direct_entity_weight * online_direct_entity_ranking_loss
        )
        total_loss = online_total_loss
        if replay_ready:
            (
                replay_loss_output,
                replay_sampled,
                replay_direct_entity_ranking_loss,
            ) = self._compute_replay_loss(
                num_graphs=trajectory_batch.num_graphs,
                rollouts_per_graph=rollouts_per_graph,
                device=replay_source_batch.node_ptr.device,
            )
            if replay_loss_output is not None:
                effective_replay_mix_alpha = float(replay_cfg.mix_alpha)
                replay_total_loss = replay_loss_output.loss + (
                    direct_entity_weight * replay_direct_entity_ranking_loss
                )
                total_loss = self._mix_coverage_losses(
                    online_loss=online_total_loss,
                    replay_loss=replay_total_loss,
                    replay_mix_alpha=effective_replay_mix_alpha,
                )
        rollout_metrics = self._compute_training_rollout_metrics(
            batch=trajectory_batch,
            sample_batch=sample_batch,
        )
        root_diagnostics = self._compute_root_diagnostics(
            prepared_batch=prepared_batch,
            sample_batch=sample_batch,
        )
        self._raise_on_nonfinite_training_loss(
            total_loss=total_loss,
            batch=trajectory_batch,
        )
        replay_subtb_loss = torch.zeros(
            (), device=total_loss.device, dtype=torch.float32
        )
        if replay_loss_output is not None:
            replay_subtb_loss = replay_loss_output.subtb_loss
        metrics = self._build_training_metrics(
            online_loss_output=online_loss_output,
            total_loss=total_loss,
            online_direct_entity_ranking_loss=online_direct_entity_ranking_loss,
            online_direct_gold_entity_mass=online_direct_gold_entity_mass,
            online_direct_entity_count=online_direct_entity_count,
            rollouts_per_graph=rollouts_per_graph,
            sampling_temperature=sampling_temperature,
            action_prior_scale=action_prior_scale,
            rollout_metrics=rollout_metrics,
            root_diagnostics=root_diagnostics,
            success_replay_effective_mix_alpha=effective_replay_mix_alpha,
            success_replay_buffer_size=len(self.success_replay_buffer),
            success_replay_ready=replay_ready,
            success_replay_added=replay_added,
            success_replay_sampled=replay_sampled,
            replay_subtb_loss=replay_subtb_loss,
            replay_direct_entity_ranking_loss=replay_direct_entity_ranking_loss,
        )
        self._latest_train_metrics = self._build_train_metrics_payload(metrics)
        self._log_metric_bundle(
            metrics=metrics,
            prefix="train",
            batch_size=trajectory_batch.num_graphs,
            on_step=True,
            on_epoch=False,
            prog_bar_key="train/loss",
        )
        return total_loss

    def _evaluate_batch_output(
        self, *, batch: TrajectoryBatch
    ) -> MetricEvaluationOutput:
        return self.runtime_controller.evaluate_batch_output(
            batch=batch,
            include_answer_support=False,
        )

    def _evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
    ) -> tuple[
        dict[str, float],
        list[PredictionResult],
        dict[str, float],
        dict[str, float],
    ]:
        return self.runtime_controller.evaluate_batch(
            batch=batch,
            include_answer_support=False,
        )

    def _log_eval_outputs(
        self,
        *,
        stage: str,
        batch: TrajectoryBatch,
        outputs: MetricEvaluationOutput,
    ) -> None:
        prefix = f"{stage}/{batch.dataset_scope}"
        batch_size = int(batch.num_graphs)
        effective_pass = self._resolve_effective_pass(after_current_step=False)
        for metrics in (
            outputs.model_metrics,
            outputs.metrics,
            outputs.diagnostics,
        ):
            self._log_metric_bundle(
                metrics=metrics,
                prefix=prefix,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        if effective_pass is not None:
            self._log_metric_bundle(
                metrics={"effective_pass": effective_pass},
                prefix=prefix,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        outputs = self._evaluate_batch_output(batch=trajectory_batch)
        self._log_eval_outputs(stage="val", batch=trajectory_batch, outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        outputs = self._evaluate_batch_output(batch=trajectory_batch)
        self._log_eval_outputs(stage="test", batch=trajectory_batch, outputs=outputs)

    def on_predict_epoch_start(self) -> None:
        self.reset_prediction_state()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[PredictionResult]:
        del batch_idx, dataloader_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        return self.runtime_controller.predict_batch(
            batch=trajectory_batch,
        )

    def on_predict_batch_end(
        self,
        outputs: list[PredictionResult] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch_idx, dataloader_idx
        if outputs is None:
            return
        trajectory_batch = self._require_trajectory_batch(batch)
        self.runtime_controller.record_prediction_batch(
            batch=trajectory_batch,
            outputs=outputs,
        )

    def on_predict_epoch_end(self) -> None:
        self.runtime_controller.finalize_prediction_epoch()

    def get_predict_metrics(self) -> dict[str, float]:
        return self.runtime_controller.get_predict_metrics()

    def write_prediction_artifacts(
        self,
        *,
        write_config: PredictionArtifactWriteConfig | None = None,
        output_dir: str | Path | None = None,
        split: str | None = None,
        artifact_name: str | None = None,
        schema_version: int | None = None,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        overwrite: bool | None = None,
    ) -> dict[str, Path] | None:
        if write_config is not None:
            has_explicit_overrides = any(
                value is not None
                for value in (
                    output_dir,
                    split,
                    artifact_name,
                    schema_version,
                    entity_vocab_path,
                    relation_vocab_path,
                    questions_path,
                    overwrite,
                )
            )
            if has_explicit_overrides:
                raise ValueError(
                    "Provide either write_config or individual artifact arguments, not both."
                )
        else:
            if output_dir is None or split is None:
                raise ValueError(
                    "write_prediction_artifacts requires either write_config or both output_dir and split."
                )
            write_config = PredictionArtifactWriteConfig(
                output_dir=output_dir,
                split=split,
                artifact_name="rankflow" if artifact_name is None else artifact_name,
                schema_version=1 if schema_version is None else schema_version,
                entity_vocab_path=entity_vocab_path,
                relation_vocab_path=relation_vocab_path,
                questions_path=questions_path,
                overwrite=True if overwrite is None else overwrite,
            )
        return self.runtime_controller.write_prediction_artifacts(
            settings=write_config,
        )


__all__ = [
    "GFlowNetConfig",
    "GFlowNetModule",
    "PredictionArtifactWriteConfig",
]
