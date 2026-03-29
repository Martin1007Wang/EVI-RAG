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
from src.models.configs.policy import SUBGRAPH_STATE_MODE
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
from .gflownet.schedules import (
    ActionPriorScheduler,
    SamplingTemperatureScheduler,
    TrainingScheduleContext,
    TransitionBiasScheduler,
)
from .gflownet.subgraph.losses import (
    SubgraphSubTrajectoryBalanceLoss,
    SubgraphSubTrajectoryBalanceLossOutput,
)
from .gflownet.subgraph.sampler import (
    SubgraphSampler,
    SubgraphTrajectorySampleBatch,
)
from .gflownet.module_factory import GFlowNetPolicyFactory
from .gflownet.module_optim import (
    build_optimizer_and_scheduler,
    build_optimizer_param_groups,
    cfg_to_dict,
    use_zero_weight_decay,
)
from .gflownet.module_types import GFlowNetConfig


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
        self._validate_subgraph_only_config(
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            eval_cfg=eval_cfg,
        )
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
            report_profile=str(self.cfg.eval_cfg.report_profile),
            on_invalid_start=self._log_invalid_start,
        )
        self.sampler = self.runtime_controller.sampler
        self.loss_fn = SubgraphSubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=training_cfg.sampling_temperature,
            config=training_cfg.sampling_temperature_schedule,
        )
        self.action_prior_scheduler = ActionPriorScheduler(
            base_scale=1.0,
            config=training_cfg.action_prior_schedule,
        )
        self.transition_bias_scheduler = TransitionBiasScheduler(
            base_scale=1.0,
            config=training_cfg.transition_bias_schedule,
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self._schedule_context_override: TrainingScheduleContext | None = None
        self._invalid_start_count = 0
        self._latest_train_metrics: dict[str, float] | None = None

    @staticmethod
    def _validate_subgraph_only_config(
        *,
        policy_cfg: PolicyConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
    ) -> None:
        if str(policy_cfg.state_mode) != SUBGRAPH_STATE_MODE:
            raise ValueError(
                "GFlowNetModule supports only policy.state_mode='subgraph'."
            )
        if training_cfg.success_replay.enabled:
            raise ValueError(
                "Subgraph mode does not support success replay yet; set training.success_replay.mix_alpha=0.0."
            )
        if training_cfg.answer_quotient.active:
            raise ValueError(
                "Subgraph mode does not support answer_quotient yet; disable training.answer_quotient."
            )
        if training_cfg.answer_quotient.direct_entity_ranking_active:
            raise ValueError(
                "Subgraph mode does not support direct entity ranking yet."
            )
        if training_cfg.potential_reward.active:
            raise ValueError(
                "Subgraph mode does not support legacy potential_reward shaping; use training.subgraph_reward instead."
            )
        if eval_cfg.runtime_task != "answer_search":
            raise ValueError(
                "Subgraph mode currently supports only answer_search evaluation."
            )

    @property
    def report_profile(self) -> str:
        return str(self.runtime_controller.report_profile)

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
        self._validate_subgraph_only_config(
            policy_cfg=self.cfg.policy_cfg,
            training_cfg=self.cfg.training_cfg,
            eval_cfg=eval_cfg,
        )
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
            report_profile=str(eval_cfg.report_profile),
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

    @staticmethod
    def _build_train_metrics_payload(metrics: dict[str, Any]) -> dict[str, float]:
        payload: dict[str, float] = {}
        for name, value in metrics.items():
            if torch.is_tensor(value):
                scalar = float(value.detach().to(dtype=torch.float32).item())
            else:
                scalar = float(value)
            payload[f"train/{name}"] = scalar
        return payload

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
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.sampling_temperature_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_action_prior_scale(self, *, global_step: int | None = None) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.action_prior_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_transition_bias_scale(
        self, *, global_step: int | None = None
    ) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.transition_bias_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_schedule_step(self, *, global_step: int | None = None) -> int:
        trainer = getattr(self, "_trainer", None)
        current_step = 0 if trainer is None else int(trainer.global_step)
        if global_step is not None:
            current_step = int(global_step)
        return current_step

    def _require_subgraph_sampler(self) -> SubgraphSampler:
        if not isinstance(self.sampler, SubgraphSampler):
            raise TypeError("Subgraph training requires SubgraphSampler.")
        return self.sampler

    def _build_subgraph_training_metrics(
        self,
        *,
        loss_output: SubgraphSubTrajectoryBalanceLossOutput,
        sample_batch: SubgraphTrajectorySampleBatch,
        total_loss: torch.Tensor,
        rollouts_per_graph: int,
        sampling_temperature: float,
        action_prior_scale: float,
        transition_bias_scale: float,
    ) -> dict[str, Any]:
        mean_selected_edges = (
            (sample_batch.chosen_edge_ids >= 0)
            .to(dtype=torch.float32)
            .sum(dim=-1)
            .mean()
        )
        mean_termination_action_step = sample_batch.termination_action_steps.to(
            dtype=torch.float32
        ).mean()
        return {
            "loss": total_loss.detach(),
            "actor_loss": total_loss.detach(),
            "subtb_loss": loss_output.subtb_loss.detach(),
            "subtb_residual": loss_output.residual_abs.detach(),
            "subtb_residual_variance_per_batch": loss_output.residual_variance.detach(),
            "subtb_root": loss_output.root_abs.detach(),
            "rollout_success": loss_output.success_rate.detach(),
            "terminal_answer_count": loss_output.average_terminal_answer_count.detach(),
            "terminal_component_count": loss_output.average_terminal_component_count.detach(),
            "log_z_mean": loss_output.log_z_mean.detach(),
            "log_z_variance": loss_output.log_z_variance.detach(),
            "mean_selected_edges": mean_selected_edges.detach(),
            "mean_termination_action_step": mean_termination_action_step.detach(),
            "rollouts_per_graph": float(rollouts_per_graph),
            "sampling_temperature": float(sampling_temperature),
            "proposal_action_prior_scale": float(action_prior_scale),
            "proposal_transition_bias_scale": float(transition_bias_scale),
            "subgraph_reward_c_step": float(
                self.cfg.training_cfg.subgraph_reward.c_step
            ),
            "subgraph_reward_lambda_conn": float(
                self.cfg.training_cfg.subgraph_reward.lambda_conn
            ),
            "subgraph_reward_beta_hit": float(
                self.cfg.training_cfg.subgraph_reward.beta_hit
            ),
            "subgraph_reward_beta_cnt": float(
                self.cfg.training_cfg.subgraph_reward.beta_cnt
            ),
            "subgraph_reward_beta_early": float(
                self.cfg.training_cfg.subgraph_reward.beta_early
            ),
        }

    def _training_step_subgraph(self, batch: TrajectoryBatch) -> torch.Tensor:
        sampler = self._require_subgraph_sampler()
        prepared_batch = self.policy.prepare_batch(batch)
        rollouts_per_graph = int(self.cfg.training_cfg.rollouts_per_graph)
        sampling_temperature = self._resolve_sampling_temperature()
        action_prior_scale = self._resolve_action_prior_scale()
        transition_bias_scale = self._resolve_transition_bias_scale()
        trajectory_batch = batch.without_raw_features()
        sample_batch = sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=rollouts_per_graph,
            temperature=sampling_temperature,
            action_prior_scale=action_prior_scale,
            transition_bias_scale=transition_bias_scale,
        )
        loss_output = self.loss_fn.compute(sample_batch)
        total_loss = loss_output.loss
        self._raise_on_nonfinite_training_loss(
            total_loss=total_loss,
            batch=trajectory_batch,
        )
        metrics = self._build_subgraph_training_metrics(
            loss_output=loss_output,
            sample_batch=sample_batch,
            total_loss=total_loss,
            rollouts_per_graph=rollouts_per_graph,
            sampling_temperature=sampling_temperature,
            action_prior_scale=action_prior_scale,
            transition_bias_scale=transition_bias_scale,
        )
        effective_pass = self._resolve_effective_pass(after_current_step=True)
        if effective_pass is not None:
            metrics["effective_pass"] = float(effective_pass)
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
        return self._training_step_subgraph(self._require_trajectory_batch(batch))

    def _evaluate_batch_output(
        self, *, batch: TrajectoryBatch
    ) -> MetricEvaluationOutput:
        return self.runtime_controller.evaluate_batch_output(
            batch=batch,
            include_answer_support=self.cfg.eval_cfg.include_answer_support,
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
            include_answer_support=self.cfg.eval_cfg.include_answer_support,
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
