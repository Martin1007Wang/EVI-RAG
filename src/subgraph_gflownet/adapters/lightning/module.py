from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from omegaconf import OmegaConf
from src.graph import TrajectoryBatch
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeFactoryProtocol
from src.metrics.search_eval_utils import (
    RUNTIME_ANSWER_TASK,
    normalize_search_eval_cfg,
    search_eval_include_answer_support,
    search_eval_runtime_task,
)
from ...core.optimization import build_optimizer_and_scheduler
from ...core.schedules import (
    ReplayMixScheduler,
    SamplingTemperatureScheduler,
    TrainingScheduleContext,
)
from src.runs.fit_schedule import ResolvedPassFitSchedule
from src.utils.logging_utils import get_logger, log_event, log_metric

from ...application.config.common import to_plain_mapping
from ...application.training.orchestrator import SubgraphTrainingOrchestrator
from ...application.training.schedules import resolve_supervision_phase
from ...core.config_utils import normalize_training_cfg
from ...core.losses import SubgraphDetailedBalanceLoss
from ...core.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from ...core.replay import SubgraphSuccessReplayBuffer
from ...core.sampler import SubgraphSampler
from .prediction_state import (
    MetricRuntimeController,
    PredictionArtifactWriteConfig,
    PredictionLabel,
    PredictionResult,
)


logger = get_logger(__name__)


class GFlowNetModule(LightningModule):
    def __init__(
        self,
        *,
        horizon_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        policy_cfg: dict[str, Any],
        eval_cfg: dict[str, Any],
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        metric_runtime_factory: MetricRuntimeFactoryProtocol,
    ) -> None:
        super().__init__()
        horizon_cfg = to_plain_mapping(horizon_cfg, field_name="horizon_cfg")
        training_cfg = normalize_training_cfg(training_cfg)
        policy_cfg = to_plain_mapping(policy_cfg, field_name="policy_cfg")
        eval_cfg = normalize_search_eval_cfg(eval_cfg)
        optimizer_cfg = to_plain_mapping(optimizer_cfg, field_name="optimizer_cfg")
        scheduler_cfg = to_plain_mapping(scheduler_cfg, field_name="scheduler_cfg")
        self.cfg = OmegaConf.create(
            {
                "horizon_cfg": horizon_cfg,
                "training_cfg": training_cfg,
                "policy_cfg": policy_cfg,
                "eval_cfg": eval_cfg,
                "optimizer_cfg": optimizer_cfg,
                "scheduler_cfg": scheduler_cfg,
            }
        )
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(self.cfg, resolve=True)},
            logger=False,
        )
        self._validate_subgraph_only_config(
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            eval_cfg=eval_cfg,
        )
        self.policy = SubgraphPolicy(
            state_mode=str(policy_cfg["state_mode"]),
            backbone=dict(policy_cfg["backbone"]),
            flow_head=dict(policy_cfg["flow_head"]),
            state_encoder=dict(policy_cfg["state_encoder"]),
            actor=dict(policy_cfg["actor"]),
            answer_reward=dict(training_cfg["answer_reward"]),
            proposal_prior=dict(training_cfg["auxiliary"]["proposal"]["prior"]),
            max_steps=int(horizon_cfg["max_steps"]),
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
            report_profile=str(self.cfg.eval_cfg["report_profile"]),
            on_invalid_start=self._log_invalid_start,
        )
        self.sampler = SubgraphSampler(max_steps=int(horizon_cfg["max_steps"]))
        self.loss_fn = SubgraphDetailedBalanceLoss(
            **dict(training_cfg["detailed_balance"])
        )
        proposal_aux_cfg = dict(training_cfg["auxiliary"]["proposal"])
        replay_aux_cfg = dict(training_cfg["auxiliary"]["replay"])
        replay_buffer_cfg = dict(replay_aux_cfg["buffer"])
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=float(training_cfg["sampling_temperature"]),
            **dict(training_cfg["sampling_temperature_schedule"]),
        )
        self.replay_mix_scheduler = ReplayMixScheduler(
            base_alpha=(
                float(replay_aux_cfg.get("mix_alpha", 0.0))
                if bool(replay_aux_cfg.get("enabled", False))
                else 0.0
            ),
            **dict(replay_aux_cfg["schedule"]),
        )
        self.success_replay_buffer = SubgraphSuccessReplayBuffer(
            capacity=int(replay_buffer_cfg.get("capacity", 1024)),
            deduplicate=bool(replay_buffer_cfg.get("deduplicate", True)),
        )
        self.training_orchestrator = SubgraphTrainingOrchestrator(
            cfg=self.cfg,
            policy=self.policy,
            sampler=self.sampler,
            loss_fn=self.loss_fn,
            success_replay_buffer=self.success_replay_buffer,
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self._schedule_context_override: TrainingScheduleContext | None = None
        self._invalid_start_count = 0
        self._latest_train_metrics: dict[str, float] | None = None

    @staticmethod
    def _validate_subgraph_only_config(
        *,
        policy_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        eval_cfg: dict[str, Any],
    ) -> None:
        del training_cfg
        if str(policy_cfg["state_mode"]) != SUBGRAPH_STATE_MODE:
            raise ValueError(
                "GFlowNetModule supports only policy.state_mode='subgraph'."
            )
        if search_eval_runtime_task(eval_cfg) != RUNTIME_ANSWER_TASK:
            raise ValueError(
                "Subgraph mode currently supports only answer_search evaluation."
            )

    @property
    def report_profile(self) -> str:
        return str(self.runtime_controller.report_profile)

    @property
    def evaluation_task(self) -> str:
        return search_eval_runtime_task(self.cfg.eval_cfg)

    @property
    def predict_results(self) -> list[PredictionResult]:
        return self.runtime_controller.get_predict_results()

    @property
    def predict_labels(self) -> list[PredictionLabel]:
        return self.runtime_controller.get_predict_labels()

    @property
    def predict_metrics(self) -> dict[str, float]:
        return self.runtime_controller.get_predict_metrics()

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

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        del dataloader_idx
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

    def _resolve_schedule_step(self, *, global_step: int | None = None) -> int:
        trainer = getattr(self, "_trainer", None)
        current_step = 0 if trainer is None else int(trainer.global_step)
        if global_step is not None:
            current_step = int(global_step)
        return current_step

    def _resolve_sampling_temperature(self, *, global_step: int | None = None) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.sampling_temperature_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_replay_mix_alpha(self, *, global_step: int | None = None) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.replay_mix_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_supervision_phase(
        self, *, global_step: int | None = None
    ) -> dict[str, float | bool]:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return resolve_supervision_phase(
            self.cfg.training_cfg["auxiliary"]["supervision"],
            current_step=current_step,
        )

    def _build_replay_batch(
        self,
        *,
        batch: TrajectoryBatch,
        prepared_batch: Any,
    ) -> tuple[TrajectoryBatch, tuple[tuple[int, ...], ...], dict[str, float]] | None:
        return self.training_orchestrator._build_replay_batch(
            batch=batch,
            prepared_batch=prepared_batch,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        step_result = self.training_orchestrator.run_step(
            batch=trajectory_batch,
            sampling_temperature=self._resolve_sampling_temperature(),
            replay_mix_alpha=self._resolve_replay_mix_alpha(),
            supervision_phase=self._resolve_supervision_phase(),
            effective_pass=self._resolve_effective_pass(after_current_step=True),
        )
        self._latest_train_metrics = self._build_train_metrics_payload(
            step_result.metrics
        )
        self._log_metric_bundle(
            metrics=step_result.metrics,
            prefix="train",
            batch_size=trajectory_batch.num_graphs,
            on_step=True,
            on_epoch=False,
            prog_bar_key="train/loss",
        )
        return step_result.total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        schedule_context = self._trainer_schedule_context()
        return build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=dict(self.cfg.optimizer_cfg),
            scheduler_cfg=dict(self.cfg.scheduler_cfg),
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

    def _evaluate_batch_output(
        self, *, batch: TrajectoryBatch
    ) -> MetricEvaluationOutput:
        return self.runtime_controller.evaluate_batch_output(
            batch=batch,
            include_answer_support=search_eval_include_answer_support(
                self.cfg.eval_cfg
            ),
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
            include_answer_support=search_eval_include_answer_support(
                self.cfg.eval_cfg
            ),
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
        self.runtime_controller.reset_prediction_state()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[PredictionResult]:
        del batch_idx, dataloader_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        return self.runtime_controller.predict_batch(batch=trajectory_batch)

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
        return self.runtime_controller.write_prediction_artifacts(settings=write_config)


__all__ = ["GFlowNetModule", "PredictionArtifactWriteConfig"]
