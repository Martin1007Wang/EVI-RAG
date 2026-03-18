from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
)
from src.models.components.heuristic_heads import LearnedHeuristicHead
from src.models.configs import (
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SearchEvalConfig,
    SchedulerConfig,
)
from src.graph_runtime import TrajectoryBatch
from src.metrics.answer_reachability.exact_analysis import ExactReachabilityAnalyzer
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeFactoryProtocol
from src.utils.fit_schedule import ResolvedPassFitSchedule
from src.utils.logging_utils import get_logger, log_event, log_metric

from .evaluation_controller import (
    MetricRuntimeController,
    PredictionLabel,
    PredictionResult,
)
from .gflownet import (
    BaseSearchPolicy,
    GFlowNetPolicy,
    SamplingTemperatureScheduler,
    SearchHeuristic,
    SuccessfulTrajectoryReplayBuffer,
    SubTrajectoryBalanceLoss,
    TrainingScheduleContext,
    build_replay_sample_batch,
    normalize_scheduler_interval,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class GFlowNetConfig:
    horizon_cfg: HorizonConfig
    training_cfg: GFlowNetTrainingConfig
    heuristic_cfg: HeuristicConfig
    policy_cfg: PolicyConfig
    eval_cfg: SearchEvalConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig


@dataclass(frozen=True)
class AuxiliaryLossResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class GFlowNetModule(LightningModule):
    @staticmethod
    def _build_base_policy(
        *,
        policy_cfg: PolicyConfig,
        max_steps: int,
    ) -> BaseSearchPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        backbone = EmbeddingBackbone(policy_cfg.backbone)
        state_score_head = NodeFlowHead(
            node_dim=graph_hidden_dim,
            question_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.state_score_head.hidden_dim),
            num_layers=int(policy_cfg.state_score_head.num_layers),
            dropout=float(policy_cfg.state_score_head.dropout),
        )
        return BaseSearchPolicy(
            config=policy_cfg,
            max_steps=max_steps,
            backbone=backbone,
            state_score_head=state_score_head,
        )

    @staticmethod
    def _build_search_heuristic(
        *,
        heuristic_cfg: HeuristicConfig,
        graph_hidden_dim: int,
    ) -> SearchHeuristic:
        learned_head = None
        if heuristic_cfg.kind == "learned":
            learned_head = LearnedHeuristicHead(
                hidden_dim=int(heuristic_cfg.learned_hidden_dim),
                dropout=float(heuristic_cfg.learned_dropout),
                feature_dim=graph_hidden_dim,
            )
        return SearchHeuristic(
            config=heuristic_cfg,
            learned_head=learned_head,
        )

    @staticmethod
    def _build_policy(
        *,
        policy_cfg: PolicyConfig,
        heuristic_cfg: HeuristicConfig,
        max_steps: int,
    ) -> GFlowNetPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        base_policy = GFlowNetModule._build_base_policy(
            policy_cfg=policy_cfg,
            max_steps=max_steps,
        )
        search_heuristic = GFlowNetModule._build_search_heuristic(
            heuristic_cfg=heuristic_cfg,
            graph_hidden_dim=graph_hidden_dim,
        )
        return GFlowNetPolicy(
            base_policy=base_policy,
            heuristic_cfg=heuristic_cfg,
            search_heuristic=search_heuristic,
        )

    @staticmethod
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        if is_dataclass(cfg):
            return asdict(cfg)  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            return dict(cfg)
        raise TypeError(f"Expected dataclass or dict config, got {type(cfg)!r}.")

    @classmethod
    def _build_optimizer_and_scheduler(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        estimated_stepping_batches: int | None,
        trainer_max_steps: int | None = None,
        trainer_max_epochs: int | None = None,
    ) -> dict[str, Any]:
        schedule_context = TrainingScheduleContext(
            estimated_stepping_batches=estimated_stepping_batches,
            trainer_max_steps=trainer_max_steps,
            trainer_max_epochs=trainer_max_epochs,
        )
        trainable_params = [
            parameter for _, parameter in model_parameters if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in model.")

        opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
        if opt_type != "adamw":
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        optimizer = AdamW(
            trainable_params,
            lr=float(optimizer_cfg.get("lr", 1.0e-4)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        )
        scheduler = None
        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
        interval = normalize_scheduler_interval(scheduler_cfg)
        explicit_t_max = (
            int(scheduler_cfg["t_max"])
            if scheduler_cfg.get("t_max") is not None
            else None
        )
        schedule_horizon = schedule_context.resolve_horizon(
            explicit_horizon=explicit_t_max,
            interval=interval,
        )
        if schedule_horizon is not None:
            eta_min = float(scheduler_cfg.get("eta_min", 0.0))
            if scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=schedule_horizon,
                    eta_min=eta_min,
                )
            elif scheduler_type == "cosine_warm_restarts":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=schedule_horizon,
                    T_mult=int(scheduler_cfg.get("t_mult", 1)),
                    eta_min=eta_min,
                )
            elif scheduler_type == "onecycle":
                if interval != "step":
                    raise ValueError(
                        "onecycle scheduler requires interval='step' because it must advance per optimizer step."
                    )
                configured_training_steps = schedule_context.configured_training_steps()
                if (
                    configured_training_steps is not None
                    and schedule_horizon < configured_training_steps
                ):
                    raise ValueError(
                        "onecycle scheduler would exhaust before training ends: "
                        f"t_max={schedule_horizon} configured_steps={configured_training_steps}. "
                        "Set trainer.max_steps and scheduler t_max consistently."
                    )
                scheduler_lr = scheduler_cfg.get("lr", optimizer_cfg.get("lr", 1.0e-4))
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=float(scheduler_lr),
                    total_steps=schedule_horizon,
                    pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
                    anneal_strategy=str(scheduler_cfg.get("anneal", "cos")),
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        result: dict[str, Any] = {"optimizer": optimizer}
        if scheduler is not None:
            result["lr_scheduler"] = {"scheduler": scheduler, "interval": interval}
        return result

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
        heuristic_cfg: HeuristicConfig = HeuristicConfig(),
    ) -> None:
        super().__init__()
        self.cfg = GFlowNetConfig(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            heuristic_cfg=heuristic_cfg,
            policy_cfg=policy_cfg,
            eval_cfg=eval_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
        self.save_hyperparameters({"config": asdict(self.cfg)}, logger=False)
        self.policy = self._build_policy(
            policy_cfg=policy_cfg,
            heuristic_cfg=heuristic_cfg,
            max_steps=int(horizon_cfg.max_steps),
        )
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
        self.training_exact_analyzer = ExactReachabilityAnalyzer(
            max_steps=int(horizon_cfg.max_steps)
        )
        self.sampler = self.runtime_controller.sampler
        self.loss_fn = SubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=float(training_cfg.sampling_temperature),
            config=training_cfg.sampling_temperature_schedule,
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self.success_replay_buffer: SuccessfulTrajectoryReplayBuffer | None = None
        replay_cfg = self.cfg.training_cfg.success_replay
        if bool(replay_cfg.enabled):
            if self.sampler is None or not hasattr(
                self.sampler, "trajectory_supervisor"
            ):
                raise RuntimeError(
                    "Successful trajectory replay requires a sampler with a trajectory_supervisor."
                )
            self.success_replay_buffer = SuccessfulTrajectoryReplayBuffer(
                max_buffer_size=int(replay_cfg.max_buffer_size),
                max_trajectories_per_sample=int(replay_cfg.max_trajectories_per_sample),
            )

    @property
    def metrics_profile(self) -> str:
        return str(self.runtime_controller.metrics_profile)

    @property
    def evaluation_task(self) -> str:
        return str(self.cfg.eval_cfg.task)

    @property
    def predict_results(self) -> list[PredictionResult]:
        return self.runtime_controller.prediction_state.results

    @predict_results.setter
    def predict_results(self, value: list[PredictionResult]) -> None:
        self.runtime_controller.prediction_state.results = list(value)

    @property
    def predict_labels(self) -> list[PredictionLabel]:
        return self.runtime_controller.prediction_state.labels

    @predict_labels.setter
    def predict_labels(self, value: list[PredictionLabel]) -> None:
        self.runtime_controller.prediction_state.labels = list(value)

    @property
    def predict_metrics(self) -> dict[str, float]:
        return self.runtime_controller.prediction_state.metrics

    @predict_metrics.setter
    def predict_metrics(self, value: dict[str, float]) -> None:
        self.runtime_controller.prediction_state.metrics = dict(value)

    @staticmethod
    def _require_trajectory_batch(batch: object) -> TrajectoryBatch:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError(
                "GFlowNetModule expects TrajectoryBatch inputs from the datamodule."
            )
        return batch

    def set_fit_schedule(self, schedule: ResolvedPassFitSchedule) -> None:
        self._fit_schedule = schedule

    def _resolve_effective_pass(self, *, after_current_step: bool) -> float | None:
        if self._fit_schedule is None:
            return None
        current_step = int(self.global_step)
        if after_current_step:
            current_step += 1
        return self._fit_schedule.effective_pass(global_step=current_step)

    def _resolve_success_replay_rollouts_per_graph(self) -> int:
        replay_cfg = self.cfg.training_cfg.success_replay
        if not bool(replay_cfg.enabled):
            return 0
        total_rollouts = int(self.cfg.training_cfg.rollout_batch_size)
        if total_rollouts < 1:
            return 0
        replay_rollouts = int(round(total_rollouts * float(replay_cfg.ratio)))
        return max(replay_rollouts, 0)

    def _exact_aux_is_ready(self) -> bool:
        exact_cfg = self.cfg.training_cfg.exact_aux
        if not bool(exact_cfg.enabled):
            return False
        if (
            float(exact_cfg.success_weight) <= 0.0
            and float(exact_cfg.coverage_weight) <= 0.0
        ):
            return False
        current_step = int(self.global_step)
        if current_step % int(exact_cfg.interval_steps) != 0:
            return False
        effective_pass = self._resolve_effective_pass(after_current_step=False)
        if effective_pass is None:
            return float(exact_cfg.warmup_passes) <= 0.0
        return float(effective_pass) >= float(exact_cfg.warmup_passes)

    @staticmethod
    def _aggregate_entity_mass(
        *,
        node_entity_ids: torch.Tensor,
        node_mass: torch.Tensor,
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        if int(entity_ids.numel()) == 0:
            return node_mass.new_empty((0,))
        matches = node_entity_ids.unsqueeze(1) == entity_ids.unsqueeze(0)
        return (node_mass.unsqueeze(1) * matches.to(dtype=node_mass.dtype)).sum(dim=0)

    def _compute_exact_auxiliary_loss(
        self,
        *,
        batch: TrajectoryBatch,
    ) -> AuxiliaryLossResult | None:
        if not self._exact_aux_is_ready():
            return None
        exact_cfg = self.cfg.training_cfg.exact_aux
        num_selected = min(int(batch.num_graphs), int(exact_cfg.max_graphs_per_batch))
        if num_selected < 1:
            return None

        success_losses: list[torch.Tensor] = []
        coverage_losses: list[torch.Tensor] = []
        success_masses: list[torch.Tensor] = []
        coverage_masses: list[torch.Tensor] = []
        for graph_idx in range(num_selected):
            single_batch = batch.select_graph(graph_idx)
            single_prepared = self.policy.prepare_batch(single_batch)
            dp_result = self.training_exact_analyzer._run_dynamic_program(
                batch=single_batch,
                policy=self.policy,
                prepared_batch=single_prepared,
            )
            if float(exact_cfg.success_weight) > 0.0:
                success_mass = dp_result.terminal_mass.sum()
                success_losses.append(
                    -torch.log(success_mass.clamp_min(float(exact_cfg.eps)))
                )
                success_masses.append(success_mass.detach())
            if float(exact_cfg.coverage_weight) > 0.0:
                gold_entity_ids = torch.unique(single_batch.answer_entity_ids)
                if int(gold_entity_ids.numel()) > 0:
                    gold_retrieval_mass = self._aggregate_entity_mass(
                        node_entity_ids=single_batch.node_global_ids,
                        node_mass=dp_result.retrieval_terminal_mass,
                        entity_ids=gold_entity_ids,
                    )
                    coverage_losses.append(
                        -torch.log(
                            gold_retrieval_mass.clamp_min(float(exact_cfg.eps))
                        ).mean()
                    )
                    coverage_masses.append(gold_retrieval_mass.detach().mean())

        if not success_losses and not coverage_losses:
            return None

        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        metrics: dict[str, torch.Tensor] = {
            "exact_aux_graphs": torch.tensor(float(num_selected), device=self.device)
        }
        if success_losses:
            success_loss = torch.stack(success_losses).mean()
            loss = loss + float(exact_cfg.success_weight) * success_loss
            metrics["exact_aux_success_loss"] = success_loss.detach()
            metrics["exact_aux_success_mass"] = torch.stack(success_masses).mean()
        if coverage_losses:
            coverage_loss = torch.stack(coverage_losses).mean()
            loss = loss + float(exact_cfg.coverage_weight) * coverage_loss
            metrics["exact_aux_coverage_loss"] = coverage_loss.detach()
            metrics["exact_aux_coverage_mass"] = torch.stack(coverage_masses).mean()
        metrics["exact_aux_loss"] = loss.detach()
        return AuxiliaryLossResult(loss=loss, metrics=metrics)

    def _success_replay_is_ready(self) -> bool:
        replay_cfg = self.cfg.training_cfg.success_replay
        if not bool(replay_cfg.enabled) or self.success_replay_buffer is None:
            return False
        if len(self.success_replay_buffer) < int(replay_cfg.min_buffer_size):
            return False
        effective_pass = self._resolve_effective_pass(after_current_step=False)
        if effective_pass is None:
            return float(replay_cfg.warmup_passes) <= 0.0
        return float(effective_pass) >= float(replay_cfg.warmup_passes)

    def _compute_success_replay_loss(
        self,
        *,
        batch: TrajectoryBatch,
        replay_rollouts_per_graph: int,
    ) -> tuple[Any, int, int] | None:
        if (
            replay_rollouts_per_graph < 1
            or self.success_replay_buffer is None
            or not self._success_replay_is_ready()
        ):
            return None
        plan = self.success_replay_buffer.plan_for_batch(
            batch=batch,
            replay_rollouts_per_graph=replay_rollouts_per_graph,
        )
        if plan is None:
            return None
        trajectory_supervisor = getattr(self.sampler, "trajectory_supervisor", None)
        if trajectory_supervisor is None:
            raise RuntimeError(
                "Successful trajectory replay requires sampler.trajectory_supervisor."
            )
        replay_batch = TrajectoryBatch.concatenate(
            [batch.select_graph(graph_idx) for graph_idx in plan.graph_indices]
        )
        replay_prepared_batch = self.policy.prepare_batch(replay_batch)
        replay_sample_batch = build_replay_sample_batch(
            batch=replay_batch,
            policy=self.policy,
            prepared_batch=replay_prepared_batch,
            trajectory_supervisor=trajectory_supervisor,
            replay_records=plan.records_by_graph,
            max_steps=int(self.cfg.horizon_cfg.max_steps),
        )
        replay_loss_output = self.loss_fn.compute(replay_sample_batch)
        return replay_loss_output, int(plan.num_trajectories), len(plan.graph_indices)

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, TrajectoryBatch):
            if batch.node_embeddings.device == device:
                return batch
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _trainer_schedule_context(self) -> TrainingScheduleContext:
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

    def configure_optimizers(self) -> dict[str, Any]:
        schedule_context = self._trainer_schedule_context()
        return self._build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=self._cfg_to_dict(self.cfg.optimizer_cfg),
            scheduler_cfg=self._cfg_to_dict(self.cfg.scheduler_cfg),
            estimated_stepping_batches=schedule_context.estimated_stepping_batches,
            trainer_max_steps=schedule_context.trainer_max_steps,
            trainer_max_epochs=schedule_context.trainer_max_epochs,
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
        log_event(
            logger,
            "gflownet_invalid_start_skipped",
            level=logging.WARNING,
            sample_id=batch.sample_ids[0],
            dataset_scope=batch.dataset_scope,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        trajectory_batch = self._require_trajectory_batch(batch)
        if self.sampler is None:
            raise RuntimeError(
                "Current metric runtime does not define a training sampler; this model cannot train with the configured metric_runtime_factory."
            )
        prepared_batch = self.policy.prepare_batch(trajectory_batch)
        sampling_temperature = self._resolve_sampling_temperature()
        replay_rollouts_per_graph = self._resolve_success_replay_rollouts_per_graph()
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollout_batch_size=int(self.cfg.training_cfg.rollout_batch_size),
            temperature=sampling_temperature,
        )
        loss_output = self.loss_fn.compute(sample_batch)
        replay_result = self._compute_success_replay_loss(
            batch=trajectory_batch,
            replay_rollouts_per_graph=replay_rollouts_per_graph,
        )
        replay_loss_output = None
        replay_trajectories = 0
        replay_graphs = 0
        total_loss = loss_output.loss
        exact_aux_result = self._compute_exact_auxiliary_loss(batch=trajectory_batch)
        on_policy_trajectories = int(trajectory_batch.num_graphs) * int(
            self.cfg.training_cfg.rollout_batch_size
        )
        if replay_result is not None:
            replay_loss_output, replay_trajectories, replay_graphs = replay_result
            total_trajectories = on_policy_trajectories + replay_trajectories
            total_loss = (
                loss_output.loss * float(on_policy_trajectories)
                + replay_loss_output.loss * float(replay_trajectories)
            ) / float(total_trajectories)
        if exact_aux_result is not None:
            total_loss = total_loss + exact_aux_result.loss
        if self.success_replay_buffer is not None:
            self.success_replay_buffer.add_successes(
                batch=trajectory_batch,
                sample_batch=sample_batch,
            )
        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "subtb_loss": loss_output.subtb_loss,
            "subtb_residual": loss_output.residual_abs,
            "subtb_root": loss_output.root_abs,
            "rollout_success": loss_output.success_rate,
            "log_z_mean": loss_output.log_z_mean,
            "log_z_variance": loss_output.log_z_variance,
            "sampling_temperature": sampling_temperature,
        }
        if exact_aux_result is not None:
            metrics.update(exact_aux_result.metrics)
        if self.success_replay_buffer is not None:
            metrics["success_replay_buffer_size"] = float(
                len(self.success_replay_buffer)
            )
            metrics["success_replay_ratio"] = (
                float(replay_trajectories)
                / float(on_policy_trajectories + replay_trajectories)
                if replay_trajectories > 0
                else 0.0
            )
            metrics["success_replay_trajectories"] = float(replay_trajectories)
            metrics["success_replay_graphs"] = float(replay_graphs)
        if replay_loss_output is not None:
            metrics["on_policy_loss"] = loss_output.loss.detach()
            metrics["success_replay_loss"] = replay_loss_output.loss.detach()
        effective_pass = self._resolve_effective_pass(after_current_step=True)
        if effective_pass is not None:
            metrics["effective_pass"] = effective_pass
        self._log_metric_bundle(
            metrics=metrics,
            prefix="train",
            batch_size=int(trajectory_batch.num_graphs),
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
            outputs.secondary_metrics,
            outputs.primary_metrics,
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
        output_dir: str | Path,
        split: str,
        artifact_name: str = "rankflow",
        schema_version: int = 1,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> dict[str, Path] | None:
        return self.runtime_controller.write_prediction_artifacts(
            output_dir=output_dir,
            split=split,
            artifact_name=artifact_name,
            schema_version=schema_version,
            entity_vocab_path=entity_vocab_path,
            relation_vocab_path=relation_vocab_path,
            questions_path=questions_path,
            overwrite=overwrite,
        )


__all__ = [
    "GFlowNetConfig",
    "GFlowNetModule",
]
