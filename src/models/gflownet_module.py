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
    TransitionPolicyHead,
)
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
from .gflownet import (
    BaseSearchPolicy,
    GFlowNetPolicy,
    SearchActionPrior,
    SamplingTemperatureScheduler,
    SubTrajectoryBalanceLoss,
    SubTrajectoryBalanceLossOutput,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
    normalize_scheduler_interval,
)
from .gflownet.success_paths import (
    collect_success_rollout_key_rows,
    deduplicate_success_rollout_key_rows,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class GFlowNetConfig:
    horizon_cfg: HorizonConfig
    training_cfg: GFlowNetTrainingConfig
    action_prior_cfg: ActionPriorConfig
    policy_cfg: PolicyConfig
    eval_cfg: SearchEvalConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig

    @property
    def heuristic_cfg(self) -> ActionPriorConfig:
        return self.action_prior_cfg


@dataclass(frozen=True)
class TrainingRolloutMetrics:
    unique_success_paths_per_100_rollouts: float
    new_success_paths: int
    start_node_entropy: torch.Tensor
    start_node_entropy_normalized: torch.Tensor
    active_forward_states: float
    unique_forward_states: float
    forward_state_dedup_keep_ratio: float
    raw_graph_candidates: float
    scored_graph_candidates: float
    raw_graph_candidates_per_unique_state: float
    scored_graph_candidates_per_unique_state: float


class GFlowNetPolicyFactory:
    @staticmethod
    def build_base_policy(
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
            conditioning=str(policy_cfg.state_score_head.conditioning),
        )
        forward_policy_head = TransitionPolicyHead(
            state_dim=graph_hidden_dim,
            relation_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.forward_policy_head.hidden_dim),
            num_layers=int(policy_cfg.forward_policy_head.num_layers),
            dropout=float(policy_cfg.forward_policy_head.dropout),
            detach_input_features=bool(
                policy_cfg.forward_policy_head.detach_input_features
            ),
        )
        return BaseSearchPolicy(
            config=policy_cfg,
            max_steps=max_steps,
            backbone=backbone,
            state_score_head=state_score_head,
            forward_policy_head=forward_policy_head,
        )

    @staticmethod
    def build_action_prior(
        *,
        action_prior_cfg: ActionPriorConfig,
    ) -> SearchActionPrior:
        return SearchActionPrior(config=action_prior_cfg)

    @staticmethod
    def build_policy(
        *,
        policy_cfg: PolicyConfig,
        action_prior_cfg: ActionPriorConfig,
        max_steps: int,
    ) -> GFlowNetPolicy:
        base_policy = GFlowNetPolicyFactory.build_base_policy(
            policy_cfg=policy_cfg,
            max_steps=max_steps,
        )
        search_action_prior = GFlowNetPolicyFactory.build_action_prior(
            action_prior_cfg=action_prior_cfg,
        )
        return GFlowNetPolicy(
            base_policy=base_policy,
            action_prior_cfg=action_prior_cfg,
            search_action_prior=search_action_prior,
        )


class GFlowNetModule(LightningModule):
    @staticmethod
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        if is_dataclass(cfg) and not isinstance(cfg, type):
            return asdict(cfg)  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            return dict(cfg)
        raise TypeError(f"Expected dataclass or dict config, got {type(cfg)!r}.")

    @classmethod
    def _use_zero_weight_decay(
        cls, *, name: str, parameter: torch.nn.Parameter
    ) -> bool:
        del cls
        if name.endswith(".bias"):
            return True
        return parameter.ndim <= 1

    @classmethod
    def _build_optimizer_param_groups(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        trainable_named_params = [
            (name, parameter)
            for name, parameter in model_parameters
            if parameter.requires_grad
        ]
        if not trainable_named_params:
            raise RuntimeError("No trainable parameters found in model.")

        base_lr = float(optimizer_cfg.get("lr", 1.0e-4))
        base_weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
        if not bool(optimizer_cfg.get("no_decay_on_bias_and_norm", True)):
            return [
                {
                    "params": [parameter for _, parameter in trainable_named_params],
                    "lr": base_lr,
                    "weight_decay": base_weight_decay,
                    "group_name": "default",
                }
            ]

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, parameter in trainable_named_params:
            if cls._use_zero_weight_decay(name=name, parameter=parameter):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "lr": base_lr,
                    "weight_decay": base_weight_decay,
                    "group_name": "decay",
                }
            )
        if no_decay_params:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                    "group_name": "no_decay",
                }
            )
        return param_groups

    @classmethod
    def _build_optimizer_and_scheduler(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        schedule_context: TrainingScheduleContext,
    ) -> dict[str, Any]:
        optimizer_param_groups = cls._build_optimizer_param_groups(
            model_parameters=model_parameters,
            optimizer_cfg=optimizer_cfg,
        )

        opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
        if opt_type != "adamw":
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        optimizer = AdamW(
            optimizer_param_groups,
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
        action_prior_cfg: ActionPriorConfig | None = None,
        heuristic_cfg: ActionPriorConfig | None = None,
    ) -> None:
        super().__init__()
        if action_prior_cfg is not None and heuristic_cfg is not None:
            raise ValueError("Pass either action_prior_cfg or heuristic_cfg, not both.")
        if action_prior_cfg is None:
            action_prior_cfg = heuristic_cfg or ActionPriorConfig()
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
        self.loss_fn = SubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=training_cfg.sampling_temperature,
            config=training_cfg.sampling_temperature_schedule,
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
        return str(self.cfg.eval_cfg.task)

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
        total_rollouts = int(sample_batch.success_mask.numel())
        success_path_rows = collect_success_rollout_key_rows(
            batch=batch,
            sample_batch=sample_batch,
        )
        unique_success_path_rows = deduplicate_success_rollout_key_rows(
            success_path_rows
        )
        new_success_paths = (
            0
            if unique_success_path_rows is None
            else int(unique_success_path_rows.size(0))
        )
        start_entropy = sample_batch.behavior_start_entropy
        start_entropy_normalized = sample_batch.behavior_start_entropy_normalized
        mean_start_entropy = (
            start_entropy.detach().to(dtype=torch.float32).mean()
            if start_entropy is not None and int(start_entropy.numel()) > 0
            else torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
        )
        mean_start_entropy_normalized = (
            start_entropy_normalized.detach().to(dtype=torch.float32).mean()
            if start_entropy_normalized is not None
            and int(start_entropy_normalized.numel()) > 0
            else torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
        )
        unique_success_rate = (
            (100.0 * float(new_success_paths)) / float(total_rollouts)
            if total_rollouts > 0
            else 0.0
        )
        active_forward_states = float(sample_batch.total_active_agent_count)
        unique_forward_states = float(sample_batch.total_unique_active_state_count)
        raw_graph_candidates = float(sample_batch.total_raw_graph_candidate_count)
        scored_graph_candidates = float(sample_batch.total_scored_graph_candidate_count)
        forward_state_dedup_keep_ratio = (
            unique_forward_states / active_forward_states
            if active_forward_states > 0.0
            else 0.0
        )
        raw_graph_candidates_per_unique_state = (
            raw_graph_candidates / unique_forward_states
            if unique_forward_states > 0.0
            else 0.0
        )
        scored_graph_candidates_per_unique_state = (
            scored_graph_candidates / unique_forward_states
            if unique_forward_states > 0.0
            else 0.0
        )
        return TrainingRolloutMetrics(
            unique_success_paths_per_100_rollouts=unique_success_rate,
            new_success_paths=new_success_paths,
            start_node_entropy=mean_start_entropy,
            start_node_entropy_normalized=mean_start_entropy_normalized,
            active_forward_states=active_forward_states,
            unique_forward_states=unique_forward_states,
            forward_state_dedup_keep_ratio=forward_state_dedup_keep_ratio,
            raw_graph_candidates=raw_graph_candidates,
            scored_graph_candidates=scored_graph_candidates,
            raw_graph_candidates_per_unique_state=raw_graph_candidates_per_unique_state,
            scored_graph_candidates_per_unique_state=scored_graph_candidates_per_unique_state,
        )

    @staticmethod
    def _safe_batch_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.detach().to(dtype=torch.float32).view(-1)
        y = y.detach().to(dtype=torch.float32).view(-1)
        if int(x.numel()) < 2 or int(y.numel()) < 2:
            return 0.0
        x = x - x.mean()
        y = y - y.mean()
        x_norm = torch.linalg.vector_norm(x)
        y_norm = torch.linalg.vector_norm(y)
        if float(x_norm.item()) == 0.0 or float(y_norm.item()) == 0.0:
            return 0.0
        corr = torch.dot(x, y) / (x_norm * y_norm)
        return float(corr.clamp(min=-1.0, max=1.0).item())

    def _compute_root_diagnostics(
        self,
        *,
        prepared_batch: Any,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> dict[str, float]:
        topology = prepared_batch.topology
        device = sample_batch.graph_log_z.device
        node_counts = (
            topology.graph_node_offsets[1:] - topology.graph_node_offsets[:-1]
        ).to(
            device=device,
            dtype=torch.float32,
        )
        edge_counts = torch.zeros_like(node_counts)
        if int(topology.edge_index.size(1)) > 0:
            edge_graph_ids = topology.graph_index_from_nodes(
                topology.edge_index[0].to(device=device)
            )
            edge_counts.scatter_add_(
                0,
                edge_graph_ids,
                torch.ones_like(edge_graph_ids, dtype=torch.float32),
            )
        start_counts = prepared_batch.observation.q_local_indices.counts().to(
            device=device,
            dtype=torch.float32,
        )
        log_z = sample_batch.graph_log_z.detach().to(dtype=torch.float32)
        return {
            "log_z_num_nodes_corr": self._safe_batch_correlation(
                log_z, torch.log1p(node_counts)
            ),
            "log_z_num_edges_corr": self._safe_batch_correlation(
                log_z, torch.log1p(edge_counts)
            ),
            "log_z_start_candidates_corr": self._safe_batch_correlation(
                log_z,
                torch.log1p(start_counts),
            ),
        }

    def _build_training_metrics(
        self,
        *,
        loss_output: SubTrajectoryBalanceLossOutput,
        total_loss: torch.Tensor,
        rollout_batch_size: int,
        sampling_temperature: float,
        rollout_metrics: TrainingRolloutMetrics,
        root_diagnostics: dict[str, float],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "actor_loss": loss_output.loss.detach(),
            "subtb_loss": loss_output.subtb_loss,
            "subtb_root_loss": loss_output.root_component_loss,
            "subtb_pairwise_loss": loss_output.pairwise_component_loss,
            "subtb_terminal_loss": loss_output.terminal_component_loss,
            "subtb_residual": loss_output.residual_abs,
            "subtb_residual_variance_per_batch": loss_output.residual_variance,
            "subtb_root": loss_output.root_abs,
            "rollout_success": loss_output.success_rate,
            "unique_success_paths_per_100_rollouts": (
                rollout_metrics.unique_success_paths_per_100_rollouts
            ),
            "new_success_paths": float(rollout_metrics.new_success_paths),
            "start_node_entropy": rollout_metrics.start_node_entropy,
            "start_node_entropy_normalized": rollout_metrics.start_node_entropy_normalized,
            "active_forward_states": rollout_metrics.active_forward_states,
            "unique_forward_states": rollout_metrics.unique_forward_states,
            "forward_state_dedup_keep_ratio": (
                rollout_metrics.forward_state_dedup_keep_ratio
            ),
            "raw_graph_candidates": rollout_metrics.raw_graph_candidates,
            "scored_graph_candidates": rollout_metrics.scored_graph_candidates,
            "raw_graph_candidates_per_unique_state": (
                rollout_metrics.raw_graph_candidates_per_unique_state
            ),
            "scored_graph_candidates_per_unique_state": (
                rollout_metrics.scored_graph_candidates_per_unique_state
            ),
            "log_z_mean": loss_output.log_z_mean,
            "log_z_variance": loss_output.log_z_variance,
            "rollout_batch_size": float(rollout_batch_size),
            "sampling_temperature": sampling_temperature,
            "step_log_penalty": float(self.cfg.training_cfg.step_log_penalty or 0.0),
            "terminal_failure_log_reward": float(
                self.cfg.training_cfg.terminal_failure_log_reward
            ),
        }
        metrics.update(root_diagnostics)
        effective_pass = self._resolve_effective_pass(after_current_step=True)
        if effective_pass is not None:
            metrics["effective_pass"] = effective_pass
        return metrics

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
        trajectory_batch = self._require_trajectory_batch(batch)
        if self.sampler is None:
            raise RuntimeError(
                "Current metric runtime does not define a training sampler; this model cannot train with the configured metric_runtime_factory."
            )
        prepared_batch = self.policy.prepare_batch(trajectory_batch)
        rollout_batch_size = int(self.cfg.training_cfg.rollout_batch_size)
        sampling_temperature = self._resolve_sampling_temperature()
        trajectory_batch = trajectory_batch.without_raw_features()
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollout_batch_size=rollout_batch_size,
            temperature=sampling_temperature,
        )
        loss_output = self.loss_fn.compute(sample_batch)
        rollout_metrics = self._compute_training_rollout_metrics(
            batch=trajectory_batch,
            sample_batch=sample_batch,
        )
        root_diagnostics = self._compute_root_diagnostics(
            prepared_batch=prepared_batch,
            sample_batch=sample_batch,
        )
        total_loss = loss_output.loss
        self._raise_on_nonfinite_training_loss(
            total_loss=total_loss,
            batch=trajectory_batch,
        )
        metrics = self._build_training_metrics(
            loss_output=loss_output,
            total_loss=total_loss,
            rollout_batch_size=rollout_batch_size,
            sampling_temperature=sampling_temperature,
            rollout_metrics=rollout_metrics,
            root_diagnostics=root_diagnostics,
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
