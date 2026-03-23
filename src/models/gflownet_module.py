from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
from src.models.components.heuristic_heads import LearnedHeuristicHead
from src.models.configs import (
    GFlowNetTrainingConfig,
    GuidanceLossConfig,
    HeuristicConfig,
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
    SamplingTemperatureScheduler,
    SearchHeuristic,
    SearchState,
    ShortestPathRewardScheduler,
    SubTrajectoryBalanceLoss,
    SubTrajectoryBalanceLossOutput,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
    normalize_scheduler_interval,
)
from .gflownet.path import reconstruct_trace_path_token_ids
from .gflownet.reward_shaping import (
    ShortestPathRewardOracle,
    build_shortest_path_reward_oracle,
    compute_shortest_path_alignment_trace,
    compute_shortest_path_prefix_alignment,
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
    heuristic_cfg: HeuristicConfig
    policy_cfg: PolicyConfig
    eval_cfg: SearchEvalConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig


@dataclass(frozen=True)
class GuidanceLossResult:
    loss: torch.Tensor
    target_mean: torch.Tensor
    prediction_mean: torch.Tensor
    active_states: int


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


@dataclass(frozen=True)
class ShortestPathRewardMetrics:
    lambda_weight: float
    mean_alignment: float
    mean_bonus: float
    reachable_start_rate: float
    full_match_rate: float
    success_alignment: float
    failure_alignment: float


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
        )
        forward_policy_head = TransitionPolicyHead(
            state_dim=graph_hidden_dim,
            relation_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.forward_policy_head.hidden_dim),
            num_layers=int(policy_cfg.forward_policy_head.num_layers),
            dropout=float(policy_cfg.forward_policy_head.dropout),
            microbatch_size=int(policy_cfg.forward_policy_head.microbatch_size),
        )
        return BaseSearchPolicy(
            config=policy_cfg,
            max_steps=max_steps,
            backbone=backbone,
            state_score_head=state_score_head,
            forward_policy_head=forward_policy_head,
        )

    @staticmethod
    def build_search_heuristic(
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
    def build_policy(
        *,
        policy_cfg: PolicyConfig,
        heuristic_cfg: HeuristicConfig,
        max_steps: int,
    ) -> GFlowNetPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        base_policy = GFlowNetPolicyFactory.build_base_policy(
            policy_cfg=policy_cfg,
            max_steps=max_steps,
        )
        search_heuristic = GFlowNetPolicyFactory.build_search_heuristic(
            heuristic_cfg=heuristic_cfg,
            graph_hidden_dim=graph_hidden_dim,
        )
        return GFlowNetPolicy(
            base_policy=base_policy,
            heuristic_cfg=heuristic_cfg,
            search_heuristic=search_heuristic,
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
    def _build_optimizer_and_scheduler(
        cls,
        *,
        model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        schedule_context: TrainingScheduleContext,
    ) -> dict[str, Any]:
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
        self.policy = GFlowNetPolicyFactory.build_policy(
            policy_cfg=policy_cfg,
            heuristic_cfg=heuristic_cfg,
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
        self._validate_auxiliary_config(
            heuristic_cfg=heuristic_cfg,
            guidance_cfg=training_cfg.guidance,
        )
        self.sampler = self.runtime_controller.sampler
        if self.sampler is not None and hasattr(
            self.sampler, "trajectory_length_discount"
        ):
            setattr(
                self.sampler,
                "trajectory_length_discount",
                float(training_cfg.trajectory_length_discount),
            )
        self.loss_fn = SubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=training_cfg.sampling_temperature,
            config=training_cfg.sampling_temperature_schedule,
        )
        self.shortest_path_reward_scheduler = ShortestPathRewardScheduler(
            config=training_cfg.shortest_path_reward,
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self._schedule_context_override: TrainingScheduleContext | None = None
        self._invalid_start_count = 0
        self._shortest_path_reward_cache: dict[str, ShortestPathRewardOracle] = {}

    @staticmethod
    def _validate_auxiliary_config(
        *,
        heuristic_cfg: HeuristicConfig,
        guidance_cfg: GuidanceLossConfig,
    ) -> None:
        if float(guidance_cfg.loss_weight) > 0.0 and heuristic_cfg.kind != "learned":
            raise ValueError(
                "training.guidance.loss_weight > 0 requires heuristic.kind='learned'."
            )
        if (
            heuristic_cfg.kind == "learned"
            and float(heuristic_cfg.beta) > 0.0
            and float(guidance_cfg.loss_weight) <= 0.0
        ):
            raise ValueError(
                "heuristic.kind='learned' with heuristic.beta > 0 requires "
                "training.guidance.loss_weight > 0 because the learned proposal "
                "cache is built without gradients."
            )

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
            heuristic_cfg=self.cfg.heuristic_cfg,
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

    def _compute_guidance_loss(
        self,
        *,
        prepared_batch: Any,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> GuidanceLossResult | None:
        guidance_cfg = self.cfg.training_cfg.guidance
        if (
            float(guidance_cfg.loss_weight) <= 0.0
            or self.cfg.heuristic_cfg.kind != "learned"
        ):
            return None
        trace_mask = sample_batch.trace_mask
        if not bool(trace_mask.any().item()):
            return None
        flat_shape = (
            int(sample_batch.trace_nodes.size(0) * sample_batch.trace_nodes.size(1)),
            int(sample_batch.trace_nodes.size(2)),
        )
        trace_path_token_ids = reconstruct_trace_path_token_ids(
            start_nodes=sample_batch.start_nodes,
            trace_edge_ids=sample_batch.trace_edge_ids,
            trace_num_steps=sample_batch.trace_num_steps,
            trace_stop_mask=sample_batch.trace_stop_mask,
            edge_index=prepared_batch.topology.edge_index,
            edge_type=prepared_batch.topology.edge_type,
            max_steps=int(self.cfg.horizon_cfg.max_steps),
        )
        state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=sample_batch.trace_nodes.view(flat_shape),
            done_mask=(~trace_mask).view(flat_shape),
            num_steps=sample_batch.trace_num_steps.view(flat_shape),
            path_token_ids=trace_path_token_ids.view(
                *flat_shape,
                int(trace_path_token_ids.size(-1)),
            ),
            absorbing_mask=torch.zeros_like(trace_mask).view(flat_shape),
        )
        logits = self.policy.compute_guidance_logits(
            prepared_batch,
            state,
            detach_features=bool(guidance_cfg.detach_features),
        ).view_as(sample_batch.trace_nodes)
        targets = (
            sample_batch.success_mask.to(dtype=torch.float32)
            .unsqueeze(-1)
            .expand_as(logits)
        )
        per_state_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        active_loss = per_state_loss[trace_mask]
        if int(active_loss.numel()) == 0:
            return None
        active_logits = logits[trace_mask]
        return GuidanceLossResult(
            loss=active_loss.mean() * float(guidance_cfg.loss_weight),
            target_mean=targets[trace_mask].mean().detach(),
            prediction_mean=torch.sigmoid(active_logits).mean().detach(),
            active_states=int(active_loss.numel()),
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
        shortest_path_reward_metrics: ShortestPathRewardMetrics,
        guidance_result: GuidanceLossResult | None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "actor_loss": loss_output.loss.detach(),
            "subtb_loss": loss_output.subtb_loss,
            "subtb_residual": loss_output.residual_abs,
            "subtb_residual_variance_per_batch": loss_output.residual_variance,
            "subtb_root": loss_output.root_abs,
            "rollout_success": loss_output.success_rate,
            "unique_success_paths_per_100_rollouts": rollout_metrics.unique_success_paths_per_100_rollouts,
            "new_success_paths": float(rollout_metrics.new_success_paths),
            "start_node_entropy": rollout_metrics.start_node_entropy,
            "start_node_entropy_normalized": rollout_metrics.start_node_entropy_normalized,
            "active_forward_states": rollout_metrics.active_forward_states,
            "unique_forward_states": rollout_metrics.unique_forward_states,
            "forward_state_dedup_keep_ratio": rollout_metrics.forward_state_dedup_keep_ratio,
            "raw_graph_candidates": rollout_metrics.raw_graph_candidates,
            "scored_graph_candidates": rollout_metrics.scored_graph_candidates,
            "raw_graph_candidates_per_unique_state": rollout_metrics.raw_graph_candidates_per_unique_state,
            "scored_graph_candidates_per_unique_state": rollout_metrics.scored_graph_candidates_per_unique_state,
            "log_z_mean": loss_output.log_z_mean,
            "log_z_variance": loss_output.log_z_variance,
            "rollout_batch_size": float(rollout_batch_size),
            "sampling_temperature": sampling_temperature,
            "shortest_path_reward_lambda": shortest_path_reward_metrics.lambda_weight,
            "oracle_prefix_align_mean": shortest_path_reward_metrics.mean_alignment,
            "shortest_path_reward_bonus_mean": shortest_path_reward_metrics.mean_bonus,
            "oracle_reachable_start_rate": shortest_path_reward_metrics.reachable_start_rate,
            "oracle_prefix_full_match_rate": shortest_path_reward_metrics.full_match_rate,
            "oracle_prefix_align_on_success": shortest_path_reward_metrics.success_alignment,
            "oracle_prefix_align_on_failure": shortest_path_reward_metrics.failure_alignment,
        }
        metrics.update(root_diagnostics)
        if guidance_result is not None:
            metrics["guidance_loss"] = guidance_result.loss.detach()
            metrics["guidance_target_mean"] = guidance_result.target_mean
            metrics["guidance_prediction_mean"] = guidance_result.prediction_mean
            metrics["guidance_active_states"] = float(guidance_result.active_states)
        effective_pass = self._resolve_effective_pass(after_current_step=True)
        if effective_pass is not None:
            metrics["effective_pass"] = effective_pass
        return metrics

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

    def _resolve_shortest_path_reward_lambda(
        self, *, global_step: int | None = None
    ) -> float:
        trainer = getattr(self, "_trainer", None)
        current_step = 0 if trainer is None else int(trainer.global_step)
        if global_step is not None:
            current_step = int(global_step)
        return self.shortest_path_reward_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    @staticmethod
    def _mean_or_zero(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _shortest_path_reward_cache_key(
        self, *, batch: TrajectoryBatch, graph_idx: int
    ) -> str:
        sample_id = (
            str(batch.sample_ids[graph_idx]) if batch.sample_ids else str(graph_idx)
        )
        return f"{batch.dataset_scope}:{sample_id}"

    def _get_shortest_path_reward_oracle(
        self, *, batch: TrajectoryBatch, graph_idx: int
    ) -> ShortestPathRewardOracle:
        cache_key = self._shortest_path_reward_cache_key(
            batch=batch, graph_idx=graph_idx
        )
        cached = self._shortest_path_reward_cache.get(cache_key)
        if cached is not None:
            return cached
        node_start = int(batch.node_ptr[graph_idx].item())
        node_end = int(batch.node_ptr[graph_idx + 1].item())
        edge_mask = batch.edge_batch == int(graph_idx)
        local_edge_index = batch.edge_index[:, edge_mask] - node_start
        local_edge_relations = batch.edge_rel_global[edge_mask]
        answer_start = int(batch.a_ptr[graph_idx].item())
        answer_end = int(batch.a_ptr[graph_idx + 1].item())
        oracle = build_shortest_path_reward_oracle(
            sample_id=str(batch.sample_ids[graph_idx])
            if batch.sample_ids
            else str(graph_idx),
            edge_index=local_edge_index,
            edge_relations=local_edge_relations,
            answer_local_indices=batch.a_local_indices[answer_start:answer_end],
            num_nodes=node_end - node_start,
        )
        self._shortest_path_reward_cache[cache_key] = oracle
        return oracle

    def _apply_shortest_path_reward_shaping(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> tuple[TrajectoryGFNSampleBatch, ShortestPathRewardMetrics]:
        if float(self.cfg.training_cfg.shortest_path_reward.weight) == 0.0:
            return sample_batch, ShortestPathRewardMetrics(
                lambda_weight=0.0,
                mean_alignment=0.0,
                mean_bonus=0.0,
                reachable_start_rate=0.0,
                full_match_rate=0.0,
                success_alignment=0.0,
                failure_alignment=0.0,
            )
        lambda_weight = self._resolve_shortest_path_reward_lambda()
        completion_power = float(
            self.cfg.training_cfg.shortest_path_reward.completion_power
        )
        trace_edge_ids = sample_batch.trace_edge_ids.detach().to(dtype=torch.long)
        relation_trace_ids = torch.full_like(trace_edge_ids, fill_value=-1)
        graph_move_mask = trace_edge_ids >= 0
        if bool(graph_move_mask.any().item()):
            relation_trace_ids[graph_move_mask] = batch.edge_rel_global.index_select(
                0,
                trace_edge_ids[graph_move_mask],
            )
        log_reward_steps = torch.zeros_like(
            sample_batch.log_pf_steps, dtype=torch.float32
        )
        alignments = torch.zeros_like(
            sample_batch.terminal_log_rewards,
            dtype=torch.float32,
            device=sample_batch.terminal_log_rewards.device,
        )
        potentials = torch.zeros_like(alignments)
        node_ptr_cpu = batch.node_ptr.detach().to(device="cpu", dtype=torch.long)
        start_nodes_cpu = sample_batch.start_nodes.detach().to(
            device="cpu", dtype=torch.long
        )
        relation_trace_cpu = relation_trace_ids.detach().to(
            device="cpu", dtype=torch.long
        )
        success_mask_cpu = sample_batch.success_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        total_rollouts = int(start_nodes_cpu.numel())
        reachable_starts = 0
        full_matches = 0
        success_alignments: list[float] = []
        failure_alignments: list[float] = []
        for graph_idx in range(batch.num_graphs):
            oracle = self._get_shortest_path_reward_oracle(
                batch=batch, graph_idx=graph_idx
            )
            node_offset = int(node_ptr_cpu[graph_idx].item())
            for rollout_idx in range(int(start_nodes_cpu.size(1))):
                start_local = (
                    int(start_nodes_cpu[graph_idx, rollout_idx].item()) - node_offset
                )
                relation_row = relation_trace_cpu[graph_idx, rollout_idx]
                step_positions = (
                    torch.nonzero(relation_row >= 0, as_tuple=False).view(-1).tolist()
                )
                relation_sequence = relation_row[relation_row >= 0].tolist()
                alignment_trace = compute_shortest_path_alignment_trace(
                    oracle=oracle,
                    start_node=start_local,
                    relation_ids=relation_sequence,
                )
                potential_trace = [
                    float(step_alignment) ** completion_power
                    for step_alignment in alignment_trace
                ]
                previous_potential = 0.0
                for step_position, step_potential in zip(
                    step_positions, potential_trace
                ):
                    increment = float(lambda_weight) * (
                        float(step_potential) - float(previous_potential)
                    )
                    log_reward_steps[graph_idx, rollout_idx, int(step_position)] = (
                        increment
                    )
                    previous_potential = float(step_potential)
                alignment = (
                    float(alignment_trace[-1])
                    if alignment_trace
                    else compute_shortest_path_prefix_alignment(
                        oracle=oracle,
                        start_node=start_local,
                        relation_ids=relation_sequence,
                    )
                )
                alignments[graph_idx, rollout_idx] = float(alignment)
                potential = (
                    float(potential_trace[-1])
                    if potential_trace
                    else float(alignment) ** completion_power
                )
                potentials[graph_idx, rollout_idx] = float(potential)
                if oracle.distance_to_answer(start_local) >= 0:
                    reachable_starts += 1
                if alignment >= 1.0 - 1.0e-6:
                    full_matches += 1
                if bool(success_mask_cpu[graph_idx, rollout_idx].item()):
                    success_alignments.append(float(alignment))
                else:
                    failure_alignments.append(float(alignment))
        shaped_bonus = float(lambda_weight) * potentials
        shaped_batch = replace(
            sample_batch,
            log_reward_steps=log_reward_steps,
        )
        return shaped_batch, ShortestPathRewardMetrics(
            lambda_weight=float(lambda_weight),
            mean_alignment=float(alignments.mean().item())
            if total_rollouts > 0
            else 0.0,
            mean_bonus=float(shaped_bonus.mean().item()) if total_rollouts > 0 else 0.0,
            reachable_start_rate=(
                float(reachable_starts) / float(total_rollouts)
                if total_rollouts > 0
                else 0.0
            ),
            full_match_rate=(
                float(full_matches) / float(total_rollouts)
                if total_rollouts > 0
                else 0.0
            ),
            success_alignment=self._mean_or_zero(success_alignments),
            failure_alignment=self._mean_or_zero(failure_alignments),
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
        (
            sample_batch,
            shortest_path_reward_metrics,
        ) = self._apply_shortest_path_reward_shaping(
            batch=trajectory_batch,
            sample_batch=sample_batch,
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
        guidance_result = self._compute_guidance_loss(
            prepared_batch=prepared_batch,
            sample_batch=sample_batch,
        )
        total_loss = loss_output.loss
        if guidance_result is not None:
            total_loss = total_loss + guidance_result.loss
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
            shortest_path_reward_metrics=shortest_path_reward_metrics,
            guidance_result=guidance_result,
        )
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
