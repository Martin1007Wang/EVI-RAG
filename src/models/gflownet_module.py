from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
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
from src.graph_runtime import TrajectoryBatch
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
    SubTrajectoryBalanceLossOutput,
    SuccessfulTrajectoryReplayBuffer,
    SubTrajectoryBalanceLoss,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
    build_replay_sample_batch,
    normalize_scheduler_interval,
)
from .gflownet.adaptive_sampling import (
    AdaptiveSamplingController,
    AdaptiveSamplingMetrics,
)
from .gflownet.path import reconstruct_trace_path_token_ids
from .gflownet.success_paths import (
    collect_success_rollout_key_rows,
    compute_success_path_hash_pairs,
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
class ReplayLossResult:
    loss_output: SubTrajectoryBalanceLossOutput
    num_trajectories: int
    num_graphs: int


@dataclass(frozen=True)
class TrainingLossAggregation:
    total_loss: torch.Tensor
    replay_result: ReplayLossResult | None


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
    shortlist_active_states: float
    candidate_shortlist_activation_rate: float
    candidate_shortlist_keep_ratio: float


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
        self.loss_fn = SubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=training_cfg.sampling_temperature,
            config=training_cfg.sampling_temperature_schedule,
        )
        self.adaptive_sampling_controller = AdaptiveSamplingController(
            config=training_cfg.adaptive_sampling,
            base_rollout_batch_size=training_cfg.rollout_batch_size,
        )
        self.search = self.runtime_controller.search
        self._fit_schedule: ResolvedPassFitSchedule | None = None
        self._schedule_context_override: TrainingScheduleContext | None = None
        self._invalid_start_count = 0
        self._pending_adaptive_observation_steps = 0
        self._pending_adaptive_success_rate: torch.Tensor | None = None
        self._pending_adaptive_unique_success = 0.0
        self._pending_adaptive_residual_variance: torch.Tensor | None = None
        self._seen_success_path_hashes_by_sample: dict[str, set[tuple[int, int]]] = {}
        self.success_replay_buffer: SuccessfulTrajectoryReplayBuffer | None = None
        replay_cfg = self.cfg.training_cfg.success_replay
        if replay_cfg.enabled:
            if self.sampler is None or not hasattr(
                self.sampler, "trajectory_supervisor"
            ):
                raise RuntimeError(
                    "Successful trajectory replay requires a sampler with a trajectory_supervisor."
                )
            self.success_replay_buffer = SuccessfulTrajectoryReplayBuffer(
                max_buffer_size=replay_cfg.max_buffer_size,
                max_trajectories_per_sample=replay_cfg.max_trajectories_per_sample,
            )

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

    def _warmup_passes_completed(
        self, *, required_passes: float, after_current_step: bool = False
    ) -> bool:
        effective_pass = self._resolve_effective_pass(
            after_current_step=after_current_step
        )
        if effective_pass is None:
            return required_passes <= 0.0
        return effective_pass >= required_passes

    def _resolve_success_replay_rollouts_per_graph(
        self, *, on_policy_rollouts_per_graph: int | None = None
    ) -> int:
        replay_cfg = self.cfg.training_cfg.success_replay
        if not replay_cfg.enabled:
            return 0
        ratio = replay_cfg.ratio
        if not 0.0 <= ratio < 1.0:
            raise ValueError(
                f"training.success_replay.ratio must be in [0, 1). Got {ratio}."
            )
        total_rollouts = (
            self.cfg.training_cfg.rollout_batch_size
            if on_policy_rollouts_per_graph is None
            else int(on_policy_rollouts_per_graph)
        )
        if total_rollouts < 1:
            return 0
        replay_rollouts = int(round((total_rollouts * ratio) / (1.0 - ratio)))
        replay_rollouts = max(replay_rollouts, 0)
        max_rollouts_per_graph = replay_cfg.max_rollouts_per_graph
        if max_rollouts_per_graph is None:
            max_rollouts_per_graph = replay_cfg.max_trajectories_per_sample
        return min(replay_rollouts, int(max_rollouts_per_graph))

    @staticmethod
    def _graph_sample_id(batch: TrajectoryBatch, graph_idx: int) -> str:
        sample_ids = getattr(batch, "sample_ids", None)
        if sample_ids is not None and len(sample_ids) > graph_idx:
            return str(sample_ids[graph_idx])
        return str(graph_idx)

    def _should_track_global_success_paths(self) -> bool:
        return bool(self.cfg.training_cfg.adaptive_sampling.enabled)

    def _count_new_success_paths(
        self,
        *,
        batch: TrajectoryBatch,
        unique_success_path_rows: torch.Tensor,
    ) -> int:
        if int(unique_success_path_rows.numel()) == 0:
            return 0
        graph_idx_and_hashes = torch.cat(
            (
                unique_success_path_rows[:, :1],
                compute_success_path_hash_pairs(unique_success_path_rows[:, 1:]),
            ),
            dim=1,
        )
        hash_rows_cpu = graph_idx_and_hashes.detach().to(device="cpu", dtype=torch.long)
        new_success_paths = 0
        for graph_idx, hash_a, hash_b in hash_rows_cpu.tolist():
            sample_id = self._graph_sample_id(batch, graph_idx)
            seen_for_sample = self._seen_success_path_hashes_by_sample.setdefault(
                sample_id, set()
            )
            path_hash = (int(hash_a), int(hash_b))
            if path_hash not in seen_for_sample:
                seen_for_sample.add(path_hash)
                new_success_paths += 1
        return new_success_paths

    def _compute_training_rollout_metrics(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> TrainingRolloutMetrics:
        total_rollouts = int(sample_batch.success_mask.numel())
        new_success_paths = 0
        success_path_rows = collect_success_rollout_key_rows(
            batch=batch,
            sample_batch=sample_batch,
        )
        unique_success_path_rows = deduplicate_success_rollout_key_rows(
            success_path_rows
        )
        if unique_success_path_rows is not None:
            if self._should_track_global_success_paths():
                new_success_paths = self._count_new_success_paths(
                    batch=batch,
                    unique_success_path_rows=unique_success_path_rows,
                )
            else:
                new_success_paths = int(unique_success_path_rows.size(0))
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
        shortlist_active_states = float(sample_batch.total_shortlist_active_state_count)
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
        candidate_shortlist_activation_rate = (
            shortlist_active_states / unique_forward_states
            if unique_forward_states > 0.0
            else 0.0
        )
        candidate_shortlist_keep_ratio = (
            scored_graph_candidates / raw_graph_candidates
            if raw_graph_candidates > 0.0
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
            shortlist_active_states=shortlist_active_states,
            candidate_shortlist_activation_rate=candidate_shortlist_activation_rate,
            candidate_shortlist_keep_ratio=candidate_shortlist_keep_ratio,
        )

    def _success_replay_enabled(self) -> bool:
        replay_cfg = self.cfg.training_cfg.success_replay
        return replay_cfg.enabled and self.success_replay_buffer is not None

    def _success_replay_buffer_ready(self) -> bool:
        if self.success_replay_buffer is None:
            return False
        return (
            len(self.success_replay_buffer)
            >= self.cfg.training_cfg.success_replay.min_buffer_size
        )

    def _success_replay_warmup_done(self) -> bool:
        return self._warmup_passes_completed(
            required_passes=self.cfg.training_cfg.success_replay.warmup_passes
        )

    def _success_replay_is_ready(self) -> bool:
        return (
            self._success_replay_enabled()
            and self._success_replay_buffer_ready()
            and self._success_replay_warmup_done()
        )

    def _compute_success_replay_loss(
        self,
        *,
        batch: TrajectoryBatch,
        replay_rollouts_per_graph: int,
    ) -> ReplayLossResult | None:
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
            [
                batch.select_graph(graph_idx, validate=False)
                for graph_idx in plan.graph_indices
            ],
            validate=False,
        )
        replay_prepared_batch = self.policy.prepare_batch(replay_batch)
        replay_batch = replay_batch.without_raw_features()
        replay_sample_batch = build_replay_sample_batch(
            batch=replay_batch,
            policy=self.policy,
            prepared_batch=replay_prepared_batch,
            trajectory_supervisor=trajectory_supervisor,
            replay_records=plan.records_by_graph,
            max_steps=self.cfg.horizon_cfg.max_steps,
        )
        replay_loss_output = self.loss_fn.compute(replay_sample_batch)
        return ReplayLossResult(
            loss_output=replay_loss_output,
            num_trajectories=int(plan.num_trajectories),
            num_graphs=len(plan.graph_indices),
        )

    def _aggregate_total_loss(
        self,
        *,
        loss_output: SubTrajectoryBalanceLossOutput,
        replay_result: ReplayLossResult | None,
        on_policy_trajectories: int,
    ) -> TrainingLossAggregation:
        total_loss = loss_output.loss
        if replay_result is not None:
            total_trajectories = on_policy_trajectories + replay_result.num_trajectories
            total_loss = (
                loss_output.loss * float(on_policy_trajectories)
                + replay_result.loss_output.loss * float(replay_result.num_trajectories)
            ) / float(total_trajectories)
        return TrainingLossAggregation(
            total_loss=total_loss, replay_result=replay_result
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

    def _adaptive_sampling_observe_interval(self) -> int:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return 1
        return max(1, int(getattr(trainer, "log_every_n_steps", 1) or 1))

    def _flush_pending_adaptive_sampling_metrics(self) -> None:
        pending_steps = int(self._pending_adaptive_observation_steps)
        if pending_steps < 1 or self._pending_adaptive_success_rate is None:
            return
        residual_variance = self._pending_adaptive_residual_variance
        if residual_variance is None:
            raise RuntimeError(
                "adaptive sampling accumulator is missing tensor statistics before flush."
            )
        denom = float(pending_steps)
        self.adaptive_sampling_controller.observe(
            AdaptiveSamplingMetrics(
                success_rate=float(
                    (self._pending_adaptive_success_rate / denom).detach().item()
                ),
                unique_success_paths_per_100_rollouts=(
                    self._pending_adaptive_unique_success / denom
                ),
                subtb_residual_variance_per_batch=float(
                    (residual_variance / denom).detach().item()
                ),
            )
        )
        self._pending_adaptive_observation_steps = 0
        self._pending_adaptive_success_rate = None
        self._pending_adaptive_unique_success = 0.0
        self._pending_adaptive_residual_variance = None

    def _buffer_adaptive_sampling_metrics(
        self,
        *,
        loss_output: SubTrajectoryBalanceLossOutput,
        rollout_metrics: TrainingRolloutMetrics,
    ) -> None:
        if not self.cfg.training_cfg.adaptive_sampling.enabled:
            return
        success_rate = loss_output.success_rate.detach().to(dtype=torch.float32)
        residual_variance = loss_output.residual_variance.detach().to(
            dtype=torch.float32
        )
        if self._pending_adaptive_success_rate is None:
            self._pending_adaptive_success_rate = success_rate
            self._pending_adaptive_residual_variance = residual_variance
        else:
            self._pending_adaptive_success_rate = (
                self._pending_adaptive_success_rate + success_rate
            )
            self._pending_adaptive_residual_variance = (
                self._pending_adaptive_residual_variance + residual_variance
            )
        self._pending_adaptive_unique_success += float(
            rollout_metrics.unique_success_paths_per_100_rollouts
        )
        self._pending_adaptive_observation_steps += 1
        if (
            self._pending_adaptive_observation_steps
            >= self._adaptive_sampling_observe_interval()
        ):
            self._flush_pending_adaptive_sampling_metrics()

    def _build_training_metrics(
        self,
        *,
        loss_output: SubTrajectoryBalanceLossOutput,
        loss_aggregation: TrainingLossAggregation,
        total_loss: torch.Tensor,
        rollout_batch_size: int,
        replay_rollouts_per_graph: int,
        sampling_temperature: float,
        on_policy_trajectories: int,
        controller_metrics: dict[str, float],
        rollout_metrics: TrainingRolloutMetrics,
        guidance_result: GuidanceLossResult | None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "actor_loss": loss_aggregation.total_loss.detach(),
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
            "shortlist_active_states": rollout_metrics.shortlist_active_states,
            "candidate_shortlist_activation_rate": rollout_metrics.candidate_shortlist_activation_rate,
            "candidate_shortlist_keep_ratio": rollout_metrics.candidate_shortlist_keep_ratio,
            "log_z_mean": loss_output.log_z_mean,
            "log_z_variance": loss_output.log_z_variance,
            "rollout_batch_size": float(rollout_batch_size),
            "sampling_temperature": sampling_temperature,
        }
        metrics.update(controller_metrics)
        replay_result = loss_aggregation.replay_result
        if self.success_replay_buffer is not None:
            metrics["success_replay_buffer_size"] = float(
                len(self.success_replay_buffer)
            )
            metrics["success_replay_rollouts_per_graph"] = float(
                replay_rollouts_per_graph
            )
            metrics["success_replay_ratio"] = (
                float(replay_result.num_trajectories)
                / float(on_policy_trajectories + replay_result.num_trajectories)
                if replay_result is not None and replay_result.num_trajectories > 0
                else 0.0
            )
            metrics["success_replay_trajectories"] = float(
                0 if replay_result is None else replay_result.num_trajectories
            )
            metrics["success_replay_graphs"] = float(
                0 if replay_result is None else replay_result.num_graphs
            )
        if replay_result is not None:
            metrics["on_policy_loss"] = loss_output.loss.detach()
            metrics["success_replay_loss"] = replay_result.loss_output.loss.detach()
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
            "Non-finite training loss detected. Check SubTB, replay, and reward inputs."
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
        sampling_plan = self.adaptive_sampling_controller.decision(
            base_temperature=self._resolve_sampling_temperature()
        )
        controller_metrics = self.adaptive_sampling_controller.snapshot_metrics()
        replay_rollouts_per_graph = self._resolve_success_replay_rollouts_per_graph(
            on_policy_rollouts_per_graph=sampling_plan.rollout_batch_size
        )
        replay_result = self._compute_success_replay_loss(
            batch=trajectory_batch,
            replay_rollouts_per_graph=replay_rollouts_per_graph,
        )
        trajectory_batch = trajectory_batch.without_raw_features()
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollout_batch_size=sampling_plan.rollout_batch_size,
            temperature=sampling_plan.sampling_temperature,
        )
        loss_output = self.loss_fn.compute(sample_batch)
        rollout_metrics = self._compute_training_rollout_metrics(
            batch=trajectory_batch,
            sample_batch=sample_batch,
        )
        on_policy_trajectories = (
            trajectory_batch.num_graphs * sampling_plan.rollout_batch_size
        )
        loss_aggregation = self._aggregate_total_loss(
            loss_output=loss_output,
            replay_result=replay_result,
            on_policy_trajectories=on_policy_trajectories,
        )
        guidance_result = self._compute_guidance_loss(
            prepared_batch=prepared_batch,
            sample_batch=sample_batch,
        )
        total_loss = loss_aggregation.total_loss
        if guidance_result is not None:
            total_loss = total_loss + guidance_result.loss
        self._raise_on_nonfinite_training_loss(
            total_loss=total_loss,
            batch=trajectory_batch,
        )
        if self.success_replay_buffer is not None:
            self.success_replay_buffer.add_successes(
                batch=trajectory_batch,
                sample_batch=sample_batch,
            )
        metrics = self._build_training_metrics(
            loss_output=loss_output,
            loss_aggregation=loss_aggregation,
            total_loss=total_loss,
            rollout_batch_size=sampling_plan.rollout_batch_size,
            replay_rollouts_per_graph=replay_rollouts_per_graph,
            sampling_temperature=sampling_plan.sampling_temperature,
            on_policy_trajectories=on_policy_trajectories,
            controller_metrics=controller_metrics,
            rollout_metrics=rollout_metrics,
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
        self._buffer_adaptive_sampling_metrics(
            loss_output=loss_output,
            rollout_metrics=rollout_metrics,
        )
        return total_loss

    def on_train_epoch_end(self) -> None:
        self._flush_pending_adaptive_sampling_metrics()

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
