from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from src.graph import TrajectoryBatch
from src.metrics.search_eval_utils import (
    RUNTIME_ANSWER_TASK,
    normalize_search_eval_cfg,
    search_eval_include_answer_support,
    search_eval_runtime_task,
)
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeFactoryProtocol
from src.utils.fit_schedule import ResolvedPassFitSchedule
from src.utils.logging_utils import get_logger, log_event, log_metric

from .evaluation_controller import (
    MetricRuntimeController,
    PredictionArtifactWriteConfig,
    PredictionLabel,
    PredictionResult,
)
from .gflownet.losses import (
    SubgraphSubTrajectoryBalanceLoss,
    SubgraphSubTrajectoryBalanceLossOutput,
)
from .gflownet.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from .gflownet.config_utils import normalize_training_cfg
from .gflownet.replay import SubgraphReplayRecord, SubgraphSuccessReplayBuffer
from .gflownet.sampler import SubgraphSampler, SubgraphTrajectorySampleBatch
from src.utils.optimizer_utils import build_optimizer_and_scheduler
from src.utils.training_schedules import (
    ProposalBiasScheduler,
    ReplayMixScheduler,
    SamplingTemperatureScheduler,
    TrainingScheduleContext,
)


logger = get_logger(__name__)


def _to_plain_mapping(node: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(node, DictConfig):
        container = OmegaConf.to_container(node, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Expected {field_name} to resolve to a mapping.")
        return dict(container)
    if isinstance(node, Mapping):
        return {str(key): deepcopy(value) for key, value in node.items()}
    raise TypeError(f"Expected {field_name} to be a mapping, got {type(node)!r}.")


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
        horizon_cfg = _to_plain_mapping(horizon_cfg, field_name="horizon_cfg")
        training_cfg = normalize_training_cfg(training_cfg)
        policy_cfg = _to_plain_mapping(policy_cfg, field_name="policy_cfg")
        eval_cfg = normalize_search_eval_cfg(eval_cfg)
        optimizer_cfg = _to_plain_mapping(optimizer_cfg, field_name="optimizer_cfg")
        scheduler_cfg = _to_plain_mapping(scheduler_cfg, field_name="scheduler_cfg")
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
        self.sampler = self.runtime_controller.sampler
        self.loss_fn = SubgraphSubTrajectoryBalanceLoss(**dict(training_cfg["subtb"]))
        proposal_aux_cfg = dict(training_cfg["auxiliary"]["proposal"])
        replay_aux_cfg = dict(training_cfg["auxiliary"]["replay"])
        replay_buffer_cfg = dict(replay_aux_cfg["buffer"])
        self.sampling_temperature_scheduler = SamplingTemperatureScheduler(
            base_temperature=float(training_cfg["sampling_temperature"]),
            **dict(training_cfg["sampling_temperature_schedule"]),
        )
        self.proposal_bias_scheduler = ProposalBiasScheduler(
            base_scale=1.0 if bool(proposal_aux_cfg.get("enabled", False)) else 0.0,
            **dict(proposal_aux_cfg["schedule"]),
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

    def reset_prediction_state(self) -> None:
        self.runtime_controller.reset_prediction_state()

    def reconfigure_evaluation(self, *, eval_cfg: dict[str, Any]) -> None:
        eval_cfg = normalize_search_eval_cfg(eval_cfg)
        self._validate_subgraph_only_config(
            policy_cfg=self.cfg.policy_cfg,
            training_cfg=self.cfg.training_cfg,
            eval_cfg=eval_cfg,
        )
        self.cfg = OmegaConf.create(
            {
                "horizon_cfg": OmegaConf.to_container(
                    self.cfg.horizon_cfg,
                    resolve=True,
                ),
                "training_cfg": OmegaConf.to_container(
                    self.cfg.training_cfg,
                    resolve=True,
                ),
                "policy_cfg": OmegaConf.to_container(
                    self.cfg.policy_cfg,
                    resolve=True,
                ),
                "eval_cfg": eval_cfg,
                "optimizer_cfg": OmegaConf.to_container(
                    self.cfg.optimizer_cfg,
                    resolve=True,
                ),
                "scheduler_cfg": OmegaConf.to_container(
                    self.cfg.scheduler_cfg,
                    resolve=True,
                ),
            }
        )
        self.metric_runtime = self.metric_runtime_factory.build_runtime(
            horizon_cfg=self.cfg.horizon_cfg,
            training_cfg=self.cfg.training_cfg,
            eval_cfg=eval_cfg,
            policy=self.policy,
        )
        self.runtime_controller = MetricRuntimeController(
            metric_runtime=self.metric_runtime,
            report_profile=str(eval_cfg["report_profile"]),
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

    def _resolve_proposal_bias_scale(self, *, global_step: int | None = None) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.proposal_bias_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _resolve_replay_mix_alpha(self, *, global_step: int | None = None) -> float:
        current_step = self._resolve_schedule_step(global_step=global_step)
        return self.replay_mix_scheduler.value(
            global_step=current_step,
            schedule_context=self._trainer_schedule_context(),
        )

    def _auxiliary_cfg(self) -> Mapping[str, Any]:
        return self.cfg.training_cfg["auxiliary"]

    def _auxiliary_proposal_cfg(self) -> Mapping[str, Any]:
        return self._auxiliary_cfg()["proposal"]

    def _auxiliary_replay_cfg(self) -> Mapping[str, Any]:
        return self._auxiliary_cfg()["replay"]

    def _auxiliary_replay_buffer_cfg(self) -> Mapping[str, Any]:
        return self._auxiliary_replay_cfg()["buffer"]

    def _auxiliary_replay_guidance_cfg(self) -> Mapping[str, Any]:
        return self._auxiliary_replay_cfg()["guidance"]

    def _training_action_pruning_cfg(self) -> Mapping[str, Any] | None:
        action_pruning_cfg = self.cfg.training_cfg.get("action_pruning")
        if action_pruning_cfg is None:
            return None
        per_node_top_k = int(action_pruning_cfg.get("per_node_top_k", 0))
        per_state_top_k = int(action_pruning_cfg.get("per_state_top_k", 0))
        if per_node_top_k <= 0 and per_state_top_k <= 0:
            return None
        return {
            "per_node_top_k": per_node_top_k,
            "per_state_top_k": per_state_top_k,
        }

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

    def _replay_trajectory_budget(self, *, batch: TrajectoryBatch) -> int:
        configured = self._auxiliary_replay_buffer_cfg().get(
            "replay_trajectories_per_step"
        )
        if configured is None:
            return int(batch.num_graphs)
        return max(int(configured), 0)

    @staticmethod
    def _replay_feature_dtype(batch: TrajectoryBatch) -> torch.dtype:
        if batch.node_embeddings is None:
            return torch.float32
        return batch.node_embeddings.dtype

    def _compute_expand_imitation_loss(
        self,
        *,
        prepared_batch: Any,
        sample_batch: SubgraphTrajectorySampleBatch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        guidance_cfg = self._auxiliary_replay_guidance_cfg()
        from_anchor_bonus = float(
            guidance_cfg.get("expand_imitation_from_anchor_bonus", 0.0)
        )
        answer_finish_bonus = float(
            guidance_cfg.get("expand_imitation_answer_finish_bonus", 0.0)
        )
        weighted_log_probs: list[torch.Tensor] = []
        weight_values: list[float] = []
        from_anchor_steps = 0.0
        answer_finish_steps = 0.0
        if sample_batch.chosen_source_graph_nodes is None:
            raise RuntimeError(
                "Replay expand imitation requires chosen_source_graph_nodes metadata."
            )
        for graph_idx in range(int(sample_batch.num_graphs)):
            anchor_nodes = {
                int(node_id)
                for node_id in prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)]
            }
            for rollout_idx in range(int(sample_batch.num_rollouts)):
                state = self.policy.initialize_rollout_batch(
                    prepared_batch=prepared_batch,
                    num_rollouts=1,
                ).states[int(graph_idx)]
                for action_step in range(int(self.policy.max_steps)):
                    edge_id = int(
                        sample_batch.chosen_edge_ids[
                            graph_idx, rollout_idx, action_step
                        ]
                        .detach()
                        .item()
                    )
                    if edge_id < 0:
                        continue
                    source_graph_node = int(
                        sample_batch.chosen_source_graph_nodes[
                            graph_idx, rollout_idx, action_step
                        ]
                        .detach()
                        .item()
                    )
                    if source_graph_node < 0:
                        continue
                    current_analysis = self.policy.analyze_state(
                        prepared_batch=prepared_batch,
                        graph_idx=int(graph_idx),
                        state=state,
                    )
                    next_state = state.with_edge(int(edge_id))
                    next_analysis = self.policy.analyze_state(
                        prepared_batch=prepared_batch,
                        graph_idx=int(graph_idx),
                        state=next_state,
                    )
                    weight = 1.0
                    src = int(source_graph_node)
                    if src in anchor_nodes:
                        weight += from_anchor_bonus
                        from_anchor_steps += 1.0
                    current_gold_admissible_count, _ = (
                        self.policy.count_gold_admissible_answers(
                            prepared_batch=prepared_batch,
                            graph_idx=int(graph_idx),
                            analysis=current_analysis,
                        )
                    )
                    next_gold_admissible_count, _ = (
                        self.policy.count_gold_admissible_answers(
                            prepared_batch=prepared_batch,
                            graph_idx=int(graph_idx),
                            analysis=next_analysis,
                        )
                    )
                    if int(next_gold_admissible_count) > int(
                        current_gold_admissible_count
                    ):
                        weight += answer_finish_bonus
                        answer_finish_steps += 1.0
                    weighted_log_probs.append(
                        sample_batch.log_pf_actions[
                            graph_idx, rollout_idx, action_step
                        ].to(dtype=torch.float32)
                    )
                    weight_values.append(float(weight))
                    state = next_state
        if not weight_values:
            return sample_batch.log_pf_actions.new_zeros((), dtype=torch.float32), {
                "from_anchor_steps": 0.0,
                "answer_finish_steps": 0.0,
                "mean_weight": 0.0,
            }
        weight_tensor = torch.tensor(
            weight_values,
            device=sample_batch.log_pf_actions.device,
            dtype=torch.float32,
        )
        log_prob_tensor = torch.stack(weighted_log_probs).to(dtype=torch.float32)
        loss = -(log_prob_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(
            1.0
        )
        return loss, {
            "from_anchor_steps": float(from_anchor_steps),
            "answer_finish_steps": float(answer_finish_steps),
            "mean_weight": float(weight_tensor.mean().item()),
        }

    @staticmethod
    def _mask_replay_stop_actions(
        sample_batch: SubgraphTrajectorySampleBatch,
    ) -> SubgraphTrajectorySampleBatch:
        return replace(
            sample_batch,
            action_mask=sample_batch.action_mask.to(dtype=torch.bool)
            & ~sample_batch.stop_actions.to(dtype=torch.bool),
        )

    def _build_teacher_guidance_records(
        self,
        *,
        batch: TrajectoryBatch,
        prepared_batch: Any,
        max_records: int,
    ) -> list[SubgraphReplayRecord]:
        if max_records <= 0:
            return []
        if not bool(
            self._auxiliary_replay_guidance_cfg().get(
                "add_shortest_path_guidance", False
            )
        ):
            return []
        candidate_graphs: list[tuple[int, int, int]] = []
        for graph_idx, teacher_edge_ids in enumerate(
            prepared_batch.graph_teacher_action_edge_ids
        ):
            if teacher_edge_ids is None:
                continue
            teacher_edge_count = prepared_batch.graph_teacher_edge_count[graph_idx]
            if teacher_edge_count is None or teacher_edge_count > int(
                self.policy.max_steps
            ):
                continue
            num_anchors = int(len(prepared_batch.graph_anchor_abs_node_ids[graph_idx]))
            candidate_graphs.append(
                (0 if num_anchors > 1 else 1, int(teacher_edge_count), graph_idx)
            )
        candidate_graphs.sort()
        records: list[SubgraphReplayRecord] = []
        for _, _, graph_idx in candidate_graphs[:max_records]:
            teacher_edge_ids = prepared_batch.graph_teacher_action_edge_ids[graph_idx]
            if teacher_edge_ids is None:
                continue
            edge_start = int(batch.edge_ptr[graph_idx].item())
            records.append(
                SubgraphReplayRecord(
                    trajectory_batch=batch.select_graph(int(graph_idx)).to(
                        device="cpu",
                        feature_dtype=torch.float16,
                    ),
                    edge_ids=tuple(
                        int(edge_id) - edge_start for edge_id in teacher_edge_ids
                    ),
                    source="guidance",
                )
            )
        return records

    def _sample_buffer_replay_records(
        self, *, max_records: int
    ) -> list[SubgraphReplayRecord]:
        min_buffer_size = int(
            self._auxiliary_replay_buffer_cfg().get("min_buffer_size", 0)
        )
        if len(self.success_replay_buffer) < min_buffer_size:
            return []
        return self.success_replay_buffer.sample(max_records=max_records)

    def _build_replay_batch(
        self,
        *,
        batch: TrajectoryBatch,
        prepared_batch: Any,
    ) -> tuple[TrajectoryBatch, tuple[tuple[int, ...], ...], dict[str, float]] | None:
        replay_budget = self._replay_trajectory_budget(batch=batch)
        if replay_budget <= 0:
            return None
        guidance_records = self._build_teacher_guidance_records(
            batch=batch,
            prepared_batch=prepared_batch,
            max_records=replay_budget,
        )
        remaining_budget = max(int(replay_budget) - len(guidance_records), 0)
        buffer_records = self._sample_buffer_replay_records(
            max_records=remaining_budget
        )
        replay_records = [*guidance_records, *buffer_records]
        if not replay_records:
            return None
        replay_batch = TrajectoryBatch.concatenate(
            [record.trajectory_batch for record in replay_records],
            validate=True,
        ).to(
            device=batch.node_ptr.device,
            feature_dtype=self._replay_feature_dtype(batch),
        )
        edge_offset = 0
        replay_sequences_list: list[tuple[int, ...]] = []
        for record in replay_records:
            replay_sequences_list.append(
                tuple(int(edge_id) + edge_offset for edge_id in record.edge_ids)
            )
            edge_offset += int(record.trajectory_batch.edge_ptr[-1].item())
        replay_sequences = tuple(replay_sequences_list)
        replay_metadata = {
            "guidance_records": float(len(guidance_records)),
            "buffer_records": float(len(buffer_records)),
            "buffer_size": float(len(self.success_replay_buffer)),
            "teacher_unavailable": float(
                sum(
                    1
                    for teacher_edge_count in prepared_batch.graph_teacher_edge_count
                    if teacher_edge_count is None
                )
            ),
            "teacher_over_budget": float(
                sum(
                    1
                    for teacher_edge_count in prepared_batch.graph_teacher_edge_count
                    if teacher_edge_count is not None
                    and teacher_edge_count > int(self.policy.max_steps)
                )
            ),
            "multi_anchor_over_budget": float(
                sum(
                    1
                    for graph_idx, teacher_edge_count in enumerate(
                        prepared_batch.graph_teacher_edge_count
                    )
                    if len(prepared_batch.graph_anchor_abs_node_ids[graph_idx]) > 1
                    and teacher_edge_count is not None
                    and teacher_edge_count > int(self.policy.max_steps)
                )
            ),
        }
        return replay_batch, replay_sequences, replay_metadata

    def _build_subgraph_training_metrics(
        self,
        *,
        loss_output: SubgraphSubTrajectoryBalanceLossOutput,
        sample_batch: SubgraphTrajectorySampleBatch,
        total_loss: torch.Tensor,
        rollouts_per_graph: int,
        sampling_temperature: float,
        proposal_bias_scale: float,
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
        mean_answer_commit_rate = total_loss.new_zeros((), dtype=torch.float32)
        if sample_batch.chosen_answer_entity_ids is not None:
            mean_answer_commit_rate = (
                (sample_batch.chosen_answer_entity_ids >= 0)
                .to(dtype=torch.float32)
                .mean()
            )
        return {
            "loss": total_loss.detach(),
            "objective/subtb_loss": loss_output.subtb_loss.detach(),
            "objective/subtb_residual_abs": loss_output.residual_abs.detach(),
            "objective/subtb_residual_variance": loss_output.residual_variance.detach(),
            "objective/subtb_root_abs": loss_output.root_abs.detach(),
            "answer_commit/gold_commit_rate": loss_output.success_rate.detach(),
            "answer_commit/admissible_count": (
                loss_output.average_terminal_commit_candidate_count.detach()
            ),
            "answer_commit/gold_admissible_count": (
                loss_output.average_terminal_gold_answer_count.detach()
            ),
            "witness/anchor_component_count": (
                loss_output.average_terminal_component_count.detach()
            ),
            "flow/log_z_mean": loss_output.log_z_mean.detach(),
            "flow/log_z_variance": loss_output.log_z_variance.detach(),
            "witness/selected_edge_count": mean_selected_edges.detach(),
            "rollout/stop_step": mean_termination_action_step.detach(),
            "answer_commit/commit_rate": mean_answer_commit_rate.detach(),
            "rollout/rollouts_per_graph": float(rollouts_per_graph),
            "rollout/sampling_temperature": float(sampling_temperature),
            "rollout/proposal_bias_scale": float(proposal_bias_scale),
            "reward/gold_answer_bonus": float(
                self.cfg.training_cfg["answer_reward"]["gold_answer_bonus"]
            ),
            "reward/wrong_answer_penalty": float(
                self.cfg.training_cfg["answer_reward"]["wrong_answer_penalty"]
            ),
            "reward/failure_penalty": float(
                self.cfg.training_cfg["answer_reward"]["failure_penalty"]
            ),
            "reward/size_penalty": float(
                self.cfg.training_cfg["answer_reward"]["size_penalty"]
            ),
            "reward/redundancy_penalty": float(
                self.cfg.training_cfg["answer_reward"]["redundancy_penalty"]
            ),
            "reward/component_penalty": float(
                self.cfg.training_cfg["answer_reward"]["component_penalty"]
            ),
        }

    def _training_step_subgraph(self, batch: TrajectoryBatch) -> torch.Tensor:
        sampler = self._require_subgraph_sampler()
        prepared_batch = self.policy.prepare_batch(batch)
        replay_cfg = self._auxiliary_replay_cfg()
        replay_guidance_cfg = self._auxiliary_replay_guidance_cfg()
        action_pruning_cfg = self._training_action_pruning_cfg()
        rollouts_per_graph = int(self.cfg.training_cfg["rollouts_per_graph"])
        sampling_temperature = self._resolve_sampling_temperature()
        proposal_bias_scale = self._resolve_proposal_bias_scale()
        replay_mix_alpha = self._resolve_replay_mix_alpha()
        trajectory_batch = batch.without_raw_features()
        if not bool(replay_cfg.get("enabled", False)):
            batch = trajectory_batch
        sample_batch = sampler.sample(
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=rollouts_per_graph,
            temperature=sampling_temperature,
            proposal_bias_scale=proposal_bias_scale,
            action_pruning=action_pruning_cfg,
        )
        online_loss_output = self.loss_fn.compute(sample_batch)
        if bool(replay_cfg.get("enabled", False)):
            self.success_replay_buffer.add_successful_trajectories(
                batch=batch,
                sample_batch=sample_batch,
            )
        replay_payload = None
        replay_loss_output: SubgraphSubTrajectoryBalanceLossOutput | None = None
        replay_expand_imitation_loss: torch.Tensor | None = None
        replay_expand_imitation_stats: dict[str, float] = {
            "from_anchor_steps": 0.0,
            "answer_finish_steps": 0.0,
            "mean_weight": 0.0,
        }
        replay_stop_loss_masked_count = 0.0
        replay_metadata: dict[str, float] = {
            "guidance_records": 0.0,
            "buffer_records": 0.0,
            "buffer_size": float(len(self.success_replay_buffer)),
            "teacher_unavailable": 0.0,
            "teacher_over_budget": 0.0,
            "multi_anchor_over_budget": 0.0,
        }
        replay_expand_imitation_weight = float(
            replay_guidance_cfg.get("expand_imitation_weight", 0.0)
        )
        replay_mask_stop_loss = bool(replay_guidance_cfg.get("mask_stop_loss", True))
        if float(replay_mix_alpha) > 0.0:
            replay_payload = self._build_replay_batch(
                batch=batch,
                prepared_batch=prepared_batch,
            )
        if replay_payload is not None:
            replay_batch, replay_sequences, replay_metadata = replay_payload
            replay_prepared_batch = self.policy.prepare_batch(replay_batch)
            replay_sample_batch = sampler.teacher_force(
                policy=self.policy,
                prepared_batch=replay_prepared_batch,
                edge_sequences=replay_sequences,
            )
            replay_loss_input = replay_sample_batch
            if replay_mask_stop_loss:
                replay_stop_loss_masked_count = float(
                    (
                        replay_sample_batch.action_mask.to(dtype=torch.bool)
                        & replay_sample_batch.stop_actions.to(dtype=torch.bool)
                    )
                    .to(dtype=torch.float32)
                    .sum()
                    .item()
                )
                replay_loss_input = self._mask_replay_stop_actions(replay_sample_batch)
            replay_loss_output = self.loss_fn.compute(replay_loss_input)
            replay_branch_loss = replay_loss_output.loss
            if replay_expand_imitation_weight > 0.0:
                expand_imitation_loss, replay_expand_imitation_stats = (
                    self._compute_expand_imitation_loss(
                        prepared_batch=replay_prepared_batch,
                        sample_batch=replay_sample_batch,
                    )
                )
                replay_expand_imitation_loss = expand_imitation_loss
                replay_branch_loss = (
                    replay_branch_loss
                    + replay_expand_imitation_weight * expand_imitation_loss
                )
            total_loss = (
                1.0 - float(replay_mix_alpha)
            ) * online_loss_output.loss + float(replay_mix_alpha) * replay_branch_loss
        else:
            total_loss = online_loss_output.loss
            replay_mix_alpha = 0.0
        self._raise_on_nonfinite_training_loss(
            total_loss=total_loss,
            batch=trajectory_batch,
        )
        metrics = self._build_subgraph_training_metrics(
            loss_output=online_loss_output,
            sample_batch=sample_batch,
            total_loss=total_loss,
            rollouts_per_graph=rollouts_per_graph,
            sampling_temperature=sampling_temperature,
            proposal_bias_scale=proposal_bias_scale,
        )
        metrics["replay/mix_alpha"] = float(replay_mix_alpha)
        metrics["replay/buffer_size"] = float(len(self.success_replay_buffer))
        metrics["replay/guidance_records"] = float(
            replay_metadata.get("guidance_records", 0.0)
        )
        metrics["replay/buffer_records"] = float(
            replay_metadata.get("buffer_records", 0.0)
        )
        metrics["replay/multi_anchor_over_budget"] = float(
            replay_metadata.get("multi_anchor_over_budget", 0.0)
        )
        metrics["replay/teacher_unavailable"] = float(
            replay_metadata.get("teacher_unavailable", 0.0)
        )
        metrics["replay/teacher_over_budget"] = float(
            replay_metadata.get("teacher_over_budget", 0.0)
        )
        metrics["replay/stop_loss_mask_enabled"] = float(replay_mask_stop_loss)
        metrics["replay/stop_loss_masked_count"] = float(replay_stop_loss_masked_count)
        metrics["replay/imitation_weight"] = float(replay_expand_imitation_weight)
        metrics["replay/imitation_from_anchor_steps"] = float(
            replay_expand_imitation_stats.get("from_anchor_steps", 0.0)
        )
        metrics["replay/imitation_answer_finish_steps"] = float(
            replay_expand_imitation_stats.get("answer_finish_steps", 0.0)
        )
        metrics["replay/imitation_mean_weight"] = float(
            replay_expand_imitation_stats.get("mean_weight", 0.0)
        )
        metrics["auxiliary/proposal_enabled"] = float(
            bool(self._auxiliary_proposal_cfg().get("enabled", False))
        )
        metrics["auxiliary/replay_enabled"] = float(
            bool(replay_cfg.get("enabled", False))
        )
        if replay_loss_output is not None:
            metrics["replay/loss"] = replay_loss_output.loss.detach()
            metrics["replay/answer_commit/gold_commit_rate"] = (
                replay_loss_output.success_rate.detach()
            )
            metrics["replay/answer_commit/admissible_count"] = (
                replay_loss_output.average_terminal_commit_candidate_count.detach()
            )
            metrics["replay/answer_commit/gold_admissible_count"] = (
                replay_loss_output.average_terminal_gold_answer_count.detach()
            )
        if replay_expand_imitation_loss is not None:
            metrics["replay/imitation_loss"] = replay_expand_imitation_loss.detach()
        effective_pass = self._resolve_effective_pass(after_current_step=True)
        if effective_pass is not None:
            metrics["rollout/effective_pass"] = float(effective_pass)
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._training_step_subgraph(self._require_trajectory_batch(batch))

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
    "GFlowNetModule",
    "PredictionArtifactWriteConfig",
]
