from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import logging
from typing import Any

import torch

from src.graph import TrajectoryBatch
from src.utils.logging_utils import get_logger, log_event

from ...core.losses import (
    SubgraphDetailedBalanceLoss,
    SubgraphDetailedBalanceLossOutput,
)
from ...core.policy import SubgraphPolicy
from ...core.replay import SubgraphReplayRecord, SubgraphSuccessReplayBuffer
from ...core.sampler import SubgraphSampler, SubgraphTrajectorySampleBatch
from ...core.subgraph_batch import SubgraphBatch, SubgraphBatchBuildOptions
from ...core.supervision import (
    SequenceSupervisionLossOutput,
    compute_expand_imitation_loss,
    compute_sequence_supervision_losses,
)
from .schedules import resolve_action_pruning_cfg


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingStepResult:
    total_loss: torch.Tensor
    metrics: dict[str, Any]


class SubgraphTrainingOrchestrator:
    def __init__(
        self,
        *,
        cfg: Any,
        policy: SubgraphPolicy,
        sampler: SubgraphSampler,
        loss_fn: SubgraphDetailedBalanceLoss,
        success_replay_buffer: SubgraphSuccessReplayBuffer,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.success_replay_buffer = success_replay_buffer

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

    @staticmethod
    def _action_pruning_enabled(action_pruning_cfg: Mapping[str, Any] | None) -> bool:
        if action_pruning_cfg is None:
            return False
        return (
            int(action_pruning_cfg.get("per_node_top_k", 0)) > 0
            or int(action_pruning_cfg.get("per_state_top_k", 0)) > 0
        )

    @staticmethod
    def _sequence_supervision_enabled(
        supervision_phase: Mapping[str, float | bool],
    ) -> bool:
        return bool(supervision_phase["enabled"]) and (
            float(supervision_phase["imitation_weight"]) > 0.0
            or float(supervision_phase["success_action_weight"]) > 0.0
        )

    def _training_batch_build_options(
        self,
        *,
        action_pruning_cfg: Mapping[str, Any] | None,
        supervision_phase: Mapping[str, float | bool],
        replay_mix_alpha: float,
        replay_guidance_cfg: Mapping[str, Any],
    ) -> SubgraphBatchBuildOptions:
        proposal_prior_cfg = self._auxiliary_proposal_cfg()["prior"]
        return SubgraphBatchBuildOptions(
            include_edge_question_similarity=self._action_pruning_enabled(
                action_pruning_cfg
            )
            or float(proposal_prior_cfg.get("prior_question_similarity_weight", 0.0))
            != 0.0,
            include_oracle_distance=float(
                proposal_prior_cfg.get("oracle_answer_distance_weight", 0.0)
            )
            != 0.0,
            include_teacher_banks=self._sequence_supervision_enabled(supervision_phase)
            or (
                float(replay_mix_alpha) > 0.0
                and bool(replay_guidance_cfg.get("add_shortest_path_guidance", False))
            ),
        )

    @staticmethod
    def _replay_batch_build_options() -> SubgraphBatchBuildOptions:
        return SubgraphBatchBuildOptions(
            include_edge_question_similarity=False,
            include_oracle_distance=False,
            include_teacher_banks=False,
        )

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
        prepared_batch: SubgraphBatch,
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
        prepared_batch: SubgraphBatch,
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
        loss_output: SubgraphDetailedBalanceLossOutput,
        sample_batch: SubgraphTrajectorySampleBatch,
        total_loss: torch.Tensor,
        rollouts_per_graph: int,
        sampling_temperature: float,
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
        mean_terminal_answer_candidate_rate = (
            (sample_batch.terminal_answer_candidate_counts > 0)
            .to(dtype=torch.float32)
            .mean()
        )
        return {
            "loss": total_loss.detach(),
            "objective/db_loss": loss_output.db_loss.detach(),
            "objective/db_residual_abs": loss_output.residual_abs.detach(),
            "objective/db_residual_variance": loss_output.residual_variance.detach(),
            "objective/db_root_abs": loss_output.root_abs.detach(),
            "witness/gold_answer_in_state_rate": loss_output.success_rate.detach(),
            "witness/answer_candidate_count": (
                loss_output.average_terminal_answer_candidate_count.detach()
            ),
            "witness/gold_answer_in_state_count": (
                loss_output.average_terminal_gold_answer_count.detach()
            ),
            "witness/anchor_component_count": (
                loss_output.average_terminal_component_count.detach()
            ),
            "flow/log_z_mean": loss_output.log_z_mean.detach(),
            "flow/log_z_variance": loss_output.log_z_variance.detach(),
            "witness/selected_edge_count": mean_selected_edges.detach(),
            "rollout/stop_step": mean_termination_action_step.detach(),
            "witness/nonempty_answer_candidate_rate": (
                mean_terminal_answer_candidate_rate.detach()
            ),
            "rollout/rollouts_per_graph": float(rollouts_per_graph),
            "rollout/sampling_temperature": float(sampling_temperature),
            "reward/hit_bonus": float(
                self.cfg.training_cfg["answer_reward"]["hit_bonus"]
            ),
            "reward/frontier_bonus": float(
                self.cfg.training_cfg["answer_reward"]["frontier_bonus"]
            ),
            "reward/coverage_bonus": float(
                self.cfg.training_cfg["answer_reward"]["coverage_bonus"]
            ),
            "reward/size_penalty": float(
                self.cfg.training_cfg["answer_reward"]["size_penalty"]
            ),
            "reward/component_penalty": float(
                self.cfg.training_cfg["answer_reward"]["component_penalty"]
            ),
        }

    @staticmethod
    def _raise_on_nonfinite_training_loss(
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
            "Non-finite training loss detected. Check Detailed Balance residuals and reward inputs."
        )

    def run_step(
        self,
        *,
        batch: TrajectoryBatch,
        sampling_temperature: float,
        replay_mix_alpha: float,
        supervision_phase: Mapping[str, float | bool],
        effective_pass: float | None,
    ) -> TrainingStepResult:
        replay_cfg = self._auxiliary_replay_cfg()
        replay_guidance_cfg = self._auxiliary_replay_guidance_cfg()
        action_pruning_cfg = resolve_action_pruning_cfg(self.cfg.training_cfg)
        rollouts_per_graph = int(self.cfg.training_cfg["rollouts_per_graph"])
        prepared_batch = self.policy.prepare_batch(
            batch,
            build_options=self._training_batch_build_options(
                action_pruning_cfg=action_pruning_cfg,
                supervision_phase=supervision_phase,
                replay_mix_alpha=replay_mix_alpha,
                replay_guidance_cfg=replay_guidance_cfg,
            ),
        )
        trajectory_batch = batch.without_raw_features()
        if not bool(replay_cfg.get("enabled", False)):
            batch = trajectory_batch
        sample_batch = self.sampler.sample(
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=rollouts_per_graph,
            temperature=sampling_temperature,
            action_pruning=action_pruning_cfg,
        )
        online_loss_output = self.loss_fn.compute(sample_batch)
        teacher_supervision_enabled = self._sequence_supervision_enabled(
            supervision_phase
        )
        teacher_supervision_output = SequenceSupervisionLossOutput(
            imitation_loss=torch.zeros(
                (), device=prepared_batch.device, dtype=torch.float32
            ),
            success_action_loss=torch.zeros(
                (), device=prepared_batch.device, dtype=torch.float32
            ),
            prefix_count=0,
            sequence_count=0,
            positive_edge_count=0,
            candidate_edge_count=0,
        )
        if teacher_supervision_enabled:
            teacher_supervision_output = compute_sequence_supervision_losses(
                policy=self.policy,
                prepared_batch=prepared_batch,
                sequence_banks=prepared_batch.graph_teacher_sequence_bank,
            )
        teacher_supervision_loss = (
            float(supervision_phase["imitation_weight"])
            * teacher_supervision_output.imitation_loss
            + float(supervision_phase["success_action_weight"])
            * teacher_supervision_output.success_action_loss
        )
        if bool(replay_cfg.get("enabled", False)):
            self.success_replay_buffer.add_successful_trajectories(
                batch=batch,
                sample_batch=sample_batch,
            )
        replay_payload = None
        replay_loss_output: SubgraphDetailedBalanceLossOutput | None = None
        replay_expand_imitation_loss: torch.Tensor | None = None
        replay_success_action_loss: torch.Tensor | None = None
        replay_supervision_output = SequenceSupervisionLossOutput(
            imitation_loss=torch.zeros(
                (), device=prepared_batch.device, dtype=torch.float32
            ),
            success_action_loss=torch.zeros(
                (), device=prepared_batch.device, dtype=torch.float32
            ),
            prefix_count=0,
            sequence_count=0,
            positive_edge_count=0,
            candidate_edge_count=0,
        )
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
            replay_prepared_batch = self.policy.prepare_batch(
                replay_batch,
                build_options=self._replay_batch_build_options(),
            )
            replay_sample_batch = self.sampler.teacher_force(
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
            replay_sequence_banks = tuple(
                (tuple(int(edge_id) for edge_id in edge_ids),)
                for edge_ids in replay_sequences
            )
            replay_success_edge_banks = tuple(
                (tuple(sorted({int(edge_id) for edge_id in edge_ids})),)
                for edge_ids in replay_sequences
            )
            replay_supervision_enabled = teacher_supervision_enabled
            if replay_supervision_enabled:
                replay_supervision_output = compute_sequence_supervision_losses(
                    policy=self.policy,
                    prepared_batch=replay_prepared_batch,
                    sequence_banks=replay_sequence_banks,
                    success_edge_banks=replay_success_edge_banks,
                )
            replay_branch_loss = (
                float(supervision_phase["db_weight"]) * replay_loss_output.loss
            )
            if replay_expand_imitation_weight > 0.0:
                expand_imitation_loss, replay_expand_imitation_stats = (
                    compute_expand_imitation_loss(
                        policy=self.policy,
                        prepared_batch=replay_prepared_batch,
                        sample_batch=replay_sample_batch,
                        from_anchor_bonus=float(
                            replay_guidance_cfg.get(
                                "expand_imitation_from_anchor_bonus", 0.0
                            )
                        ),
                        answer_finish_bonus=float(
                            replay_guidance_cfg.get(
                                "expand_imitation_answer_finish_bonus", 0.0
                            )
                        ),
                    )
                )
                replay_expand_imitation_loss = expand_imitation_loss
                replay_branch_loss = (
                    replay_branch_loss
                    + replay_expand_imitation_weight * expand_imitation_loss
                )
            if replay_supervision_enabled:
                replay_success_action_loss = (
                    replay_supervision_output.success_action_loss
                )
                replay_branch_loss = (
                    replay_branch_loss
                    + float(supervision_phase["imitation_weight"])
                    * replay_supervision_output.imitation_loss
                    + float(supervision_phase["success_action_weight"])
                    * replay_supervision_output.success_action_loss
                )
            total_loss = (
                teacher_supervision_loss
                + (1.0 - float(replay_mix_alpha))
                * (float(supervision_phase["db_weight"]) * online_loss_output.loss)
                + float(replay_mix_alpha) * replay_branch_loss
            )
        else:
            total_loss = (
                teacher_supervision_loss
                + float(supervision_phase["db_weight"]) * online_loss_output.loss
            )
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
        )
        metrics["replay/mix_alpha"] = float(replay_mix_alpha)
        metrics["supervision/enabled"] = float(bool(supervision_phase["enabled"]))
        metrics["supervision/warmup_active"] = float(
            bool(supervision_phase["warmup_active"])
        )
        metrics["supervision/db_weight"] = float(supervision_phase["db_weight"])
        metrics["supervision/imitation_weight"] = float(
            supervision_phase["imitation_weight"]
        )
        metrics["supervision/success_action_weight"] = float(
            supervision_phase["success_action_weight"]
        )
        metrics["supervision/loss"] = teacher_supervision_loss.detach()
        metrics["supervision/imitation_loss"] = (
            teacher_supervision_output.imitation_loss.detach()
        )
        metrics["supervision/success_action_loss"] = (
            teacher_supervision_output.success_action_loss.detach()
        )
        metrics["supervision/prefix_count"] = float(
            teacher_supervision_output.prefix_count
        )
        metrics["supervision/sequence_count"] = float(
            teacher_supervision_output.sequence_count
        )
        metrics["supervision/positive_edge_count"] = float(
            teacher_supervision_output.positive_edge_count
        )
        metrics["supervision/candidate_edge_count"] = float(
            teacher_supervision_output.candidate_edge_count
        )
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
        metrics["auxiliary/proposal_feature_hints_active"] = float(
            any(
                float(value) != 0.0
                for value in self._auxiliary_proposal_cfg()["prior"].values()
            )
        )
        metrics["auxiliary/replay_enabled"] = float(
            bool(replay_cfg.get("enabled", False))
        )
        if replay_loss_output is not None:
            metrics["replay/loss"] = replay_loss_output.loss.detach()
            metrics["replay/witness/gold_answer_in_state_rate"] = (
                replay_loss_output.success_rate.detach()
            )
            metrics["replay/witness/answer_candidate_count"] = (
                replay_loss_output.average_terminal_answer_candidate_count.detach()
            )
            metrics["replay/witness/gold_answer_in_state_count"] = (
                replay_loss_output.average_terminal_gold_answer_count.detach()
            )
        if replay_expand_imitation_loss is not None:
            metrics["replay/imitation_loss"] = replay_expand_imitation_loss.detach()
        if replay_success_action_loss is not None:
            metrics["replay/success_action_loss"] = replay_success_action_loss.detach()
            metrics["replay/supervision_prefix_count"] = float(
                replay_supervision_output.prefix_count
            )
            metrics["replay/supervision_positive_edge_count"] = float(
                replay_supervision_output.positive_edge_count
            )
        if effective_pass is not None:
            metrics["rollout/effective_pass"] = float(effective_pass)
        return TrainingStepResult(total_loss=total_loss, metrics=metrics)


__all__ = ["SubgraphTrainingOrchestrator", "TrainingStepResult"]
