from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.data.schema import RetrievalBatch
from src.eval.metrics import (
    UnionSubgraphMasks,
    compute_distribution_expectations,
    compute_exploration_diversity,
    compute_high_reward_discovery,
    compute_union_subgraph_masks,
)
from src.models.losses import SubTrajectoryBalanceLoss
from src.models.mask_debug import log_mask_debug_summaries
from src.models.policy import Policy
from src.models.replay import (
    OnlineReplayBuffer,
    ReplayConfig,
    TrajectoryTrace,
    residual_priority,
)
from src.models.reward import RewardModel
from src.models.rollout import RolloutBatch, RolloutEngine
from src.models.teacher_guidance import TeacherGuidance, TeacherGuidanceConfig
from src.models.training_schedule import (
    CurriculumSchedule,
    CurriculumScheduleConfig,
)
from src.utils.logging_utils import get_logger
from src.utils.optimization_utils import build_optimizer_and_scheduler


_DEFAULT_EVAL_BUDGETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_WEIGHTS_ONLY_FALLBACK_EXCEPTIONS = (pickle.UnpicklingError, RuntimeError)
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_rollout_chunk_size(chunk_size: int | None, total_rollouts: int) -> int:
    if total_rollouts < 1:
        raise ValueError(f"num_rollout must be >= 1, got {total_rollouts}.")
    if chunk_size is None:
        return total_rollouts
    resolved = int(chunk_size)
    if resolved < 1:
        raise ValueError(f"rollout chunk size must be >= 1, got {resolved}.")
    return min(resolved, total_rollouts)


def _normalize_eval_budgets(eval_budgets: Sequence[int] | None) -> tuple[int, ...]:
    raw = _DEFAULT_EVAL_BUDGETS if eval_budgets is None else eval_budgets
    return tuple(sorted(int(k) for k in raw if int(k) >= 1))


def _flatten_metric_groups(
    groups: dict[str, dict[str, float]], *, prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}/{group_name}/{metric_name}": value
        for group_name, metrics in groups.items()
        for metric_name, value in metrics.items()
    }


def _safe_tensor_variance(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    return values.var(unbiased=True)

# ---------------------------------------------------------------------------
# GFlowNetModule
# ---------------------------------------------------------------------------


class GFlowNetModule(LightningModule):
    """GFlowNet Lightning module trained with forward-looking SubTB."""

    def __init__(
        self,
        *,
        max_steps: int = 4,
        num_rollout: int = 8,
        eval_num_rollout: int | None = None,
        rollout_chunk_size: int | None = None,
        eval_rollout_chunk_size: int | None = None,
        eval_budgets: Sequence[int] | None = None,
        temperature: float = 1.0,
        temperature_schedule: dict[str, Any] | None = None,
        backbone: dict[str, Any] | None = None,
        policy_hidden_dim: int = 1024,
        action_head: dict[str, Any] | None = None,
        edge_scorer: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        teacher: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
        replay: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.num_rollout = int(num_rollout)
        if self.num_rollout < 1:
            raise ValueError(f"num_rollout must be >= 1, got {self.num_rollout}.")
        self.eval_num_rollout = (
            self.num_rollout if eval_num_rollout is None else int(eval_num_rollout)
        )
        if self.eval_num_rollout < 1:
            raise ValueError(
                f"eval_num_rollout must be >= 1, got {self.eval_num_rollout}."
            )

        self.rollout_chunk_size = _resolve_rollout_chunk_size(
            rollout_chunk_size, self.num_rollout
        )
        self.eval_rollout_chunk_size = _resolve_rollout_chunk_size(
            eval_rollout_chunk_size
            if eval_rollout_chunk_size is not None
            else rollout_chunk_size,
            self.eval_num_rollout,
        )
        self.eval_budgets = _normalize_eval_budgets(eval_budgets)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}.")

        temperature_schedule_cfg = dict(temperature_schedule or {})
        self.temperature_start = float(
            temperature_schedule_cfg.pop("start", self.temperature)
        )
        self.temperature_end = float(
            temperature_schedule_cfg.pop("end", self.temperature)
        )
        self.temperature_warmup_steps = int(
            temperature_schedule_cfg.pop("warmup_steps", 0)
        )
        if temperature_schedule_cfg:
            raise ValueError(
                f"Unsupported temperature_schedule keys: "
                f"{sorted(temperature_schedule_cfg.keys())}."
            )
        if self.temperature_start <= 0.0:
            raise ValueError(
                f"temperature_schedule.start must be > 0, got {self.temperature_start}."
            )
        if self.temperature_end <= 0.0:
            raise ValueError(
                f"temperature_schedule.end must be > 0, got {self.temperature_end}."
            )
        if self.temperature_warmup_steps < 0:
            raise ValueError(
                f"temperature_schedule.warmup_steps must be >= 0, "
                f"got {self.temperature_warmup_steps}."
            )

        self.policy = Policy(
            backbone_cfg=backbone or {},
            hidden_dim=policy_hidden_dim,
            max_steps=max_steps,
            action_head_cfg=action_head,
            edge_scorer_cfg=edge_scorer,
        )
        self.reward_model = RewardModel(**(reward or {}))
        self.rollout_engine = RolloutEngine(max_steps=max_steps)

        loss_cfg = dict(loss or {})
        loss_cfg.setdefault("max_trajectory_len", max_steps + 1)
        self.loss_fn = SubTrajectoryBalanceLoss(**loss_cfg)

        teacher_cfg = TeacherGuidanceConfig(**(teacher or {}))
        self.teacher_cfg = teacher_cfg
        self.teacher_guidance = (
            TeacherGuidance(
                mode=teacher_cfg.mode,
                score_exponent=teacher_cfg.score_exponent,
                undirected=self.policy.undirected,
                fallback_to_policy=teacher_cfg.fallback_to_policy,
            )
            if teacher_cfg.enabled
            else None
        )

        schedule_cfg = CurriculumScheduleConfig(**(schedule or {}))
        self.schedule = CurriculumSchedule(
            warmup_steps=schedule_cfg.warmup_steps,
            decay_steps=schedule_cfg.decay_steps,
            initial_teacher_prob=schedule_cfg.initial_teacher_prob,
            final_teacher_prob=schedule_cfg.final_teacher_prob,
        )

        replay_cfg = ReplayConfig(**(replay or {}))
        self.replay_cfg = replay_cfg
        self.replay_buffer = (
            OnlineReplayBuffer(capacity=replay_cfg.capacity)
            if replay_cfg.enabled
            else None
        )

        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluation_budgets(self) -> tuple[int, ...]:
        budgets = {k for k in self.eval_budgets if k <= self.eval_num_rollout}
        budgets.add(self.eval_num_rollout)
        return tuple(sorted(budgets))

    def _debug_batch_masks(
        self, batch: RetrievalBatch, *, stage: str, batch_idx: int
    ) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
            return
        if batch_idx != 0:
            return
        log_mask_debug_summaries(
            batch, stage=stage, batch_idx=batch_idx, max_graphs=None
        )

    def _replay_enabled(self) -> bool:
        return (
            self.replay_cfg.enabled
            and self.replay_buffer is not None
            and self.replay_cfg.loss_coef > 0.0
        )

    def _replay_buffer_size(self) -> int:
        return len(self.replay_buffer) if self.replay_buffer is not None else 0

    def clear_replay_buffer(self) -> None:
        if self.replay_buffer is not None:
            self.replay_buffer.clear()

    def _global_step_int(self) -> int:
        try:
            return int(getattr(self, "global_step"))
        except (AttributeError, TypeError, RuntimeError):
            return 0

    # [FIX] 实现 on_fit_start 钩子，使 reset_on_fit_start 真正生效。
    # 原代码中 ReplayConfig.reset_on_fit_start 字段有定义但从未被读取，
    # 导致跨 run 的旧污染数据永远留在 buffer 中被优先采样。
    def on_fit_start(self) -> None:
        if not self._replay_enabled() or self.replay_buffer is None:
            return
        if not self.replay_cfg.reset_on_fit_start:
            return
        stale_count = self._replay_buffer_size()
        self.clear_replay_buffer()
        if stale_count > 0:
            log.warning(
                "Cleared %d stale replay traces at fit start to avoid mixing "
                "contaminated rewards with the current reward configuration.",
                stale_count,
            )

    def _validate_replay_runtime(self) -> None:
        if not self._replay_enabled():
            return
        world_size = int(getattr(self.trainer, "world_size", 1) or 1)
        if world_size != 1:
            raise RuntimeError(
                "Trace-based prioritized replay currently supports only "
                f"single-process training. Got world_size={world_size}."
            )

    def _build_replay_traces(
        self,
        rollouts: Sequence[RolloutBatch],
        loss_outputs: Sequence[Any],
    ) -> list[TrajectoryTrace]:
        if not self._replay_enabled():
            return []
        traces: list[TrajectoryTrace] = []
        for rollout, loss_output in zip(rollouts, loss_outputs):
            if rollout.trajectory_traces is None:
                raise RuntimeError(
                    "Replay is enabled but rollout traces were not recorded. "
                    "Ensure training rollouts carry sample_id and trace metadata."
                )
            per_traj_loss = getattr(loss_output, "per_trajectory_loss", None)
            if per_traj_loss is None:
                raise RuntimeError(
                    "Replay requires per-trajectory losses to compute residual priorities."
                )
            for idx, trace in enumerate(rollout.trajectory_traces):
                if trace.source != "online":
                    continue
                priority = float(
                    residual_priority(
                        per_traj_loss[idx],
                        epsilon=self.replay_cfg.priority_epsilon,
                        exponent=self.replay_cfg.priority_exponent,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                traces.append(
                    TrajectoryTrace(
                        sample_id=trace.sample_id,
                        edge_trace_local=trace.edge_trace_local,
                        traj_len=trace.traj_len,
                        terminal_log_reward=float(
                            rollout.terminal_log_rewards[idx].detach().cpu()
                        ),
                        priority=float(priority),
                        insert_step=self._global_step_int(),
                        source=trace.source,
                        positive_edge_hit_count=trace.positive_edge_hit_count,
                        positive_prefix_hit_len=trace.positive_prefix_hit_len,
                        relation_only_score_mean=trace.relation_only_score_mean,
                        relation_only_score_max=trace.relation_only_score_max,
                        final_score_mean=trace.final_score_mean,
                        teacher_forced_action_count=trace.teacher_forced_action_count,
                    )
                )
        return traces

    def _store_replay_traces(
        self,
        rollouts: Sequence[RolloutBatch],
        loss_outputs: Sequence[Any],
    ) -> int:
        if not self._replay_enabled() or self.replay_buffer is None:
            return 0
        traces = self._build_replay_traces(rollouts, loss_outputs)
        if traces:
            self.replay_buffer.add_many(traces)
        return len(traces)

    def _should_run_replay(self) -> bool:
        if not self._replay_enabled():
            return False
        if self._global_step_int() < self.replay_cfg.warmup_steps:
            return False
        return self._replay_buffer_size() >= self.replay_cfg.min_size

    def _sample_replay_rollout(self) -> tuple[RolloutBatch, torch.Tensor, torch.Tensor]:
        if not self._replay_enabled() or self.replay_buffer is None:
            raise RuntimeError("Replay buffer is not enabled.")
        replay_sample = self.replay_buffer.sample(
            self.replay_cfg.sample_size,
            current_step=self._global_step_int(),
            age_decay=self.replay_cfg.age_decay,
            importance_sampling_exponent=self.replay_cfg.importance_sampling_exponent,
        )
        if not replay_sample.traces:
            raise RuntimeError("Replay buffer sampling returned no traces.")

        datamodule = getattr(self.trainer, "datamodule", None)
        build_batch = getattr(datamodule, "build_train_batch_from_ids", None)
        if not callable(build_batch):
            raise RuntimeError(
                "Trace-based prioritized replay requires "
                "trainer.datamodule.build_train_batch_from_ids(...)."
            )

        replay_batch = build_batch(
            [trace.sample_id for trace in replay_sample.traces]
        ).to(self.device)
        rollout = self.rollout_engine.replay_trajectories(
            policy=self.policy,
            base_graph=replay_batch,
            reward_model=self.reward_model,
            traces=replay_sample.traces,
            collect_terminal_state=False,
        )
        return (
            rollout,
            replay_sample.indices,
            replay_sample.importance_weights.to(device=self.device),
        )

    @contextmanager
    def _eval_mode(self) -> Generator[None, None, None]:
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            if was_training:
                self.train()

    def _is_optimizer_step_due(self, batch_idx: int, accumulation_batches: int) -> bool:
        if (batch_idx + 1) % accumulation_batches == 0:
            return True
        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        if (
            isinstance(num_training_batches, int)
            and num_training_batches > 0
            and (batch_idx + 1) == num_training_batches
            and (num_training_batches % accumulation_batches != 0)
        ):
            return True
        return False

    def _step_schedulers_by_interval(self, target_interval: str) -> None:
        configs = getattr(self.trainer, "lr_scheduler_configs", [])
        for config in configs:
            if config.interval != target_interval:
                continue
            scheduler = config.scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor = config.monitor
                if monitor is None:
                    warnings.warn(
                        "ReduceLROnPlateau is active but no 'monitor' was specified "
                        "in configure_optimizers. Scheduler step skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                metric_val = self.trainer.callback_metrics.get(monitor)
                if metric_val is not None:
                    scheduler.step(metric_val)
            else:
                scheduler.step()

    def _high_reward_threshold(self) -> float:
        return 0.0

    def _training_rollout_temperature(self) -> float:
        if self.temperature_warmup_steps == 0:
            return self.temperature_end
        progress = min(
            max(float(self._global_step_int()), 0.0)
            / float(self.temperature_warmup_steps),
            1.0,
        )
        return (
            self.temperature_start
            + (self.temperature_end - self.temperature_start) * progress
        )

    # ------------------------------------------------------------------
    # Rollout pipeline
    # ------------------------------------------------------------------

    def _iter_chunk_sizes(self, total_rollouts: int, chunk_size: int) -> list[int]:
        sizes: list[int] = []
        remaining = total_rollouts
        while remaining > 0:
            sizes.append(min(chunk_size, remaining))
            remaining -= chunk_size
        return sizes

    def _teacher_force_prob(self) -> float:
        if self.teacher_guidance is None:
            return 0.0
        return self.schedule.teacher_force_prob(self._global_step_int())

    def _yield_rollout_chunks(
        self,
        *,
        batch: RetrievalBatch,
        total_rollouts: int,
        chunk_size: int,
        temperature: float,
        collect_terminal_state: bool,
        terminal_state_device: torch.device | str | None = None,
        teacher_force_prob: float = 0.0,
    ) -> Iterator[list[RolloutBatch]]:
        for size in self._iter_chunk_sizes(total_rollouts, chunk_size):
            yield self.rollout_engine.run_exploration(
                policy=self.policy,
                base_graph=batch,
                reward_model=self.reward_model,
                num_rollouts=size,
                temperature=temperature,
                collect_terminal_state=collect_terminal_state,
                terminal_state_device=terminal_state_device,
                teacher_guidance=self.teacher_guidance,
                teacher_force_prob=teacher_force_prob,
            )

    def _generate_terminal_rollouts(
        self,
        batch: RetrievalBatch,
        num_rollouts: int,
        temperature: float | None = None,
    ) -> list[RolloutBatch]:
        resolved_temp = self.temperature if temperature is None else float(temperature)
        return [
            rollout
            for chunk in self._yield_rollout_chunks(
                batch=batch,
                total_rollouts=num_rollouts,
                chunk_size=self.eval_rollout_chunk_size,
                temperature=resolved_temp,
                collect_terminal_state=True,
                terminal_state_device="cpu",
            )
            for rollout in chunk
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: RetrievalBatch, batch_idx: int) -> None:
        self._validate_replay_runtime()
        self._debug_batch_masks(batch, stage="train", batch_idx=batch_idx)
        optimizer = self.optimizers()
        accumulation_batches = getattr(self.trainer, "accumulate_grad_batches", 1)
        optimizer_step_due = self._is_optimizer_step_due(
            batch_idx, accumulation_batches
        )
        global_step = self._global_step_int()
        self.policy.update_training_schedule(global_step=global_step)

        total_loss: float = 0.0
        total_fl_subtb_loss: float = 0.0
        total_rm_loss: float = 0.0
        total_log_z_mean: float = 0.0
        total_log_reward_mean: float = 0.0
        total_step_shaping_mean: float = 0.0
        total_terminal_flow_vs_reward: float = 0.0
        total_terminal_flow_mean: float = 0.0

        terminal_reward_samples: list[torch.Tensor] = []
        trajectory_length_samples: list[torch.Tensor] = []
        forced_stop_samples: list[torch.Tensor] = []
        log_z_samples: list[torch.Tensor] = []
        relation_only_score_samples: list[torch.Tensor] = []
        residual_score_samples: list[torch.Tensor] = []
        teacher_action_ratio_samples: list[torch.Tensor] = []

        rollout_temperature = self._training_rollout_temperature()
        teacher_force_prob = self._teacher_force_prob()
        exploration_loss_value: float = 0.0
        replay_loss_value: float = 0.0
        replay_fl_subtb_loss: float = 0.0
        replay_rm_loss: float = 0.0
        replay_log_reward_mean: float = 0.0
        replay_sample_count: float = 0.0
        replay_store_count = 0
        teacher_traj_count = 0.0
        mixed_traj_count = 0.0
        online_traj_count = 0.0

        for rollouts in self._yield_rollout_chunks(
            batch=batch,
            total_rollouts=self.num_rollout,
            chunk_size=self.rollout_chunk_size,
            temperature=rollout_temperature,
            collect_terminal_state=False,
            teacher_force_prob=teacher_force_prob,
        ):
            loss_outputs = [self.loss_fn(r) for r in rollouts]
            chunk_loss = torch.stack([o.loss for o in loss_outputs]).sum() / self.num_rollout
            self.manual_backward(chunk_loss / accumulation_batches)

            with torch.no_grad():
                n = self.num_rollout
                scaled_loss_value = float((chunk_loss / accumulation_batches).item())
                total_loss += scaled_loss_value
                exploration_loss_value += scaled_loss_value
                total_fl_subtb_loss += (
                    torch.stack([o.metric("fl_subtb_loss") for o in loss_outputs]).sum()
                    / n
                ).item()
                total_rm_loss += (
                    torch.stack(
                        [o.metric("reward_matching_loss") for o in loss_outputs]
                    ).sum()
                    / n
                ).item()
                total_log_z_mean += (
                    torch.stack([o.metric("log_z_mean") for o in loss_outputs]).sum() / n
                ).item()
                total_log_reward_mean += (
                    torch.stack([o.metric("log_reward_mean") for o in loss_outputs]).sum()
                    / n
                ).item()
                total_step_shaping_mean += (
                    torch.stack(
                        [o.metric("step_log_shaping_mean") for o in loss_outputs]
                    ).sum()
                    / n
                ).item()
                total_terminal_flow_vs_reward += (
                    torch.stack(
                        [o.metric("terminal_flow_vs_reward") for o in loss_outputs]
                    ).sum()
                    / n
                ).item()
                total_terminal_flow_mean += (
                    torch.stack(
                        [o.metric("terminal_flow_mean") for o in loss_outputs]
                    ).sum()
                    / n
                ).item()
                terminal_reward_samples.append(
                    torch.cat([r.terminal_log_rewards.detach().float() for r in rollouts])
                )
                trajectory_length_samples.append(
                    torch.cat([r.traj_len.detach().float() for r in rollouts])
                )
                for r in rollouts:
                    forced_stop_samples.append(
                        r.traj_len.eq(self.rollout_engine.max_steps).float()
                    )
                    if r.teacher_forced_action_count is not None:
                        teacher_action_ratio_samples.append(
                            r.teacher_forced_action_count.detach().float()
                            / r.traj_len.detach().float().clamp_min(1.0)
                        )
                    if r.selected_relation_only_logits is not None and r.selected_edge_ids is not None:
                        relation_mask = r.selected_edge_ids.ge(0)
                        if bool(relation_mask.any().item()):
                            relation_only_score_samples.append(
                                r.selected_relation_only_logits[relation_mask].detach().float()
                            )
                    if (
                        r.selected_final_logits is not None
                        and r.selected_relation_only_logits is not None
                        and r.selected_edge_ids is not None
                    ):
                        residual_mask = r.selected_edge_ids.ge(0)
                        if bool(residual_mask.any().item()):
                            residual_score_samples.append(
                                (
                                    r.selected_final_logits[residual_mask]
                                    - r.selected_relation_only_logits[residual_mask]
                                )
                                .detach()
                                .float()
                            )
                    if r.trajectory_traces is not None:
                        for trace in r.trajectory_traces:
                            if trace.source == "teacher":
                                teacher_traj_count += 1.0
                            elif trace.source == "mixed":
                                mixed_traj_count += 1.0
                            else:
                                online_traj_count += 1.0
                log_z_samples.append(
                    torch.stack(
                        [o.metric("log_z_mean").detach().float() for o in loss_outputs]
                    )
                )
            replay_store_count += self._store_replay_traces(rollouts, loss_outputs)
            del rollouts, loss_outputs

        if self._should_run_replay():
            replay_rollout, replay_indices, replay_weights = self._sample_replay_rollout()
            replay_output = self.loss_fn(
                replay_rollout,
                trajectory_weights=replay_weights,
            )
            replay_scaled_loss = replay_output.loss * self.replay_cfg.loss_coef
            self.manual_backward(replay_scaled_loss / accumulation_batches)

            with torch.no_grad():
                replay_loss_value = float(
                    (replay_scaled_loss / accumulation_batches).item()
                )
                replay_fl_subtb_loss = float(
                    replay_output.metric("fl_subtb_loss").item()
                )
                replay_rm_loss = float(
                    replay_output.metric("reward_matching_loss").item()
                )
                replay_log_reward_mean = float(
                    replay_output.metric("log_reward_mean").item()
                )
                replay_sample_count = float(replay_rollout.traj_len.numel())
                total_loss += replay_loss_value
                if replay_output.per_trajectory_loss is None:
                    raise RuntimeError(
                        "Replay output is missing per-trajectory losses for priority updates."
                    )
                self.replay_buffer.update_priorities(
                    replay_indices.detach().cpu(),
                    residual_priority(
                        replay_output.per_trajectory_loss.detach().cpu(),
                        epsilon=self.replay_cfg.priority_epsilon,
                        exponent=self.replay_cfg.priority_exponent,
                    ),
                )

        if optimizer_step_due:
            clip_val = getattr(self.trainer, "gradient_clip_val", None)
            if clip_val is not None:
                self.clip_gradients(
                    optimizer,  # type: ignore[arg-type]
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm=getattr(
                        self.trainer, "gradient_clip_algorithm", "norm"
                    ),
                )
            optimizer.step()
            optimizer.zero_grad()
            self._step_schedulers_by_interval("step")

        dev = self.device
        all_terminal_rewards = (
            torch.cat(terminal_reward_samples)
            if terminal_reward_samples
            else torch.zeros(1, device=dev)
        )
        all_trajectory_lengths = (
            torch.cat(trajectory_length_samples)
            if trajectory_length_samples
            else torch.zeros(1, device=dev)
        )
        all_forced_stops = (
            torch.cat(forced_stop_samples)
            if forced_stop_samples
            else torch.zeros(1, device=dev)
        )
        all_log_z = (
            torch.cat(log_z_samples) if log_z_samples else torch.zeros(1, device=dev)
        )
        all_relation_only_scores = (
            torch.cat(relation_only_score_samples)
            if relation_only_score_samples
            else torch.zeros(1, device=dev)
        )
        all_residual_scores = (
            torch.cat(residual_score_samples)
            if residual_score_samples
            else torch.zeros(1, device=dev)
        )

        all_teacher_action_ratios = (
            torch.cat(teacher_action_ratio_samples)
            if teacher_action_ratio_samples
            else torch.zeros(1, device=dev)
        )
        high_reward_ratio = all_terminal_rewards.ge(self._high_reward_threshold()).float().mean()
        forced_stop_ratio = all_forced_stops.mean()
        current_lr = float(self.optimizers().param_groups[0]["lr"])  # type: ignore[union-attr]
        scorer = self.policy.expand_edge_scorer
        prior_scale_value = float(scorer.prior_scale.detach().item())
        residual_scale_value = float(getattr(scorer, "residual_scale", 0.0))
        schedule_phase_value = {"warmup": 0.0, "mix": 1.0, "online": 2.0}[
            self.schedule.phase(global_step)
        ]

        def _t(v: float) -> torch.Tensor:
            return torch.tensor(v, device=dev)

        self.log_dict(
            {
                "train/loss": _t(total_loss),
                "train/fl_subtb_loss": _t(total_fl_subtb_loss),
                "train/reward_matching_loss": _t(total_rm_loss),
                "train/log_z_mean": _t(total_log_z_mean),
                "train/log_z_variance": _safe_tensor_variance(all_log_z),
                "train/terminal_flow_mean": _t(total_terminal_flow_mean),
                "train/terminal_flow_vs_reward": _t(total_terminal_flow_vs_reward),
                "train/log_reward_mean": _t(total_log_reward_mean),
                "train/log_reward_variance": _safe_tensor_variance(
                    all_terminal_rewards
                ),
                "train/high_reward_ratio": high_reward_ratio,
                "train/step_log_shaping_mean": _t(total_step_shaping_mean),
                "train/exploration_loss": _t(exploration_loss_value),
                "train/teacher_force_prob": _t(teacher_force_prob),
                "train/teacher_phase": _t(schedule_phase_value),
                "train/teacher_trajectory_count": _t(teacher_traj_count),
                "train/mixed_trajectory_count": _t(mixed_traj_count),
                "train/online_trajectory_count": _t(online_traj_count),
                "train/teacher_action_ratio": all_teacher_action_ratios.mean(),
                "train/replay_loss": _t(replay_loss_value),
                "train/replay_fl_subtb_loss": _t(replay_fl_subtb_loss),
                "train/replay_reward_matching_loss": _t(replay_rm_loss),
                "train/replay_log_reward_mean": _t(replay_log_reward_mean),
                "train/replay_sample_count": _t(replay_sample_count),
                "train/replay_buffer_size": _t(float(self._replay_buffer_size())),
                "train/replay_store_count": _t(float(replay_store_count)),
                "train/relation_only_logit_mean": all_relation_only_scores.mean(),
                "train/residual_logit_mean": all_residual_scores.mean(),
                "train/residual_to_prior_ratio": _t(
                    float(all_residual_scores.abs().mean().item())
                    / max(float(all_relation_only_scores.abs().mean().item()), 1.0e-8)
                ),
                "train/edge_prior_scale": _t(prior_scale_value),
                "train/edge_residual_scale": _t(residual_scale_value),
                "train/trajectory_length_mean": all_trajectory_lengths.mean(),
                "train/forced_stop_ratio": forced_stop_ratio,
                "train/rollout_temperature": _t(rollout_temperature),
                "train/lr": _t(current_lr),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )

    def on_train_epoch_end(self) -> None:
        self._step_schedulers_by_interval("epoch")

    # ------------------------------------------------------------------
    # Inference & evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        batch: RetrievalBatch,
        num_rollouts: int = 1,
        temperature: float | None = None,
    ) -> UnionSubgraphMasks:
        with self._eval_mode():
            rollouts = self._generate_terminal_rollouts(
                batch, num_rollouts, temperature
            )
            return compute_union_subgraph_masks(rollouts, batch)

    @torch.no_grad()
    def evaluate_subgraph_retrieval(self, batch: RetrievalBatch) -> dict[str, Any]:
        with self._eval_mode():
            rollouts = self._generate_terminal_rollouts(
                batch, self.eval_num_rollout, self.temperature
            )
            return {
                "distribution": compute_distribution_expectations(rollouts, batch),
                "high_reward": compute_high_reward_discovery(
                    rollouts, batch, ks=self._evaluation_budgets()
                ),
                "diversity": compute_exploration_diversity(rollouts, batch),
            }

    def _shared_eval_step(self, batch: RetrievalBatch, prefix: str) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)
        self.log_dict(
            _flatten_metric_groups(results, prefix=prefix),
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix == "val"),
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        return results

    def validation_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        self._debug_batch_masks(batch, stage="val", batch_idx=batch_idx)
        return self._shared_eval_step(batch, prefix="val")

    def test_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        return self._shared_eval_step(batch, prefix="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return build_optimizer_and_scheduler(
            module=self,
            optimizer_cfg=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
        )

    def load_pretrained_weights(
        self, checkpoint_path: str, strict: bool = False
    ) -> tuple[list[str], list[str]]:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")
        allowed_suffixes = {".ckpt", ".pt", ".pth"}
        if os.path.splitext(checkpoint_path)[1].lower() not in allowed_suffixes:
            raise ValueError(
                f"Unsupported checkpoint extension for {checkpoint_path!r}. "
                f"Allowed: {allowed_suffixes}"
            )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except _WEIGHTS_ONLY_FALLBACK_EXCEPTIONS as exc:
            raise RuntimeError(
                f"Failed to load {checkpoint_path!r} with weights_only=True. "
                "If you trust the checkpoint source, load it manually with "
                "torch.load(path, map_location='cpu', weights_only=False) "
                "and pass the resulting state_dict to load_state_dict()."
            ) from exc
        return self.load_state_dict(
            checkpoint.get("state_dict", checkpoint), strict=strict
        )


__all__ = ["GFlowNetModule"]
