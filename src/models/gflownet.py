from __future__ import annotations

import os
import math
import pickle
import warnings
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast, List

from sympy import true
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
from src.models.losses import LossOutput, SubTrajectoryBalanceLoss
from src.models.policy import Policy
from src.models.reward import RewardModel
from src.models.rollout import RolloutBatch, RolloutEngine
from src.models.guidance import TeacherGuidance
from src.utils.logging_utils import get_logger
from src.utils.optimization_utils import build_optimizer_and_scheduler

log = get_logger(__name__)

class GFlowNetModule(LightningModule):
    """GFlowNet Lightning module trained with forward-looking SubTB."""

    def __init__(
        self,
        *,
        expand_budget: int = 4,
        num_rollout: int = 8,
        eval_num_rollout: int | None = None,
        rollout_chunk_size: int | None = None,
        eval_rollout_chunk_size: int | None = None,
        eval_budgets: List[int],
        temperature: float = 1.0,
        policy_hidden_dim: int = 1024,
        temperature_cfg: dict[str, Any] | None = None,
        backbone_cfg: dict[str, Any] | None = None,
        action_head_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
        reward_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        curriculum_cfg: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.num_rollout = int(num_rollout)
        self.eval_num_rollout = self.num_rollout if eval_num_rollout is None else int(eval_num_rollout)
        self.rollout_chunk_size = min(int(rollout_chunk_size), self.num_rollout) if rollout_chunk_size else self.num_rollout
        _eval_chunk = eval_rollout_chunk_size or rollout_chunk_size
        self.eval_rollout_chunk_size = min(int(_eval_chunk), self.eval_num_rollout) if _eval_chunk else self.eval_num_rollout
        self.eval_budgets = tuple(sorted({int(k) for k in (eval_budgets or [1, 2, 4, 8, 16, 32, 64]) if int(k) >= 1}))

        self.temperature = float(temperature)
        temperature_cfg = dict(temperature_cfg or {})
        self.temperature_start = float(temperature_cfg.pop("temperature_start", self.temperature))
        self.temperature_end = float(temperature_cfg.pop("temperature_end", self.temperature))
        self.temperature_warmup_steps = int(temperature_cfg.pop("temperature_warmup_steps", 0))

        curriculum_cfg = dict(curriculum_cfg or {})
        self.curriculum_warmup_steps = int(curriculum_cfg.pop("curriculum_warmup_steps", 0))
        self.curriculum_decay_steps = int(curriculum_cfg.pop("curriculum_decay_steps", 0))
        self.curriculum_initial_prob = float(curriculum_cfg.pop("curriculum_initial_prob", 1.0))
        self.curriculum_final_prob = float(curriculum_cfg.pop("curriculum_final_prob", 0.0))

        self.policy = Policy(
            backbone_cfg=backbone_cfg or {}, 
            hidden_dim=policy_hidden_dim, 
            action_head_cfg=action_head_cfg, 
            edge_scorer_cfg=edge_scorer_cfg
        )
        self.reward_model = RewardModel(**(reward_cfg or {}))
        self.rollout_engine = RolloutEngine(expand_budget=expand_budget)

        loss_cfg = dict(loss_cfg or {})
        loss_cfg.setdefault("max_trajectory_len", expand_budget + 1)
        self.loss_fn = SubTrajectoryBalanceLoss(**loss_cfg)
        
        teacher_cfg = dict(teacher_cfg or {})
        self.teacher_guidance = TeacherGuidance(score_exponent=teacher_cfg.pop("teacher_score_exponent", 1.0)) if teacher_cfg.pop("teacher_enabled", true) else None

        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        """Bind W&B metric families to a stable x-axis when available."""
        logger = self.logger
        if logger is None:
            return

        experiment = getattr(logger, "experiment", None)
        define_metric = getattr(experiment, "define_metric", None)
        if not callable(define_metric):
            return

        define_metric("trainer/global_step")
        for prefix in ("train/*", "val/*", "test/*"):
            define_metric(prefix, step_metric="trainer/global_step")

    def _is_optimizer_step_due(self, batch_idx: int, accumulation_batches: int) -> bool:
        if (batch_idx + 1) % accumulation_batches == 0: return True
        num_batches = getattr(self.trainer, "num_training_batches", None)
        return isinstance(num_batches, int) and num_batches > 0 and (batch_idx + 1) == num_batches
                
    def _training_rollout_temperature(self) -> float:
        if self.temperature_warmup_steps <= 0: return self.temperature_end
        progress = min(max(float(self.global_step), 0.0) / float(self.temperature_warmup_steps), 1.0)
        return self.temperature_start + (self.temperature_end - self.temperature_start) * progress

    def _teacher_force_prob(self) -> float:
        if self.teacher_guidance is None: 
            return 0.0
        step = max(float(self.global_step), 0.0)
        if step < self.curriculum_warmup_steps: 
            return self.curriculum_initial_prob
        if self.curriculum_decay_steps <= 0: 
            return self.curriculum_final_prob
        progress = min((step - self.curriculum_warmup_steps) / float(self.curriculum_decay_steps), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.curriculum_final_prob + (self.curriculum_initial_prob - self.curriculum_final_prob) * cosine_decay

    def _yield_rollout_chunks(
        self,
        *,
        batch: RetrievalBatch,
        total_rollouts: int,
        chunk_size: int,
        temperature: float,
        teacher_force_prob: float = 0.0,
    ) -> Iterator[list[RolloutBatch]]:
        remaining = total_rollouts
        while remaining > 0:
            current_size = min(chunk_size, remaining)
            remaining -= current_size
            yield self.rollout_engine.run_exploration(
                policy=self.policy,
                retrieval_batch=batch,
                reward_model=self.reward_model,
                num_rollouts=current_size,
                temperature=temperature,
                teacher_guidance=self.teacher_guidance,
                teacher_force_prob=teacher_force_prob,
            )
            
    def _generate_terminal_rollouts(self, batch: RetrievalBatch, num_rollouts: int, temperature: float | None = None) -> list[RolloutBatch]:
        return [r for chunk in self._yield_rollout_chunks(batch=batch, total_rollouts=num_rollouts, chunk_size=self.eval_rollout_chunk_size, temperature=temperature or self.temperature) for r in chunk]
    
    def training_step(self, batch: RetrievalBatch, batch_idx: int) -> None:
        opts = self.optimizers()
        optimizer = opts[0] if isinstance(opts, list) else opts
        accumulation_batches = getattr(self.trainer, "accumulate_grad_batches", 1)
        optimizer_step_due = self._is_optimizer_step_due(batch_idx, accumulation_batches)
        rollout_temperature = self._training_rollout_temperature()
        teacher_force_prob = self._teacher_force_prob()
        chunk_loss_outputs: list[LossOutput] = []
        metrics = {"rewards": [], "stops": [], "ratios": []}
        traj_counts = torch.zeros(3, dtype=torch.long, device=self.device)
        for rollouts in self._yield_rollout_chunks(
            batch=batch, total_rollouts=self.num_rollout, chunk_size=self.rollout_chunk_size, 
            temperature=rollout_temperature, teacher_force_prob=teacher_force_prob
        ):
            loss_outputs = [self.loss_fn(r) for r in rollouts]
            raw_sum = cast(torch.Tensor, sum(o.loss for o in loss_outputs))
            chunk_loss = raw_sum / self.num_rollout
            self.manual_backward(chunk_loss / accumulation_batches)
            with torch.no_grad():
                chunk_loss_outputs.append(LossOutput.aggregate(loss_outputs))
                c_rewards = torch.cat([r.stats.terminal_log_rewards for r in rollouts])
                c_lens = torch.cat([r.stats.traj_len for r in rollouts])
                c_teachers = torch.cat([r.stats.teacher_forced_action_count for r in rollouts])
                metrics["rewards"].append(c_rewards)
                metrics["stops"].append(c_lens == self.rollout_engine.expand_budget + 1)
                metrics["ratios"].append(c_teachers / c_lens.clamp_min(1))
                is_teacher = c_teachers >= c_lens
                is_online = c_teachers <= 0
                traj_counts[0] += is_teacher.sum()
                traj_counts[1] += is_online.sum()
                traj_counts[2] += (c_lens.numel() - is_teacher.sum() - is_online.sum())
        if optimizer_step_due:
            clip_val = getattr(self.trainer, "gradient_clip_val", None)
            if clip_val is not None:
                self.clip_gradients(
                    optimizer,  # type: ignore[arg-type]
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
                )
            optimizer.step()
            optimizer.zero_grad()
            self.lr_schedulers().step() # type: ignore[call-arg, union-attr]
        dev = self.device
        all_rewards = torch.cat(metrics["rewards"]).float() if metrics["rewards"] else torch.zeros(1, device=dev)
        all_stops = torch.cat(metrics["stops"]).float() if metrics["stops"] else torch.zeros(1, device=dev)
        all_ratios = torch.cat(metrics["ratios"]).float() if metrics["ratios"] else torch.zeros(1, device=dev)
        primary_loss_out = LossOutput.aggregate(chunk_loss_outputs)
        log_data: dict[str, Any] = {
            **primary_loss_out.prefixed_metrics("train"),
            
            "train/log_reward_variance": all_rewards.var(unbiased=True) if all_rewards.numel() > 1 else all_rewards.new_zeros(()),
            "train/high_reward_ratio": all_rewards.ge(0).float().mean(),
            "train/forced_stop_ratio": all_stops.mean(),
            
            "train/teacher_force_prob": teacher_force_prob,
            "train/teacher_trajectory_count": traj_counts[0].float(),
            "train/online_trajectory_count": traj_counts[1].float(),
            "train/mixed_trajectory_count": traj_counts[2].float(),
            "train/teacher_action_ratio": all_ratios.mean(),
            
            "train/edge_prior_scale": float(self.policy.expand_edge_scorer.prior_scale.detach()),
            "train/edge_residual_scale": float(getattr(self.policy.expand_edge_scorer, "residual_scale", 0.0)),
            "train/rollout_temperature": rollout_temperature,
            "train/lr": float(optimizer.param_groups[0]["lr"]),
        }
        self.log_dict(log_data, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.num_graphs)

    @torch.no_grad()
    def forward(self, batch: RetrievalBatch, num_rollouts: int = 1, temperature: float | None = None) -> UnionSubgraphMasks:
        rollouts = self._generate_terminal_rollouts(batch, num_rollouts, temperature)
        return compute_union_subgraph_masks(rollouts, batch)

    @torch.no_grad()
    def evaluate_subgraph_retrieval(self, batch: RetrievalBatch) -> dict[str, Any]:
        rollouts = self._generate_terminal_rollouts(batch, self.eval_num_rollout, self.temperature)
        return {
            "distribution": compute_distribution_expectations(rollouts, batch),
            "high_reward": compute_high_reward_discovery(rollouts, batch, ks=self.eval_budgets),
            "diversity": compute_exploration_diversity(rollouts, batch),
        }

    def _shared_eval_step(self, batch: RetrievalBatch, prefix: str) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)
        flat_metrics = {
            f"{prefix}/{group}/{metric}": value 
            for group, metrics in results.items() 
            for metric, value in metrics.items()
        }
        self.log_dict(
            flat_metrics, 
            on_step=False, on_epoch=True, prog_bar=(prefix == "val"), sync_dist=True, batch_size=batch.num_graphs
        )
        return results

    def validation_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        return self._shared_eval_step(batch, prefix="val")

    def test_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        return self._shared_eval_step(batch, prefix="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return build_optimizer_and_scheduler(module=self, optimizer_cfg=self.optimizer_cfg, scheduler_cfg=self.scheduler_cfg)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        if not os.path.isfile(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return self.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=strict)

__all__ = ["GFlowNetModule"]
