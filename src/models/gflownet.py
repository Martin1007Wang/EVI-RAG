from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.data.schema import RetrievalBatch
from src.eval.metrics import (
    build_union_context_graph,
    compute_distribution_expectations,
    compute_exploration_diversity,
    compute_high_reward_discovery,
)
from src.models.losses import TrajectoryBalanceLoss
from src.models.policy import Policy
from src.models.reward import RewardModel
from src.models.rollout import RolloutEngine
from src.models.schedules import SamplingTemperatureScheduler
from src.models.teacher_warmstart import ShortestPathTeacherWarmup
from src.utils.optimization_utils import build_optimizer_and_scheduler


class GFlowNetModule(LightningModule):
    _GPU_BATCH_FIELDS = frozenset(
        {
            "node_tokens",
            "edge_relation_tokens",
            "question_emb",
            "edge_index",
            "batch",
            "edge_batch",
            "is_anchor_mask",
            "is_target_mask",
        }
    )

    def __init__(
        self,
        *,
        max_steps: int = 20,
        num_rollout: int = 8,
        strict_on_policy: bool = True,
        sampling_temperature_schedule: dict[str, Any] | None = None,
        backbone: dict[str, Any] | None = None,
        policy_hidden_dim: int = 512,
        relation_prior: dict[str, Any] | None = None,
        answer_reward: dict[str, Any] | None = None,
        teacher_warmstart: dict[str, Any] | None = None,
        training_curriculum: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_rollout = num_rollout
        self.strict_on_policy = bool(strict_on_policy)

        self.policy = Policy(
            backbone_cfg=backbone or {},
            hidden_dim=policy_hidden_dim,
            relation_prior_cfg=relation_prior,
        )
        self.reward_model = RewardModel(**(answer_reward or {}))
        self.rollout_engine = RolloutEngine(max_steps=max_steps)
        self.loss_fn = TrajectoryBalanceLoss()
        self.temp_scheduler = SamplingTemperatureScheduler(
            **(sampling_temperature_schedule or {})
        )
        self.teacher_warmstart = ShortestPathTeacherWarmup(
            max_steps=max_steps,
            **(teacher_warmstart or {}),
        )
        self.training_curriculum = training_curriculum or {}
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}

    def _curriculum_weights(self, global_step: int) -> dict[str, float]:
        warmup_steps = int(self.training_curriculum.get("warmup_steps", 2000))
        blend_steps = int(self.training_curriculum.get("blend_steps", 3000))
        edge_teacher_weight = float(
            self.training_curriculum.get("edge_teacher_weight", 1.0)
        )
        stop_teacher_weight = float(
            self.training_curriculum.get("stop_teacher_weight", 3.0)
        )
        tb_final_weight = float(self.training_curriculum.get("tb_final_weight", 1.0))

        if global_step < warmup_steps:
            progress = 0.0
        elif blend_steps <= 0:
            progress = 1.0
        else:
            progress = min(max((global_step - warmup_steps) / blend_steps, 0.0), 1.0)

        teacher_scale = 1.0 - progress
        tb_weight = tb_final_weight * progress
        return {
            "tb_weight": tb_weight,
            "edge_teacher_weight": edge_teacher_weight * teacher_scale,
            "stop_teacher_weight": stop_teacher_weight * teacher_scale,
            "teacher_scale": teacher_scale,
            "progress": progress,
        }

    def _evaluation_budgets(self) -> list[int]:
        """
        Evaluation budgets reported for best-of-K metrics.

        K=1 measures native single-sample quality; K=4 and K=8 give practical
        search budgets; K=num_rollout exposes the full evaluation budget.
        K=8 is the default checkpoint-monitoring budget because it matches the
        default training rollout count in `configs/model/gflownet.yaml`.
        """
        return sorted({k for k in (1, 4, 8, self.num_rollout) if k <= self.num_rollout})

    @staticmethod
    def _flatten_metric_groups(
        groups: dict[str, dict[str, float]],
        *,
        prefix: str,
    ) -> dict[str, float]:
        flat: dict[str, float] = {}
        for group_name, group_metrics in groups.items():
            for metric_name, value in group_metrics.items():
                flat[f"{prefix}/{group_name}/{metric_name}"] = value
        return flat

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if not isinstance(batch, RetrievalBatch):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        for field in self._GPU_BATCH_FIELDS:
            if not hasattr(batch, field):
                continue
            value = getattr(batch, field)
            if torch.is_tensor(value):
                setattr(batch, field, value.to(device))
        return batch

    # ------------------------------------------------------------------
    # Inference interface (for downstream RAG pipelines)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        batch: RetrievalBatch,
        num_rollouts: int = 1,
        temperature: float = 1.0,
    ) -> Any:
        """
        Public inference interface for downstream RAG pipelines.

        Runs ``num_rollouts`` independent trajectories under the current
        policy and returns the topologically merged union context graph.
        Gradients are explicitly disabled; this method must not be used
        inside the training loop.
        """
        rollouts = self.rollout_engine.run_exploration(
            policy=self.policy,
            base_graph=batch,
            reward_model=self.reward_model,
            num_rollouts=num_rollouts,
            temperature=temperature,
            collect_terminal_state=True,
            terminal_state_device="cpu",
        )
        return build_union_context_graph(rollouts, batch)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: RetrievalBatch, batch_idx: int) -> torch.Tensor:
        """
        Trajectory Balance training step.

        Samples ``num_rollout`` trajectories per batch so that the
        TB loss estimate has low variance. The temperature is read from the
        annealing schedule so that exploration is warm early in training and
        cools as the policy sharpens.
        """
        temp = (
            1.0
            if self.strict_on_policy
            else self.temp_scheduler.value(self.global_step)
        )

        rollouts = self.rollout_engine.run_exploration(
            policy=self.policy,
            base_graph=batch,
            reward_model=self.reward_model,
            num_rollouts=self.num_rollout,
            temperature=temp,
            collect_terminal_state=False,
        )

        loss_outputs = [self.loss_fn(r) for r in rollouts]
        tb_loss = torch.stack([o.loss for o in loss_outputs]).mean()
        zero = torch.zeros((), device=self.device)
        curriculum = self._curriculum_weights(int(self.global_step))
        if curriculum["teacher_scale"] > 0.0:
            teacher_output = self.teacher_warmstart(policy=self.policy, batch=batch)
            teacher_edge_loss = teacher_output.edge_loss
            teacher_type_loss = teacher_output.type_loss
        else:
            teacher_output = None
            teacher_edge_loss = zero
            teacher_type_loss = zero
        total_loss = (
            tb_loss * curriculum["tb_weight"]
            + teacher_edge_loss * curriculum["edge_teacher_weight"]
            + teacher_type_loss * curriculum["stop_teacher_weight"]
        )

        log_z_mean = torch.stack([o.log_z_mean for o in loss_outputs]).mean()
        log_reward_mean = torch.stack(
            [r.terminal_log_rewards.mean() for r in rollouts]
        ).mean()

        self.log_dict(
            {
                "train/loss": total_loss,
                "train/tb_loss": tb_loss,
                "train/tb_weight": torch.tensor(
                    curriculum["tb_weight"], device=self.device
                ),
                "train/log_z_mean": log_z_mean,
                "train/log_reward_mean": log_reward_mean,
                "train/prior_lambda": self.policy.action_head.prior_scale.detach(),
                "train/teacher_loss": (
                    teacher_edge_loss.detach() + teacher_type_loss.detach()
                ),
                "train/teacher_scale": torch.tensor(
                    curriculum["teacher_scale"], device=self.device
                ),
                "train/teacher_type_loss": (teacher_type_loss.detach()),
                "train/teacher_edge_loss": (teacher_edge_loss.detach()),
                "train/teacher_stop_weight": torch.tensor(
                    curriculum["stop_teacher_weight"], device=self.device
                ),
                "train/teacher_edge_weight": torch.tensor(
                    curriculum["edge_teacher_weight"], device=self.device
                ),
                "train/teacher_states": (
                    teacher_output.supervised_states.detach()
                    if teacher_output is not None
                    else zero
                ),
                "train/temp": torch.tensor(temp, device=self.device),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )

        return total_loss

    # ------------------------------------------------------------------
    # Evaluation (shared logic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_subgraph_retrieval(self, batch: RetrievalBatch) -> dict[str, Any]:
        """
        Run Monte-Carlo rollouts and compute retrieval metrics.

        Uses temperature=1.0 so that the reported numbers reflect the true
        learned distribution rather than a sharpened/flattened variant.
        Gradients are disabled for the entire evaluation pass.
        """
        rollouts = self.rollout_engine.run_exploration(
            policy=self.policy,
            base_graph=batch,
            reward_model=self.reward_model,
            num_rollouts=self.num_rollout,
            temperature=1.0,
            collect_terminal_state=True,
            terminal_state_device="cpu",
        )

        return {
            "distribution": compute_distribution_expectations(rollouts, batch),
            "high_reward": compute_high_reward_discovery(
                rollouts,
                batch,
                ks=self._evaluation_budgets(),
            ),
            "diversity": compute_exploration_diversity(rollouts, batch),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)

        self.log_dict(
            self._flatten_metric_groups(results, prefix="val"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        return results

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)

        self.log_dict(
            self._flatten_metric_groups(results, prefix="test"),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        return results

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return build_optimizer_and_scheduler(
            module=self,
            optimizer_cfg=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Load weights from a checkpoint file.

        ``weights_only=True`` prevents arbitrary pickle execution and should
        be kept unless the checkpoint was deliberately saved with non-tensor
        Python objects. If loading fails with this flag, inspect the
        checkpoint contents before setting it to False.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        return self.load_state_dict(state_dict, strict=strict)


def collect_bool_tensors(obj, prefix=""):
    result = {}
    if isinstance(obj, torch.Tensor) and obj.dtype == torch.bool:
        result[prefix] = obj
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            result.update(collect_bool_tensors(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(collect_bool_tensors(v, f"{prefix}[{i}]"))
    return result


__all__ = ["GFlowNetModule"]
