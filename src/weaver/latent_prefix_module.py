from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, OptimizerLRSchedulerConfig

from src.data.schema import RetrievalBatch
from src.training.config import EvalRuntimeConfig, OptimizationRuntimeConfig
from src.training.optimization import (
    build_lightning_scheduler_config,
    build_optimizer,
    build_scheduler,
    resolve_scheduler_horizon,
)
from src.weaver.context import GraphContext, TargetContext
from src.weaver.module import graph_batch_size
from src.weaver.nn.feature_encoder import FeatureEncoder
from src.weaver.objectives import (
    LatentPrefixObjective,
    LatentPrefixObjectiveInput,
    recompute_trajectory_log_prob,
)
from src.weaver.policy import EdgeOnlyProposalPolicy, PrefixSelector
from src.weaver.rollout import LatentPrefixRollout, LatentPrefixRolloutEngine
from src.weaver.utility import TrueTerminalReward

Scalar = torch.Tensor | float | int


@dataclass(frozen=True, slots=True)
class LatentPrefixStepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, Scalar]


class LatentPrefixWeaverModule(LightningModule):
    def __init__(
        self,
        *,
        policy_feature_encoder: FeatureEncoder,
        proposal_policy: EdgeOnlyProposalPolicy,
        prefix_selector: PrefixSelector,
        reward_model: TrueTerminalReward,
        objective: LatentPrefixObjective,
        rollout_engine: LatentPrefixRolloutEngine,
        train_num_rollouts: int,
        eval_num_rollouts: int,
        optimization: OptimizationRuntimeConfig,
        evaluation: EvalRuntimeConfig,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.policy_feature_encoder = policy_feature_encoder
        self.proposal_policy = proposal_policy
        self.prefix_selector = prefix_selector
        self.reward_model = reward_model
        self.objective = objective
        self.rollout_engine = rollout_engine
        self.train_num_rollouts = int(train_num_rollouts)
        self.eval_num_rollouts = int(eval_num_rollouts)
        self.optimization = optimization
        self.evaluation = evaluation
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.save_hyperparameters(
            {
                "train_num_rollouts": self.train_num_rollouts,
                "eval_num_rollouts": self.eval_num_rollouts,
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": self.gradient_clip_algorithm,
            }
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            modules=(
                self.policy_feature_encoder,
                self.proposal_policy,
                self.prefix_selector,
                self.objective,
            ),
            cfg=self.optimization.optimizer,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            cfg=self.optimization.scheduler,
            trainer=self.trainer,
            base_lr=self.optimization.optimizer.lr,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": build_lightning_scheduler_config(
                scheduler=scheduler,
                interval=self.optimization.scheduler.interval,
            ),
        }

    def training_step(self, batch: RetrievalBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        output = self.compute_step(batch=batch, num_rollouts=self.train_num_rollouts)
        optimizer = self.optimizer()
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(output.loss)
        self.clip_gradients_if_needed(optimizer)
        optimizer.step()
        self.step_scheduler_if_needed(interval="step")
        self.log_scalar("train/latent_prefix/loss", output.loss, batch_size=graph_batch_size(batch), prog_bar=True)
        self.log_scalars(prefix="train/latent_prefix", values=output.metrics, batch_size=graph_batch_size(batch))
        return output.loss.detach()

    def validation_step(self, batch: RetrievalBatch, batch_idx: int) -> None:
        del batch_idx
        with torch.no_grad():
            output = self.compute_step(batch=batch, num_rollouts=self.eval_num_rollouts)
        self.log_scalar("val/latent_prefix/loss", output.loss, batch_size=graph_batch_size(batch))
        self.log_scalars(prefix="val/latent_prefix", values=output.metrics, batch_size=graph_batch_size(batch))

    def compute_step(
        self,
        *,
        batch: RetrievalBatch,
        num_rollouts: int,
    ) -> LatentPrefixStepOutput:
        graph = GraphContext.from_batch(batch)
        target = TargetContext.from_batch(batch=batch, graph_context=graph)
        features = self.policy_feature_encoder(batch)
        rollout = self.rollout_engine.sample_rollouts(
            policy=self.proposal_policy,
            context=graph,
            features=features,
            rollouts_per_graph=int(num_rollouts),
        )
        expansion_log_prob = self.recompute_expansion_log_prob(
            rollout=rollout,
            graph=graph,
            features=features,
        )
        trajectory_log_prob = recompute_trajectory_log_prob(
            expansion_log_prob=expansion_log_prob,
            expansions=rollout.expansions,
            num_trajectories=rollout.num_trajectories,
        )
        _, selector_log_prob = self.prefix_selector(
            features=features,
            state=rollout.prefixes.state,
            context=graph,
            trajectory_ids=rollout.prefixes.trajectory_ids,
            num_trajectories=rollout.num_trajectories,
        )
        reward = self.reward_model(
            state=rollout.prefixes.state,
            graph_context=graph,
            target_context=target,
        ).log_reward
        objective_input = LatentPrefixObjectiveInput(
            trajectory_log_prob=trajectory_log_prob,
            prefix_trajectory_ids=rollout.prefixes.trajectory_ids,
            prefix_step=rollout.prefixes.prefix_step,
            prefix_log_reward=reward,
            selector_log_prob=selector_log_prob,
            num_trajectories=rollout.num_trajectories,
            edge_aux_log_prob=expansion_log_prob,
        )
        output = self.objective(objective_input)
        return LatentPrefixStepOutput(
            loss=output.loss,
            metrics={
                **output.metrics,
                "num_trajectories": float(rollout.num_trajectories),
                "num_prefixes": float(rollout.prefixes.num_items),
                "dead_end_rate": rollout.dead_end.float().mean().detach(),
            },
        )

    def recompute_expansion_log_prob(
        self,
        *,
        rollout: LatentPrefixRollout,
        graph: GraphContext,
        features,
    ) -> torch.Tensor:
        expansions = rollout.expansions
        if expansions.num_items <= 0:
            return torch.empty(0, dtype=torch.float32, device=graph.device)
        frontier = expansions.parent.frontier(graph, expand_budget=rollout.expand_budget)
        out = self.proposal_policy(
            features=features,
            state=expansions.parent,
            context=graph,
            frontier=frontier,
        )
        rows = torch.arange(expansions.num_items, dtype=torch.long, device=graph.device)
        return out.gather_log_prob(row_ids=rows, edge_ids=expansions.edge_ids)

    def optimizer(self) -> torch.optim.Optimizer:
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 1:
                raise RuntimeError("LatentPrefixWeaverModule expects exactly one optimizer.")
            return optimizer[0]
        return optimizer

    def clip_gradients_if_needed(self, optimizer: torch.optim.Optimizer) -> None:
        if self.gradient_clip_val is None or self.gradient_clip_val <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(self.gradient_clip_val),
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

    def step_scheduler_if_needed(self, *, interval: str) -> None:
        cfg = self.optimization.scheduler
        if cfg is None or cfg.interval != interval:
            return
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def on_train_epoch_end(self) -> None:
        self.step_scheduler_if_needed(interval="epoch")

    def training_progress(self) -> float:
        try:
            trainer = self.trainer
        except RuntimeError:
            return 0.0
        horizon = resolve_scheduler_horizon(trainer=trainer, explicit_t_max=None, interval="step")
        if horizon <= 0:
            return 0.0
        return min(1.0, max(0.0, float(int(trainer.global_step)) / float(horizon)))

    def log_scalar(
        self,
        name: str,
        value: Scalar,
        *,
        batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        log_value = value.detach() if isinstance(value, torch.Tensor) else float(value)
        self.log(name, log_value, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size, sync_dist=True)

    def log_scalars(
        self,
        *,
        prefix: str,
        values: Mapping[str, Scalar],
        batch_size: int,
    ) -> None:
        for name, value in values.items():
            self.log_scalar(f"{prefix}/{name}", value, batch_size=batch_size)


__all__ = [
    "LatentPrefixStepOutput",
    "LatentPrefixWeaverModule",
]
