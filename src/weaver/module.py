from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, OptimizerLRSchedulerConfig

from src.data.schema import RetrievalBatch
from src.training.config import EvalRuntimeConfig, OptimizationRuntimeConfig
from src.training.metrics import WeaverMetricSuite
from src.training.optimization import (
    build_lightning_scheduler_config,
    build_optimizer,
    build_scheduler,
    resolve_scheduler_horizon,
)
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import EncodedFeatures, FeatureEncoder
from src.weaver.objectives import SubTBLoss, build_subtb_input, single_step_branch_losses
from src.weaver.policy import ForwardPolicy, PolicyOutput, UniformValidPredecessorBackwardPolicy
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.runner import RolloutBatch, RolloutRunner
from src.weaver.state import Frontier, State
from src.weaver.transition import TrainingBatch
from src.weaver.utility import TrueTerminalReward

Scalar = torch.Tensor | float | int


@dataclass(frozen=True, slots=True)
class StepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, Scalar]
    expansion_branch_loss: torch.Tensor
    terminal_branch_loss: torch.Tensor


class WeaverModule(LightningModule):
    def __init__(
        self,
        *,
        policy_feature_encoder: FeatureEncoder,
        policy: ForwardPolicy,
        reward_model: TrueTerminalReward,
        policy_objective: SubTBLoss,
        runner: RolloutRunner,
        optimization: OptimizationRuntimeConfig,
        evaluation: EvalRuntimeConfig,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.policy_feature_encoder = policy_feature_encoder
        self.policy = policy
        self.reward_model = reward_model
        self.policy_objective = policy_objective
        self.backward_policy = UniformValidPredecessorBackwardPolicy()

        self.runner = runner
        self.optimization = optimization
        self.evaluation = evaluation

        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        self.metric_suite = WeaverMetricSuite(
            k_windows=evaluation.k_windows,
            exclude_anchors_from_retrieved=evaluation.exclude_anchors_from_retrieved,
            use_reachable_targets=evaluation.use_reachable_targets,
        )

        self.runner.progress_fn = self.training_progress

        self.save_hyperparameters(
            {
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": self.gradient_clip_algorithm,
            }
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            modules=(
                self.policy_feature_encoder,
                self.policy,
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

        config: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": build_lightning_scheduler_config(
                scheduler=scheduler,
                interval=self.optimization.scheduler.interval,
            ),
        }
        return config

    def training_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx

        output = self.compute_step(
            batch=batch,
        )

        optimizer = self.optimizer()
        optimizer.zero_grad(set_to_none=True)
        branch_grad_metrics = stop_branch_gradient_metrics(
            stop_head=self.policy.stop_head,
            terminal_loss=output.terminal_branch_loss,
            expansion_loss=output.expansion_branch_loss,
        )
        self.manual_backward(output.loss)
        grad_metrics = gradient_norm_metrics(self.policy)
        self.clip_gradients_if_needed(optimizer)
        optimizer.step()
        self.step_scheduler_if_needed(interval="step")

        batch_n = graph_batch_size(batch)
        metrics = {
            **output.metrics,
            **branch_grad_metrics,
            **grad_metrics,
        }
        self.log_scalar(
            "train/policy/loss",
            output.loss,
            batch_size=batch_n,
            prog_bar=True,
        )
        self.log_scalars(
            prefix="train/policy",
            values=metrics,
            batch_size=batch_n,
        )
        return output.loss.detach()

    def compute_step(
        self,
        *,
        batch: RetrievalBatch,
    ) -> StepOutput:
        graph = GraphContext.from_batch(batch)
        target = TargetContext.from_batch(
            batch=batch,
            graph_context=graph,
        )

        policy_features = self.policy_feature_encoder(batch)

        rollout = self.sample_train_rollout(
            batch=batch,
            graph=graph,
            target=target,
            features=policy_features,
        )
        rollout_entropy = rollout_action_entropy(rollout.rollouts)
        training = rollout.training
        if training is None or training.num_items <= 0:
            raise RuntimeError("No transition samples were produced.")

        output = self.policy_step_output(
            graph=graph,
            target=target,
            policy_features=policy_features,
            training=training,
        )
        return StepOutput(
            loss=output.loss,
            metrics={
                **output.metrics,
                "rollout_action_entropy": rollout_entropy.detach(),
                **rollout_replay_metrics(rollout, policy_features.query_model),
            },
            expansion_branch_loss=output.expansion_branch_loss,
            terminal_branch_loss=output.terminal_branch_loss,
        )

    def policy_step_output(
        self,
        *,
        graph: GraphContext,
        target: TargetContext,
        policy_features: EncodedFeatures,
        training: TrainingBatch,
    ) -> StepOutput:
        expansions = training.expansions
        terminals = training.terminals
        if terminals.num_items <= 0:
            zero = policy_features.query_model.new_zeros(())
            return StepOutput(
                loss=zero,
                metrics={},
                expansion_branch_loss=zero,
                terminal_branch_loss=zero,
            )

        expand_budget = int(self.runner.engine.expand_budget)
        if expansions.num_items > 0:
            parent_frontier = expansions.parent.frontier(
                graph,
                expand_budget=expand_budget,
            )
            child_frontier = expansions.child.frontier(
                graph,
                expand_budget=expand_budget,
            )

            parent_out = self.policy(
                features=policy_features,
                state=expansions.parent,
                context=graph,
                frontier=parent_frontier,
            )
            # Child flow must come from an explicit child-state forward pass.
            child_out = self.policy(
                features=policy_features,
                state=expansions.child,
                context=graph,
                frontier=child_frontier,
            )
            backward_log_prob = backward_action_log_prob(
                backward_policy=self.backward_policy,
                child_state=expansions.child,
                context=graph,
                action_edge_ids=expansions.edge_ids,
            )
        else:
            parent_out = empty_policy_output(
                device=policy_features.query_model.device,
                dtype=policy_features.query_model.dtype,
                num_edges=graph.num_edges,
            )
            child_out = parent_out
            backward_log_prob = torch.empty(
                0,
                dtype=torch.float32,
                device=policy_features.query_model.device,
            )
        terminal_frontier = terminals.state.frontier(
            graph,
            expand_budget=expand_budget,
        )
        terminal_out = self.policy(
            features=policy_features,
            state=terminals.state,
            context=graph,
            frontier=terminal_frontier,
        )

        reward_out = call_reward_model(
            reward_model=self.reward_model,
            state=terminals.state,
            graph_context=graph,
            target_context=target,
        )

        subtb_input = build_subtb_input(
            parent_out=parent_out,
            child_out=child_out,
            terminal_out=terminal_out,
            reward_out=reward_out,
            backward_log_prob=backward_log_prob,
            expansions=expansions,
            terminals=terminals,
        )

        output = self.policy_objective(subtb_input)
        expansion_branch_loss, terminal_branch_loss = single_step_branch_losses(
            subtb_input,
            loss_type=self.policy_objective.residual_loss,
            huber_delta=self.policy_objective.huber_delta,
        )
        policy_metrics = policy_diagnostic_metrics(
            expansion_out=parent_out,
            expansion_depth=expansions.parent.depth,
            terminal_out=terminal_out,
            terminal_depth=terminals.state.depth,
        )

        return StepOutput(
            loss=output.loss,
            metrics={
                **output.metrics,
                **policy_metrics,
            },
            expansion_branch_loss=expansion_branch_loss,
            terminal_branch_loss=terminal_branch_loss,
        )

    def sample_train_rollout(
        self,
        *,
        batch: RetrievalBatch,
        graph: GraphContext,
        target: TargetContext,
        features: EncodedFeatures,
    ) -> RolloutBatch:
        with torch.no_grad():
            return self.runner.train_rollouts(
                policy=self.policy,
                batch=batch,
                context=graph,
                features=features,
                reward_model=self.reward_model,
                target_context=target,
            )

    def validation_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self.eval_step(split="val", batch=batch)

    def test_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self.eval_step(split="test", batch=batch)

    def predict_step(
        self,
        batch: RetrievalBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[RolloutResult, ...]:
        del batch_idx, dataloader_idx
        with torch.no_grad():
            graph = GraphContext.from_batch(batch)
            features = self.policy_feature_encoder(batch)
            return self.runner.eval_rollouts(
                policy=self.policy,
                context=graph,
                features=features,
            )

    def eval_step(
        self,
        *,
        split: str,
        batch: RetrievalBatch,
    ) -> None:
        with torch.no_grad():
            graph = GraphContext.from_batch(batch)
            target = TargetContext.from_batch(
                batch=batch,
                graph_context=graph,
            )
            features = self.policy_feature_encoder(batch)
            rollouts = self.runner.eval_rollouts(
                policy=self.policy,
                context=graph,
                features=features,
            )
            metrics = self.metric_suite.eval_metrics(
                rollout_samples=rollouts,
                batch=batch,
                stage="",
                context=graph,
                features=features,
                reward_model=self.reward_model,
                target_context=target,
                policy=self.policy,
            )

        batch_n = graph_batch_size(batch)
        self.log_scalar(
            f"{split}/num_rollouts",
            float(len(rollouts)),
            batch_size=batch_n,
        )
        self.log_scalars(
            prefix=split,
            values=metrics,
            batch_size=batch_n,
        )

    def optimizer(self) -> torch.optim.Optimizer:
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 1:
                raise RuntimeError("WeaverModule expects exactly one optimizer.")
            return optimizer[0]
        return optimizer

    def clip_gradients_if_needed(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.gradient_clip_val is None or self.gradient_clip_val <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(self.gradient_clip_val),
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

    def step_scheduler_if_needed(
        self,
        *,
        interval: str,
    ) -> None:
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

        horizon = resolve_scheduler_horizon(
            trainer=trainer,
            explicit_t_max=None,
            interval="step",
        )
        if horizon <= 0:
            return 0.0
        step = int(trainer.global_step)
        return min(1.0, max(0.0, float(step) / float(horizon)))

    def log_scalar(
        self,
        name: str,
        value: Scalar,
        *,
        batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        if name.count("/") >= 3:
            raise ValueError(f"Metric name must have at most three slash-separated levels: {name}")
        log_value = value.detach() if isinstance(value, torch.Tensor) else float(value)
        self.log(
            name,
            log_value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=True,
        )

    def log_scalars(
        self,
        *,
        prefix: str,
        values: Mapping[str, Scalar],
        batch_size: int,
    ) -> None:
        for name, value in values.items():
            self.log_scalar(
                f"{prefix}/{name}",
                value,
                batch_size=batch_size,
            )


def backward_action_log_prob(
    *,
    backward_policy: UniformValidPredecessorBackwardPolicy,
    child_state: State,
    context: GraphContext,
    action_edge_ids: torch.Tensor,
) -> torch.Tensor:
    action_edge_ids = action_edge_ids.to(device=child_state.device, dtype=torch.long).view(-1)
    out = torch.zeros(
        action_edge_ids.numel(),
        dtype=torch.float32,
        device=child_state.device,
    )
    expand = action_edge_ids.ge(0)
    if bool(expand.any()):
        out[expand] = backward_policy.log_prob(
            child_state=child_state.select_rows(expand.nonzero(as_tuple=False).flatten()),
            context=context,
            action_edge_ids=action_edge_ids[expand],
        )
    return out


def empty_policy_output(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_edges: int,
) -> PolicyOutput:
    return PolicyOutput(
        stop_logit=torch.empty(0, dtype=dtype, device=device),
        log_flow=torch.empty(0, dtype=dtype, device=device),
        edge_logit=torch.empty(0, dtype=dtype, device=device),
        frontier=Frontier(
            row_ids=torch.empty(0, dtype=torch.long, device=device),
            edge_ids=torch.empty(0, dtype=torch.long, device=device),
        ),
        num_rows=0,
        num_edges=int(num_edges),
    )


def call_reward_model(
    *,
    reward_model: TrueTerminalReward,
    state: State,
    graph_context: GraphContext,
    target_context: TargetContext,
):
    return reward_model(
        state=state,
        graph_context=graph_context,
        target_context=target_context,
    )


def policy_diagnostic_metrics(
    *,
    expansion_out: PolicyOutput,
    expansion_depth: torch.Tensor,
    terminal_out: PolicyOutput,
    terminal_depth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    stop_logit = torch.cat(
        [
            expansion_out.stop_logit.float(),
            terminal_out.stop_logit.float(),
        ],
        dim=0,
    )
    depth = torch.cat(
        [
            expansion_depth.to(device=stop_logit.device, dtype=torch.long).view(-1),
            terminal_depth.to(device=stop_logit.device, dtype=torch.long).view(-1),
        ],
        dim=0,
    )
    stop_prob = torch.cat(
        [
            expansion_out.stop_prob().float(),
            terminal_out.stop_prob().float(),
        ],
        dim=0,
    )

    frontier_size = torch.cat(
        [
            expansion_out.frontier_size(),
            terminal_out.frontier_size(),
        ],
        dim=0,
    )
    edge_cond_entropy = torch.cat(
        [
            expansion_out.edge_cond_entropy(),
            terminal_out.edge_cond_entropy(),
        ],
        dim=0,
    )
    continue_prob = torch.cat(
        [
            expansion_out.continue_prob().float(),
            terminal_out.continue_prob().float(),
        ],
        dim=0,
    )
    has_frontier = frontier_size.gt(0)

    metrics: dict[str, torch.Tensor] = {
        "policy_continue_prob_mean": masked_mean_or_zero(continue_prob, has_frontier).detach(),
        "policy_edge_cond_entropy_mean": masked_mean_or_zero(edge_cond_entropy, has_frontier).detach(),
        "policy_frontier_size_mean": mean_or_zero(frontier_size).detach(),
        "policy_frontier_size_p90": quantile_or_zero(frontier_size, 0.90).detach(),
        "policy_frontier_size_p99": quantile_or_zero(frontier_size, 0.99).detach(),
    }
    for bucket in range(4):
        mask = depth.eq(bucket)
        metrics[f"policy_stop_logit_depth{bucket}_mean"] = masked_mean_or_zero(
            stop_logit,
            mask,
        ).detach()
        metrics[f"policy_stop_prob_depth{bucket}_mean"] = masked_mean_or_zero(
            stop_prob,
            mask,
        ).detach()
    return metrics


def stop_branch_gradient_metrics(
    *,
    stop_head: torch.nn.Module,
    terminal_loss: torch.Tensor,
    expansion_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    params = [param for param in stop_head.parameters() if param.requires_grad]
    terminal_grad = gradient_vector(
        loss=terminal_loss,
        params=params,
        retain_graph=True,
    )
    expansion_grad = gradient_vector(
        loss=expansion_loss,
        params=params,
        retain_graph=True,
    )
    terminal_norm = terminal_grad.norm()
    expansion_norm = expansion_grad.norm()
    denom = terminal_norm * expansion_norm
    cosine = torch.where(
        denom.gt(0),
        torch.dot(terminal_grad, expansion_grad) / denom.clamp_min(torch.finfo(terminal_grad.dtype).tiny),
        terminal_grad.new_zeros(()),
    )
    return {
        "grad_stop_head_from_terminal_loss": terminal_norm.detach(),
        "grad_stop_head_from_expansion_loss": expansion_norm.detach(),
        "grad_stop_head_terminal_expansion_cosine": cosine.detach(),
    }


def gradient_vector(
    *,
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    retain_graph: bool,
) -> torch.Tensor:
    if not params:
        return loss.new_zeros((0,))
    if not loss.requires_grad:
        return torch.cat([param.detach().new_zeros(param.numel()) for param in params])
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [
        torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1)
        for param, grad in zip(params, grads, strict=True)
    ]
    return torch.cat(values) if values else loss.new_zeros((0,))


def gradient_norm_metrics(policy: ForwardPolicy) -> dict[str, torch.Tensor]:
    reference = next(policy.parameters())
    return {
        "grad_stop_head_norm": module_grad_norm(policy.stop_head, reference).detach(),
        "grad_flow_head_norm": module_grad_norm(policy.flow_head, reference).detach(),
        "grad_edge_head_norm": module_grad_norm(policy.edge_head, reference).detach(),
        "grad_state_encoder_norm": module_grad_norm(policy.state_encoder, reference).detach(),
    }


def module_grad_norm(
    module: torch.nn.Module,
    reference: torch.Tensor,
) -> torch.Tensor:
    total = reference.new_zeros(())
    for param in module.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().float().square().sum()
    return total.sqrt()


def masked_mean_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or not bool(mask.any()):
        return values.new_zeros(())
    return values.float()[mask].mean()


def mean_or_zero(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.float().mean()


def quantile_or_zero(
    values: torch.Tensor,
    q: float,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return torch.quantile(values.float(), float(q))


def graph_batch_size(
    batch: RetrievalBatch,
) -> int:
    return int(batch.num_graphs_total)


def rollout_action_entropy(
    rollouts: tuple[RolloutResult, ...],
) -> torch.Tensor:
    if not rollouts:
        return torch.zeros((), dtype=torch.float32)
    values = torch.cat(
        [
            rollout.policy_action_log_prob[rollout.valid_mask].reshape(-1)
            for rollout in rollouts
        ],
        dim=0,
    )
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=rollouts[0].device)
    return (-values.float()).mean()


def rollout_replay_metrics(
    rollout: RolloutBatch,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    replay = rollout.replay
    if replay is None:
        return {
            "replay_eligible_graphs": reference.new_zeros(()).detach(),
            "replay_skipped_by_reward": reference.new_zeros(()).detach(),
            "replay_generated_trajectories": reference.new_zeros(()).detach(),
            "replay_covered_graphs": reference.new_zeros(()).detach(),
            "replay_expansion_transitions": reference.new_zeros(()).detach(),
            "replay_terminal_transitions": reference.new_zeros(()).detach(),
        }
    stats = replay.stats
    return {
        "replay_eligible_graphs": reference.new_tensor(float(stats.eligible_graphs)).detach(),
        "replay_skipped_by_reward": reference.new_tensor(float(stats.skipped_by_reward)).detach(),
        "replay_generated_trajectories": reference.new_tensor(float(stats.generated_trajectories)).detach(),
        "replay_covered_graphs": reference.new_tensor(float(stats.covered_graphs)).detach(),
        "replay_expansion_transitions": reference.new_tensor(float(rollout.num_replay_transitions)).detach(),
        "replay_terminal_transitions": reference.new_tensor(float(rollout.num_replay_terminal_transitions)).detach(),
    }


__all__ = [
    "WeaverModule",
]
