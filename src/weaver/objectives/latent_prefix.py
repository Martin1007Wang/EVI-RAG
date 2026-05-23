from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_logsumexp, segment_softmax
from src.weaver.transition import ExpansionBatch
from src.weaver.utility import RewardOutput

from .common import ObjectiveOutput
from .subtb import mean_or_zero


@dataclass(frozen=True, slots=True)
class LatentPrefixObjectiveInput:
    trajectory_log_prob: torch.Tensor
    prefix_trajectory_ids: torch.Tensor
    prefix_step: torch.Tensor
    prefix_log_reward: torch.Tensor
    selector_log_prob: torch.Tensor
    num_trajectories: int
    edge_aux_log_prob: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class LatentPrefixTargets:
    prefix_log_target: torch.Tensor
    prefix_target_prob: torch.Tensor
    trajectory_log_reward: torch.Tensor


class LatentPrefixObjective(nn.Module):
    def __init__(
        self,
        *,
        prefix_temperature: float = 1.0,
        length_prior_gamma: float = 0.05,
        lambda_prefix: float = 1.0,
        lambda_edge: float = 0.1,
        initial_log_z: float = 0.0,
    ) -> None:
        super().__init__()
        if prefix_temperature <= 0.0:
            raise ValueError(f"prefix_temperature must be positive, got {prefix_temperature}.")
        if length_prior_gamma < 0.0:
            raise ValueError(f"length_prior_gamma must be non-negative, got {length_prior_gamma}.")
        if lambda_prefix < 0.0:
            raise ValueError(f"lambda_prefix must be non-negative, got {lambda_prefix}.")
        if lambda_edge < 0.0:
            raise ValueError(f"lambda_edge must be non-negative, got {lambda_edge}.")
        self.prefix_temperature = float(prefix_temperature)
        self.length_prior_gamma = float(length_prior_gamma)
        self.lambda_prefix = float(lambda_prefix)
        self.lambda_edge = float(lambda_edge)
        self.log_z = nn.Parameter(torch.tensor(float(initial_log_z), dtype=torch.float32))

    def forward(self, x: LatentPrefixObjectiveInput) -> ObjectiveOutput:
        targets = latent_prefix_targets(
            prefix_log_reward=x.prefix_log_reward,
            prefix_step=x.prefix_step,
            prefix_trajectory_ids=x.prefix_trajectory_ids,
            num_trajectories=int(x.num_trajectories),
            temperature=self.prefix_temperature,
            length_prior_gamma=self.length_prior_gamma,
        )
        residual = self.log_z + x.trajectory_log_prob.float() - targets.trajectory_log_reward.detach()
        traj_loss = residual.square().mean() if residual.numel() > 0 else residual.new_zeros(())
        prefix_loss = prefix_cross_entropy(
            target_prob=targets.prefix_target_prob.detach(),
            selector_log_prob=x.selector_log_prob,
        )
        edge_loss = edge_auxiliary_loss(x.edge_aux_log_prob)
        if edge_loss.device != traj_loss.device:
            edge_loss = traj_loss.new_zeros(())
        loss = traj_loss + float(self.lambda_prefix) * prefix_loss + float(self.lambda_edge) * edge_loss
        metrics = {
            "latent_prefix_loss_traj": traj_loss.detach(),
            "latent_prefix_loss_prefix": prefix_loss.detach(),
            "latent_prefix_loss_edge_aux": edge_loss.detach(),
            "latent_prefix_log_z": self.log_z.detach(),
            "latent_prefix_log_reward_mean": mean_or_zero(x.prefix_log_reward).detach(),
            "latent_prefix_log_rbar_mean": mean_or_zero(targets.trajectory_log_reward).detach(),
            "latent_prefix_residual_abs_mean": mean_or_zero(residual.abs()).detach(),
        }
        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_states=int(x.prefix_trajectory_ids.numel()),
            per_unit_loss=residual.detach().square(),
        )


def latent_prefix_targets(
    *,
    prefix_log_reward: torch.Tensor,
    prefix_step: torch.Tensor,
    prefix_trajectory_ids: torch.Tensor,
    num_trajectories: int,
    temperature: float,
    length_prior_gamma: float,
) -> LatentPrefixTargets:
    temp = float(temperature)
    prefix_log_target = prefix_log_reward.float() - float(length_prior_gamma) * prefix_step.float()
    scaled = prefix_log_target / temp
    logsum = segment_logsumexp(
        values=scaled,
        segment_ids=prefix_trajectory_ids.to(device=scaled.device, dtype=torch.long),
        num_segments=int(num_trajectories),
    )
    trajectory_log_reward = temp * logsum
    prefix_target_prob = segment_softmax(
        scaled,
        prefix_trajectory_ids.to(device=scaled.device, dtype=torch.long),
        num_segments=int(num_trajectories),
    )
    return LatentPrefixTargets(
        prefix_log_target=prefix_log_target,
        prefix_target_prob=prefix_target_prob,
        trajectory_log_reward=trajectory_log_reward,
    )


def prefix_cross_entropy(
    *,
    target_prob: torch.Tensor,
    selector_log_prob: torch.Tensor,
) -> torch.Tensor:
    if target_prob.numel() == 0:
        return selector_log_prob.new_zeros(())
    return -(target_prob.float() * selector_log_prob.float()).sum() / target_prob.float().sum().clamp_min(1.0)


def edge_auxiliary_loss(edge_aux_log_prob: torch.Tensor | None) -> torch.Tensor:
    if edge_aux_log_prob is None or edge_aux_log_prob.numel() == 0:
        reference = edge_aux_log_prob if edge_aux_log_prob is not None else torch.tensor(0.0)
        return reference.new_zeros(())
    return -edge_aux_log_prob.float().mean()


def joint_prefix_score(
    *,
    trajectory_log_prob: torch.Tensor,
    selector_log_prob: torch.Tensor,
    prefix_trajectory_ids: torch.Tensor,
) -> torch.Tensor:
    return trajectory_log_prob.float().index_select(
        0,
        prefix_trajectory_ids.to(device=trajectory_log_prob.device, dtype=torch.long),
    ) + selector_log_prob.float()


def recompute_trajectory_log_prob(
    *,
    expansion_log_prob: torch.Tensor,
    expansions: ExpansionBatch,
    num_trajectories: int,
) -> torch.Tensor:
    out = expansion_log_prob.new_zeros((int(num_trajectories),))
    if expansion_log_prob.numel() == 0:
        return out
    out.scatter_add_(
        0,
        expansions.meta.trajectory_ids.to(device=expansion_log_prob.device, dtype=torch.long),
        expansion_log_prob.float(),
    )
    return out


def reward_log_values(reward_out: RewardOutput) -> torch.Tensor:
    return reward_out.log_reward.float()


__all__ = [
    "LatentPrefixObjective",
    "LatentPrefixObjectiveInput",
    "LatentPrefixTargets",
    "edge_auxiliary_loss",
    "joint_prefix_score",
    "latent_prefix_targets",
    "prefix_cross_entropy",
    "recompute_trajectory_log_prob",
    "reward_log_values",
]
