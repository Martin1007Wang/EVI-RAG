from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.weaver.rollout.schema import RolloutBatch


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    per_trajectory_loss: torch.Tensor | None = None

    @classmethod
    def aggregate(cls, outputs: Iterable["LossOutput"]) -> "LossOutput":
        outputs = tuple(outputs)
        if not outputs:
            zero = torch.zeros(())
            return cls(loss=zero, metrics={}, per_trajectory_loss=None)

        metrics: dict[str, torch.Tensor] = {}
        for output in outputs:
            for key, value in output.metrics.items():
                metrics[key] = value if key not in metrics else metrics[key] + value

        scale = float(len(outputs))
        metrics = {key: value / scale for key, value in metrics.items()}

        if "loss/total" in metrics:
            loss = metrics["loss/total"]
        elif "loss" in metrics:
            loss = metrics["loss"]
        else:
            device = outputs[0].loss.device
            loss = torch.zeros((), device=device)

        return cls(loss=loss, metrics=metrics, per_trajectory_loss=None)


class SubTrajectoryBalanceLoss(nn.Module):
    """
    SubTrajectory Balance loss over rollout traces.

    For each valid subtrajectory i..j:

        residual(i, j) =
            log F(s_i)
            + sum_{t=i..j} [log P_F(a_t | s_t) - log P_B(s_t | s_{t+1})]
            - target(j)

    where target(j) is:
        - log F(s_{j+1}) for non-terminal j
        - log R(x) for the terminal STOP step

    The terminal STOP transition has no backward action, so its effective
    log P_B is set to 0 inside the loss.
    """

    metric_keys = (
        "loss/total",
        "loss/subtb",
        "loss/stop_tb",
        "loss/advantage_aux",
        "loss/advantage_aux_coef",
        "subtb/residual_abs_mean",
        "subtb/residual_square_mean",
        "subtb/residual_mean",
        "subtb/residual_std",
        "subtb/subtrajectory_count_mean",
        "stop_tb/residual_abs_mean",
        "stop_tb/residual_square_mean",
        "stop_tb/residual_mean",
        "stop_tb/residual_std",
        "stop_tb/valid_count_mean",
        "advantage_aux/valid_count_mean",
        "diagnostic/terminal_stop_balance_mse",
        "diagnostic/terminal_stop_residual_abs_mean",
        "diagnostic/terminal_step_log_pb_abs_max",
        "flow/log_z_mean",
        "flow/log_z_std",
        "flow/state_log_flow_mean",
        "flow/state_log_flow_std",
        "reward/log_reward_mean",
        "reward/log_reward_std",
        "reward/log_reward_clamped_mean",
        "reward/clipped_ratio",
        "prob/step_log_pf_mean",
        "prob/step_log_pb_mean",
    )

    def __init__(
        self,
        *,
        max_trajectory_len: int,
        subtb_lambda: float = 0.8,
        stop_tb_coef: float = 0.0,
        advantage_aux_coef_start: float = 0.0,
        advantage_aux_coef_end: float | None = None,
        advantage_aux_anneal_steps: int = 0,
        advantage_teacher_temperature: float = 0.15,
        advantage_top_k: int = 32,
        advantage_topk_prior: int | None = None,
        advantage_topk_final: int | None = None,
        advantage_random_k: int | None = None,
        log_reward_clip_min: float = -100.0,
        debug: bool = False,
    ) -> None:
        super().__init__()

        if max_trajectory_len < 1:
            raise ValueError(
                f"max_trajectory_len must be >= 1, got {max_trajectory_len}."
            )
        if not 0.0 <= subtb_lambda <= 1.0:
            raise ValueError(f"subtb_lambda must be in [0, 1], got {subtb_lambda}.")
        if stop_tb_coef < 0.0:
            raise ValueError(f"stop_tb_coef must be >= 0, got {stop_tb_coef}.")
        if advantage_aux_coef_start < 0.0:
            raise ValueError(
                "advantage_aux_coef_start must be >= 0, "
                f"got {advantage_aux_coef_start}."
            )
        if advantage_aux_coef_end is not None and advantage_aux_coef_end < 0.0:
            raise ValueError(
                "advantage_aux_coef_end must be >= 0, " f"got {advantage_aux_coef_end}."
            )
        if advantage_aux_anneal_steps < 0:
            raise ValueError(
                "advantage_aux_anneal_steps must be >= 0, "
                f"got {advantage_aux_anneal_steps}."
            )
        if advantage_teacher_temperature <= 0.0:
            raise ValueError(
                "advantage_teacher_temperature must be > 0, "
                f"got {advantage_teacher_temperature}."
            )
        if advantage_top_k < 1:
            raise ValueError(f"advantage_top_k must be >= 1, got {advantage_top_k}.")
        if advantage_topk_prior is not None and advantage_topk_prior < 0:
            raise ValueError(
                "advantage_topk_prior must be >= 0, " f"got {advantage_topk_prior}."
            )
        if advantage_topk_final is not None and advantage_topk_final < 0:
            raise ValueError(
                "advantage_topk_final must be >= 0, " f"got {advantage_topk_final}."
            )
        if advantage_random_k is not None and advantage_random_k < 0:
            raise ValueError(
                f"advantage_random_k must be >= 0, got {advantage_random_k}."
            )

        self.max_trajectory_len = int(max_trajectory_len)
        self.subtb_lambda = float(subtb_lambda)
        self.stop_tb_coef = float(stop_tb_coef)
        self.advantage_aux_coef_start = float(advantage_aux_coef_start)
        self.advantage_aux_coef_end = (
            float(advantage_aux_coef_start)
            if advantage_aux_coef_end is None
            else float(advantage_aux_coef_end)
        )
        self.advantage_aux_anneal_steps = int(advantage_aux_anneal_steps)
        self.advantage_teacher_temperature = float(advantage_teacher_temperature)
        self.advantage_top_k = int(advantage_top_k)
        if (
            advantage_topk_prior is None
            and advantage_topk_final is None
            and advantage_random_k is None
        ):
            self.advantage_topk_prior = 0
            self.advantage_topk_final = int(advantage_top_k)
            self.advantage_random_k = 0
        else:
            self.advantage_topk_prior = int(advantage_topk_prior or 0)
            self.advantage_topk_final = int(advantage_topk_final or 0)
            self.advantage_random_k = int(advantage_random_k or 0)
            if (
                self.advantage_topk_prior
                + self.advantage_topk_final
                + self.advantage_random_k
                < 1
            ):
                raise ValueError(
                    "At least one advantage candidate source must be enabled: "
                    "advantage_topk_prior + advantage_topk_final + "
                    "advantage_random_k >= 1."
                )
        self._global_step = 0
        self.log_reward_clip_min = float(log_reward_clip_min)
        self.debug = bool(debug)

    @property
    def advantage_aux_enabled(self) -> bool:
        return max(self.advantage_aux_coef_start, self.advantage_aux_coef_end) > 0.0

    def set_global_step(self, step: int) -> None:
        self._global_step = max(int(step), 0)

    def current_advantage_aux_coef(self) -> float:
        if self.advantage_aux_anneal_steps <= 0:
            return self.advantage_aux_coef_end

        progress = min(
            float(self._global_step) / float(self.advantage_aux_anneal_steps),
            1.0,
        )
        return self.advantage_aux_coef_start + progress * (
            self.advantage_aux_coef_end - self.advantage_aux_coef_start
        )

    def forward(self, rollout: RolloutBatch) -> LossOutput:
        traces = rollout.traces
        stats = rollout.stats

        state_log_flows = traces.state_log_flows.float()
        step_log_pf = traces.log_pf.float()
        step_log_pb = traces.log_pb.float()

        device = state_log_flows.device
        dtype = state_log_flows.dtype
        batch_size, horizon = state_log_flows.shape

        if horizon > self.max_trajectory_len:
            raise ValueError(
                f"rollout horizon={horizon} exceeds "
                f"max_trajectory_len={self.max_trajectory_len}."
            )

        lengths = stats.trajectory_length.to(device=device, dtype=torch.long)
        valid = lengths.gt(0)

        if not bool(valid.any()):
            return self._zero_output(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if self.debug:
            self._assert_terminal_stop(
                stop_mask=traces.stop_mask,
                lengths=lengths,
                valid=valid,
                horizon=horizon,
            )

        raw_log_rewards = stats.terminal_log_reward.to(device=device, dtype=dtype)
        log_rewards = raw_log_rewards.clamp(min=self.log_reward_clip_min)

        terminal_index = lengths.clamp(1, horizon) - 1
        entry_mask = _subtrajectory_mask(
            lengths=lengths,
            horizon=horizon,
            device=device,
        )

        effective_log_pb = step_log_pb.clone()
        effective_log_pb[torch.arange(batch_size, device=device), terminal_index] = 0.0

        segment_log_ratio = _segment_sums(step_log_pf - effective_log_pb)
        targets = _subtrajectory_targets(
            state_log_flows=state_log_flows,
            log_rewards=log_rewards,
            terminal_index=terminal_index,
        )
        weights = self._subtrajectory_weights(
            lengths=lengths,
            horizon=horizon,
            device=device,
            dtype=dtype,
        )

        residuals = (
            state_log_flows.unsqueeze(2) + segment_log_ratio - targets.unsqueeze(1)
        ) * entry_mask.to(dtype)

        weights = weights * entry_mask.to(dtype)
        denom = weights.sum(dim=(1, 2)).clamp_min(torch.finfo(dtype).eps)
        per_trajectory_loss = (weights * residuals.square()).sum(dim=(1, 2)) / denom
        subtb_loss = per_trajectory_loss[valid].mean()

        stop_tb_valid_mask = traces.stop_tb_valid_mask.to(
            device=device, dtype=torch.bool
        ) & traces.stop_now_valid_mask.to(device=device, dtype=torch.bool)
        stop_now_log_reward = traces.stop_now_log_reward.to(
            device=device,
            dtype=dtype,
        ).clamp(min=self.log_reward_clip_min)
        stop_tb_loss = stop_now_tb_loss(
            state_log_flow=state_log_flows,
            stop_log_pf=traces.stop_log_pf.to(device=device, dtype=dtype),
            stop_now_log_reward=stop_now_log_reward,
            valid_mask=stop_tb_valid_mask,
        )
        advantage_aux_loss = masked_mean(
            traces.advantage_aux_loss.to(device=device, dtype=dtype),
            traces.advantage_aux_valid_mask.to(device=device, dtype=torch.bool),
        )
        advantage_aux_coef = self.current_advantage_aux_coef()
        loss = (
            subtb_loss
            + (self.stop_tb_coef * stop_tb_loss)
            + (advantage_aux_coef * advantage_aux_loss)
        )

        terminal_stop_residual = residuals[
            torch.arange(batch_size, device=device),
            terminal_index,
            terminal_index,
        ]

        metrics = self._metrics(
            loss=loss,
            subtb_loss=subtb_loss,
            stop_tb_loss=stop_tb_loss,
            advantage_aux_loss=advantage_aux_loss,
            advantage_aux_coef=state_log_flows.new_tensor(advantage_aux_coef),
            raw_log_rewards=raw_log_rewards,
            log_rewards=log_rewards,
            stop_tb_residuals=(
                state_log_flows
                + traces.stop_log_pf.to(device=device, dtype=dtype)
                - stop_now_log_reward
            ),
            stop_tb_valid_mask=stop_tb_valid_mask,
            advantage_aux_valid_mask=traces.advantage_aux_valid_mask.to(
                device=device,
                dtype=torch.bool,
            ),
            state_log_flows=state_log_flows,
            step_log_pf=step_log_pf,
            step_log_pb=step_log_pb,
            residuals=residuals,
            entry_mask=entry_mask,
            terminal_stop_residual=terminal_stop_residual,
            lengths=lengths,
            valid=valid,
            root_log_z=stats.root_log_z.to(device=device, dtype=torch.float32),
        )

        return LossOutput(
            loss=loss,
            metrics=metrics,
            per_trajectory_loss=per_trajectory_loss.detach(),
        )

    def _subtrajectory_weights(
        self,
        *,
        lengths: torch.Tensor,
        horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        idx = torch.arange(horizon, device=device)
        span = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp_min(0).to(dtype)
        upper = torch.ones(horizon, horizon, dtype=dtype, device=device).triu()

        bucket_count = (lengths.to(dtype).view(-1, 1, 1) - span.unsqueeze(0)).clamp_min(
            1.0
        )

        return (
            (self.subtb_lambda**span).unsqueeze(0) * upper.unsqueeze(0) / bucket_count
        )

    def _metrics(
        self,
        *,
        loss: torch.Tensor,
        subtb_loss: torch.Tensor,
        stop_tb_loss: torch.Tensor,
        advantage_aux_loss: torch.Tensor,
        advantage_aux_coef: torch.Tensor,
        raw_log_rewards: torch.Tensor,
        log_rewards: torch.Tensor,
        stop_tb_residuals: torch.Tensor,
        stop_tb_valid_mask: torch.Tensor,
        advantage_aux_valid_mask: torch.Tensor,
        state_log_flows: torch.Tensor,
        step_log_pf: torch.Tensor,
        step_log_pb: torch.Tensor,
        residuals: torch.Tensor,
        entry_mask: torch.Tensor,
        terminal_stop_residual: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
        root_log_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batch_size, horizon = state_log_flows.shape
            device = state_log_flows.device
            zero = state_log_flows.new_zeros(())

            valid_lengths = lengths[valid]
            valid_entry_mask = entry_mask[valid]
            valid_residuals = residuals[valid][valid_entry_mask]
            valid_stop_tb_residuals = stop_tb_residuals[stop_tb_valid_mask]

            step_ids = torch.arange(horizon, device=device).unsqueeze(0)
            valid_steps = step_ids < valid_lengths.unsqueeze(1)

            valid_state_flows = state_log_flows[valid][valid_steps]
            valid_log_pf = step_log_pf[valid][valid_steps]
            valid_log_pb = step_log_pb[valid][valid_steps]

            valid_count = int(valid_lengths.numel())
            terminal_index = valid_lengths.clamp(1, horizon) - 1

            valid_terminal_log_pb = step_log_pb[valid][
                torch.arange(valid_count, device=device),
                terminal_index,
            ]

            valid_raw_rewards = raw_log_rewards[valid]
            valid_rewards = log_rewards[valid]
            valid_log_z = root_log_z[valid]
            valid_terminal_residual = terminal_stop_residual[valid]
            subtrajectory_counts = valid_entry_mask.sum(dim=(1, 2)).float()

            return {
                "loss/total": loss.detach(),
                "loss/subtb": subtb_loss.detach(),
                "loss/stop_tb": stop_tb_loss.detach(),
                "loss/advantage_aux": advantage_aux_loss.detach(),
                "loss/advantage_aux_coef": advantage_aux_coef.detach(),
                "subtb/residual_abs_mean": _mean_or_zero(
                    valid_residuals.abs(),
                    zero,
                ),
                "subtb/residual_square_mean": _mean_or_zero(
                    valid_residuals.square(),
                    zero,
                ),
                "subtb/residual_mean": _mean_or_zero(valid_residuals, zero),
                "subtb/residual_std": _std_or_zero(valid_residuals, zero),
                "subtb/subtrajectory_count_mean": subtrajectory_counts.mean().detach(),
                "stop_tb/residual_abs_mean": _mean_or_zero(
                    valid_stop_tb_residuals.abs(),
                    zero,
                ),
                "stop_tb/residual_square_mean": _mean_or_zero(
                    valid_stop_tb_residuals.square(),
                    zero,
                ),
                "stop_tb/residual_mean": _mean_or_zero(
                    valid_stop_tb_residuals,
                    zero,
                ),
                "stop_tb/residual_std": _std_or_zero(
                    valid_stop_tb_residuals,
                    zero,
                ),
                "stop_tb/valid_count_mean": (
                    stop_tb_valid_mask.to(dtype=torch.float32)
                    .sum(dim=1)
                    .mean()
                    .detach()
                ),
                "advantage_aux/valid_count_mean": (
                    advantage_aux_valid_mask.to(dtype=torch.float32)
                    .sum(dim=1)
                    .mean()
                    .detach()
                ),
                "diagnostic/terminal_stop_balance_mse": (
                    valid_terminal_residual.square().mean().detach()
                ),
                "diagnostic/terminal_stop_residual_abs_mean": (
                    valid_terminal_residual.abs().mean().detach()
                ),
                "diagnostic/terminal_step_log_pb_abs_max": (
                    valid_terminal_log_pb.abs().max().detach()
                ),
                "flow/log_z_mean": valid_log_z.mean().detach(),
                "flow/log_z_std": _std_or_zero(valid_log_z, zero),
                "flow/state_log_flow_mean": _mean_or_zero(valid_state_flows, zero),
                "flow/state_log_flow_std": _std_or_zero(valid_state_flows, zero),
                "reward/log_reward_mean": valid_raw_rewards.mean().detach(),
                "reward/log_reward_std": _std_or_zero(valid_raw_rewards, zero),
                "reward/log_reward_clamped_mean": valid_rewards.mean().detach(),
                "reward/clipped_ratio": (
                    valid_raw_rewards.lt(self.log_reward_clip_min)
                    .float()
                    .mean()
                    .detach()
                ),
                "prob/step_log_pf_mean": _mean_or_zero(valid_log_pf, zero),
                "prob/step_log_pb_mean": _mean_or_zero(valid_log_pb, zero),
            }

    def _zero_output(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LossOutput:
        zero = torch.zeros((), device=device, dtype=dtype)
        return LossOutput(
            loss=zero,
            metrics={key: zero for key in self.metric_keys},
            per_trajectory_loss=torch.zeros(batch_size, device=device, dtype=dtype),
        )

    @staticmethod
    def _assert_terminal_stop(
        *,
        stop_mask: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
        horizon: int,
    ) -> None:
        device = lengths.device
        stop_mask = stop_mask.to(device=device, dtype=torch.bool)

        terminal_index = lengths.clamp(1, horizon) - 1
        batch_index = torch.arange(lengths.numel(), device=device)
        terminal_is_stop = stop_mask[batch_index, terminal_index]

        bad = torch.nonzero(valid & ~terminal_is_stop, as_tuple=False).view(-1)
        if bad.numel() > 0:
            raise ValueError(
                "Every valid trajectory must end with STOP. "
                f"Bad trajectory ids: {bad.tolist()}."
            )


def _subtrajectory_mask(
    *,
    lengths: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    step = torch.arange(horizon, device=device)
    inside = step.unsqueeze(0) < lengths.unsqueeze(1)
    upper = torch.ones(horizon, horizon, dtype=torch.bool, device=device).triu()
    return upper.unsqueeze(0) & inside.unsqueeze(1) & inside.unsqueeze(2)


def _segment_sums(step_values: torch.Tensor) -> torch.Tensor:
    batch_size = step_values.size(0)
    cumsum = step_values.cumsum(dim=1)
    shifted = torch.cat(
        [
            step_values.new_zeros(batch_size, 1),
            cumsum[:, :-1],
        ],
        dim=1,
    )
    return cumsum.unsqueeze(1) - shifted.unsqueeze(2)


def _subtrajectory_targets(
    *,
    state_log_flows: torch.Tensor,
    log_rewards: torch.Tensor,
    terminal_index: torch.Tensor,
) -> torch.Tensor:
    targets = torch.zeros_like(state_log_flows)

    if state_log_flows.size(1) > 1:
        targets[:, :-1] = state_log_flows[:, 1:]

    batch_index = torch.arange(state_log_flows.size(0), device=state_log_flows.device)
    targets[batch_index, terminal_index] = log_rewards

    return targets


def stop_now_tb_loss(
    *,
    state_log_flow: torch.Tensor,
    stop_log_pf: torch.Tensor,
    stop_now_log_reward: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    residual = state_log_flow + stop_log_pf - stop_now_log_reward
    residual = residual[valid_mask]
    if residual.numel() == 0:
        return state_log_flow.new_zeros(())
    return residual.square().mean()


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = values[valid_mask]
    if valid.numel() == 0:
        return values.new_zeros(())
    return valid.mean()


def _mean_or_zero(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return values.mean().detach() if values.numel() > 0 else zero.detach()


def _std_or_zero(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False).detach() if values.numel() > 1 else zero.detach()


__all__ = ["LossOutput", "SubTrajectoryBalanceLoss", "stop_now_tb_loss"]
