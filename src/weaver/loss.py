from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.weaver.rollout.schema import RolloutBatch


@dataclass(frozen=True, slots=True)
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

        loss = metrics.get("loss/total")
        if loss is None:
            loss = torch.zeros((), device=outputs[0].loss.device)

        return cls(loss=loss, metrics=metrics, per_trajectory_loss=None)


class SubTrajectoryBalanceLoss(nn.Module):
    """
    SubTrajectory Balance with explicit stopping supervision.

    Main GFlowNet objective:

        eps(i,j)
            = log F(s_i)
              + sum_{t=i..j} [
                    log P_F(a_t | s_t)
                    - log P_B(s_t | s_{t+1})
                ]
              - target(j)

    where:

        target(j) =
            log F(s_{j+1})    if j is non-terminal
            log R(x)          if j is the terminal STOP step

    Terminal STOP has no backward removal action, so the effective terminal
    log P_B is set to 0.

    StopTB counterfactual:

        log F(s) + log P_F(Stop | s) ~= log R_stop(s)

    StopAdv boundary supervision:

        y_stop(s) ~= sigmoid((J_stop(s) - J_continue(s)) / tau)

        L_StopAdv
            = - y_stop log P_F(Stop | s)
              - (1 - y_stop) log P_F(Expand | s)

    StopAdv is not an edge imitation loss. It only supervises the Stop/Expand
    option boundary.
    """

    metric_keys = (
        "loss/total",
        "loss/subtb",
        "loss/stop_tb",
        "loss/stop_adv",
        "loss/subtb_coef",
        "loss/stop_tb_coef",
        "loss/stop_adv_coef",
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
        "stop_adv/valid_count_mean",
        "stop_adv/target_mean",
        "stop_adv/target_stop_ratio",
        "stop_adv/pred_stop_prob_mean",
        "stop_adv/pred_stop_prob_when_target_stop",
        "stop_adv/pred_stop_prob_when_target_continue",
        "stop_adv/stop_now_better_ratio",
        "stop_adv/continue_minus_stop_log_reward_mean",
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
        subtb_lambda: float = 0.9,
        subtb_coef: float = 1.0,
        stop_tb_coef: float = 1.0,
        stop_adv_coef: float = 0.0,
        log_reward_clip_min: float = -30.0,
        debug: bool = False,
    ) -> None:
        super().__init__()

        if int(max_trajectory_len) < 1:
            raise ValueError(
                f"max_trajectory_len must be >= 1, got {max_trajectory_len}."
            )
        if not 0.0 <= float(subtb_lambda) <= 1.0:
            raise ValueError(f"subtb_lambda must be in [0, 1], got {subtb_lambda}.")
        if float(subtb_coef) < 0.0:
            raise ValueError(f"subtb_coef must be >= 0, got {subtb_coef}.")
        if float(stop_tb_coef) < 0.0:
            raise ValueError(f"stop_tb_coef must be >= 0, got {stop_tb_coef}.")
        if float(stop_adv_coef) < 0.0:
            raise ValueError(f"stop_adv_coef must be >= 0, got {stop_adv_coef}.")

        self.max_trajectory_len = int(max_trajectory_len)
        self.subtb_lambda = float(subtb_lambda)
        self.subtb_coef = float(subtb_coef)
        self.stop_tb_coef = float(stop_tb_coef)
        self.stop_adv_coef = float(stop_adv_coef)
        self.log_reward_clip_min = float(log_reward_clip_min)
        self.debug = bool(debug)

    @property
    def requires_stop_now_reward(self) -> bool:
        return self.stop_tb_coef > 0.0

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
        effective_log_pb[
            torch.arange(batch_size, device=device),
            terminal_index,
        ] = 0.0

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
        )

        valid_entry = entry_mask.to(dtype)
        residuals = residuals * valid_entry
        weights = weights * valid_entry

        denom = weights.sum(dim=(1, 2)).clamp_min(torch.finfo(dtype).eps)
        per_trajectory_loss = (weights * residuals.square()).sum(dim=(1, 2)) / denom
        subtb_loss = per_trajectory_loss[valid].mean()

        stop_now_log_reward = traces.stop_now_log_reward.to(
            device=device,
            dtype=dtype,
        ).clamp(min=self.log_reward_clip_min)

        stop_now_valid_mask = traces.stop_now_valid_mask.to(
            device=device,
            dtype=torch.bool,
        )

        stop_log_pf = traces.stop_log_pf.to(device=device, dtype=dtype)

        stop_tb_valid_mask = (
            traces.stop_tb_valid_mask.to(device=device, dtype=torch.bool)
            & stop_now_valid_mask
        )

        stop_tb_loss = stop_now_tb_loss(
            state_log_flow=state_log_flows,
            stop_log_pf=stop_log_pf,
            stop_now_log_reward=stop_now_log_reward,
            valid_mask=stop_tb_valid_mask,
        )

        stop_adv_target = _optional_tensor(
            getattr(traces, "stop_adv_target", None),
            device=device,
            dtype=dtype,
        )
        stop_adv_valid_mask = _optional_tensor(
            getattr(traces, "stop_adv_valid_mask", None),
            device=device,
            dtype=torch.bool,
        )

        stop_adv_loss = stop_advantage_loss(
            stop_log_pf=stop_log_pf,
            target=stop_adv_target,
            valid_mask=stop_adv_valid_mask,
        )

        loss = (
            self.subtb_coef * subtb_loss
            + self.stop_tb_coef * stop_tb_loss
            + self.stop_adv_coef * stop_adv_loss
        )

        terminal_stop_residual = residuals[
            torch.arange(batch_size, device=device),
            terminal_index,
            terminal_index,
        ]

        stop_adv_continue_log_reward = _optional_tensor(
            getattr(traces, "stop_adv_continue_log_reward", None),
            device=device,
            dtype=dtype,
        )

        metrics = self._metrics(
            loss=loss,
            subtb_loss=subtb_loss,
            stop_tb_loss=stop_tb_loss,
            stop_adv_loss=stop_adv_loss,
            raw_log_rewards=raw_log_rewards,
            log_rewards=log_rewards,
            state_log_flows=state_log_flows,
            step_log_pf=step_log_pf,
            step_log_pb=step_log_pb,
            stop_log_pf=stop_log_pf,
            stop_now_log_reward=stop_now_log_reward,
            stop_tb_residuals=state_log_flows + stop_log_pf - stop_now_log_reward,
            stop_tb_valid_mask=stop_tb_valid_mask,
            stop_adv_target=stop_adv_target,
            stop_adv_valid_mask=stop_adv_valid_mask,
            stop_adv_continue_log_reward=stop_adv_continue_log_reward,
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
        stop_adv_loss: torch.Tensor,
        raw_log_rewards: torch.Tensor,
        log_rewards: torch.Tensor,
        state_log_flows: torch.Tensor,
        step_log_pf: torch.Tensor,
        step_log_pb: torch.Tensor,
        stop_log_pf: torch.Tensor,
        stop_now_log_reward: torch.Tensor,
        stop_tb_residuals: torch.Tensor,
        stop_tb_valid_mask: torch.Tensor,
        stop_adv_target: torch.Tensor | None,
        stop_adv_valid_mask: torch.Tensor | None,
        stop_adv_continue_log_reward: torch.Tensor | None,
        residuals: torch.Tensor,
        entry_mask: torch.Tensor,
        terminal_stop_residual: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
        root_log_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            horizon = state_log_flows.size(1)
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

            terminal_index = valid_lengths.clamp(1, horizon) - 1
            valid_terminal_log_pb = step_log_pb[valid][
                torch.arange(valid_lengths.numel(), device=device),
                terminal_index,
            ]

            valid_raw_rewards = raw_log_rewards[valid]
            valid_rewards = log_rewards[valid]
            valid_log_z = root_log_z[valid]
            valid_terminal_residual = terminal_stop_residual[valid]
            subtrajectory_counts = valid_entry_mask.sum(dim=(1, 2)).float()

            stop_adv_valid = (
                stop_adv_valid_mask
                if stop_adv_valid_mask is not None
                else torch.zeros_like(stop_log_pf, dtype=torch.bool)
            )

            if stop_adv_target is not None:
                valid_stop_adv_target = stop_adv_target[stop_adv_valid]
            else:
                valid_stop_adv_target = state_log_flows.new_zeros((0,))

            pred_stop_prob = stop_log_pf.exp().clamp(0.0, 1.0)
            valid_pred_stop_prob = pred_stop_prob[stop_adv_valid]

            target_stop_mask = (
                stop_adv_target.ge(0.5) & stop_adv_valid
                if stop_adv_target is not None
                else torch.zeros_like(stop_adv_valid)
            )
            target_continue_mask = (
                stop_adv_target.lt(0.5) & stop_adv_valid
                if stop_adv_target is not None
                else torch.zeros_like(stop_adv_valid)
            )

            if stop_adv_continue_log_reward is not None:
                valid_stop_adv_delta = (
                    stop_adv_continue_log_reward[stop_adv_valid]
                    - stop_now_log_reward[stop_adv_valid]
                )
            else:
                valid_stop_adv_delta = state_log_flows.new_zeros((0,))

            return {
                "loss/total": loss.detach(),
                "loss/subtb": subtb_loss.detach(),
                "loss/stop_tb": stop_tb_loss.detach(),
                "loss/stop_adv": stop_adv_loss.detach(),
                "loss/subtb_coef": state_log_flows.new_tensor(self.subtb_coef).detach(),
                "loss/stop_tb_coef": state_log_flows.new_tensor(
                    self.stop_tb_coef
                ).detach(),
                "loss/stop_adv_coef": state_log_flows.new_tensor(
                    self.stop_adv_coef
                ).detach(),
                "subtb/residual_abs_mean": _mean_or_zero(valid_residuals.abs(), zero),
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
                "stop_adv/valid_count_mean": (
                    stop_adv_valid.to(dtype=torch.float32).sum(dim=1).mean().detach()
                ),
                "stop_adv/target_mean": _mean_or_zero(
                    valid_stop_adv_target,
                    zero,
                ),
                "stop_adv/target_stop_ratio": _mean_or_zero(
                    valid_stop_adv_target.ge(0.5).to(dtype=torch.float32),
                    zero,
                ),
                "stop_adv/pred_stop_prob_mean": _mean_or_zero(
                    valid_pred_stop_prob,
                    zero,
                ),
                "stop_adv/pred_stop_prob_when_target_stop": _mean_or_zero(
                    pred_stop_prob[target_stop_mask],
                    zero,
                ),
                "stop_adv/pred_stop_prob_when_target_continue": _mean_or_zero(
                    pred_stop_prob[target_continue_mask],
                    zero,
                ),
                "stop_adv/stop_now_better_ratio": _mean_or_zero(
                    valid_stop_adv_delta.le(0.0).to(dtype=torch.float32),
                    zero,
                ),
                "stop_adv/continue_minus_stop_log_reward_mean": _mean_or_zero(
                    valid_stop_adv_delta,
                    zero,
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


def stop_advantage_loss(
    *,
    stop_log_pf: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    BCE over the Stop/Expand option using log P(Stop | s).

    target:
        Soft stop target in [0, 1].

    valid_mask:
        States where the stop-advantage oracle was evaluated.

    This function intentionally computes the BCE here instead of consuming a
    precomputed loss tensor, so gradients flow through stop_log_pf.
    """
    if target is None:
        return stop_log_pf.new_zeros(())

    if valid_mask is None:
        raise ValueError("stop_adv_valid_mask is required when stop_adv_target exists.")

    target = target.to(device=stop_log_pf.device, dtype=stop_log_pf.dtype)
    valid_mask = valid_mask.to(device=stop_log_pf.device, dtype=torch.bool)

    if target.shape != stop_log_pf.shape:
        raise ValueError(
            "stop_adv_target must have the same shape as stop_log_pf: "
            f"{tuple(target.shape)} != {tuple(stop_log_pf.shape)}."
        )
    if valid_mask.shape != stop_log_pf.shape:
        raise ValueError(
            "stop_adv_valid_mask must have the same shape as stop_log_pf: "
            f"{tuple(valid_mask.shape)} != {tuple(stop_log_pf.shape)}."
        )

    logp_stop = stop_log_pf[valid_mask]
    target = target[valid_mask].clamp(0.0, 1.0)

    if logp_stop.numel() == 0:
        return stop_log_pf.new_zeros(())

    p_stop = logp_stop.exp().clamp(min=float(eps), max=1.0 - float(eps))
    logp_stop = p_stop.log()
    logp_expand = torch.log1p(-p_stop)

    loss = -(target * logp_stop + (1.0 - target) * logp_expand)
    return loss.mean()


def _optional_tensor(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device=device, dtype=dtype)


def _mean_or_zero(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return values.mean().detach() if values.numel() > 0 else zero.detach()


def _std_or_zero(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False).detach() if values.numel() > 1 else zero.detach()


__all__ = [
    "LossOutput",
    "SubTrajectoryBalanceLoss",
    "stop_advantage_loss",
    "stop_now_tb_loss",
]
