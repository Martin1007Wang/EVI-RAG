from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .rollout import RolloutBatch


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    per_trajectory_loss: torch.Tensor | None = None

    def prefixed_metrics(self, prefix: str = "") -> dict[str, torch.Tensor]:
        if not prefix:
            return self.metrics
        return {f"{prefix}/{k}": v for k, v in self.metrics.items()}

    def metric(self, name: str) -> torch.Tensor:
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise KeyError(f"Unknown loss metric {name!r}.") from exc

    @classmethod
    def aggregate(cls, outputs: Iterable["LossOutput"]) -> "LossOutput":
        """Average a collection of LossOutputs (for logging only, not backprop)."""
        aggregated: dict[str, torch.Tensor] = {}
        count = 0
        ref_device: torch.device | None = None

        for out in outputs:
            count += 1
            for key, value in out.metrics.items():
                if ref_device is None:
                    ref_device = value.device
                aggregated[key] = (
                    value if key not in aggregated else aggregated[key] + value
                )

        if count == 0:
            device = ref_device if ref_device is not None else torch.device("cpu")
            return cls(loss=torch.zeros((), device=device), metrics={}, per_trajectory_loss=None)

        avg_metrics = {k: v / count for k, v in aggregated.items()}
        avg_loss = avg_metrics.get(
            "loss",
            avg_metrics.get("fl_subtb_loss", torch.zeros((), device=ref_device)),
        )
        return cls(loss=avg_loss, metrics=avg_metrics, per_trajectory_loss=None)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _safe_var(t: torch.Tensor, *, unbiased: bool = False) -> torch.Tensor:
    return t.var(unbiased=unbiased) if t.numel() > 1 else t.new_zeros(())


def _safe_mean(t: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    return t.mean() if t.numel() > 0 else fallback


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _build_entry_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T, T) bool mask — True iff i ≤ j and both steps are within the trajectory.

    A trajectory of length L has valid *step indices* 0 … L-1 (the Stop step
    is recorded at index L-1).  entry_mask[b, i, j] is True when:
      - i ≤ j  (upper-triangular → sub-trajectory starts no later than it ends)
      - i < L  (start step within trajectory)
      - j < L  (end step within trajectory, including the terminal Stop step)
    """
    device = lengths.device
    idx = torch.arange(T, device=device)
    step_valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # (B, T): True iff step t is valid
    triu = torch.ones(T, T, device=device, dtype=torch.bool).triu()
    # step_valid[:, j] → unsqueeze(2) gives column mask (j axis)
    # step_valid[:, i] → unsqueeze(1) gives row mask (i axis)
    return triu.unsqueeze(0) & step_valid.unsqueeze(2) & step_valid.unsqueeze(1)


# ---------------------------------------------------------------------------
# Weight / shape matrices (pre-computed, stored as buffers)
# ---------------------------------------------------------------------------


def _subtb_weight_matrix(
    T: int,
    lam: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Upper-triangular W[i,j] = λ^(j-i).

    Special cases:
      lam = 0   → identity (Detailed Balance: only diagonal entries survive)
      lam = inf → only the full-trajectory entry [0, T-1] survives (Trajectory Balance)
    """
    if math.isinf(lam):
        W = torch.zeros(T, T, device=device, dtype=dtype)
        if T > 0:
            W[0, T - 1] = 1.0
        return W
    if lam == 0.0:
        return torch.eye(T, device=device, dtype=dtype)
    idx = torch.arange(T, device=device, dtype=dtype)
    return (lam ** (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp_min(0)).triu()


def _subtraj_length_matrix(T: int, *, device: torch.device) -> torch.Tensor:
    """Upper-triangular (T, T): entry [i,j] = j - i + 1 (sub-trajectory length)."""
    idx = torch.arange(T, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1) + 1).triu()


# ---------------------------------------------------------------------------
# Core FL-SubTB math
# ---------------------------------------------------------------------------


def _compute_residuals(
    state_log_flows: torch.Tensor,  # (B, T)
    net_log_ratio: torch.Tensor,    # (B, T)  η_t = log P_F - log P_B
    log_rewards: torch.Tensor,      # (B,)
    lengths: torch.Tensor,          # (B,) long — number of steps incl. Stop
    entry_mask: torch.Tensor,       # (B, T, T) bool
) -> torch.Tensor:
    """Compute FL-SubTB residual matrices, shape (B, T, T).

    δ(i, j) = F̃(s_i) + Σ_{t=i}^{j} η_t  −  target(j)

    where
        target(j) = F̃(s_{j+1})   for non-terminal j
        target(j) = log R(x)      for the terminal step j = L-1

    The residual is zeroed outside entry_mask.
    """
    B, T = state_log_flows.shape
    device = state_log_flows.device
    dtype = state_log_flows.dtype

    # Σ_{t=i}^{j} η_t = cumsum[j] - cumsum[i-1]
    cumsum      = torch.cumsum(net_log_ratio, dim=1)                              # (B, T)
    cumsum_prev = torch.cat(
        [torch.zeros(B, 1, device=device, dtype=dtype), cumsum[:, :-1]], dim=1
    )                                                                              # (B, T)
    cross = cumsum.unsqueeze(1) - cumsum_prev.unsqueeze(2)                        # (B, T, T)

    # Build target tensor: default = next-step flow; terminal = log R
    targets = torch.zeros_like(state_log_flows)
    if T > 1:
        targets[:, :-1] = state_log_flows[:, 1:]
    term_idx = lengths.clamp(min=1, max=T) - 1
    targets[torch.arange(B, device=device), term_idx] = log_rewards

    # Residual: (B, T, 1) - (B, 1, T) + (B, T, T)
    residuals = state_log_flows.unsqueeze(2) - targets.unsqueeze(1) + cross       # (B, T, T)

    return residuals * entry_mask  # zero out padding


def _weighted_mse(
    residuals: torch.Tensor,   # (B, T, T)
    weights: torch.Tensor,     # (T, T)   broadcast weight matrix W
    entry_mask: torch.Tensor,  # (B, T, T) bool
    *,
    global_normalize: bool = True,
) -> torch.Tensor:
    """Weighted MSE of residuals.

    global_normalize=True  → scalar: Σ(w·δ²) / Σw          (used for batch loss)
    global_normalize=False → (B,)  : Σ(w·δ²)[b] / Σw[b]    (used for per-traj loss)
    """
    w = weights.unsqueeze(0) * entry_mask.float()  # (B, T, T)
    if global_normalize:
        total_w = w.sum().clamp_min(torch.finfo(w.dtype).eps)
        return (w * residuals.square()).sum() / total_w
    else:
        denom = w.sum(dim=(1, 2)).clamp_min(torch.finfo(w.dtype).eps)
        return (w * residuals.square()).sum(dim=(1, 2)) / denom  # (B,)


def _reward_matching_loss(
    state_log_flows: torch.Tensor,  # (B, T)
    log_rewards: torch.Tensor,       # (B,)
    lengths: torch.Tensor,           # (B,) long
    T: int,
) -> torch.Tensor:
    """Anchor terminal-state flow to log R(x).

    Eliminates the additive-constant degree of freedom in FL-SubTB.
    """
    B = state_log_flows.shape[0]
    device = state_log_flows.device
    term_idx = lengths.clamp(min=1, max=T) - 1
    terminal_flows = state_log_flows[torch.arange(B, device=device), term_idx]
    return (terminal_flows - log_rewards).square().mean()


# ---------------------------------------------------------------------------
# Loss Module
# ---------------------------------------------------------------------------


class SubTrajectoryBalanceLoss(nn.Module):
    """Forward-Looking SubTrajectoryBalance(λ) loss for GFlowNet training.

    Minimises for every sub-trajectory [i, j]:

        ( F̃(s_i) + Σ_{t=i}^{j} (log P_F − log P_B) − target(j) )²

    weighted by λ^(j−i), where
        target(j) = log F̃(s_{j+1})   non-terminal j
        target(j) = log R(x)          terminal j

    Special cases controlled by subtb_lambda
    -----------------------------------------
    λ = 0   → Detailed Balance  (lowest variance, highest bias)
    λ = ∞   → Trajectory Balance (lowest bias, highest variance)
    0 < λ < ∞ → interpolation  (λ ≈ 0.9 is a good default)

    Reward-Matching regularisation (reward_matching_coef > 0) anchors the
    absolute scale of flow estimates to log R(x), eliminating the additive
    constant degree of freedom that SubTB alone cannot resolve.
    """

    _subtb_weight: torch.Tensor
    _subtraj_len: torch.Tensor

    def __init__(
        self,
        *,
        max_trajectory_len: int,
        variant: str = "fl_subtb",
        subtb_lambda: float = 0.9,
        log_reward_clip_min: float = -100.0,
        global_normalize: bool = True,
        reward_matching_coef: float = 0.0,
    ) -> None:
        super().__init__()
        if variant != "fl_subtb":
            raise ValueError(f"Unsupported variant {variant!r}; only 'fl_subtb' is available.")
        if subtb_lambda < 0.0:
            raise ValueError(f"subtb_lambda must be >= 0, got {subtb_lambda}.")
        if max_trajectory_len < 1:
            raise ValueError(f"max_trajectory_len must be >= 1, got {max_trajectory_len}.")
        if reward_matching_coef < 0.0:
            raise ValueError(f"reward_matching_coef must be >= 0, got {reward_matching_coef}.")

        self.variant = variant
        self.subtb_lambda = float(subtb_lambda)
        self.log_reward_clip_min = log_reward_clip_min
        self.global_normalize = bool(global_normalize)
        self.max_trajectory_len = max_trajectory_len
        self.reward_matching_coef = float(reward_matching_coef)

        self.register_buffer(
            "_subtb_weight",
            _subtb_weight_matrix(
                max_trajectory_len,
                subtb_lambda,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "_subtraj_len",
            _subtraj_length_matrix(max_trajectory_len, device=torch.device("cpu")).float(),
        )

    # ------------------------------------------------------------------

    def forward(self,rollout_batch: RolloutBatch,trajectory_weights: torch.Tensor | None = None,) -> LossOutput:
        stats  = rollout_batch.stats
        traces = rollout_batch.traces
        lengths    = stats.traj_len.long()
        valid_mask = lengths.gt(0)
        slf = traces.state_log_flows   # (B, T)
        pf  = traces.step_log_pf       # (B, T)
        pb  = traces.step_log_pb       # (B, T)

        # Keep raw log_reward for unbiased logging; use clamped version for loss.
        log_reward         = stats.terminal_log_rewards.float()
        log_reward_clamped = log_reward.clamp(min=self.log_reward_clip_min)

        B, T = slf.shape
        if T > self.max_trajectory_len:
            raise ValueError(f"T={T} exceeds max_trajectory_len={self.max_trajectory_len}.")

        # Slice pre-computed buffers to actual trajectory length T
        W           = self._subtb_weight[:T, :T]
        subtraj_len = self._subtraj_len[:T, :T]
        entry_mask  = _build_entry_mask(lengths, T)

        # η_t = log P_F(a_t|s_t) − log P_B(a_{t-1}|s_t)
        net_log_ratio = pf - pb

        residuals = _compute_residuals(
            state_log_flows=slf,
            net_log_ratio=net_log_ratio,
            log_rewards=log_reward_clamped,
            lengths=lengths,
            entry_mask=entry_mask,
        )  # (B, T, T)

        # Per-trajectory FL and RM losses (used for weighted aggregation and logging)
        traj_fl_losses = _weighted_mse(residuals, W, entry_mask, global_normalize=False)  # (B,)

        term_idx_all      = lengths.clamp(min=1, max=T) - 1
        terminal_flows_all = slf[torch.arange(B, device=slf.device), term_idx_all]
        traj_rm_losses    = (terminal_flows_all - log_reward_clamped).square()             # (B,)
        traj_total_losses = traj_fl_losses + self.reward_matching_coef * traj_rm_losses    # (B,)

        # ── Aggregate to scalar loss ───────────────────────────────────
        if trajectory_weights is not None:
            if trajectory_weights.shape != (B,):
                raise ValueError(
                    f"trajectory_weights must have shape ({B},), "
                    f"got {tuple(trajectory_weights.shape)}."
                )
            valid_w = (
                trajectory_weights.to(device=slf.device, dtype=slf.dtype)[valid_mask]
                .clamp_min(0.0)
            )
            denom   = valid_w.sum().clamp_min(torch.finfo(valid_w.dtype).eps)

            fl_loss = (traj_fl_losses[valid_mask] * valid_w).sum() / denom
            if self.reward_matching_coef > 0.0:
                rm_loss = (traj_rm_losses[valid_mask] * valid_w).sum() / denom
                loss    = (traj_total_losses[valid_mask] * valid_w).sum() / denom
            else:
                rm_loss = slf.new_zeros(())
                loss    = fl_loss
        else:
            if self.global_normalize:
                fl_loss = _weighted_mse(
                    residuals[valid_mask],
                    W,
                    entry_mask[valid_mask],
                    global_normalize=True,
                )
            else:
                fl_loss = traj_fl_losses[valid_mask].mean()

            if self.reward_matching_coef > 0.0:
                rm_loss = _reward_matching_loss(
                    state_log_flows=slf[valid_mask],
                    log_rewards=log_reward_clamped[valid_mask],
                    lengths=lengths[valid_mask],
                    T=T,
                )
                loss = fl_loss + self.reward_matching_coef * rm_loss
            else:
                rm_loss = slf.new_zeros(())
                loss    = fl_loss

        # ── Monitoring metrics (no-grad) ───────────────────────────────
        with torch.no_grad():
            entry_mask_valid = entry_mask[valid_mask]   # (V, T, T)
            n_valid          = entry_mask_valid.shape[0]

            v_slf    = slf[valid_mask]
            v_pf     = stats.trajectory_log_pf.float()[valid_mask]
            v_pb     = stats.trajectory_log_pb.float()[valid_mask]
            v_log_z  = stats.root_log_z.float()[valid_mask]
            # Use CLAMPED reward for all loss-related metrics (matches what the
            # loss was actually computed against), but log the RAW reward mean
            # separately so monitoring reflects true reward distribution.
            v_reward_clamped = log_reward_clamped[valid_mask]
            v_reward_raw     = log_reward[valid_mask]
            v_len            = lengths.float()[valid_mask]

            step_valid_v = (
                torch.arange(T, device=slf.device).unsqueeze(0)
                < lengths[valid_mask].unsqueeze(1)
            )  # (V, T) bool

            valid_residuals    = residuals[valid_mask][entry_mask_valid]    # (N_entries,)
            valid_flows        = v_slf[step_valid_v]                        # (N_steps,)
            # Expand subtraj_len to (n_valid, T, T) and index with entry_mask
            valid_subtraj_len  = (
                subtraj_len.unsqueeze(0).expand(n_valid, -1, -1)[entry_mask_valid].float()
            )
            subtrajectory_count = entry_mask_valid.sum(dim=(1, 2)).float()  # (V,)

            term_idx_v      = lengths[valid_mask].clamp(min=1, max=T) - 1
            terminal_flows_v = v_slf[torch.arange(n_valid, device=slf.device), term_idx_v]

            fallback = slf.new_zeros(())
            metrics: dict[str, torch.Tensor] = {
                # --- primary loss signals ---
                "loss":                    loss.detach(),
                "fl_subtb_loss":           fl_loss.detach(),
                "reward_matching_loss":    rm_loss.detach(),
                # --- residual diagnostics ---
                "residual_abs_mean":       _safe_mean(valid_residuals.abs(), fallback).detach(),
                "residual_variance":       _safe_var(valid_residuals).detach(),
                # --- flow diagnostics ---
                "log_z_mean":              v_log_z.mean().detach(),
                "log_z_variance":          _safe_var(v_log_z).detach(),
                "terminal_flow_mean":      terminal_flows_v.mean().detach(),
                "terminal_flow_vs_reward": (terminal_flows_v - v_reward_clamped).mean().detach(),
                "state_flow_mean":         _safe_mean(valid_flows, fallback).detach(),
                "state_flow_variance":     _safe_var(valid_flows).detach(),
                # --- reward diagnostics (raw = unclipped, for faithful monitoring) ---
                "log_reward_mean":         v_reward_raw.mean().detach(),
                "log_reward_clamped_mean": v_reward_clamped.mean().detach(),
                # Fraction of trajectories whose reward hit the clip floor
                "log_reward_clipped_ratio": (
                    (v_reward_raw < self.log_reward_clip_min).float().mean().detach()
                ),
                # --- policy / trajectory diagnostics ---
                "log_pf_mean":               v_pf.mean().detach(),
                "log_pb_mean":               v_pb.mean().detach(),
                "trajectory_length_mean":    v_len.mean().detach(),
                "valid_trajectory_ratio":    valid_mask.float().mean().detach(),
                "subtrajectory_length_mean": _safe_mean(valid_subtraj_len, fallback).detach(),
                "subtrajectory_count_mean":  subtrajectory_count.mean().detach(),
            }

        return LossOutput(
            loss=loss,
            metrics=metrics,
            per_trajectory_loss=traj_total_losses.detach(),
        )


__all__ = ["LossOutput", "SubTrajectoryBalanceLoss"]
