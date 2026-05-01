from __future__ import annotations

from typing import TYPE_CHECKING

from src.training.metric_utils import scalar_float

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch
    from src.weaver.loss import LossOutput
    from src.weaver.policy import Policy
    from src.weaver.rollout.schema import RolloutBatch


class TrainingDiagnosticsCollector:
    """Collect train-side diagnostics without owning logging or IO."""

    CORE_LOSS_KEYS = {
        "loss/total": "loss/total",
        "loss/subtb": "loss/subtb",
        "loss/stop_tb": "loss/stop_tb",
        "loss/stop_adv": "loss/stop_adv",
        "loss/potential_coef": "loss/potential_coef",
        "subtb/residual_abs_mean": "loss/residual_abs_mean",
        "subtb/residual_square_mean": "loss/residual_square_mean",
        "subtb/residual_mean": "loss/residual_mean",
        "stop_tb/residual_abs_mean": "stop_tb/residual_abs_mean",
        "stop_tb/residual_square_mean": "stop_tb/residual_square_mean",
        "stop_tb/residual_mean": "stop_tb/residual_mean",
        "stop_adv/valid_count_mean": "stop_adv/valid_count_mean",
        "stop_adv/target_mean": "stop_adv/target_mean",
        "stop_adv/stop_now_better_ratio": "stop_adv/stop_now_better_ratio",
        "stop_adv/continue_minus_stop_log_reward_mean": (
            "stop_adv/continue_minus_stop_log_reward_mean"
        ),
        "flow/log_z_mean": "flow/root_log_z_mean",
        "flow/log_z_std": "flow/root_log_z_std",
        "flow/state_log_flow_mean": "flow/state_log_flow_mean",
        "flow/state_log_flow_std": "flow/state_log_flow_std",
        "reward/log_reward_mean": "reward/log_reward_mean",
        "reward/log_reward_std": "reward/log_reward_std",
        "reward/clipped_ratio": "reward/clipped_ratio",
    }

    DEBUG_LOSS_KEYS = {
        "subtb/residual_std": "debug/residual_std",
        "stop_tb/residual_std": "debug/stop_tb_residual_std",
        "stop_tb/valid_count_mean": "debug/stop_tb_valid_count_mean",
        "potential/valid_count_mean": "debug/potential_valid_count_mean",
        "potential/delta_mean": "debug/potential_delta_mean",
        "potential/delta_abs_mean": "debug/potential_delta_abs_mean",
        "potential/delta_std": "debug/potential_delta_std",
        "diagnostic/terminal_stop_balance_mse": "debug/terminal_stop_balance_mse",
        "diagnostic/terminal_stop_residual_abs_mean": (
            "debug/terminal_stop_residual_abs_mean"
        ),
        "diagnostic/terminal_step_log_pb_abs_max": "debug/terminal_step_log_pb_abs_max",
        "reward/log_reward_clamped_mean": "debug/log_reward_clamped_mean",
        "prob/step_log_pf_mean": "debug/step_log_pf_mean",
        "prob/step_log_pb_mean": "debug/step_log_pb_mean",
        "subtb/subtrajectory_count_mean": "debug/subtrajectory_count_mean",
    }

    def __init__(
        self,
        *,
        debug: bool,
        rollout_diagnostics: bool = True,
        rollout_diagnostics_interval: int = 1,
        policy_diagnostics: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.rollout_diagnostics = bool(rollout_diagnostics)
        self.rollout_diagnostics_interval = int(rollout_diagnostics_interval)
        self.policy_diagnostics = bool(policy_diagnostics)
        if self.rollout_diagnostics_interval < 0:
            raise ValueError(
                "rollout_diagnostics_interval must be >= 0, "
                f"got {self.rollout_diagnostics_interval}."
            )

    def collect(
        self,
        *,
        loss_output: "LossOutput",
        batch: RetrievalBatch | None = None,
        online_rollouts: tuple["RolloutBatch", ...] = (),
        policy: "Policy | None" = None,
        root_expand_budget: int = 3,
        global_step: int | None = None,
    ) -> dict[str, float]:
        metrics = self._loss_metrics(loss_output)
        all_rollouts = online_rollouts
        if all_rollouts:
            from src.training.rollout_diagnostics import (
                compute_terminal_reward_diagnostics,
            )

            metrics.update(compute_terminal_reward_diagnostics(all_rollouts))
        collect_rollout_diagnostics = (
            batch is not None and self._should_collect_rollout_diagnostics(global_step)
        )
        if collect_rollout_diagnostics and all_rollouts:
            from src.training.rollout_diagnostics import (
                collect_training_rollout_diagnostics,
            )

            metrics.update(
                collect_training_rollout_diagnostics(
                    all_rollouts,
                    batch=batch,
                    debug=self.debug,
                )
            )
        if self.policy_diagnostics and policy is not None and batch is not None:
            from src.training.rollout_diagnostics import (
                compute_root_answer_edge_ranking_diagnostics,
            )

            metrics.update(
                compute_root_answer_edge_ranking_diagnostics(
                    policy,
                    batch=batch,
                    expand_budget=int(root_expand_budget),
                )
            )
        return {f"train/{name}": value for name, value in metrics.items()}

    def _loss_metrics(self, loss_output: "LossOutput") -> dict[str, float]:
        key_map = dict(self.CORE_LOSS_KEYS)
        if self.debug:
            key_map.update(self.DEBUG_LOSS_KEYS)

        return {
            output_name: scalar_float(loss_output.metrics[input_name])
            for input_name, output_name in key_map.items()
            if input_name in loss_output.metrics
        }

    def _should_collect_rollout_diagnostics(self, global_step: int | None) -> bool:
        if not self.rollout_diagnostics:
            return False
        if self.rollout_diagnostics_interval <= 0:
            return False
        if global_step is None:
            return True
        return int(global_step) % self.rollout_diagnostics_interval == 0


__all__ = ["TrainingDiagnosticsCollector"]
