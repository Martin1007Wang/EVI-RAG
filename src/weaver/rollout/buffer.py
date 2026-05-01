from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.weaver.reward import TerminalRewardOutput

from .schema import StepResult


@dataclass(slots=True)
class RolloutBuffer:
    """
    Mutable storage for one batched rollout.

    Shapes:
        B: number of physical graphs
        T: maximum number of decision steps

    Action convention:
        0 = Continue / Expand one edge
        1 = Stop

    Stored trajectory convention:
        state_log_flows[b, t] = log F(s_t | q)
        step_log_pf[b, t]     = log P_F(a_t | s_t)
        step_log_pb[b, t]     = log P_B(s_t | s_{t+1})
        traj_len[b]           = number of executed transitions, including Stop

    StopAdv convention:
        The buffer stores only oracle targets and diagnostics:

            stop_adv_target[b, t]
            stop_adv_valid_mask[b, t]
            stop_adv_continue_log_reward[b, t]

        It never stores stop_adv_loss. StopAdv BCE is computed in loss.py from
        stop_log_pf and stop_adv_target so gradients flow through the Stop/Expand
        option gate.
    """

    B: int
    T: int
    device: torch.device

    is_terminated: torch.Tensor = field(init=False)
    traj_len: torch.Tensor = field(init=False)

    terminal_log_reward: torch.Tensor = field(init=False)
    terminal_answer_f1: torch.Tensor = field(init=False)
    terminal_complexity_penalty: torch.Tensor = field(init=False)
    terminal_base_log_reward: torch.Tensor = field(init=False)
    terminal_utility: torch.Tensor = field(init=False)
    terminal_expanded_edge_count: torch.Tensor = field(init=False)
    terminal_answer_degree_excess: torch.Tensor = field(init=False)

    root_log_z: torch.Tensor = field(init=False)
    root_log_z_written: bool = field(init=False)
    state_log_flows: torch.Tensor = field(init=False)
    step_log_pf: torch.Tensor = field(init=False)
    step_log_pb: torch.Tensor = field(init=False)

    action_type: torch.Tensor = field(init=False)
    selected_edge_ids: torch.Tensor = field(init=False)

    continue_mask: torch.Tensor = field(init=False)
    stop_mask: torch.Tensor = field(init=False)

    stop_now_log_reward: torch.Tensor = field(init=False)
    stop_now_answer_f1: torch.Tensor = field(init=False)
    stop_now_valid_mask: torch.Tensor = field(init=False)

    stop_log_pf: torch.Tensor = field(init=False)
    stop_tb_valid_mask: torch.Tensor = field(init=False)

    target_continue_prob: torch.Tensor = field(init=False)
    target_stop_prob: torch.Tensor = field(init=False)
    policy_action_valid_mask: torch.Tensor = field(init=False)

    edge_action_entropy: torch.Tensor = field(init=False)
    edge_action_entropy_valid_mask: torch.Tensor = field(init=False)

    budget_exhausted_mask: torch.Tensor = field(init=False)

    stop_adv_target: torch.Tensor = field(init=False)
    stop_adv_valid_mask: torch.Tensor = field(init=False)
    stop_adv_continue_log_reward: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.B = int(self.B)
        self.T = int(self.T)

        if self.B <= 0:
            raise ValueError(f"B must be positive, got {self.B}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")

        b = (self.B,)
        bt = (self.B, self.T)

        self.is_terminated = self._zeros_bool(b)
        self.traj_len = self._zeros_long(b)

        self.terminal_log_reward = self._zeros_float(b)
        self.terminal_answer_f1 = self._zeros_float(b)
        self.terminal_complexity_penalty = self._zeros_float(b)
        self.terminal_base_log_reward = self._zeros_float(b)
        self.terminal_utility = self._zeros_float(b)
        self.terminal_expanded_edge_count = self._zeros_float(b)
        self.terminal_answer_degree_excess = self._zeros_float(b)

        self.root_log_z = self._zeros_float(b)
        self.root_log_z_written = False
        self.state_log_flows = self._zeros_float(bt)
        self.step_log_pf = self._zeros_float(bt)
        self.step_log_pb = self._zeros_float(bt)

        self.action_type = torch.full(
            bt,
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.selected_edge_ids = torch.full(
            bt,
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.continue_mask = self._zeros_bool(bt)
        self.stop_mask = self._zeros_bool(bt)

        self.stop_now_log_reward = self._zeros_float(bt)
        self.stop_now_answer_f1 = self._zeros_float(bt)
        self.stop_now_valid_mask = self._zeros_bool(bt)

        self.stop_log_pf = self._zeros_float(bt)
        self.stop_tb_valid_mask = self._zeros_bool(bt)

        self.target_continue_prob = self._zeros_float(bt)
        self.target_stop_prob = self._zeros_float(bt)
        self.policy_action_valid_mask = self._zeros_bool(bt)

        self.edge_action_entropy = self._zeros_float(bt)
        self.edge_action_entropy_valid_mask = self._zeros_bool(bt)

        self.budget_exhausted_mask = self._zeros_bool(bt)

        self.stop_adv_target = self._zeros_float(bt)
        self.stop_adv_valid_mask = self._zeros_bool(bt)
        self.stop_adv_continue_log_reward = self._zeros_float(bt)

    @property
    def active(self) -> torch.Tensor:
        return ~self.is_terminated

    def write_root_log_z(self, root_log_z: torch.Tensor) -> None:
        if self.root_log_z_written:
            raise RuntimeError("root_log_z has already been written.")

        self.root_log_z = self._as_float(
            root_log_z,
            shape=(self.B,),
            name="root_log_z",
        )
        self.root_log_z_written = True

    def write_state_log_flow(
        self,
        *,
        t: int,
        active: torch.Tensor,
        state_log_flow: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = self._as_bool(active, shape=(self.B,), name="active")
        flow = self._as_float(
            state_log_flow,
            shape=(self.B,),
            name="state_log_flow",
        )

        self.state_log_flows[active, t] = flow[active]

    def write_step(self, t: int, result: StepResult) -> None:
        self._check_t(t)

        active = self.active

        log_pf = self._as_float(result.log_pf, shape=(self.B,), name="log_pf")
        log_pb = self._as_float(result.log_pb, shape=(self.B,), name="log_pb")
        action_type = self._as_long(
            result.action_type,
            shape=(self.B,),
            name="action_type",
        )
        selected_edge_ids = self._as_long(
            result.selected_edge_ids,
            shape=(self.B,),
            name="selected_edge_ids",
        )
        continue_mask = self._as_bool(
            result.continue_mask,
            shape=(self.B,),
            name="continue_mask",
        )
        stop_mask = self._as_bool(
            result.stop_mask,
            shape=(self.B,),
            name="stop_mask",
        )

        self.step_log_pf[active, t] = log_pf[active]
        self.step_log_pb[active, t] = log_pb[active]
        self.action_type[active, t] = action_type[active]
        self.selected_edge_ids[active, t] = selected_edge_ids[active]
        self.continue_mask[active, t] = continue_mask[active]
        self.stop_mask[active, t] = stop_mask[active]

        stopped = active & stop_mask
        if not bool(stopped.any()):
            return

        self.is_terminated[stopped] = True
        self.traj_len[stopped] = int(t) + 1
        self._write_terminal_from_step(stopped=stopped, result=result)

    def write_stop_now_reward(
        self,
        *,
        t: int,
        active: torch.Tensor,
        reward_output: TerminalRewardOutput,
    ) -> None:
        self._check_t(t)

        active = self._as_bool(active, shape=(self.B,), name="active")

        log_reward = self._as_float(
            reward_output.log_reward,
            shape=(self.B,),
            name="stop_now.log_reward",
        )
        answer_f1 = self._as_float(
            reward_output.answer_f1,
            shape=(self.B,),
            name="stop_now.answer_f1",
        )

        self.stop_now_log_reward[active, t] = log_reward[active]
        self.stop_now_answer_f1[active, t] = answer_f1[active]
        self.stop_now_valid_mask[active, t] = True

    def write_stop_counterfactual(
        self,
        *,
        t: int,
        active: torch.Tensor,
        reward_output: TerminalRewardOutput,
    ) -> None:
        self.write_stop_now_reward(
            t=t,
            active=active,
            reward_output=reward_output,
        )

    def write_policy_step_diagnostics(
        self,
        *,
        t: int,
        active: torch.Tensor,
        target_stop_prob: torch.Tensor,
        target_continue_prob: torch.Tensor,
        stop_log_pf: torch.Tensor,
        action_valid_mask: torch.Tensor,
        stop_tb_valid_mask: torch.Tensor,
        edge_action_entropy: torch.Tensor,
        edge_action_entropy_valid_mask: torch.Tensor,
        budget_exhausted_mask: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = self._as_bool(active, shape=(self.B,), name="active")

        target_stop_prob = self._as_float(
            target_stop_prob,
            shape=(self.B,),
            name="target_stop_prob",
        )
        target_continue_prob = self._as_float(
            target_continue_prob,
            shape=(self.B,),
            name="target_continue_prob",
        )
        stop_log_pf = self._as_float(
            stop_log_pf,
            shape=(self.B,),
            name="stop_log_pf",
        )
        action_valid_mask = self._as_bool(
            action_valid_mask,
            shape=(self.B,),
            name="action_valid_mask",
        )
        stop_tb_valid = self._as_bool(
            stop_tb_valid_mask,
            shape=(self.B,),
            name="stop_tb_valid_mask",
        )
        edge_entropy = self._as_float(
            edge_action_entropy,
            shape=(self.B,),
            name="edge_action_entropy",
        )
        edge_entropy_valid = self._as_bool(
            edge_action_entropy_valid_mask,
            shape=(self.B,),
            name="edge_action_entropy_valid_mask",
        )
        budget_exhausted = self._as_bool(
            budget_exhausted_mask,
            shape=(self.B,),
            name="budget_exhausted_mask",
        )

        self.target_stop_prob[active, t] = target_stop_prob[active]
        self.target_continue_prob[active, t] = target_continue_prob[active]
        self.stop_log_pf[active, t] = stop_log_pf[active]
        self.policy_action_valid_mask[active, t] = action_valid_mask[active]
        self.stop_tb_valid_mask[active, t] = stop_tb_valid[active]
        self.edge_action_entropy[active, t] = edge_entropy[active]
        self.edge_action_entropy_valid_mask[active, t] = edge_entropy_valid[active]
        self.budget_exhausted_mask[active, t] = budget_exhausted[active]

    def write_stop_advantage(
        self,
        *,
        t: int,
        active: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        continue_log_reward: torch.Tensor,
    ) -> None:
        """
        Write StopAdv oracle targets.

        This method intentionally does not accept a loss tensor. StopAdv BCE is
        computed in loss.py from stop_log_pf and stop_adv_target, preserving
        gradients through the Stop/Expand option gate.
        """
        self._check_t(t)

        active = self._as_bool(active, shape=(self.B,), name="active")
        target = self._as_float(target, shape=(self.B,), name="stop_adv.target")
        valid_mask = self._as_bool(
            valid_mask,
            shape=(self.B,),
            name="stop_adv.valid_mask",
        )
        continue_log_reward = self._as_float(
            continue_log_reward,
            shape=(self.B,),
            name="stop_adv.continue_log_reward",
        )

        write_mask = active & valid_mask
        self.stop_adv_target[write_mask, t] = target[write_mask]
        self.stop_adv_valid_mask[write_mask, t] = True
        self.stop_adv_continue_log_reward[write_mask, t] = continue_log_reward[
            write_mask
        ]

    def finalize_unfinished(
        self,
        *,
        t: int,
        terminal: TerminalRewardOutput,
    ) -> None:
        """
        Mark still-active trajectories as terminal at the current step.

        Use this only if the rollout loop exits because the horizon/budget was
        reached without an explicit STOP action. The caller must ensure the
        corresponding step fields were already written for t.
        """
        self._check_t(t)

        unfinished = self.active
        if not bool(unfinished.any()):
            return

        self.is_terminated[unfinished] = True
        self.traj_len[unfinished] = int(t) + 1

        self.terminal_log_reward[unfinished] = self._as_float(
            terminal.log_reward,
            shape=(self.B,),
            name="terminal.log_reward",
        )[unfinished]
        self.terminal_answer_f1[unfinished] = self._as_float(
            terminal.answer_f1,
            shape=(self.B,),
            name="terminal.answer_f1",
        )[unfinished]
        self.terminal_complexity_penalty[unfinished] = self._as_float(
            terminal.complexity_penalty,
            shape=(self.B,),
            name="terminal.complexity_penalty",
        )[unfinished]
        self.terminal_base_log_reward[unfinished] = self._as_float(
            terminal.base_log_reward,
            shape=(self.B,),
            name="terminal.base_log_reward",
        )[unfinished]
        self.terminal_utility[unfinished] = self._as_float(
            terminal.utility,
            shape=(self.B,),
            name="terminal.utility",
        )[unfinished]
        self.terminal_expanded_edge_count[unfinished] = self._as_float(
            terminal.expanded_edge_count,
            shape=(self.B,),
            name="terminal.expanded_edge_count",
        )[unfinished]
        self.terminal_answer_degree_excess[unfinished] = self._as_float(
            terminal.answer_degree_excess,
            shape=(self.B,),
            name="terminal.answer_degree_excess",
        )[unfinished]

    def _write_terminal_from_step(
        self,
        *,
        stopped: torch.Tensor,
        result: StepResult,
    ) -> None:
        fields = (
            ("terminal_log_reward", result.terminal_log_reward),
            ("terminal_answer_f1", result.terminal_answer_f1),
            ("terminal_complexity_penalty", result.terminal_complexity_penalty),
            ("terminal_base_log_reward", result.terminal_base_log_reward),
            ("terminal_utility", result.terminal_utility),
            ("terminal_expanded_edge_count", result.terminal_expanded_edge_count),
            ("terminal_answer_degree_excess", result.terminal_answer_degree_excess),
        )

        for name, source in fields:
            target = getattr(self, name)
            value = self._as_float(source, shape=(self.B,), name=name)
            target[stopped] = value[stopped]

    def _check_t(self, t: int) -> None:
        if not 0 <= int(t) < self.T:
            raise IndexError(f"t must be in [0, {self.T}), got {t}.")

    def _as_float(
        self,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _as_long(
        self,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=torch.long)
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _as_bool(
        self,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=torch.bool)
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _zeros_float(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _zeros_long(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=self.device)

    def _zeros_bool(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bool, device=self.device)


__all__ = ["RolloutBuffer"]
