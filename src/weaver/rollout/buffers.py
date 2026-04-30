from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.weaver.reward import TerminalRewardOutput

from .schema import StepResult


@dataclass
class RolloutBuffer:
    """
    Mutable storage for one batched rollout.

    Shapes:
        B: number of physical graphs
        T: maximum number of decision steps

    Action type convention:
        0 = Continue / Expand one edge
        1 = Stop

    Stored trajectory convention:
        state_log_flows[b, t] = log F(s_t | q)
        step_log_pf[b, t]     = log P_F(a_t | s_t)
        step_log_pb[b, t]     = log P_B(s_t | s_{t+1})
        traj_len[b]           = number of executed transitions, including Stop
    """

    B: int
    T: int
    device: torch.device

    is_terminated: torch.Tensor = field(init=False)  # [B]
    traj_len: torch.Tensor = field(init=False)  # [B]

    terminal_log_reward: torch.Tensor = field(init=False)  # [B]
    terminal_answer_f1: torch.Tensor = field(init=False)  # [B]
    terminal_edge_penalty: torch.Tensor = field(init=False)  # [B]
    terminal_base_log_reward: torch.Tensor = field(init=False)  # [B]
    terminal_utility: torch.Tensor = field(init=False)  # [B]
    terminal_expanded_edge_count: torch.Tensor = field(init=False)  # [B]
    terminal_minimal_edge_count: torch.Tensor = field(init=False)  # [B]
    terminal_minimality_gap: torch.Tensor = field(init=False)  # [B]
    terminal_minimality_penalty: torch.Tensor = field(init=False)  # [B]

    state_log_flows: torch.Tensor = field(init=False)  # [B, T]
    step_log_pf: torch.Tensor = field(init=False)  # [B, T]
    step_log_pb: torch.Tensor = field(init=False)  # [B, T]

    action_type: torch.Tensor = field(init=False)  # [B, T]
    selected_edge_ids: torch.Tensor = field(init=False)  # [B, T]

    continue_mask: torch.Tensor = field(init=False)  # [B, T]
    stop_mask: torch.Tensor = field(init=False)  # [B, T]

    stop_now_log_reward: torch.Tensor = field(init=False)  # [B, T]
    stop_now_answer_f1: torch.Tensor = field(init=False)  # [B, T]
    stop_now_valid_mask: torch.Tensor = field(init=False)  # [B, T]

    stop_log_pf: torch.Tensor = field(init=False)  # [B, T]
    stop_tb_valid_mask: torch.Tensor = field(init=False)  # [B, T]

    target_continue_prob: torch.Tensor = field(init=False)  # [B, T]
    target_stop_prob: torch.Tensor = field(init=False)  # [B, T]
    policy_action_valid_mask: torch.Tensor = field(init=False)  # [B, T]

    edge_action_entropy: torch.Tensor = field(init=False)  # [B, T]
    edge_action_entropy_valid_mask: torch.Tensor = field(init=False)  # [B, T]

    advantage_aux_loss: torch.Tensor = field(init=False)  # [B, T]
    advantage_aux_valid_mask: torch.Tensor = field(init=False)  # [B, T]

    budget_exhausted_mask: torch.Tensor = field(init=False)  # [B, T]
    proposal_intervention_mask: torch.Tensor = field(init=False)  # [B, T]

    def __post_init__(self) -> None:
        self.B = int(self.B)
        self.T = int(self.T)

        if self.B <= 0:
            raise ValueError(f"B must be positive, got {self.B}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")

        b = (self.B,)
        bt = (self.B, self.T)

        self.is_terminated = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.traj_len = torch.zeros(b, dtype=torch.long, device=self.device)

        self.terminal_log_reward = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_answer_f1 = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_edge_penalty = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_base_log_reward = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_utility = torch.zeros(b, dtype=torch.float32, device=self.device)
        self.terminal_expanded_edge_count = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_minimal_edge_count = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_minimality_gap = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )
        self.terminal_minimality_penalty = torch.zeros(
            b, dtype=torch.float32, device=self.device
        )

        self.state_log_flows = torch.zeros(bt, dtype=torch.float32, device=self.device)
        self.step_log_pf = torch.zeros(bt, dtype=torch.float32, device=self.device)
        self.step_log_pb = torch.zeros(bt, dtype=torch.float32, device=self.device)

        self.action_type = torch.full(bt, -1, dtype=torch.long, device=self.device)
        self.selected_edge_ids = torch.full(
            bt, -1, dtype=torch.long, device=self.device
        )

        self.continue_mask = torch.zeros(bt, dtype=torch.bool, device=self.device)
        self.stop_mask = torch.zeros(bt, dtype=torch.bool, device=self.device)

        self.stop_now_log_reward = torch.zeros(
            bt, dtype=torch.float32, device=self.device
        )
        self.stop_now_answer_f1 = torch.zeros(
            bt, dtype=torch.float32, device=self.device
        )
        self.stop_now_valid_mask = torch.zeros(bt, dtype=torch.bool, device=self.device)

        self.stop_log_pf = torch.zeros(bt, dtype=torch.float32, device=self.device)
        self.stop_tb_valid_mask = torch.zeros(bt, dtype=torch.bool, device=self.device)

        self.target_continue_prob = torch.zeros(
            bt, dtype=torch.float32, device=self.device
        )
        self.target_stop_prob = torch.zeros(bt, dtype=torch.float32, device=self.device)
        self.policy_action_valid_mask = torch.zeros(
            bt, dtype=torch.bool, device=self.device
        )

        self.edge_action_entropy = torch.zeros(
            bt, dtype=torch.float32, device=self.device
        )
        self.edge_action_entropy_valid_mask = torch.zeros(
            bt, dtype=torch.bool, device=self.device
        )

        self.advantage_aux_loss = torch.zeros(
            bt, dtype=torch.float32, device=self.device
        )
        self.advantage_aux_valid_mask = torch.zeros(
            bt, dtype=torch.bool, device=self.device
        )

        self.budget_exhausted_mask = torch.zeros(
            bt, dtype=torch.bool, device=self.device
        )
        self.proposal_intervention_mask = torch.zeros(
            bt, dtype=torch.bool, device=self.device
        )

    def write_state_log_flow(
        self,
        *,
        t: int,
        active: torch.Tensor,
        state_log_flow: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = active.to(device=self.device, dtype=torch.bool)

        flow = state_log_flow.to(device=self.device, dtype=torch.float32)
        self.state_log_flows[active, t] = flow[active]

    def write_step(self, t: int, result: StepResult) -> None:
        self._check_t(t)

        active = ~self.is_terminated

        log_pf = result.log_pf.to(device=self.device, dtype=torch.float32)
        log_pb = result.log_pb.to(device=self.device, dtype=torch.float32)
        action_type = result.action_type.to(device=self.device, dtype=torch.long)
        selected_edge_ids = result.selected_edge_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        continue_mask = result.continue_mask.to(device=self.device, dtype=torch.bool)
        stop_mask = result.stop_mask.to(device=self.device, dtype=torch.bool)
        intervention = result.proposal_intervention_mask.to(
            device=self.device,
            dtype=torch.bool,
        )

        self.step_log_pf[active, t] = log_pf[active]
        self.step_log_pb[active, t] = log_pb[active]
        self.action_type[active, t] = action_type[active]
        self.selected_edge_ids[active, t] = selected_edge_ids[active]
        self.continue_mask[active, t] = continue_mask[active]
        self.stop_mask[active, t] = stop_mask[active]
        self.proposal_intervention_mask[active, t] = intervention[active]

        stopped = active & stop_mask

        self.is_terminated[stopped] = True
        self.traj_len[stopped] = int(t) + 1

        terminal_log_reward = result.terminal_log_reward.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_answer_f1 = result.terminal_answer_f1.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_edge_penalty = result.terminal_edge_penalty.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_base_log_reward = result.terminal_base_log_reward.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_utility = result.terminal_utility.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_expanded_edge_count = result.terminal_expanded_edge_count.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_minimal_edge_count = result.terminal_minimal_edge_count.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_minimality_gap = result.terminal_minimality_gap.to(
            device=self.device,
            dtype=torch.float32,
        )
        terminal_minimality_penalty = result.terminal_minimality_penalty.to(
            device=self.device,
            dtype=torch.float32,
        )

        self.terminal_log_reward[stopped] = terminal_log_reward[stopped]
        self.terminal_answer_f1[stopped] = terminal_answer_f1[stopped]
        self.terminal_edge_penalty[stopped] = terminal_edge_penalty[stopped]
        self.terminal_base_log_reward[stopped] = terminal_base_log_reward[stopped]
        self.terminal_utility[stopped] = terminal_utility[stopped]
        self.terminal_expanded_edge_count[stopped] = terminal_expanded_edge_count[
            stopped
        ]
        self.terminal_minimal_edge_count[stopped] = terminal_minimal_edge_count[stopped]
        self.terminal_minimality_gap[stopped] = terminal_minimality_gap[stopped]
        self.terminal_minimality_penalty[stopped] = terminal_minimality_penalty[stopped]

    def write_stop_counterfactual(
        self,
        *,
        t: int,
        active: torch.Tensor,
        reward_output: TerminalRewardOutput,
    ) -> None:
        self._check_t(t)

        active = active.to(device=self.device, dtype=torch.bool)

        log_reward = reward_output.log_reward.to(
            device=self.device,
            dtype=torch.float32,
        )
        answer_f1 = reward_output.answer_f1.to(
            device=self.device,
            dtype=torch.float32,
        )

        self.stop_now_log_reward[active, t] = log_reward[active]
        self.stop_now_answer_f1[active, t] = answer_f1[active]
        self.stop_now_valid_mask[active, t] = True

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

        active = active.to(device=self.device, dtype=torch.bool)

        target_stop_prob = target_stop_prob.to(
            device=self.device,
            dtype=torch.float32,
        )
        target_continue_prob = target_continue_prob.to(
            device=self.device,
            dtype=torch.float32,
        )
        stop_log_pf = stop_log_pf.to(
            device=self.device,
            dtype=torch.float32,
        )
        action_valid_mask = action_valid_mask.to(
            device=self.device,
            dtype=torch.bool,
        )
        stop_tb_valid_mask = stop_tb_valid_mask.to(
            device=self.device,
            dtype=torch.bool,
        )
        edge_action_entropy = edge_action_entropy.to(
            device=self.device,
            dtype=torch.float32,
        )
        edge_entropy_valid = edge_action_entropy_valid_mask.to(
            device=self.device,
            dtype=torch.bool,
        )
        budget_exhausted = budget_exhausted_mask.to(
            device=self.device,
            dtype=torch.bool,
        )

        self.target_stop_prob[active, t] = target_stop_prob[active]
        self.target_continue_prob[active, t] = target_continue_prob[active]
        self.stop_log_pf[active, t] = stop_log_pf[active]
        self.policy_action_valid_mask[active, t] = action_valid_mask[active]
        self.stop_tb_valid_mask[active, t] = stop_tb_valid_mask[active]
        self.edge_action_entropy[active, t] = edge_action_entropy[active]
        self.edge_action_entropy_valid_mask[active, t] = edge_entropy_valid[active]
        self.budget_exhausted_mask[active, t] = budget_exhausted[active]

    def write_advantage_auxiliary(
        self,
        *,
        t: int,
        active: torch.Tensor,
        advantage_loss: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = active.to(device=self.device, dtype=torch.bool)
        advantage_loss = advantage_loss.to(device=self.device, dtype=torch.float32)
        valid_mask = valid_mask.to(device=self.device, dtype=torch.bool)

        self.advantage_aux_loss[active, t] = advantage_loss[active]
        self.advantage_aux_valid_mask[active, t] = valid_mask[active]

    def _check_t(self, t: int) -> None:
        if not 0 <= int(t) < self.T:
            raise IndexError(f"t must be in [0, {self.T}), got {t}.")


__all__ = ["RolloutBuffer"]
