from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.weaver.reward import TerminalRewardOutput

from .schema import RolloutTraceSpec, StepResult


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
        step_log_pf[b, t]     = log P_F(a_t | s_t)
        step_log_pb[b, t]     = log P_B(s_t | s_{t+1})
        state_log_flow[b, t]  = log F_theta(s_t)
        db_*[b, t]            = legacy DB/DAGDB-shaped diagnostic trace, not
                                the active SubTB training objective
        traj_len[b]           = number of executed transitions, including Stop
    """

    B: int
    T: int
    device: torch.device
    trace_spec: RolloutTraceSpec = field(default_factory=RolloutTraceSpec)

    is_terminated: torch.Tensor = field(init=False)
    traj_len: torch.Tensor = field(init=False)

    terminal_log_reward: torch.Tensor = field(init=False)
    terminal_answer_f1: torch.Tensor = field(init=False)
    terminal_complexity_penalty: torch.Tensor = field(init=False)
    terminal_base_log_reward: torch.Tensor = field(init=False)
    terminal_utility: torch.Tensor = field(init=False)
    terminal_shortest_path_potential: torch.Tensor = field(init=False)
    terminal_expanded_edge_count: torch.Tensor = field(init=False)
    terminal_answer_degree_excess: torch.Tensor = field(init=False)

    step_log_pf: torch.Tensor = field(init=False)
    step_log_pb: torch.Tensor = field(init=False)
    state_log_flow: torch.Tensor = field(init=False)
    log_p_stop: torch.Tensor = field(init=False)

    db_parent_log_reward: torch.Tensor = field(init=False)
    db_child_log_reward: torch.Tensor = field(init=False)
    db_parent_shortest_path_potential: torch.Tensor = field(init=False)
    db_child_shortest_path_potential: torch.Tensor = field(init=False)
    db_parent_process_log_bonus: torch.Tensor = field(init=False)
    db_child_process_log_bonus: torch.Tensor = field(init=False)
    db_log_p_stop_parent: torch.Tensor = field(init=False)
    db_log_p_stop_child: torch.Tensor = field(init=False)
    db_log_pf_expand: torch.Tensor = field(init=False)
    db_log_pb: torch.Tensor = field(init=False)
    db_valid_mask: torch.Tensor = field(init=False)

    action_type: torch.Tensor = field(init=False)
    selected_edge_ids: torch.Tensor = field(init=False)

    continue_mask: torch.Tensor = field(init=False)
    stop_mask: torch.Tensor = field(init=False)

    stop_now_log_reward: torch.Tensor | None = field(init=False)
    stop_now_answer_f1: torch.Tensor | None = field(init=False)
    stop_now_valid_mask: torch.Tensor | None = field(init=False)

    target_continue_prob: torch.Tensor = field(init=False)
    target_stop_prob: torch.Tensor = field(init=False)
    policy_action_valid_mask: torch.Tensor = field(init=False)

    edge_action_entropy: torch.Tensor = field(init=False)
    edge_action_entropy_valid_mask: torch.Tensor = field(init=False)
    budget_exhausted_mask: torch.Tensor = field(init=False)

    te_bfm_loss: torch.Tensor | None = field(init=False)
    te_bfm_valid_mask: torch.Tensor | None = field(init=False)
    te_bfm_residual_abs: torch.Tensor | None = field(init=False)
    te_bfm_target_log_value: torch.Tensor | None = field(init=False)
    te_bfm_log_reward: torch.Tensor | None = field(init=False)
    te_bfm_stop_prob: torch.Tensor | None = field(init=False)
    te_bfm_frontier_edge_count: torch.Tensor | None = field(init=False)
    te_bfm_counterfactual_child_loss: torch.Tensor | None = field(init=False)
    te_bfm_frontier_cap_used: torch.Tensor | None = field(init=False)
    te_bfm_frontier_cap_dropped_edge_count: torch.Tensor | None = field(init=False)
    bdb_stop_loss: torch.Tensor | None = field(init=False)
    bdb_edge_loss: torch.Tensor | None = field(init=False)
    bdb_base_loss: torch.Tensor | None = field(init=False)
    bdb_stop_valid_mask: torch.Tensor | None = field(init=False)
    bdb_edge_valid_mask: torch.Tensor | None = field(init=False)
    bdb_base_valid_mask: torch.Tensor | None = field(init=False)
    bdb_delta_stop: torch.Tensor | None = field(init=False)
    bdb_delta_edge: torch.Tensor | None = field(init=False)
    bdb_delta_base: torch.Tensor | None = field(init=False)
    bdb_frontier_size: torch.Tensor | None = field(init=False)
    bdb_parent_count: torch.Tensor | None = field(init=False)
    bdb_log_reward: torch.Tensor | None = field(init=False)
    bdb_log_flow: torch.Tensor | None = field(init=False)

    def __post_init__(self) -> None:
        self.B = int(self.B)
        self.T = int(self.T)

        if self.B <= 0:
            raise ValueError(f"B must be positive, got {self.B}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if not isinstance(self.trace_spec, RolloutTraceSpec):
            raise TypeError(
                "trace_spec must be a RolloutTraceSpec, "
                f"got {type(self.trace_spec).__name__}."
            )

        b = (self.B,)
        bt = (self.B, self.T)

        self.is_terminated = self._zeros_bool(b)
        self.traj_len = self._zeros_long(b)

        self.terminal_log_reward = self._zeros_float(b)
        self.terminal_answer_f1 = self._zeros_float(b)
        self.terminal_complexity_penalty = self._zeros_float(b)
        self.terminal_base_log_reward = self._zeros_float(b)
        self.terminal_utility = self._zeros_float(b)
        self.terminal_shortest_path_potential = self._zeros_float(b)
        self.terminal_expanded_edge_count = self._zeros_float(b)
        self.terminal_answer_degree_excess = self._zeros_float(b)

        self.step_log_pf = self._zeros_float(bt)
        self.step_log_pb = self._zeros_float(bt)
        self.state_log_flow = self._zeros_float(bt)
        self.log_p_stop = self._zeros_float(bt)
        self.db_parent_log_reward = self._zeros_float(bt)
        self.db_child_log_reward = self._zeros_float(bt)
        self.db_parent_shortest_path_potential = self._zeros_float(bt)
        self.db_child_shortest_path_potential = self._zeros_float(bt)
        self.db_parent_process_log_bonus = self._zeros_float(bt)
        self.db_child_process_log_bonus = self._zeros_float(bt)
        self.db_log_p_stop_parent = self._zeros_float(bt)
        self.db_log_p_stop_child = self._zeros_float(bt)
        self.db_log_pf_expand = self._zeros_float(bt)
        self.db_log_pb = self._zeros_float(bt)
        self.db_valid_mask = self._zeros_bool(bt)

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

        if self.trace_spec.store_stop_now_reward:
            self.stop_now_log_reward = self._zeros_float(bt)
            self.stop_now_answer_f1 = self._zeros_float(bt)
            self.stop_now_valid_mask = self._zeros_bool(bt)
        else:
            self.stop_now_log_reward = None
            self.stop_now_answer_f1 = None
            self.stop_now_valid_mask = None

        self.target_continue_prob = self._zeros_float(bt)
        self.target_stop_prob = self._zeros_float(bt)
        self.policy_action_valid_mask = self._zeros_bool(bt)

        self.edge_action_entropy = self._zeros_float(bt)
        self.edge_action_entropy_valid_mask = self._zeros_bool(bt)
        self.budget_exhausted_mask = self._zeros_bool(bt)

        if self.trace_spec.store_te_bfm:
            self.te_bfm_loss = self._zeros_float(bt)
            self.te_bfm_valid_mask = self._zeros_bool(bt)
            self.te_bfm_residual_abs = self._zeros_float(bt)
            self.te_bfm_target_log_value = self._zeros_float(bt)
            self.te_bfm_log_reward = self._zeros_float(bt)
            self.te_bfm_stop_prob = self._zeros_float(bt)
            self.te_bfm_frontier_edge_count = self._zeros_float(bt)
            self.te_bfm_counterfactual_child_loss = self._zeros_float(bt)
            self.te_bfm_frontier_cap_used = self._zeros_float(bt)
            self.te_bfm_frontier_cap_dropped_edge_count = self._zeros_float(bt)
        else:
            self.te_bfm_loss = None
            self.te_bfm_valid_mask = None
            self.te_bfm_residual_abs = None
            self.te_bfm_target_log_value = None
            self.te_bfm_log_reward = None
            self.te_bfm_stop_prob = None
            self.te_bfm_frontier_edge_count = None
            self.te_bfm_counterfactual_child_loss = None
            self.te_bfm_frontier_cap_used = None
            self.te_bfm_frontier_cap_dropped_edge_count = None

        if self.trace_spec.store_bdb:
            self.bdb_stop_loss = self._zeros_float(bt)
            self.bdb_edge_loss = self._zeros_float(bt)
            self.bdb_base_loss = self._zeros_float(bt)
            self.bdb_stop_valid_mask = self._zeros_bool(bt)
            self.bdb_edge_valid_mask = self._zeros_bool(bt)
            self.bdb_base_valid_mask = self._zeros_bool(bt)
            self.bdb_delta_stop = self._zeros_float(bt)
            self.bdb_delta_edge = self._zeros_float(bt)
            self.bdb_delta_base = self._zeros_float(bt)
            self.bdb_frontier_size = self._zeros_float(bt)
            self.bdb_parent_count = self._zeros_float(bt)
            self.bdb_log_reward = self._zeros_float(bt)
            self.bdb_log_flow = self._zeros_float(bt)
        else:
            self.bdb_stop_loss = None
            self.bdb_edge_loss = None
            self.bdb_base_loss = None
            self.bdb_stop_valid_mask = None
            self.bdb_edge_valid_mask = None
            self.bdb_base_valid_mask = None
            self.bdb_delta_stop = None
            self.bdb_delta_edge = None
            self.bdb_delta_base = None
            self.bdb_frontier_size = None
            self.bdb_parent_count = None
            self.bdb_log_reward = None
            self.bdb_log_flow = None

    @property
    def active(self) -> torch.Tensor:
        return ~self.is_terminated

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

    def write_state_log_flow(
        self,
        *,
        t: int,
        active: torch.Tensor,
        state_log_flow: torch.Tensor,
    ) -> None:
        self._check_t(t)
        active = self._as_bool(active, shape=(self.B,), name="active")
        state_log_flow = self._as_float(
            state_log_flow,
            shape=(self.B,),
            name="state_log_flow",
        )
        self.state_log_flow[active, t] = state_log_flow[active]

    def write_log_p_stop(
        self,
        *,
        t: int,
        active: torch.Tensor,
        log_p_stop: torch.Tensor,
    ) -> None:
        self._check_t(t)
        active = self._as_bool(active, shape=(self.B,), name="active")
        log_p_stop = self._as_float(
            log_p_stop,
            shape=(self.B,),
            name="log_p_stop",
        )
        self.log_p_stop[active, t] = log_p_stop[active]

    def write_dag_db_transition(
        self,
        *,
        t: int,
        valid_mask: torch.Tensor,
        parent_log_reward: torch.Tensor,
        child_log_reward: torch.Tensor,
        parent_shortest_path_potential: torch.Tensor,
        child_shortest_path_potential: torch.Tensor,
        parent_process_log_bonus: torch.Tensor,
        child_process_log_bonus: torch.Tensor,
        log_p_stop_parent: torch.Tensor,
        log_p_stop_child: torch.Tensor,
        log_pf_expand: torch.Tensor,
        log_pb: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = self.active
        valid = self._as_bool(valid_mask, shape=(self.B,), name="db_valid_mask")
        mask = active & valid
        if not bool(mask.any()):
            return

        parent_log_reward = self._as_float(
            parent_log_reward,
            shape=(self.B,),
            name="db_parent_log_reward",
        )
        child_log_reward = self._as_float(
            child_log_reward,
            shape=(self.B,),
            name="db_child_log_reward",
        )
        parent_shortest_path_potential = self._as_float(
            parent_shortest_path_potential,
            shape=(self.B,),
            name="db_parent_shortest_path_potential",
        )
        child_shortest_path_potential = self._as_float(
            child_shortest_path_potential,
            shape=(self.B,),
            name="db_child_shortest_path_potential",
        )
        parent_process_log_bonus = self._as_float(
            parent_process_log_bonus,
            shape=(self.B,),
            name="db_parent_process_log_bonus",
        )
        child_process_log_bonus = self._as_float(
            child_process_log_bonus,
            shape=(self.B,),
            name="db_child_process_log_bonus",
        )
        log_p_stop_parent = self._as_float(
            log_p_stop_parent,
            shape=(self.B,),
            name="db_log_p_stop_parent",
        )
        log_p_stop_child = self._as_float(
            log_p_stop_child,
            shape=(self.B,),
            name="db_log_p_stop_child",
        )
        log_pf_expand = self._as_float(
            log_pf_expand,
            shape=(self.B,),
            name="db_log_pf_expand",
        )
        log_pb = self._as_float(log_pb, shape=(self.B,), name="db_log_pb")

        self.db_parent_log_reward[mask, t] = parent_log_reward[mask]
        self.db_child_log_reward[mask, t] = child_log_reward[mask]
        self.db_parent_shortest_path_potential[mask, t] = parent_shortest_path_potential[mask]
        self.db_child_shortest_path_potential[mask, t] = child_shortest_path_potential[mask]
        self.db_parent_process_log_bonus[mask, t] = parent_process_log_bonus[mask]
        self.db_child_process_log_bonus[mask, t] = child_process_log_bonus[mask]
        self.db_log_p_stop_parent[mask, t] = log_p_stop_parent[mask]
        self.db_log_p_stop_child[mask, t] = log_p_stop_child[mask]
        self.db_log_pf_expand[mask, t] = log_pf_expand[mask]
        self.db_log_pb[mask, t] = log_pb[mask]
        self.db_valid_mask[mask, t] = True

    def write_stop_now_reward(
        self,
        *,
        t: int,
        active: torch.Tensor,
        reward_output: TerminalRewardOutput,
    ) -> None:
        self._check_t(t)
        self._require_stop_now_storage()

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

        assert self.stop_now_log_reward is not None
        assert self.stop_now_answer_f1 is not None
        assert self.stop_now_valid_mask is not None
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
        action_valid_mask: torch.Tensor,
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
        action_valid_mask = self._as_bool(
            action_valid_mask,
            shape=(self.B,),
            name="action_valid_mask",
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
        self.policy_action_valid_mask[active, t] = action_valid_mask[active]
        self.edge_action_entropy[active, t] = edge_entropy[active]
        self.edge_action_entropy_valid_mask[active, t] = edge_entropy_valid[active]
        self.budget_exhausted_mask[active, t] = budget_exhausted[active]

    def write_stop_decision_context(
        self,
        *,
        t: int,
        active: torch.Tensor,
        action_valid_mask: torch.Tensor,
        budget_exhausted_mask: torch.Tensor,
    ) -> None:
        self._check_t(t)

        active = self._as_bool(active, shape=(self.B,), name="active")
        action_valid = self._as_bool(
            action_valid_mask,
            shape=(self.B,),
            name="action_valid_mask",
        )
        budget_exhausted = self._as_bool(
            budget_exhausted_mask,
            shape=(self.B,),
            name="budget_exhausted_mask",
        )

        self.policy_action_valid_mask[active, t] = action_valid[active]
        self.budget_exhausted_mask[active, t] = budget_exhausted[active]

    def write_te_bfm(
        self,
        *,
        t: int,
        active: torch.Tensor,
        loss: torch.Tensor,
        valid_mask: torch.Tensor,
        residual_abs: torch.Tensor,
        target_log_value: torch.Tensor,
        log_reward: torch.Tensor,
        stop_prob: torch.Tensor,
        frontier_edge_count: torch.Tensor,
        counterfactual_child_loss: torch.Tensor,
        frontier_cap_used: torch.Tensor,
        frontier_cap_dropped_edge_count: torch.Tensor,
    ) -> None:
        self._check_t(t)
        self._require_te_bfm_storage()

        active = self._as_bool(active, shape=(self.B,), name="active")
        valid = self._as_bool(valid_mask, shape=(self.B,), name="te_bfm_valid_mask")
        mask = active & valid
        if not bool(mask.any()):
            return

        loss = self._as_float(loss, shape=(self.B,), name="te_bfm_loss")
        residual_abs = self._as_float(
            residual_abs,
            shape=(self.B,),
            name="te_bfm_residual_abs",
        )
        target_log_value = self._as_float(
            target_log_value,
            shape=(self.B,),
            name="te_bfm_target_log_value",
        )
        log_reward = self._as_float(
            log_reward,
            shape=(self.B,),
            name="te_bfm_log_reward",
        )
        stop_prob = self._as_float(
            stop_prob,
            shape=(self.B,),
            name="te_bfm_stop_prob",
        )
        frontier_edge_count = self._as_float(
            frontier_edge_count,
            shape=(self.B,),
            name="te_bfm_frontier_edge_count",
        )
        counterfactual_child_loss = self._as_float(
            counterfactual_child_loss,
            shape=(self.B,),
            name="te_bfm_counterfactual_child_loss",
        )
        frontier_cap_used = self._as_float(
            frontier_cap_used,
            shape=(self.B,),
            name="te_bfm_frontier_cap_used",
        )
        frontier_cap_dropped_edge_count = self._as_float(
            frontier_cap_dropped_edge_count,
            shape=(self.B,),
            name="te_bfm_frontier_cap_dropped_edge_count",
        )

        assert self.te_bfm_loss is not None
        assert self.te_bfm_valid_mask is not None
        assert self.te_bfm_residual_abs is not None
        assert self.te_bfm_target_log_value is not None
        assert self.te_bfm_log_reward is not None
        assert self.te_bfm_stop_prob is not None
        assert self.te_bfm_frontier_edge_count is not None
        assert self.te_bfm_counterfactual_child_loss is not None
        assert self.te_bfm_frontier_cap_used is not None
        assert self.te_bfm_frontier_cap_dropped_edge_count is not None
        self.te_bfm_loss[mask, t] = loss[mask]
        self.te_bfm_valid_mask[mask, t] = True
        self.te_bfm_residual_abs[mask, t] = residual_abs[mask]
        self.te_bfm_target_log_value[mask, t] = target_log_value[mask]
        self.te_bfm_log_reward[mask, t] = log_reward[mask]
        self.te_bfm_stop_prob[mask, t] = stop_prob[mask]
        self.te_bfm_frontier_edge_count[mask, t] = frontier_edge_count[mask]
        self.te_bfm_counterfactual_child_loss[mask, t] = (
            counterfactual_child_loss[mask]
        )
        self.te_bfm_frontier_cap_used[mask, t] = frontier_cap_used[mask]
        self.te_bfm_frontier_cap_dropped_edge_count[mask, t] = (
            frontier_cap_dropped_edge_count[mask]
        )

    def write_bdb(
        self,
        *,
        t: int,
        active: torch.Tensor,
        stop_loss: torch.Tensor,
        edge_loss: torch.Tensor,
        base_loss: torch.Tensor,
        stop_valid_mask: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        base_valid_mask: torch.Tensor,
        delta_stop: torch.Tensor,
        delta_edge: torch.Tensor,
        delta_base: torch.Tensor,
        frontier_size: torch.Tensor,
        parent_count: torch.Tensor,
        log_reward: torch.Tensor,
        log_flow: torch.Tensor,
    ) -> None:
        self._check_t(t)
        self._require_bdb_storage()

        active = self._as_bool(active, shape=(self.B,), name="active")
        assert self.bdb_stop_loss is not None
        assert self.bdb_edge_loss is not None
        assert self.bdb_base_loss is not None
        assert self.bdb_stop_valid_mask is not None
        assert self.bdb_edge_valid_mask is not None
        assert self.bdb_base_valid_mask is not None
        assert self.bdb_delta_stop is not None
        assert self.bdb_delta_edge is not None
        assert self.bdb_delta_base is not None
        assert self.bdb_frontier_size is not None
        assert self.bdb_parent_count is not None
        assert self.bdb_log_reward is not None
        assert self.bdb_log_flow is not None

        self.bdb_stop_loss[active, t] = self._as_float(
            stop_loss, shape=(self.B,), name="bdb_stop_loss"
        )[active]
        self.bdb_edge_loss[active, t] = self._as_float(
            edge_loss, shape=(self.B,), name="bdb_edge_loss"
        )[active]
        self.bdb_base_loss[active, t] = self._as_float(
            base_loss, shape=(self.B,), name="bdb_base_loss"
        )[active]
        self.bdb_stop_valid_mask[active, t] = self._as_bool(
            stop_valid_mask, shape=(self.B,), name="bdb_stop_valid_mask"
        )[active]
        self.bdb_edge_valid_mask[active, t] = self._as_bool(
            edge_valid_mask, shape=(self.B,), name="bdb_edge_valid_mask"
        )[active]
        self.bdb_base_valid_mask[active, t] = self._as_bool(
            base_valid_mask, shape=(self.B,), name="bdb_base_valid_mask"
        )[active]
        self.bdb_delta_stop[active, t] = self._as_float(
            delta_stop, shape=(self.B,), name="bdb_delta_stop"
        )[active]
        self.bdb_delta_edge[active, t] = self._as_float(
            delta_edge, shape=(self.B,), name="bdb_delta_edge"
        )[active]
        self.bdb_delta_base[active, t] = self._as_float(
            delta_base, shape=(self.B,), name="bdb_delta_base"
        )[active]
        self.bdb_frontier_size[active, t] = self._as_float(
            frontier_size, shape=(self.B,), name="bdb_frontier_size"
        )[active]
        self.bdb_parent_count[active, t] = self._as_float(
            parent_count, shape=(self.B,), name="bdb_parent_count"
        )[active]
        self.bdb_log_reward[active, t] = self._as_float(
            log_reward, shape=(self.B,), name="bdb_log_reward"
        )[active]
        self.bdb_log_flow[active, t] = self._as_float(
            log_flow, shape=(self.B,), name="bdb_log_flow"
        )[active]

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
        self.terminal_shortest_path_potential[unfinished] = self._as_float(
            terminal.shortest_path_potential,
            shape=(self.B,),
            name="terminal.shortest_path_potential",
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
            (
                "terminal_shortest_path_potential",
                result.terminal_shortest_path_potential,
            ),
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

    def _require_stop_now_storage(self) -> None:
        if (
            self.stop_now_log_reward is None
            or self.stop_now_answer_f1 is None
            or self.stop_now_valid_mask is None
        ):
            raise RuntimeError(
                "RolloutBuffer was created without stop-now trace storage."
            )

    def _require_te_bfm_storage(self) -> None:
        if (
            self.te_bfm_loss is None
            or self.te_bfm_valid_mask is None
            or self.te_bfm_residual_abs is None
            or self.te_bfm_target_log_value is None
            or self.te_bfm_log_reward is None
            or self.te_bfm_stop_prob is None
            or self.te_bfm_frontier_edge_count is None
        ):
            raise RuntimeError(
                "RolloutBuffer was created without TE-BFM trace storage."
            )

    def _require_bdb_storage(self) -> None:
        if (
            self.bdb_stop_loss is None
            or self.bdb_edge_loss is None
            or self.bdb_base_loss is None
            or self.bdb_stop_valid_mask is None
            or self.bdb_edge_valid_mask is None
            or self.bdb_base_valid_mask is None
            or self.bdb_delta_stop is None
            or self.bdb_delta_edge is None
            or self.bdb_delta_base is None
            or self.bdb_frontier_size is None
            or self.bdb_parent_count is None
            or self.bdb_log_reward is None
            or self.bdb_log_flow is None
        ):
            raise RuntimeError("RolloutBuffer was created without BDB trace storage.")

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
