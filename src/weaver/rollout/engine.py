from __future__ import annotations

from dataclasses import dataclass
import torch

from src.data.schema import RetrievalBatch
from src.weaver.policy import Policy, PolicyContext, PolicyOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.state import RolloutState, State

from .batch_ops import split_static_rollout_batch
from .buffer import RolloutBuffer
from .diagnostics import write_policy_diagnostics
from .executor import (
    FusedStepExecutor,
    StepContext,
    budget_exhausted_mask,
    has_frontier,
)
from .sampling import sample_action_for_generation
from .schema import RolloutBatch, RolloutTraceSpec


@dataclass(frozen=True, slots=True)
class RolloutEngineConfig:
    temperature: float
    collect_policy_diagnostics: bool = False
    validate_synchronous_depth: bool = False
    trace_spec: RolloutTraceSpec = RolloutTraceSpec()


class RolloutEngine:
    """
    Finite-horizon vectorized rollout driver.

    Responsibilities:
        1. initialize canonical subgraph state;
        2. evaluate stop-now terminal reward eagerly only when diagnostics
           need it;
        3. query policy at each active state;
        4. write diagnostics into RolloutBuffer;
        5. execute one transition;
        6. write rollout traces used by SubTB and diagnostics;
        7. return RolloutBatch.

    It does not implement loss computation, target networks, or rollout-time
    external intervention.

    State invariant:
        V_s = anchors union endpoints(E_s)

    State stores V_s as a mutable mask for fast readout. Engine snapshots that
    mask before each policy call; Executor validates the invariant and mutates
    the live state for Expand transitions.
    """

    def __init__(self, expand_budget: int) -> None:
        self.expand_budget = int(expand_budget)
        if self.expand_budget < 0:
            raise ValueError(
                f"expand_budget must be non-negative, got {self.expand_budget}."
            )

    def run_vectorized(
        self,
        *,
        policy: Policy,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        num_rollouts: int,
        temperature: float = 1.0,
        collect_policy_diagnostics: bool = False,
        validate_synchronous_depth: bool = False,
        store_stop_now_reward: bool | None = None,
        store_te_bfm: bool | None = None,
        store_bdb: bool | None = None,
    ) -> list[RolloutBatch]:
        """
        Run K independent rollouts.

        The static RetrievalBatch and FeatureBank stay at size B. Dynamic
        rollout state is fused into R = K * B rows, then split back into K
        logical rollout batches.
        """
        num_rollouts = int(num_rollouts)
        if num_rollouts <= 0:
            return []

        store_stop_now = (
            bool(store_stop_now_reward)
            if store_stop_now_reward is not None
            else False
        )
        store_te_bfm_trace = bool(store_te_bfm) if store_te_bfm is not None else False
        store_bdb_trace = bool(store_bdb) if store_bdb is not None else False

        config = RolloutEngineConfig(
            temperature=float(temperature),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
            trace_spec=RolloutTraceSpec(
                store_stop_now_reward=store_stop_now,
                store_te_bfm=store_te_bfm_trace,
                store_bdb=store_bdb_trace,
            ),
        )

        rollout_context = policy.prepare_rollout_context(retrieval_batch)
        rollout = self._run_fused_static_batch(
            policy=policy,
            retrieval_batch=retrieval_batch,
            reward_model=reward_model,
            rollout_context=rollout_context,
            num_rollouts=num_rollouts,
            config=config,
        )

        return split_static_rollout_batch(
            rollout=rollout,
            original_batch=retrieval_batch,
            repeats=num_rollouts,
        )

    def _run_fused_static_batch(
        self,
        *,
        policy: Policy,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        rollout_context: PolicyContext,
        num_rollouts: int,
        config: RolloutEngineConfig,
    ) -> RolloutBatch:
        if config.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {config.temperature}.")

        device = retrieval_batch.edge_index.device
        static_graphs = int(retrieval_batch.num_graphs)
        dynamic_graphs = int(num_rollouts) * static_graphs

        rollout_to_graph = torch.arange(
            static_graphs,
            dtype=torch.long,
            device=device,
        ).repeat(int(num_rollouts))

        state = RolloutState.create_initial(
            retrieval_batch,
            expand_budget=self.expand_budget,
            rollout_to_graph=rollout_to_graph,
        )

        buffer = RolloutBuffer(
            B=dynamic_graphs,
            T=self.expand_budget + 1,
            device=device,
            trace_spec=config.trace_spec,
        )

        executor = FusedStepExecutor(
            retrieval_batch=retrieval_batch,
            reward_model=reward_model,
        )

        need_policy_step_traces = bool(config.collect_policy_diagnostics)

        for t in range(self.expand_budget + 1):
            active = ~buffer.is_terminated
            if not bool(active.any()):
                break
            active_rollout_ids = active.nonzero(as_tuple=False).view(-1)

            if config.validate_synchronous_depth:
                self._assert_synchronous_depth(
                    state=state,
                    retrieval_batch=retrieval_batch,
                    active=active,
                    t=t,
                    num_graphs=dynamic_graphs,
                )

            policy_state = state.select_rollouts(active_rollout_ids)

            active_remaining_budget = policy_state.remaining_budget_per_graph(
                edge_batch=retrieval_batch.edge_batch,
                num_graphs=int(active_rollout_ids.numel()),
            )
            remaining_budget = _scatter_active_rows(
                values=active_remaining_budget,
                active_rollout_ids=active_rollout_ids,
                num_graphs=dynamic_graphs,
                fill_value=0,
            )

            stop_now_reward = (
                _scatter_terminal_reward_output(
                    self._evaluate_stop_now(
                        reward_model=reward_model,
                        retrieval_batch=retrieval_batch,
                        state=policy_state,
                    ),
                    active_rollout_ids=active_rollout_ids,
                    num_graphs=dynamic_graphs,
                )
                if config.trace_spec.store_stop_now_reward
                else None
            )

            active_step_out = policy(
                retrieval_batch,
                policy_state,
                rollout_context=rollout_context,
                reward_model=reward_model,
                remaining_budget=active_remaining_budget,
                return_edge_diagnostics=False,
                compute_bdb_trace=config.trace_spec.store_bdb,
            )
            step_out = _scatter_policy_output(
                active_step_out,
                active_rollout_ids=active_rollout_ids,
                num_graphs=dynamic_graphs,
            )
            buffer.write_state_log_flow(
                t=t,
                active=active,
                state_log_flow=step_out.state_log_flow,
            )
            buffer.write_log_p_stop(
                t=t,
                active=active,
                log_p_stop=step_out.log_p_stop,
            )
            if config.trace_spec.store_te_bfm:
                buffer.write_te_bfm(
                    t=t,
                    active=active,
                    loss=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_loss,
                            "te_bfm_loss",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    valid_mask=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_valid_mask,
                            "te_bfm_valid_mask",
                        ).to(dtype=torch.float32),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ).to(dtype=torch.bool),
                    residual_abs=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_residual_abs,
                            "te_bfm_residual_abs",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    target_log_value=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_target_log_value,
                            "te_bfm_target_log_value",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    log_reward=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_log_reward,
                            "te_bfm_log_reward",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    stop_prob=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_stop_prob,
                            "te_bfm_stop_prob",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    frontier_edge_count=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_frontier_edge_count,
                            "te_bfm_frontier_edge_count",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    counterfactual_child_loss=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_counterfactual_child_loss,
                            "te_bfm_counterfactual_child_loss",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    frontier_cap_used=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_frontier_cap_used,
                            "te_bfm_frontier_cap_used",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                    frontier_cap_dropped_edge_count=_scatter_active_rows(
                        values=_require_policy_trace(
                            active_step_out.te_bfm_frontier_cap_dropped_edge_count,
                            "te_bfm_frontier_cap_dropped_edge_count",
                        ),
                        active_rollout_ids=active_rollout_ids,
                        num_graphs=dynamic_graphs,
                        fill_value=0.0,
                    ),
                )
            if config.trace_spec.store_bdb:
                buffer.write_bdb(
                    t=t,
                    active=active,
                    stop_loss=_scatter_required_trace(
                        active_step_out.bdb_stop_loss,
                        "bdb_stop_loss",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    edge_loss=_scatter_required_trace(
                        active_step_out.bdb_edge_loss,
                        "bdb_edge_loss",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    base_loss=_scatter_required_trace(
                        active_step_out.bdb_base_loss,
                        "bdb_base_loss",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    stop_valid_mask=_scatter_required_trace(
                        active_step_out.bdb_stop_valid_mask,
                        "bdb_stop_valid_mask",
                        active_rollout_ids,
                        dynamic_graphs,
                    ).to(dtype=torch.bool),
                    edge_valid_mask=_scatter_required_trace(
                        active_step_out.bdb_edge_valid_mask,
                        "bdb_edge_valid_mask",
                        active_rollout_ids,
                        dynamic_graphs,
                    ).to(dtype=torch.bool),
                    base_valid_mask=_scatter_required_trace(
                        active_step_out.bdb_base_valid_mask,
                        "bdb_base_valid_mask",
                        active_rollout_ids,
                        dynamic_graphs,
                    ).to(dtype=torch.bool),
                    delta_stop=_scatter_required_trace(
                        active_step_out.bdb_delta_stop,
                        "bdb_delta_stop",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    delta_edge=_scatter_required_trace(
                        active_step_out.bdb_delta_edge,
                        "bdb_delta_edge",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    delta_base=_scatter_required_trace(
                        active_step_out.bdb_delta_base,
                        "bdb_delta_base",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    frontier_size=_scatter_required_trace(
                        active_step_out.bdb_frontier_size,
                        "bdb_frontier_size",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    parent_count=_scatter_required_trace(
                        active_step_out.bdb_parent_count,
                        "bdb_parent_count",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    log_reward=_scatter_required_trace(
                        active_step_out.bdb_log_reward,
                        "bdb_log_reward",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                    log_flow=_scatter_required_trace(
                        active_step_out.bdb_log_flow,
                        "bdb_log_flow",
                        active_rollout_ids,
                        dynamic_graphs,
                    ),
                )

            step_context = self._build_step_context(
                t=t,
                active=active,
                remaining_budget=remaining_budget,
                step_out=step_out,
                num_graphs=dynamic_graphs,
                device=device,
            )

            if stop_now_reward is not None and config.trace_spec.store_stop_now_reward:
                buffer.write_stop_now_reward(
                    t=t,
                    active=active,
                    reward_output=stop_now_reward,
                )
            if need_policy_step_traces:
                write_policy_diagnostics(
                    buffer=buffer,
                    step_out=step_out,
                    step_context=step_context,
                    num_graphs=dynamic_graphs,
                )
            else:
                buffer.write_stop_decision_context(
                    t=t,
                    active=active,
                    action_valid_mask=step_context.can_expand,
                    budget_exhausted_mask=step_context.budget_exhausted,
                )

            action = sample_action_for_generation(
                stop_logits=step_out.stop_logits,
                edge_logits=step_out.edge_logits,
                frontier_edge_ids=step_out.frontier_edge_ids,
                frontier_batch_ids=step_out.frontier_batch_ids,
                active=step_context.active_mask,
                can_expand=step_context.can_expand,
                temperature=float(config.temperature),
                batch_size=dynamic_graphs,
            )
            result = executor.execute_step(
                step_out=step_out,
                state=state,
                step_context=step_context,
                temperature=float(config.temperature),
                stop_now_reward=stop_now_reward,
                action=action,
            )

            buffer.write_step(t, result)

        unfinished = (~buffer.is_terminated).nonzero(as_tuple=False).view(-1)
        if unfinished.numel() > 0:
            raise RuntimeError(
                f"Rollout ended with unfinished rollout rows: {unfinished.tolist()}."
            )

        return RolloutBatch.from_buffer(
            buffer=buffer,
            source_graph_id=rollout_to_graph,
        )

    @staticmethod
    def _evaluate_stop_now(
        *,
        reward_model: RewardModel,
        retrieval_batch: RetrievalBatch,
        state: State | RolloutState,
    ) -> TerminalRewardOutput:
        if isinstance(state, RolloutState):
            return reward_model.evaluate_terminal_state(
                retrieval_batch=retrieval_batch,
                state=state,
                diagnostics="basic",
            )
        return reward_model.evaluate_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
            diagnostics="basic",
        )

    @staticmethod
    def _build_step_context(
        *,
        t: int,
        active: torch.Tensor,
        remaining_budget: torch.Tensor,
        step_out: PolicyOutput,
        num_graphs: int,
        device: torch.device,
    ) -> StepContext:
        has_edge = has_frontier(
            frontier_batch_ids=step_out.frontier_batch_ids,
            num_graphs=int(num_graphs),
            device=device,
            validate_values=False,
        )
        exhausted = budget_exhausted_mask(
            remaining_budget,
            num_graphs=int(num_graphs),
            device=device,
        )
        active_mask = active.to(device=device, dtype=torch.bool)
        can_expand = active_mask & has_edge & ~exhausted
        return StepContext(
            t=int(t),
            active_mask=active_mask,
            remaining_budget=remaining_budget.to(device=device),
            has_frontier=has_edge,
            budget_exhausted=exhausted,
            can_expand=can_expand,
        )

    @staticmethod
    def _assert_synchronous_depth(
        *,
        state: State | RolloutState,
        retrieval_batch: RetrievalBatch,
        active: torch.Tensor,
        t: int,
        num_graphs: int,
    ) -> None:
        depth = state.synchronous_rollout_depth(
            edge_batch=retrieval_batch.edge_batch,
            num_graphs=int(num_graphs),
            active_graphs=active,
        )

        if depth != int(t):
            raise RuntimeError(
                "Synchronous rollout depth mismatch for unfinished graphs: "
                f"state depth={depth}, loop step={t}."
            )


__all__ = [
    "RolloutEngine",
    "RolloutEngineConfig",
]


def _scatter_policy_output(
    step_out: PolicyOutput,
    *,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
) -> PolicyOutput:
    device = step_out.stop_logits.device
    active_rollout_ids = active_rollout_ids.to(device=device, dtype=torch.long).view(-1)
    num_graphs = int(num_graphs)

    stop_logits = _scatter_active_rows(
        values=step_out.stop_logits,
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=0.0,
    )
    state_log_flow = _scatter_active_rows(
        values=step_out.state_log_flow,
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=0.0,
    )
    log_p_stop = _scatter_active_rows(
        values=step_out.log_p_stop,
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=0.0,
    )
    log_p_continue = _scatter_active_rows(
        values=step_out.log_p_continue,
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=-torch.inf,
    )
    frontier_batch_ids = active_rollout_ids.index_select(
        0,
        step_out.frontier_batch_ids.to(device=device, dtype=torch.long).view(-1),
    )

    return PolicyOutput(
        stop_logits=stop_logits,
        edge_logits=step_out.edge_logits,
        state_log_flow=state_log_flow,
        log_p_stop=log_p_stop,
        log_p_continue=log_p_continue,
        edge_cond_logprob=step_out.edge_cond_logprob,
        edge_expand_logprob=step_out.edge_expand_logprob,
        frontier_edge_ids=step_out.frontier_edge_ids,
        frontier_batch_ids=frontier_batch_ids,
        edge_policy_diagnostics=step_out.edge_policy_diagnostics,
        log_c_continue=_scatter_optional_active_rows(
            values=step_out.log_c_continue,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=-torch.inf,
        ),
        log_z_action=_scatter_optional_active_rows(
            values=step_out.log_z_action,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        terminal_energy=_scatter_optional_active_rows(
            values=step_out.terminal_energy,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        continue_energy=_scatter_optional_active_rows(
            values=step_out.continue_energy,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=-torch.inf,
        ),
        value_energy=_scatter_optional_active_rows(
            values=step_out.value_energy,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        terminal_quotient_backup_used_rate=step_out.terminal_quotient_backup_used_rate,
        terminal_quotient_parent_count=step_out.terminal_quotient_parent_count,
        terminal_quotient_edge_count=step_out.terminal_quotient_edge_count,
        terminal_quotient_group_count_mean=step_out.terminal_quotient_group_count_mean,
        terminal_quotient_floor_edge_count_mean=(
            step_out.terminal_quotient_floor_edge_count_mean
        ),
        terminal_quotient_positive_edge_count_mean=(
            step_out.terminal_quotient_positive_edge_count_mean
        ),
        terminal_quotient_speedup_estimate=step_out.terminal_quotient_speedup_estimate,
        te_bfm_loss=_scatter_optional_active_rows(
            values=step_out.te_bfm_loss,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_valid_mask=_scatter_optional_active_rows(
            values=(
                None
                if step_out.te_bfm_valid_mask is None
                else step_out.te_bfm_valid_mask.to(dtype=torch.float32)
            ),
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ).to(dtype=torch.bool)
        if step_out.te_bfm_valid_mask is not None
        else None,
        te_bfm_residual_abs=_scatter_optional_active_rows(
            values=step_out.te_bfm_residual_abs,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_target_log_value=_scatter_optional_active_rows(
            values=step_out.te_bfm_target_log_value,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_log_reward=_scatter_optional_active_rows(
            values=step_out.te_bfm_log_reward,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_stop_prob=_scatter_optional_active_rows(
            values=step_out.te_bfm_stop_prob,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_frontier_edge_count=_scatter_optional_active_rows(
            values=step_out.te_bfm_frontier_edge_count,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_counterfactual_child_loss=_scatter_optional_active_rows(
            values=step_out.te_bfm_counterfactual_child_loss,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        te_bfm_frontier_cap_used=step_out.te_bfm_frontier_cap_used,
        te_bfm_frontier_cap_dropped_edge_count=(
            step_out.te_bfm_frontier_cap_dropped_edge_count
        ),
        bdb_stop_loss=_scatter_optional_active_rows(
            values=step_out.bdb_stop_loss,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_edge_loss=_scatter_optional_active_rows(
            values=step_out.bdb_edge_loss,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_base_loss=_scatter_optional_active_rows(
            values=step_out.bdb_base_loss,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_stop_valid_mask=_scatter_optional_bool_rows(
            values=step_out.bdb_stop_valid_mask,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
        ),
        bdb_edge_valid_mask=_scatter_optional_bool_rows(
            values=step_out.bdb_edge_valid_mask,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
        ),
        bdb_base_valid_mask=_scatter_optional_bool_rows(
            values=step_out.bdb_base_valid_mask,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
        ),
        bdb_delta_stop=_scatter_optional_active_rows(
            values=step_out.bdb_delta_stop,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_delta_edge=_scatter_optional_active_rows(
            values=step_out.bdb_delta_edge,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_delta_base=_scatter_optional_active_rows(
            values=step_out.bdb_delta_base,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_frontier_size=_scatter_optional_active_rows(
            values=step_out.bdb_frontier_size,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_parent_count=_scatter_optional_active_rows(
            values=step_out.bdb_parent_count,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_log_reward=_scatter_optional_active_rows(
            values=step_out.bdb_log_reward,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
        bdb_log_flow=_scatter_optional_active_rows(
            values=step_out.bdb_log_flow,
            active_rollout_ids=active_rollout_ids,
            num_graphs=num_graphs,
            fill_value=0.0,
        ),
    )


def _scatter_active_rows(
    *,
    values: torch.Tensor,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
    fill_value: float | int,
) -> torch.Tensor:
    values = values.view(-1)
    active_rollout_ids = active_rollout_ids.to(
        device=values.device,
        dtype=torch.long,
    ).view(-1)
    if values.shape != active_rollout_ids.shape:
        raise ValueError(
            "active-row values must match active_rollout_ids shape: "
            f"{tuple(values.shape)} != {tuple(active_rollout_ids.shape)}."
        )

    out = values.new_full((int(num_graphs),), fill_value)
    out[active_rollout_ids] = values
    return out


def _scatter_optional_active_rows(
    *,
    values: torch.Tensor | None,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
    fill_value: float | int,
) -> torch.Tensor | None:
    if values is None:
        return None
    return _scatter_active_rows(
        values=values,
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=fill_value,
    )


def _scatter_optional_bool_rows(
    *,
    values: torch.Tensor | None,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor | None:
    if values is None:
        return None
    return _scatter_active_rows(
        values=values.to(dtype=torch.float32),
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=0.0,
    ).to(dtype=torch.bool)


def _scatter_required_trace(
    values: torch.Tensor | None,
    name: str,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    return _scatter_active_rows(
        values=_require_policy_trace(values, name).to(dtype=torch.float32),
        active_rollout_ids=active_rollout_ids,
        num_graphs=num_graphs,
        fill_value=0.0,
    )


def _require_policy_trace(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise RuntimeError(f"PolicyOutput.{name} is required for TE-BFM tracing.")
    return values


def _scatter_terminal_reward_output(
    reward: TerminalRewardOutput,
    *,
    active_rollout_ids: torch.Tensor,
    num_graphs: int,
) -> TerminalRewardOutput:
    def scatter(values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        return _scatter_active_rows(
            values=values,
            active_rollout_ids=active_rollout_ids.to(
                device=values.device,
                dtype=torch.long,
            ),
            num_graphs=int(num_graphs),
            fill_value=fill_value,
        )

    return TerminalRewardOutput(
        log_reward=scatter(reward.log_reward),
        raw_log_reward=scatter(reward.raw_log_reward),
        utility=scatter(reward.utility),
        base_log_reward=scatter(reward.base_log_reward),
        process_potential=scatter(reward.process_potential),
        shortest_path_potential=scatter(reward.shortest_path_potential),
        process_utility=scatter(reward.process_utility),
        process_log_bonus=scatter(reward.process_log_bonus),
        complexity_log_prior=scatter(reward.complexity_log_prior),
        complexity_penalty=scatter(reward.complexity_penalty),
        supported_answer_recall=scatter(reward.supported_answer_recall),
        supported_answer_precision=scatter(reward.supported_answer_precision),
        supported_answer_f_beta=scatter(reward.supported_answer_f_beta),
        supported_answer_count=scatter(reward.supported_answer_count),
        reward_answer_count=scatter(reward.reward_answer_count),
        supported_retrieved_count=scatter(reward.supported_retrieved_count),
        expanded_edge_count=scatter(reward.expanded_edge_count),
        directed_supported_answer_count=scatter(reward.directed_supported_answer_count),
        directed_supported_answer_recall=scatter(reward.directed_supported_answer_recall),
        has_supported_answer=scatter(reward.has_supported_answer),
        fail_penalty_applied=scatter(reward.fail_penalty_applied),
        answer_credit=scatter(reward.answer_credit),
        edge_cost=scatter(reward.edge_cost),
        fail_penalty=scatter(reward.fail_penalty),
        answer_f1=scatter(reward.answer_f1),
        answer_precision=scatter(reward.answer_precision),
        answer_recall=scatter(reward.answer_recall),
        answer_hits=scatter(reward.answer_hits),
        answer_gold=scatter(reward.answer_gold),
        retrieved_node_count=scatter(reward.retrieved_node_count),
        answer_degree_excess=scatter(reward.answer_degree_excess),
    )
