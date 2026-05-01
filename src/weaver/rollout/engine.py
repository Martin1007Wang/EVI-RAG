from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from src.data.schema import RetrievalBatch, repeat_retrieval_batch
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import Policy, PolicyOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.state import RolloutState, State

from .batch_ops import split_repeated_rollout_batch, split_static_rollout_batch
from .buffer import RolloutBuffer
from .diagnostics import write_policy_diagnostics
from .executor import (
    FusedStepExecutor,
    StepContext,
    StepExecutor,
    budget_exhausted_mask,
    has_candidate,
)
from .schema import RolloutBatch

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 compatibility.

    class StrEnum(str, Enum):
        pass


class RewardMode(StrEnum):
    EAGER_STOP_NOW = "eager_stop_now"
    LAZY_TERMINAL = "lazy_terminal"


class StepAuxiliary(Protocol):
    """
    Optional rollout-time auxiliary writer.

    The auxiliary may inspect the current state, reward model, and policy output
    and write extra targets or diagnostics into RolloutBuffer.

    It must not mutate State, sample actions, or compute the main loss.
    """

    def write_step(
        self,
        *,
        buffer: RolloutBuffer,
        t: int,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        state: State,
        step_out: PolicyOutput,
        step_context: StepContext,
        stop_now_reward: TerminalRewardOutput,
    ) -> None: ...

    @property
    def requires_stop_now_reward(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class RolloutEngineConfig:
    temperature: float
    collect_stop_counterfactual: bool = True
    collect_policy_diagnostics: bool = False
    validate_synchronous_depth: bool = False
    edge_logit_mode: str = "final"
    reward_mode: RewardMode = RewardMode.EAGER_STOP_NOW
    use_static_batch_rollouts: bool = False
    use_fused_static_batch_rollouts: bool = False


class RolloutEngine:
    """
    Finite-horizon vectorized rollout driver.

    Responsibilities:
        1. initialize canonical subgraph state;
        2. evaluate stop-now terminal reward eagerly only when diagnostics,
           StopTB, or auxiliaries need it;
        3. query policy at each active state;
        4. write state flow and diagnostics into RolloutBuffer;
        5. call optional step auxiliary writer before action execution;
        6. execute one transition;
        7. return RolloutBatch.

    It does not implement loss computation or proposal/teacher intervention.

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
        auxiliary: StepAuxiliary | None = None,
        collect_stop_counterfactual: bool = True,
        collect_policy_diagnostics: bool = False,
        validate_synchronous_depth: bool = False,
        edge_logit_mode: str = "final",
        reward_mode: RewardMode | str | None = None,
        use_static_batch_rollouts: bool = False,
        use_fused_static_batch_rollouts: bool = False,
    ) -> list[RolloutBatch]:
        """
        Run K independent rollouts.

        For K > 1, physically repeat the RetrievalBatch once, run one
        vectorized rollout, then split the result back into K logical rollout
        batches.

        When use_static_batch_rollouts=True, keep the original RetrievalBatch
        and FeatureBank at size B, then execute K independent dynamic states
        against that shared static context. This is the conservative first step
        toward static-batch/dynamic-rollout separation; the default physical
        repeat path remains available for parity and fallback.

        When use_fused_static_batch_rollouts=True, keep the static batch at B
        and execute one fused horizon loop over R=K*B dynamic rollout rows.
        """
        num_rollouts = int(num_rollouts)
        if num_rollouts <= 0:
            return []

        config = RolloutEngineConfig(
            temperature=float(temperature),
            collect_stop_counterfactual=bool(collect_stop_counterfactual),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
            edge_logit_mode=str(edge_logit_mode),
            reward_mode=self._resolve_reward_mode(
                reward_mode=reward_mode,
                collect_stop_counterfactual=bool(collect_stop_counterfactual),
                auxiliary=auxiliary,
            ),
            use_static_batch_rollouts=bool(use_static_batch_rollouts),
            use_fused_static_batch_rollouts=bool(use_fused_static_batch_rollouts),
        )

        if config.edge_logit_mode not in {"final", "semantic"}:
            raise ValueError(
                "edge_logit_mode must be 'final' or 'semantic', "
                f"got {config.edge_logit_mode!r}."
            )
        if (
            config.reward_mode == RewardMode.LAZY_TERMINAL
            and config.collect_stop_counterfactual
        ):
            raise ValueError(
                "reward_mode='lazy_terminal' cannot collect stop counterfactuals; "
                "use reward_mode='eager_stop_now'."
            )
        if (
            config.reward_mode == RewardMode.LAZY_TERMINAL
            and auxiliary is not None
            and bool(getattr(auxiliary, "requires_stop_now_reward", True))
        ):
            raise ValueError(
                "reward_mode='lazy_terminal' is incompatible with a rollout "
                "auxiliary that requires stop_now_reward."
            )

        if num_rollouts == 1:
            rollout_context = policy.prepare_rollout_context(retrieval_batch)
            rollout = self._run_one_batch(
                policy=policy,
                retrieval_batch=retrieval_batch,
                reward_model=reward_model,
                rollout_context=rollout_context,
                auxiliary=auxiliary,
                config=config,
            )
            return [rollout]

        if config.use_fused_static_batch_rollouts:
            if auxiliary is not None:
                raise ValueError(
                    "Fused static-batch rollouts do not support step auxiliaries yet; "
                    "disable fused static rollouts or run without StopAdvantage."
                )
            rollout_context = policy.prepare_rollout_context(retrieval_batch)
            rollout = self._run_fused_static_batch(
                policy=policy,
                retrieval_batch=retrieval_batch,
                reward_model=reward_model,
                rollout_context=rollout_context,
                num_rollouts=num_rollouts,
                auxiliary=auxiliary,
                config=config,
            )
            return split_static_rollout_batch(
                rollout=rollout,
                original_batch=retrieval_batch,
                repeats=num_rollouts,
            )

        if config.use_static_batch_rollouts:
            rollout_context = policy.prepare_rollout_context(retrieval_batch)
            return [
                self._run_one_batch(
                    policy=policy,
                    retrieval_batch=retrieval_batch,
                    reward_model=reward_model,
                    rollout_context=rollout_context,
                    auxiliary=auxiliary,
                    config=config,
                )
                for _ in range(num_rollouts)
            ]

        repeated_batch = repeat_retrieval_batch(retrieval_batch, num_rollouts)
        rollout_context = policy.prepare_rollout_context(repeated_batch)

        rollout = self._run_one_batch(
            policy=policy,
            retrieval_batch=repeated_batch,
            reward_model=reward_model,
            rollout_context=rollout_context,
            auxiliary=auxiliary,
            config=config,
        )

        return split_repeated_rollout_batch(
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
        rollout_context: FeatureBank,
        num_rollouts: int,
        auxiliary: StepAuxiliary | None,
        config: RolloutEngineConfig,
    ) -> RolloutBatch:
        if config.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {config.temperature}.")
        if auxiliary is not None:
            raise ValueError("Fused static-batch rollouts do not support auxiliaries.")

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
        )

        executor = FusedStepExecutor(
            retrieval_batch=retrieval_batch,
            reward_model=reward_model,
        )

        need_policy_step_traces = (
            config.collect_policy_diagnostics
            or config.collect_stop_counterfactual
            or config.reward_mode == RewardMode.EAGER_STOP_NOW
        )

        for t in range(self.expand_budget + 1):
            active = ~buffer.is_terminated
            if not bool(active.any()):
                break

            if config.validate_synchronous_depth:
                self._assert_synchronous_depth(
                    state=state,
                    retrieval_batch=retrieval_batch,
                    active=active,
                    t=t,
                    num_graphs=dynamic_graphs,
                )

            remaining_budget = state.remaining_budget_per_graph(
                edge_batch=retrieval_batch.edge_batch,
                num_graphs=dynamic_graphs,
            )

            policy_state = state.detach()

            stop_now_reward: TerminalRewardOutput | None = None
            if config.reward_mode == RewardMode.EAGER_STOP_NOW:
                stop_now_reward = self._evaluate_stop_now(
                    reward_model=reward_model,
                    retrieval_batch=retrieval_batch,
                    state=policy_state,
                )

            step_out = policy(
                retrieval_batch,
                policy_state,
                rollout_context=rollout_context,
                return_edge_breakdown=False,
                edge_logit_mode=config.edge_logit_mode,
            )

            step_context = self._build_step_context(
                t=t,
                active=active,
                remaining_budget=remaining_budget,
                step_out=step_out,
                num_graphs=dynamic_graphs,
                device=device,
            )

            self._write_flow(
                buffer=buffer,
                step_out=step_out,
                active=active,
                t=t,
            )

            if stop_now_reward is not None:
                buffer.write_stop_now_reward(
                    t=t,
                    active=active,
                    reward_output=stop_now_reward,
                )
            elif config.collect_stop_counterfactual:
                raise RuntimeError(
                    "stop_now_reward is required when stop counterfactuals are collected."
                )

            if need_policy_step_traces:
                write_policy_diagnostics(
                    buffer=buffer,
                    step_out=step_out,
                    step_context=step_context,
                    num_graphs=dynamic_graphs,
                )

            result = executor.execute_step(
                step_out=step_out,
                state=state,
                step_context=step_context,
                temperature=float(config.temperature),
                stop_now_reward=stop_now_reward,
            )

            buffer.write_step(t, result)

        unfinished = (~buffer.is_terminated).nonzero(as_tuple=False).view(-1)
        if unfinished.numel() > 0:
            raise RuntimeError(
                f"Rollout ended with unfinished rollout rows: {unfinished.tolist()}."
            )

        return RolloutBatch.from_buffer(buffer=buffer)

    def _run_one_batch(
        self,
        *,
        policy: Policy,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        rollout_context: FeatureBank,
        auxiliary: StepAuxiliary | None,
        config: RolloutEngineConfig,
    ) -> RolloutBatch:
        if config.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {config.temperature}.")

        device = retrieval_batch.edge_index.device
        num_graphs = int(retrieval_batch.num_graphs)

        state = State.create_initial(
            retrieval_batch,
            expand_budget=self.expand_budget,
        )

        buffer = RolloutBuffer(
            B=num_graphs,
            T=self.expand_budget + 1,
            device=device,
        )

        executor = StepExecutor(
            retrieval_batch=retrieval_batch,
            reward_model=reward_model,
        )

        need_policy_step_traces = (
            config.collect_policy_diagnostics
            or config.collect_stop_counterfactual
            or config.reward_mode == RewardMode.EAGER_STOP_NOW
            or auxiliary is not None
        )

        for t in range(self.expand_budget + 1):
            active = ~buffer.is_terminated
            if not bool(active.any()):
                break

            if config.validate_synchronous_depth:
                self._assert_synchronous_depth(
                    state=state,
                    retrieval_batch=retrieval_batch,
                    active=active,
                    t=t,
                    num_graphs=num_graphs,
                )

            remaining_budget = state.remaining_budget_per_graph(
                edge_batch=retrieval_batch.edge_batch,
                num_graphs=num_graphs,
            )

            # Policy and reward observe a canonical snapshot. Executor mutates
            # the live state and validates node/edge closure before transition.
            policy_state = state.detach()

            stop_now_reward: TerminalRewardOutput | None = None
            if config.reward_mode == RewardMode.EAGER_STOP_NOW:
                stop_now_reward = self._evaluate_stop_now(
                    reward_model=reward_model,
                    retrieval_batch=retrieval_batch,
                    state=policy_state,
                )

            step_out = policy(
                retrieval_batch,
                policy_state,
                rollout_context=rollout_context,
                return_edge_breakdown=auxiliary is not None,
                edge_logit_mode=config.edge_logit_mode,
            )

            step_context = self._build_step_context(
                t=t,
                active=active,
                remaining_budget=remaining_budget,
                step_out=step_out,
                num_graphs=num_graphs,
                device=device,
            )

            self._write_flow(
                buffer=buffer,
                step_out=step_out,
                active=active,
                t=t,
            )

            if stop_now_reward is not None:
                buffer.write_stop_now_reward(
                    t=t,
                    active=active,
                    reward_output=stop_now_reward,
                )
            elif config.collect_stop_counterfactual:
                raise RuntimeError(
                    "stop_now_reward is required when stop counterfactuals are collected."
                )

            if need_policy_step_traces:
                write_policy_diagnostics(
                    buffer=buffer,
                    step_out=step_out,
                    step_context=step_context,
                    num_graphs=num_graphs,
                )

            if auxiliary is not None:
                if stop_now_reward is None:
                    raise RuntimeError(
                        "stop_now_reward is required when rollout auxiliary is enabled."
                    )
                auxiliary.write_step(
                    buffer=buffer,
                    t=t,
                    retrieval_batch=retrieval_batch,
                    reward_model=reward_model,
                    state=policy_state,
                    step_out=step_out,
                    step_context=step_context,
                    stop_now_reward=stop_now_reward,
                )

            result = executor.execute_step(
                step_out=step_out,
                state=state,
                step_context=step_context,
                temperature=float(config.temperature),
                stop_now_reward=stop_now_reward,
            )

            buffer.write_step(t, result)

        unfinished = (~buffer.is_terminated).nonzero(as_tuple=False).view(-1)
        if unfinished.numel() > 0:
            raise RuntimeError(
                f"Rollout ended with unfinished graphs: {unfinished.tolist()}."
            )

        return RolloutBatch.from_buffer(buffer=buffer)

    @staticmethod
    def _evaluate_stop_now(
        *,
        reward_model: RewardModel,
        retrieval_batch: RetrievalBatch,
        state: State | RolloutState,
    ) -> TerminalRewardOutput:
        return reward_model.evaluate_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
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
        has_edge = has_candidate(
            candidate_batch_ids=step_out.candidate_batch_ids,
            num_graphs=int(num_graphs),
            device=device,
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
            has_candidate=has_edge,
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

    @staticmethod
    def _write_flow(
        *,
        buffer: RolloutBuffer,
        step_out: PolicyOutput,
        active: torch.Tensor,
        t: int,
    ) -> None:
        """
        Write log F(s_t | q). At t=0, root flow is also log Z(q).
        """
        if t == 0:
            root_log_z = step_out.state_log_flow.to(dtype=torch.float32)
            buffer.write_root_log_z(root_log_z)

            buffer.write_state_log_flow(
                t=0,
                active=torch.ones_like(active, dtype=torch.bool),
                state_log_flow=root_log_z,
            )
            return

        buffer.write_state_log_flow(
            t=t,
            active=active,
            state_log_flow=step_out.state_log_flow,
        )

    @staticmethod
    def _resolve_reward_mode(
        *,
        reward_mode: RewardMode | str | None,
        collect_stop_counterfactual: bool,
        auxiliary: StepAuxiliary | None,
    ) -> RewardMode:
        if reward_mode is None:
            if bool(collect_stop_counterfactual) or auxiliary is not None:
                return RewardMode.EAGER_STOP_NOW
            return RewardMode.LAZY_TERMINAL

        if isinstance(reward_mode, RewardMode):
            return reward_mode

        try:
            return RewardMode(str(reward_mode))
        except ValueError as exc:
            raise ValueError(
                "reward_mode must be 'eager_stop_now' or 'lazy_terminal', "
                f"got {reward_mode!r}."
            ) from exc


__all__ = [
    "RewardMode",
    "RolloutEngine",
    "RolloutEngineConfig",
    "StepAuxiliary",
]
