from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.data.schema import RetrievalBatch, repeat_retrieval_batch
from src.weaver.nn.backbone import FeatureBank
from src.weaver.policy import Policy, PolicyStepOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.rollout.sampling import option_action_log_probs
from src.weaver.state import State

from .batch_ops import split_repeated_rollout_batch
from .buffers import RolloutBuffer
from .diagnostics import write_policy_diagnostics
from .executor import StepExecutor
from .schema import RolloutBatch


class StepAuxiliary(Protocol):
    """
    Optional rollout-time auxiliary writer.

    This is intentionally a narrow protocol. RolloutEngine should not know
    whether the auxiliary is VIGOR, a debug oracle, or something else.
    """

    def write_step(
        self,
        *,
        buffer: RolloutBuffer,
        t: int,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        state: State,
        step_out: PolicyStepOutput,
        active: torch.Tensor,
        remaining_budget: torch.Tensor,
        current_reward: TerminalRewardOutput,
    ) -> None: ...


@dataclass(frozen=True)
class RolloutEngineConfig:
    temperature: float
    collect_stop_counterfactual: bool = True
    collect_policy_diagnostics: bool = False
    validate_synchronous_depth: bool = False


class RolloutEngine:
    """
    Finite-horizon vectorized rollout driver.

    Responsibilities:
        1. initialize canonical subgraph state;
        2. evaluate stop-now terminal reward for diagnostics / StopTB / auxiliary;
        3. query policy at each active state;
        4. write state flow and diagnostics into RolloutBuffer;
        5. call optional step auxiliary writer;
        6. execute one transition;
        7. return RolloutBatch.

    Non-responsibilities:
        - no coverage proposal;
        - no behavior-policy teacher;
        - no proposal intervention;
        - no VIGOR math;
        - no candidate top-k construction;
        - no loss computation.
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
    ) -> list[RolloutBatch]:
        """
        Run K independent rollouts.

        For K > 1, physically repeat the retrieval batch once, run one
        vectorized rollout, then split the result back into K logical rollout
        batches.
        """
        num_rollouts = int(num_rollouts)
        if num_rollouts <= 0:
            return []

        config = RolloutEngineConfig(
            temperature=float(temperature),
            collect_stop_counterfactual=bool(collect_stop_counterfactual),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
        )

        if num_rollouts == 1:
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

        root_log_z: torch.Tensor | None = None

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

            remaining_budget = torch.full(
                (num_graphs,),
                max(self.expand_budget - int(t), 0),
                dtype=torch.long,
                device=device,
            )

            # Policy observes an immutable snapshot. Executor mutates real state.
            policy_state = state.detach()

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
            )

            root_log_z = self._write_flow(
                buffer=buffer,
                step_out=step_out,
                active=active,
                t=t,
                root_log_z=root_log_z,
            )

            if config.collect_stop_counterfactual:
                buffer.write_stop_counterfactual(
                    t=t,
                    active=active,
                    reward_output=stop_now_reward,
                )

            if config.collect_policy_diagnostics or config.collect_stop_counterfactual:
                write_policy_diagnostics(
                    buffer=buffer,
                    step_out=step_out,
                    active=active,
                    t=t,
                    num_graphs=num_graphs,
                    remaining_budget=remaining_budget,
                )

            if auxiliary is not None:
                auxiliary.write_step(
                    buffer=buffer,
                    t=t,
                    retrieval_batch=retrieval_batch,
                    reward_model=reward_model,
                    state=policy_state,
                    step_out=step_out,
                    active=active,
                    remaining_budget=remaining_budget,
                    current_reward=stop_now_reward,
                )

            result = executor.execute_step(
                step_out=step_out,
                state=state,
                active=active,
                temperature=float(config.temperature),
                remaining_budget=remaining_budget,
                stop_now_reward=stop_now_reward,
            )

            buffer.write_step(t, result)

        unfinished = (~buffer.is_terminated).nonzero(as_tuple=False).view(-1)
        if unfinished.numel() > 0:
            raise RuntimeError(
                f"Rollout ended with unfinished graphs: {unfinished.tolist()}."
            )

        if root_log_z is None:
            raise RuntimeError("root_log_z was never computed.")

        return RolloutBatch.from_buffer(
            buffer=buffer,
            root_log_z=root_log_z,
        )

    @staticmethod
    def _evaluate_stop_now(
        *,
        reward_model: RewardModel,
        retrieval_batch: RetrievalBatch,
        state: State,
    ) -> TerminalRewardOutput:
        return reward_model.evaluate_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )

    @staticmethod
    def _assert_synchronous_depth(
        *,
        state: State,
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
        step_out: PolicyStepOutput,
        active: torch.Tensor,
        t: int,
        root_log_z: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Write log F(s_t | q). At t=0, root flow is also log Z(q).
        """
        if t == 0:
            if step_out.root_log_z is None:
                raise RuntimeError("Policy did not return root_log_z at root state.")

            root_log_z = step_out.root_log_z.to(dtype=torch.float32)
            root_active = torch.ones_like(active, dtype=torch.bool)

            buffer.write_state_log_flow(
                t=0,
                active=root_active,
                state_log_flow=root_log_z,
            )
            return root_log_z

        if root_log_z is None:
            raise RuntimeError("root_log_z must be available after t=0.")

        buffer.write_state_log_flow(
            t=t,
            active=active,
            state_log_flow=step_out.state_log_flow,
        )

        return root_log_z


__all__ = [
    "RolloutEngine",
    "RolloutEngineConfig",
    "StepAuxiliary",
]
