from __future__ import annotations
from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.models.policy import Policy
from src.models.state import State
from src.models.guidance import TeacherGuidance

from .buffers import RolloutBuffer
from .executor import Executor
from .sampling import ActionSampler
from .types import RolloutBatch


class RolloutEngine:
    def __init__(self, expand_budget: int) -> None:
        if expand_budget < 0:
            raise ValueError(f"expand_budget must be >= 0, got {expand_budget}.")
        self.expand_budget = expand_budget

    def run_exploration(
        self,
        policy: Policy,
        retrieval_batch: RetrievalBatch,
        reward_model: Any,
        num_rollouts: int = 1,
        temperature: float = 1.0,
        teacher_guidance: TeacherGuidance | None = None,
        teacher_force_prob: float = 0.0,
    ) -> list[RolloutBatch]:
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}.")
        rollout_ctx = policy.prepare_rollout_context(retrieval_batch)
        return [
            self._run_single(
                policy=policy,
                retrieval_batch=retrieval_batch,
                reward_model=reward_model,
                temperature=temperature,
                rollout_ctx=rollout_ctx,
                teacher_guidance=teacher_guidance,
                teacher_force_prob=teacher_force_prob,
            )
            for _ in range(num_rollouts)
        ]

    def _run_single(
        self,
        *,
        policy: Policy,
        retrieval_batch: RetrievalBatch,
        reward_model: Any,
        temperature: float,
        rollout_ctx: Any | None,
        teacher_guidance: TeacherGuidance | None,
        teacher_force_prob: float,
    ) -> RolloutBatch:
        batch_size = int(retrieval_batch.ptr.numel()) - 1
        device = retrieval_batch.node_tokens.device
        state = State.create_initial(retrieval_batch, expand_budget=self.expand_budget)
        buffer = RolloutBuffer(B=batch_size, T=self.expand_budget + 1, device=device)
        sampler = ActionSampler(
            teacher_guidance=teacher_guidance,
            teacher_force_prob=teacher_force_prob,
            edge_ptr=retrieval_batch.edge_ptr.long(),
            batch_size=batch_size,
            device=device,
            expand_budget=self.expand_budget,
        )
        executor = Executor(
            expand_budget=self.expand_budget,
            retrieval_batch=retrieval_batch,
            reward_model=reward_model,
            sampler=sampler,
        )
        root_log_z = None
        for num_expands in range(self.expand_budget + 1):
            active = ~buffer.is_terminated
            if not active.any():
                break
            state.num_expands = num_expands
            step_out = policy(retrieval_batch, state.as_policy_input(), rollout_context=rollout_ctx)
            if num_expands == 0:
                assert step_out.root_log_z is not None
                root_log_z = step_out.root_log_z.to(dtype=torch.float32)
                buffer.state_log_flows[:, 0] = root_log_z
            else:
                buffer.state_log_flows[active, num_expands] = step_out.state_log_flow[active].to(dtype=torch.float32)
            result = executor.execute_step(
                num_expands=num_expands,
                step_out=step_out,
                state=state,
                active=active,
                temperature=temperature,
            )
            buffer.write_step(num_expands, result)
        unfinished = ~buffer.is_terminated
        if unfinished.any():
            raise RuntimeError(f"Rollout ended with unfinished graphs: {torch.nonzero(unfinished, as_tuple=False).view(-1).tolist()}")
        buffer.finalize_teacher_counts(sampler.teacher_action_counts())
        assert root_log_z is not None, "root_log_z was never computed."
        return RolloutBatch.from_run(buffer=buffer, root_log_z=root_log_z)
__all__ = ["RolloutEngine"]
