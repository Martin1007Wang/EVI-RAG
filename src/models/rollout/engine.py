from __future__ import annotations

import warnings
from typing import Any, Sequence

import torch

from src.data.schema import RetrievalBatch
from src.models.policy import Policy
from src.models.replay import TrajectoryTrace
from src.models.state import State
from src.models.teacher_guidance import TeacherGuidance

from .buffers import RolloutAccumulators
from .executor import StepExecutor
from .sampling import ActionSampler, segmented_gumbel_sample
from .traces import (
    build_trajectory_traces,
    resolve_batch_sample_ids,
    resolve_edge_ptr,
    validate_traces,
)
from .types import ROLLOUT_DTYPE, RolloutBatch


def _extract_flow_state_h(step_output: Any) -> torch.Tensor:
    flow_state_h = getattr(step_output, "flow_state_h", None)
    if flow_state_h is not None:
        return flow_state_h
    warnings.warn(
        "StepOutput.flow_state_h not found; falling back to state_h. "
        "Declare flow_state_h explicitly in StepOutput to decouple FlowHead gradients.",
        stacklevel=3,
    )
    return step_output.state_h


class RolloutEngine:
    def __init__(
        self,
        max_steps: int,
        *,
        terminal_backward_mode: str = "deterministic",
    ) -> None:
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
        if terminal_backward_mode not in {"deterministic", "policy"}:
            raise ValueError(
                "terminal_backward_mode must be 'deterministic' or 'policy', "
                f"got {terminal_backward_mode!r}."
            )
        self.max_steps = max_steps
        self.terminal_backward_mode = terminal_backward_mode

    def run_exploration(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        num_rollouts: int = 1,
        temperature: float = 1.0,
        collect_terminal_state: bool = True,
        terminal_state_device: torch.device | str | None = None,
        teacher_guidance: TeacherGuidance | None = None,
        teacher_force_prob: float = 0.0,
    ) -> list[RolloutBatch]:
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}.")
        backbone_ctx = self._precompute_backbone(policy, base_graph)
        return [
            self._run_single(
                policy=policy,
                base_graph=base_graph,
                reward_model=reward_model,
                temperature=temperature,
                collect_terminal_state=collect_terminal_state,
                terminal_state_device=terminal_state_device,
                backbone_ctx=backbone_ctx,
                forced_traces=None,
                teacher_guidance=teacher_guidance,
                teacher_force_prob=teacher_force_prob,
            )
            for _ in range(num_rollouts)
        ]

    def replay_trajectories(
        self,
        *,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        traces: Sequence[TrajectoryTrace],
        collect_terminal_state: bool = False,
        terminal_state_device: torch.device | str | None = None,
    ) -> RolloutBatch:
        backbone_ctx = self._precompute_backbone(policy, base_graph)
        return self._run_single(
            policy=policy,
            base_graph=base_graph,
            reward_model=reward_model,
            temperature=1.0,
            collect_terminal_state=collect_terminal_state,
            terminal_state_device=terminal_state_device,
            backbone_ctx=backbone_ctx,
            forced_traces=traces,
            teacher_guidance=None,
            teacher_force_prob=0.0,
        )

    def _run_exploration_once(
        self,
        *,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        temperature: float = 1.0,
        collect_terminal_state: bool = True,
        terminal_state_device: torch.device | str | None = None,
    ) -> RolloutBatch:
        backbone_ctx = self._precompute_backbone(policy, base_graph)
        return self._run_single(
            policy=policy,
            base_graph=base_graph,
            reward_model=reward_model,
            temperature=temperature,
            collect_terminal_state=collect_terminal_state,
            terminal_state_device=terminal_state_device,
            backbone_ctx=backbone_ctx,
            forced_traces=None,
            teacher_guidance=None,
            teacher_force_prob=0.0,
        )

    def _run_single(
        self,
        *,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        temperature: float,
        collect_terminal_state: bool,
        terminal_state_device: torch.device | str | None,
        backbone_ctx: Any | None,
        forced_traces: Sequence[TrajectoryTrace] | None,
        teacher_guidance: TeacherGuidance | None,
        teacher_force_prob: float,
    ) -> RolloutBatch:
        batch_size = int(base_graph.ptr.numel()) - 1
        device = base_graph.node_tokens.device
        trajectory_horizon = self.max_steps + 1
        sample_ids = resolve_batch_sample_ids(base_graph)

        validated_traces: tuple[TrajectoryTrace, ...] | None = None
        if forced_traces is not None:
            validated_traces = validate_traces(forced_traces, batch_size=batch_size)
            if sample_ids is not None:
                expected = [trace.sample_id for trace in validated_traces]
                if sample_ids != expected:
                    raise ValueError(
                        f"Forced replay sample_id mismatch: batch={sample_ids}, traces={expected}."
                    )

        state = State.create_initial(base_graph)
        accumulators = RolloutAccumulators(
            B=batch_size,
            T=trajectory_horizon,
            device=device,
        )
        sampler = ActionSampler(
            forced_traces=validated_traces,
            teacher_guidance=teacher_guidance,
            teacher_force_prob=teacher_force_prob,
            edge_ptr=resolve_edge_ptr(base_graph),
            batch_size=batch_size,
            device=device,
            max_steps=self.max_steps,
        )
        executor = StepExecutor(
            max_steps=self.max_steps,
            terminal_backward_mode=self.terminal_backward_mode,
        )

        if not sampler.is_replay and sample_ids is None:
            warnings.warn(
                "run_exploration: base_graph.sample_id is not set. "
                "TrajectoryTrace will NOT be recorded and the replay buffer "
                "will remain empty. Set RetrievalBatch.sample_id to enable "
                "off-policy replay.",
                stacklevel=3,
            )

        recorded_edge_traces = (
            [[] for _ in range(batch_size)]
            if sample_ids is not None and not sampler.is_replay
            else None
        )

        state.rollout_step = 0
        initial_output = self._policy_forward(policy, base_graph, state, backbone_ctx)
        root_log_z = self._compute_root_log_z(policy, initial_output).to(
            dtype=ROLLOUT_DTYPE
        )
        accumulators.state_log_flows[:, 0] = root_log_z

        snapshot = lambda tensor: self._snapshot(
            tensor,
            collect=collect_terminal_state,
            device=terminal_state_device,
        )
        pre_stop_nodes = (
            torch.zeros_like(state.active_nodes) if collect_terminal_state else None
        )
        pre_stop_edges = (
            torch.zeros_like(state.active_edges) if collect_terminal_state else None
        )

        for t in range(trajectory_horizon):
            active = ~accumulators.is_terminated
            if not active.any():
                break

            state.rollout_step = t
            step_out = (
                initial_output
                if t == 0
                else self._policy_forward(policy, base_graph, state, backbone_ctx)
            )

            if t > 0:
                flow_h = _extract_flow_state_h(step_out)
                flow_t = policy.state_log_flow(
                    query_h=step_out.query_h,
                    flow_state_h=flow_h,
                ).to(dtype=ROLLOUT_DTYPE)
                accumulators.state_log_flows[active, t] = flow_t[active]

            result = executor.execute_step(
                t=t,
                step_out=step_out,
                state=state,
                active=active,
                policy=policy,
                base_graph=base_graph,
                reward_model=reward_model,
                backbone_static_context=backbone_ctx,
                sampler=sampler,
                temperature=temperature,
                collect_terminal_state=collect_terminal_state,
                pre_stop_nodes=pre_stop_nodes,
                pre_stop_edges=pre_stop_edges,
                recorded_edge_traces=recorded_edge_traces,
            )
            accumulators.write_step(t, result)

        unfinished = ~accumulators.is_terminated
        if unfinished.any():
            raise RuntimeError(
                "Rollout ended with unfinished graphs: "
                f"{torch.nonzero(unfinished, as_tuple=False).view(-1).tolist()}"
            )

        trajectory_traces = build_trajectory_traces(
            base_graph=base_graph,
            batch_size=batch_size,
            acc=accumulators,
            validated_traces=validated_traces,
            recorded_edge_traces=recorded_edge_traces,
            sample_ids=sample_ids,
            teacher_action_counts=sampler.teacher_action_counts(),
        )

        return RolloutBatch(
            root_log_z=root_log_z,
            traj_len=accumulators.traj_len,
            trajectory_log_pf=accumulators.trajectory_log_pf,
            trajectory_log_pb=accumulators.trajectory_log_pb,
            terminal_log_rewards=accumulators.terminal_log_rewards,
            state_log_flows=accumulators.state_log_flows,
            step_log_pf=accumulators.step_log_pf,
            step_log_pb=accumulators.step_log_pb,
            step_log_shaping=accumulators.step_log_shaping,
            terminal_active_nodes=snapshot(state.active_nodes),
            terminal_active_edges=snapshot(state.active_edges),
            terminal_pre_stop_active_nodes=(
                snapshot(pre_stop_nodes) if pre_stop_nodes is not None else None
            ),
            terminal_pre_stop_active_edges=(
                snapshot(pre_stop_edges) if pre_stop_edges is not None else None
            ),
            terminal_stop_mask=snapshot(accumulators.terminal_stop_mask),
            terminal_stop_log_pb=snapshot(accumulators.terminal_stop_log_pb),
            selected_edge_ids=accumulators.selected_edge_ids,
            selected_relation_only_logits=accumulators.selected_relation_only_logits,
            selected_final_logits=accumulators.selected_final_logits,
            teacher_forced_action_count=sampler.teacher_action_counts(),
            trajectory_traces=trajectory_traces,
        )

    @staticmethod
    def _segmented_gumbel_sample(
        logits: torch.Tensor,
        batch_idx: torch.Tensor,
        num_segments: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return segmented_gumbel_sample(
            logits=logits,
            batch_idx=batch_idx,
            num_segments=num_segments,
            temperature=temperature,
        )

    @staticmethod
    def _masked_logit_value(reference: torch.Tensor) -> float:
        return torch.finfo(reference.dtype).min

    @staticmethod
    def _precompute_backbone(policy: Policy, base_graph: RetrievalBatch) -> Any | None:
        fn = getattr(policy, "precompute_backbone_static_context", None)
        return fn(base_graph) if callable(fn) else None

    @staticmethod
    def _policy_forward(
        policy: Policy,
        base_graph: RetrievalBatch,
        state: State,
        backbone_ctx: Any | None,
    ) -> Any:
        policy_state = state.as_policy_input()
        if backbone_ctx is None:
            return policy(base_graph, policy_state)
        return policy(base_graph, policy_state, backbone_static_context=backbone_ctx)

    @staticmethod
    def _compute_root_log_z(policy: Policy, initial_output: Any) -> torch.Tensor:
        flow_h = _extract_flow_state_h(initial_output)
        return policy.root_log_z(query_h=initial_output.query_h, root_state_h=flow_h)

    @staticmethod
    def _snapshot(
        tensor: torch.Tensor,
        *,
        collect: bool,
        device: torch.device | str | None,
    ) -> torch.Tensor | None:
        if not collect:
            return None
        snap = tensor.detach().clone()
        return snap.to(device) if device is not None else snap


__all__ = ["RolloutEngine"]
