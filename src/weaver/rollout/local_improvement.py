from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.policy import PolicyOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.rollout.buffer import RolloutBuffer
from src.weaver.rollout.executor import StepContext
from src.weaver.state import RolloutState, State


@dataclass(frozen=True, slots=True)
class LocalImprovementConfig:
    enabled: bool = False
    temperature: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict[str, object] | None) -> "LocalImprovementConfig":
        cfg = dict(cfg or {})
        defaults = cls()
        enabled = bool(cfg.pop("enabled", defaults.enabled))
        temperature = float(cfg.pop("temperature", defaults.temperature))
        if temperature <= 0.0:
            raise ValueError(
                f"local_improvement.temperature must be > 0, got {temperature}."
            )
        if cfg:
            raise ValueError(f"Unused local_improvement config keys: {sorted(cfg)}.")
        return cls(enabled=enabled, temperature=temperature)


class LocalImprovementAuxiliary:
    """
    Reward-consistent local credit assignment over Expand edges.

    It constructs Q_R(e|s) from one-step stop-reward improvements and writes a
    per-state KL(Q_R || P_F(.|s, Expand)) term. This is an auxiliary warmup
    signal; SubTB remains the main GFlowNet objective.
    """

    def __init__(self, cfg: LocalImprovementConfig) -> None:
        self.cfg = cfg

    @property
    def supports_fused_rollouts(self) -> bool:
        return True

    @property
    def requires_stop_now_reward(self) -> bool:
        return self.cfg.enabled

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
    ) -> None:
        if not self.cfg.enabled:
            return

        device = step_out.edge_logits.device
        num_graphs = int(step_out.stop_logits.numel())
        active = step_context.active_mask.to(device=device, dtype=torch.bool)
        valid = step_context.can_expand.to(device=device, dtype=torch.bool)
        loss = step_out.stop_logits.new_zeros(num_graphs)

        if not bool(valid.any()) or step_out.candidate_edge_ids.numel() == 0:
            buffer.write_local_improvement(
                t=t,
                active=active,
                loss=loss,
                valid_mask=torch.zeros_like(valid),
            )
            return

        child = evaluate_candidate_child_rewards(
            batch=retrieval_batch,
            state=state,
            reward_model=reward_model,
            candidate_edge_ids=step_out.candidate_edge_ids,
            candidate_batch_ids=step_out.candidate_batch_ids,
        )
        stop_values = stop_now_reward.log_reward.index_select(
            0,
            step_out.candidate_batch_ids.to(
                device=stop_now_reward.log_reward.device,
                dtype=torch.long,
            ),
        ).to(device=device, dtype=step_out.edge_logits.dtype)
        advantage = child.to(device=device, dtype=step_out.edge_logits.dtype)
        advantage = advantage - stop_values

        valid_mask = valid.clone()
        for graph_id_tensor in valid.nonzero(as_tuple=False).view(-1):
            graph_id = int(graph_id_tensor.item())
            pos = (
                step_out.candidate_batch_ids.eq(graph_id)
                .nonzero(as_tuple=False)
                .view(-1)
            )
            if pos.numel() == 0:
                valid_mask[graph_id] = False
                continue

            q_logp = torch.log_softmax(
                advantage.index_select(0, pos) / float(self.cfg.temperature),
                dim=0,
            )
            p_logp = torch.log_softmax(step_out.edge_logits.index_select(0, pos), dim=0)
            q = q_logp.exp()
            loss[graph_id] = (q * (q_logp - p_logp)).sum()

        buffer.write_local_improvement(
            t=t,
            active=active,
            loss=loss,
            valid_mask=valid_mask,
        )


@torch.no_grad()
def evaluate_candidate_child_rewards(
    *,
    batch: RetrievalBatch,
    state: State | RolloutState,
    reward_model: RewardModel,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate log R(s + e) for every candidate expansion.

    This uses RewardModel itself rather than duplicating reward algebra, so the
    local-improvement teacher stays aligned with the terminal reward definition.
    """
    device = state.active_nodes.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    row_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_ids.shape != row_ids.shape:
        raise ValueError(
            "candidate_edge_ids and candidate_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(row_ids.shape)}."
        )
    if edge_ids.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    child_state = _candidate_child_state(
        batch=batch,
        state=state,
        candidate_edge_ids=edge_ids,
        candidate_batch_ids=row_ids,
    )
    reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=child_state.active_nodes,
        active_edges=child_state.active_edges,
        state=child_state,
    )
    return reward.log_reward.to(device=device, dtype=torch.float32)


def _candidate_child_state(
    *,
    batch: RetrievalBatch,
    state: State | RolloutState,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> RolloutState:
    device = state.active_nodes.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    row_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
    num_candidates = int(edge_ids.numel())

    if isinstance(state, RolloutState) or state.active_nodes.ndim == 2:
        if not isinstance(state, RolloutState):
            raise TypeError("2D active masks require RolloutState.")
        child = RolloutState(
            active_nodes=state.active_nodes.index_select(0, row_ids).clone(),
            active_edges=state.active_edges.index_select(0, row_ids).clone(),
            root_edges=state.root_edges.index_select(0, row_ids).clone(),
            anchor_nodes=state.anchor_nodes.index_select(0, row_ids).clone(),
            rollout_to_graph=state.rollout_to_graph.index_select(0, row_ids),
            expand_budget=int(state.expand_budget),
        )
    else:
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        node_batch = batch.batch.to(device=device, dtype=torch.long)
        static_ids = row_ids
        node_belongs = node_batch.view(1, -1).eq(static_ids.view(-1, 1))
        edge_belongs = edge_batch.view(1, -1).eq(static_ids.view(-1, 1))

        anchors = torch.zeros_like(state.active_nodes, dtype=torch.bool, device=device)
        anchor_ids = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        valid_anchors = anchor_ids.ge(0) & anchor_ids.lt(anchors.numel())
        if bool(valid_anchors.any()):
            anchors[anchor_ids[valid_anchors]] = True
        child = RolloutState(
            active_nodes=(
                state.active_nodes.view(1, -1).expand_as(node_belongs) & node_belongs
            ).clone(),
            active_edges=(
                state.active_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs
            ).clone(),
            root_edges=(
                state.root_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs
            ).clone(),
            anchor_nodes=(
                anchors.view(1, -1).expand_as(node_belongs) & node_belongs
            ).clone(),
            rollout_to_graph=static_ids,
            expand_budget=int(state.expand_budget),
        )

    child.apply_expansion(
        rollout_ids=torch.arange(num_candidates, dtype=torch.long, device=device),
        chosen_edges=edge_ids,
        edge_index=batch.edge_index,
    )
    return child


__all__ = [
    "LocalImprovementAuxiliary",
    "LocalImprovementConfig",
    "evaluate_candidate_child_rewards",
]
