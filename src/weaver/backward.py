from __future__ import annotations

import torch
from torch import nn

from src.weaver.rollout.engine import RolloutContext
from src.weaver.state import State, assert_anchor_connected_state, derive_node_mask


class BackwardKernel(nn.Module):
    def log_prob(
        self,
        *,
        parent_state: State,
        child_state: State,
        action_edge_ids: torch.Tensor,
        context: RolloutContext,
    ) -> torch.Tensor:
        raise NotImplementedError


class DeterministicBackwardKernel(BackwardKernel):
    """
    Deterministic backward kernel.

    Use this only when the state space is ordered, i.e. each child state has a
    unique predecessor for the sampled transition.

    Do not use this for unordered evidence-subgraph states.
    """

    def log_prob(
        self,
        *,
        parent_state: State,
        child_state: State,
        action_edge_ids: torch.Tensor,
        context: RolloutContext,
    ) -> torch.Tensor:
        del parent_state, child_state, context
        return torch.zeros_like(action_edge_ids.view(-1), dtype=torch.float32)


class UniformSubgraphBackwardKernel(BackwardKernel):
    """
    Uniform backward kernel for unordered evidence-subgraph states.

    The state is an unordered selected-edge set. Therefore the same child
    subgraph may have multiple legal parents.

    For a transition x --e--> y:

        B(y) = {
            e' in selected_edges(y):
                remove(y, e') is anchor-connected
                and e' is in frontier(remove(y, e'))
        }

        P_B(x | y) = 1 / |B(y)|

    This implementation also verifies that:

        remove(y, e) == x

    If this is false, the transition is not a valid forward/backward pair.
    """

    def log_prob(
        self,
        *,
        parent_state: State,
        child_state: State,
        action_edge_ids: torch.Tensor,
        context: RolloutContext,
    ) -> torch.Tensor:
        device = child_state.device
        actions = action_edge_ids.to(device=device, dtype=torch.long).view(-1)

        if parent_state.num_rollouts != child_state.num_rollouts:
            raise ValueError(
                "parent_state and child_state must have the same number of rows, "
                f"got parent={parent_state.num_rollouts}, child={child_state.num_rollouts}."
            )

        if actions.numel() != child_state.num_rollouts:
            raise ValueError("action_edge_ids must have one edge per transition, " f"got actions={actions.numel()}, rows={child_state.num_rollouts}.")

        counts = torch.empty(
            child_state.num_rollouts,
            dtype=torch.long,
            device=device,
        )

        for row in range(child_state.num_rollouts):
            row_index = torch.tensor([row], dtype=torch.long, device=device)

            parent = parent_state.select_rows(row_index)
            child = child_state.select_rows(row_index)
            action_edge_id = int(actions[row].item())

            valid_removable_edges = self.valid_removable_edges(
                child=child,
                context=context,
            )

            if valid_removable_edges.numel() == 0:
                raise ValueError("Child state has no valid removable parent under " "UniformSubgraphBackwardKernel.")

            if not torch.any(valid_removable_edges.eq(action_edge_id)):
                raise ValueError(
                    "The transition action is not a valid backward-removable edge. "
                    f"action_edge_id={action_edge_id}, "
                    f"valid_edges={valid_removable_edges.detach().cpu().tolist()}."
                )

            reconstructed_parent = self.remove_edge(
                child=child,
                edge_id=action_edge_id,
                context=context,
            )

            if not same_state(reconstructed_parent, parent):
                raise ValueError("Backward transition mismatch: remove(child, action_edge_id) " "does not equal parent_state.")

            counts[row] = valid_removable_edges.numel()

        return -counts.to(dtype=torch.float32).log()

    def valid_removable_edges(
        self,
        *,
        child: State,
        context: RolloutContext,
    ) -> torch.Tensor:
        device = child.device
        selected_edges = child.edge_mask[0].nonzero(as_tuple=False).view(-1)

        if selected_edges.numel() == 0:
            return selected_edges

        valid_edges: list[int] = []

        for edge_id_tensor in selected_edges:
            edge_id = int(edge_id_tensor.item())

            candidate_parent = self.remove_edge(
                child=child,
                edge_id=edge_id,
                context=context,
            )

            try:
                assert_anchor_connected_state(
                    state=candidate_parent,
                    edge_index=edge_index_from_context(context),
                )
            except AssertionError:
                continue

            if frontier_contains(
                context=context,
                state=candidate_parent,
                edge_id=edge_id,
            ):
                valid_edges.append(edge_id)

        if not valid_edges:
            return torch.empty(0, dtype=torch.long, device=device)

        return torch.tensor(valid_edges, dtype=torch.long, device=device)

    @staticmethod
    def remove_edge(
        *,
        child: State,
        edge_id: int,
        context: RolloutContext,
    ) -> State:
        edge_mask = child.edge_mask.clone()
        edge_mask[0, int(edge_id)] = False

        raw_parent = State(
            node_mask=torch.zeros_like(child.node_mask),
            edge_mask=edge_mask,
            max_budget_by_row=child.max_budget_by_row.clone(),
            row_to_graph=child.row_to_graph.clone(),
        )

        node_mask = derive_node_mask(
            state=raw_parent,
            edge_index=edge_index_from_context(context),
        )

        return State(
            node_mask=node_mask,
            edge_mask=edge_mask,
            max_budget_by_row=child.max_budget_by_row.clone(),
            row_to_graph=child.row_to_graph.clone(),
        )


def frontier_contains(
    *,
    context: RolloutContext,
    state: State,
    edge_id: int,
) -> bool:
    return bool(
        context.frontier_builder.contains(
            state=state,
            row=0,
            edge_id=int(edge_id),
        )
    )


def edge_index_from_context(
    context: RolloutContext,
) -> object:
    return context.frontier_builder.edge_index


def same_state(
    left: State,
    right: State,
) -> bool:
    return (
        torch.equal(left.edge_mask, right.edge_mask)
        and torch.equal(left.max_budget_by_row, right.max_budget_by_row)
        and torch.equal(left.row_to_graph, right.row_to_graph)
    )


__all__ = [
    "BackwardKernel",
    "DeterministicBackwardKernel",
    "UniformSubgraphBackwardKernel",
]
