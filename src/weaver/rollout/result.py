from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RolloutResult:
    source_graph_id: torch.Tensor
    selected_edge_ids: torch.Tensor
    policy_action_log_prob: torch.Tensor
    behavior_action_log_prob: torch.Tensor
    terminal_step: torch.Tensor
    forced_terminal: torch.Tensor
    expand_budget: int

    @property
    def num_rollouts(self) -> int:
        return int(self.source_graph_id.numel())

    @property
    def max_steps(self) -> int:
        return int(self.selected_edge_ids.size(1))

    @property
    def device(self) -> torch.device:
        return self.selected_edge_ids.device

    @property
    def valid_mask(self) -> torch.Tensor:
        step = torch.arange(self.max_steps, device=self.device)
        return step.unsqueeze(0) <= self.terminal_step.unsqueeze(1)

    @property
    def terminal_mask(self) -> torch.Tensor:
        step = torch.arange(self.max_steps, device=self.device)
        return step.unsqueeze(0) == self.terminal_step.unsqueeze(1)

    @property
    def expand_mask(self) -> torch.Tensor:
        return self.valid_mask & self.selected_edge_ids.ge(0)

    @property
    def forced_terminal_mask(self) -> torch.Tensor:
        return self.terminal_mask & self.forced_terminal.unsqueeze(1)

    @property
    def policy_trajectory_log_prob(self) -> torch.Tensor:
        return self.policy_action_log_prob.masked_fill(~self.valid_mask, 0.0).sum(dim=1)

    @property
    def behavior_trajectory_log_prob(self) -> torch.Tensor:
        return self.behavior_action_log_prob.masked_fill(~self.valid_mask, 0.0).sum(dim=1)

    def select_rows(self, rows: torch.Tensor) -> RolloutResult:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return RolloutResult(
            source_graph_id=self.source_graph_id.index_select(0, rows),
            selected_edge_ids=self.selected_edge_ids.index_select(0, rows),
            policy_action_log_prob=self.policy_action_log_prob.index_select(0, rows),
            behavior_action_log_prob=self.behavior_action_log_prob.index_select(0, rows),
            terminal_step=self.terminal_step.index_select(0, rows),
            forced_terminal=self.forced_terminal.index_select(0, rows),
            expand_budget=self.expand_budget,
        )

    @classmethod
    def concat(cls, rollouts: list[RolloutResult]) -> RolloutResult:
        first = rollouts[0]
        return cls(
            source_graph_id=torch.cat([rollout.source_graph_id for rollout in rollouts], dim=0),
            selected_edge_ids=torch.cat([rollout.selected_edge_ids for rollout in rollouts], dim=0),
            policy_action_log_prob=torch.cat([rollout.policy_action_log_prob for rollout in rollouts], dim=0),
            behavior_action_log_prob=torch.cat([rollout.behavior_action_log_prob for rollout in rollouts], dim=0),
            terminal_step=torch.cat([rollout.terminal_step for rollout in rollouts], dim=0),
            forced_terminal=torch.cat([rollout.forced_terminal for rollout in rollouts], dim=0),
            expand_budget=first.expand_budget,
        )


__all__ = [
    "RolloutResult",
]
