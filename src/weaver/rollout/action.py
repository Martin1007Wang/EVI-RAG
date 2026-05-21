from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.policy import PolicyOutput

STOP_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class StepAction:
    row_ids: torch.Tensor
    edge_ids: torch.Tensor
    policy_log_prob: torch.Tensor
    behavior_log_prob: torch.Tensor
    forced: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.row_ids.numel())
        if int(self.edge_ids.numel()) != count:
            raise ValueError("edge_ids must match row_ids length.")
        if int(self.policy_log_prob.numel()) != count:
            raise ValueError("policy_log_prob must match row_ids length.")
        if int(self.behavior_log_prob.numel()) != count:
            raise ValueError("behavior_log_prob must match row_ids length.")
        if int(self.forced.numel()) != count:
            raise ValueError("forced must match row_ids length.")
        invalid_negative = self.edge_ids.lt(STOP_EDGE_ID)
        if bool(invalid_negative.any()):
            raise ValueError("Only STOP_EDGE_ID may be negative.")

    @property
    def stop_mask(self) -> torch.Tensor:
        return self.edge_ids.eq(STOP_EDGE_ID)

    @property
    def expand_mask(self) -> torch.Tensor:
        return self.edge_ids.ge(0)

    @property
    def stop_rows(self) -> torch.Tensor:
        return self.row_ids[self.stop_mask]

    @property
    def expand_rows(self) -> torch.Tensor:
        return self.row_ids[self.expand_mask]

    @property
    def expand_edge_ids(self) -> torch.Tensor:
        return self.edge_ids[self.expand_mask]

    @property
    def expand_log_prob(self) -> torch.Tensor:
        return self.policy_log_prob[self.expand_mask]

    @classmethod
    def forced_stop(
        cls,
        *,
        rows: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> StepAction:
        rows = rows.to(
            device=device or rows.device,
            dtype=torch.long,
        ).view(-1)
        return cls(
            row_ids=rows,
            edge_ids=torch.full(
                (rows.numel(),),
                STOP_EDGE_ID,
                dtype=torch.long,
                device=rows.device,
            ),
            policy_log_prob=torch.zeros(
                rows.numel(),
                dtype=dtype,
                device=rows.device,
            ),
            behavior_log_prob=torch.zeros(
                rows.numel(),
                dtype=dtype,
                device=rows.device,
            ),
            forced=torch.ones(
                rows.numel(),
                dtype=torch.bool,
                device=rows.device,
            ),
        )

    @classmethod
    def concat(cls, actions: Sequence[StepAction]) -> StepAction:
        non_empty = [action for action in actions if action.row_ids.numel() > 0]
        if not non_empty:
            device = actions[0].row_ids.device if actions else torch.device("cpu")
            return cls(
                row_ids=torch.empty(0, dtype=torch.long, device=device),
                edge_ids=torch.empty(0, dtype=torch.long, device=device),
                policy_log_prob=torch.empty(0, dtype=torch.float32, device=device),
                behavior_log_prob=torch.empty(0, dtype=torch.float32, device=device),
                forced=torch.empty(0, dtype=torch.bool, device=device),
            )

        row_ids = torch.cat([action.row_ids for action in non_empty], dim=0)
        edge_ids = torch.cat([action.edge_ids for action in non_empty], dim=0)
        policy_log_prob = torch.cat([action.policy_log_prob for action in non_empty], dim=0)
        behavior_log_prob = torch.cat([action.behavior_log_prob for action in non_empty], dim=0)
        forced = torch.cat([action.forced for action in non_empty], dim=0)
        order = torch.argsort(row_ids)
        return cls(
            row_ids=row_ids.index_select(0, order),
            edge_ids=edge_ids.index_select(0, order),
            policy_log_prob=policy_log_prob.index_select(0, order),
            behavior_log_prob=behavior_log_prob.index_select(0, order),
            forced=forced.index_select(0, order),
        )


def sample_step(
    *,
    policy_out: PolicyOutput,
    rows: torch.Tensor,
    temperature: float = 1.0,
) -> StepAction:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    rows = rows.to(
        device=policy_out.stop_logit.device,
        dtype=torch.long,
    ).view(-1)
    if rows.numel() == 0:
        return StepAction(
            row_ids=rows,
            edge_ids=rows.new_empty((0,)),
            policy_log_prob=policy_out.stop_logit.new_empty((0,)).float(),
            behavior_log_prob=policy_out.stop_logit.new_empty((0,)).float(),
            forced=torch.zeros(0, dtype=torch.bool, device=rows.device),
        )

    picked_edge_ids = policy_out.sample(
        rows=rows,
        temperature=float(temperature),
    )
    policy_log_prob = policy_out.gather_log_prob(
        row_ids=rows,
        edge_ids=picked_edge_ids,
        temperature=1.0,
    ).float()
    behavior_log_prob = policy_out.gather_log_prob(
        row_ids=rows,
        edge_ids=picked_edge_ids,
        temperature=float(temperature),
    ).float()
    return StepAction(
        row_ids=rows,
        edge_ids=picked_edge_ids,
        policy_log_prob=policy_log_prob,
        behavior_log_prob=behavior_log_prob,
        forced=torch.zeros(rows.numel(), dtype=torch.bool, device=rows.device),
    )


__all__ = [
    "STOP_EDGE_ID",
    "StepAction",
    "sample_step",
]
