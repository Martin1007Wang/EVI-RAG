from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.state import State

SRC_UNKNOWN = -1
SRC_POLICY = 0
SRC_REPLAY = 1


@dataclass(frozen=True, slots=True)
class SampleMeta:
    trajectory_ids: torch.Tensor
    step_ids: torch.Tensor
    source_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.trajectory_ids is None:
            raise ValueError("trajectory_ids cannot be None.")
        if self.step_ids is None:
            raise ValueError("step_ids cannot be None.")
        if self.source_ids is None:
            raise ValueError("source_ids cannot be None.")
        count = int(self.trajectory_ids.numel())
        if int(self.step_ids.numel()) != count:
            raise ValueError("step_ids must match trajectory_ids length.")
        if int(self.source_ids.numel()) != count:
            raise ValueError("source_ids must match trajectory_ids length.")

    @property
    def num_items(self) -> int:
        return int(self.trajectory_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.trajectory_ids.device

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> SampleMeta:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return SampleMeta(
            trajectory_ids=self.trajectory_ids.index_select(0, rows),
            step_ids=self.step_ids.index_select(0, rows),
            source_ids=self.source_ids.index_select(0, rows),
        )

    def with_source_id(
        self,
        source_id: int,
    ) -> SampleMeta:
        return SampleMeta(
            trajectory_ids=self.trajectory_ids,
            step_ids=self.step_ids,
            source_ids=torch.full_like(
                self.source_ids,
                int(source_id),
            ),
        )

    @classmethod
    def empty(
        cls,
        device: torch.device,
    ) -> SampleMeta:
        empty = _empty_long(device)
        return cls(
            trajectory_ids=empty,
            step_ids=empty,
            source_ids=empty,
        )

    @classmethod
    def concat(
        cls,
        metas: Sequence[SampleMeta],
    ) -> SampleMeta:
        if not metas:
            raise ValueError("Cannot concatenate an empty SampleMeta sequence.")
        if len(metas) == 1:
            return metas[0]
        return cls(
            trajectory_ids=torch.cat([meta.trajectory_ids for meta in metas], dim=0),
            step_ids=torch.cat([meta.step_ids for meta in metas], dim=0),
            source_ids=torch.cat([meta.source_ids for meta in metas], dim=0),
        )


@dataclass(frozen=True, slots=True)
class ExpansionBatch:
    parent: State
    child: State
    edge_ids: torch.Tensor
    meta: SampleMeta

    @property
    def num_items(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.edge_ids.device

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> ExpansionBatch:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return ExpansionBatch(
            parent=self.parent.select_rows(rows),
            child=self.child.select_rows(rows),
            edge_ids=self.edge_ids.index_select(0, rows),
            meta=self.meta.select_rows(rows),
        )

    @classmethod
    def empty_like(
        cls,
        *,
        graph_like: State,
    ) -> ExpansionBatch:
        rows = _empty_long(graph_like.device)
        empty_state = graph_like.select_rows(rows)
        return cls(
            parent=empty_state,
            child=empty_state,
            edge_ids=rows,
            meta=SampleMeta.empty(graph_like.device),
        )

    @classmethod
    def concat(
        cls,
        batches: Sequence[ExpansionBatch],
    ) -> ExpansionBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty ExpansionBatch sequence.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            parent=State.concat([batch.parent for batch in batches]),
            child=State.concat([batch.child for batch in batches]),
            edge_ids=torch.cat([batch.edge_ids for batch in batches], dim=0),
            meta=SampleMeta.concat([batch.meta for batch in batches]),
        )


@dataclass(frozen=True, slots=True)
class TerminalBatch:
    state: State
    meta: SampleMeta
    forced_terminal: torch.Tensor

    @property
    def num_items(self) -> int:
        return int(self.meta.num_items)

    @property
    def device(self) -> torch.device:
        return self.meta.device

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> TerminalBatch:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return TerminalBatch(
            state=self.state.select_rows(rows),
            meta=self.meta.select_rows(rows),
            forced_terminal=self.forced_terminal.index_select(0, rows),
        )

    @classmethod
    def empty_like(
        cls,
        *,
        graph_like: State,
    ) -> TerminalBatch:
        rows = _empty_long(graph_like.device)
        empty_state = graph_like.select_rows(rows)
        return cls(
            state=empty_state,
            meta=SampleMeta.empty(graph_like.device),
            forced_terminal=torch.empty(0, dtype=torch.bool, device=graph_like.device),
        )

    @classmethod
    def concat(
        cls,
        batches: Sequence[TerminalBatch],
    ) -> TerminalBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty TerminalBatch sequence.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            state=State.concat([batch.state for batch in batches]),
            meta=SampleMeta.concat([batch.meta for batch in batches]),
            forced_terminal=torch.cat([batch.forced_terminal for batch in batches], dim=0),
        )


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    expansions: ExpansionBatch
    terminals: TerminalBatch

    @property
    def expand(self) -> ExpansionBatch:
        return self.expansions

    @property
    def stop(self) -> TerminalBatch:
        return self.terminals

    @property
    def num_expansions(self) -> int:
        return int(self.expansions.num_items)

    @property
    def num_terminals(self) -> int:
        return int(self.terminals.num_items)

    @property
    def num_items(self) -> int:
        return int(self.num_expansions + self.num_terminals)

    def with_source_id(
        self,
        source_id: int,
    ) -> TrainingBatch:
        return TrainingBatch(
            expansions=ExpansionBatch(
                parent=self.expansions.parent,
                child=self.expansions.child,
                edge_ids=self.expansions.edge_ids,
                meta=self.expansions.meta.with_source_id(source_id),
            ),
            terminals=TerminalBatch(
                state=self.terminals.state,
                meta=self.terminals.meta.with_source_id(source_id),
                forced_terminal=self.terminals.forced_terminal,
            ),
        )

    @classmethod
    def concat(
        cls,
        batches: Sequence[TrainingBatch],
    ) -> TrainingBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty TrainingBatch sequence.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            expansions=ExpansionBatch.concat([batch.expansions for batch in batches]),
            terminals=TerminalBatch.concat([batch.terminals for batch in batches]),
        )

    @classmethod
    def concat_reindex_trajectories(
        cls,
        batches: Sequence[TrainingBatch],
    ) -> TrainingBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty TrainingBatch sequence.")
        if len(batches) == 1:
            return batches[0]

        out: list[TrainingBatch] = []
        offset = 0

        for batch in batches:
            expansions = batch.expansions
            terminals = batch.terminals

            all_traj_ids = torch.cat(
                [
                    expansions.meta.trajectory_ids,
                    terminals.meta.trajectory_ids,
                ],
                dim=0,
            )
            local_traj_ids = _reindex_trajectory_ids(all_traj_ids)

            num_expand = expansions.num_items
            expand_traj_ids = local_traj_ids[:num_expand]
            terminal_traj_ids = local_traj_ids[num_expand:]
            num_local_traj = (
                int(local_traj_ids.max().item()) + 1
                if local_traj_ids.numel() > 0
                else 0
            )

            out.append(
                TrainingBatch(
                    expansions=ExpansionBatch(
                        parent=expansions.parent,
                        child=expansions.child,
                        edge_ids=expansions.edge_ids,
                        meta=_with_trajectory_ids(
                            expansions.meta,
                            expand_traj_ids + offset,
                        ),
                    ),
                terminals=TerminalBatch(
                    state=terminals.state,
                    meta=_with_trajectory_ids(
                        terminals.meta,
                        terminal_traj_ids + offset,
                    ),
                    forced_terminal=terminals.forced_terminal,
                ),
            )
        )
            offset += num_local_traj

        return cls.concat(out)


def _with_trajectory_ids(
    meta: SampleMeta,
    trajectory_ids: torch.Tensor,
) -> SampleMeta:
    return SampleMeta(
        trajectory_ids=trajectory_ids,
        step_ids=meta.step_ids,
        source_ids=meta.source_ids,
    )


def _reindex_trajectory_ids(
    trajectory_ids: torch.Tensor,
) -> torch.Tensor:
    if trajectory_ids.numel() == 0:
        return trajectory_ids
    _, inverse = torch.unique(
        trajectory_ids,
        sorted=True,
        return_inverse=True,
    )
    return inverse


def _empty_long(
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )


__all__ = [
    "ExpansionBatch",
    "SRC_POLICY",
    "SRC_REPLAY",
    "SRC_UNKNOWN",
    "SampleMeta",
    "TerminalBatch",
    "TrainingBatch",
]
