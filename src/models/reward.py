from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter_max

from src.data.schema import RetrievalBatch
from src.models.state import State
from src.utils.nn_utils import cosine_scores
from src.utils.reward_utils import build_anchor_induced_edge_mask, per_graph_mask_count


class RewardModel(nn.Module):
    """Terminal log-reward for KG subgraph retrieval (subgraph MDP only).

    Primary objective — F1-based recall/precision balance
    ------------------------------------------------------
    For KGQA the ground-truth evaluation metric is F1 over answer entities.
    A pure-recall objective incentivises the model to expand the entire graph
    (recall = 1 trivially), so we use F1 as the primary signal:

        F1(x) = 2 · precision(x) · recall(x) / (precision(x) + recall(x))

    where
        recall(x)    = |active_gold| / |gold|
        precision(x) = |active_gold| / |active_nodes - (anchor_nodes - gold_nodes)|

    and active_gold = active answer nodes in the retrieved sub-graph.

    This precision definition excludes fixed non-gold anchors from the predicted
    positive set, but still counts gold anchors in the denominator. That keeps
    root-hit trajectories well-defined when anchor and target sets overlap, and
    guarantees precision <= 1.

    log R(x) is then:

        log R(x) = log F1(x)                                           if F1(x) > 0
        log R(x) = log_r_min + edge_bonus_scale * best_edge_relation_match(x) otherwise

    ``log_r_min`` is part of the terminal reward definition itself: it keeps
    zero-F1 trajectories finite and trainable instead of collapsing to -inf.

    The zero-F1 edge bonus weakly ranks failed trajectories by
    the best query-relation match among the agent's selected non-root edges.
    This is a set-level terminal tie-breaker: it rewards committed evidence in
    the final subgraph, not unchosen frontier opportunities or path order.

    Parameters
    ----------
    log_r_min : float
        Finite floor for zero-F1 trajectories. Must be < 0.
    zero_f1_edge_bonus_scale : float
        Weight of the selected-edge relation bonus when F1 = 0.
        Set to 0.0 to recover a hard floor with no ranking among failures.
    """

    def __init__(
        self,
        log_r_min: float = -5.0,
        zero_f1_edge_bonus_scale: float = 0.25,
    ) -> None:
        super().__init__()

        if log_r_min >= 0.0:
            raise ValueError(f"log_r_min must be < 0 (a penalty floor), got {log_r_min}.")
        if zero_f1_edge_bonus_scale < 0.0:
            raise ValueError(
                "zero_f1_edge_bonus_scale must be >= 0, got "
                f"{zero_f1_edge_bonus_scale}."
            )

        self.log_r_min = float(log_r_min)
        self.zero_f1_edge_bonus_scale = float(zero_f1_edge_bonus_scale)

    # ------------------------------------------------------------------
    # Primary terminal reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        retrieval_batch: RetrievalBatch,
        active_nodes: torch.Tensor,   # (N_nodes,) bool
        active_edges: torch.Tensor,   # (N_edges,) bool
        state: State | None = None,
    ) -> torch.Tensor:
        """Compute per-graph terminal log R(x).

        Returns
        -------
        log_reward : Tensor, shape (num_graphs,), dtype float32
            log R(x) for each graph. Guaranteed: values >= log_r_min.
        """
        num_graphs  = retrieval_batch.num_graphs
        node_batch  = retrieval_batch.batch
        dtype       = torch.float32
        device      = active_nodes.device

        target_mask = _get_train_target_mask(retrieval_batch)  # (N_nodes,) bool
        anchor_mask  = retrieval_batch.is_anchor_mask   # (N_nodes,) bool

        active_gold  = active_nodes & target_mask

        # ── recall ──────────────────────────────────────────────────────
        hits_per_graph = per_graph_mask_count(active_gold,  node_batch, num_graphs, dtype=dtype)
        gold_per_graph = per_graph_mask_count(target_mask,  node_batch, num_graphs, dtype=dtype)

        recall = torch.zeros(num_graphs, dtype=dtype, device=device)
        valid_gold = gold_per_graph.gt(0.0)
        recall[valid_gold] = hits_per_graph[valid_gold] / gold_per_graph[valid_gold]

        # ── precision ────────────────────────────────────────────────────
        # Predicted positives = active nodes after excluding only non-gold anchors:
        #   active_nodes \ (anchor_nodes \ gold_nodes)
        # This preserves the original intent of not penalising fixed context
        # anchors, while still treating gold anchors as true retrieved positives.
        effective_retrieved = active_nodes & (~anchor_mask | target_mask)
        retrieved_per_graph = per_graph_mask_count(
            effective_retrieved, node_batch, num_graphs, dtype=dtype
        )
        # precision = hits / effective_retrieved; if that set is empty, precision = 0
        precision = torch.zeros(num_graphs, dtype=dtype, device=device)
        valid_retrieved = retrieved_per_graph.gt(0.0)
        precision[valid_retrieved] = (
            hits_per_graph[valid_retrieved] / retrieved_per_graph[valid_retrieved]
        )

        # ── F1 ───────────────────────────────────────────────────────────
        denom = (precision + recall).clamp_min(torch.finfo(dtype).eps)
        f1    = torch.where(
            (precision + recall).gt(0.0),
            2.0 * precision * recall / denom,
            torch.zeros_like(recall),
        )

        # ── log R ────────────────────────────────────────────────────────
        log_reward    = torch.full((num_graphs,), self.log_r_min, dtype=dtype, device=device)
        positive_f1   = f1.gt(0.0)
        log_reward[positive_f1] = f1[positive_f1].log()

        # Zero-F1 fallback: only computed for failed graphs (avoids wasted work)
        need_fallback = ~positive_f1
        if bool(need_fallback.any()):
            zero_f1_edge_bonus = self._zero_f1_edge_bonus(
                retrieval_batch,
                active_edges,
                state=state,
                graph_mask=need_fallback,
            )
            log_reward[need_fallback] = (
                self.log_r_min + zero_f1_edge_bonus[need_fallback]
            )

        return log_reward.clamp(min=self.log_r_min).to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Zero-F1 edge bonus (vectorised)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _zero_f1_edge_bonus(
        self,
        retrieval_batch: RetrievalBatch,
        active_edges: torch.Tensor,
        *,
        state: State | None = None,
        graph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph query-relation match over selected non-root edges.

        The score is a weak set-level tie-breaker for zero-F1 trajectories:
        we look only at edges the agent actually selected into the final
        subgraph, exclude root edges that were present from the start, and take
        the best query-relation cosine per graph.

        Parameters
        ----------
        graph_mask : (num_graphs,) bool | None
            If provided, only graphs where graph_mask=True are computed;
            the rest receive 0.  Pass ``~positive_f1`` to avoid wasted work.

        Returns
        -------
        bonus : Tensor, shape (num_graphs,), dtype float32, values in
            [0, zero_f1_edge_bonus_scale]
        """
        num_graphs = retrieval_batch.num_graphs
        device = active_edges.device

        bonus = torch.zeros(num_graphs, dtype=torch.float32, device=device)

        if self.zero_f1_edge_bonus_scale == 0.0:
            return bonus

        edge_batch = retrieval_batch.edge_batch
        root_active_edges = (
            state.root_active_edges
            if state is not None
            else build_anchor_induced_edge_mask(
                retrieval_batch.edge_index, retrieval_batch.is_anchor_mask
            )
        )

        if graph_mask is not None:
            relevant_edges = graph_mask[edge_batch]
        else:
            relevant_edges = torch.ones(active_edges.shape[0], dtype=torch.bool, device=device)

        selected_non_root_edges = active_edges & ~root_active_edges & relevant_edges

        if not bool(selected_non_root_edges.any()):
            return bonus

        selected_edge_ids = torch.nonzero(selected_non_root_edges, as_tuple=False).view(-1)
        selected_batch = edge_batch[selected_edge_ids]
        query_h = retrieval_batch.question_emb.index_select(0, selected_batch)
        rel_h = retrieval_batch.relation_tokens.index_select(0, selected_edge_ids)
        per_edge_sim = cosine_scores(query_h, rel_h).clamp(min=0.0, max=1.0)

        sim_max, _ = scatter_max(
            per_edge_sim,
            selected_batch,
            dim=0,
            dim_size=num_graphs,
        )  # (G,) — scatter_max fills missing entries with 0

        bonus = self.zero_f1_edge_bonus_scale * sim_max.to(dtype=torch.float32)
        return bonus


__all__ = ["RewardModel"]


def _get_train_target_mask(retrieval_batch: RetrievalBatch) -> torch.Tensor:
    return retrieval_batch.train_target_mask
