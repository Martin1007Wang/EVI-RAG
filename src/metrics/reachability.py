from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torchmetrics import Metric


class AnswerReachability(Metric):
    full_state_update: bool = False

    def __init__(self, k_values: Optional[Sequence[int]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k_values = self._normalize_k_values(k_values)
        for k in self.k_values:
            self.add_state(f"hits_at_{k}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _normalize_k_values(raw_values: Optional[Sequence[int]]) -> List[int]:
        if raw_values is None:
            return []
        normalized: List[int] = []
        seen = set()
        for item in raw_values:
            try:
                k = int(item)
            except (TypeError, ValueError):
                continue
            if k <= 0 or k in seen:
                continue
            normalized.append(k)
            seen.add(k)
        normalized.sort()
        return normalized

    @staticmethod
    def _get_attr(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    @classmethod
    def _require_attr(cls, obj: Any, name: str, error: str) -> Any:
        value = cls._get_attr(obj, name)
        if value is None:
            raise ValueError(error)
        return value

    @staticmethod
    def _infer_num_graphs(node_ptr: torch.Tensor, num_graphs: Optional[int]) -> int:
        if num_graphs is None:
            return int(node_ptr.numel() - 1)
        return int(num_graphs)

    @staticmethod
    def _resolve_edge_ptr(slice_dict: Any, num_graphs: int) -> Optional[List[int]]:
        if isinstance(slice_dict, dict) and "edge_index" in slice_dict:
            candidate = torch.as_tensor(slice_dict.get("edge_index"), dtype=torch.long).view(-1)
            if candidate.numel() == num_graphs + 1:
                return candidate.tolist()
        return None

    @staticmethod
    def _resolve_query_ids(
        query_ids: Optional[torch.Tensor],
        scores_all: torch.Tensor,
        edge_ptr: Optional[List[int]],
    ) -> Optional[torch.Tensor]:
        if edge_ptr is not None:
            return None
        if query_ids is None:
            raise ValueError("query_ids required for reachability metrics when edge ptr is unavailable.")
        query_ids_all = query_ids.detach().view(-1)
        if query_ids_all.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids_all.shape} vs {scores_all.shape}")
        return query_ids_all

    def _resolve_local_indices(
        self,
        batch: Any,
        slice_dict: Any,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_local_indices_all = self._get_attr(batch, "q_local_indices")
        a_local_indices_all = self._get_attr(batch, "a_local_indices")
        q_ptr_raw = self._get_attr(batch, "q_local_indices_ptr")
        a_ptr_raw = self._get_attr(batch, "a_local_indices_ptr")
        if q_ptr_raw is None and isinstance(slice_dict, dict):
            q_ptr_raw = slice_dict.get("q_local_indices")
        if a_ptr_raw is None and isinstance(slice_dict, dict):
            a_ptr_raw = slice_dict.get("a_local_indices")
        if q_local_indices_all is None or a_local_indices_all is None or q_ptr_raw is None or a_ptr_raw is None:
            raise ValueError("Batch missing q_local_indices/a_local_indices required for reachability metrics.")

        q_ptr = torch.as_tensor(q_ptr_raw, dtype=torch.long, device=device).view(-1)
        a_ptr = torch.as_tensor(a_ptr_raw, dtype=torch.long, device=device).view(-1)
        q_local_indices = torch.as_tensor(q_local_indices_all, dtype=torch.long, device=device).view(-1)
        a_local_indices = torch.as_tensor(a_local_indices_all, dtype=torch.long, device=device).view(-1)
        if q_ptr.numel() != num_graphs + 1:
            raise ValueError(f"q_local_indices_ptr length mismatch: {q_ptr.numel()} vs expected {num_graphs + 1}")
        if a_ptr.numel() != num_graphs + 1:
            raise ValueError(f"a_local_indices_ptr length mismatch: {a_ptr.numel()} vs expected {num_graphs + 1}")
        return q_ptr, a_ptr, q_local_indices, a_local_indices

    @staticmethod
    def _select_graph_edges(
        gid: int,
        *,
        edge_ptr: Optional[List[int]],
        scores_all: torch.Tensor,
        edge_index_all: torch.Tensor,
        query_ids_all: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if edge_ptr is not None:
            start = int(edge_ptr[gid])
            end = int(edge_ptr[gid + 1])
            if end <= start:
                return None, None
            return scores_all[start:end], edge_index_all[:, start:end]
        if query_ids_all is None:
            return None, None
        mask = query_ids_all == gid
        if not bool(mask.any().item()):
            return None, None
        return scores_all[mask], edge_index_all[:, mask]

    def _compute_graph_reachability(
        self,
        *,
        gid: int,
        scores: torch.Tensor,
        edge_index_g: torch.Tensor,
        q_ptr: torch.Tensor,
        a_ptr: torch.Tensor,
        q_local_indices: torch.Tensor,
        a_local_indices: torch.Tensor,
        node_ptr: torch.Tensor,
        max_reach_k: int,
        k_values_sorted: List[int],
    ) -> Optional[Dict[int, bool]]:
        num_edges = int(scores.numel())
        if num_edges <= 0:
            return None
        k_top = min(num_edges, max_reach_k)
        top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True).indices

        q_start = int(q_ptr[gid])
        q_end = int(q_ptr[gid + 1])
        a_start = int(a_ptr[gid])
        a_end = int(a_ptr[gid + 1])
        q_nodes = q_local_indices[q_start:q_end]
        a_nodes = a_local_indices[a_start:a_end]
        if q_nodes.numel() == 0 or a_nodes.numel() == 0:
            return None

        node_start = int(node_ptr[gid].item())
        node_end = int(node_ptr[gid + 1].item())
        num_nodes = max(0, node_end - node_start)
        if num_nodes <= 0:
            return None
        q_local = q_nodes[(q_nodes >= node_start) & (q_nodes < node_end)] - node_start
        a_local = a_nodes[(a_nodes >= node_start) & (a_nodes < node_end)] - node_start
        if q_local.numel() == 0 or a_local.numel() == 0:
            return None

        edge_index_local = edge_index_g - node_start
        reach_map = self._compute_reachability_at_k(
            edge_index=edge_index_local,
            top_idx=top_idx,
            start_nodes=q_local,
            answer_nodes=a_local,
            num_nodes=num_nodes,
            k_values=k_values_sorted,
        )
        if not reach_map:
            return None
        return reach_map

    def _accumulate_hits(
        self,
        *,
        num_graphs: int,
        edge_ptr: Optional[List[int]],
        scores_all: torch.Tensor,
        edge_index_all: torch.Tensor,
        query_ids_all: Optional[torch.Tensor],
        q_ptr: torch.Tensor,
        a_ptr: torch.Tensor,
        q_local_indices: torch.Tensor,
        a_local_indices: torch.Tensor,
        node_ptr: torch.Tensor,
        max_reach_k: int,
    ) -> tuple[Dict[int, float], float]:
        hits = {int(k): 0.0 for k in self.k_values}
        valid = 0.0
        k_values_sorted = sorted(self.k_values)

        for gid in range(num_graphs):
            scores, edge_index_g = self._select_graph_edges(
                gid,
                edge_ptr=edge_ptr,
                scores_all=scores_all,
                edge_index_all=edge_index_all,
                query_ids_all=query_ids_all,
            )
            if scores is None or edge_index_g is None:
                continue
            reach_map = self._compute_graph_reachability(
                gid=gid,
                scores=scores,
                edge_index_g=edge_index_g,
                q_ptr=q_ptr,
                a_ptr=a_ptr,
                q_local_indices=q_local_indices,
                a_local_indices=a_local_indices,
                node_ptr=node_ptr,
                max_reach_k=max_reach_k,
                k_values_sorted=k_values_sorted,
            )
            if not reach_map:
                continue
            valid += 1.0
            for k, reachable in reach_map.items():
                if reachable:
                    hits[int(k)] += 1.0
        return hits, valid

    def update(
        self,
        preds: torch.Tensor,
        batch: Any,
        query_ids: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> None:
        if not self.k_values:
            return
        scores_all = preds.detach().view(-1)
        if scores_all.numel() == 0:
            return

        edge_index_all = self._require_attr(
            batch,
            "edge_index",
            "Batch missing edge_index required for reachability metrics.",
        )
        node_ptr = self._require_attr(batch, "ptr", "Batch missing ptr required for reachability metrics.")
        slice_dict = self._get_attr(batch, "_slice_dict")
        num_graphs = self._infer_num_graphs(node_ptr, num_graphs)
        if num_graphs <= 0:
            return

        edge_ptr = self._resolve_edge_ptr(slice_dict, num_graphs)
        query_ids_all = self._resolve_query_ids(query_ids, scores_all, edge_ptr)
        q_ptr, a_ptr, q_local_indices, a_local_indices = self._resolve_local_indices(
            batch,
            slice_dict,
            num_graphs,
            scores_all.device,
        )

        max_reach_k = max(self.k_values)
        if max_reach_k <= 0:
            return

        hits, valid = self._accumulate_hits(
            num_graphs=num_graphs,
            edge_ptr=edge_ptr,
            scores_all=scores_all,
            edge_index_all=edge_index_all,
            query_ids_all=query_ids_all,
            q_ptr=q_ptr,
            a_ptr=a_ptr,
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            node_ptr=node_ptr,
            max_reach_k=max_reach_k,
        )
        if valid <= 0:
            return

        self.total += valid
        for k in self.k_values:
            getattr(self, f"hits_at_{int(k)}").add_(hits[int(k)])

    def compute(self) -> Dict[str, torch.Tensor]:
        if not self.k_values:
            return {}
        denom = self.total.clamp(min=1.0)
        return {
            f"answer/reachability@{k}": getattr(self, f"hits_at_{k}") / denom
            for k in self.k_values
        }

    @staticmethod
    def _uf_init(num_nodes: int) -> tuple[List[int], List[int]]:
        return list(range(num_nodes)), [0] * num_nodes

    @staticmethod
    def _uf_find(parent: List[int], x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @classmethod
    def _uf_union(cls, parent: List[int], rank: List[int], a: int, b: int) -> None:
        pa = cls._uf_find(parent, a)
        pb = cls._uf_find(parent, b)
        if pa == pb:
            return
        if rank[pa] < rank[pb]:
            parent[pa] = pb
        elif rank[pa] > rank[pb]:
            parent[pb] = pa
        else:
            parent[pb] = pa
            rank[pa] += 1

    @classmethod
    def _uf_reachable(cls, parent: List[int], start_nodes: List[int], answer_nodes: List[int]) -> bool:
        roots = {cls._uf_find(parent, s) for s in start_nodes}
        for a in answer_nodes:
            if cls._uf_find(parent, a) in roots:
                return True
        return False

    @staticmethod
    def _compute_reachability_at_k(
        *,
        edge_index: torch.Tensor,
        top_idx: torch.Tensor,
        start_nodes: torch.Tensor,
        answer_nodes: torch.Tensor,
        num_nodes: int,
        k_values: Iterable[int],
    ) -> Dict[int, bool]:
        if num_nodes <= 0:
            return {}
        k_values = [int(k) for k in k_values if int(k) > 0]
        if not k_values:
            return {}
        start_nodes = start_nodes.view(-1)
        answer_nodes = answer_nodes.view(-1)
        if start_nodes.numel() == 0 or answer_nodes.numel() == 0:
            return {int(k): False for k in k_values}

        k_top = min(int(top_idx.numel()), max(k_values))
        if k_top <= 0:
            return {int(k): False for k in k_values}

        edge_index_top = edge_index[:, top_idx[:k_top]].detach().cpu()
        start_nodes_cpu = start_nodes.detach().cpu().view(-1).tolist()
        answer_nodes_cpu = answer_nodes.detach().cpu().view(-1).tolist()
        parent, rank = AnswerReachability._uf_init(num_nodes)

        k_check = sorted({min(int(k), k_top) for k in k_values})
        reach_map: Dict[int, bool] = {}
        next_idx = 0
        for idx in range(k_top):
            u = int(edge_index_top[0, idx].item())
            v = int(edge_index_top[1, idx].item())
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                AnswerReachability._uf_union(parent, rank, u, v)
            while next_idx < len(k_check) and idx + 1 >= k_check[next_idx]:
                reach_map[k_check[next_idx]] = AnswerReachability._uf_reachable(
                    parent,
                    start_nodes_cpu,
                    answer_nodes_cpu,
                )
                next_idx += 1
        while next_idx < len(k_check):
            reach_map[k_check[next_idx]] = AnswerReachability._uf_reachable(
                parent,
                start_nodes_cpu,
                answer_nodes_cpu,
            )
            next_idx += 1

        return {int(k): reach_map[min(int(k), k_top)] for k in k_values}
