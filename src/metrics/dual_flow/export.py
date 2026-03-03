from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutExportInputs:
    stop_nodes: torch.Tensor
    num_moves: torch.Tensor
    log_pf_sum: torch.Tensor
    stop_reason: torch.Tensor


@dataclass(frozen=True)
class GraphExportInputs:
    q_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_ptr: torch.Tensor
    node_ptr: torch.Tensor
    node_global_ids: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    sample_ids: Sequence[str]
    num_graphs: int


class DualFlowRolloutExporter:
    @staticmethod
    def build_eval_metrics(
        *,
        prefix: str,
        reward_metrics: dict[str, torch.Tensor],
        rollout: RolloutExportInputs,
        step_offset: int,
        num_graphs: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        reward_mean_scaled = reward_metrics.get(
            "reward/reward_mean_scaled", reward_metrics["reward/reward_mean"]
        )
        num_moves_raw = rollout.num_moves.float()
        num_moves_real = (num_moves_raw - float(step_offset)).clamp(min=0.0)
        return {
            f"{prefix}/hit@1": reward_metrics["reward/hit@1"],
            f"{prefix}/hit@beam": reward_metrics["reward/hit@beam"],
            f"{prefix}/coverage@beam": reward_metrics["reward/coverage@beam"],
            f"{prefix}/reward_mean": reward_metrics["reward/reward_mean"],
            f"{prefix}/reward_mean_scaled": reward_mean_scaled,
            f"{prefix}/num_moves": num_moves_real.mean().detach(),
            f"{prefix}/num_moves_raw": num_moves_raw.mean().detach(),
            f"{prefix}/log_pf_sum": rollout.log_pf_sum.mean().detach(),
            f"{prefix}/num_graphs": torch.tensor(float(num_graphs), device=device),
        }

    @staticmethod
    def build_rollout_probe_metrics(
        *,
        prefix: str,
        reward_metrics: dict[str, torch.Tensor],
        rollout: RolloutExportInputs,
        step_offset: int,
    ) -> dict[str, torch.Tensor]:
        reward_mean_scaled = reward_metrics.get(
            "reward/reward_mean_scaled", reward_metrics["reward/reward_mean"]
        )
        num_moves_raw = rollout.num_moves.float()
        num_moves_real = (num_moves_raw - float(step_offset)).clamp(min=0.0)
        rollout_k = (
            int(rollout.stop_nodes.size(1)) if rollout.stop_nodes.dim() > 1 else 1
        )
        return {
            f"{prefix}/hit@1_rollout": reward_metrics["reward/hit@1"],
            f"{prefix}/hit@k_rollout": reward_metrics["reward/hit@beam"],
            f"{prefix}/coverage@k_rollout": reward_metrics["reward/coverage@beam"],
            f"{prefix}/reward_mean_rollout": reward_metrics["reward/reward_mean"],
            f"{prefix}/reward_mean_scaled_rollout": reward_mean_scaled,
            f"{prefix}/num_moves_rollout": num_moves_real.mean().detach(),
            f"{prefix}/num_moves_raw_rollout": num_moves_raw.mean().detach(),
            f"{prefix}/log_pf_sum_rollout": rollout.log_pf_sum.mean().detach(),
            f"{prefix}/hit@{rollout_k}_rollout": reward_metrics["reward/hit@beam"],
        }

    @staticmethod
    def to_2d_numpy(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        if array.ndim == 1:
            return array[:, None]
        return array

    @staticmethod
    def resolve_stop_entity_ids(
        stop_nodes_cpu: np.ndarray, node_global_ids_cpu: np.ndarray
    ) -> np.ndarray:
        safe_stop_nodes = np.maximum(stop_nodes_cpu, 0)
        stop_entity_ids_cpu = node_global_ids_cpu[safe_stop_nodes]
        return np.where(stop_nodes_cpu >= 0, stop_entity_ids_cpu, -1)

    @staticmethod
    def extract_start_entity_ids(
        *,
        graph_idx: int,
        q_local_indices_cpu: np.ndarray,
        q_ptr_cpu: np.ndarray,
        node_ptr_cpu: np.ndarray,
        node_global_ids_cpu: np.ndarray,
    ) -> list[int]:
        q_start = int(q_ptr_cpu[graph_idx])
        q_end = int(q_ptr_cpu[graph_idx + 1])
        node_offset = int(node_ptr_cpu[graph_idx])
        local_q = q_local_indices_cpu[q_start:q_end]
        return (node_global_ids_cpu[node_offset + local_q]).astype(int).tolist()

    @staticmethod
    def extract_answer_entity_ids(
        *,
        graph_idx: int,
        answer_entity_ids_cpu: np.ndarray,
        answer_ptr_cpu: np.ndarray,
    ) -> list[int]:
        ans_start = int(answer_ptr_cpu[graph_idx])
        ans_end = int(answer_ptr_cpu[graph_idx + 1])
        return answer_entity_ids_cpu[ans_start:ans_end].astype(int).tolist()

    @staticmethod
    def build_graph_rollout_records(
        *,
        graph_idx: int,
        stop_entity_ids_cpu: np.ndarray,
        stop_reason_cpu: np.ndarray,
        num_moves_cpu: np.ndarray,
        log_pf_sum_cpu: np.ndarray,
        step_offset: int,
    ) -> list[dict[str, Any]]:
        num_rollouts = int(stop_entity_ids_cpu.shape[1])
        rollout_records: list[dict[str, Any]] = []
        for rollout_idx in range(num_rollouts):
            score = float(log_pf_sum_cpu[graph_idx, rollout_idx])
            num_moves_raw = int(num_moves_cpu[graph_idx, rollout_idx])
            num_moves_real = max(num_moves_raw - int(step_offset), 0)
            rollout_records.append(
                {
                    "rollout_index": int(rollout_idx),
                    "stop_node_entity_id": int(
                        stop_entity_ids_cpu[graph_idx, rollout_idx]
                    ),
                    "stop_reason": int(stop_reason_cpu[graph_idx, rollout_idx]),
                    "num_moves": num_moves_real,
                    "num_moves_raw": num_moves_raw,
                    "log_pf_sum": score,
                    "score": score,
                    "edges": [],
                }
            )
        return rollout_records

    def build_predict_records(
        self,
        *,
        rollout: RolloutExportInputs,
        graph: GraphExportInputs,
        questions: list[str],
        step_offset: int,
    ) -> list[dict[str, Any]]:
        stop_nodes_cpu = self.to_2d_numpy(rollout.stop_nodes)
        num_moves_cpu = self.to_2d_numpy(rollout.num_moves)
        log_pf_sum_cpu = self.to_2d_numpy(rollout.log_pf_sum)
        stop_reason_cpu = self.to_2d_numpy(rollout.stop_reason)
        q_local_indices_cpu = graph.q_local_indices.detach().cpu().numpy()
        q_ptr_cpu = graph.q_ptr.detach().cpu().numpy()
        a_ptr_cpu = graph.a_ptr.detach().cpu().numpy()
        node_ptr_cpu = graph.node_ptr.detach().cpu().numpy()
        node_global_ids_cpu = graph.node_global_ids.detach().cpu().numpy()
        answer_entity_ids_cpu = graph.answer_entity_ids.detach().cpu().numpy()
        answer_ptr_cpu = graph.answer_ptr.detach().cpu().numpy()
        stop_entity_ids_cpu = self.resolve_stop_entity_ids(
            stop_nodes_cpu, node_global_ids_cpu
        )

        records: list[dict[str, Any]] = []
        sample_ids = list(graph.sample_ids)
        for graph_idx in range(int(graph.num_graphs)):
            start_entity_ids = self.extract_start_entity_ids(
                graph_idx=graph_idx,
                q_local_indices_cpu=q_local_indices_cpu,
                q_ptr_cpu=q_ptr_cpu,
                node_ptr_cpu=node_ptr_cpu,
                node_global_ids_cpu=node_global_ids_cpu,
            )
            answer_ids = self.extract_answer_entity_ids(
                graph_idx=graph_idx,
                answer_entity_ids_cpu=answer_entity_ids_cpu,
                answer_ptr_cpu=answer_ptr_cpu,
            )
            has_answer_in_subgraph = int(a_ptr_cpu[graph_idx + 1]) > int(
                a_ptr_cpu[graph_idx]
            )
            rollout_records = self.build_graph_rollout_records(
                graph_idx=graph_idx,
                stop_entity_ids_cpu=stop_entity_ids_cpu,
                stop_reason_cpu=stop_reason_cpu,
                num_moves_cpu=num_moves_cpu,
                log_pf_sum_cpu=log_pf_sum_cpu,
                step_offset=step_offset,
            )
            records.append(
                {
                    "sample_id": sample_ids[graph_idx],
                    "question": questions[graph_idx],
                    "start_entity_ids": start_entity_ids,
                    "answer_entity_ids": answer_ids,
                    "a_entity_in_graph": bool(has_answer_in_subgraph),
                    "decoder": "beam",
                    "rollouts": rollout_records,
                }
            )
        return records


__all__ = [
    "RolloutExportInputs",
    "GraphExportInputs",
    "DualFlowRolloutExporter",
]
