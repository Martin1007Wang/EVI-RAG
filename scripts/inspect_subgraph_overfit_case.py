from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.preprocess.labels.edge_retrieval import (  # noqa: E402
    ForwardMultiAnchorUnionTrajectory,
    ForwardShortestPathTrajectory,
    resolve_forward_multi_anchor_union_trajectory,
    resolve_forward_shortest_path_trajectory,
)
from src.datasets.graph_retrieval_collate import BatchAugmenter  # noqa: E402
from src.datasets.graph_retrieval_dataset import create_graph_retrieval_dataset  # noqa: E402
from src.datasets.components.embeddings import attach_embeddings_to_batch  # noqa: E402
from src.datasets.components.shared_resources import SharedDataResources  # noqa: E402
from src.graph import TrajectoryBatch  # noqa: E402
from src.metrics.subgraph_answer_search_runtime import (  # noqa: E402
    SubgraphAnswerSearchRuntime,
)
from src.models.gflownet.reward import resolve_subgraph_answer_entities  # noqa: E402
from src.models.gflownet.sampler import SubgraphTrajectorySampleBatch  # noqa: E402
from src.models.gflownet.state import SubgraphAction  # noqa: E402
from src.utils.segment_ops import sample_segmented_one_1d  # noqa: E402


@dataclass(frozen=True)
class SelectedSample:
    index: int
    sample_id: str
    question: str | None
    data: Any
    raw_sample: dict[str, Any]
    teacher_name: str
    teacher_edge_ids: tuple[int, ...]
    shortest_path: ForwardShortestPathTrajectory | None
    union_trajectory: ForwardMultiAnchorUnionTrajectory | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one multi-anchor multi-hop subgraph sample end-to-end."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Hydra run dir containing .hydra/config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional exact sample id. Defaults to the first multi-anchor multi-hop sample.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoints/last.ckpt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. cpu or cuda:0. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used for traced rollout sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many candidate actions / predicted answers to print.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Override rollouts_per_graph during traced training analysis.",
    )
    return parser.parse_args()


def _resolve_num_nodes(raw_sample: dict[str, Any]) -> int:
    raw_num_nodes = raw_sample.get("num_nodes")
    if raw_num_nodes is not None:
        return int(torch.as_tensor(raw_num_nodes).view(-1)[0].item())
    return int(torch.as_tensor(raw_sample["node_entity_ids"]).numel())


def _load_cfg(run_dir: Path) -> DictConfig:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _instantiate_model(cfg: DictConfig, ckpt_path: Path, device: torch.device) -> Any:
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {ckpt_path} does not contain a valid state_dict.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        json.dumps(
            {
                "loaded_checkpoint": str(ckpt_path),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            ensure_ascii=True,
        )
    )
    model.to(device)
    return model


def _select_sample(cfg: DictConfig, *, sample_id: str | None) -> SelectedSample:
    dataset = create_graph_retrieval_dataset(cfg.dataset, split_name="validation")
    try:
        for idx, current_sample_id in enumerate(dataset.sample_ids):
            if sample_id is not None and current_sample_id != sample_id:
                continue
            raw_sample = dataset._load_raw_sample(current_sample_id)
            raw_edge_index = torch.as_tensor(raw_sample["edge_index"], dtype=torch.long)
            num_nodes = _resolve_num_nodes(raw_sample)
            anchor_local_indices = torch.as_tensor(
                raw_sample["anchor_local_indices"], dtype=torch.long
            )
            a_local_indices = torch.as_tensor(
                raw_sample["a_local_indices"], dtype=torch.long
            )
            shortest_path = resolve_forward_shortest_path_trajectory(
                edge_index=raw_edge_index,
                anchor_local_indices=anchor_local_indices,
                a_local_indices=a_local_indices,
                num_nodes=num_nodes,
            )
            if int(anchor_local_indices.numel()) < 2:
                continue
            union_trajectory = resolve_forward_multi_anchor_union_trajectory(
                edge_index=raw_edge_index,
                anchor_local_indices=anchor_local_indices,
                a_local_indices=a_local_indices,
                num_nodes=num_nodes,
            )
            teacher_name = "shortest_path"
            teacher_edge_ids: tuple[int, ...] = ()
            if union_trajectory is not None:
                teacher_name = "multi_anchor_union"
                teacher_edge_ids = tuple(
                    int(edge_id) for edge_id in union_trajectory.ordered_edge_ids
                )
            elif shortest_path is not None:
                teacher_edge_ids = tuple(
                    int(edge_id) for edge_id in shortest_path.path_edge_ids
                )
            if len(teacher_edge_ids) < 2:
                continue
            data = dataset.get(idx)
            question = getattr(data, "question", None)
            return SelectedSample(
                index=idx,
                sample_id=str(current_sample_id),
                question=None if question is None else str(question),
                data=data,
                raw_sample=raw_sample,
                teacher_name=teacher_name,
                teacher_edge_ids=teacher_edge_ids,
                shortest_path=shortest_path,
                union_trajectory=union_trajectory,
            )
    finally:
        dataset.close()
    if sample_id is None:
        raise RuntimeError(
            "Could not find a multi-anchor multi-hop sample in the run subset."
        )
    raise RuntimeError(
        f"Requested sample_id not found or not multi-anchor multi-hop: {sample_id}"
    )


def _build_single_graph_batch(
    sample: SelectedSample,
    cfg: DictConfig,
    device: torch.device,
    dataset_scope: str,
) -> TrajectoryBatch:
    pyg_batch = Batch.from_data_list([sample.data])
    pyg_batch = BatchAugmenter(precompute_edge_batch=True)(pyg_batch)
    resources = SharedDataResources(
        entity_vocab_path=Path(str(cfg.dataset.paths.entity_vocab)),
        relation_vocab_path=Path(str(cfg.dataset.paths.relation_vocab)),
        embeddings_dir=Path(str(cfg.dataset.paths.embeddings)),
        embeddings_device=None,
    )
    attach_embeddings_to_batch(
        pyg_batch,
        global_embeddings=resources.global_embeddings,
        embeddings_device=device,
    )
    pyg_batch = pyg_batch.to(device)
    return TrajectoryBatch.from_pyg_batch(
        pyg_batch,
        device=device,
        dataset_scope=str(dataset_scope),
    )


def _edge_tuple(prepared_batch: Any, edge_id: int) -> tuple[int, int, int]:
    src = int(prepared_batch.topology.edge_index[0, edge_id].item())
    dst = int(prepared_batch.topology.edge_index[1, edge_id].item())
    relation = int(prepared_batch.topology.edge_type[edge_id].item())
    return src, relation, dst


def _edge_payload(prepared_batch: Any, edge_id: int) -> dict[str, Any]:
    src, relation, dst = _edge_tuple(prepared_batch, edge_id)
    return {
        "edge_id": int(edge_id),
        "src": src,
        "relation": relation,
        "dst": dst,
        "src_entity": int(prepared_batch.node_entity_ids[src].item()),
        "dst_entity": int(prepared_batch.node_entity_ids[dst].item()),
    }


def _teacher_summary_payload(selected: SelectedSample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "teacher_name": selected.teacher_name,
        "teacher_edge_ids": [int(edge_id) for edge_id in selected.teacher_edge_ids],
    }
    if selected.union_trajectory is not None:
        payload.update(
            {
                "answer_node": int(selected.union_trajectory.answer_node),
                "anchor_nodes": [
                    int(node_id) for node_id in selected.union_trajectory.anchor_nodes
                ],
                "anchor_path_nodes": [
                    [int(node_id) for node_id in path_nodes]
                    for path_nodes in selected.union_trajectory.anchor_path_nodes
                ],
                "anchor_path_edge_ids": [
                    [int(edge_id) for edge_id in path_edge_ids]
                    for path_edge_ids in selected.union_trajectory.anchor_path_edge_ids
                ],
                "ordered_edge_ids": [
                    int(edge_id)
                    for edge_id in selected.union_trajectory.ordered_edge_ids
                ],
                "union_edge_ids": [
                    int(edge_id) for edge_id in selected.union_trajectory.union_edge_ids
                ],
                "total_hop_length": int(selected.union_trajectory.total_hop_length),
            }
        )
    elif selected.shortest_path is not None:
        payload.update(
            {
                "anchor_node": int(selected.shortest_path.anchor_node),
                "path_nodes": [
                    int(node_id) for node_id in selected.shortest_path.path_nodes
                ],
                "path_edge_ids": [
                    int(edge_id) for edge_id in selected.shortest_path.path_edge_ids
                ],
                "hop_length": int(selected.shortest_path.hop_length),
            }
        )
    return payload


def _trace_oracle_path(
    model: Any, batch: TrajectoryBatch, selected: SelectedSample
) -> None:
    prepared_batch = model.policy.prepare_batch(batch)
    state = model.policy.initial_state()
    analysis = model.policy.analyze_state(
        prepared_batch=prepared_batch, graph_idx=0, state=state
    )
    print("\n=== Sample ===")
    print(
        json.dumps(
            {
                "sample_id": batch.sample_ids[0],
                "question": batch.questions[0],
                "anchor_local_indices": batch.anchor_local_indices.detach()
                .cpu()
                .tolist(),
                "answer_local_indices": batch.a_local_indices.detach().cpu().tolist(),
                "answer_entity_ids": batch.answer_entity_ids.detach().cpu().tolist(),
                "num_nodes": int(batch.num_nodes_total),
                "num_edges": int(batch.edge_index.size(1)),
            },
            ensure_ascii=True,
        )
    )
    print("\n=== Teacher Supervision ===")
    print(json.dumps(_teacher_summary_payload(selected), ensure_ascii=True))
    print(
        json.dumps(
            {
                "path_edges": [
                    _edge_payload(prepared_batch, int(edge_id))
                    for edge_id in selected.teacher_edge_ids
                ]
            },
            ensure_ascii=True,
        )
    )
    print("\n=== Initial State ===")
    print(
        json.dumps(
            {
                "selected_node_ids": [
                    int(node_id) for node_id in analysis.selected_node_ids
                ],
                "selected_node_entities": [
                    int(prepared_batch.node_entity_ids[int(node_id)].item())
                    for node_id in analysis.selected_node_ids
                ],
                "reachability_bits": {
                    str(node_id): int(bits)
                    for node_id, bits in analysis.reachability_bits.items()
                },
                "anchor_component_count": int(analysis.anchor_component_count),
            },
            ensure_ascii=True,
        )
    )

    rollout_batch = model.policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    analyses = model.policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
    )
    distribution = model.policy.compute_action_distribution(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
        analyses=analyses,
    )
    target_log_probs = model.policy.compute_target_log_probs(distribution)
    proposal_bias = model.policy.compute_proposal_bias(
        prepared_batch=prepared_batch,
        distribution=distribution,
        proposal_bias_scale=1.0,
    )
    proposal_logits = distribution.logits + proposal_bias

    print("\n=== Initial Action Ranking ===")
    initial_candidates: list[dict[str, Any]] = []
    for position in torch.nonzero(distribution.segment_ids == 0, as_tuple=False).view(
        -1
    ):
        pos = int(position.item())
        action = distribution.actions[pos]
        candidate = {
            "position": pos,
            "is_stop": bool(action.is_stop),
            "edge_id": None if action.edge_id is None else int(action.edge_id),
            "target_log_prob": float(target_log_probs[pos].item()),
            "raw_logit": float(distribution.logits[pos].item()),
            "proposal_bias": float(proposal_bias[pos].item()),
            "proposal_logit": float(proposal_logits[pos].item()),
        }
        if action.edge_id is not None:
            candidate.update(_edge_payload(prepared_batch, int(action.edge_id)))
            candidate["on_teacher_path"] = int(action.edge_id) in set(
                selected.teacher_edge_ids
            )
        initial_candidates.append(candidate)
    initial_candidates.sort(key=lambda item: item["proposal_logit"], reverse=True)
    print(
        json.dumps(
            {"top_proposal_candidates": initial_candidates[:10]}, ensure_ascii=True
        )
    )
    target_sorted = sorted(
        initial_candidates, key=lambda item: item["target_log_prob"], reverse=True
    )
    print(json.dumps({"top_target_candidates": target_sorted[:10]}, ensure_ascii=True))
    path_rankings = []
    for edge_id in selected.teacher_edge_ids:
        edge_id = int(edge_id)
        candidate = next(
            (
                candidate
                for candidate in initial_candidates
                if candidate.get("edge_id") == edge_id
            ),
            None,
        )
        proposal_rank = None
        target_rank = None
        if candidate is not None:
            proposal_rank = next(
                idx + 1
                for idx, ranked_candidate in enumerate(initial_candidates)
                if ranked_candidate.get("edge_id") == edge_id
            )
            target_rank = next(
                idx + 1
                for idx, ranked_candidate in enumerate(target_sorted)
                if ranked_candidate.get("edge_id") == edge_id
            )
        path_rankings.append(
            {
                "edge_id": edge_id,
                "proposal_rank": proposal_rank,
                "target_rank": target_rank,
                "target_log_prob": None
                if candidate is None
                else candidate["target_log_prob"],
                "proposal_bias": None
                if candidate is None
                else candidate["proposal_bias"],
                "proposal_logit": None
                if candidate is None
                else candidate["proposal_logit"],
                "available_at_initial_step": candidate is not None,
            }
        )
    print(json.dumps({"teacher_initial_ranks": path_rankings}, ensure_ascii=True))

    print("\n=== Teacher Path Rollout ===")
    running_state = model.policy.initial_state()
    for step_idx, edge_id in enumerate(selected.teacher_edge_ids):
        current_analysis = model.policy.analyze_state(
            prepared_batch=prepared_batch,
            graph_idx=0,
            state=running_state,
        )
        next_state = running_state.with_edge(int(edge_id))
        next_analysis = model.policy.analyze_state(
            prepared_batch=prepared_batch,
            graph_idx=0,
            state=next_state,
        )
        expand_reward = model.policy.compute_expand_log_reward(
            current_analysis=current_analysis,
            next_analysis=next_analysis,
        )
        stop_reward, answer_count, hit = model.policy.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=next_analysis,
        )
        print(
            json.dumps(
                {
                    "step": int(step_idx),
                    "edge": _edge_payload(prepared_batch, int(edge_id)),
                    "expand_log_reward": float(expand_reward),
                    "next_anchor_component_count": int(
                        next_analysis.anchor_component_count
                    ),
                    "next_selected_nodes": [
                        int(node_id) for node_id in next_analysis.selected_node_ids
                    ],
                    "next_answer_entities": [
                        int(entity_id)
                        for entity_id in resolve_subgraph_answer_entities(
                            prepared_batch=prepared_batch,
                            graph_idx=0,
                            analysis=next_analysis,
                        )
                    ],
                    "stop_reward_if_stop_now": float(stop_reward),
                    "stop_answer_count_if_stop_now": int(answer_count),
                    "stop_hit_if_stop_now": bool(hit),
                },
                ensure_ascii=True,
            )
        )
        running_state = next_state

    final_analysis = model.policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=running_state,
    )
    final_stop_reward, final_answer_count, final_hit = (
        model.policy.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=final_analysis,
        )
    )
    print(
        json.dumps(
            {
                "final_state_edge_ids": [
                    int(edge_id) for edge_id in running_state.edge_ids
                ],
                "final_answer_entities": [
                    int(entity_id)
                    for entity_id in resolve_subgraph_answer_entities(
                        prepared_batch=prepared_batch,
                        graph_idx=0,
                        analysis=final_analysis,
                    )
                ],
                "final_stop_reward": float(final_stop_reward),
                "final_answer_count": int(final_answer_count),
                "final_hit": bool(final_hit),
            },
            ensure_ascii=True,
        )
    )


def _run_prediction_analysis(
    model: Any, cfg: DictConfig, batch: TrajectoryBatch, top_k: int
) -> None:
    print("\n=== Validation-Time Search Output ===")
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=OmegaConf.to_container(cfg.model.eval_cfg, resolve=True),
        policy=model.policy,
        sampler=model.sampler,
    )
    result = runtime._predict_single_graph(batch=batch, include_answer_support=True)
    print(
        json.dumps(
            {
                "gold_answer_entity_ids": result["gold_answer_entity_ids"],
                "predicted_answer_entity_ids": result["predicted_answer_entity_ids"][
                    :top_k
                ],
                "answer_log_masses": result["answer_log_masses"][:top_k],
                "terminal_subgraph_count": int(result["terminal_subgraph_count"]),
                "frontier_state_count": int(result["frontier_state_count"]),
                "frontier_answering_state_count": int(
                    result["frontier_answering_state_count"]
                ),
            },
            ensure_ascii=True,
        )
    )
    print(
        json.dumps(
            {
                "terminal_subgraphs_head": result.get("terminal_subgraphs", [])[
                    : min(5, top_k)
                ]
            },
            ensure_ascii=True,
        )
    )


def _param_grad_summary(model: Any) -> dict[str, float]:
    summaries = {
        "encoder.backbone": 0.0,
        "encoder.state_encoder": 0.0,
        "encoder.flow_head": 0.0,
        "actor.candidate_encoder": 0.0,
        "actor.action_head": 0.0,
        "actor.stop_head": 0.0,
    }
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_norm = float(grad.detach().norm().item())
        for prefix in summaries:
            if name.startswith(f"policy.{prefix}"):
                summaries[prefix] += grad_norm
    return summaries


def _trace_training_rollouts(
    model: Any,
    cfg: DictConfig,
    batch: TrajectoryBatch,
    selected: SelectedSample,
    *,
    rollouts_per_graph: int,
    seed: int,
) -> None:
    print("\n=== Traced Training Rollouts + Backprop ===")
    torch.manual_seed(int(seed))
    model.train()
    model.zero_grad(set_to_none=True)
    prepared_batch = model.policy.prepare_batch(batch)
    max_steps = int(model.sampler.max_steps)
    max_actions = max_steps + 1
    flat_states = int(batch.num_graphs) * int(rollouts_per_graph)
    rollout_batch = model.policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=rollouts_per_graph,
    )
    state_log_flows = torch.zeros(
        (flat_states, max_actions),
        device=prepared_batch.device,
        dtype=torch.float32,
    )
    log_pf_actions = torch.zeros_like(state_log_flows)
    log_reward_actions = torch.zeros_like(state_log_flows)
    action_mask = torch.zeros_like(state_log_flows, dtype=torch.bool)
    chosen_edge_ids = torch.full(
        (flat_states, max_steps),
        fill_value=-1,
        device=prepared_batch.device,
        dtype=torch.long,
    )
    stop_actions = torch.zeros_like(state_log_flows, dtype=torch.bool)
    termination_action_steps = torch.zeros(
        (flat_states,), device=prepared_batch.device, dtype=torch.long
    )
    terminal_answer_counts = torch.zeros_like(termination_action_steps)
    terminal_hit_mask = torch.zeros_like(termination_action_steps, dtype=torch.bool)
    terminal_component_counts = torch.zeros_like(termination_action_steps)
    step_traces: list[dict[str, Any]] = []
    teacher_edge_set = {int(edge_id) for edge_id in selected.teacher_edge_ids}
    sampling_temperature = float(cfg.model.training_cfg.sampling_temperature)
    proposal_bias_scale = float(
        cfg.model.training_cfg.proposal_bias_schedule.initial_scale
        if cfg.model.training_cfg.proposal_bias_schedule.initial_scale is not None
        else 1.0
    )

    for action_step in range(max_actions):
        if not bool((~rollout_batch.done_mask).any().item()):
            break
        analyses = model.policy.analyze_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
        )
        current_log_flow = model.policy.compute_log_flows(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        state_log_flows[:, action_step] = current_log_flow
        distribution = model.policy.compute_action_distribution(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        distribution.logits.retain_grad()
        target_log_probs = model.policy.compute_target_log_probs(distribution)
        proposal_bias = model.policy.compute_proposal_bias(
            prepared_batch=prepared_batch,
            distribution=distribution,
            proposal_bias_scale=proposal_bias_scale,
        )
        proposal_logits = distribution.logits + proposal_bias
        chosen_positions, _, has_values = sample_segmented_one_1d(
            logits=proposal_logits,
            segment_ids=distribution.segment_ids,
            num_segments=int(distribution.flat_state_indices.numel()),
            temperature=sampling_temperature,
        )
        if not bool(has_values.all().item()):
            raise RuntimeError(
                "Failed to sample a legal action for every active state."
            )

        chosen_actions: list[SubgraphAction] = [SubgraphAction.stop()] * flat_states
        for state_idx in range(flat_states):
            if bool(rollout_batch.done_mask[state_idx].item()):
                chosen_actions[state_idx] = SubgraphAction.stop()

        trace_segments: list[dict[str, Any]] = []
        for local_state_idx, flat_state_idx in enumerate(
            distribution.flat_state_indices.detach().cpu().tolist()
        ):
            action_pos = int(chosen_positions[local_state_idx].item())
            action = distribution.actions[action_pos]
            current_analysis = analyses[int(flat_state_idx)]
            chosen_actions[int(flat_state_idx)] = action
            action_mask[int(flat_state_idx), action_step] = True
            stop_actions[int(flat_state_idx), action_step] = bool(action.is_stop)
            log_pf_actions[int(flat_state_idx), action_step] = target_log_probs[
                action_pos
            ]

            segment_positions = torch.nonzero(
                distribution.segment_ids == int(local_state_idx), as_tuple=False
            ).view(-1)
            shortest_candidates: list[dict[str, Any]] = []
            for position in segment_positions.detach().cpu().tolist():
                position = int(position)
                edge_id = int(distribution.edge_ids[position].item())
                if edge_id not in teacher_edge_set:
                    continue
                shortest_candidates.append(
                    {
                        "edge_id": edge_id,
                        "target_log_prob": float(target_log_probs[position].item()),
                        "proposal_bias": float(proposal_bias[position].item()),
                        "proposal_logit": float(proposal_logits[position].item()),
                        "candidate_answer_count": int(
                            distribution.candidate_answer_counts[position].item()
                        ),
                    }
                )

            current_components = int(
                distribution.current_component_counts[action_pos].item()
            )
            reward_value = 0.0
            answer_count = 0
            hit = False
            if action.is_stop:
                reward_value, answer_count, hit = model.policy.compute_stop_log_reward(
                    prepared_batch=prepared_batch,
                    graph_idx=int(rollout_batch.graph_ids[flat_state_idx].item()),
                    analysis=current_analysis,
                )
                log_reward_actions[int(flat_state_idx), action_step] = float(
                    reward_value
                )
                termination_action_steps[int(flat_state_idx)] = int(action_step + 1)
                terminal_answer_counts[int(flat_state_idx)] = int(answer_count)
                terminal_hit_mask[int(flat_state_idx)] = bool(hit)
                terminal_component_counts[int(flat_state_idx)] = int(current_components)
            else:
                if action.edge_id is None:
                    raise RuntimeError("Expand actions must carry an edge_id.")
                edge_id = int(action.edge_id)
                next_state = rollout_batch.states[int(flat_state_idx)].with_edge(
                    edge_id
                )
                next_analysis = model.policy.analyze_state(
                    prepared_batch=prepared_batch,
                    graph_idx=int(rollout_batch.graph_ids[flat_state_idx].item()),
                    state=next_state,
                )
                chosen_edge_ids[int(flat_state_idx), action_step] = edge_id
                reward_value = model.policy.compute_expand_log_reward(
                    current_analysis=current_analysis,
                    next_analysis=next_analysis,
                )
                log_reward_actions[int(flat_state_idx), action_step] = float(
                    reward_value
                )

            trace_segments.append(
                {
                    "rollout_index": int(flat_state_idx),
                    "chosen_action": {
                        "is_stop": bool(action.is_stop),
                        "edge_id": None
                        if action.edge_id is None
                        else int(action.edge_id),
                    },
                    "chosen_target_log_prob": float(
                        target_log_probs[action_pos].item()
                    ),
                    "chosen_proposal_bias": float(proposal_bias[action_pos].item()),
                    "chosen_proposal_logit": float(proposal_logits[action_pos].item()),
                    "current_component_count": current_components,
                    "reward_value": float(reward_value),
                    "stop_answer_count": int(answer_count),
                    "stop_hit": bool(hit),
                    "teacher_path_candidates": shortest_candidates,
                }
            )

        step_traces.append(
            {
                "step": int(action_step),
                "distribution": distribution,
                "target_log_probs": target_log_probs,
                "proposal_bias": proposal_bias,
                "proposal_logits": proposal_logits,
                "segments": trace_segments,
            }
        )
        rollout_batch = model.policy.transition(
            rollout_batch=rollout_batch,
            chosen_actions=tuple(chosen_actions),
        )

    terminal_analyses = model.policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
    )
    terminal_edge_ids = tuple(state.edge_ids for state in rollout_batch.states)
    terminal_node_ids = tuple(
        analysis.selected_node_ids for analysis in terminal_analyses
    )
    terminal_reachability_bits = tuple(
        dict(analysis.reachability_bits) for analysis in terminal_analyses
    )
    sample_batch = SubgraphTrajectorySampleBatch(
        state_log_flows=state_log_flows.view(1, rollouts_per_graph, max_actions),
        log_pf_actions=log_pf_actions.view(1, rollouts_per_graph, max_actions),
        log_reward_actions=log_reward_actions.view(1, rollouts_per_graph, max_actions),
        action_mask=action_mask.view(1, rollouts_per_graph, max_actions),
        termination_action_steps=termination_action_steps.view(1, rollouts_per_graph),
        chosen_edge_ids=chosen_edge_ids.view(1, rollouts_per_graph, max_steps),
        stop_actions=stop_actions.view(1, rollouts_per_graph, max_actions),
        terminal_answer_counts=terminal_answer_counts.view(1, rollouts_per_graph),
        terminal_hit_mask=terminal_hit_mask.view(1, rollouts_per_graph),
        terminal_component_counts=terminal_component_counts.view(1, rollouts_per_graph),
        terminal_edge_ids=terminal_edge_ids,
        terminal_node_ids=terminal_node_ids,
        terminal_reachability_bits=terminal_reachability_bits,
        sample_ids=tuple(batch.sample_ids),
        question_ids=tuple(batch.questions),
        num_graphs=1,
        num_rollouts=rollouts_per_graph,
    )
    loss_output = model.loss_fn.compute(sample_batch)
    total_loss = loss_output.loss
    total_loss.backward()

    print(
        json.dumps(
            {
                "loss": float(total_loss.detach().item()),
                "success_rate": float(loss_output.success_rate.detach().item()),
                "average_terminal_answer_count": float(
                    loss_output.average_terminal_answer_count.detach().item()
                ),
                "average_terminal_component_count": float(
                    loss_output.average_terminal_component_count.detach().item()
                ),
                "terminal_hit_mask": terminal_hit_mask.view(-1).detach().cpu().tolist(),
                "terminal_answer_counts": terminal_answer_counts.view(-1)
                .detach()
                .cpu()
                .tolist(),
                "termination_action_steps": termination_action_steps.view(-1)
                .detach()
                .cpu()
                .tolist(),
                "chosen_edge_ids": chosen_edge_ids.view(rollouts_per_graph, max_steps)
                .detach()
                .cpu()
                .tolist(),
            },
            ensure_ascii=True,
        )
    )
    print(
        json.dumps(
            {"parameter_grad_norms": _param_grad_summary(model)}, ensure_ascii=True
        )
    )

    print("\n=== Per-Step Gradient Trace ===")
    for step_trace in step_traces:
        distribution = step_trace["distribution"]
        logits_grad = distribution.logits.grad
        if logits_grad is None:
            continue
        gradient_segments: list[dict[str, Any]] = []
        for segment in step_trace["segments"]:
            rollout_index = int(segment["rollout_index"])
            segment_positions = torch.nonzero(
                distribution.segment_ids == rollout_index,
                as_tuple=False,
            ).view(-1)
            top_grad_positions = sorted(
                [int(position.item()) for position in segment_positions],
                key=lambda position: abs(float(logits_grad[position].item())),
                reverse=True,
            )[:5]
            top_grad_entries = []
            for position in top_grad_positions:
                edge_id = int(distribution.edge_ids[position].item())
                top_grad_entries.append(
                    {
                        "position": position,
                        "edge_id": None if edge_id < 0 else edge_id,
                        "is_stop": bool(distribution.is_stop_action[position].item()),
                        "grad": float(logits_grad[position].item()),
                        "target_log_prob": float(
                            step_trace["target_log_probs"][position].item()
                        ),
                        "proposal_bias": float(
                            step_trace["proposal_bias"][position].item()
                        ),
                    }
                )
            shortest_grads = []
            for position in [int(pos.item()) for pos in segment_positions]:
                edge_id = int(distribution.edge_ids[position].item())
                if edge_id not in teacher_edge_set:
                    continue
                shortest_grads.append(
                    {
                        "edge_id": edge_id,
                        "grad": float(logits_grad[position].item()),
                        "target_log_prob": float(
                            step_trace["target_log_probs"][position].item()
                        ),
                        "proposal_bias": float(
                            step_trace["proposal_bias"][position].item()
                        ),
                    }
                )
            gradient_segments.append(
                {
                    "rollout_index": rollout_index,
                    "chosen_action": segment["chosen_action"],
                    "reward_value": segment["reward_value"],
                    "stop_hit": segment["stop_hit"],
                    "teacher_path_grads": shortest_grads,
                    "top_abs_logit_grads": top_grad_entries,
                }
            )
        print(
            json.dumps(
                {"step": step_trace["step"], "segments": gradient_segments},
                ensure_ascii=True,
            )
        )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = (
        args.ckpt.resolve()
        if args.ckpt is not None
        else (run_dir / "checkpoints" / "last.ckpt")
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    selected = _select_sample(cfg, sample_id=args.sample_id)
    model = _instantiate_model(cfg, ckpt_path=ckpt_path, device=device)
    batch = _build_single_graph_batch(
        selected,
        cfg,
        device=device,
        dataset_scope=str(cfg.dataset.dataset_scope),
    )
    _trace_oracle_path(model, batch, selected)
    model.eval()
    _run_prediction_analysis(model, cfg, batch, top_k=int(args.top_k))
    _trace_training_rollouts(
        model,
        cfg,
        batch,
        selected,
        rollouts_per_graph=int(
            args.rollouts or cfg.model.training_cfg.rollouts_per_graph
        ),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
