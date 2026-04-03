from __future__ import annotations

import argparse
import json
import math
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
from src.data.retrieval.collate import BatchAugmenter  # noqa: E402
from src.data.retrieval.components.embeddings import attach_embeddings_to_batch  # noqa: E402
from src.data.retrieval.components.shared_resources import SharedDataResources  # noqa: E402
from src.data.retrieval.dataset import create_graph_retrieval_dataset  # noqa: E402
from src.graph import TrajectoryBatch  # noqa: E402
from src.subgraph_gflownet.application.evaluation import (  # noqa: E402
    SubgraphAnswerSearchRuntime,
)
from src.subgraph_gflownet.core.state import SubgraphAction, SubgraphState  # noqa: E402


@dataclass(frozen=True)
class SelectedSample:
    index: int
    sample_id: str
    question: str | None
    data: Any
    raw_sample: dict[str, Any]
    reference_trajectory_name: str
    reference_edge_ids: tuple[int, ...]
    shortest_path: ForwardShortestPathTrajectory | None
    union_trajectory: ForwardMultiAnchorUnionTrajectory | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one answer-set RankFlow sample end-to-end."
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
        help="Seed used for rollout sampling diagnostics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many action / answer candidates to print per section.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Override rollouts_per_graph during sampled training diagnostics.",
    )
    return parser.parse_args()


def _plain_mapping(node: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(node, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected mapping config, got {type(container)!r}.")
    return dict(container)


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
            reference_trajectory_name = "shortest_path"
            reference_edge_ids: tuple[int, ...] = ()
            if union_trajectory is not None:
                reference_trajectory_name = "multi_anchor_union"
                reference_edge_ids = tuple(
                    int(edge_id) for edge_id in union_trajectory.ordered_edge_ids
                )
            elif shortest_path is not None:
                reference_edge_ids = tuple(
                    int(edge_id) for edge_id in shortest_path.path_edge_ids
                )
            if len(reference_edge_ids) < 2:
                continue
            data = dataset.get(idx)
            question = getattr(data, "question", None)
            return SelectedSample(
                index=idx,
                sample_id=str(current_sample_id),
                question=None if question is None else str(question),
                data=data,
                raw_sample=raw_sample,
                reference_trajectory_name=reference_trajectory_name,
                reference_edge_ids=reference_edge_ids,
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


def _reference_trajectory_payload(selected_sample: SelectedSample) -> dict[str, Any]:
    return {
        "sample_id": selected_sample.sample_id,
        "question": selected_sample.question,
        "reference_trajectory_name": selected_sample.reference_trajectory_name,
        "reference_edge_ids": [
            int(edge_id) for edge_id in selected_sample.reference_edge_ids
        ],
        "reference_trajectory_length": int(len(selected_sample.reference_edge_ids)),
    }


def _action_log_prob_summary(state_distribution: Any) -> list[dict[str, Any]]:
    gate_logits = torch.stack(
        [state_distribution.stop_logit, state_distribution.continue_logit], dim=0
    ).to(dtype=torch.float32)
    gate_log_probs = torch.log_softmax(gate_logits, dim=0)
    records: list[dict[str, Any]] = []

    stop_logits = torch.stack(
        [choice.logit for choice in state_distribution.stop_choices], dim=0
    ).to(dtype=torch.float32)
    stop_log_probs = torch.log_softmax(stop_logits, dim=0)
    for stop_idx, stop_choice in enumerate(state_distribution.stop_choices):
        total_log_prob = float(
            gate_log_probs[0].item() + stop_log_probs[int(stop_idx)].item()
        )
        records.append(
            {
                "kind": "stop",
                "terminal_answer_set_size": int(stop_choice.support_node_count),
                "log_prob": total_log_prob,
                "prob": float(math.exp(total_log_prob)),
            }
        )

    if not state_distribution.node_choices:
        return sorted(records, key=lambda item: -float(item["log_prob"]))

    node_logits = torch.stack(
        [choice.logit for choice in state_distribution.node_choices], dim=0
    ).to(dtype=torch.float32)
    node_log_probs = torch.log_softmax(node_logits, dim=0)
    continue_log_prob = float(gate_log_probs[1].item())
    for node_idx, node_choice in enumerate(state_distribution.node_choices):
        relation_logits = torch.stack(
            [choice.logit for choice in node_choice.relations], dim=0
        ).to(dtype=torch.float32)
        relation_log_probs = torch.log_softmax(relation_logits, dim=0)
        for relation_idx, relation_choice in enumerate(node_choice.relations):
            edge_logits = torch.stack(
                [choice.logit for choice in relation_choice.edges], dim=0
            ).to(dtype=torch.float32)
            edge_log_probs = torch.log_softmax(edge_logits, dim=0)
            prefix_log_prob = (
                continue_log_prob
                + float(node_log_probs[int(node_idx)].item())
                + float(relation_log_probs[int(relation_idx)].item())
            )
            for edge_idx, edge_choice in enumerate(relation_choice.edges):
                total_log_prob = prefix_log_prob + float(
                    edge_log_probs[int(edge_idx)].item()
                )
                records.append(
                    {
                        "kind": "add_edge",
                        "edge_id": int(edge_choice.action.edge_id),
                        "source_graph_node": int(node_choice.graph_node_id),
                        "relation_id": int(relation_choice.relation_id),
                        "target_graph_node": int(edge_choice.target_graph_node),
                        "next_component_count": int(edge_choice.next_component_count),
                        "answer_candidate_count": int(
                            edge_choice.answer_candidate_count
                        ),
                        "log_prob": total_log_prob,
                        "prob": float(math.exp(total_log_prob)),
                    }
                )
    return sorted(records, key=lambda item: -float(item["log_prob"]))


def _terminal_answer_set_payload(
    *, policy: Any, prepared_batch: Any, analysis: Any, graph_idx: int
) -> dict[str, Any]:
    answer_set = policy.admissible_answer_set(
        prepared_batch=prepared_batch,
        graph_idx=int(graph_idx),
        analysis=analysis,
    )
    return {
        "terminal_answer_set_entity_ids": [
            int(entity_id) for entity_id in answer_set.entities
        ],
        "gold_answer_entities_in_terminal_answer_set": [
            int(entity_id) for entity_id in answer_set.gold_entities
        ],
        "non_gold_answer_entities_in_terminal_answer_set": [
            int(entity_id)
            for entity_id in answer_set.entities
            if int(entity_id) not in answer_set.gold_entities
        ],
    }


def _terminal_reward_payload(
    *, policy: Any, prepared_batch: Any, graph_idx: int, analysis: Any
) -> dict[str, Any]:
    terminal_reward = policy.compute_terminal_reward(
        prepared_batch=prepared_batch,
        graph_idx=int(graph_idx),
        analysis=analysis,
    )
    return {
        **_terminal_answer_set_payload(
            policy=policy,
            prepared_batch=prepared_batch,
            analysis=analysis,
            graph_idx=graph_idx,
        ),
        "gold_answer_entities_in_state": [
            int(entity_id)
            for entity_id in terminal_reward.gold_answer_entities_in_graph
        ],
        "terminal_reward_summary": {
            "log_reward": float(terminal_reward.log_reward),
            "gold_answer_in_state": bool(terminal_reward.hit),
            "terminal_answer_set_size": int(terminal_reward.answer_set.count),
            "gold_answer_entities_in_state_count": int(
                terminal_reward.gold_answer_count
            ),
            "frontier_hits_gold_answer": bool(terminal_reward.frontier_hit),
            "anchor_coverage": float(terminal_reward.anchor_coverage),
            "utility": float(terminal_reward.utility),
            "redundancy_edge_count": int(terminal_reward.redundancy_edges),
        },
    }


def _initial_action_summary(
    *, model: Any, prepared_batch: Any, top_k: int
) -> dict[str, Any]:
    rollout_batch = model.policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    distribution = model.policy.compute_action_distribution(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
        action_pruning=model._training_action_pruning_cfg(),
    )
    state_distribution = distribution.state_distributions[0]
    return {
        "current_anchor_component_count": int(
            state_distribution.current_component_count
        ),
        "current_terminal_answer_set_size": int(
            state_distribution.current_answer_candidate_count
        ),
        "top_action_candidates": _action_log_prob_summary(state_distribution)[
            : int(top_k)
        ],
    }


def _trace_reference_trajectory(
    model: Any,
    batch: TrajectoryBatch,
    selected_sample: SelectedSample,
    top_k: int,
) -> None:
    print("\n=== Reference Trajectory Trace ===")
    prepared_batch = model.policy.prepare_batch(batch)
    print(json.dumps(_reference_trajectory_payload(selected_sample), ensure_ascii=True))
    print(
        json.dumps(
            _initial_action_summary(
                model=model, prepared_batch=prepared_batch, top_k=top_k
            ),
            ensure_ascii=True,
        )
    )
    rollout_batch = model.policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    for step_idx, reference_edge_id in enumerate(selected_sample.reference_edge_ids):
        analyses = model.policy.analyze_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
        )
        state_distribution = model.policy.compute_action_distribution(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
            action_pruning=model._training_action_pruning_cfg(),
        ).state_distributions[0]
        action_candidates = _action_log_prob_summary(state_distribution)
        reference_rank = next(
            (
                int(rank)
                for rank, candidate in enumerate(action_candidates, start=1)
                if candidate.get("edge_id") == int(reference_edge_id)
            ),
            None,
        )
        chosen_action = SubgraphAction.add_edge(int(reference_edge_id))
        rollout_batch = model.policy.transition(
            rollout_batch=rollout_batch,
            chosen_actions=(chosen_action,),
        )
        next_analysis = model.policy.analyze_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
        )[0]
        print(
            json.dumps(
                {
                    "step_index": int(step_idx),
                    "reference_edge": _edge_payload(
                        prepared_batch, int(reference_edge_id)
                    ),
                    "reference_edge_rank_before_step": (
                        None if reference_rank is None else int(reference_rank)
                    ),
                    "top_action_candidates_before_step": action_candidates[
                        : int(top_k)
                    ],
                    "next_state_edge_ids": [
                        int(edge_id) for edge_id in rollout_batch.states[0].edge_ids
                    ],
                    "next_state_node_ids": [
                        int(node_id) for node_id in next_analysis.selected_node_ids
                    ],
                    "next_anchor_component_count": int(
                        next_analysis.anchor_component_count
                    ),
                    **_terminal_reward_payload(
                        policy=model.policy,
                        prepared_batch=prepared_batch,
                        graph_idx=0,
                        analysis=next_analysis,
                    ),
                },
                ensure_ascii=True,
            )
        )


def _run_posterior_surrogate_analysis(
    model: Any, cfg: DictConfig, batch: TrajectoryBatch, top_k: int
) -> None:
    print("\n=== Validation-Time Posterior-Surrogate Output ===")
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_plain_mapping(cfg.model.eval_cfg),
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
                "answer_log_posterior_surrogate_masses": result[
                    "answer_log_posterior_surrogate_masses"
                ][:top_k],
                "posterior_surrogate_aggregation_backend": result[
                    "posterior_surrogate_aggregation_backend"
                ],
                "requested_rollout_count": int(result["requested_rollout_count"]),
                "executed_rollout_count": int(result["rollout_count"]),
                "nonempty_terminal_answer_set_rollout_count": int(
                    result["nonempty_terminal_answer_set_rollout_count"]
                ),
                "gold_answer_in_state_rollout_count": int(
                    result["gold_answer_in_state_rollout_count"]
                ),
                "terminal_witness_count": int(result["terminal_witness_count"]),
                "mean_stop_step": float(result["mean_stop_step"]),
                "mean_anchor_component_count": float(
                    result["mean_terminal_component_count"]
                ),
            },
            ensure_ascii=True,
        )
    )
    print(
        json.dumps(
            {
                "witness_support_head": result.get("witness_supports", [])[
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
        "actor.node_focus_head": 0.0,
        "actor.stop_choice_head": 0.0,
        "actor.failure_stop_head": 0.0,
        "actor.relation_head": 0.0,
        "actor.candidate_encoder": 0.0,
        "actor.action_head": 0.0,
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


def _run_training_rollout_diagnostics(
    model: Any,
    batch: TrajectoryBatch,
    *,
    rollouts_per_graph: int,
    seed: int,
    top_k: int,
) -> None:
    del top_k
    print("\n=== Sampled Training Rollouts + Gradient Diagnostics ===")
    torch.manual_seed(int(seed))
    model.train()
    model.zero_grad(set_to_none=True)
    prepared_batch = model.policy.prepare_batch(batch)
    sample_batch = model.sampler.sample(
        policy=model.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=int(rollouts_per_graph),
        temperature=float(model._resolve_sampling_temperature()),
        proposal_bias_scale=float(model._resolve_proposal_bias_scale()),
        action_pruning=model._training_action_pruning_cfg(),
    )
    loss_output = model.loss_fn.compute(sample_batch)
    total_loss = loss_output.loss
    total_loss.backward()
    metric_payload = model._build_subgraph_training_metrics(
        loss_output=loss_output,
        sample_batch=sample_batch,
        total_loss=total_loss,
        rollouts_per_graph=int(rollouts_per_graph),
        sampling_temperature=float(model._resolve_sampling_temperature()),
        proposal_bias_scale=float(model._resolve_proposal_bias_scale()),
    )
    terminal_rollout_summaries = []
    for rollout_idx in range(int(sample_batch.num_rollouts)):
        terminal_rollout_summaries.append(
            {
                "rollout_index": int(rollout_idx),
                "chosen_edge_ids": [
                    int(edge_id)
                    for edge_id in sample_batch.chosen_edge_ids[0, rollout_idx].tolist()
                    if int(edge_id) >= 0
                ],
                "terminal_answer_set_entity_ids": [
                    int(entity_id)
                    for entity_id in sample_batch.terminal_answer_set_entity_ids[
                        int(rollout_idx)
                    ]
                ],
                "terminal_answer_set_size": int(
                    sample_batch.terminal_answer_candidate_counts[0, rollout_idx].item()
                ),
                "gold_answer_entities_in_state_count": int(
                    sample_batch.terminal_gold_answer_counts[0, rollout_idx].item()
                ),
                "gold_answer_in_state": bool(
                    sample_batch.terminal_hit_mask[0, rollout_idx].item()
                ),
                "anchor_component_count": int(
                    sample_batch.terminal_component_counts[0, rollout_idx].item()
                ),
                "stop_step": int(
                    sample_batch.termination_action_steps[0, rollout_idx].item()
                ),
            }
        )
    print(
        json.dumps(
            {
                "training_metrics": {
                    key: (
                        float(value.detach().item())
                        if torch.is_tensor(value)
                        else float(value)
                    )
                    for key, value in metric_payload.items()
                },
                "rollout_terminal_summaries": terminal_rollout_summaries,
            },
            ensure_ascii=True,
        )
    )
    print(
        json.dumps(
            {"parameter_grad_norms": _param_grad_summary(model)}, ensure_ascii=True
        )
    )


def _run_reference_trajectory_forced_replay_summary(
    model: Any,
    batch: TrajectoryBatch,
    selected_sample: SelectedSample,
) -> None:
    print("\n=== Forced Reference-Trajectory Replay Summary ===")
    prepared_batch = model.policy.prepare_batch(batch)
    sample_batch = model.sampler.teacher_force(
        policy=model.policy,
        prepared_batch=prepared_batch,
        edge_sequences=(selected_sample.reference_edge_ids,),
    )
    print(
        json.dumps(
            {
                "reference_edge_ids": [
                    int(edge_id)
                    for edge_id in sample_batch.chosen_edge_ids[0, 0].tolist()
                ],
                "terminal_answer_set_entity_ids": [
                    int(entity_id)
                    for entity_id in sample_batch.terminal_answer_set_entity_ids[0]
                ],
                "terminal_answer_set_size": int(
                    sample_batch.terminal_answer_candidate_counts[0, 0].item()
                ),
                "gold_answer_entities_in_state_count": int(
                    sample_batch.terminal_gold_answer_counts[0, 0].item()
                ),
                "gold_answer_in_state": bool(
                    sample_batch.terminal_hit_mask[0, 0].item()
                ),
                "anchor_component_count": int(
                    sample_batch.terminal_component_counts[0, 0].item()
                ),
            },
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
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )
    selected_sample = _select_sample(cfg, sample_id=args.sample_id)
    model = _instantiate_model(cfg, ckpt_path=ckpt_path, device=device)
    batch = _build_single_graph_batch(
        selected_sample,
        cfg,
        device=device,
        dataset_scope=str(cfg.dataset.dataset_scope),
    )
    model.eval()
    _trace_reference_trajectory(model, batch, selected_sample, top_k=int(args.top_k))
    _run_reference_trajectory_forced_replay_summary(model, batch, selected_sample)
    _run_posterior_surrogate_analysis(model, cfg, batch, top_k=int(args.top_k))
    _run_training_rollout_diagnostics(
        model,
        batch,
        rollouts_per_graph=int(
            args.rollouts or cfg.model.training_cfg.rollouts_per_graph
        ),
        seed=int(args.seed),
        top_k=int(args.top_k),
    )


if __name__ == "__main__":
    main()
