from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.collate import RetrievalCollator
from src.utils.logging_utils import get_logger
from src.utils.path_utils import compute_shortest_path_labels
from src.models.rollout import RolloutState

log = get_logger(__name__)


@dataclass(frozen=True)
class StateRankingExample:
    graph_index: int
    step_index: int
    best_rank: int
    num_candidates: int
    num_gold_candidates: int
    path_length: int | None


def _path_length_bucket(path_length: int | None) -> str:
    if path_length is None:
        return "unknown"
    if path_length <= 1:
        return "1"
    if path_length == 2:
        return "2"
    return "3+"


def _candidate_bucket(num_candidates: int) -> str:
    if num_candidates <= 10:
        return "1-10"
    if num_candidates <= 50:
        return "11-50"
    return "51+"


def _select_dataset(datamodule: Any, split_name: str) -> Any:
    split_name = str(split_name)
    split_to_dataset = {
        str(
            datamodule.dataset_cfg.get("train_split", "train")
        ): datamodule.train_dataset,
        str(
            datamodule.dataset_cfg.get("val_split", "validation")
        ): datamodule.val_dataset,
        str(datamodule.dataset_cfg.get("eval_split", "test")): datamodule.test_dataset,
        str(
            datamodule.dataset_cfg.get(
                "predict_split", datamodule.dataset_cfg.get("eval_split", "test")
            )
        ): datamodule.predict_dataset,
    }
    dataset = split_to_dataset.get(split_name)
    if dataset is None:
        available = sorted(
            name for name, value in split_to_dataset.items() if value is not None
        )
        raise ValueError(
            f"Split {split_name!r} is unavailable. Available loaded splits: {available}."
        )
    return dataset


def _sample_dataset_indices(
    dataset_size: int, *, max_graphs: int | None, seed: int
) -> list[int]:
    all_indices = list(range(dataset_size))
    if max_graphs is None or max_graphs <= 0 or max_graphs >= dataset_size:
        return all_indices
    rng = random.Random(seed)
    return sorted(rng.sample(all_indices, k=max_graphs))


def _build_query_like(query_emb: torch.Tensor, target_width: int) -> torch.Tensor:
    query_width = int(query_emb.numel())
    if query_width == target_width:
        return query_emb
    if target_width % query_width != 0:
        raise ValueError(
            f"Cannot broadcast query width {query_width} to target width {target_width}."
        )
    return query_emb.repeat(target_width // query_width)


def _cosine_scores(query_emb: torch.Tensor, edge_repr: torch.Tensor) -> torch.Tensor:
    if edge_repr.numel() == 0:
        return edge_repr.new_empty((0,))
    query_like = _build_query_like(query_emb, edge_repr.size(-1)).unsqueeze(0)
    query_like = F.normalize(query_like, dim=-1)
    edge_repr = F.normalize(edge_repr, dim=-1)
    return F.cosine_similarity(query_like, edge_repr, dim=-1)


def _score_candidates(
    *,
    method: str,
    query_emb: torch.Tensor,
    node_tokens: torch.Tensor,
    edge_relation_tokens: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    if candidate_edge_ids.numel() == 0:
        return torch.empty((0,), dtype=query_emb.dtype, device=query_emb.device)

    src = edge_index[0].index_select(0, candidate_edge_ids)
    dst = edge_index[1].index_select(0, candidate_edge_ids)
    rel = edge_relation_tokens.index_select(0, candidate_edge_ids)
    src_h = node_tokens.index_select(0, src)
    dst_h = node_tokens.index_select(0, dst)

    if method == "random":
        return torch.rand(
            candidate_edge_ids.numel(), generator=rng, device=query_emb.device
        )
    if method == "relation_only":
        return _cosine_scores(query_emb, rel)
    if method == "tail_only":
        return _cosine_scores(query_emb, dst_h)
    if method == "rel_tail_concat":
        return _cosine_scores(query_emb, torch.cat([rel, dst_h], dim=-1))
    if method == "src_rel_tail_concat":
        return _cosine_scores(query_emb, torch.cat([src_h, rel, dst_h], dim=-1))
    raise ValueError(
        f"Unsupported scoring method {method!r}. Expected one of random, relation_only, "
        "tail_only, rel_tail_concat, src_rel_tail_concat."
    )


def _best_rank(scores: torch.Tensor, gold_mask: torch.Tensor) -> int:
    sorted_idx = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(sorted_idx)
    ranks[sorted_idx] = torch.arange(1, sorted_idx.numel() + 1, device=scores.device)
    return int(ranks[gold_mask].min().item())


def _choose_teacher_edge(
    *,
    gold_edge_ids: torch.Tensor,
    active_nodes: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    src = edge_index[0].index_select(0, gold_edge_ids)
    dst = edge_index[1].index_select(0, gold_edge_ids)
    activates_new_node = (~active_nodes.index_select(0, src)) | (
        ~active_nodes.index_select(0, dst)
    )
    preferred = gold_edge_ids[activates_new_node]
    if preferred.numel() == 0:
        preferred = gold_edge_ids
    return preferred.min().view(1)


def _aggregate_examples(
    examples: list[StateRankingExample], top_ks: list[int]
) -> dict[str, float | int]:
    if not examples:
        return {
            "num_states": 0,
            "mrr": 0.0,
            **{f"hit@{k}": 0.0 for k in top_ks},
            "mean_rank": 0.0,
            "median_rank": 0.0,
        }

    best_ranks = torch.tensor(
        [example.best_rank for example in examples], dtype=torch.float32
    )
    metrics: dict[str, float | int] = {
        "num_states": len(examples),
        "mrr": float((1.0 / best_ranks).mean().item()),
        "mean_rank": float(best_ranks.mean().item()),
        "median_rank": float(best_ranks.median().item()),
    }
    for k in top_ks:
        metrics[f"hit@{k}"] = float(best_ranks.le(float(k)).float().mean().item())
    return metrics


def _group_examples(
    examples: list[StateRankingExample],
    *,
    key_fn: Any,
) -> dict[str, list[StateRankingExample]]:
    grouped: dict[str, list[StateRankingExample]] = defaultdict(list)
    for example in examples:
        grouped[str(key_fn(example))].append(example)
    return dict(grouped)


def _evaluate_graph(
    *,
    graph_index: int,
    batch: Any,
    scoring_methods: list[str],
    path_mode: str,
    top_ks: list[int],
    rng: torch.Generator,
) -> dict[str, Any]:
    del top_ks  # aggregated at the corpus level; kept for signature symmetry.

    edge_index = batch.edge_index
    is_anchor_mask = batch.is_anchor_mask
    is_target_mask = batch.is_target_mask
    node_tokens = batch.node_tokens.float()
    edge_relation_tokens = batch.edge_relation_tokens.float()
    question_emb = batch.question_emb.view(-1).float()

    sp_labels = compute_shortest_path_labels(
        edge_index=edge_index.cpu(),
        is_anchor_mask=is_anchor_mask.cpu(),
        is_target_mask=is_target_mask.cpu(),
        num_nodes=batch.num_nodes,
        path_mode=path_mode,
    )

    if (
        sp_labels.positive_edge_ids.numel() == 0
        or sp_labels.reachable_target_node_ids.numel() == 0
    ):
        return {"status": "skipped_no_path"}

    positive_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    positive_edge_mask[sp_labels.positive_edge_ids.long()] = True

    rollout_state = RolloutState.initialize(batch)
    target_active = rollout_state.active_nodes & is_target_mask
    if bool(target_active.any().item()):
        return {"status": "skipped_root_hit"}

    src = edge_index[0]
    dst = edge_index[1]
    per_method_examples: dict[str, list[StateRankingExample]] = {
        method: [] for method in scoring_methods
    }

    for step_index in range(int(edge_index.size(1)) + 1):
        target_active = rollout_state.active_nodes & is_target_mask
        if bool(target_active.any().item()):
            break

        valid_edges = (
            rollout_state.active_nodes[src] | rollout_state.active_nodes[dst]
        ) & ~rollout_state.active_edges
        candidate_edge_ids = torch.nonzero(valid_edges, as_tuple=False).view(-1)
        if candidate_edge_ids.numel() == 0:
            break

        gold_mask_in_candidates = positive_edge_mask.index_select(0, candidate_edge_ids)
        if not bool(gold_mask_in_candidates.any().item()):
            break

        num_candidates = int(candidate_edge_ids.numel())
        num_gold_candidates = int(gold_mask_in_candidates.sum().item())
        for method in scoring_methods:
            scores = _score_candidates(
                method=method,
                query_emb=question_emb,
                node_tokens=node_tokens,
                edge_relation_tokens=edge_relation_tokens,
                edge_index=edge_index,
                candidate_edge_ids=candidate_edge_ids,
                rng=rng,
            )
            per_method_examples[method].append(
                StateRankingExample(
                    graph_index=graph_index,
                    step_index=step_index,
                    best_rank=_best_rank(scores, gold_mask_in_candidates),
                    num_candidates=num_candidates,
                    num_gold_candidates=num_gold_candidates,
                    path_length=sp_labels.max_path_length,
                )
            )

        teacher_gold_edges = candidate_edge_ids[gold_mask_in_candidates]
        chosen_teacher_edge = _choose_teacher_edge(
            gold_edge_ids=teacher_gold_edges,
            active_nodes=rollout_state.active_nodes,
            edge_index=edge_index,
        )
        rollout_state.apply_expansion(
            chosen_edges=chosen_teacher_edge, src=src, dst=dst
        )

    total_states = max(
        (len(examples) for examples in per_method_examples.values()), default=0
    )
    if total_states == 0:
        return {"status": "skipped_no_teacher_states"}
    return {"status": "ok", "examples": per_method_examples}


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage=None)

    if datamodule._shared_resources is None:
        raise RuntimeError(
            "Data resources were not initialized by the datamodule setup."
        )

    dataset = _select_dataset(datamodule, cfg.split)
    collator = RetrievalCollator(datamodule._shared_resources)

    chosen_indices = _sample_dataset_indices(
        len(dataset),
        max_graphs=cfg.max_graphs,
        seed=int(cfg.seed),
    )
    scoring_methods = [str(method) for method in cfg.scoring_methods]
    top_ks = sorted({int(k) for k in cfg.top_ks if int(k) >= 1})
    if not top_ks:
        raise ValueError("top_ks must contain at least one integer >= 1.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.seed))

    skipped_counts: dict[str, int] = defaultdict(int)
    examples_by_method: dict[str, list[StateRankingExample]] = {
        method: [] for method in scoring_methods
    }
    successful_graphs = 0

    log.info(
        "Evaluating semantic prior on split=%s with %d sampled graphs.",
        cfg.split,
        len(chosen_indices),
    )

    try:
        for graph_offset, dataset_idx in enumerate(chosen_indices):
            batch = collator([dataset[dataset_idx]])
            result = _evaluate_graph(
                graph_index=graph_offset,
                batch=batch,
                scoring_methods=scoring_methods,
                path_mode=str(cfg.path_mode),
                top_ks=top_ks,
                rng=rng,
            )
            status = str(result["status"])
            if status != "ok":
                skipped_counts[status] += 1
                continue
            successful_graphs += 1
            for method, examples in result["examples"].items():
                examples_by_method[method].extend(examples)
    finally:
        datamodule.teardown(stage=None)

    metrics: dict[str, Any] = {
        "split": str(cfg.split),
        "seed": int(cfg.seed),
        "path_mode": str(cfg.path_mode),
        "requested_max_graphs": None if cfg.max_graphs is None else int(cfg.max_graphs),
        "sampled_graphs": len(chosen_indices),
        "graphs_with_teacher_states": successful_graphs,
        "scoring_methods": scoring_methods,
        "top_ks": top_ks,
        "skipped": dict(sorted(skipped_counts.items())),
        "methods": {},
    }

    for method, examples in examples_by_method.items():
        path_groups = _group_examples(
            examples, key_fn=lambda item: _path_length_bucket(item.path_length)
        )
        candidate_groups = _group_examples(
            examples, key_fn=lambda item: _candidate_bucket(item.num_candidates)
        )
        metrics["methods"][method] = {
            "overall": _aggregate_examples(examples, top_ks),
            "by_path_length": {
                group_name: _aggregate_examples(group_examples, top_ks)
                for group_name, group_examples in sorted(path_groups.items())
            },
            "by_candidate_count": {
                group_name: _aggregate_examples(group_examples, top_ks)
                for group_name, group_examples in sorted(candidate_groups.items())
            },
        }

    return metrics


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="eval_semantic_prior.yaml",
)
def main(cfg: DictConfig) -> None:
    metrics = run_experiment(cfg)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_prior_oracle_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Semantic prior oracle metrics written to %s", output_path)
    log.info("%s", json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
