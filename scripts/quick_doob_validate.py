from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import rootutils
except ModuleNotFoundError:
    rootutils = None
else:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from lightning import seed_everything  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.schema import RetrievalBatch  # noqa: E402
from src.training.factory import build_datamodule, build_model  # noqa: E402
from src.training.resources import setup_datamodule  # noqa: E402
from src.weaver.nn.edge_scorer import EdgeScoreBreakdown  # noqa: E402
from src.weaver.reward import RewardModel, TerminalRewardOutput  # noqa: E402
from src.weaver.state import RolloutState, State  # noqa: E402


@dataclass
class MeanAgg:
    values: dict[str, list[float]]

    @classmethod
    def create(cls) -> "MeanAgg":
        return cls(values=defaultdict(list))

    def add(self, key: str, value: float | int | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().item())
        self.values[key].append(float(value))

    def extend(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.add(key, value)

    def mean(self) -> dict[str, float]:
        return {
            key: sum(vals) / float(len(vals)) if vals else 0.0
            for key, vals in sorted(self.values.items())
        }


@dataclass(frozen=True)
class CandidateSet:
    pos: torch.Tensor
    edge_ids: torch.Tensor
    graph_ids: torch.Tensor
    log_p0: torch.Tensor


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    return float(value)


def _rate(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0.0 else 0.0


def _load_shape_matched_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> tuple[int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = len(state_dict) - len(matched)
    missing = len(model_state) - len(matched)
    model.load_state_dict(matched, strict=False)
    return len(matched), missing, skipped


def _as_device_batch(batch: RetrievalBatch, device: torch.device) -> RetrievalBatch:
    return batch.to(device)  # type: ignore[return-value]


def _topk_prior_candidates(
    *,
    model: torch.nn.Module,
    batch: RetrievalBatch,
    state: State,
    rollout_context: object,
    active: torch.Tensor,
    top_k: int,
) -> CandidateSet:
    out = model.policy(
        batch,
        state,
        rollout_context=rollout_context,
        return_edge_breakdown=True,
        edge_logit_mode="semantic",
    )
    if not isinstance(out.edge_score_breakdown, EdgeScoreBreakdown):
        raise TypeError("Doob validation requires edge score breakdown.")
    prior_logits = out.edge_score_breakdown.semantic_logits.detach()
    candidate_batch = out.candidate_batch_ids.detach()
    candidate_edges = out.candidate_edge_ids.detach()

    selected: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    for graph_id in active.nonzero(as_tuple=False).view(-1).tolist():
        graph_id = int(graph_id)
        mask = candidate_batch.eq(graph_id)
        if not bool(mask.any()):
            continue
        pos = mask.nonzero(as_tuple=False).view(-1)
        vals = prior_logits.index_select(0, pos)
        k = min(int(top_k), int(vals.numel()))
        top_vals, order = torch.topk(vals, k=k)
        top_pos = pos.index_select(0, order)
        selected.append(top_pos)
        log_probs.append(torch.log_softmax(top_vals, dim=0))

    if not selected:
        empty = candidate_edges.new_empty((0,), dtype=torch.long)
        empty_float = prior_logits.new_empty((0,), dtype=torch.float32)
        return CandidateSet(
            pos=empty, edge_ids=empty, graph_ids=empty, log_p0=empty_float
        )

    pos = torch.cat(selected, dim=0)
    return CandidateSet(
        pos=pos,
        edge_ids=candidate_edges.index_select(0, pos),
        graph_ids=candidate_batch.index_select(0, pos),
        log_p0=torch.cat(log_probs, dim=0).to(dtype=torch.float32),
    )


def _candidate_next_state(
    *,
    batch: RetrievalBatch,
    state: State,
    candidates: CandidateSet,
) -> RolloutState:
    device = state.active_nodes.device
    rollout_to_graph = candidates.graph_ids.to(device=device, dtype=torch.long)
    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
    node_belongs = node_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
    edge_belongs = edge_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))

    active_nodes = state.active_nodes.view(1, -1).expand_as(node_belongs) & node_belongs
    active_edges = state.active_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs
    root_edges = state.root_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs

    next_state = RolloutState(
        active_nodes=active_nodes.clone(),
        active_edges=active_edges.clone(),
        root_edges=root_edges.clone(),
        anchor_nodes=active_nodes.clone(),
        rollout_to_graph=rollout_to_graph,
        expand_budget=int(state.expand_budget),
    )
    next_state.apply_expansion(
        rollout_ids=torch.arange(
            candidates.edge_ids.numel(),
            dtype=torch.long,
            device=device,
        ),
        chosen_edges=candidates.edge_ids,
        edge_index=batch.edge_index,
    )
    return next_state


def _successor_values(
    *,
    mode: str,
    model: torch.nn.Module,
    batch: RetrievalBatch,
    next_state: RolloutState,
    rollout_context: object,
    reward_model: RewardModel,
) -> torch.Tensor:
    if mode == "oracle":
        reward = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=next_state.active_nodes,
            active_edges=next_state.active_edges,
            state=next_state,
        )
        return reward.log_reward.to(dtype=torch.float32)
    if mode == "flow":
        context = model.policy.state_readout(
            fb=rollout_context,
            batch=batch,
            state=next_state,
        )
        return model.policy.flow_head(state_h=context.state_h).to(dtype=torch.float32)
    raise ValueError(f"Unsupported successor value mode: {mode!r}.")


@torch.no_grad()
def doob_greedy_rollout(
    *,
    model: torch.nn.Module,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    top_k: int,
    value_mode: str,
) -> dict[str, float]:
    expand_budget = int(model.expand_budget)
    state = State.create_initial(batch, expand_budget=expand_budget)
    rollout_context = model.policy.prepare_rollout_context(batch)
    device = batch.edge_index.device
    num_graphs = int(batch.num_graphs)
    active = torch.ones(num_graphs, dtype=torch.bool, device=device)
    stop_depth = torch.full(
        (num_graphs,), expand_budget, dtype=torch.long, device=device
    )
    first_hit_depth = torch.full((num_graphs,), -1, dtype=torch.long, device=device)

    candidate_counts: list[float] = []
    chosen_ranks: list[float] = []
    stop_probs: list[float] = []
    continue_probs: list[float] = []

    for depth in range(expand_budget + 1):
        current_reward = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )
        newly_hit = active & first_hit_depth.lt(0) & current_reward.answer_f1.gt(0.0)
        first_hit_depth[newly_hit] = int(depth)

        if depth >= expand_budget or not bool(active.any()):
            stopping = active.clone()
            stop_depth[stopping] = int(depth)
            active[stopping] = False
            break

        remaining = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=num_graphs,
        )
        can_expand = active & remaining.gt(0)
        if not bool(can_expand.any()):
            stopping = active.clone()
            stop_depth[stopping] = int(depth)
            active[stopping] = False
            break

        candidates = _topk_prior_candidates(
            model=model,
            batch=batch,
            state=state,
            rollout_context=rollout_context,
            active=can_expand,
            top_k=top_k,
        )
        if candidates.edge_ids.numel() == 0:
            stopping = active.clone()
            stop_depth[stopping] = int(depth)
            active[stopping] = False
            break

        next_state = _candidate_next_state(
            batch=batch, state=state, candidates=candidates
        )
        values = _successor_values(
            mode=value_mode,
            model=model,
            batch=batch,
            next_state=next_state,
            rollout_context=rollout_context,
            reward_model=reward_model,
        )
        edge_logits = candidates.log_p0.to(device=device) + values.to(device=device)

        chosen_edges: list[int] = []
        stopping_graphs: list[int] = []
        for graph_id in active.nonzero(as_tuple=False).view(-1).tolist():
            graph_id = int(graph_id)
            graph_mask = candidates.graph_ids.eq(graph_id)
            if not bool(graph_mask.any()) or not bool(can_expand[graph_id].item()):
                stopping_graphs.append(graph_id)
                continue

            pos = graph_mask.nonzero(as_tuple=False).view(-1)
            graph_edge_logits = edge_logits.index_select(0, pos)
            continue_logit = torch.logsumexp(graph_edge_logits, dim=0)
            stop_logit = current_reward.log_reward[graph_id].to(dtype=torch.float32)
            option = torch.softmax(torch.stack([stop_logit, continue_logit]), dim=0)
            stop_probs.append(float(option[0].item()))
            continue_probs.append(float(option[1].item()))
            candidate_counts.append(float(pos.numel()))

            if stop_logit >= continue_logit:
                stopping_graphs.append(graph_id)
                continue

            best_local = int(graph_edge_logits.argmax().item())
            best_pos = pos[best_local]
            chosen_edges.append(int(candidates.edge_ids[best_pos].item()))
            rank = 1.0 + float(
                (graph_edge_logits > graph_edge_logits[best_local]).sum().item()
            )
            chosen_ranks.append(rank)

        if stopping_graphs:
            ids = torch.tensor(stopping_graphs, dtype=torch.long, device=device)
            stop_depth[ids] = int(depth)
            active[ids] = False

        if chosen_edges:
            state.apply_expansion(
                chosen_edges=torch.tensor(
                    chosen_edges, dtype=torch.long, device=device
                ),
                edge_index=batch.edge_index,
            )

    final_reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    budget_exhausted = stop_depth.ge(expand_budget)
    hit = final_reward.answer_f1.gt(0.0)
    hit_first = first_hit_depth.ge(0)
    return {
        f"doob_{value_mode}/nonzero_f1_rate": _safe_float(hit.float().mean()),
        f"doob_{value_mode}/answer_f1_mean": _safe_float(final_reward.answer_f1.mean()),
        f"doob_{value_mode}/utility_mean": _safe_float(final_reward.utility.mean()),
        f"doob_{value_mode}/log_reward_mean": _safe_float(
            final_reward.log_reward.mean()
        ),
        f"doob_{value_mode}/expanded_edge_count_mean": _safe_float(
            final_reward.expanded_edge_count.mean()
        ),
        f"doob_{value_mode}/budget_exhausted_rate": _safe_float(
            budget_exhausted.float().mean()
        ),
        f"doob_{value_mode}/stop_depth_mean": _safe_float(stop_depth.float().mean()),
        f"doob_{value_mode}/first_hit_depth_mean": (
            _safe_float(first_hit_depth[hit_first].float().mean())
            if bool(hit_first.any())
            else 0.0
        ),
        f"doob_{value_mode}/candidate_count_mean": _mean(candidate_counts),
        f"doob_{value_mode}/chosen_rank_mean": _mean(chosen_ranks),
        f"doob_{value_mode}/stop_prob_mean": _mean(stop_probs),
        f"doob_{value_mode}/continue_prob_mean": _mean(continue_probs),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _format_metric_value(value: float) -> str:
    value = float(value)
    if value != 0.0 and abs(value) < 1.0e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _write_report(
    *,
    output_path: Path,
    checkpoint_path: str | None,
    sample_count: int,
    metrics: dict[str, float],
) -> None:
    lines = [
        "# Quick Doob Validation",
        "",
        f"- checkpoint: `{checkpoint_path or 'random initialization'}`",
        f"- validation samples: {sample_count}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"| `{key}` | {_format_metric_value(value)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="diagnose_weaver_rollout",
)
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed), workers=True)
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(datamodule, stage="fit")
    model = build_model(cfg, resources)

    ckpt_path = str(cfg.get("diagnose_ckpt_path") or cfg.get("ckpt_path") or "")
    if not ckpt_path:
        ckpt_cfg = cfg.get("ckpt", None)
        if ckpt_cfg is not None:
            ckpt_path = str(ckpt_cfg.get("path") or ckpt_cfg.get("pretrained") or "")
    if ckpt_path:
        matched, missing, skipped = _load_shape_matched_checkpoint(model, ckpt_path)
        print(
            "Loaded shape-matched checkpoint "
            f"{ckpt_path!r}; matched={matched}, missing={missing}, skipped={skipped}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = datamodule.val_dataloader()
    limit = int(cfg.get("limit", 32))
    top_k = int(cfg.get("doob_top_k", 32))
    aggs = MeanAgg.create()
    sample_count = 0

    for batch in loader:
        if sample_count >= limit:
            break
        batch = _as_device_batch(batch, device)
        sample_count += int(batch.num_graphs)
        aggs.extend(
            doob_greedy_rollout(
                model=model,
                batch=batch,
                reward_model=model.reward_model,
                top_k=top_k,
                value_mode="oracle",
            )
        )
        aggs.extend(
            doob_greedy_rollout(
                model=model,
                batch=batch,
                reward_model=model.reward_model,
                top_k=top_k,
                value_mode="flow",
            )
        )

    original_cwd = Path(get_original_cwd())
    output_path = Path(str(cfg.get("output_path", "quick_doob_validation.md")))
    if not output_path.is_absolute():
        output_path = original_cwd / output_path
    _write_report(
        output_path=output_path,
        checkpoint_path=ckpt_path or None,
        sample_count=min(sample_count, limit),
        metrics=aggs.mean(),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
