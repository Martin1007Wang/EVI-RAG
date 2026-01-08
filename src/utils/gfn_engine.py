from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from src.metrics import gflownet as gfn_metrics
from src.models.components import GFlowNetActor, GraphEnv, RewardOutput, RolloutResult
from src.models.components.gflownet_env import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    STOP_RELATION,
)
from src.utils.gfn import GFlowNetBatchProcessor, GFlowNetInputValidator, RolloutInputs, pool_nodes_mean_by_ptr

_ZERO = 0
_ONE = 1
_NAN = float("nan")
_DIST_UNREACHABLE = -1


@dataclass(frozen=True)
class GFlowNetRolloutConfig:
    num_rollouts: int
    eval_rollout_prefixes: Sequence[int]
    eval_rollout_temperature: float
    vectorized_rollouts: bool
    is_training: bool

    def __post_init__(self) -> None:
        if self.num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be > 0, got {self.num_rollouts}.")
        if self.eval_rollout_temperature < 0.0:
            raise ValueError(
                f"eval_rollout_temperature must be >= 0, got {self.eval_rollout_temperature}."
            )

    def use_vectorized(self) -> bool:
        return bool(self.vectorized_rollouts and self.num_rollouts > 1)


class GFlowNetEngine:
    def __init__(
        self,
        *,
        actor: GFlowNetActor,
        reward_fn: torch.nn.Module,
        env: GraphEnv,
        log_z: torch.nn.Module,
        batch_processor: GFlowNetBatchProcessor,
        input_validator: GFlowNetInputValidator,
        reward_embedding_source: str,
        vectorized_rollouts: bool,
    ) -> None:
        self.actor = actor
        self.reward_fn = reward_fn
        self.env = env
        self.log_z = log_z
        self.batch_processor = batch_processor
        self.input_validator = input_validator
        self.reward_embedding_source = str(reward_embedding_source).strip().lower()
        self.vectorized_rollouts = bool(vectorized_rollouts)
        if self.reward_embedding_source not in ("raw", "backbone"):
            raise ValueError(
                "reward_embedding_source must be 'raw' or 'backbone', "
                f"got {self.reward_embedding_source!r}."
            )

    def compute_batch_loss(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=rollout_cfg.is_training,
        )
        full_inputs = self.batch_processor.prepare_full_rollout_inputs(batch, device)
        num_graphs = int(full_inputs.node_ptr.numel() - 1)
        vectorized = rollout_cfg.use_vectorized()
        inputs = (
            self.batch_processor.repeat_rollout_inputs(full_inputs, rollout_cfg.num_rollouts)
            if vectorized
            else full_inputs
        )
        graph_cache = self.batch_processor.build_graph_cache(inputs, device=device)
        edge_debug = self._compute_edge_debug_metrics(inputs, graph_cache, device=device)
        log_z = self._compute_log_z(inputs)
        graph_mask = ~inputs.dummy_mask
        temperature = None if rollout_cfg.is_training else rollout_cfg.eval_rollout_temperature
        if vectorized:
            loss, metrics = self._compute_vectorized_rollout_loss(
                inputs=inputs,
                num_rollouts=rollout_cfg.num_rollouts,
                num_graphs=num_graphs,
                graph_cache=graph_cache,
                log_z=log_z,
                graph_mask=graph_mask,
                temperature=temperature,
                rollout_cfg=rollout_cfg,
            )
            if edge_debug:
                edge_debug = self._reduce_rollout_metrics(
                    edge_debug,
                    num_rollouts=rollout_cfg.num_rollouts,
                    num_graphs=num_graphs,
                    best_of=False,
                )
                metrics.update({k: v.detach() for k, v in edge_debug.items()})
            return loss, metrics
        loss, metrics = self._compute_loop_rollout_loss(
            inputs=inputs,
            num_rollouts=rollout_cfg.num_rollouts,
            num_graphs=num_graphs,
            graph_cache=graph_cache,
            log_z=log_z,
            graph_mask=graph_mask,
            temperature=temperature,
            rollout_cfg=rollout_cfg,
        )
        if edge_debug:
            metrics.update({k: v.detach() for k, v in edge_debug.items()})
        return loss, metrics

    def compute_rollout_records(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> list[Dict[str, Any]]:
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=False,
        )
        inputs = self.batch_processor.prepare_rollout_inputs(batch, device)
        num_graphs = int(inputs.node_ptr.numel() - 1)
        graph_cache = self.batch_processor.build_graph_cache(inputs, device=device)
        rollout_logs: list[Dict[str, torch.Tensor]] = []
        for _ in range(rollout_cfg.num_rollouts):
            rollout = self.actor.rollout(
                graph=graph_cache,
                temperature=rollout_cfg.eval_rollout_temperature,
            )
            rollout_logs.append(
                {
                    "actions_seq": rollout.actions_seq.detach().cpu(),
                    "log_pf": rollout.log_pf.detach().cpu(),
                    "directions_seq": rollout.directions_seq.detach().cpu(),
                    "reach_success": rollout.reach_success.detach().cpu(),
                    "stop_node_locals": rollout.stop_node_locals.detach().cpu(),
                }
            )
        return self._build_rollout_records(
            batch=batch,
            rollout_logs=rollout_logs,
            node_ptr=inputs.node_ptr.detach().cpu(),
            edge_ptr=inputs.edge_ptr.detach().cpu(),
            edge_index=inputs.edge_index.detach().cpu(),
            edge_relations=inputs.edge_relations.detach().cpu(),
            num_graphs=num_graphs,
        )

    def sample_edge_targets(
        self,
        *,
        batch: Any,
        device: torch.device,
        num_rollouts: int,
        temperature: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=False,
        )
        inputs = self.batch_processor.prepare_rollout_inputs(batch, device)
        num_graphs = int(inputs.node_ptr.numel() - 1)
        graph_cache = self.batch_processor.build_graph_cache(inputs, device=device)
        edge_targets = torch.zeros(inputs.edge_batch.numel(), device=device, dtype=torch.bool)
        success_counts = torch.zeros(num_graphs, device=device, dtype=torch.long)
        for _ in range(num_rollouts):
            rollout = self.actor.rollout(
                graph=graph_cache,
                temperature=temperature,
            )
            reach_success = rollout.reach_success.to(device=device, dtype=torch.bool)
            success_counts += reach_success.to(dtype=torch.long)
            selected_mask = rollout.selected_mask.to(device=device, dtype=torch.bool)
            if selected_mask.numel() != edge_targets.numel():
                raise ValueError("selected_mask length mismatch in sample_edge_targets.")
            edge_targets |= selected_mask & reach_success[inputs.edge_batch]
        metrics = {
            "success_counts": success_counts,
            "num_rollouts": torch.full_like(success_counts, int(num_rollouts)),
        }
        return edge_targets.to(dtype=torch.float32), metrics

    def _validate_batch_inputs(
        self,
        batch: Any,
        *,
        device: torch.device,
        require_rollout: bool,
        is_training: bool,
    ) -> None:
        if require_rollout:
            self.input_validator.validate_rollout_batch(
                batch,
                device=device,
                is_training=is_training,
            )
            return
        self.input_validator.validate_edge_batch(batch, device=device)

    def _build_reward_kwargs(
        self,
        rollout: RolloutResult,
        *,
        inputs: RolloutInputs,
    ) -> Dict[str, Any]:
        if self.reward_embedding_source == "raw":
            node_tokens = inputs.reward_node_embeddings
            question_tokens = inputs.reward_question_emb
        else:
            node_tokens = inputs.node_tokens
            question_tokens = inputs.question_tokens
        return {
            "answer_hit": rollout.reach_success,
            "answer_node_locals": inputs.answer_node_locals,
            "dummy_mask": inputs.dummy_mask,
            "edge_index": inputs.edge_index,
            "node_ptr": inputs.node_ptr,
            "node_min_dists": inputs.node_min_dists,
            "node_tokens": node_tokens,
            "path_length": rollout.length,
            "question_tokens": question_tokens,
            "start_node_locals": inputs.start_node_locals,
            "start_ptr": inputs.start_ptr,
            "stop_node_locals": rollout.stop_node_locals,
        }

    @staticmethod
    def _reduce_rollout_metrics(
        metrics: Dict[str, torch.Tensor],
        *,
        num_rollouts: int,
        num_graphs: int,
        best_of: bool,
    ) -> Dict[str, torch.Tensor]:
        return gfn_metrics.reduce_rollout_metrics(
            metrics,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=best_of,
        )

    @staticmethod
    def _stack_rollout_metrics(metrics_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return gfn_metrics.stack_rollout_metrics(metrics_list)

    def _finalize_rollout_metrics(
        self,
        loss_list: list[torch.Tensor],
        metrics_list: list[Dict[str, torch.Tensor]],
        *,
        num_rollouts: int,
        num_graphs: int,
        best_of: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return gfn_metrics.finalize_rollout_metrics(
            loss_list,
            metrics_list,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=best_of,
        )

    def _finalize_loop_rollout_metrics(
        self,
        *,
        loss_list: list[torch.Tensor],
        metrics_list: list[Dict[str, torch.Tensor]],
        num_rollouts: int,
        num_graphs: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._finalize_rollout_metrics(
            loss_list,
            metrics_list,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=False,
        )

    def _run_rollout_and_metrics(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        log_z: torch.Tensor,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
    ) -> tuple[RolloutResult, torch.Tensor, Dict[str, torch.Tensor]]:
        rollout = self.actor.rollout(
            graph=graph_cache,
            temperature=temperature,
        )
        tb_loss, metrics = self._compute_rollout_loss_metrics(
            rollout=rollout,
            inputs=inputs,
            log_z=log_z,
            graph_mask=graph_mask,
        )
        return rollout, tb_loss, metrics

    def _compute_rollout_loss_metrics(
        self,
        *,
        rollout: RolloutResult,
        inputs: RolloutInputs,
        log_z: torch.Tensor,
        graph_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reward_out: RewardOutput = self.reward_fn(
            **self._build_reward_kwargs(rollout, inputs=inputs)
        )
        log_reward_for_loss = torch.where(
            inputs.dummy_mask,
            torch.zeros_like(reward_out.log_reward),
            reward_out.log_reward,
        )
        tb_loss = self._compute_tb_loss(
            log_pf_steps=rollout.log_pf_steps,
            log_pb_steps=rollout.log_pb_steps,
            log_z=log_z,
            log_reward=log_reward_for_loss,
            lengths=rollout.length.long(),
            graph_mask=graph_mask,
        )
        self._raise_if_non_finite_loss(
            tb_loss=tb_loss,
            log_pf_steps=rollout.log_pf_steps,
            log_pb_steps=rollout.log_pb_steps,
            log_z=log_z,
            log_reward=log_reward_for_loss,
            lengths=rollout.length.long(),
            dummy_mask=inputs.dummy_mask,
        )
        metrics = self._build_tb_metrics(
            rollout=rollout,
            reward_out=reward_out,
            tb_loss=tb_loss,
            log_reward=log_reward_for_loss,
            log_z=log_z,
        )
        return tb_loss, metrics

    def _raise_if_non_finite_loss(
        self,
        *,
        tb_loss: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_z: torch.Tensor,
        log_reward: torch.Tensor,
        lengths: torch.Tensor,
        dummy_mask: torch.Tensor,
    ) -> None:
        if torch.isfinite(tb_loss).all():
            return
        num_steps = int(log_pf_steps.size(1))
        step_mask = self._build_step_mask(lengths, num_steps).to(dtype=log_pf_steps.dtype)
        sum_log_pf = (log_pf_steps * step_mask).sum(dim=1)
        sum_log_pb = (log_pb_steps * step_mask).sum(dim=1)
        residual = log_z + sum_log_pf - sum_log_pb - log_reward
        message = self._format_non_finite_report(
            tb_loss=tb_loss,
            log_z=log_z,
            log_reward=log_reward,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            sum_log_pf=sum_log_pf,
            sum_log_pb=sum_log_pb,
            residual=residual,
        )
        dummy_mask = dummy_mask.to(dtype=torch.bool)
        dummy_total = int(dummy_mask.numel())
        dummy_count = int(dummy_mask.sum().item())
        dummy_ratio = float(dummy_count) / float(dummy_total) if dummy_total > _ZERO else float(_ZERO)
        extra = [
            f"dummy_mask: count={dummy_count}, total={dummy_total}, ratio={dummy_ratio}",
            self._summarize_tensor_stats("log_z_stats", log_z),
            self._summarize_tensor_stats("log_reward_stats", log_reward),
            self._summarize_tensor_stats("sum_log_pf_stats", sum_log_pf),
            self._summarize_tensor_stats("sum_log_pb_stats", sum_log_pb),
            self._summarize_tensor_stats("residual_stats", residual),
            self._summarize_tensor_stats("lengths_stats", lengths),
        ]
        message = "\n".join([message, *extra])
        raise RuntimeError(message)

    def _format_non_finite_report(
        self,
        *,
        tb_loss: torch.Tensor,
        log_z: torch.Tensor,
        log_reward: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        sum_log_pf: torch.Tensor,
        sum_log_pb: torch.Tensor,
        residual: torch.Tensor,
    ) -> str:
        header = "Non-finite TB loss detected; training aborted."
        summary = self._summarize_tensor("tb_loss", tb_loss)
        lines = [header]
        if summary:
            lines.append(summary)
        detail = self._summarize_tensor("log_z", log_z)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("log_reward", log_reward)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("log_pf_steps", log_pf_steps)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("log_pb_steps", log_pb_steps)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("sum_log_pf", sum_log_pf)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("sum_log_pb", sum_log_pb)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("residual", residual)
        if detail:
            lines.append(detail)
        if len(lines) == _ONE:
            lines.append("No additional non-finite components detected.")
        return "\n".join(lines)

    @staticmethod
    def _summarize_tensor(name: str, tensor: torch.Tensor) -> str | None:
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            return None
        non_finite = ~finite
        num_non_finite = int(non_finite.sum().item())
        num_nan = int(torch.isnan(tensor).sum().item())
        num_inf = int(torch.isinf(tensor).sum().item())
        finite_vals = tensor[finite]
        if finite_vals.numel() > _ZERO:
            finite_min = float(finite_vals.min().item())
            finite_max = float(finite_vals.max().item())
        else:
            finite_min = _NAN
            finite_max = _NAN
        return (
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"non_finite={num_non_finite} (nan={num_nan}, inf={num_inf}), "
            f"finite_min={finite_min}, finite_max={finite_max}"
        )

    @staticmethod
    def _summarize_tensor_stats(name: str, tensor: torch.Tensor) -> str:
        if tensor.numel() == 0:
            return f"{name}: empty"
        finite = torch.isfinite(tensor)
        non_finite = int((~finite).sum().item())
        finite_vals = tensor[finite]
        if finite_vals.numel() == 0:
            return f"{name}: non_finite={non_finite} (all)"
        calc = finite_vals.to(dtype=torch.float32)
        min_val = float(calc.min().item())
        max_val = float(calc.max().item())
        mean_val = float(calc.mean().item())
        std_val = float(calc.std(unbiased=False).item())
        abs_max = float(calc.abs().max().item())
        q_tensor = torch.quantile(
            calc,
            torch.tensor((0.0, 0.5, 0.9, 0.99, 1.0), device=calc.device, dtype=calc.dtype),
        )
        q_vals = [float(q.item()) for q in q_tensor]
        q_parts = " ".join(f"q{idx}={val}" for idx, val in enumerate(q_vals))
        return (
            f"{name}: non_finite={non_finite}, min={min_val}, max={max_val}, "
            f"mean={mean_val}, std={std_val}, abs_max={abs_max}, {q_parts}"
        )

    @staticmethod
    def _build_step_mask(lengths: torch.Tensor, num_steps: int) -> torch.Tensor:
        if lengths.dim() != 1:
            raise ValueError("lengths must be [B] for step mask.")
        step_ids = torch.arange(num_steps, device=lengths.device, dtype=lengths.dtype).view(1, -1)
        return step_ids <= lengths.view(-1, 1)

    def _compute_tb_loss(
        self,
        *,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_z: torch.Tensor,
        log_reward: torch.Tensor,
        lengths: torch.Tensor,
        graph_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_steps = int(log_pf_steps.size(1))
        step_mask = self._build_step_mask(lengths, num_steps).to(dtype=log_pf_steps.dtype)
        sum_log_pf = (log_pf_steps * step_mask).sum(dim=1)
        sum_log_pb = (log_pb_steps * step_mask).sum(dim=1)
        residual = log_z + sum_log_pf - sum_log_pb - log_reward
        if graph_mask is not None:
            mask = graph_mask.to(dtype=residual.dtype).view(-1)
            residual = torch.where(mask > _ZERO, residual, torch.zeros_like(residual))
        loss_per_graph = residual.pow(2)
        return self._reduce_graph_loss(loss_per_graph, graph_mask)

    @staticmethod
    def _reduce_graph_loss(loss_per_graph: torch.Tensor, graph_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if graph_mask is None:
            return loss_per_graph.mean()
        weights = graph_mask.to(dtype=loss_per_graph.dtype)
        denom = weights.sum().clamp(min=float(_ONE))
        return (loss_per_graph * weights).sum() / denom

    def _compute_edge_debug_metrics(
        self,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        state_vec = self.actor.state_encoder.init_state(
            num_graphs,
            device=inputs.node_tokens.device,
            dtype=inputs.node_tokens.dtype,
        )
        edge_scores = self.actor._score_edges_forward(
            graph_cache,
            state_vec,
            active_nodes=graph_cache["node_is_start"],
            autocast_ctx=self.actor._autocast_context(device),
        )
        return gfn_metrics.compute_edge_debug_metrics(
            edge_scores=edge_scores,
            edge_batch=inputs.edge_batch,
            edge_index=inputs.edge_index,
            node_ptr=inputs.node_ptr,
            node_min_dists=inputs.node_min_dists,
            start_ptr=inputs.start_ptr,
            dummy_mask=inputs.dummy_mask,
            node_is_start=graph_cache["node_is_start"],
            node_is_answer=graph_cache["node_is_answer"],
            node_batch=graph_cache["node_batch"],
            stop_on_answer=bool(self.env.stop_on_answer),
        )

    @staticmethod
    def _build_tb_metrics(
        *,
        rollout: RolloutResult,
        reward_out: RewardOutput,
        tb_loss: torch.Tensor,
        log_reward: torch.Tensor,
        log_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return gfn_metrics.build_tb_metrics(
            rollout=rollout,
            reward_out=reward_out,
            tb_loss=tb_loss,
            log_reward=log_reward,
            log_z=log_z,
        )

    @staticmethod
    def _build_log_z_graph_stats(
        *,
        node_ptr: torch.Tensor,
        edge_ptr: torch.Tensor,
        start_ptr: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        node_counts = (node_ptr[1:] - node_ptr[:-1]).to(dtype=dtype)
        edge_counts = (edge_ptr[1:] - edge_ptr[:-1]).to(dtype=dtype)
        start_counts = (start_ptr[1:] - start_ptr[:-1]).to(dtype=dtype)
        counts = torch.stack((node_counts, edge_counts, start_counts), dim=1)
        return torch.log1p(counts)

    def _compute_log_z(self, inputs: RolloutInputs) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        start_tokens = pool_nodes_mean_by_ptr(
            node_tokens=inputs.node_tokens,
            node_locals=inputs.start_node_locals,
            ptr=inputs.start_ptr,
            num_graphs=num_graphs,
        )
        graph_stats = self._build_log_z_graph_stats(
            node_ptr=inputs.node_ptr,
            edge_ptr=inputs.edge_ptr,
            start_ptr=inputs.start_ptr,
            dtype=inputs.question_tokens.dtype,
        )
        return self.log_z(
            question_tokens=inputs.question_tokens,
            start_tokens=start_tokens,
            graph_stats=graph_stats,
        )

    def _compute_vectorized_rollout_loss(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        graph_cache: Dict[str, torch.Tensor],
        log_z: torch.Tensor,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rollout, tb_loss, metrics = self._run_rollout_and_metrics(
            inputs=inputs,
            graph_cache=graph_cache,
            log_z=log_z,
            graph_mask=graph_mask,
            temperature=temperature,
        )
        raw_metrics = metrics
        metrics = self._reduce_rollout_metrics(
            metrics,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=False,
        )
        if "log_reward" in raw_metrics and "pass@1" in raw_metrics:
            metrics["reward_gap"] = gfn_metrics.compute_reward_gap(
                log_reward=raw_metrics["log_reward"],
                pass_hits=raw_metrics["pass@1"],
                num_rollouts=num_rollouts,
                num_graphs=num_graphs,
            )
        k_values = rollout_cfg.eval_rollout_prefixes if not rollout_cfg.is_training else [num_rollouts]
        if k_values:
            terminal_hits = gfn_metrics.compute_terminal_hits(
                stop_node_locals=rollout.stop_node_locals,
                node_ptr=inputs.node_ptr,
                node_is_answer=graph_cache["node_is_answer"],
            ).reshape(num_rollouts, num_graphs)
            metrics.update(
                gfn_metrics.compute_terminal_hit_prefixes(
                    terminal_hits=terminal_hits,
                    k_values=k_values,
                )
            )
        if not rollout_cfg.is_training:
            node_ptr = inputs.node_ptr[: num_graphs + _ONE]
            total_nodes = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else _ZERO
            expected_nodes = total_nodes * num_rollouts
            visited_nodes = rollout.visited_nodes
            if visited_nodes.numel() != expected_nodes:
                raise ValueError("visited_nodes length mismatch for eval metrics.")
            visited_stack = visited_nodes.reshape(num_rollouts, total_nodes)
            metrics = self._attach_eval_metrics(
                metrics=metrics,
                actions_seq=rollout.actions_seq,
                directions_seq=rollout.directions_seq,
                visited_stack=visited_stack,
                inputs=inputs,
                num_rollouts=num_rollouts,
                num_graphs=num_graphs,
                k_values=rollout_cfg.eval_rollout_prefixes,
            )
        return tb_loss, metrics

    def _collect_rollout_outputs(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        graph_cache: Dict[str, torch.Tensor],
        log_z: torch.Tensor,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> tuple[
        list[torch.Tensor],
        list[Dict[str, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        loss_list: list[torch.Tensor] = []
        metrics_list: list[Dict[str, torch.Tensor]] = []
        rollout_stop_nodes: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_directions: list[torch.Tensor] = []
        rollout_visited: list[torch.Tensor] = []
        collect_eval = not rollout_cfg.is_training
        for _ in range(num_rollouts):
            rollout, tb_loss, metrics = self._run_rollout_and_metrics(
                inputs=inputs,
                graph_cache=graph_cache,
                log_z=log_z,
                graph_mask=graph_mask,
                temperature=temperature,
            )
            rollout_stop_nodes.append(rollout.stop_node_locals.detach())
            if collect_eval:
                rollout_actions.append(rollout.actions_seq.detach())
                rollout_directions.append(rollout.directions_seq.detach())
                rollout_visited.append(rollout.visited_nodes.detach())
            loss_list.append(tb_loss)
            metrics_list.append(metrics)
        return (
            loss_list,
            metrics_list,
            rollout_stop_nodes,
            rollout_actions,
            rollout_directions,
            rollout_visited,
        )

    def _compute_loop_rollout_loss(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        graph_cache: Dict[str, torch.Tensor],
        log_z: torch.Tensor,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        (
            loss_list,
            metrics_list,
            rollout_stop_nodes,
            rollout_actions,
            rollout_directions,
            rollout_visited,
        ) = self._collect_rollout_outputs(
            inputs=inputs,
            num_rollouts=num_rollouts,
            graph_cache=graph_cache,
            log_z=log_z,
            graph_mask=graph_mask,
            temperature=temperature,
            rollout_cfg=rollout_cfg,
        )
        loss, metrics = self._finalize_loop_rollout_metrics(
            loss_list=loss_list,
            metrics_list=metrics_list,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        stacked = self._stack_rollout_metrics(metrics_list)
        if "log_reward" in stacked and "pass@1" in stacked:
            metrics["reward_gap"] = gfn_metrics.compute_reward_gap(
                log_reward=stacked["log_reward"],
                pass_hits=stacked["pass@1"],
                num_rollouts=num_rollouts,
                num_graphs=num_graphs,
            )
        k_values = rollout_cfg.eval_rollout_prefixes if not rollout_cfg.is_training else [num_rollouts]
        if k_values:
            terminal_hits = torch.stack(
                [
                    gfn_metrics.compute_terminal_hits(
                        stop_node_locals=stop_nodes,
                        node_ptr=inputs.node_ptr,
                        node_is_answer=graph_cache["node_is_answer"],
                    )
                    for stop_nodes in rollout_stop_nodes
                ],
                dim=0,
            )
            metrics.update(
                gfn_metrics.compute_terminal_hit_prefixes(
                    terminal_hits=terminal_hits,
                    k_values=k_values,
                )
            )
        if not rollout_cfg.is_training:
            metrics = self._attach_eval_metrics_from_rollout_lists(
                metrics=metrics,
                rollout_actions=rollout_actions,
                rollout_directions=rollout_directions,
                rollout_visited=rollout_visited,
                inputs=inputs,
                num_rollouts=num_rollouts,
                num_graphs=num_graphs,
                k_values=rollout_cfg.eval_rollout_prefixes,
            )
        return loss, metrics

    def _attach_eval_metrics(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        actions_seq: torch.Tensor,
        directions_seq: torch.Tensor,
        visited_stack: torch.Tensor,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        k_values: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        node_ptr = inputs.node_ptr[: num_graphs + _ONE]
        answer_ptr = inputs.answer_ptr[: num_graphs + _ONE]
        start_ptr = inputs.start_ptr[: num_graphs + _ONE]
        total_nodes = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else _ZERO
        total_starts = int(start_ptr[-1].item()) if start_ptr.numel() > 0 else _ZERO
        total_answers = int(answer_ptr[-1].item()) if answer_ptr.numel() > 0 else _ZERO
        start_nodes = inputs.start_node_locals
        answer_nodes = inputs.answer_node_locals
        node_is_start = torch.zeros(total_nodes, device=visited_stack.device, dtype=torch.bool)
        node_is_answer = torch.zeros(total_nodes, device=visited_stack.device, dtype=torch.bool)
        if total_starts > _ZERO:
            node_is_start[start_nodes[:total_starts].to(device=visited_stack.device, dtype=torch.long)] = True
        if total_answers > _ZERO:
            node_is_answer[answer_nodes[:total_answers].to(device=visited_stack.device, dtype=torch.long)] = True
        metrics.update(
            gfn_metrics.compute_context_metrics(
                visited_stack=visited_stack,
                node_ptr=node_ptr,
                node_is_answer=node_is_answer,
                node_is_start=node_is_start,
                answer_ptr=answer_ptr,
                k_values=k_values,
            )
        )
        metrics["path_diversity"] = gfn_metrics.compute_path_diversity(
            actions_seq=actions_seq,
            directions_seq=directions_seq,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            edge_ptr=inputs.edge_ptr,
        )
        context_stats = self._collect_context_debug_stats(
            visited_stack=visited_stack,
            inputs=inputs,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            node_is_start=node_is_start,
        )
        if context_stats:
            metrics.update({f"context_debug/{k}": v for k, v in context_stats.items()})
        return metrics

    def _attach_eval_metrics_from_rollout_lists(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        rollout_actions: list[torch.Tensor],
        rollout_directions: list[torch.Tensor],
        rollout_visited: list[torch.Tensor],
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        k_values: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        if not rollout_actions or not rollout_directions or not rollout_visited:
            raise ValueError("Missing rollout buffers for eval metrics.")
        actions_stack = torch.stack(rollout_actions, dim=0)
        directions_stack = torch.stack(rollout_directions, dim=0)
        num_steps = int(actions_stack.size(2))
        actions_seq = actions_stack.reshape(num_rollouts * num_graphs, num_steps)
        directions_seq = directions_stack.reshape(num_rollouts * num_graphs, num_steps)
        visited_stack = torch.stack(rollout_visited, dim=0)
        return self._attach_eval_metrics(
            metrics=metrics,
            actions_seq=actions_seq,
            directions_seq=directions_seq,
            visited_stack=visited_stack,
            inputs=inputs,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            k_values=k_values,
        )

    @staticmethod
    def _fraction(mask: torch.Tensor, *, denom_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if denom_mask is None:
            denom = torch.tensor(mask.numel(), device=mask.device, dtype=torch.float32)
        else:
            denom = denom_mask.to(dtype=torch.float32).sum()
        return mask.to(dtype=torch.float32).sum() / denom.clamp(min=float(_ONE))

    def _compute_start_out_stats(
        self,
        *,
        edge_index: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_counts: torch.Tensor,
        num_graphs: int,
        total_nodes: int,
        missing_start: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if total_nodes <= _ZERO or start_node_locals.numel() == 0:
            return {
                "start_out_zero_frac": torch.tensor(_ZERO, device=edge_index.device, dtype=torch.float32),
                "start_out_degree_mean": torch.tensor(_ZERO, device=edge_index.device, dtype=torch.float32),
            }
        out_degree = torch.zeros(total_nodes, device=edge_index.device, dtype=torch.long)
        if edge_index.numel() > 0:
            ones = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.long)
            out_degree.index_add_(0, edge_index[0], ones)
        start_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=edge_index.device),
            start_counts.clamp(min=_ZERO),
        )
        start_degrees = out_degree[start_node_locals]
        start_has_out = start_degrees > _ZERO
        start_has_out_counts = torch.zeros(num_graphs, device=edge_index.device, dtype=torch.long)
        start_has_out_counts.index_add_(0, start_batch, start_has_out.to(dtype=torch.long))
        start_out_zero = (start_has_out_counts == _ZERO) & (~missing_start)
        return {
            "start_out_zero_frac": self._fraction(start_out_zero),
            "start_out_degree_mean": start_degrees.to(dtype=torch.float32).mean(),
        }

    def _compute_min_start_dist_stats(
        self,
        *,
        node_min_dists: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_counts: torch.Tensor,
        num_graphs: int,
        missing_start: torch.Tensor,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        if start_node_locals.numel() == 0:
            zero = torch.tensor(_ZERO, device=node_min_dists.device, dtype=torch.float32)
            return {
                "reachable_frac": zero,
                "reachable_horizon_frac": zero,
                "min_start_dist_mean": zero,
            }
        start_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_min_dists.device),
            start_counts.clamp(min=_ZERO),
        )
        start_dists = node_min_dists[start_node_locals]
        init_val = torch.full(
            (num_graphs,),
            torch.iinfo(start_dists.dtype).max,
            device=node_min_dists.device,
            dtype=start_dists.dtype,
        )
        min_start_dist = init_val.scatter_reduce_(0, start_batch, start_dists, reduce="amin", include_self=True)
        min_start_dist = torch.where(
            missing_start,
            torch.full_like(min_start_dist, _DIST_UNREACHABLE),
            min_start_dist,
        )
        reachable = (min_start_dist >= _ZERO) & (~missing_start)
        reachable_count = reachable.to(dtype=torch.float32).sum()
        reachable_sum = (min_start_dist.to(dtype=torch.float32) * reachable.to(dtype=torch.float32)).sum()
        reachable_mean = reachable_sum / reachable_count.clamp(min=float(_ONE))
        return {
            "reachable_frac": self._fraction(reachable),
            "reachable_horizon_frac": self._fraction(reachable & (min_start_dist <= max_steps)),
            "min_start_dist_mean": reachable_mean,
        }

    def _compute_candidate_edge_stats(
        self,
        *,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        node_is_start: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        if edge_index.numel() == 0 or num_graphs <= _ZERO:
            zero = torch.tensor(_ZERO, device=edge_index.device, dtype=torch.float32)
            return {
                "candidate_edges_mean": zero,
                "candidate_edges_zero_frac": zero,
            }
        heads = edge_index[0]
        tails = edge_index[1]
        candidate_mask = node_is_start[heads] & (~node_is_start[tails])
        candidate_counts = torch.zeros(num_graphs, device=edge_index.device, dtype=torch.long)
        candidate_counts.index_add_(0, edge_batch, candidate_mask.to(dtype=torch.long))
        return {
            "candidate_edges_mean": candidate_counts.to(dtype=torch.float32).mean(),
            "candidate_edges_zero_frac": self._fraction(candidate_counts == _ZERO),
        }

    def _compute_visited_stats(
        self,
        *,
        visited_stack: torch.Tensor,
        node_ptr: torch.Tensor,
        answer_node_locals: torch.Tensor,
        answer_counts: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        if num_graphs <= _ZERO or visited_stack.numel() == 0:
            zero = torch.tensor(_ZERO, device=visited_stack.device, dtype=torch.float32)
            return {
                "visited_frac_first": zero,
                "answer_hit_first_frac": zero,
            }
        node_counts = node_ptr[1:] - node_ptr[:-1]
        node_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=visited_stack.device),
            node_counts.to(dtype=torch.long),
        )
        visited_first = visited_stack[0].to(dtype=torch.long)
        visited_counts = torch.zeros(num_graphs, device=visited_stack.device, dtype=torch.long)
        visited_counts.index_add_(0, node_batch, visited_first)
        node_counts_safe = node_counts.clamp(min=_ONE).to(dtype=torch.float32)
        visited_frac = (visited_counts.to(dtype=torch.float32) / node_counts_safe).mean()
        if answer_node_locals.numel() == 0:
            return {
                "visited_frac_first": visited_frac,
                "answer_hit_first_frac": torch.tensor(_ZERO, device=visited_stack.device, dtype=torch.float32),
            }
        answer_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=visited_stack.device),
            answer_counts.clamp(min=_ZERO),
        )
        answer_visited = visited_first[answer_node_locals].to(dtype=torch.long)
        answer_hit_counts = torch.zeros(num_graphs, device=visited_stack.device, dtype=torch.long)
        answer_hit_counts.index_add_(0, answer_batch, answer_visited)
        answer_hit = answer_hit_counts > _ZERO
        answer_has_any = answer_counts > _ZERO
        answer_hit_frac = self._fraction(answer_hit, denom_mask=answer_has_any)
        return {
            "visited_frac_first": visited_frac,
            "answer_hit_first_frac": answer_hit_frac,
        }

    @staticmethod
    def _tensor_stats_to_floats(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {key: float(val.item()) for key, val in stats.items()}

    def _build_context_debug_base(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
    ) -> tuple[Dict[str, float], Dict[str, torch.Tensor | int]]:
        if num_graphs <= _ZERO:
            return {}, {}
        node_ptr = inputs.node_ptr[: num_graphs + _ONE]
        edge_ptr = inputs.edge_ptr[: num_graphs + _ONE]
        start_ptr = inputs.start_ptr[: num_graphs + _ONE]
        answer_ptr = inputs.answer_ptr[: num_graphs + _ONE]
        total_nodes = int(node_ptr[-_ONE].item()) if node_ptr.numel() > _ZERO else _ZERO
        total_edges = int(edge_ptr[-_ONE].item()) if edge_ptr.numel() > _ZERO else _ZERO
        start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).to(dtype=torch.long)
        answer_counts = (answer_ptr[_ONE:] - answer_ptr[:-_ONE]).to(dtype=torch.long)
        missing_start = start_counts == _ZERO
        missing_answer = answer_counts == _ZERO
        dummy_mask = inputs.dummy_mask[:num_graphs].to(dtype=torch.bool)
        stats: Dict[str, float] = {
            "num_graphs": float(num_graphs),
            "num_rollouts": float(num_rollouts),
            "max_steps": float(getattr(self.env, "max_steps", _ZERO)),
            "total_nodes": float(total_nodes),
            "total_edges": float(total_edges),
            "missing_start_frac": float(self._fraction(missing_start).item()),
            "missing_answer_frac": float(self._fraction(missing_answer).item()),
            "dummy_frac": float(self._fraction(dummy_mask).item()),
            "start_count_mean": float(start_counts.to(dtype=torch.float32).mean().item()),
            "answer_count_mean": float(answer_counts.to(dtype=torch.float32).mean().item()),
        }
        context = {
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "start_counts": start_counts,
            "answer_counts": answer_counts,
            "missing_start": missing_start,
            "total_nodes": total_nodes,
        }
        return stats, context

    def _collect_context_debug_stats(
        self,
        *,
        visited_stack: torch.Tensor,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        node_is_start: torch.Tensor,
    ) -> Dict[str, float]:
        stats, context = self._build_context_debug_base(
            inputs=inputs,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        if not stats:
            return {}
        edge_index = inputs.edge_index
        edge_batch = inputs.edge_batch
        start_node_locals = inputs.start_node_locals
        answer_node_locals = inputs.answer_node_locals
        node_min_dists = inputs.node_min_dists
        node_ptr = context["node_ptr"]
        start_counts = context["start_counts"]
        answer_counts = context["answer_counts"]
        missing_start = context["missing_start"]
        total_nodes = int(context["total_nodes"])
        max_steps = int(getattr(self.env, "max_steps", _ZERO))
        stats.update(self._tensor_stats_to_floats(self._compute_start_out_stats(edge_index=edge_index, start_node_locals=start_node_locals, start_counts=start_counts, num_graphs=num_graphs, total_nodes=total_nodes, missing_start=missing_start)))
        stats.update(self._tensor_stats_to_floats(self._compute_min_start_dist_stats(node_min_dists=node_min_dists, start_node_locals=start_node_locals, start_counts=start_counts, num_graphs=num_graphs, missing_start=missing_start, max_steps=max_steps)))
        stats.update(self._tensor_stats_to_floats(self._compute_candidate_edge_stats(edge_index=edge_index, edge_batch=edge_batch, node_is_start=node_is_start, num_graphs=num_graphs)))
        stats.update(self._tensor_stats_to_floats(self._compute_visited_stats(visited_stack=visited_stack, node_ptr=node_ptr, answer_node_locals=answer_node_locals, answer_counts=answer_counts, num_graphs=num_graphs)))
        return stats

    @staticmethod
    def _extract_batch_meta(batch: Any, num_graphs: int) -> tuple[list[str], list[str]]:
        raw_ids = getattr(batch, "sample_id", None)
        if raw_ids is None:
            raise ValueError("Batch missing sample_id required for rollout artifacts.")
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError(f"batch.sample_id must be list/tuple, got {type(raw_ids)!r}.")
        sample_ids = [str(s) for s in raw_ids]
        if len(sample_ids) != num_graphs:
            raise ValueError(f"sample_id length {len(sample_ids)} != num_graphs {num_graphs}.")
        raw_q = getattr(batch, "question", None)
        if raw_q is None:
            questions = ["" for _ in range(num_graphs)]
        else:
            if not isinstance(raw_q, (list, tuple)):
                raise ValueError(f"batch.question must be list/tuple, got {type(raw_q)!r}.")
            questions = [str(q) for q in raw_q]
            if len(questions) != num_graphs:
                raise ValueError(f"question length {len(questions)} != num_graphs {num_graphs}.")
        return sample_ids, questions

    @staticmethod
    def _extract_answer_entity_ids(batch: Any, num_graphs: int) -> list[list[int]]:
        raw_ids = getattr(batch, "answer_entity_ids", None)
        if not torch.is_tensor(raw_ids):
            raise ValueError("Batch missing answer_entity_ids tensor required for rollout artifacts.")
        if raw_ids.dtype != torch.long:
            raise ValueError(f"answer_entity_ids must be torch.long, got {raw_ids.dtype}.")
        raw_ids = raw_ids.view(-1)
        ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if not torch.is_tensor(ptr):
            raise ValueError("Batch missing answer_entity_ids_ptr required for rollout artifacts.")
        if ptr.dtype != torch.long:
            raise ValueError(f"answer_entity_ids_ptr must be torch.long, got {ptr.dtype}.")
        ptr = ptr.view(-1)
        if ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"answer_entity_ids_ptr length {ptr.numel()} != num_graphs+1 ({num_graphs + _ONE}).")
        if int(ptr[0].item()) != _ZERO:
            raise ValueError("answer_entity_ids_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError("answer_entity_ids_ptr must be non-decreasing.")
        if int(ptr[-1].item()) != raw_ids.numel():
            raise ValueError(
                f"answer_entity_ids_ptr must end at {raw_ids.numel()}, got {int(ptr[-1].item())}."
            )
        answer_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].item())
            end = int(ptr[gid + 1].item())
            answer_lists.append([int(x) for x in raw_ids[start:end].detach().cpu().tolist()])
        return answer_lists

    @staticmethod
    def _extract_start_local_indices(batch: Any, num_graphs: int) -> list[list[int]]:
        raw = getattr(batch, "q_local_indices", None)
        if raw is None:
            return [[] for _ in range(num_graphs)]
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw, dtype=torch.long)
        raw = raw.view(-1)
        slice_dict = getattr(batch, "_slice_dict", None)
        ptr = None
        if isinstance(slice_dict, dict):
            ptr = slice_dict.get("q_local_indices")
        if ptr is None:
            ptr = getattr(batch, "q_local_indices_ptr", None)
        if not torch.is_tensor(ptr):
            raise ValueError("Batch missing q_local_indices_ptr required for rollout artifacts.")
        if ptr.dtype != torch.long:
            raise ValueError(f"q_local_indices_ptr must be torch.long, got {ptr.dtype}.")
        ptr = ptr.view(-1)
        if ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"q_local_indices_ptr length {ptr.numel()} != num_graphs+1 ({num_graphs + _ONE}).")
        if int(ptr[0].item()) != _ZERO:
            raise ValueError("q_local_indices_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError("q_local_indices_ptr must be non-decreasing.")
        if int(ptr[-1].item()) != raw.numel():
            raise ValueError(
                f"q_local_indices_ptr must end at {raw.numel()}, got {int(ptr[-1].item())}."
            )
        start_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].item())
            end = int(ptr[gid + 1].item())
            indices = raw[start:end]
            start_lists.append([int(x) for x in indices.detach().cpu().tolist()])
        return start_lists

    @staticmethod
    def _extract_start_entity_ids(
        batch: Any,
        num_graphs: int,
        node_global_ids: torch.Tensor,
    ) -> list[list[int]]:
        raw = getattr(batch, "q_local_indices", None)
        if raw is None:
            return [[] for _ in range(num_graphs)]
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw, dtype=torch.long)
        raw = raw.view(-1)
        slice_dict = getattr(batch, "_slice_dict", None)
        ptr = None
        if isinstance(slice_dict, dict):
            ptr = slice_dict.get("q_local_indices")
        if ptr is None:
            ptr = getattr(batch, "q_local_indices_ptr", None)
        if not torch.is_tensor(ptr):
            raise ValueError("Batch missing q_local_indices_ptr required for rollout artifacts.")
        if ptr.dtype != torch.long:
            raise ValueError(f"q_local_indices_ptr must be torch.long, got {ptr.dtype}.")
        ptr = ptr.view(-1)
        if ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"q_local_indices_ptr length {ptr.numel()} != num_graphs+1 ({num_graphs + _ONE}).")
        if int(ptr[0].item()) != _ZERO:
            raise ValueError("q_local_indices_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError("q_local_indices_ptr must be non-decreasing.")
        if int(ptr[-1].item()) != raw.numel():
            raise ValueError(
                f"q_local_indices_ptr must end at {raw.numel()}, got {int(ptr[-1].item())}."
            )
        start_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].item())
            end = int(ptr[gid + 1].item())
            indices = raw[start:end]
            if indices.numel() == 0:
                start_lists.append([])
                continue
            ids = node_global_ids.index_select(0, indices.to(device=node_global_ids.device))
            start_lists.append([int(x) for x in ids.detach().cpu().tolist()])
        return start_lists

    @staticmethod
    def _build_undirected_rollout_edges(
        *,
        actions: torch.Tensor,
        directions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        node_global_ids: torch.Tensor,
        edge_start: int,
        edge_end: int,
        node_offset: int,
        start_locals: Sequence[int],
    ) -> tuple[list[int], list[int], list[Dict[str, Any]]]:
        active = {int(x) for x in start_locals}
        edge_ids: list[int] = []
        dir_ids: list[int] = []
        edges_meta: list[Dict[str, Any]] = []
        actions = actions.view(-1)
        directions = directions.view(-1)
        if actions.numel() != directions.numel():
            raise ValueError("actions/directions length mismatch in rollout edge metadata.")
        for step_idx, action in enumerate(actions.tolist()):
            if action < _ZERO:
                if action == STOP_RELATION and active:
                    local = min(active)
                    head_idx = node_offset + local
                    head_gid = int(node_global_ids[head_idx].item())
                    edges_meta.append(
                        {
                            "head_entity_id": head_gid,
                            "tail_entity_id": head_gid,
                            "relation_id": STOP_RELATION,
                            "src_entity_id": head_gid,
                            "dst_entity_id": head_gid,
                        }
                    )
                break
            edge_id = int(action)
            if edge_id < edge_start or edge_id >= edge_end:
                raise ValueError(
                    f"rollout edge id {edge_id} out of range [{edge_start},{edge_end})."
                )
            rel_id = int(edge_relations[edge_id].item())
            head_idx = int(edge_index[0, edge_id].item())
            tail_idx = int(edge_index[1, edge_id].item())
            head_local = head_idx - node_offset
            tail_local = tail_idx - node_offset
            dir_id = int(directions[step_idx].item())
            if dir_id not in (DIRECTION_FORWARD, DIRECTION_BACKWARD):
                raise ValueError(f"rollout directions_seq contains invalid values: {dir_id}.")
            edge_ids.append(edge_id - edge_start)
            dir_ids.append(dir_id)
            head_gid = int(node_global_ids[head_idx].item())
            tail_gid = int(node_global_ids[tail_idx].item())
            head_active = head_local in active
            tail_active = tail_local in active
            if head_active == tail_active:
                raise ValueError(
                    "Undirected rollout edge has ambiguous active endpoint; "
                    f"head_active={head_active}, tail_active={tail_active}."
                )
            if head_active:
                src_idx = head_idx
                dst_idx = tail_idx
                active = {tail_local}
            else:
                src_idx = tail_idx
                dst_idx = head_idx
                active = {head_local}
            edges_meta.append(
                {
                    "head_entity_id": head_gid,
                    "tail_entity_id": tail_gid,
                    "relation_id": rel_id,
                    "src_entity_id": int(node_global_ids[src_idx].item()),
                    "dst_entity_id": int(node_global_ids[dst_idx].item()),
                }
            )
        return edge_ids, dir_ids, edges_meta

    def _build_rollout_records(
        self,
        *,
        batch: Any,
        rollout_logs: list[Dict[str, torch.Tensor]],
        node_ptr: torch.Tensor,
        edge_ptr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        num_graphs: int,
    ) -> list[Dict[str, Any]]:
        if num_graphs <= 0:
            raise ValueError("node_ptr must encode at least one graph.")
        edge_ptr = edge_ptr.to(dtype=torch.long).view(-1)
        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError(f"edge_ptr length {edge_ptr.numel()} != num_graphs+1 ({num_graphs + 1}).")
        if edge_ptr.numel() == 0:
            raise ValueError("edge_ptr must be non-empty.")
        if int(edge_ptr[0].item()) != 0:
            raise ValueError(f"edge_ptr must start at 0, got {int(edge_ptr[0].item())}.")
        if bool((edge_ptr[1:] < edge_ptr[:-1]).any().item()):
            raise ValueError("edge_ptr must be non-decreasing.")
        total_edges = int(edge_ptr[-1].item())
        sample_ids, questions = self._extract_batch_meta(batch, num_graphs)
        answer_entity_ids = self._extract_answer_entity_ids(batch, num_graphs)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if node_global_ids is None:
            raise ValueError("Batch missing node_global_ids required for rollout artifacts.")
        if not torch.is_tensor(node_global_ids):
            node_global_ids = torch.as_tensor(node_global_ids, dtype=torch.long)
        node_global_ids = node_global_ids.view(-1).detach().cpu()
        start_entity_ids = self._extract_start_entity_ids(batch, num_graphs, node_global_ids)
        start_local_indices = self._extract_start_local_indices(batch, num_graphs)
        edge_index = edge_index.to(dtype=torch.long)
        edge_relations = edge_relations.to(dtype=torch.long)
        if not rollout_logs:
            raise ValueError("rollout_logs must be non-empty.")
        normalized_logs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ridx, log in enumerate(rollout_logs):
            actions_seq = log.get("actions_seq")
            directions_seq = log.get("directions_seq")
            log_pf = log.get("log_pf")
            reach_success = log.get("reach_success")
            stop_node_locals = log.get("stop_node_locals")
            if actions_seq is None or directions_seq is None or log_pf is None:
                raise ValueError(f"rollout_logs[{ridx}] missing required keys (actions_seq/directions_seq/log_pf).")
            if reach_success is None or stop_node_locals is None:
                raise ValueError(f"rollout_logs[{ridx}] missing reach_success/stop_node_locals.")
            if actions_seq.dim() != 2:
                raise ValueError(f"rollout_logs[{ridx}].actions_seq must be [B,T], got shape={tuple(actions_seq.shape)}.")
            if directions_seq.shape != actions_seq.shape:
                raise ValueError(
                    f"rollout_logs[{ridx}].directions_seq shape mismatch with actions_seq: "
                    f"{tuple(directions_seq.shape)} vs {tuple(actions_seq.shape)}."
                )
            if actions_seq.size(0) != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].actions_seq batch {actions_seq.size(0)} != num_graphs {num_graphs}.")
            if log_pf.dim() != 1 or log_pf.numel() != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].log_pf must be [B], got shape={tuple(log_pf.shape)}.")
            if reach_success.numel() != num_graphs:
                raise ValueError(
                    f"rollout_logs[{ridx}].reach_success must be [B], got shape={tuple(reach_success.shape)}."
                )
            if stop_node_locals.numel() != num_graphs:
                raise ValueError(
                    f"rollout_logs[{ridx}].stop_node_locals must be [B], got shape={tuple(stop_node_locals.shape)}."
                )
            normalized_logs.append(
                (
                    actions_seq.to(dtype=torch.long),
                    directions_seq.to(dtype=torch.long),
                    log_pf,
                    reach_success.to(dtype=torch.float32),
                    stop_node_locals.to(dtype=torch.long),
                )
            )
        records: list[Dict[str, Any]] = []
        for g in range(num_graphs):
            rollouts: list[Dict[str, Any]] = []
            edge_start = int(edge_ptr[g].item())
            edge_end = int(edge_ptr[g + 1].item())
            node_offset = int(node_ptr[g].item())
            start_locals = start_local_indices[g] if g < len(start_local_indices) else []
            for ridx, (actions_seq, directions_seq, log_pf, reach_success, stop_node_locals) in enumerate(
                normalized_logs
            ):
                actions = actions_seq[g].to(dtype=torch.long)
                directions = directions_seq[g].to(dtype=torch.long)
                if actions.numel() != directions.numel():
                    raise ValueError(
                        f"rollout_logs[{ridx}] actions/directions length mismatch for graph {g}: "
                        f"{actions.numel()} vs {directions.numel()}."
                    )
                if bool((actions < STOP_RELATION).any().item()):
                    bad = actions[actions < STOP_RELATION][:5].tolist()
                    raise ValueError(f"rollout_logs[{ridx}] actions_seq contains invalid negatives: {bad}.")
                edge_ids, dir_ids, edges_meta = self._build_undirected_rollout_edges(
                    actions=actions,
                    directions=directions,
                    edge_index=edge_index,
                    edge_relations=edge_relations,
                    node_global_ids=node_global_ids,
                    edge_start=edge_start,
                    edge_end=edge_end,
                    node_offset=node_offset,
                    start_locals=start_locals,
                )
                if edge_ids and total_edges <= 0:
                    raise ValueError("actions_seq selects edges but edge_ptr indicates zero total edges.")
                rollouts.append(
                    {
                        "rollout_index": ridx,
                        "log_pf": float(log_pf[g].item()),
                        "reach_success": bool(reach_success[g].item()),
                        "stop_node_local": int(stop_node_locals[g].item()),
                        "stop_node_entity_id": (
                            int(node_global_ids[int(node_ptr[g].item()) + int(stop_node_locals[g].item())].item())
                            if int(stop_node_locals[g].item()) >= _ZERO
                            else None
                        ),
                        "edge_ids": edge_ids,
                        "directions": dir_ids,
                        "edges": edges_meta,
                    }
                )
            records.append(
                {
                    "sample_id": sample_ids[g] if g < len(sample_ids) else str(g),
                    "question": questions[g] if g < len(questions) else "",
                    "answer_entity_ids": answer_entity_ids[g] if g < len(answer_entity_ids) else [],
                    "start_entity_ids": start_entity_ids[g] if g < len(start_entity_ids) else [],
                    "rollouts": rollouts,
                }
            )
        return records
