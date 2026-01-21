from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import inspect
import math
import functools
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Callable

import torch
from torch_scatter import scatter_max
from src.metrics import gflownet as gfn_metrics
from src.models.components import EntrySelector, GFlowNetActor, GraphEnv, RewardOutput
from src.models.components.gflownet_env import STOP_RELATION
from src.gfn.ops import (
    compute_policy_log_probs,
    EDGE_POLICY_MASK_KEY,
    GFlowNetBatchProcessor,
    GFlowNetInputValidator,
    RolloutInputs,
    STOP_NODE_MASK_KEY,
    gumbel_noise_like,
    neg_inf_value,
    scatter_logsumexp,
)
from src.gfn.trajectory_utils import TrajectoryStats, derive_trajectory, reflect_backward_to_forward, resolve_current_stop_locals
from src.utils.logging_utils import get_logger, log_event
_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_PI = math.pi
_NAN = float("nan")
_EDGE_INV_PREVIEW = 5
_DUAL_STREAM_CFG_KEY = "dual_stream"
_DUAL_STREAM_ENABLED_KEY = "enabled"
_DUAL_STREAM_WEIGHT_KEY = "stream_forward_weight"
_DUAL_STREAM_FORWARD_MAX_STEPS_KEY = "stream_forward_max_steps"
_DUAL_STREAM_SCHEDULE_KEY = "schedule"
_DUAL_STREAM_DISABLE_STOP_KEY = "disable_stop"
_DUAL_STREAM_WEIGHT_START_KEY = "stream_forward_weight_start"
_DUAL_STREAM_WEIGHT_SCHEDULE_KEY = "stream_forward_weight_schedule"
_DUAL_STREAM_WEIGHT_ANNEAL_EPOCHS_KEY = "stream_forward_weight_anneal_epochs"
_DUAL_STREAM_WEIGHT_SCHEDULE_NONE = "none"
_DUAL_STREAM_WEIGHT_SCHEDULE_LINEAR = "linear"
_DUAL_STREAM_WEIGHT_SCHEDULE_COSINE = "cosine"
_DUAL_STREAM_WEIGHT_SCHEDULES = {
    _DUAL_STREAM_WEIGHT_SCHEDULE_NONE,
    _DUAL_STREAM_WEIGHT_SCHEDULE_LINEAR,
    _DUAL_STREAM_WEIGHT_SCHEDULE_COSINE,
}
_DUAL_STREAM_SCHEDULE_LINEAR = "linear"
_DUAL_STREAM_SCHEDULES = {_DUAL_STREAM_SCHEDULE_LINEAR}
_DEFAULT_DUAL_STREAM_WEIGHT = 1.0
_DEFAULT_DUAL_STREAM_WEIGHT_START = 0.0
_DEFAULT_DUAL_STREAM_WEIGHT_SCHEDULE = _DUAL_STREAM_WEIGHT_SCHEDULE_NONE
_DEFAULT_DUAL_STREAM_WEIGHT_ANNEAL_EPOCHS = None
_SUBTB_CFG_KEY = "subtb"
_SUBTB_ENABLED_KEY = "enabled"
_SUBTB_NUM_KEY = "num_subtrajectories"
_DEFAULT_SUBTB_ENABLED = False
logger = get_logger(__name__)
_DEFAULT_SUBTB_NUM = 1
_Z_ALIGN_ENABLED_KEY = "enabled"
_Z_ALIGN_WEIGHT_KEY = "weight"
_DEFAULT_Z_ALIGN_ENABLED = False
_DEFAULT_Z_ALIGN_WEIGHT = 1.0
_H_GUIDANCE_CFG_KEY = "h_guidance"
_H_GUIDANCE_ENABLED_KEY = "enabled"
_H_GUIDANCE_BETA_START_KEY = "beta_start"
_H_GUIDANCE_BETA_END_KEY = "beta_end"
_H_GUIDANCE_WARMUP_KEY = "warmup_progress"
_H_GUIDANCE_APPLY_EVAL_KEY = "apply_eval"
_H_GUIDANCE_STOP_GRAD_KEY = "stop_gradient"
_H_GUIDANCE_SCALE_KEY = "scale"
_DEFAULT_H_GUIDANCE_ENABLED = False
_DEFAULT_H_GUIDANCE_BETA_START = 0.0
_DEFAULT_H_GUIDANCE_BETA_END = 1.0
_DEFAULT_H_GUIDANCE_WARMUP = 0.1
_DEFAULT_H_GUIDANCE_APPLY_EVAL = False
_DEFAULT_H_GUIDANCE_STOP_GRAD = True
_DEFAULT_H_GUIDANCE_SCALE = 1.0
_IMITATION_CFG_KEY = "imitation"
_IMITATION_ENABLED_KEY = "enabled"
_IMITATION_WEIGHT_KEY = "weight"
_DEFAULT_IMITATION_WEIGHT = 1.0
_INVERSE_EDGE_IDS_KEY = "inverse_edge_ids"
_TARGET_SAMPLING_CFG_KEY = "target_sampling"
_TARGET_SAMPLING_MODE_KEY = "mode"
_TARGET_SAMPLING_MODE_ALL = "all"
_TARGET_SAMPLING_MODE_RANDOM_ONE = "random_one"
_TARGET_SAMPLING_MODES = {_TARGET_SAMPLING_MODE_ALL, _TARGET_SAMPLING_MODE_RANDOM_ONE}
_DEFAULT_TARGET_SAMPLING_MODE = _TARGET_SAMPLING_MODE_ALL
_ENTRY_SELECTOR_CFG_KEY = "selector_cfg"
_ENTRY_SELECTOR_ENABLED_KEY = "enabled"
_ENTRY_SELECTOR_EPSILON_KEY = "epsilon"
_DEFAULT_ENTRY_SELECTOR_ENABLED = False
_DEFAULT_ENTRY_SELECTOR_EPSILON = 0.1
_ENTRY_LOG_PROB_MISS_DEFAULT = 0.0
_ENTRY_SELECTOR_STATS_EVENT = "gfn_entry_selector_stats"
_FLOW_STATS_DIM = 2
_BACKWARD_LOG_PB_MAX_REPEATED_EDGES = 500_000
_NONFINITE_ROLLOUT_EVENT = "gfn_nonfinite_rollout"
_NONFINITE_ROLLOUT_PREVIEW = 5
_ANCHOR_OOB_EVENT = "gfn_anchor_indices_oob"
_ANCHOR_OOB_SAMPLE_PREVIEW = 5
_ANCHOR_OOB_VALUE_PREVIEW = 8
_START_NODE_NONE = -1
_STOP_NODE_NONE = -1


@dataclass(frozen=True)
class FlowFeatureSpec:
    graph_stats_log1p: bool
    stats_dim: int


@dataclass(frozen=True)
class TargetSamplingSpec:
    mode: str


@dataclass(frozen=True)
class EntrySelectorSpec:
    epsilon: float


def resolve_flow_spec(flow_cfg: Mapping[str, object] | None) -> FlowFeatureSpec:
    if flow_cfg is None:
        raise ValueError("flow_cfg must be provided for Flow features.")
    graph_stats_log1p = bool(flow_cfg.get("graph_stats_log1p", True))
    return FlowFeatureSpec(
        graph_stats_log1p=graph_stats_log1p,
        stats_dim=_FLOW_STATS_DIM,
    )


def _compute_flow_graph_stats(
    *,
    node_ptr: torch.Tensor,
    edge_ptr: torch.Tensor,
    log1p: bool,
) -> torch.Tensor:
    node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).to(dtype=torch.float32)
    edge_counts = (edge_ptr[_ONE:] - edge_ptr[:-_ONE]).to(dtype=torch.float32)
    stats = torch.stack((node_counts, edge_counts), dim=1)
    if log1p:
        stats = torch.log1p(stats)
    return stats


def build_flow_features(
    *,
    node_ptr: torch.Tensor,
    edge_ptr: torch.Tensor,
    spec: FlowFeatureSpec,
) -> torch.Tensor:
    num_graphs = int(node_ptr.numel() - _ONE)
    if num_graphs <= _ZERO:
        return torch.zeros((num_graphs, spec.stats_dim), device=node_ptr.device, dtype=torch.float32)
    return _compute_flow_graph_stats(
        node_ptr=node_ptr,
        edge_ptr=edge_ptr,
        log1p=spec.graph_stats_log1p,
    )


@dataclass(frozen=True)
class GFlowNetRolloutConfig:
    num_rollouts: int
    eval_rollout_prefixes: Sequence[int]
    eval_rollout_temperature: float
    rollout_chunk_size: int
    is_training: bool

    def __post_init__(self) -> None:
        if self.num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be > 0, got {self.num_rollouts}.")
        if self.eval_rollout_temperature < 0.0:
            raise ValueError(f"eval_rollout_temperature must be >= 0, got {self.eval_rollout_temperature}.")
        if self.rollout_chunk_size <= 0:
            raise ValueError(f"rollout_chunk_size must be > 0, got {self.rollout_chunk_size}.")
        if self.rollout_chunk_size > self.num_rollouts:
            raise ValueError(
                "rollout_chunk_size must be <= num_rollouts " f"({self.num_rollouts}), got {self.rollout_chunk_size}."
            )


@dataclass(frozen=True)
class RolloutChunkInputs:
    inputs: RolloutInputs
    graph_cache: Dict[str, torch.Tensor]
    flow_features: torch.Tensor
    log_f_start: torch.Tensor
    log_pf_start: Optional[torch.Tensor]
    graph_mask: torch.Tensor
    num_graphs: int
    node_ptr: torch.Tensor
    node_is_target: torch.Tensor


RolloutHook = Callable[
    [
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RolloutInputs,
        Dict[str, torch.Tensor],
    ],
    None,
]


@dataclass
class RolloutChunkState:
    loss_list: list[torch.Tensor]
    metrics_list: list[Dict[str, torch.Tensor]]
    rollout_stop_nodes: list[torch.Tensor]
    rollout_actions: list[torch.Tensor]


@dataclass
class StreamingChunkContext:
    node_ptr: Optional[torch.Tensor]
    node_is_target: Optional[torch.Tensor]
    inputs: Optional[RolloutInputs]
    graph_cache: Optional[Dict[str, torch.Tensor]]
    flow_features: Optional[torch.Tensor]
    log_f_start: Optional[torch.Tensor]
    log_pf_start: Optional[torch.Tensor]
    graph_mask: Optional[torch.Tensor]
    num_graphs: int
    cache_inputs: bool


@dataclass(frozen=True)
class EntrySelection:
    selected_locals: torch.Tensor
    selected_ptr: torch.Tensor
    log_prob: torch.Tensor


@dataclass(frozen=True)
class RolloutLossRecord:
    reward_out: RewardOutput
    log_f_start: torch.Tensor
    log_reward: torch.Tensor
    log_target: torch.Tensor
    sum_log_pf: torch.Tensor
    sum_log_pb: torch.Tensor
    residual: torch.Tensor
    reach_success: torch.Tensor
    num_moves: torch.Tensor
    imitation_active: torch.Tensor
    imitation_loss: torch.Tensor


@dataclass(frozen=True)
class FlowModules:
    log_f: torch.nn.Module


@dataclass(frozen=True)
class SubTrajectorySpec:
    num_subtrajectories: int


@dataclass(frozen=True)
class ImitationSpec:
    weight: float


@dataclass(frozen=True)
class DualStreamSpec:
    stream_forward_weight: float
    stream_forward_max_steps: int
    schedule: str
    stream_forward_weight_start: float
    stream_forward_weight_schedule: str
    stream_forward_weight_anneal_epochs: Optional[int]


@dataclass(frozen=True)
class HGuidanceSpec:
    beta_start: float
    beta_end: float
    warmup_progress: float
    apply_eval: bool
    stop_gradient: bool
    scale: float


@dataclass(frozen=True)
class ZAlignSpec:
    weight: float


class GFlowNetEngine:
    def __init__(
        self,
        *,
        actor: GFlowNetActor,
        actor_backward: Optional[GFlowNetActor] = None,
        reward_fn: torch.nn.Module,
        env: GraphEnv,
        log_f: torch.nn.Module,
        log_f_backward: Optional[torch.nn.Module] = None,
        flow_spec: FlowFeatureSpec,
        relation_inverse_map: Optional[torch.Tensor] = None,
        relation_is_inverse: Optional[torch.Tensor] = None,
        batch_processor: GFlowNetBatchProcessor,
        input_validator: GFlowNetInputValidator,
        composite_score_cfg: Optional[Any] = None,
        dual_stream_cfg: Optional[Mapping[str, Any]] = None,
        subtb_cfg: Optional[Mapping[str, Any]] = None,
        z_align_cfg: Optional[Mapping[str, Any]] = None,
        h_guidance_cfg: Optional[Mapping[str, Any]] = None,
        imitation_cfg: Optional[Mapping[str, Any]] = None,
        target_sampling_cfg: Optional[Mapping[str, Any]] = None,
        entry_selector_cfg: Optional[Mapping[str, Any]] = None,
        source_selector: Optional[EntrySelector] = None,
        sink_selector: Optional[EntrySelector] = None,
        cache_action_keys: bool = False,
        require_precomputed_edge_batch_train: bool = True,
        require_precomputed_edge_batch_eval: bool = True,
        debug_rollout_steps: bool = False,
    ) -> None:
        self.actor = actor
        self.actor_backward = actor if actor_backward is None else actor_backward
        self.reward_fn = reward_fn
        self.env = env
        self.log_f = log_f
        self.log_f_backward = log_f_backward if log_f_backward is not None else log_f
        self.flow_spec = flow_spec
        self._relation_inverse_map = relation_inverse_map
        self._relation_is_inverse = relation_is_inverse
        self.batch_processor = batch_processor
        self.input_validator = input_validator
        self._composite_score_cfg = gfn_metrics.resolve_composite_score_cfg(composite_score_cfg)
        self._dual_stream_cfg = dual_stream_cfg
        self._subtb_cfg = subtb_cfg
        self._z_align_cfg = z_align_cfg
        self._z_align_spec: Optional[ZAlignSpec] = self._resolve_z_align_spec(z_align_cfg)
        self._h_guidance_cfg = h_guidance_cfg
        self._h_guidance_spec: Optional[HGuidanceSpec] = self._resolve_h_guidance_spec(h_guidance_cfg)
        self._imitation_cfg = imitation_cfg
        self._imitation_spec: Optional[ImitationSpec] = self._resolve_imitation_spec(imitation_cfg)
        self._replay_spec = None
        self._replay_buffer = None
        self._target_sampling_cfg = target_sampling_cfg
        self._target_sampling_spec: TargetSamplingSpec = self._resolve_target_sampling_spec(target_sampling_cfg)
        self._entry_selector_cfg = entry_selector_cfg
        self._entry_selector_spec: Optional[EntrySelectorSpec] = self._resolve_entry_selector_spec(entry_selector_cfg)
        self._source_selector = source_selector
        self._sink_selector = sink_selector
        self._validate_entry_selector_components()
        self._log_pb_max_repeated_edges = int(_BACKWARD_LOG_PB_MAX_REPEATED_EDGES)
        self._cache_action_keys = bool(cache_action_keys)
        self._require_precomputed_edge_batch_train = bool(require_precomputed_edge_batch_train)
        self._require_precomputed_edge_batch_eval = bool(require_precomputed_edge_batch_eval)
        self._debug_rollout_steps = bool(debug_rollout_steps)
        self._debug_rollout_logged = False
        self._debug_rollout_epoch: Optional[int] = None

    def set_composite_score_cfg(self, composite_score_cfg: Optional[Any]) -> None:
        self._composite_score_cfg = gfn_metrics.resolve_composite_score_cfg(composite_score_cfg)

    def set_debug_rollout_steps(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._debug_rollout_steps:
            self._debug_rollout_logged = False
        self._debug_rollout_steps = enabled

    def set_debug_rollout_epoch(self, epoch: Optional[int]) -> None:
        if epoch is None:
            self._debug_rollout_epoch = None
            return
        self._debug_rollout_epoch = int(epoch)

    def _set_edge_batch_requirement(self, *, is_training: bool) -> None:
        required = self._require_precomputed_edge_batch_train if is_training else self._require_precomputed_edge_batch_eval
        self.batch_processor.require_precomputed_edge_batch = required

    def _action_keys_cache_key(self, actor: GFlowNetActor) -> str:
        if self.actor_backward is self.actor:
            return "action_keys_shared"
        if actor is self.actor_backward:
            return "action_keys_backward"
        if actor is self.actor:
            return "action_keys_forward"
        return f"action_keys_{id(actor)}"

    def _get_action_keys(
        self,
        *,
        graph_cache: Dict[str, torch.Tensor],
        actor: GFlowNetActor,
    ) -> torch.Tensor:
        cache_key = self._action_keys_cache_key(actor)
        cached = graph_cache.get(cache_key)
        if cached is not None:
            return cached
        action_keys = actor.agent.precompute_action_keys(
            relation_tokens=graph_cache["relation_tokens"],
            node_tokens=graph_cache["node_tokens"],
            edge_index=graph_cache["edge_index"],
        )
        if self._cache_action_keys:
            graph_cache[cache_key] = action_keys
        return action_keys

    def _resolve_dual_stream_spec(
        self,
        *,
        is_training: bool,
    ) -> Optional[DualStreamSpec]:
        if not is_training:
            return None
        cfg = self._dual_stream_cfg
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            cfg = {} if cfg else None
        if cfg is None:
            return None
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.dual_stream must be a mapping or bool.")
        enabled = cfg.get(_DUAL_STREAM_ENABLED_KEY, True)
        if not bool(enabled):
            return None
        stream_forward_weight = self._require_non_negative_float(cfg.get(_DUAL_STREAM_WEIGHT_KEY, _DEFAULT_DUAL_STREAM_WEIGHT))
        max_steps_raw = cfg.get(_DUAL_STREAM_FORWARD_MAX_STEPS_KEY)
        stream_forward_max_steps = int(self.env.max_steps) if max_steps_raw is None else self._require_positive_int(max_steps_raw)
        schedule_raw = cfg.get(_DUAL_STREAM_SCHEDULE_KEY, _DUAL_STREAM_SCHEDULE_LINEAR)
        schedule = str(schedule_raw or _DUAL_STREAM_SCHEDULE_LINEAR).strip().lower()
        if schedule not in _DUAL_STREAM_SCHEDULES:
            raise ValueError(
                "training_cfg.dual_stream.schedule must be one of " f"{sorted(_DUAL_STREAM_SCHEDULES)}, got {schedule!r}."
            )
        if _DUAL_STREAM_DISABLE_STOP_KEY in cfg:
            raise ValueError("training_cfg.dual_stream.disable_stop has been removed; stop is always allowed.")
        weight_start = self._require_non_negative_float(
            cfg.get(_DUAL_STREAM_WEIGHT_START_KEY, _DEFAULT_DUAL_STREAM_WEIGHT_START)
        )
        weight_schedule_raw = cfg.get(_DUAL_STREAM_WEIGHT_SCHEDULE_KEY, _DEFAULT_DUAL_STREAM_WEIGHT_SCHEDULE)
        weight_schedule = str(weight_schedule_raw or _DEFAULT_DUAL_STREAM_WEIGHT_SCHEDULE).strip().lower()
        if weight_schedule not in _DUAL_STREAM_WEIGHT_SCHEDULES:
            raise ValueError(
                "training_cfg.dual_stream.stream_forward_weight_schedule must be one of "
                f"{sorted(_DUAL_STREAM_WEIGHT_SCHEDULES)}, got {weight_schedule!r}."
            )
        weight_anneal_raw = cfg.get(_DUAL_STREAM_WEIGHT_ANNEAL_EPOCHS_KEY, _DEFAULT_DUAL_STREAM_WEIGHT_ANNEAL_EPOCHS)
        weight_anneal_epochs = None if weight_anneal_raw is None else self._require_positive_int(weight_anneal_raw)
        return DualStreamSpec(
            stream_forward_weight=float(stream_forward_weight),
            stream_forward_max_steps=stream_forward_max_steps,
            schedule=schedule,
            stream_forward_weight_start=float(weight_start),
            stream_forward_weight_schedule=weight_schedule,
            stream_forward_weight_anneal_epochs=weight_anneal_epochs,
        )

    def _resolve_subtb_spec(self, *, is_training: bool) -> SubTrajectorySpec:
        _ = is_training
        cfg = self._subtb_cfg
        if cfg is None:
            raise ValueError("training_cfg.subtb must be set; only SubTB is supported.")
        if isinstance(cfg, bool):
            cfg = {} if cfg else None
        if cfg is None:
            raise ValueError("training_cfg.subtb must enable SubTB; got disabled config.")
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.subtb must be a mapping or bool.")
        enabled = cfg.get(_SUBTB_ENABLED_KEY, _DEFAULT_SUBTB_ENABLED)
        if not bool(enabled):
            raise ValueError("training_cfg.subtb.enabled must be true; only SubTB is supported.")
        num_sub = self._require_positive_int(cfg.get(_SUBTB_NUM_KEY, _DEFAULT_SUBTB_NUM))
        return SubTrajectorySpec(num_subtrajectories=num_sub)

    def _resolve_z_align_spec(self, cfg: Optional[Mapping[str, Any]]) -> Optional[ZAlignSpec]:
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            cfg = {} if cfg else None
        if cfg is None:
            return None
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.z_align must be a mapping or bool.")
        enabled = cfg.get(_Z_ALIGN_ENABLED_KEY, _DEFAULT_Z_ALIGN_ENABLED)
        if not bool(enabled):
            return None
        weight = self._require_non_negative_float(cfg.get(_Z_ALIGN_WEIGHT_KEY, _DEFAULT_Z_ALIGN_WEIGHT))
        return ZAlignSpec(weight=float(weight))


    def _resolve_h_guidance_spec(self, cfg: Optional[Mapping[str, Any]]) -> Optional[HGuidanceSpec]:
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            cfg = {} if cfg else None
        if cfg is None:
            return None
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.h_guidance must be a mapping or bool.")
        enabled = cfg.get(_H_GUIDANCE_ENABLED_KEY, _DEFAULT_H_GUIDANCE_ENABLED)
        if not bool(enabled):
            return None
        beta_start = self._require_non_negative_float(cfg.get(_H_GUIDANCE_BETA_START_KEY, _DEFAULT_H_GUIDANCE_BETA_START))
        beta_end = self._require_non_negative_float(cfg.get(_H_GUIDANCE_BETA_END_KEY, _DEFAULT_H_GUIDANCE_BETA_END))
        warmup = self._require_probability_closed(cfg.get(_H_GUIDANCE_WARMUP_KEY, _DEFAULT_H_GUIDANCE_WARMUP))
        apply_eval = bool(cfg.get(_H_GUIDANCE_APPLY_EVAL_KEY, _DEFAULT_H_GUIDANCE_APPLY_EVAL))
        stop_gradient = bool(cfg.get(_H_GUIDANCE_STOP_GRAD_KEY, _DEFAULT_H_GUIDANCE_STOP_GRAD))
        scale = self._require_non_negative_float(cfg.get(_H_GUIDANCE_SCALE_KEY, _DEFAULT_H_GUIDANCE_SCALE))
        return HGuidanceSpec(
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            warmup_progress=float(warmup),
            apply_eval=apply_eval,
            stop_gradient=stop_gradient,
            scale=float(scale),
        )

    def _resolve_imitation_spec(self, cfg: Optional[Mapping[str, Any]]) -> Optional[ImitationSpec]:
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            cfg = {} if cfg else None
        if cfg is None:
            return None
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.imitation must be a mapping or bool.")
        enabled = cfg.get(_IMITATION_ENABLED_KEY, True)
        if not bool(enabled):
            return None
        weight = self._require_non_negative_float(cfg.get(_IMITATION_WEIGHT_KEY, _DEFAULT_IMITATION_WEIGHT))
        return ImitationSpec(weight=float(weight))

    def _resolve_target_sampling_spec(self, cfg: Optional[Mapping[str, Any]]) -> TargetSamplingSpec:
        if cfg is None:
            return TargetSamplingSpec(mode=_DEFAULT_TARGET_SAMPLING_MODE)
        if isinstance(cfg, bool):
            mode = _TARGET_SAMPLING_MODE_RANDOM_ONE if cfg else _DEFAULT_TARGET_SAMPLING_MODE
            return TargetSamplingSpec(mode=mode)
        if not isinstance(cfg, Mapping):
            raise TypeError("training_cfg.target_sampling must be a mapping or bool.")
        mode_raw = cfg.get(_TARGET_SAMPLING_MODE_KEY, _DEFAULT_TARGET_SAMPLING_MODE)
        mode = str(mode_raw or _DEFAULT_TARGET_SAMPLING_MODE).strip().lower()
        if mode not in _TARGET_SAMPLING_MODES:
            raise ValueError(
                "training_cfg.target_sampling.mode must be one of "
                f"{sorted(_TARGET_SAMPLING_MODES)}, got {mode!r}."
            )
        return TargetSamplingSpec(mode=mode)

    def _resolve_entry_selector_spec(self, cfg: Optional[Mapping[str, Any]]) -> Optional[EntrySelectorSpec]:
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            cfg = { _ENTRY_SELECTOR_ENABLED_KEY: cfg }
        if not isinstance(cfg, Mapping):
            raise TypeError("model.selector_cfg must be a mapping or bool.")
        enabled = bool(cfg.get(_ENTRY_SELECTOR_ENABLED_KEY, _DEFAULT_ENTRY_SELECTOR_ENABLED))
        if not enabled:
            return None
        epsilon = self._require_probability_closed(cfg.get(_ENTRY_SELECTOR_EPSILON_KEY, _DEFAULT_ENTRY_SELECTOR_EPSILON))
        return EntrySelectorSpec(epsilon=float(epsilon))

    def _validate_entry_selector_components(self) -> None:
        if self._entry_selector_spec is None:
            return
        if self._source_selector is None or self._sink_selector is None:
            raise ValueError("selector_cfg enabled but source/sink selectors are missing.")
        if not isinstance(self._source_selector, EntrySelector) or not isinstance(self._sink_selector, EntrySelector):
            raise TypeError("source/sink selectors must be EntrySelector instances.")

    def _entry_selectors_enabled(self) -> bool:
        return self._entry_selector_spec is not None

    def _resolve_entry_selectors_for_flow(
        self,
        *,
        log_f_module: torch.nn.Module,
    ) -> tuple[Optional[EntrySelector], Optional[EntrySelector]]:
        if not self._entry_selectors_enabled():
            return None, None
        if log_f_module is self.log_f_backward:
            return self._sink_selector, self._source_selector
        return self._source_selector, self._sink_selector

    def _select_start_nodes_for_flow(
        self,
        *,
        log_f_module: torch.nn.Module,
        query_tokens: torch.Tensor,
        candidate_locals: torch.Tensor,
        candidate_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
        stream: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_selector, _ = self._resolve_entry_selectors_for_flow(log_f_module=log_f_module)
        if start_selector is None:
            return self._sample_one_per_graph(candidate_locals=candidate_locals, candidate_ptr=candidate_ptr)
        selection = self._select_entry_nodes(
            selector=start_selector,
            query_tokens=query_tokens,
            candidate_locals=candidate_locals,
            candidate_ptr=candidate_ptr,
            node_ptr=node_ptr,
            node_tokens=node_tokens,
            stream=stream,
            role="start",
        )
        return selection.selected_locals, selection.selected_ptr, selection.log_prob

    def _maybe_attach_entry_end_cache(
        self,
        *,
        log_f_module: torch.nn.Module,
        query_tokens: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        stream: str,
    ) -> None:
        _, end_selector = self._resolve_entry_selectors_for_flow(log_f_module=log_f_module)
        if end_selector is None:
            return
        self._attach_entry_log_prob_cache(
            graph_cache=graph_cache,
            selector=end_selector,
            inputs=inputs,
            query_tokens=query_tokens,
            candidate_locals=inputs.target_node_locals,
            candidate_ptr=inputs.target_ptr,
            cache_prefix="entry_end",
            stream=stream,
            role="end",
        )

    def _override_start_nodes(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        start_nodes: torch.Tensor,
        start_ptr: torch.Tensor,
    ) -> tuple[RolloutInputs, Dict[str, torch.Tensor]]:
        device = inputs.node_ptr.device
        start_nodes = start_nodes.to(device=device, dtype=torch.long).view(-1)
        start_ptr = start_ptr.to(device=device, dtype=torch.long).view(-1)
        updated_inputs = replace(inputs, start_node_locals=start_nodes, start_ptr=start_ptr)
        updated_cache = dict(graph_cache)
        updated_cache["start_node_locals"] = start_nodes
        updated_cache["start_ptr"] = start_ptr
        num_nodes_total = int(inputs.node_ptr[-1].detach().tolist()) if inputs.node_ptr.numel() > 0 else _ZERO
        node_is_start, _ = self.batch_processor.compute_node_flags(
            num_nodes_total,
            start_nodes,
            inputs.target_node_locals,
            device,
        )
        updated_cache["node_is_start"] = node_is_start
        return updated_inputs, updated_cache

    @staticmethod
    def _build_candidate_view(
        *,
        candidate_locals: torch.Tensor,
        candidate_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = node_ptr.device
        candidate_locals = candidate_locals.to(device=device, dtype=torch.long).view(-1)
        candidate_ptr = candidate_ptr.to(device=device, dtype=torch.long).view(-1)
        num_graphs = int(node_ptr.numel() - 1)
        if candidate_ptr.numel() != num_graphs + _ONE:
            raise ValueError("candidate_ptr length mismatch with num_graphs.")
        counts = (candidate_ptr[_ONE:] - candidate_ptr[:-_ONE]).clamp(min=_ZERO)
        if not bool((counts > _ZERO).all().detach().tolist()):
            raise ValueError("Entry selector requires non-empty candidate sets for all graphs.")
        total = int(counts.sum().detach().tolist())
        if total != int(candidate_locals.numel()):
            raise ValueError("candidate_locals length mismatch with candidate_ptr counts.")
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)
        pairs = torch.stack((graph_ids, candidate_locals), dim=_ONE)
        unique_pairs = torch.unique(pairs, dim=_ZERO)
        graph_ids = unique_pairs[:, _ZERO]
        candidate_locals = unique_pairs[:, _ONE]
        unique_counts = torch.bincount(graph_ids, minlength=num_graphs)
        if not bool((unique_counts > _ZERO).all().detach().tolist()):
            raise ValueError("Entry selector requires non-empty candidate sets after dedup.")
        node_start = node_ptr.index_select(0, graph_ids)
        node_end = node_ptr.index_select(0, graph_ids + _ONE)
        in_range = (candidate_locals >= node_start) & (candidate_locals < node_end)
        if bool((~in_range).any().detach().tolist()):
            raise ValueError("Entry selector candidate indices fall outside per-graph node ranges.")
        candidate_globals = candidate_locals
        candidate_tokens = node_tokens.index_select(0, candidate_globals)
        return unique_counts, graph_ids, candidate_globals, candidate_tokens, candidate_locals

    def _compute_entry_log_probs(
        self,
        *,
        selector: EntrySelector,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_graph_ids: torch.Tensor,
        candidate_counts: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        logits = selector.score(
            query_tokens=query_tokens,
            candidate_tokens=candidate_tokens,
            candidate_batch=candidate_graph_ids,
        )
        log_denom = scatter_logsumexp(logits, candidate_graph_ids, num_graphs)
        log_soft = logits - log_denom.index_select(0, candidate_graph_ids)
        counts = candidate_counts.index_select(0, candidate_graph_ids).to(dtype=log_soft.dtype)
        log_uniform = -torch.log(counts)
        epsilon = float(self._entry_selector_spec.epsilon)
        eps_tensor = torch.tensor(epsilon, device=log_soft.device, dtype=log_soft.dtype)
        log_eps = torch.log(eps_tensor)
        log_one_minus = torch.log1p(-eps_tensor)
        return torch.logaddexp(log_soft + log_one_minus, log_uniform + log_eps)

    @staticmethod
    def _sample_entry_indices(
        *,
        log_probs: torch.Tensor,
        candidate_graph_ids: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = scatter_max(scores, candidate_graph_ids, dim=0, dim_size=num_graphs)
        if argmax.numel() != num_graphs:
            raise ValueError("Entry selector argmax length mismatch with num_graphs.")
        return argmax

    def _maybe_log_entry_selector_stats(
        self,
        *,
        log_probs: torch.Tensor,
        selected_log_prob: Optional[torch.Tensor],
        stream: str,
        role: str,
    ) -> None:
        if not self._debug_rollout_steps:
            return
        if log_probs.numel() == _ZERO:
            return
        log_probs = log_probs.detach()
        fields = {
            "stream": str(stream),
            "role": str(role),
            "log_prob_mean": float(log_probs.mean().item()),
            "log_prob_min": float(log_probs.min().item()),
            "log_prob_max": float(log_probs.max().item()),
        }
        if selected_log_prob is not None and selected_log_prob.numel() > _ZERO:
            selected_log_prob = selected_log_prob.detach()
            fields["selected_log_prob_mean"] = float(selected_log_prob.mean().item())
        log_event(
            logger,
            _ENTRY_SELECTOR_STATS_EVENT,
            **fields,
        )

    def _select_entry_nodes(
        self,
        *,
        selector: EntrySelector,
        query_tokens: torch.Tensor,
        candidate_locals: torch.Tensor,
        candidate_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
        stream: str,
        role: str,
    ) -> EntrySelection:
        counts, graph_ids, candidate_globals, candidate_tokens, candidate_locals = self._build_candidate_view(
            candidate_locals=candidate_locals,
            candidate_ptr=candidate_ptr,
            node_ptr=node_ptr,
            node_tokens=node_tokens,
        )
        num_graphs = int(node_ptr.numel() - 1)
        log_probs = self._compute_entry_log_probs(
            selector=selector,
            query_tokens=query_tokens,
            candidate_tokens=candidate_tokens,
            candidate_graph_ids=graph_ids,
            candidate_counts=counts,
            num_graphs=num_graphs,
        )
        selected_idx = self._sample_entry_indices(
            log_probs=log_probs,
            candidate_graph_ids=graph_ids,
            num_graphs=num_graphs,
        )
        selected_locals = candidate_locals.index_select(0, selected_idx)
        selected_ptr = torch.arange(num_graphs + _ONE, device=node_ptr.device, dtype=torch.long)
        log_prob = log_probs.index_select(0, selected_idx)
        self._maybe_log_entry_selector_stats(
            log_probs=log_probs,
            selected_log_prob=log_prob,
            stream=stream,
            role=role,
        )
        _ = candidate_globals
        return EntrySelection(
            selected_locals=selected_locals,
            selected_ptr=selected_ptr,
            log_prob=log_prob,
        )

    def _attach_entry_log_prob_cache(
        self,
        *,
        graph_cache: Dict[str, torch.Tensor],
        selector: EntrySelector,
        inputs: RolloutInputs,
        query_tokens: torch.Tensor,
        candidate_locals: torch.Tensor,
        candidate_ptr: torch.Tensor,
        cache_prefix: str,
        stream: str,
        role: str,
    ) -> None:
        counts, graph_ids, candidate_globals, candidate_tokens, _ = self._build_candidate_view(
            candidate_locals=candidate_locals,
            candidate_ptr=candidate_ptr,
            node_ptr=inputs.node_ptr,
            node_tokens=inputs.node_tokens,
        )
        num_graphs = int(inputs.node_ptr.numel() - 1)
        log_probs = self._compute_entry_log_probs(
            selector=selector,
            query_tokens=query_tokens,
            candidate_tokens=candidate_tokens,
            candidate_graph_ids=graph_ids,
            candidate_counts=counts,
            num_graphs=num_graphs,
        )
        self._maybe_log_entry_selector_stats(
            log_probs=log_probs,
            selected_log_prob=None,
            stream=stream,
            role=role,
        )
        graph_cache[f"{cache_prefix}_globals"] = candidate_globals
        graph_cache[f"{cache_prefix}_log_probs"] = log_probs

    @staticmethod
    def _maybe_attach_sample_ids(
        graph_cache: Dict[str, torch.Tensor],
        *,
        batch: Any,
        num_graphs: int,
    ) -> None:
        raw_ids = getattr(batch, "sample_id", None)
        if raw_ids is None:
            return
        if isinstance(raw_ids, (list, tuple)):
            sample_ids = [str(s) for s in raw_ids]
        elif torch.is_tensor(raw_ids):
            sample_ids = [str(s.detach().tolist()) for s in raw_ids.view(-1)]
        else:
            return
        if len(sample_ids) != num_graphs:
            return
        graph_cache["sample_ids"] = sample_ids

    def _build_graph_cache_with_masks(
        self,
        *,
        inputs: RolloutInputs,
        device: torch.device,
        flow_direction: str,
        batch: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        graph_cache = self.batch_processor.build_graph_cache(inputs, device=device)
        if batch is not None:
            num_graphs = int(inputs.node_ptr.numel() - 1)
            self._maybe_attach_sample_ids(graph_cache, batch=batch, num_graphs=num_graphs)
        self._attach_inverse_edge_mask(graph_cache)
        self._attach_stream_policy_masks(graph_cache, flow_direction=flow_direction)
        return graph_cache

    @staticmethod
    def _validate_packed_node_locals(
        *,
        node_locals: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        name: str,
        flow_direction: str,
        sample_ids: Optional[Sequence[str]],
    ) -> None:
        device = node_ptr.device
        node_ptr = node_ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        node_locals = node_locals.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        ptr = ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        num_graphs = int(node_ptr.numel() - _ONE)
        if ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"{name}_ptr length {ptr.numel()} != num_graphs+1 ({num_graphs + _ONE}).")
        if int(ptr[_ZERO].detach().tolist()) != _ZERO:
            raise ValueError(f"{name}_ptr must start at 0.")
        if int(ptr[-_ONE].detach().tolist()) != int(node_locals.numel()):
            raise ValueError(
                f"{name}_ptr must end at {int(node_locals.numel())}, got {int(ptr[-_ONE].detach().tolist())}."
            )
        if bool((ptr[_ONE:] < ptr[:-_ONE]).any().detach().tolist()):
            raise ValueError(f"{name}_ptr must be non-decreasing.")
        if node_locals.numel() == _ZERO:
            return
        positions = torch.arange(node_locals.numel(), device=device, dtype=ptr.dtype)
        graph_ids = torch.bucketize(positions, ptr[_ONE:], right=True)
        node_start = node_ptr.index_select(0, graph_ids)
        node_end = node_ptr.index_select(0, graph_ids + _ONE)
        invalid = (node_locals < node_start) | (node_locals >= node_end)
        if not bool(invalid.any().detach().tolist()):
            return
        bad_pos = invalid.nonzero(as_tuple=False).view(-1)
        bad_graph_ids = torch.unique(graph_ids.index_select(0, bad_pos))
        preview_graph_ids = bad_graph_ids[:_ANCHOR_OOB_SAMPLE_PREVIEW].to(device="cpu").tolist()
        preview_sample_ids = None
        if sample_ids is not None:
            preview_sample_ids = [str(sample_ids[i]) for i in preview_graph_ids]
        preview_vals = node_locals.index_select(0, bad_pos[:_ANCHOR_OOB_VALUE_PREVIEW]).to(device="cpu").tolist()
        log_event(
            logger,
            _ANCHOR_OOB_EVENT,
            name=name,
            flow_direction=flow_direction,
            bad_graph_count=int(bad_graph_ids.numel()),
            graph_ids=preview_graph_ids,
            sample_ids=preview_sample_ids,
            values=preview_vals,
        )
        raise ValueError(f"{name} indices fall outside per-graph node ranges; see {_ANCHOR_OOB_EVENT} logs for details.")

    @staticmethod
    def _resolve_stop_globals(
        *,
        stop_node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stop_locals = stop_node_locals.to(dtype=torch.long)
        valid = stop_locals >= _ZERO
        node_counts = node_ptr[_ONE:] - node_ptr[:-_ONE]
        if bool((stop_locals >= node_counts).any().detach().tolist()):
            raise ValueError("stop_node_locals out of range for entry selector lookup.")
        stop_globals = node_ptr[:-_ONE] + stop_locals.clamp(min=_ZERO)
        return stop_globals, valid

    def _lookup_entry_log_prob_for_stop_nodes(
        self,
        *,
        candidate_globals: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        stop_node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
        default_log_prob: Optional[float] = None,
    ) -> torch.Tensor:
        stop_globals, valid_stop = self._resolve_stop_globals(stop_node_locals=stop_node_locals, node_ptr=node_ptr)
        if candidate_globals.numel() == _ZERO:
            raise ValueError("entry selector candidate set is empty.")
        sorted_globals, sort_idx = torch.sort(candidate_globals)
        if sorted_globals.numel() > _ONE:
            dup = sorted_globals[_ONE:] == sorted_globals[:-_ONE]
            if bool(dup.any().detach().tolist()):
                preview = sorted_globals[_ONE:][dup][:_EDGE_INV_PREVIEW].tolist()
                raise ValueError(f"Duplicate entry candidates detected (preview: {preview}).")
        sorted_log_probs = candidate_log_probs.index_select(0, sort_idx)
        pos = torch.searchsorted(sorted_globals, stop_globals)
        num_keys = int(sorted_globals.numel())
        max_pos = max(num_keys - _ONE, _ZERO)
        pos_safe = pos.clamp(min=_ZERO, max=max_pos)
        matched = sorted_globals.index_select(0, pos_safe) == stop_globals
        within = pos < num_keys
        valid = valid_stop & within & matched
        if default_log_prob is None:
            default_log_prob = neg_inf_value(candidate_log_probs)
        log_prob = sorted_log_probs.index_select(0, pos_safe)
        return torch.where(valid, log_prob, torch.full_like(log_prob, float(default_log_prob)))

    @staticmethod
    def _append_rollout_outputs(
        actions_seq: Optional[torch.Tensor],
        stop_node_locals: torch.Tensor,
        tb_loss: torch.Tensor,
        record: RolloutLossRecord,
        *,
        collect_terminal_hits: bool,
        collect_eval: bool,
        rollout_stop_nodes: list[torch.Tensor],
        rollout_actions: list[torch.Tensor],
        loss_list: list[torch.Tensor],
        metric_records: list[RolloutLossRecord],
    ) -> None:
        if collect_terminal_hits:
            rollout_stop_nodes.append(stop_node_locals.detach())
        if collect_eval:
            if actions_seq is None:
                raise RuntimeError("rollout missing actions; record_actions=True required.")
            rollout_actions.append(actions_seq.detach())
        loss_list.append(tb_loss)
        metric_records.append(record)

    @staticmethod
    def _require_float(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("Expected float, got bool.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return float(text)
            except ValueError as exc:
                raise TypeError(f"Expected float, got {value!r}.") from exc
        raise TypeError(f"Expected float, got {type(value).__name__}.")

    @classmethod
    def _require_non_negative_float(cls, value: Any) -> float:
        parsed = cls._require_float(value)
        if parsed < float(_ZERO):
            raise ValueError(f"Value must be >= 0, got {parsed}.")
        return parsed

    @classmethod
    def _require_positive_float(cls, value: Any) -> float:
        parsed = cls._require_float(value)
        if parsed <= float(_ZERO):
            raise ValueError(f"Value must be > 0, got {parsed}.")
        return parsed

    @classmethod
    def _require_probability_closed(cls, value: Any) -> float:
        parsed = cls._require_non_negative_float(value)
        if parsed > float(_ONE):
            raise ValueError(f"Value must be <= 1, got {parsed}.")
        return parsed

    @classmethod
    def _require_positive_int(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("Expected positive int, got bool.")
        if isinstance(value, int):
            parsed = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"Expected positive int, got {value}.")
            parsed = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise TypeError("Expected positive int, got empty string.")
            if any(ch in text for ch in ".eE"):
                parsed_float = cls._require_float(text)
                if not parsed_float.is_integer():
                    raise TypeError(f"Expected positive int, got {value!r}.")
                parsed = int(parsed_float)
            else:
                parsed = int(text)
        else:
            raise TypeError(f"Expected positive int, got {type(value).__name__}.")
        if parsed <= _ZERO:
            raise ValueError(f"Value must be > 0, got {parsed}.")
        return parsed

    @classmethod
    def _require_non_negative_int(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("Expected non-negative int, got bool.")
        if isinstance(value, int):
            parsed = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"Expected non-negative int, got {value}.")
            parsed = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise TypeError("Expected non-negative int, got empty string.")
            if any(ch in text for ch in ".eE"):
                parsed_float = cls._require_float(text)
                if not parsed_float.is_integer():
                    raise TypeError(f"Expected non-negative int, got {value!r}.")
                parsed = int(parsed_float)
            else:
                parsed = int(text)
        else:
            raise TypeError(f"Expected non-negative int, got {type(value).__name__}.")
        if parsed < _ZERO:
            raise ValueError(f"Value must be >= 0, got {parsed}.")
        return parsed

    def _compute_single_stream_loss(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        guidance_beta: float,
        graph_mask: torch.Tensor,
        rollout_cfg: GFlowNetRolloutConfig,
        temperature: Optional[float],
        subtb_spec: Optional[SubTrajectorySpec],
        max_steps_override: Optional[int],
        sample_ids: Optional[Sequence[str]] = None,
        rollout_hook: Optional[RolloutHook] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        loss, metrics = self._compute_loop_rollout_loss(
            inputs=inputs,
            num_rollouts=rollout_cfg.num_rollouts,
            num_graphs=num_graphs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            flow_modules=flow_modules,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            guidance_beta=guidance_beta,
            graph_mask=graph_mask,
            temperature=temperature,
            rollout_cfg=rollout_cfg,
            rollout_chunk_size=rollout_cfg.rollout_chunk_size,
            subtb_spec=subtb_spec,
            max_steps_override=max_steps_override,
            rollout_hook=rollout_hook,
        )
        if (
            self._replay_spec is not None
            and self._replay_buffer is not None
            and flow_modules.log_f is self.log_f
            and sample_ids is not None
        ):
            replay_inputs, replay_cache, replay_features, _ = self._prepare_replay_inputs(
                batch=None,
                device=inputs.node_ptr.device,
                is_training=rollout_cfg.is_training,
                base_inputs=inputs,
                base_graph_cache=graph_cache,
                base_flow_features=flow_features,
            )
            replay_loss, replay_metrics = self._compute_replay_loss(
                inputs=replay_inputs,
                graph_cache=replay_cache,
                flow_features=replay_features,
                flow_modules=flow_modules,
                temperature=temperature,
                subtb_spec=subtb_spec,
                sample_ids=sample_ids,
            )
            if replay_loss is not None:
                mix_ratio = float(self._replay_spec.mix_ratio)
                loss = (float(_ONE) - mix_ratio) * loss + mix_ratio * replay_loss
                replay_metrics["replay/mix_ratio"] = torch.as_tensor(
                    mix_ratio,
                    device=loss.device,
                    dtype=loss.dtype,
                )
                metrics.update(replay_metrics)
        return loss, metrics

    def _compute_dual_stream_loss(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        log_f_start_forward: torch.Tensor,
        log_pf_start_forward: Optional[torch.Tensor],
        graph_mask: torch.Tensor,
        rollout_cfg: GFlowNetRolloutConfig,
        temperature: Optional[float],
        spec: DualStreamSpec,
        subtb_spec: Optional[SubTrajectorySpec],
        progress: float,
        stream_forward_progress: Optional[float],
        guidance_beta: float,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flow_modules_backward = self._resolve_flow_modules(direction="backward")
        flow_modules_forward = self._resolve_flow_modules(direction="forward")
        inputs_backward, graph_cache_backward, log_pf_start_backward = self._prepare_backward_inputs(
            inputs=inputs,
            graph_cache=graph_cache,
            temperature=temperature,
            is_training=True,
        )
        flow_features_backward = self._compute_flow_features(inputs_backward)
        log_f_start_backward = self._compute_log_f_start(
            inputs=inputs_backward,
            graph_cache=graph_cache_backward,
            flow_features=flow_features_backward,
            log_f_module=flow_modules_backward.log_f,
        )
        graph_mask_backward = ~inputs_backward.dummy_mask
        replay_hook = None
        if self._replay_buffer is not None and sample_ids is not None:
            def _hook(
                actions_seq: Optional[torch.Tensor],
                _log_pf: torch.Tensor,
                _log_pf_steps: torch.Tensor,
                _log_pb_steps: torch.Tensor,
                reach_success: torch.Tensor,
                num_moves: torch.Tensor,
                stop_node_locals: torch.Tensor,
                rollout_inputs: RolloutInputs,
                rollout_cache: Dict[str, torch.Tensor],
            ) -> None:
                if actions_seq is None:
                    raise RuntimeError("Replay hook requires actions_seq; record_actions=True required.")
                self._store_replay_from_backward_rollout(
                    actions_seq=actions_seq,
                    reach_success=reach_success,
                    stop_node_locals=stop_node_locals,
                    inputs=rollout_inputs,
                    graph_cache=rollout_cache,
                    sample_ids=sample_ids,
                )
            replay_hook = _hook
        loss_backward, metrics_backward = self._compute_single_stream_loss(
            inputs=inputs_backward,
            graph_cache=graph_cache_backward,
            flow_features=flow_features_backward,
            flow_modules=flow_modules_backward,
            log_f_start=log_f_start_backward,
            log_pf_start=log_pf_start_backward,
            guidance_beta=guidance_beta,
            graph_mask=graph_mask_backward,
            rollout_cfg=rollout_cfg,
            temperature=temperature,
            subtb_spec=subtb_spec,
            max_steps_override=None,
            rollout_hook=replay_hook,
        )
        loss_forward, metrics_forward = self._compute_single_stream_loss(
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            flow_modules=flow_modules_forward,
            log_f_start=log_f_start_forward,
            log_pf_start=log_pf_start_forward,
            guidance_beta=guidance_beta,
            graph_mask=graph_mask,
            rollout_cfg=rollout_cfg,
            temperature=temperature,
            subtb_spec=subtb_spec,
            max_steps_override=spec.stream_forward_max_steps,
            sample_ids=sample_ids,
        )
        weight_progress = stream_forward_progress if stream_forward_progress is not None else progress
        stream_forward_weight = self._compute_stream_forward_weight(spec=spec, progress=weight_progress)
        loss = loss_backward + (float(stream_forward_weight) * loss_forward)
        metrics_backward = self._filter_stream_backward_metrics(metrics_backward)
        metrics = {}
        metrics.update(self._suffix_metrics(metrics_backward, suffix="stream_backward"))
        metrics.update(self._suffix_metrics(metrics_forward, suffix="stream_forward"))
        metrics["stream_backward_loss"] = loss_backward.detach()
        metrics["stream_forward_loss"] = loss_forward.detach()
        metrics["stream_forward_weight"] = torch.as_tensor(
            float(stream_forward_weight),
            device=loss_backward.device,
            dtype=loss_backward.dtype,
        )
        return loss, metrics

    @staticmethod
    def _slice_ptr(ptr: torch.Tensor, *, start_graph: int, end_graph: int) -> tuple[torch.Tensor, int]:
        start = int(ptr[start_graph].detach().tolist())
        end = int(ptr[end_graph].detach().tolist())
        sliced = ptr[start_graph : end_graph + _ONE] - start
        return sliced, start

    @staticmethod
    def _slice_indices(values: torch.Tensor, *, ptr: torch.Tensor, start_graph: int, end_graph: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(ptr[start_graph].detach().tolist())
        end = int(ptr[end_graph].detach().tolist())
        sliced = values[start:end]
        new_ptr = ptr[start_graph : end_graph + _ONE] - start
        return sliced, new_ptr

    def _build_batch_slice(
        self,
        batch: Any,
        *,
        start_graph: int,
        end_graph: int,
    ) -> Any:
        node_ptr = batch.ptr
        edge_ptr = getattr(batch, "edge_ptr", None)
        edge_batch = getattr(batch, "edge_batch", None)
        if not torch.is_tensor(edge_ptr) or not torch.is_tensor(edge_batch):
            raise ValueError("Dual stream forwardatch splitting requires precomputed edge_ptr and edge_batch.")
        node_ptr_slice, node_offset = self._slice_ptr(node_ptr, start_graph=start_graph, end_graph=end_graph)
        edge_ptr_slice, edge_offset = self._slice_ptr(edge_ptr, start_graph=start_graph, end_graph=end_graph)
        node_slice = slice(node_offset, node_offset + int(node_ptr_slice[-1].detach().tolist()))
        edge_slice = slice(edge_offset, edge_offset + int(edge_ptr_slice[-1].detach().tolist()))

        edge_index = batch.edge_index[:, edge_slice] - node_offset
        edge_attr = batch.edge_attr[edge_slice]
        edge_batch_slice = edge_batch[edge_slice] - int(start_graph)

        question_emb = batch.question_emb[start_graph:end_graph]
        node_embeddings = batch.node_embeddings[node_slice]
        node_embedding_ids = batch.node_embedding_ids[node_slice]
        edge_embeddings = batch.edge_embeddings[edge_slice]
        node_global_ids = getattr(batch, "node_global_ids", None)
        if torch.is_tensor(node_global_ids):
            node_global_ids = node_global_ids[node_slice]
        node_is_cvt = getattr(batch, "node_is_cvt", None)
        if torch.is_tensor(node_is_cvt):
            node_is_cvt = node_is_cvt[node_slice]
        node_type_counts_all = getattr(batch, "node_type_counts", None)
        node_type_counts = None
        node_type_ids = None
        if torch.is_tensor(node_type_counts_all):
            node_type_counts = node_type_counts_all[node_slice]
            node_type_ids_all = getattr(batch, "node_type_ids", None)
            if node_type_ids_all is None:
                raise ValueError("Batch missing node_type_ids for node_type_counts slicing.")
            if not torch.is_tensor(node_type_ids_all):
                node_type_ids_all = torch.as_tensor(node_type_ids_all, dtype=torch.long)
            if node_offset <= _ZERO:
                type_start = _ZERO
            else:
                type_start = int(node_type_counts_all[:node_offset].sum().detach().tolist())
            type_count = int(node_type_counts.sum().detach().tolist())
            node_type_ids = node_type_ids_all[type_start : type_start + type_count]

        slice_dict = getattr(batch, "_slice_dict", None)
        if not isinstance(slice_dict, dict):
            raise ValueError("Batch missing _slice_dict required for dual stream splitting.")
        q_ptr_raw = torch.as_tensor(slice_dict.get("q_local_indices"), dtype=torch.long)
        a_ptr_raw = torch.as_tensor(slice_dict.get("a_local_indices"), dtype=torch.long)
        q_local_indices, q_ptr = self._slice_indices(
            batch.q_local_indices,
            ptr=q_ptr_raw,
            start_graph=start_graph,
            end_graph=end_graph,
        )
        a_local_indices, a_ptr = self._slice_indices(
            batch.a_local_indices,
            ptr=a_ptr_raw,
            start_graph=start_graph,
            end_graph=end_graph,
        )
        if q_local_indices.numel() > 0:
            q_local_indices = q_local_indices - node_offset
        if a_local_indices.numel() > 0:
            a_local_indices = a_local_indices - node_offset
        answer_entity_ids = getattr(batch, "answer_entity_ids", None)
        answer_entity_ids_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        answer_entity_ids_slice = None
        answer_entity_ids_ptr_slice = None
        if torch.is_tensor(answer_entity_ids) and answer_entity_ids_ptr is not None:
            answer_ptr_raw = torch.as_tensor(answer_entity_ids_ptr, dtype=torch.long)
            answer_entity_ids_slice, answer_entity_ids_ptr_slice = self._slice_indices(
                answer_entity_ids.view(-1),
                ptr=answer_ptr_raw,
                start_graph=start_graph,
                end_graph=end_graph,
            )

        view = SimpleNamespace(
            ptr=node_ptr_slice,
            edge_index=edge_index,
            edge_attr=edge_attr,
            question_emb=question_emb,
            node_embeddings=node_embeddings,
            node_global_ids=node_global_ids,
            node_embedding_ids=node_embedding_ids,
            edge_embeddings=edge_embeddings,
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            edge_batch=edge_batch_slice,
            edge_ptr=edge_ptr_slice,
        )
        if answer_entity_ids_slice is not None and answer_entity_ids_ptr_slice is not None:
            view.answer_entity_ids = answer_entity_ids_slice
            view.answer_entity_ids_ptr = answer_entity_ids_ptr_slice
        retrieval_failure = getattr(batch, "retrieval_failure", None)
        if isinstance(retrieval_failure, (list, tuple)):
            view.retrieval_failure = list(retrieval_failure[start_graph:end_graph])
        elif torch.is_tensor(retrieval_failure):
            view.retrieval_failure = retrieval_failure[start_graph:end_graph]
        sample_ids = getattr(batch, "sample_id", None)
        if isinstance(sample_ids, (list, tuple)):
            view.sample_id = list(sample_ids[start_graph:end_graph])
        elif torch.is_tensor(sample_ids):
            view.sample_id = [str(s.detach().tolist()) for s in sample_ids[start_graph:end_graph]]
        if node_is_cvt is not None:
            view.node_is_cvt = node_is_cvt
        if node_type_counts is not None:
            view.node_type_counts = node_type_counts
        if node_type_ids is not None:
            view.node_type_ids = node_type_ids
        view._slice_dict = {
            "q_local_indices": q_ptr,
            "a_local_indices": a_ptr,
        }
        return view

    def _split_batch_for_dual_stream(self, batch: Any) -> tuple[Any, Any]:
        node_ptr = batch.ptr
        if not torch.is_tensor(node_ptr):
            raise ValueError("Batch missing ptr required for dual stream splitting.")
        num_graphs = int(node_ptr.numel() - _ONE)
        split_graphs = num_graphs // _TWO
        if split_graphs <= _ZERO or split_graphs >= num_graphs:
            raise ValueError("Dual stream split requires at least two graphs.")
        batch_backward = self._build_batch_slice(batch, start_graph=_ZERO, end_graph=split_graphs)
        batch_forward = self._build_batch_slice(batch, start_graph=split_graphs, end_graph=num_graphs)
        return batch_backward, batch_forward

    def _compute_dual_stream_loss_streaming(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
        backward_fn: Callable[[torch.Tensor], None],
        spec: DualStreamSpec,
        subtb_spec: Optional[SubTrajectorySpec],
        progress: float,
        stream_forward_progress: Optional[float],
        guidance_beta: float,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._z_align_spec is None:
            try:
                batch_backward, batch_forward = self._split_batch_for_dual_stream(batch)
            except ValueError:
                batch_backward = batch
                batch_forward = batch
        else:
            batch_backward = batch
            batch_forward = batch
        sample_ids_backward = None
        sample_ids_forward = None
        if self._replay_spec is not None and self._replay_buffer is not None:
            def _num_graphs(batch_obj: Any) -> int:
                ptr = getattr(batch_obj, "ptr", None)
                if torch.is_tensor(ptr):
                    return int(ptr.numel() - _ONE)
                num_graphs = getattr(batch_obj, "num_graphs", None)
                if num_graphs is None:
                    raise ValueError("Batch missing ptr/num_graphs required for replay sample_ids.")
                return int(num_graphs)

            if sample_ids is None:
                sample_ids_backward = self._extract_sample_ids_for_replay(batch_backward, _num_graphs(batch_backward))
                sample_ids_forward = self._extract_sample_ids_for_replay(batch_forward, _num_graphs(batch_forward))
            else:
                sample_ids_list = list(sample_ids)
                if batch_backward is batch_forward:
                    sample_ids_backward = sample_ids_list
                    sample_ids_forward = sample_ids_list
                else:
                    num_graphs_backward = _num_graphs(batch_backward)
                    num_graphs_forward = _num_graphs(batch_forward)
                    total_graphs = num_graphs_backward + num_graphs_forward
                    if total_graphs != len(sample_ids_list):
                        raise ValueError(
                            f"sample_ids length {len(sample_ids_list)} != split graphs {total_graphs}."
                        )
                    sample_ids_backward = sample_ids_list[:num_graphs_backward]
                    sample_ids_forward = sample_ids_list[num_graphs_backward:total_graphs]
        loss_backward, metrics_backward = self._run_stream_backward_streaming(
            batch=batch_backward,
            device=device,
            rollout_cfg=rollout_cfg,
            backward_fn=backward_fn,
            subtb_spec=subtb_spec,
            guidance_beta=guidance_beta,
            sample_ids=sample_ids_backward,
        )
        forward_loss_scale = float(_ONE)
        if self._replay_spec is not None and sample_ids_forward is not None:
            forward_loss_scale = float(_ONE) - float(self._replay_spec.mix_ratio)
        loss_forward, metrics_forward = self._run_stream_forward_streaming(
            batch=batch_forward,
            device=device,
            rollout_cfg=rollout_cfg,
            backward_fn=backward_fn,
            spec=spec,
            subtb_spec=subtb_spec,
            guidance_beta=guidance_beta,
            sample_ids=sample_ids_forward,
            loss_scale=forward_loss_scale,
        )
        weight_progress = stream_forward_progress if stream_forward_progress is not None else progress
        stream_forward_weight = self._compute_stream_forward_weight(spec=spec, progress=weight_progress)
        loss = loss_backward + (float(stream_forward_weight) * loss_forward)
        metrics_backward = self._filter_stream_backward_metrics(metrics_backward)
        metrics = {}
        metrics.update(self._suffix_metrics(metrics_backward, suffix="stream_backward"))
        metrics.update(self._suffix_metrics(metrics_forward, suffix="stream_forward"))
        metrics["stream_backward_loss"] = loss_backward.detach()
        metrics["stream_forward_loss"] = loss_forward.detach()
        metrics["stream_forward_weight"] = torch.as_tensor(
            float(stream_forward_weight),
            device=loss_backward.device,
            dtype=loss_backward.dtype,
        )
        return loss, metrics

    def _run_stream_backward_streaming(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
        backward_fn: Callable[[torch.Tensor], None],
        subtb_spec: Optional[SubTrajectorySpec],
        guidance_beta: float,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flow_modules = self._resolve_flow_modules(direction="backward")
        replay_hook = None
        if self._replay_buffer is not None and sample_ids is not None:
            def _hook(
                actions_seq: Optional[torch.Tensor],
                _log_pf: torch.Tensor,
                _log_pf_steps: torch.Tensor,
                _log_pb_steps: torch.Tensor,
                reach_success: torch.Tensor,
                num_moves: torch.Tensor,
                stop_node_locals: torch.Tensor,
                rollout_inputs: RolloutInputs,
                rollout_cache: Dict[str, torch.Tensor],
            ) -> None:
                if actions_seq is None:
                    raise RuntimeError("Replay hook requires actions_seq; record_actions=True required.")
                self._store_replay_from_backward_rollout(
                    actions_seq=actions_seq,
                    reach_success=reach_success,
                    stop_node_locals=stop_node_locals,
                    inputs=rollout_inputs,
                    graph_cache=rollout_cache,
                    sample_ids=sample_ids,
                )
            replay_hook = _hook
        loss_backward, metrics_backward = self._compute_loop_rollout_loss_streaming(
            batch=batch,
            device=device,
            num_rollouts=rollout_cfg.num_rollouts,
            rollout_cfg=rollout_cfg,
            rollout_chunk_size=rollout_cfg.rollout_chunk_size,
            temperature=None,
            backward_fn=backward_fn,
            subtb_spec=subtb_spec,
            flow_modules=flow_modules,
            guidance_beta=guidance_beta,
            max_steps_override=None,
            log_f_start_node_locals=None,
            log_f_start_ptr=None,
            rollout_hook=replay_hook,
        )
        return loss_backward, metrics_backward

    def _run_stream_forward_streaming(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
        backward_fn: Callable[[torch.Tensor], None],
        spec: DualStreamSpec,
        subtb_spec: Optional[SubTrajectorySpec],
        guidance_beta: float,
        sample_ids: Optional[Sequence[str]] = None,
        loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flow_modules = self._resolve_flow_modules(direction="forward")
        loss_forward, metrics_forward = self._compute_loop_rollout_loss_streaming(
            batch=batch,
            device=device,
            num_rollouts=rollout_cfg.num_rollouts,
            rollout_cfg=rollout_cfg,
            rollout_chunk_size=rollout_cfg.rollout_chunk_size,
            temperature=None,
            backward_fn=backward_fn,
            subtb_spec=subtb_spec,
            flow_modules=flow_modules,
            guidance_beta=guidance_beta,
            max_steps_override=spec.stream_forward_max_steps,
            loss_scale=loss_scale,
            sample_ids=sample_ids,
        )
        return loss_forward, metrics_forward

    def compute_batch_loss(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
        progress: Optional[float] = None,
        stream_forward_progress: Optional[float] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._set_edge_batch_requirement(is_training=rollout_cfg.is_training)
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=rollout_cfg.is_training,
        )
        flow_modules_forward = self._resolve_flow_modules(direction="forward")
        (
            inputs,
            graph_cache,
            flow_features,
            log_f_start_forward,
            graph_mask,
            num_graphs,
            _,
            _,
            log_pf_start_forward,
        ) = self._prepare_streaming_inputs(
            batch=batch,
            device=device,
            node_ptr=None,
            node_is_target=None,
            log_f_module=flow_modules_forward.log_f,
            log_f_start_node_locals=None,
            log_f_start_ptr=None,
            is_training=rollout_cfg.is_training,
        )
        graph_cache["flow_features"] = flow_features
        temperature = None if rollout_cfg.is_training else rollout_cfg.eval_rollout_temperature
        guidance_beta = self._resolve_h_guidance_beta(progress=progress, is_training=rollout_cfg.is_training)
        dual_spec = self._resolve_dual_stream_spec(is_training=rollout_cfg.is_training)
        subtb_spec = self._resolve_subtb_spec(is_training=rollout_cfg.is_training)
        sample_ids = None
        if self._replay_spec is not None:
            sample_ids = self._extract_sample_ids_for_replay(batch, num_graphs)
        if dual_spec is None:
            loss, metrics = self._compute_single_stream_loss(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                flow_modules=flow_modules_forward,
                log_f_start=log_f_start_forward,
                log_pf_start=log_pf_start_forward,
                guidance_beta=guidance_beta,
                graph_mask=graph_mask,
                rollout_cfg=rollout_cfg,
            temperature=temperature,
            subtb_spec=subtb_spec,
            max_steps_override=None,
            sample_ids=sample_ids,
        )
        else:
            progress_val = float(_ZERO) if progress is None else float(progress)
            loss, metrics = self._compute_dual_stream_loss(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                log_f_start_forward=log_f_start_forward,
                log_pf_start_forward=log_pf_start_forward,
                graph_mask=graph_mask,
                rollout_cfg=rollout_cfg,
                temperature=temperature,
                spec=dual_spec,
                subtb_spec=subtb_spec,
                progress=progress_val,
                stream_forward_progress=stream_forward_progress,
                guidance_beta=guidance_beta,
                sample_ids=sample_ids,
            )
        return loss, metrics

    def compute_batch_loss_streaming(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
        backward_fn: Callable[[torch.Tensor], None],
        progress: Optional[float] = None,
        stream_forward_progress: Optional[float] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._set_edge_batch_requirement(is_training=rollout_cfg.is_training)
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=rollout_cfg.is_training,
        )
        if not rollout_cfg.is_training:
            raise ValueError("compute_batch_loss_streaming requires training rollout config.")
        dual_spec = self._resolve_dual_stream_spec(is_training=True)
        subtb_spec = self._resolve_subtb_spec(is_training=True)
        guidance_beta = self._resolve_h_guidance_beta(progress=progress, is_training=True)
        if dual_spec is None:
            sample_ids = None
            if self._replay_spec is not None:
                num_graphs = int(batch.ptr.numel() - 1) if torch.is_tensor(batch.ptr) else int(getattr(batch, "num_graphs", 0))
                sample_ids = self._extract_sample_ids_for_replay(batch, num_graphs)
            flow_modules = self._resolve_flow_modules(direction="forward")
            loss, metrics = self._compute_loop_rollout_loss_streaming(
                batch=batch,
                device=device,
                num_rollouts=rollout_cfg.num_rollouts,
                rollout_cfg=rollout_cfg,
                rollout_chunk_size=rollout_cfg.rollout_chunk_size,
                temperature=None,
                backward_fn=backward_fn,
                subtb_spec=subtb_spec,
                flow_modules=flow_modules,
                guidance_beta=guidance_beta,
                max_steps_override=None,
                loss_scale=float(_ONE) - float(self._replay_spec.mix_ratio) if self._replay_spec is not None else float(_ONE),
                sample_ids=sample_ids,
            )
        else:
            progress_val = float(_ZERO) if progress is None else float(progress)
            sample_ids = None
            if self._replay_spec is not None:
                if torch.is_tensor(batch.ptr):
                    num_graphs = int(batch.ptr.numel() - _ONE)
                else:
                    num_graphs = int(getattr(batch, "num_graphs", _ZERO))
                sample_ids = self._extract_sample_ids_for_replay(batch, num_graphs)
            loss, metrics = self._compute_dual_stream_loss_streaming(
                batch=batch,
                device=device,
                rollout_cfg=rollout_cfg,
                backward_fn=backward_fn,
                spec=dual_spec,
                subtb_spec=subtb_spec,
                progress=progress_val,
                stream_forward_progress=stream_forward_progress,
                guidance_beta=guidance_beta,
                sample_ids=sample_ids,
            )
        return loss, metrics

    def compute_rollout_records(
        self,
        *,
        batch: Any,
        device: torch.device,
        rollout_cfg: GFlowNetRolloutConfig,
    ) -> list[Dict[str, Any]]:
        self._set_edge_batch_requirement(is_training=False)
        self._validate_batch_inputs(
            batch,
            device=device,
            require_rollout=True,
            is_training=False,
        )
        inputs = self.batch_processor.prepare_rollout_inputs(batch, device)
        num_graphs = int(inputs.node_ptr.numel() - 1)
        graph_cache = self._build_graph_cache_with_masks(
            inputs=inputs,
            device=device,
            flow_direction="forward",
        )
        start_nodes, start_ptr, log_pf_start = self._select_start_nodes_for_flow(
            log_f_module=self.log_f,
            query_tokens=inputs.question_tokens,
            candidate_locals=inputs.start_node_locals,
            candidate_ptr=inputs.start_ptr,
            node_ptr=inputs.node_ptr,
            node_tokens=inputs.node_tokens,
            stream="forward",
        )
        inputs, graph_cache = self._override_start_nodes(
            inputs=inputs,
            graph_cache=graph_cache,
            start_nodes=start_nodes,
            start_ptr=start_ptr,
        )
        flow_features = self._compute_flow_features(inputs)
        rollout_logs: list[Dict[str, torch.Tensor]] = []
        guidance_beta = self._resolve_h_guidance_beta(progress=None, is_training=False)
        guidance_fn = None
        if self._should_apply_h_guidance(flow_modules=self._resolve_flow_modules(direction="forward")):
            guidance_fn = self._build_h_guidance_fn(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                beta=guidance_beta,
            )
        if self._cache_action_keys:
            _ = self._get_action_keys(graph_cache=graph_cache, actor=self.actor)
        for _ in range(rollout_cfg.num_rollouts):
            actions_seq, log_pf, _, _ = self.actor.rollout(
                graph=graph_cache,
                temperature=rollout_cfg.eval_rollout_temperature,
                record_actions=True,
                guidance_fn=guidance_fn,
                mode="forward",
                init_node_locals=graph_cache["start_node_locals"],
                init_ptr=graph_cache["start_ptr"],
            )
            if log_pf_start is not None:
                log_pf = log_pf + log_pf_start.to(device=log_pf.device, dtype=log_pf.dtype).view(-1)
            if actions_seq is None:
                raise RuntimeError("rollout missing actions; record_actions=True required.")
            stop_node_locals, reach_success = self._finalize_rollout_stop(
                actions_seq=actions_seq,
                inputs=inputs,
                graph_cache=graph_cache,
                target_key="node_is_target",
            )
            rollout_logs.append(
                {
                    "actions_seq": actions_seq.detach().to(device="cpu"),
                    "log_pf": log_pf.detach().to(device="cpu"),
                    "reach_success": reach_success.detach().to(device="cpu"),
                    "stop_node_locals": stop_node_locals.detach().to(device="cpu"),
                }
            )
        return self._build_rollout_records(
            batch=batch,
            rollout_logs=rollout_logs,
            node_ptr=inputs.node_ptr.detach().to(device="cpu"),
            edge_ptr=inputs.edge_ptr.detach().to(device="cpu"),
            edge_index=inputs.edge_index.detach().to(device="cpu"),
            edge_relations=inputs.edge_relations.detach().to(device="cpu"),
            num_graphs=num_graphs,
        )

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
        *,
        stop_node_locals: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if graph_cache is None:
            raise ValueError("graph_cache required for reward computation.")
        if "node_is_target" not in graph_cache:
            raise ValueError("node_is_target missing from graph_cache; reward_fn requires explicit flags.")
        node_is_target = graph_cache["node_is_target"]
        return {
            "dummy_mask": inputs.dummy_mask,
            "node_ptr": inputs.node_ptr,
            "stop_node_locals": stop_node_locals,
            "node_is_target": node_is_target,
        }

    def build_reward_kwargs(
        self,
        *,
        stop_node_locals: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        return self._build_reward_kwargs(stop_node_locals=stop_node_locals, inputs=inputs, graph_cache=graph_cache)

    @staticmethod
    def _compute_stream_forward_weight(*, spec: DualStreamSpec, progress: Optional[float]) -> float:
        weight_end = float(spec.stream_forward_weight)
        schedule = spec.stream_forward_weight_schedule
        if schedule == _DUAL_STREAM_WEIGHT_SCHEDULE_NONE:
            return weight_end
        if progress is None:
            return weight_end
        progress_val = float(max(min(progress, float(_ONE)), float(_ZERO)))
        start = float(spec.stream_forward_weight_start)
        if schedule == _DUAL_STREAM_WEIGHT_SCHEDULE_LINEAR:
            scale = progress_val
        elif schedule == _DUAL_STREAM_WEIGHT_SCHEDULE_COSINE:
            half = float(_ONE) / float(_TWO)
            scale = half * (float(_ONE) - math.cos(float(_PI) * progress_val))
        else:
            raise ValueError(f"Unsupported stream_forward_weight_schedule: {schedule!r}.")
        return start + (weight_end - start) * scale

    def resolve_stream_forward_weight_progress(self, *, current_epoch: int) -> Optional[float]:
        spec = self._resolve_dual_stream_spec(is_training=True)
        if spec is None:
            return None
        anneal_epochs = spec.stream_forward_weight_anneal_epochs
        if anneal_epochs is None:
            return None
        epoch = max(int(current_epoch) + _ONE, _ZERO)
        return min(max(epoch / float(anneal_epochs), float(_ZERO)), float(_ONE))

    def _compute_flow_features(self, inputs: RolloutInputs) -> torch.Tensor:
        return build_flow_features(
            node_ptr=inputs.node_ptr,
            edge_ptr=inputs.edge_ptr,
            spec=self.flow_spec,
        )

    def _resolve_flow_features(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cached = graph_cache.get("flow_features")
        if cached is not None:
            return cached
        flow_features = self._compute_flow_features(inputs)
        graph_cache["flow_features"] = flow_features
        return flow_features

    def _resolve_actor_for_log_f(self, log_f_module: torch.nn.Module) -> GFlowNetActor:
        if log_f_module is self.log_f_backward:
            return self.actor_backward
        return self.actor

    def _resolve_actor_for_flow(self, flow_modules: FlowModules) -> GFlowNetActor:
        if flow_modules.log_f is self.log_f_backward:
            return self.actor_backward
        return self.actor

    def _resolve_flow_modules(self, *, direction: str) -> FlowModules:
        if direction == "backward":
            log_f = self.log_f_backward
        else:
            log_f = self.log_f
        return FlowModules(log_f=log_f)

    def _should_apply_h_guidance(self, *, flow_modules: FlowModules) -> bool:
        if self._h_guidance_spec is None:
            return False
        return flow_modules.log_f is self.log_f

    def _resolve_h_guidance_beta(self, *, progress: Optional[float], is_training: bool) -> float:
        spec = self._h_guidance_spec
        if spec is None:
            return float(_ZERO)
        if not is_training:
            return float(spec.beta_end) if spec.apply_eval else float(_ZERO)
        if progress is None:
            return float(spec.beta_start)
        progress_val = float(max(min(progress, float(_ONE)), float(_ZERO)))
        if progress_val < float(spec.warmup_progress):
            return float(spec.beta_start)
        return float(spec.beta_end)

    def _assert_inverse_relations_present(self, edge_relations: torch.Tensor) -> None:
        inverse_map = self._relation_inverse_map
        if inverse_map is None:
            raise RuntimeError("Backward stream requires relation_inverse_map to validate inverse relations.")
        edge_relations = edge_relations.to(device=inverse_map.device, dtype=torch.long).view(-1)
        if edge_relations.numel() == _ZERO:
            return
        rel_ids = torch.unique(edge_relations)
        inv_ids = inverse_map.index_select(0, rel_ids)
        if bool((inv_ids < _ZERO).any().detach().tolist()):
            bad = inv_ids[inv_ids < _ZERO][:5].tolist()
            raise ValueError(f"relation_inverse_map contains invalid ids (preview): {bad}.")
        max_id = int(torch.stack((rel_ids.max(), inv_ids.max())).max().detach().tolist())
        present = torch.zeros(max_id + _ONE, device=rel_ids.device, dtype=torch.bool)
        present.index_fill_(0, rel_ids, True)
        missing = ~present.index_select(0, inv_ids)
        if bool(missing.any().detach().tolist()):
            preview = inv_ids[missing][:5].tolist()
            raise ValueError(
                "Backward stream requires inverse relations present in the graph "
                f"(missing inverse ids preview: {preview})."
            )

    def _build_inverse_edge_ids(
        self,
        *,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        inverse_map = self._relation_inverse_map
        if inverse_map is None:
            raise RuntimeError("relation_inverse_map is required to build inverse edge ids.")
        if edge_index.numel() == _ZERO:
            return torch.zeros((0,), device=edge_index.device, dtype=torch.long)
        edge_relations = edge_relations.to(device=edge_index.device, dtype=torch.long).view(-1)
        if bool((edge_relations < _ZERO).any().detach().tolist()):
            bad = edge_relations[edge_relations < _ZERO][:_EDGE_INV_PREVIEW].tolist()
            raise ValueError(f"edge_relations contain invalid ids for inverse mapping (preview: {bad}).")
        inv_relations = inverse_map.index_select(0, edge_relations)
        if bool((inv_relations < _ZERO).any().detach().tolist()):
            bad = inv_relations[inv_relations < _ZERO][:_EDGE_INV_PREVIEW].tolist()
            raise ValueError(f"relation_inverse_map missing ids for edges (preview: {bad}).")
        heads = edge_index[_ZERO].to(dtype=torch.long)
        tails = edge_index[_ONE].to(dtype=torch.long)
        num_relations = int(inverse_map.numel())
        stride_tail = num_relations
        stride_head = num_nodes * num_relations
        edge_keys = heads * stride_head + tails * stride_tail + edge_relations
        inv_keys = tails * stride_head + heads * stride_tail + inv_relations
        sorted_keys, sorted_idx = torch.sort(edge_keys)
        if sorted_keys.numel() > _ONE:
            dup = sorted_keys[1:] == sorted_keys[:-1]
            if bool(dup.any().detach().tolist()):
                preview = sorted_keys[1:][dup][:_EDGE_INV_PREVIEW].tolist()
                raise ValueError(f"Duplicate edges detected while building inverse map (preview: {preview}).")
        pos = torch.searchsorted(sorted_keys, inv_keys)
        num_keys = int(sorted_keys.numel())
        max_pos = max(num_keys - _ONE, _ZERO)
        pos_safe = pos.clamp(min=_ZERO, max=max_pos)
        matched = sorted_keys.index_select(0, pos_safe) == inv_keys
        within = pos < num_keys
        valid = within & matched
        if not bool(valid.all().detach().tolist()):
            preview = inv_keys[~valid][:_EDGE_INV_PREVIEW].tolist()
            raise ValueError(f"Missing inverse edges for backward policy (preview keys: {preview}).")
        return sorted_idx.index_select(0, pos_safe)

    def _attach_inverse_edge_mask(self, graph_cache: Dict[str, torch.Tensor]) -> None:
        if self._relation_is_inverse is None:
            raise RuntimeError("relation_is_inverse is required for strict inverse edge policies.")
        edge_relations = graph_cache.get("edge_relations")
        if edge_relations is None:
            raise ValueError("edge_relations required to build inverse edge mask.")
        rel_ids = edge_relations.to(device=self._relation_is_inverse.device, dtype=torch.long).view(-1)
        mask = self._relation_is_inverse.index_select(0, rel_ids)
        if mask.numel() > 0:
            has_inverse = bool(mask.any().detach().tolist())
            has_forward = bool((~mask).any().detach().tolist())
            if not (has_inverse and has_forward):
                raise ValueError("edge_is_inverse mask requires both forward and inverse edges for strict mode.")
        graph_cache["edge_is_inverse"] = mask.to(device=edge_relations.device, dtype=torch.bool)

    def _attach_stream_policy_masks(
        self,
        graph_cache: Dict[str, torch.Tensor],
        *,
        flow_direction: str,
    ) -> None:
        edge_index = graph_cache.get("edge_index")
        if edge_index is None:
            raise ValueError("edge_index missing from graph_cache; cannot attach policy masks.")
        edge_is_inverse = self._require_edge_is_inverse(graph_cache, device=edge_index.device)
        direction = str(flow_direction).strip().lower()
        if direction == "forward":
            edge_policy_mask = ~edge_is_inverse
        elif direction == "backward":
            edge_policy_mask = edge_is_inverse
        else:
            raise ValueError(f"Unsupported flow_direction for policy masks: {flow_direction!r}.")
        graph_cache[EDGE_POLICY_MASK_KEY] = edge_policy_mask.to(device=edge_index.device, dtype=torch.bool)
        stop_node_mask = graph_cache.get("node_is_target")
        node_ptr = graph_cache.get("node_ptr")
        if stop_node_mask is None or node_ptr is None:
            raise ValueError("node_is_target/node_ptr missing from graph_cache; stop mask requires explicit targets.")
        node_ptr = node_ptr.to(device=edge_index.device, dtype=torch.long).view(-1)
        expected_nodes = int(node_ptr[-1].detach().item()) if node_ptr.numel() > _ZERO else _ZERO
        stop_node_mask = stop_node_mask.to(device=edge_index.device, dtype=torch.bool).view(-1)
        if stop_node_mask.numel() != expected_nodes:
            raise ValueError("stop_node_mask length mismatch with node_ptr.")
        graph_cache[STOP_NODE_MASK_KEY] = stop_node_mask

    @staticmethod
    def _require_log_pb_steps(log_pb_steps: Optional[torch.Tensor]) -> torch.Tensor:
        if log_pb_steps is None:
            raise RuntimeError("log_pb_steps missing; compute via engine before use.")
        return log_pb_steps

    @staticmethod
    def _require_edge_is_inverse(
        graph_cache: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        edge_is_inverse = graph_cache.get("edge_is_inverse")
        if edge_is_inverse is None:
            raise RuntimeError("edge_is_inverse missing from graph_cache; strict edge policy requires it.")
        return edge_is_inverse.to(device=device, dtype=torch.bool).view(-1)

    @staticmethod
    def _require_edge_policy_mask(
        graph_cache: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        edge_policy_mask = graph_cache.get(EDGE_POLICY_MASK_KEY)
        if edge_policy_mask is None:
            raise RuntimeError("edge_policy_mask missing from graph_cache; attach explicit policy mask.")
        return edge_policy_mask.to(device=device, dtype=torch.bool).view(-1)

    @staticmethod
    def _require_stop_node_mask(
        graph_cache: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        stop_node_mask = graph_cache.get(STOP_NODE_MASK_KEY)
        if stop_node_mask is None:
            raise RuntimeError("stop_node_mask missing from graph_cache; attach explicit stop mask.")
        return stop_node_mask.to(device=device, dtype=torch.bool).view(-1)

    def _prepare_backward_inputs(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        temperature: Optional[float],
        is_training: bool,
    ) -> tuple[RolloutInputs, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        self._assert_inverse_relations_present(inputs.edge_relations)
        query_tokens = inputs.question_tokens
        start_nodes = inputs.a_local_indices
        start_ptr = inputs.a_ptr
        start_nodes, start_ptr, log_pf_start = self._select_start_nodes_for_flow(
            log_f_module=self.log_f_backward,
            query_tokens=query_tokens,
            candidate_locals=start_nodes,
            candidate_ptr=start_ptr,
            node_ptr=inputs.node_ptr,
            node_tokens=inputs.node_tokens,
            stream="backward",
        )
        target_context = self._compute_target_context_tokens(
            inputs,
            node_locals=start_nodes,
            node_ptr=start_ptr,
        )
        back_inputs = replace(
            inputs,
            question_tokens=target_context,
            start_node_locals=start_nodes,
            start_ptr=start_ptr,
            target_node_locals=inputs.start_node_locals,
            target_ptr=inputs.start_ptr,
        )
        back_graph_cache = self._build_graph_cache_with_masks(
            inputs=back_inputs,
            device=inputs.node_ptr.device,
            flow_direction="backward",
        )
        back_graph_cache["query_tokens"] = query_tokens
        if "sample_ids" in graph_cache:
            back_graph_cache["sample_ids"] = graph_cache["sample_ids"]
        self._maybe_attach_entry_end_cache(
            log_f_module=self.log_f_backward,
            query_tokens=query_tokens,
            inputs=back_inputs,
            graph_cache=back_graph_cache,
            stream="backward",
        )
        return back_inputs, back_graph_cache, log_pf_start


    @staticmethod
    def _build_start_node_matrix(
        *,
        start_nodes: torch.Tensor,
        start_ptr: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_nodes = start_nodes.to(device=start_ptr.device, dtype=torch.long).view(-1)
        start_ptr = start_ptr.to(device=start_ptr.device, dtype=torch.long).view(-1)
        if start_ptr.numel() != num_graphs + _ONE:
            raise ValueError("start_ptr length mismatch for log_f_start.")
        counts = start_ptr[_ONE:] - start_ptr[:-_ONE]
        if not bool((counts > _ZERO).all().detach().tolist()):
            raise ValueError("start_node_locals missing for some graphs; filter_missing_start required.")
        total = int(start_nodes.numel())
        if total != int(counts.sum().detach().tolist()):
            raise ValueError("start_node_locals length mismatch with start_ptr counts.")
        max_count = int(counts.max().detach().tolist())
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=start_ptr.device), counts)
        offsets = torch.arange(total, device=start_ptr.device) - start_ptr.index_select(0, graph_ids)
        state_nodes = torch.zeros((num_graphs, max_count), device=start_ptr.device, dtype=torch.long)
        state_nodes.index_put_((graph_ids, offsets), start_nodes)
        col_ids = torch.arange(max_count, device=start_ptr.device).view(1, -1)
        valid_mask = col_ids < counts.view(-1, 1)
        return state_nodes, valid_mask

    def _compute_initial_hidden(
        self,
        *,
        inputs: RolloutInputs,
        actor: GFlowNetActor,
        start_node_locals: Optional[torch.Tensor] = None,
        start_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        device = inputs.node_ptr.device
        if num_graphs <= _ZERO:
            return torch.zeros((num_graphs, actor.agent.hidden_dim), device=device, dtype=inputs.node_tokens.dtype)
        if actor.context_mode == "question":
            return actor.agent.initialize_state(inputs.question_tokens, context_mode=actor.context_mode)
        if actor.context_mode == "start_node":
            if start_node_locals is None:
                start_node_locals = inputs.start_node_locals
            if start_ptr is None:
                start_ptr = inputs.start_ptr
            start_nodes, has_start = GFlowNetBatchProcessor.compute_single_start_nodes(
                start_node_locals=start_node_locals,
                start_ptr=start_ptr,
                num_graphs=num_graphs,
                device=device,
            )
            node_tokens = inputs.node_tokens.index_select(0, start_nodes.clamp(min=_ZERO))
            valid = has_start.to(device=node_tokens.device, dtype=torch.bool)
            context_tokens = torch.where(valid.unsqueeze(-1), node_tokens, torch.zeros_like(node_tokens))
            return actor.agent.initialize_state(context_tokens, context_mode=actor.context_mode)
        raise ValueError(f"Unknown actor.context_mode: {actor.context_mode!r}.")

    def _compute_target_context_tokens(
        self,
        inputs: RolloutInputs,
        *,
        node_locals: Optional[torch.Tensor] = None,
        node_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        device = inputs.node_ptr.device
        if num_graphs <= _ZERO:
            return torch.zeros((num_graphs, inputs.node_tokens.size(-1)), device=device, dtype=inputs.node_tokens.dtype)
        node_locals = inputs.target_node_locals if node_locals is None else node_locals
        node_ptr = inputs.target_ptr if node_ptr is None else node_ptr
        node_locals, node_ptr = self._maybe_sample_targets(node_locals, node_ptr)
        target_nodes, has_target = GFlowNetBatchProcessor.compute_single_start_nodes(
            start_node_locals=node_locals,
            start_ptr=node_ptr,
            num_graphs=num_graphs,
            device=device,
        )
        node_tokens = inputs.node_tokens.index_select(0, target_nodes.clamp(min=_ZERO))
        valid = has_target.to(device=node_tokens.device, dtype=torch.bool)
        return torch.where(valid.unsqueeze(-1), node_tokens, torch.zeros_like(node_tokens))

    def _maybe_sample_targets(
        self,
        node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._target_sampling_spec.mode != _TARGET_SAMPLING_MODE_RANDOM_ONE:
            return node_locals, node_ptr
        return self._sample_one_target_per_graph(node_locals, node_ptr)

    @staticmethod
    def _sample_one_per_graph(
        *,
        candidate_locals: torch.Tensor,
        candidate_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_ptr = candidate_ptr.to(dtype=torch.long).view(-1)
        candidate_locals = candidate_locals.to(device=candidate_ptr.device, dtype=torch.long).view(-1)
        num_graphs = int(candidate_ptr.numel() - _ONE)
        device = candidate_ptr.device
        if num_graphs <= _ZERO:
            empty = torch.empty((_ZERO,), device=device, dtype=torch.long)
            log_prob = torch.empty((_ZERO,), device=device, dtype=torch.float32)
            ptr = torch.zeros((_ONE,), device=device, dtype=torch.long)
            return empty, ptr, log_prob
        counts = (candidate_ptr[_ONE:] - candidate_ptr[:-_ONE]).clamp(min=_ZERO)
        counts_f = counts.to(dtype=torch.float32)
        log_prob = -torch.log(counts_f.clamp(min=_ONE))
        if candidate_locals.numel() == _ZERO:
            ptr = torch.zeros((num_graphs + _ONE), device=device, dtype=torch.long)
            return candidate_locals, ptr, log_prob
        offsets = (torch.rand((num_graphs,), device=device) * counts_f).to(dtype=torch.long)
        has_candidate = counts > _ZERO
        valid_graphs = has_candidate.nonzero(as_tuple=False).view(-1)
        if valid_graphs.numel() == _ZERO:
            empty = torch.empty((_ZERO,), device=device, dtype=torch.long)
            ptr = torch.zeros((num_graphs + _ONE), device=device, dtype=torch.long)
            return empty, ptr, log_prob
        starts = candidate_ptr[:-_ONE].index_select(0, valid_graphs)
        choices = offsets.index_select(0, valid_graphs)
        sample_idx = starts + choices
        sampled = candidate_locals.index_select(0, sample_idx)
        ptr = torch.zeros((num_graphs + _ONE), device=device, dtype=torch.long)
        ptr[_ONE:] = has_candidate.to(dtype=torch.long).cumsum(dim=0)
        return sampled, ptr, log_prob

    def _sample_one_target_per_graph(
        self,
        node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled, ptr, _ = self._sample_one_per_graph(
            candidate_locals=node_locals,
            candidate_ptr=node_ptr,
        )
        return sampled, ptr

    def _build_start_state_vec(
        self,
        *,
        inputs: RolloutInputs,
        actor: GFlowNetActor,
        state_nodes: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs, num_steps = state_nodes.shape
        device = inputs.node_ptr.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps, actor.agent.hidden_dim), device=device, dtype=inputs.node_tokens.dtype)
        if actor.context_mode == "question":
            hidden = actor.agent.initialize_state(inputs.question_tokens, context_mode=actor.context_mode)
            return hidden.unsqueeze(1).expand(num_graphs, num_steps, -1)
        if actor.context_mode == "start_node":
            node_tokens = inputs.node_tokens.index_select(0, state_nodes.to(device=device, dtype=torch.long).view(-1))
            hidden_flat = actor.agent.initialize_state(node_tokens, context_mode=actor.context_mode)
            return hidden_flat.view(num_graphs, num_steps, -1)
        raise ValueError(f"Unknown actor.context_mode: {actor.context_mode!r}.")

    def _build_action_token_sequence(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.dim() != _TWO:
            raise ValueError("actions must be [B, T] for action token encoding.")
        num_graphs, num_steps = actions.shape
        device = inputs.node_ptr.device
        action_mask = actions >= _ZERO
        token_dim = int(inputs.node_tokens.size(-1))
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            zeros = torch.zeros((num_graphs, num_steps, token_dim), device=device, dtype=inputs.node_tokens.dtype)
            return zeros, zeros, action_mask
        num_edges = int(inputs.edge_index.size(1))
        if num_edges <= _ZERO:
            zeros = torch.zeros((num_graphs, num_steps, token_dim), device=device, dtype=inputs.node_tokens.dtype)
            return zeros, zeros, action_mask
        action_ids = actions.to(device=device, dtype=torch.long).clamp(min=_ZERO)
        flat_ids = action_ids.view(-1)
        relation_tokens = inputs.relation_tokens.index_select(0, flat_ids).view(num_graphs, num_steps, -1)
        tail_nodes = inputs.edge_index[_ONE].index_select(0, flat_ids).view(num_graphs, num_steps)
        node_tokens = inputs.node_tokens.index_select(0, tail_nodes.view(-1)).view(num_graphs, num_steps, -1)
        return relation_tokens, node_tokens, action_mask

    def _compute_state_sequence(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
        actor: GFlowNetActor,
        start_node_locals: Optional[torch.Tensor] = None,
        start_ptr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TrajectoryStats]:
        stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
        relation_tokens, node_tokens, _ = self._build_action_token_sequence(actions=actions, inputs=inputs)
        hidden = self._compute_initial_hidden(
            inputs=inputs,
            actor=actor,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
        )
        state_seq, output_seq = actor.agent.encode_state_sequence(
            hidden=hidden,
            relation_tokens=relation_tokens,
            node_tokens=node_tokens,
            action_mask=stats.move_mask,
        )
        return state_seq, output_seq, stats

    def _compute_log_f_start(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        log_f_module: torch.nn.Module,
        start_node_locals: Optional[torch.Tensor] = None,
        start_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            return torch.zeros((num_graphs,), device=inputs.node_ptr.device, dtype=torch.float32)
        if start_node_locals is None:
            start_node_locals = inputs.start_node_locals
        if start_ptr is None:
            start_ptr = inputs.start_ptr
        start_node_locals = start_node_locals.to(device=inputs.node_ptr.device, dtype=torch.long)
        start_ptr = start_ptr.to(device=inputs.node_ptr.device, dtype=torch.long)
        state_nodes, valid_mask = self._build_start_node_matrix(
            start_nodes=start_node_locals,
            start_ptr=start_ptr,
            num_graphs=num_graphs,
        )
        log_f_states = self._compute_log_f_states(
            log_f_module=log_f_module,
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            state_nodes=state_nodes,
        )
        neg_inf = neg_inf_value(log_f_states)
        masked = torch.where(valid_mask, log_f_states, torch.full_like(log_f_states, neg_inf))
        return torch.logsumexp(masked, dim=1)

    def _compute_log_f_stop(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        stop_node_locals: torch.Tensor,
        log_f_module: torch.nn.Module,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        if stop_node_locals.numel() != num_graphs:
            raise ValueError("stop_node_locals length mismatch for log_f_stop.")
        stop_local = stop_node_locals.to(device=inputs.node_ptr.device, dtype=torch.long).view(-1)
        valid = stop_local >= _ZERO
        stop_local = stop_local.clamp(min=_ZERO)
        stop_global = inputs.node_ptr[:-1] + stop_local
        node_batch = graph_cache["node_batch"].index_select(0, stop_global)
        log_f_stop = log_f_module(
            node_tokens=inputs.node_tokens.index_select(0, stop_global),
            question_tokens=inputs.question_tokens,
            graph_features=flow_features,
            node_batch=node_batch,
        )
        if not bool(valid.all().detach().tolist()):
            log_f_stop = torch.where(valid, log_f_stop, torch.zeros_like(log_f_stop))
        return log_f_stop

    def _compute_start_nodes(self, *, inputs: RolloutInputs) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        device = inputs.node_ptr.device
        start_nodes, _ = GFlowNetBatchProcessor.compute_single_start_nodes(
            start_node_locals=inputs.start_node_locals,
            start_ptr=inputs.start_ptr,
            num_graphs=num_graphs,
            device=device,
        )
        return start_nodes

    def _build_state_nodes(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
    ) -> torch.Tensor:
        if actions.dim() != _TWO:
            raise ValueError("actions must be [B, T] for state node reconstruction.")
        num_graphs, num_steps = actions.shape
        device = inputs.edge_index.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.long)
        start_nodes = self._compute_start_nodes(inputs=inputs).to(device=device, dtype=torch.long)
        if inputs.edge_index.numel() == _ZERO:
            return start_nodes.view(-1, 1).expand(num_graphs, num_steps)
        actions = actions.to(device=device, dtype=torch.long)
        action_ids = actions.clamp(min=_ZERO)
        flat_ids = action_ids.view(-1)
        tails = inputs.edge_index[_ONE].index_select(0, flat_ids).view(num_graphs, num_steps)
        step_ids = torch.arange(num_steps, device=device, dtype=torch.long).view(1, -1).expand(num_graphs, -1)
        valid_idx = torch.where(actions >= _ZERO, step_ids, torch.full_like(step_ids, -_ONE))
        last_idx = torch.cummax(valid_idx, dim=1).values
        tail_last = tails.gather(1, last_idx.clamp(min=_ZERO))
        node_after = torch.where(last_idx >= _ZERO, tail_last, start_nodes.view(-1, 1))
        state_nodes = torch.empty((num_graphs, num_steps), device=device, dtype=torch.long)
        state_nodes[:, 0] = start_nodes
        if num_steps > _ONE:
            state_nodes[:, 1:] = node_after[:, :-1]
        return state_nodes

    def _compute_stop_node_locals_from_actions(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
    ) -> torch.Tensor:
        if actions.dim() != _TWO:
            raise ValueError("actions must be [B, T] for stop node reconstruction.")
        num_graphs, num_steps = actions.shape
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.full((num_graphs,), _STOP_NODE_NONE, device=actions.device, dtype=torch.long)
        stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
        state_nodes = self._build_state_nodes(actions=actions, inputs=inputs)
        stop_idx = stats.stop_idx.to(device=state_nodes.device, dtype=torch.long)
        stop_nodes = state_nodes.gather(1, stop_idx.view(-1, _ONE)).squeeze(1)
        stop_locals, has_active = resolve_current_stop_locals(
            curr_nodes=stop_nodes,
            node_ptr=inputs.node_ptr,
            dtype=torch.long,
        )
        return torch.where(has_active, stop_locals, torch.full_like(stop_locals, _STOP_NODE_NONE))

    @staticmethod
    def _compute_reach_success(
        *,
        stop_node_locals: torch.Tensor,
        node_ptr: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> torch.Tensor:
        stop_locals = stop_node_locals.to(device=node_ptr.device, dtype=torch.long).view(-1)
        valid_stop = stop_locals >= _ZERO
        stop_globals = node_ptr[:-1] + stop_locals.clamp(min=_ZERO)
        hits = node_is_target.index_select(0, stop_globals.clamp(min=_ZERO))
        return hits & valid_stop

    def _finalize_rollout_stop(
        self,
        *,
        actions_seq: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        target_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_is_target = graph_cache.get(target_key)
        if node_is_target is None:
            raise ValueError(f"Target mask '{target_key}' missing from graph cache; supply explicit target flags.")
        stop_node_locals = self._compute_stop_node_locals_from_actions(actions=actions_seq, inputs=inputs)
        reach_success = self._compute_reach_success(
            stop_node_locals=stop_node_locals,
            node_ptr=inputs.node_ptr,
            node_is_target=node_is_target,
        )
        return stop_node_locals, reach_success.to(dtype=torch.float32)

    def _maybe_log_rollout_step_stats(
        self,
        *,
        actions_seq: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_direction: str,
    ) -> None:
        if not self._debug_rollout_steps:
            return
        if self._debug_rollout_logged:
            return
        if actions_seq.dim() != _TWO:
            raise ValueError("actions_seq must be [B, T] for rollout step stats.")
        num_graphs, num_steps = actions_seq.shape
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return
        out_degree_counts = self._resolve_out_degree_counts(
            graph_cache=graph_cache,
            flow_direction=flow_direction,
        )
        stop_rate, no_edge_rate, done_ratio, mean_out_degree = self._compute_rollout_step_stats(
            actions_seq=actions_seq,
            inputs=inputs,
            out_degree_counts=out_degree_counts,
        )
        self._log_rollout_step_stats(
            stop_rate=stop_rate,
            no_edge_rate=no_edge_rate,
            done_ratio=done_ratio,
            mean_out_degree=mean_out_degree,
            stream=flow_direction,
            epoch=self._debug_rollout_epoch,
        )
        self._debug_rollout_logged = True

    def _resolve_out_degree_counts(
        self,
        *,
        graph_cache: Dict[str, torch.Tensor],
        flow_direction: str,
    ) -> torch.Tensor:
        direction = str(flow_direction).strip().lower()
        if direction not in {"forward", "backward"}:
            raise ValueError(f"Unsupported flow_direction for rollout stats: {flow_direction!r}.")
        edge_index = graph_cache["edge_index"]
        keep = self._require_edge_policy_mask(graph_cache, device=edge_index.device)
        heads = edge_index[_ZERO]
        if heads.numel() != keep.numel():
            raise ValueError("edge_policy_mask length mismatch with edge_index.")
        heads = heads[keep]
        node_ptr = graph_cache["node_ptr"]
        num_nodes_total = int(node_ptr[-1].detach().tolist()) if node_ptr.numel() > 0 else _ZERO
        out_degree_counts = torch.zeros((num_nodes_total,), device=edge_index.device, dtype=torch.long)
        if heads.numel() > _ZERO:
            ones = torch.ones_like(heads, dtype=torch.long)
            out_degree_counts.index_add_(0, heads, ones)
        return out_degree_counts

    def _compute_rollout_step_stats(
        self,
        *,
        actions_seq: torch.Tensor,
        inputs: RolloutInputs,
        out_degree_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs, num_steps = actions_seq.shape
        device = out_degree_counts.device
        state_nodes = self._build_state_nodes(actions=actions_seq, inputs=inputs).to(device=device, dtype=torch.long)
        valid_nodes = state_nodes >= _ZERO
        safe_nodes = state_nodes.clamp(min=_ZERO)
        out_degree = out_degree_counts.index_select(0, safe_nodes.view(-1)).view(num_graphs, num_steps)
        out_degree = torch.where(valid_nodes, out_degree, torch.zeros_like(out_degree))

        actions_local = actions_seq.to(device=device, dtype=torch.long)
        stop_mask = actions_local == STOP_RELATION
        stop_mask = stop_mask.to(dtype=torch.bool)
        stops_cum = stop_mask.to(dtype=torch.long).cumsum(dim=1)
        stops_before = stops_cum - stop_mask.to(dtype=torch.long)
        start_ptr = inputs.start_ptr.to(device=device, dtype=torch.long)
        start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).clamp(min=_ZERO)
        missing_start = start_counts == _ZERO
        initial_done = missing_start | inputs.dummy_mask.to(device=device, dtype=torch.bool)
        active = (~initial_done).view(num_graphs, _ONE) & (stops_before == _ZERO)

        denom = active.sum(dim=0).clamp(min=_ONE).to(dtype=torch.float32)
        stop_rate = (stop_mask & active).sum(dim=0).to(dtype=torch.float32) / denom
        no_edge_rate = ((out_degree == _ZERO) & active).sum(dim=0).to(dtype=torch.float32) / denom
        done_ratio = (initial_done.view(num_graphs, _ONE) | (stops_cum > _ZERO)).to(dtype=torch.float32).mean(dim=0)
        mean_out_degree = (out_degree.to(dtype=torch.float32) * active.to(dtype=torch.float32)).sum(dim=0) / denom
        return stop_rate, no_edge_rate, done_ratio, mean_out_degree

    @staticmethod
    def _log_rollout_step_stats(
        *,
        stop_rate: torch.Tensor,
        no_edge_rate: torch.Tensor,
        done_ratio: torch.Tensor,
        mean_out_degree: torch.Tensor,
        stream: str,
        epoch: Optional[int],
    ) -> None:
        stop_rate = stop_rate.detach().cpu()
        no_edge_rate = no_edge_rate.detach().cpu()
        done_ratio = done_ratio.detach().cpu()
        mean_out_degree = mean_out_degree.detach().cpu()
        num_steps = int(stop_rate.numel())
        epoch_val = None if epoch is None else int(epoch)
        for step_idx in range(num_steps):
            fields = {
                "step": int(step_idx),
                "stream": stream,
                "stop_rate": float(stop_rate[step_idx].item()),
                "no_edge_rate": float(no_edge_rate[step_idx].item()),
                "done_ratio": float(done_ratio[step_idx].item()),
                "mean_out_degree": float(mean_out_degree[step_idx].item()),
            }
            if epoch_val is not None:
                fields["epoch"] = epoch_val
            log_event(
                logger,
                "gfn_rollout_step_stats",
                **fields,
            )

    def _compute_state_vec_sequence(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_modules: FlowModules,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.dim() != _TWO:
            raise ValueError("actions must be [B, T] for state reconstruction.")
        actor = self._resolve_actor_for_flow(flow_modules)
        state_vec, _, stats = self._compute_state_sequence(actions=actions, inputs=inputs, actor=actor)
        return state_vec, stats.num_moves

    @staticmethod
    def _build_edge_tail_matrix(
        *,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        num_edges = int(edge_index.size(1))
        num_graphs = int(edge_ptr.numel() - 1)
        if num_edges <= _ZERO or num_graphs <= _ZERO:
            return None, None, _ZERO
        edge_batch = edge_batch.to(device=edge_index.device, dtype=torch.long).view(-1)
        edge_ptr = edge_ptr.to(device=edge_index.device, dtype=torch.long).view(-1)
        edge_counts = edge_ptr[_ONE:] - edge_ptr[:-_ONE]
        max_edges = int(edge_counts.max().detach().tolist()) if edge_counts.numel() > _ZERO else _ZERO
        if max_edges <= _ZERO:
            return None, None, _ZERO
        edge_ids = torch.arange(num_edges, device=edge_index.device, dtype=torch.long)
        edge_offsets = edge_ids - edge_ptr.index_select(0, edge_batch)
        tail_nodes = torch.zeros((num_graphs, max_edges), device=edge_index.device, dtype=torch.long)
        tail_nodes.index_put_((edge_batch, edge_offsets), edge_index[_ONE].to(dtype=torch.long), accumulate=False)
        return tail_nodes, edge_offsets, max_edges

    def _compute_h_guidance(
        self,
        *,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        context_tokens: torch.Tensor,
        edge_ids: Optional[torch.Tensor],
        log_f_module: torch.nn.Module,
        stop_gradient: bool,
    ) -> torch.Tensor:
        edge_index = graph_cache["edge_index"]
        num_edges = int(edge_index.size(1))
        if num_edges <= _ZERO:
            return torch.zeros((num_edges,), device=edge_index.device, dtype=torch.float32)
        if edge_ids is None:
            edge_ids = torch.arange(num_edges, device=edge_index.device, dtype=torch.long)
        else:
            edge_ids = edge_ids.to(device=edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            return torch.zeros((_ZERO,), device=edge_index.device, dtype=torch.float32)
        edge_tail_nodes = edge_index[_ONE].index_select(0, edge_ids)
        node_tokens = graph_cache["node_tokens"].index_select(0, edge_tail_nodes)

        def _compute_log_f() -> torch.Tensor:
            node_batch = graph_cache["node_batch"].index_select(0, edge_tail_nodes)
            return log_f_module(
                node_tokens=node_tokens,
                question_tokens=context_tokens,
                graph_features=flow_features,
                node_batch=node_batch,
            )

        if stop_gradient:
            with torch.no_grad():
                log_f_flat = _compute_log_f()
        else:
            log_f_flat = _compute_log_f()
        return log_f_flat

    def _build_h_guidance_fn(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        beta: float,
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        spec = self._h_guidance_spec
        if spec is None or beta <= float(_ZERO):
            return None
        log_f_module = self.log_f_backward
        stop_gradient = spec.stop_gradient
        scale = float(spec.scale)
        target_context = self._compute_target_context_tokens(inputs)

        def _guidance(step_counts: torch.Tensor, edge_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            _ = step_counts
            _ = hidden
            log_f = self._compute_h_guidance(
                graph_cache=graph_cache,
                flow_features=flow_features,
                context_tokens=target_context,
                edge_ids=edge_ids,
                log_f_module=log_f_module,
                stop_gradient=stop_gradient,
            )
            return torch.tanh(log_f) * float(beta) * scale

        return _guidance

    def _compute_backward_log_pb_steps(
        self,
        *,
        actions_seq: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        temperature: Optional[float],
        flow_modules: FlowModules,
    ) -> torch.Tensor:
        actions = actions_seq
        if actions.dim() != _TWO:
            raise ValueError("actions_seq must be [B, T] for backward log-prob.")
        num_graphs, num_steps = actions.shape
        device = actions.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        self._assert_inverse_relations_present(inputs.edge_relations)
        num_nodes_total = int(inputs.node_ptr[-1].detach().tolist()) if inputs.node_ptr.numel() > 0 else _ZERO
        inverse_edge_ids = self._build_inverse_edge_ids(
            edge_index=graph_cache["edge_index"],
            edge_relations=graph_cache["edge_relations"],
            num_nodes=num_nodes_total,
        )
        state_nodes = self._build_state_nodes(actions=actions, inputs=inputs)
        state_nodes_after = state_nodes.clone()
        if num_steps > _ONE:
            state_nodes_after[:, :-1] = state_nodes[:, 1:]
            state_nodes_after[:, -1] = state_nodes[:, -1]
        actor = self._resolve_actor_for_flow(flow_modules)
        edge_policy_mask = self._require_edge_policy_mask(graph_cache, device=device)
        backward_edge_mask = ~edge_policy_mask
        _, state_vec_after, stats = self._compute_state_sequence(
            actions=actions,
            inputs=inputs,
            actor=actor,
        )
        num_moves = stats.num_moves.to(device=device, dtype=torch.long)
        move_mask = stats.move_mask
        step_ids = torch.arange(num_steps, device=device, dtype=num_moves.dtype).view(1, -1)
        step_counts_after = (num_moves.view(-1, 1) - _ONE - step_ids).clamp(min=_ZERO)
        edge_index = graph_cache["edge_index"]
        num_edges = int(edge_index.size(1))
        if num_edges <= _ZERO:
            if bool(move_mask.any().detach().tolist()):
                raise ValueError("Backward log_pb encountered move actions with empty edge_index.")
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        log_pb_steps = torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        edge_batch = graph_cache["edge_batch"].to(device=device, dtype=torch.long).view(-1)
        action_keys = self._get_action_keys(graph_cache=graph_cache, actor=actor)
        edge_ids_base = torch.arange(num_edges, device=device, dtype=torch.long)
        action_ids = actions.clamp(min=_ZERO)
        inv_edge_ids = inverse_edge_ids.index_select(0, action_ids.view(-1)).view(num_graphs, num_steps)
        inv_edge_ids_t = inv_edge_ids.transpose(0, 1)
        temp, _ = actor._resolve_temperature(temperature)
        # Chunk over step dimension to cap repeated edge tensors and avoid OOM on large graphs.
        max_repeated_edges = max(int(self._log_pb_max_repeated_edges), num_edges, _ONE)
        block_span = max(_ONE, min(num_steps, max_repeated_edges // max(num_edges, _ONE)))
        for block_start in range(_ZERO, num_steps, block_span):
            block_end = min(num_steps, block_start + block_span)
            block_size = block_end - block_start
            block_mask = move_mask[:, block_start:block_end]
            if not bool(block_mask.any().detach().tolist()):
                continue
            # Flatten step-major to align with edge_batch_rep (edges repeated per step).
            state_vec_block = (
                state_vec_after[:, block_start:block_end, :].transpose(0, 1).reshape(block_size * num_graphs, -1)
            )
            curr_nodes_block = state_nodes_after[:, block_start:block_end].transpose(0, 1).reshape(-1)
            step_counts_block = step_counts_after[:, block_start:block_end].transpose(0, 1).reshape(-1)
            step_offsets = torch.arange(block_size, device=device, dtype=edge_batch.dtype)
            step_offsets = step_offsets.repeat_interleave(num_edges)
            edge_batch_rep = edge_batch.repeat(block_size) + (step_offsets * int(num_graphs))
            heads_rep = edge_index[_ZERO].repeat(block_size)
            curr_nodes_per_edge = curr_nodes_block.index_select(0, edge_batch_rep)
            step_counts_per_edge = step_counts_block.index_select(0, edge_batch_rep)
            horizon_exhausted = step_counts_per_edge >= int(self.env.max_steps)
            head_match = heads_rep == curr_nodes_per_edge
            policy_mask_rep = backward_edge_mask.repeat(block_size)
            valid_edges_rep = head_match & (~horizon_exhausted) & (curr_nodes_per_edge >= _ZERO) & policy_mask_rep
            edge_ids_rep = edge_ids_base.repeat(block_size)
            edge_scores_rep = actor.agent.score_cached(
                hidden=state_vec_block,
                action_keys=action_keys,
                edge_batch=edge_batch_rep,
                valid_edges_mask=valid_edges_rep,
                edge_ids=edge_ids_rep,
            )
            log_prob_edge_rep, _, _, _ = compute_policy_log_probs(
                edge_logits=edge_scores_rep,
                stop_logits=None,
                edge_batch=edge_batch_rep,
                valid_edges=valid_edges_rep,
                num_graphs=num_graphs * block_size,
                temperature=temp,
                allow_stop=None,
            )
            log_prob_edge_steps = log_prob_edge_rep.view(block_size, num_edges)
            valid_edges_steps = valid_edges_rep.view(block_size, num_edges)
            inv_edge_ids_block = inv_edge_ids_t[block_start:block_end]
            log_pb_block_t = log_prob_edge_steps.gather(1, inv_edge_ids_block)
            edge_valid_block = valid_edges_steps.gather(1, inv_edge_ids_block).transpose(0, 1)
            log_pb_block = log_pb_block_t.transpose(0, 1)
            log_pb_block = torch.where(block_mask, log_pb_block, torch.zeros_like(log_pb_block))
            if bool(block_mask.any().detach().tolist()) and not bool(edge_valid_block[block_mask].all().detach().tolist()):
                raise ValueError("Backward log_pb encountered invalid parent edges.")
            log_pb_steps[:, block_start:block_end] = log_pb_block
        return log_pb_steps

    def _compute_forward_log_pf_steps(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        temperature: Optional[float],
        flow_modules: FlowModules,
        enforce_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if actions.dim() != _TWO:
            raise ValueError("actions must be [B, T] for forward log-prob.")
        num_graphs, num_steps = actions.shape
        device = actions.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        edge_index = graph_cache["edge_index"]
        num_edges = int(edge_index.size(1))
        if num_edges <= _ZERO:
            stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
            if bool(stats.move_mask.any().detach().tolist()):
                raise ValueError("Forward log_pf encountered move actions with empty edge_index.")
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        actor = self._resolve_actor_for_flow(flow_modules)
        edge_policy_mask = self._require_edge_policy_mask(graph_cache, device=device)
        stop_node_mask = self._require_stop_node_mask(graph_cache, device=device)
        state_vec, _, stats = self._compute_state_sequence(
            actions=actions,
            inputs=inputs,
            actor=actor,
        )
        state_nodes = self._build_state_nodes(actions=actions, inputs=inputs)
        num_moves = stats.num_moves.to(device=device, dtype=torch.long)
        move_mask = stats.move_mask
        step_mask_incl_stop = stats.step_mask_incl_stop
        enforce = None
        if enforce_mask is not None:
            enforce = enforce_mask.to(device=device, dtype=torch.bool).view(-1)
            if enforce.numel() != num_graphs:
                raise ValueError("enforce_mask length mismatch with actions batch size.")
        step_ids = torch.arange(num_steps, device=device, dtype=num_moves.dtype).view(1, -1).expand(num_graphs, -1)
        log_pf_steps = torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        edge_batch = graph_cache["edge_batch"].to(device=device, dtype=torch.long).view(-1)
        action_keys = self._get_action_keys(graph_cache=graph_cache, actor=actor)
        edge_ids_base = torch.arange(num_edges, device=device, dtype=torch.long)
        temp, _ = actor._resolve_temperature(temperature)
        max_repeated_edges = max(int(self._log_pb_max_repeated_edges), num_edges, _ONE)
        block_span = max(_ONE, min(num_steps, max_repeated_edges // max(num_edges, _ONE)))
        for block_start in range(_ZERO, num_steps, block_span):
            block_end = min(num_steps, block_start + block_span)
            block_size = block_end - block_start
            block_active = step_mask_incl_stop[:, block_start:block_end]
            if enforce is not None:
                block_active = block_active & enforce.view(-1, 1)
            if not bool(block_active.any().detach().tolist()):
                continue
            state_vec_block = (
                state_vec[:, block_start:block_end, :].transpose(0, 1).reshape(block_size * num_graphs, -1)
            )
            curr_nodes_block = state_nodes[:, block_start:block_end].transpose(0, 1).reshape(-1)
            step_counts_block = step_ids[:, block_start:block_end].transpose(0, 1).reshape(-1)
            done_before_block = (~block_active).transpose(0, 1).reshape(-1)
            step_offsets = torch.arange(block_size, device=device, dtype=edge_batch.dtype)
            step_offsets = step_offsets.repeat_interleave(num_edges)
            edge_batch_rep = edge_batch.repeat(block_size) + (step_offsets * int(num_graphs))
            heads_rep = edge_index[_ZERO].repeat(block_size)
            curr_nodes_per_edge = curr_nodes_block.index_select(0, edge_batch_rep)
            step_counts_per_edge = step_counts_block.index_select(0, edge_batch_rep)
            done_before_per_edge = done_before_block.index_select(0, edge_batch_rep)
            horizon_exhausted = step_counts_per_edge >= int(self.env.max_steps)
            head_match = heads_rep == curr_nodes_per_edge
            policy_mask_rep = edge_policy_mask.repeat(block_size)
            valid_edges_rep = (
                head_match & (~horizon_exhausted) & (~done_before_per_edge) & (curr_nodes_per_edge >= _ZERO) & policy_mask_rep
            )
            has_edge_rep, _ = scatter_max(
                valid_edges_rep.to(dtype=torch.long),
                edge_batch_rep,
                dim=0,
                dim_size=num_graphs * block_size,
            )
            has_edge = has_edge_rep > _ZERO
            edge_ids_rep = edge_ids_base.repeat(block_size)
            edge_scores_rep = actor.agent.score_cached(
                hidden=state_vec_block,
                action_keys=action_keys,
                edge_batch=edge_batch_rep,
                valid_edges_mask=valid_edges_rep,
                edge_ids=edge_ids_rep,
            )
            stop_logits = actor._compute_stop_logits(
                state_vec=state_vec_block,
                edge_scores=edge_scores_rep,
                edge_batch=edge_batch_rep,
                num_graphs=num_graphs * block_size,
                edge_valid_mask=valid_edges_rep,
            )
            state_nodes_block = state_nodes[:, block_start:block_end].transpose(0, 1).reshape(-1)
            valid_nodes_block = state_nodes_block >= _ZERO
            is_target = stop_node_mask.index_select(0, state_nodes_block.clamp(min=_ZERO)) & valid_nodes_block
            active = block_active.transpose(0, 1).reshape(-1)
            allow_stop = active & (is_target | (~has_edge))
            log_prob_edge_rep, log_prob_stop_rep, _, _ = compute_policy_log_probs(
                edge_logits=edge_scores_rep,
                stop_logits=stop_logits,
                edge_batch=edge_batch_rep,
                valid_edges=valid_edges_rep,
                num_graphs=num_graphs * block_size,
                temperature=temp,
                allow_stop=allow_stop,
            )
            actions_block = actions[:, block_start:block_end]
            allow_stop_block = allow_stop.view(block_size, num_graphs).transpose(0, 1)
            invalid_stop = (actions_block == STOP_RELATION) & block_active & (~allow_stop_block)
            if bool(invalid_stop.any().detach().tolist()):
                count = int(invalid_stop.sum().detach().tolist())
                raise ValueError(f"Forward log_pf encountered STOP at non-target nodes for {count} graphs.")
            action_ids = actions_block.clamp(min=_ZERO)
            step_offsets = torch.arange(block_size, device=device, dtype=action_ids.dtype).view(1, -1) * num_edges
            flat_idx = (action_ids + step_offsets).view(-1)
            log_prob_edge_block = log_prob_edge_rep.index_select(0, flat_idx).view(num_graphs, block_size)
            valid_edge_block = valid_edges_rep.index_select(0, flat_idx).view(num_graphs, block_size)
            log_prob_stop_block = log_prob_stop_rep.view(block_size, num_graphs).transpose(0, 1)
            selected = torch.where(actions_block == STOP_RELATION, log_prob_stop_block, log_prob_edge_block)
            selected = torch.where(block_active, selected, torch.zeros_like(selected))
            move_mask_block = move_mask[:, block_start:block_end]
            if enforce is not None:
                move_mask_block = move_mask_block & enforce.view(-1, 1)
            if bool(move_mask_block.any().detach().tolist()) and not bool(
                valid_edge_block[move_mask_block].all().detach().tolist()
            ):
                raise ValueError("Forward log_pf encountered invalid chosen edges.")
            log_pf_steps[:, block_start:block_end] = selected
        return log_pf_steps

    def _get_inverse_edge_ids_cached(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cached = graph_cache.get(_INVERSE_EDGE_IDS_KEY)
        if cached is not None:
            return cached
        num_nodes_total = int(inputs.node_ptr[-1].detach().tolist()) if inputs.node_ptr.numel() > _ZERO else _ZERO
        inverse_edge_ids = self._build_inverse_edge_ids(
            edge_index=graph_cache["edge_index"],
            edge_relations=graph_cache["edge_relations"],
            num_nodes=num_nodes_total,
        )
        graph_cache[_INVERSE_EDGE_IDS_KEY] = inverse_edge_ids
        return inverse_edge_ids

    def _build_imitation_graph_cache(
        self,
        *,
        base_graph_cache: Dict[str, torch.Tensor],
        inputs: RolloutInputs,
        flow_direction: str,
    ) -> Dict[str, torch.Tensor]:
        device = inputs.node_ptr.device
        out = dict(base_graph_cache)
        out["start_node_locals"] = inputs.start_node_locals
        out["start_ptr"] = inputs.start_ptr
        out["target_node_locals"] = inputs.target_node_locals
        out["target_ptr"] = inputs.target_ptr
        out["question_tokens"] = inputs.question_tokens
        out["dummy_mask"] = inputs.dummy_mask
        num_nodes_total = int(inputs.node_ptr[-1].detach().tolist()) if inputs.node_ptr.numel() > _ZERO else _ZERO
        node_is_start, node_is_target = GFlowNetBatchProcessor.compute_node_flags(
            num_nodes_total,
            inputs.start_node_locals,
            inputs.target_node_locals,
            device,
        )
        out["node_is_start"] = node_is_start
        out["node_is_target"] = node_is_target
        self._attach_stream_policy_masks(out, flow_direction=flow_direction)
        return out

    @staticmethod
    def _assert_reflection_starts_from_node(
        *,
        reflected_actions: torch.Tensor,
        start_globals: torch.Tensor,
        edge_index: torch.Tensor,
        active_mask: torch.Tensor,
        label: str,
    ) -> None:
        if reflected_actions.dim() != _TWO:
            raise ValueError("reflected_actions must be [B, T].")
        first_actions = reflected_actions[:, _ZERO].to(device=start_globals.device, dtype=torch.long)
        move_first = first_actions >= _ZERO
        check = active_mask.to(device=start_globals.device, dtype=torch.bool).view(-1) & move_first
        if not bool(check.any().detach().tolist()):
            return
        edge_index = edge_index.to(device=start_globals.device, dtype=torch.long)
        heads = edge_index[_ZERO].index_select(0, first_actions.clamp(min=_ZERO))
        mismatch = check & (heads != start_globals.to(device=start_globals.device, dtype=torch.long))
        if bool(mismatch.any().detach().tolist()):
            bad = mismatch.nonzero(as_tuple=False).view(-1)
            preview = bad[:5].to(device="cpu").tolist()
            raise ValueError(f"Imitation reflection mismatch ({label}): head != start (preview idx={preview}).")

    def _compute_online_imitation_loss(
        self,
        *,
        actions_seq: torch.Tensor,
        num_moves: torch.Tensor,
        reach_success: torch.Tensor,
        stop_node_locals: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_direction: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = int(inputs.node_ptr.numel() - _ONE)
        device = inputs.node_ptr.device
        dtype = torch.float32
        zeros_graph = torch.zeros((num_graphs,), device=device, dtype=dtype)
        spec = self._imitation_spec
        if spec is None or num_graphs <= _ZERO:
            return torch.zeros((), device=device, dtype=dtype), zeros_graph, zeros_graph
        active = reach_success.to(device=device, dtype=torch.bool).view(-1) & (~inputs.dummy_mask.to(device=device, dtype=torch.bool).view(-1))
        if not bool(active.any().detach().tolist()):
            return torch.zeros((), device=device, dtype=dtype), zeros_graph, zeros_graph
        inverse_edge_ids = self._get_inverse_edge_ids_cached(inputs=inputs, graph_cache=graph_cache)
        reflected_actions, _ = reflect_backward_to_forward(
            actions_seq=actions_seq,
            num_moves=num_moves,
            edge_inverse_map=inverse_edge_ids,
            stop_value=STOP_RELATION,
        )
        stop_locals = stop_node_locals.to(device=device, dtype=torch.long).view(-1)
        node_offsets = inputs.node_ptr[:-_ONE].to(device=device, dtype=torch.long)
        start_globals = node_offsets + stop_locals.clamp(min=_ZERO)
        start_ptr = torch.arange(num_graphs + _ONE, device=device, dtype=torch.long)
        if flow_direction == "backward":
            student_direction = "forward"
            query_tokens = graph_cache.get("query_tokens")
            if query_tokens is None:
                query_tokens = inputs.question_tokens
            question_tokens = query_tokens
            target_node_locals = inputs.a_local_indices
            target_ptr = inputs.a_ptr
        elif flow_direction == "forward":
            student_direction = "backward"
            question_tokens = self._compute_target_context_tokens(
                inputs,
                node_locals=start_globals,
                node_ptr=start_ptr,
            )
            target_node_locals = inputs.q_local_indices
            target_ptr = inputs.q_ptr
        else:
            raise ValueError(f"Unsupported flow_direction for imitation: {flow_direction!r}.")
        student_inputs = replace(
            inputs,
            question_tokens=question_tokens,
            start_node_locals=start_globals,
            start_ptr=start_ptr,
            target_node_locals=target_node_locals,
            target_ptr=target_ptr,
        )
        student_cache = self._build_imitation_graph_cache(
            base_graph_cache=graph_cache,
            inputs=student_inputs,
            flow_direction=student_direction,
        )
        self._assert_reflection_starts_from_node(
            reflected_actions=reflected_actions,
            start_globals=start_globals,
            edge_index=student_cache["edge_index"],
            active_mask=active,
            label=f"{flow_direction}->{student_direction}",
        )
        student_flow_modules = self._resolve_flow_modules(direction=student_direction)
        log_pf_steps = self._compute_forward_log_pf_steps(
            actions=reflected_actions,
            inputs=student_inputs,
            graph_cache=student_cache,
            temperature=None,
            flow_modules=student_flow_modules,
            enforce_mask=active,
        )
        nll = -log_pf_steps.sum(dim=1)
        per_graph = nll * float(spec.weight)
        per_graph = per_graph * active.to(device=device, dtype=per_graph.dtype)
        denom = active.to(device=device, dtype=per_graph.dtype).sum().clamp(min=float(_ONE))
        loss = per_graph.sum() / denom
        return loss, active.to(device=device, dtype=dtype), per_graph.to(device=device, dtype=dtype)

    def _build_replay_actions_from_backward(
        self,
        *,
        actions_seq: torch.Tensor,
        reach_success: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions_seq
        if actions.dim() != _TWO:
            raise ValueError("actions_seq must be [B, T] for replay reversal.")
        num_graphs, num_steps = actions.shape
        success_mask = reach_success.to(device=actions.device, dtype=torch.bool).view(-1)
        stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
        num_moves = stats.num_moves
        num_nodes_total = int(inputs.node_ptr[-1].detach().item()) if inputs.node_ptr.numel() > _ZERO else _ZERO
        inverse_edge_ids = self._build_inverse_edge_ids(
            edge_index=graph_cache["edge_index"],
            edge_relations=graph_cache["edge_relations"],
            num_nodes=num_nodes_total,
        )
        replay_actions, _ = reflect_backward_to_forward(
            actions_seq=actions,
            num_moves=num_moves,
            edge_inverse_map=inverse_edge_ids,
            stop_value=STOP_RELATION,
        )
        return replay_actions, success_mask

    @staticmethod
    def _convert_replay_actions_to_local(
        *,
        actions: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if actions.dim() != _TWO:
            raise ValueError("Replay actions must be [B, T].")
        edge_ptr = edge_ptr.to(device=actions.device, dtype=torch.long).view(-1)
        num_graphs = int(edge_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            return actions
        if actions.size(0) != num_graphs:
            raise ValueError("Replay actions batch size mismatch with edge_ptr.")
        offsets = edge_ptr[:-_ONE].view(-1, _ONE)
        edge_end = edge_ptr[_ONE:].view(-1, _ONE)
        move_mask = actions >= _ZERO
        in_range = (~move_mask) | ((actions >= offsets) & (actions < edge_end))
        if bool((~in_range).any().detach().tolist()):
            raise ValueError("Replay actions contain edges outside per-graph ranges.")
        return torch.where(move_mask, actions - offsets, torch.full_like(actions, STOP_RELATION))

    @staticmethod
    def _convert_replay_actions_to_global(
        *,
        actions: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if actions.dim() != _TWO:
            raise ValueError("Replay actions must be [B, T].")
        edge_ptr = edge_ptr.to(device=actions.device, dtype=torch.long).view(-1)
        num_graphs = int(edge_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            return actions
        if actions.size(0) != num_graphs:
            raise ValueError("Replay actions batch size mismatch with edge_ptr.")
        offsets = edge_ptr[:-_ONE].view(-1, _ONE)
        edge_counts = (edge_ptr[_ONE:] - edge_ptr[:-_ONE]).view(-1, _ONE)
        move_mask = actions >= _ZERO
        in_range = (~move_mask) | (actions < edge_counts)
        if bool((~in_range).any().detach().tolist()):
            raise ValueError("Replay actions contain edges outside per-graph ranges.")
        return torch.where(move_mask, actions + offsets, torch.full_like(actions, STOP_RELATION))

    def _store_replay_from_backward_rollout(
        self,
        *,
        actions_seq: torch.Tensor,
        reach_success: torch.Tensor,
        stop_node_locals: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        sample_ids: Sequence[str],
    ) -> None:
        if self._replay_buffer is None:
            return
        replay_actions, success_mask = self._build_replay_actions_from_backward(
            actions_seq=actions_seq,
            reach_success=reach_success,
            inputs=inputs,
            graph_cache=graph_cache,
        )
        if replay_actions.numel() == _ZERO:
            return
        stop_locals = stop_node_locals.to(device=inputs.node_ptr.device, dtype=torch.long).view(-1)
        node_counts = (inputs.node_ptr[_ONE:] - inputs.node_ptr[:-_ONE]).to(device=stop_locals.device, dtype=torch.long)
        if stop_locals.numel() != node_counts.numel():
            raise ValueError("stop_node_locals length mismatch with node_ptr.")
        out_of_range = (stop_locals >= node_counts) & (stop_locals >= _ZERO)
        if bool(out_of_range.any().detach().tolist()):
            bad_idx = out_of_range.nonzero(as_tuple=False).view(-1)
            preview_idx = bad_idx[:_REPLAY_STOP_OOB_PREVIEW]
            preview_ids = [str(sample_ids[i]) for i in preview_idx.to(device="cpu").tolist()]
            log_event(
                logger,
                _REPLAY_STOP_OOB_EVENT,
                bad_count=int(bad_idx.numel()),
                sample_ids=preview_ids,
                stop_locals=stop_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
                node_counts=node_counts.index_select(0, preview_idx).to(device="cpu").tolist(),
            )
            raise ValueError("Replay stop_node_locals out of range; see gfn_replay_stop_oob logs for details.")
        stop_globals = inputs.node_ptr[:-_ONE] + stop_locals.clamp(min=_ZERO)
        q_globals = inputs.q_local_indices.to(device=inputs.node_ptr.device, dtype=torch.long)
        stop_in_q = torch.isin(stop_globals, q_globals) & (stop_locals >= _ZERO)
        bad_success = success_mask & (~stop_in_q)
        if bool(bad_success.any().detach().tolist()):
            bad_idx = bad_success.nonzero(as_tuple=False).view(-1)
            preview_idx = bad_idx[:_REPLAY_STOP_NOT_IN_Q_PREVIEW]
            preview_ids = [str(sample_ids[i]) for i in preview_idx.to(device="cpu").tolist()]
            log_event(
                logger,
                _REPLAY_STOP_NOT_IN_Q_EVENT,
                bad_count=int(bad_idx.numel()),
                sample_ids=preview_ids,
                stop_locals=stop_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
                stop_globals=stop_globals.index_select(0, preview_idx).to(device="cpu").tolist(),
            )
            raise ValueError("Backward stop not in E_Q; see gfn_replay_stop_not_in_q logs for details.")
        success_mask = success_mask & stop_in_q
        if not bool(success_mask.any().detach().tolist()):
            return
        first_actions = replay_actions[:, _ZERO].to(device=inputs.node_ptr.device, dtype=torch.long)
        move_first = first_actions >= _ZERO
        if bool(move_first.any().detach().tolist()):
            node_offsets = inputs.node_ptr[:-_ONE].to(device=inputs.node_ptr.device, dtype=torch.long)
            edge_index = graph_cache["edge_index"].to(device=inputs.node_ptr.device, dtype=torch.long)
            first_heads = edge_index[_ZERO].index_select(0, first_actions.clamp(min=_ZERO))
            first_head_locals = first_heads - node_offsets
            mismatch = success_mask & move_first & (first_head_locals != stop_locals)
            if bool(mismatch.any().detach().tolist()):
                bad_idx = mismatch.nonzero(as_tuple=False).view(-1)
                preview_idx = bad_idx[:_REPLAY_REFLECT_MISMATCH_PREVIEW]
                preview_ids = [str(sample_ids[i]) for i in preview_idx.to(device="cpu").tolist()]
                replay_actions_preview = replay_actions.index_select(0, preview_idx)
                preview_steps = min(_TWO, int(replay_actions_preview.size(1)))
                if int(replay_actions_preview.size(1)) != preview_steps:
                    replay_actions_preview = replay_actions_preview[:, :preview_steps]
                log_event(
                    logger,
                    _REPLAY_REFLECT_MISMATCH_EVENT,
                    level=logging.ERROR,
                    bad_count=int(bad_idx.numel()),
                    sample_ids=preview_ids,
                    stop_locals=stop_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
                    first_actions=first_actions.index_select(0, preview_idx).to(device="cpu").tolist(),
                    first_heads=first_heads.index_select(0, preview_idx).to(device="cpu").tolist(),
                    first_head_locals=first_head_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
                    node_offsets=node_offsets.index_select(0, preview_idx).to(device="cpu").tolist(),
                    replay_actions=replay_actions_preview.to(device="cpu").tolist(),
                )
                raise ValueError(
                    "Replay reflection mismatch: first head does not match stop_node_locals; "
                    "see gfn_replay_reflect_mismatch logs for details."
                )
        device = inputs.node_ptr.device
        num_graphs = int(inputs.node_ptr.numel() - 1)
        start_globals = stop_globals.to(device=device, dtype=torch.long)
        start_ptr = torch.arange(num_graphs + _ONE, device=device, dtype=torch.long)
        forward_inputs = replace(
            inputs,
            start_node_locals=start_globals,
            start_ptr=start_ptr,
            target_node_locals=inputs.a_local_indices.to(device=device, dtype=torch.long),
            target_ptr=inputs.a_ptr.to(device=device, dtype=torch.long),
        )
        forward_stop_locals = self._compute_stop_node_locals_from_actions(
            actions=replay_actions,
            inputs=forward_inputs,
        )
        num_nodes_total = (
            int(forward_inputs.node_ptr[-1].detach().tolist()) if forward_inputs.node_ptr.numel() > _ZERO else _ZERO
        )
        node_is_target = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if forward_inputs.a_local_indices.numel() > _ZERO:
            a_globals = forward_inputs.a_local_indices.to(device=device, dtype=torch.long)
            node_is_target[a_globals] = True
        reward_cache = {"node_is_target": node_is_target}
        reward_out: RewardOutput = self.reward_fn(
            **self._build_reward_kwargs(
                stop_node_locals=forward_stop_locals,
                inputs=forward_inputs,
                graph_cache=reward_cache,
            )
        )
        replay_actions = self._convert_replay_actions_to_local(
            actions=replay_actions,
            edge_ptr=graph_cache["edge_ptr"],
        )
        if len(sample_ids) != int(replay_actions.size(0)):
            raise ValueError("Replay sample_ids length mismatch with rollout batch.")
        self._replay_buffer.add(
            sample_ids=sample_ids,
            actions=replay_actions,
            mask=success_mask,
            start_nodes=stop_node_locals,
            log_reward=reward_out.log_reward,
            success=reward_out.success,
        )

    def _compute_replay_log_pf_start(
        self,
        *,
        inputs: RolloutInputs,
        start_locals: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        device = inputs.node_ptr.device
        if num_graphs <= _ZERO:
            return torch.empty((num_graphs,), device=device, dtype=torch.float32)
        if not self._entry_selectors_enabled():
            counts = (inputs.q_ptr[_ONE:] - inputs.q_ptr[:-_ONE]).clamp(min=_ZERO)
            counts_f = counts.to(dtype=torch.float32)
            return -torch.log(counts_f.clamp(min=_ONE))
        start_selector, _ = self._resolve_entry_selectors_for_flow(log_f_module=self.log_f)
        if start_selector is None:
            raise RuntimeError("entry selector enabled but start selector missing.")
        counts, graph_ids, candidate_globals, candidate_tokens, _ = self._build_candidate_view(
            candidate_locals=inputs.q_local_indices,
            candidate_ptr=inputs.q_ptr,
            node_ptr=inputs.node_ptr,
            node_tokens=inputs.node_tokens,
        )
        log_probs = self._compute_entry_log_probs(
            selector=start_selector,
            query_tokens=inputs.question_tokens,
            candidate_tokens=candidate_tokens,
            candidate_graph_ids=graph_ids,
            candidate_counts=counts,
            num_graphs=num_graphs,
        )
        start_locals = start_locals.to(device=device, dtype=torch.long).view(-1)
        if start_locals.numel() != num_graphs:
            raise ValueError("replay start_locals length mismatch with batch size.")
        return self._lookup_entry_log_prob_for_stop_nodes(
            candidate_globals=candidate_globals,
            candidate_log_probs=log_probs,
            stop_node_locals=start_locals,
            node_ptr=inputs.node_ptr,
        )

    def _raise_replay_invalid_edges(
        self,
        *,
        actions: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        sample_ids: Sequence[str],
    ) -> None:
        if actions.dim() != _TWO:
            raise ValueError("Replay actions must be [B, T].")
        num_graphs, num_steps = actions.shape
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            raise ValueError("Replay actions empty; cannot validate invalid edges.")
        device = actions.device
        edge_index = graph_cache["edge_index"].to(device=device, dtype=torch.long)
        edge_policy_mask = self._require_edge_policy_mask(graph_cache, device=device)
        stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
        move_mask = stats.move_mask
        num_moves = stats.num_moves.to(device=device, dtype=torch.long)
        step_mask_incl_stop = stats.step_mask_incl_stop
        step_ids = torch.arange(num_steps, device=device, dtype=num_moves.dtype).view(1, -1).expand(num_graphs, -1)
        active = step_mask_incl_stop
        horizon_ok = step_ids < int(self.env.max_steps)
        state_nodes = self._build_state_nodes(actions=actions, inputs=inputs)
        action_ids = actions.clamp(min=_ZERO).to(dtype=torch.long)
        flat_ids = action_ids.view(-1)
        heads = edge_index[_ZERO].index_select(0, flat_ids).view(num_graphs, num_steps)
        head_match = heads == state_nodes
        policy_ok = edge_policy_mask.index_select(0, flat_ids).view(num_graphs, num_steps)
        curr_valid = state_nodes >= _ZERO
        valid = head_match & policy_ok & curr_valid & active & horizon_ok
        invalid_move = move_mask & (~valid)
        if not bool(invalid_move.any().detach().tolist()):
            raise ValueError("Replay invalid edge check failed, but no invalid moves were found.")
        invalid_graphs = invalid_move.any(dim=1)
        invalid_idx = invalid_graphs.nonzero(as_tuple=False).view(-1)
        preview_idx = invalid_idx[:_REPLAY_INVALID_PREVIEW]
        first_bad = invalid_move.to(dtype=torch.long).argmax(dim=1).index_select(0, preview_idx)
        preview_actions = actions[preview_idx, first_bad].to(device="cpu").tolist()
        preview_heads = heads[preview_idx, first_bad].to(device="cpu").tolist()
        preview_curr = state_nodes[preview_idx, first_bad].to(device="cpu").tolist()
        preview_policy = policy_ok[preview_idx, first_bad].to(device="cpu").tolist()
        preview_head_match = head_match[preview_idx, first_bad].to(device="cpu").tolist()
        preview_active = active[preview_idx, first_bad].to(device="cpu").tolist()
        preview_horizon = horizon_ok[preview_idx, first_bad].to(device="cpu").tolist()
        preview_curr_valid = curr_valid[preview_idx, first_bad].to(device="cpu").tolist()
        sample_preview = [str(sample_ids[idx]) for idx in preview_idx.to(device="cpu").tolist()]
        log_event(
            logger,
            _REPLAY_INVALID_EVENT,
            invalid_count=int(invalid_idx.numel()),
            sample_ids=sample_preview,
            steps=first_bad.to(device="cpu").tolist(),
            actions=preview_actions,
            heads=preview_heads,
            curr_nodes=preview_curr,
            head_match=preview_head_match,
            policy_ok=preview_policy,
            active=preview_active,
            horizon_ok=preview_horizon,
            curr_valid=preview_curr_valid,
        )
        raise ValueError("Replay actions contain invalid edges; see gfn_replay_invalid_edges logs for details.")

    def _compute_replay_loss(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        temperature: Optional[float],
        subtb_spec: Optional[SubTrajectorySpec],
        sample_ids: Sequence[str],
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self._replay_spec is None or self._replay_buffer is None:
            return None, {}
        num_graphs = int(inputs.node_ptr.numel() - 1)
        if len(sample_ids) != num_graphs:
            raise ValueError("Replay sample_ids length mismatch with batch size.")
        (
            _,
            replay_mask,
            replay_start_locals,
            _,
            _,
        ) = self._replay_buffer.fetch(
            sample_ids=sample_ids,
            device=inputs.node_ptr.device,
        )
        if not bool(replay_mask.any().detach().tolist()):
            return None, {
                "replay/available_frac": torch.zeros((), device=inputs.node_ptr.device),
                "replay/invalid_frac": torch.zeros((), device=inputs.node_ptr.device),
            }
        replay_start_locals = replay_start_locals.to(device=inputs.node_ptr.device, dtype=torch.long).view(-1)
        if replay_start_locals.numel() != num_graphs:
            raise ValueError("Replay start_nodes length mismatch with batch size.")
        node_counts = (inputs.node_ptr[_ONE:] - inputs.node_ptr[:-_ONE]).to(
            device=inputs.node_ptr.device,
            dtype=torch.long,
        )
        has_replay_start = replay_start_locals >= _ZERO
        in_range = replay_start_locals < node_counts
        valid_replay_start = replay_mask & has_replay_start & in_range
        invalid_replay = replay_mask & (~valid_replay_start)
        if bool(invalid_replay.any().detach().tolist()):
            bad_idx = invalid_replay.nonzero(as_tuple=False).view(-1)
            preview_idx = bad_idx[:_REPLAY_START_INVALID_PREVIEW]
            preview_ids = [str(sample_ids[i]) for i in preview_idx.to(device="cpu").tolist()]
            log_event(
                logger,
                _REPLAY_START_INVALID_EVENT,
                bad_count=int(bad_idx.numel()),
                sample_ids=preview_ids,
                replay_start_locals=replay_start_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
            )
            invalid_ids = [str(sample_ids[i]) for i in bad_idx.to(device="cpu").tolist()]
            _ = self._replay_buffer.discard(sample_ids=invalid_ids)
        graph_mask = (~inputs.dummy_mask) & valid_replay_start
        if not bool(graph_mask.any().detach().tolist()):
            return None, {
                "replay/available_frac": graph_mask.to(dtype=torch.float32).mean(),
                "replay/invalid_frac": invalid_replay.to(dtype=torch.float32).mean(),
            }
        node_offsets = inputs.node_ptr[:-_ONE].to(device=inputs.node_ptr.device, dtype=torch.long)
        default_start_globals, has_default_start = GFlowNetBatchProcessor.compute_single_start_nodes(
            start_node_locals=inputs.start_node_locals,
            start_ptr=inputs.start_ptr,
            num_graphs=num_graphs,
            device=inputs.node_ptr.device,
        )
        if not bool(has_default_start.all().detach().tolist()):
            raise ValueError("Replay requires default start nodes for all graphs.")
        replay_start_globals = node_offsets + replay_start_locals.clamp(min=_ZERO)
        default_start_locals = default_start_globals - node_offsets
        start_locals = torch.where(valid_replay_start, replay_start_locals, default_start_locals)
        start_nodes = torch.where(valid_replay_start, replay_start_globals, default_start_globals)
        q_globals = inputs.q_local_indices.to(device=inputs.node_ptr.device, dtype=torch.long)
        start_in_q = torch.isin(start_nodes, q_globals)
        missing_q = graph_mask & (~start_in_q)
        if bool(missing_q.any().detach().tolist()):
            bad_idx = missing_q.nonzero(as_tuple=False).view(-1)
            preview_idx = bad_idx[:_REPLAY_START_MISS_PREVIEW]
            preview_ids = [str(sample_ids[i]) for i in preview_idx.to(device="cpu").tolist()]
            log_event(
                logger,
                _REPLAY_START_MISS_EVENT,
                bad_count=int(bad_idx.numel()),
                sample_ids=preview_ids,
                start_locals=start_locals.index_select(0, preview_idx).to(device="cpu").tolist(),
                start_globals=start_nodes.index_select(0, preview_idx).to(device="cpu").tolist(),
            )
            raise ValueError("Replay start nodes not in E_Q; see gfn_replay_start_missing_q logs for details.")
        start_ptr = torch.arange(num_graphs + _ONE, device=inputs.node_ptr.device, dtype=torch.long)
        replay_inputs, replay_cache = self._override_start_nodes(
            inputs=inputs,
            graph_cache=graph_cache,
            start_nodes=start_nodes,
            start_ptr=start_ptr,
        )
        if self._entry_selectors_enabled():
            globals_key = replay_cache.get("entry_end_globals")
            log_probs_key = replay_cache.get("entry_end_log_probs")
            if globals_key is None or log_probs_key is None:
                self._maybe_attach_entry_end_cache(
                    log_f_module=flow_modules.log_f,
                    query_tokens=replay_inputs.question_tokens,
                    inputs=replay_inputs,
                    graph_cache=replay_cache,
                    stream="forward",
                )
        log_f_start = self._compute_log_f_start(
            inputs=replay_inputs,
            graph_cache=replay_cache,
            flow_features=flow_features,
            log_f_module=flow_modules.log_f,
        )
        log_pf_start = self._compute_replay_log_pf_start(inputs=inputs, start_locals=start_locals)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            total_loss,
            record,
        ) = self._run_rollout_and_loss(
            inputs=replay_inputs,
            graph_cache=replay_cache,
            flow_features=flow_features,
            flow_modules=flow_modules,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            graph_mask=graph_mask,
            guidance_fn=None,
            temperature=temperature,
            subtb_spec=subtb_spec,
            max_steps_override=None,
            record_actions=True,
            compute_imitation=False,
        )
        metrics = {
            "replay/available_frac": graph_mask.to(dtype=torch.float32).mean(),
            "replay/buffer_size": torch.as_tensor(len(self._replay_buffer), device=inputs.node_ptr.device),
            "replay/loss": total_loss.detach(),
            "replay/log_reward": torch.where(
                graph_mask,
                record.log_reward,
                torch.zeros_like(record.log_reward),
            ).detach(),
            "replay/pass@1": torch.where(
                graph_mask,
                record.reward_out.success.to(dtype=torch.float32),
                torch.zeros_like(record.log_reward),
            ).detach(),
            "replay/invalid_frac": invalid_replay.to(dtype=torch.float32).mean(),
        }
        return total_loss, metrics

    @staticmethod
    def _gather_state_vec_at_step(
        state_vec: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> torch.Tensor:
        if state_vec.dim() != _THREE:
            raise ValueError("state_vec must be [B, T, S] for step selection.")
        num_steps = int(state_vec.size(1))
        if num_steps <= _ZERO:
            return torch.zeros((state_vec.size(0), state_vec.size(2)), device=state_vec.device, dtype=state_vec.dtype)
        step_idx = step_idx.to(device=state_vec.device, dtype=torch.long).clamp(min=_ZERO, max=num_steps - _ONE)
        idx = step_idx.view(-1, 1, 1).expand(-1, 1, state_vec.size(2))
        return state_vec.gather(1, idx).squeeze(1)

    def _compute_log_f_states(
        self,
        *,
        log_f_module: torch.nn.Module,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        state_nodes: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs, num_steps = state_nodes.shape
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps), device=inputs.node_ptr.device, dtype=torch.float32)
        flat_nodes = state_nodes.view(-1)
        node_tokens = inputs.node_tokens.index_select(0, flat_nodes)
        node_batch = graph_cache["node_batch"].index_select(0, flat_nodes)
        log_f_flat = log_f_module(
            node_tokens=node_tokens,
            question_tokens=inputs.question_tokens,
            graph_features=flow_features,
            node_batch=node_batch,
        )
        return log_f_flat.view(num_graphs, num_steps)

    @staticmethod
    def _sample_subtrajectory_indices(
        num_moves: torch.Tensor,
        *,
        num_subtrajectories: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_moves.dim() != _ONE:
            raise ValueError("num_moves must be [B] for sub-trajectory sampling.")
        num_graphs = int(num_moves.numel())
        if num_graphs <= _ZERO:
            empty = torch.zeros((num_graphs, num_subtrajectories), device=num_moves.device, dtype=torch.long)
            return empty, empty
        num_moves = num_moves.to(dtype=torch.long)
        max_start = (num_moves - _ONE).clamp(min=_ZERO)
        max_start_f = (max_start + _ONE).to(dtype=torch.float32)
        start = torch.floor(
            torch.rand((num_graphs, num_subtrajectories), device=num_moves.device) * max_start_f.unsqueeze(1)
        ).to(dtype=torch.long)
        span = (num_moves.unsqueeze(1) - start).clamp(min=_ONE)
        span_f = span.to(dtype=torch.float32)
        end = start + _ONE + torch.floor(
            torch.rand((num_graphs, num_subtrajectories), device=num_moves.device) * span_f
        ).to(dtype=torch.long)
        end = torch.minimum(end, num_moves.unsqueeze(1))
        return start, end

    @staticmethod
    def _segment_sum(prefix: torch.Tensor, start_idx: torch.Tensor, end_step: torch.Tensor) -> torch.Tensor:
        end_vals = prefix.gather(1, end_step)
        start_prev = (start_idx - _ONE).clamp(min=_ZERO)
        start_vals = prefix.gather(1, start_prev)
        start_vals = torch.where(start_idx > _ZERO, start_vals, torch.zeros_like(start_vals))
        return end_vals - start_vals

    def _compute_subtrajectory_values(
        self,
        *,
        actions_seq: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: Optional[torch.Tensor],
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        log_f_module: torch.nn.Module,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        log_pb_start: Optional[torch.Tensor],
        log_target: torch.Tensor,
        num_moves: torch.Tensor,
        spec: SubTrajectorySpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_nodes = self._build_state_nodes(actions=actions_seq, inputs=inputs)
        log_f_states = self._compute_log_f_states(
            log_f_module=log_f_module,
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            state_nodes=state_nodes,
        )
        start_idx, end_idx = self._sample_subtrajectory_indices(num_moves, num_subtrajectories=spec.num_subtrajectories)
        end_is_terminal = end_idx == num_moves.unsqueeze(1)
        end_step = end_idx - _ONE + end_is_terminal.to(dtype=torch.long)
        prefix_pf = log_pf_steps.cumsum(dim=1)
        log_pb_steps = self._require_log_pb_steps(log_pb_steps)
        prefix_pb = log_pb_steps.cumsum(dim=1)
        sum_pf = self._segment_sum(prefix_pf, start_idx, end_step)
        sum_pb = self._segment_sum(prefix_pb, start_idx, end_step)
        zero_len = end_idx == start_idx
        if bool(zero_len.any().detach().tolist()):
            sum_pf = torch.where(zero_len, torch.zeros_like(sum_pf), sum_pf)
            sum_pb = torch.where(zero_len, torch.zeros_like(sum_pb), sum_pb)
        log_f_start_nodes = log_f_states.gather(1, start_idx)
        log_f_start_sub = torch.where(start_idx == _ZERO, log_f_start.unsqueeze(1), log_f_start_nodes)
        if log_pf_start is not None:
            log_pf_start = log_pf_start.to(device=sum_pf.device, dtype=sum_pf.dtype).view(-1)
            if log_pf_start.numel() != sum_pf.size(0):
                raise ValueError("log_pf_start length mismatch with batch size.")
            sum_pf = sum_pf + torch.where(start_idx == _ZERO, log_pf_start.unsqueeze(1), torch.zeros_like(sum_pf))
        if log_pb_start is not None:
            log_pb_start = log_pb_start.to(device=sum_pb.device, dtype=sum_pb.dtype).view(-1)
            if log_pb_start.numel() != sum_pb.size(0):
                raise ValueError("log_pb_start length mismatch with batch size.")
            sum_pb = sum_pb + torch.where(start_idx == _ZERO, log_pb_start.unsqueeze(1), torch.zeros_like(sum_pb))
        log_f_end_nodes = log_f_states.gather(1, end_idx)
        log_target_sub = torch.where(end_is_terminal, log_target.unsqueeze(1), log_f_end_nodes)
        valid = (end_idx > start_idx) | (num_moves.unsqueeze(1) == _ZERO)
        residual = log_f_start_sub + sum_pf - sum_pb - log_target_sub
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return sum_pf, sum_pb, residual, log_f_start_sub, log_target_sub, valid, end_is_terminal

    @staticmethod
    def _reduce_subtrajectory_stats(
        *,
        sum_pf: torch.Tensor,
        sum_pb: torch.Tensor,
        residual: torch.Tensor,
        log_f_start: torch.Tensor,
        log_target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_count = valid.sum(dim=1).clamp(min=_ONE)
        weight = valid.to(dtype=residual.dtype)
        loss_per_graph = (residual.pow(2) * weight).sum(dim=1) / valid_count
        sum_pf_mean = (sum_pf * weight).sum(dim=1) / valid_count
        sum_pb_mean = (sum_pb * weight).sum(dim=1) / valid_count
        residual_mean = (residual * weight).sum(dim=1) / valid_count
        log_f_start_mean = (log_f_start * weight).sum(dim=1) / valid_count
        log_target_mean = (log_target * weight).sum(dim=1) / valid_count
        return sum_pf_mean, sum_pb_mean, residual_mean, log_f_start_mean, log_target_mean, loss_per_graph

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

    @staticmethod
    def _suffix_metrics(metrics: Dict[str, torch.Tensor], *, suffix: str) -> Dict[str, torch.Tensor]:
        return {f"{name}_{suffix}": value for name, value in metrics.items()}

    @staticmethod
    def _filter_stream_backward_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return metrics

    def _run_rollout_and_loss(
        self,
        *,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        graph_mask: torch.Tensor,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        temperature: Optional[float],
        subtb_spec: Optional[SubTrajectorySpec],
        max_steps_override: Optional[int],
        record_actions: bool,
        compute_imitation: bool = False,
    ) -> tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RolloutLossRecord,
    ]:
        if not record_actions:
            record_actions = True
        flow_direction = "backward" if flow_modules.log_f is self.log_f_backward else "forward"
        actor = self._resolve_actor_for_flow(flow_modules)
        if self._cache_action_keys:
            _ = self._get_action_keys(graph_cache=graph_cache, actor=actor)
        actions_seq, log_pf, log_pf_steps, num_moves = actor.rollout(
            graph=graph_cache,
            temperature=temperature,
            max_steps_override=max_steps_override,
            record_actions=record_actions,
            guidance_fn=guidance_fn,
            mode=flow_direction,
            init_node_locals=graph_cache["start_node_locals"],
            init_ptr=graph_cache["start_ptr"],
        )
        if log_pf_start is not None:
            log_pf_start = log_pf_start.to(device=log_pf.device, dtype=log_pf.dtype).view(-1)
            if log_pf_start.numel() != log_pf.numel():
                raise ValueError("log_pf_start length mismatch with rollout log_pf.")
            log_pf = log_pf + log_pf_start
        if actions_seq is None:
            raise RuntimeError("rollout missing actions; record_actions=True required.")
        self._maybe_log_rollout_step_stats(
            actions_seq=actions_seq,
            inputs=inputs,
            graph_cache=graph_cache,
            flow_direction=flow_direction,
        )
        target_key = "node_is_target"
        stop_node_locals, reach_success = self._finalize_rollout_stop(
            actions_seq=actions_seq,
            inputs=inputs,
            graph_cache=graph_cache,
            target_key=target_key,
        )
        log_pb_steps = self._compute_backward_log_pb_steps(
            actions_seq=actions_seq,
            inputs=inputs,
            graph_cache=graph_cache,
            temperature=temperature,
            flow_modules=flow_modules,
        )
        log_pb_start = None
        if self._entry_selectors_enabled():
            globals_key = graph_cache.get("entry_end_globals")
            log_probs_key = graph_cache.get("entry_end_log_probs")
            if globals_key is None or log_probs_key is None:
                raise RuntimeError("entry_end log-prob cache missing for selector-enabled rollout.")
            log_pb_start = self._lookup_entry_log_prob_for_stop_nodes(
                candidate_globals=globals_key,
                candidate_log_probs=log_probs_key,
                stop_node_locals=stop_node_locals,
                node_ptr=inputs.node_ptr,
                default_log_prob=_ENTRY_LOG_PROB_MISS_DEFAULT,
            )
        tb_loss, record = self._compute_rollout_loss(
            actions_seq=actions_seq,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            stop_node_locals=stop_node_locals,
            reach_success=reach_success,
            num_moves=num_moves,
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            flow_modules=flow_modules,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            log_pb_start=log_pb_start,
            graph_mask=graph_mask,
            subtb_spec=subtb_spec,
            reward_override=None,
        )
        if compute_imitation and self._imitation_spec is not None:
            imitation_loss, imitation_active, imitation_per_graph = self._compute_online_imitation_loss(
                actions_seq=actions_seq,
                num_moves=num_moves,
                reach_success=reach_success,
                stop_node_locals=stop_node_locals,
                inputs=inputs,
                graph_cache=graph_cache,
                flow_direction=flow_direction,
            )
            tb_loss = tb_loss + imitation_loss
            record = replace(
                record,
                imitation_active=imitation_active.detach(),
                imitation_loss=imitation_per_graph.detach(),
            )
        return (
            actions_seq,
            log_pf,
            log_pf_steps,
            log_pb_steps,
            reach_success,
            num_moves,
            stop_node_locals,
            tb_loss,
            record,
        )

    def _compute_rollout_loss(
        self,
        *,
        actions_seq: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        stop_node_locals: torch.Tensor,
        reach_success: torch.Tensor,
        num_moves: torch.Tensor,
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        log_pb_start: Optional[torch.Tensor],
        graph_mask: torch.Tensor,
        subtb_spec: Optional[SubTrajectorySpec],
        reward_override: Optional[RewardOutput],
    ) -> tuple[torch.Tensor, RolloutLossRecord]:
        if reward_override is None:
            reward_out: RewardOutput = self.reward_fn(
                **self._build_reward_kwargs(stop_node_locals=stop_node_locals, inputs=inputs, graph_cache=graph_cache)
            )
        else:
            reward_out = RewardOutput(
                log_reward=reward_override.log_reward.to(device=inputs.node_ptr.device, dtype=torch.float32),
                success=reward_override.success.to(device=inputs.node_ptr.device, dtype=torch.bool),
            )
        log_reward_for_loss = reward_out.log_reward
        invalid_stop = stop_node_locals < _ZERO
        if bool(invalid_stop.any().detach().tolist()):
            active_invalid = invalid_stop & graph_mask
            if bool(active_invalid.any().detach().tolist()):
                raise ValueError("stop_node_locals contains invalid entries for active graphs.")
        log_target = log_reward_for_loss
        num_moves = num_moves.to(dtype=torch.long)
        if subtb_spec is None:
            raise RuntimeError("SubTB is required for trajectory balance; configure training_cfg.subtb.")
        sum_pf, sum_pb, residuals, log_f_start_sub, log_target_sub, valid, _ = self._compute_subtrajectory_values(
            actions_seq=actions_seq,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            log_f_module=flow_modules.log_f,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            log_pb_start=log_pb_start,
            log_target=log_target,
            num_moves=num_moves,
            spec=subtb_spec,
        )
        (
            sum_log_pf,
            sum_log_pb,
            residual,
            log_f_start_used,
            log_target_used,
            loss_per_graph,
        ) = self._reduce_subtrajectory_stats(
            sum_pf=sum_pf,
            sum_pb=sum_pb,
            residual=residuals,
            log_f_start=log_f_start_sub,
            log_target=log_target_sub,
            valid=valid,
        )
        log_pb_steps = self._require_log_pb_steps(log_pb_steps)
        tb_loss = self._reduce_graph_loss(loss_per_graph, graph_mask)
        self._raise_if_non_finite_loss(
            tb_loss=tb_loss,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_f_start=log_f_start_used,
            log_target=log_target_used,
            num_moves=num_moves,
            dummy_mask=inputs.dummy_mask,
            sum_log_pf=sum_log_pf,
            sum_log_pb=sum_log_pb,
            residual=residual,
        )
        record = self._build_rollout_loss_record(
            reward_out=reward_out,
            log_f_start=log_f_start_used,
            log_reward=log_reward_for_loss,
            log_target=log_target_used,
            sum_log_pf=sum_log_pf,
            sum_log_pb=sum_log_pb,
            residual=residual,
            reach_success=reach_success,
            num_moves=num_moves,
        )
        return tb_loss, record

    def _build_rollout_loss_record(
        self,
        *,
        reward_out: RewardOutput,
        log_f_start: torch.Tensor,
        log_reward: torch.Tensor,
        log_target: torch.Tensor,
        sum_log_pf: torch.Tensor,
        sum_log_pb: torch.Tensor,
        residual: torch.Tensor,
        reach_success: torch.Tensor,
        num_moves: torch.Tensor,
        imitation_active: Optional[torch.Tensor] = None,
        imitation_loss: Optional[torch.Tensor] = None,
    ) -> RolloutLossRecord:
        reward_detached = RewardOutput(
            log_reward=reward_out.log_reward.detach(),
            success=reward_out.success.detach(),
        )
        if imitation_active is None:
            imitation_active = torch.zeros_like(reach_success.detach(), dtype=torch.float32)
        else:
            imitation_active = imitation_active.detach()
        if imitation_loss is None:
            imitation_loss = torch.zeros_like(log_reward.detach(), dtype=torch.float32)
        else:
            imitation_loss = imitation_loss.detach()
        return RolloutLossRecord(
            reward_out=reward_detached,
            log_f_start=log_f_start.detach(),
            log_reward=log_reward.detach(),
            log_target=log_target.detach(),
            sum_log_pf=sum_log_pf.detach(),
            sum_log_pb=sum_log_pb.detach(),
            residual=residual.detach(),
            reach_success=reach_success.detach(),
            num_moves=num_moves.detach(),
            imitation_active=imitation_active,
            imitation_loss=imitation_loss,
        )

    def _build_metrics_from_records(
        self,
        records: list[RolloutLossRecord],
        *,
        graph_mask: torch.Tensor,
    ) -> list[Dict[str, torch.Tensor]]:
        if not records:
            return []
        metrics_list: list[Dict[str, torch.Tensor]] = []
        for record in records:
            metrics_list.append(
                self._build_rollout_metrics_from_record(
                    record,
                    graph_mask=graph_mask,
                )
            )
        return metrics_list

    def _build_rollout_metrics_from_record(
        self,
        record: RolloutLossRecord,
        *,
        graph_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        metrics = self._build_flow_metrics(
            reach_success=record.reach_success,
            num_moves=record.num_moves,
            reward_out=record.reward_out,
            log_reward=record.log_reward,
            log_f_start=record.log_f_start,
            log_f_target=record.log_target,
        )
        metrics["imitation/active"] = record.imitation_active
        metrics["imitation/loss"] = record.imitation_loss
        metrics.update(
            self._build_tb_stats(
                residual=record.residual,
                graph_mask=graph_mask,
            )
        )
        return metrics

    def _raise_if_non_finite_loss(
        self,
        *,
        tb_loss: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_f_start: torch.Tensor,
        log_target: torch.Tensor,
        num_moves: torch.Tensor,
        dummy_mask: torch.Tensor,
        sum_log_pf: Optional[torch.Tensor] = None,
        sum_log_pb: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> None:
        if torch.isfinite(tb_loss).all():
            return
        if sum_log_pf is None or sum_log_pb is None or residual is None:
            num_steps = int(log_pf_steps.size(1))
            step_mask = self._build_step_mask_incl_stop(num_moves, num_steps).to(dtype=log_pf_steps.dtype)
            sum_log_pf = (log_pf_steps * step_mask).sum(dim=1)
            sum_log_pb = (log_pb_steps * step_mask).sum(dim=1)
            residual = log_f_start + sum_log_pf - sum_log_pb - log_target
        message = self._format_non_finite_report(
            tb_loss=tb_loss,
            log_f_start=log_f_start,
            log_target=log_target,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            sum_log_pf=sum_log_pf,
            sum_log_pb=sum_log_pb,
            residual=residual,
        )
        dummy_mask = dummy_mask.to(dtype=torch.bool)
        dummy_total = int(dummy_mask.numel())
        dummy_count = int(dummy_mask.sum().detach().tolist())
        dummy_ratio = float(dummy_count) / float(dummy_total) if dummy_total > _ZERO else float(_ZERO)
        extra = [
            f"dummy_mask: count={dummy_count}, total={dummy_total}, ratio={dummy_ratio}",
            self._summarize_tensor_stats("log_f_start_stats", log_f_start),
            self._summarize_tensor_stats("log_target_stats", log_target),
            self._summarize_tensor_stats("sum_log_pf_stats", sum_log_pf),
            self._summarize_tensor_stats("sum_log_pb_stats", sum_log_pb),
            self._summarize_tensor_stats("residual_stats", residual),
            self._summarize_tensor_stats("num_moves_stats", num_moves),
        ]
        message = "\n".join([message, *extra])
        raise RuntimeError(message)

    def _format_non_finite_report(
        self,
        *,
        tb_loss: torch.Tensor,
        log_f_start: torch.Tensor,
        log_target: torch.Tensor,
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
        detail = self._summarize_tensor("log_f_start", log_f_start)
        if detail:
            lines.append(detail)
        detail = self._summarize_tensor("log_target", log_target)
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
        if bool(finite.all().detach().tolist()):
            return None
        non_finite = ~finite
        num_non_finite = int(non_finite.sum().detach().tolist())
        num_nan = int(torch.isnan(tensor).sum().detach().tolist())
        num_inf = int(torch.isinf(tensor).sum().detach().tolist())
        finite_vals = tensor[finite]
        if finite_vals.numel() > _ZERO:
            finite_min = float(finite_vals.min().detach().tolist())
            finite_max = float(finite_vals.max().detach().tolist())
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
        non_finite = int((~finite).sum().detach().tolist())
        finite_vals = tensor[finite]
        if finite_vals.numel() == 0:
            return f"{name}: non_finite={non_finite} (all)"
        calc = finite_vals.to(dtype=torch.float32)
        min_val = float(calc.min().detach().tolist())
        max_val = float(calc.max().detach().tolist())
        mean_val = float(calc.mean().detach().tolist())
        std_val = float(calc.std(unbiased=False).detach().tolist())
        abs_max = float(calc.abs().max().detach().tolist())
        q_tensor = torch.quantile(
            calc,
            torch.tensor((0.0, 0.5, 0.9, 0.99, 1.0), device=calc.device, dtype=calc.dtype),
        )
        q_vals = [float(q.detach().tolist()) for q in q_tensor]
        q_parts = " ".join(f"q{idx}={val}" for idx, val in enumerate(q_vals))
        return (
            f"{name}: non_finite={non_finite}, min={min_val}, max={max_val}, "
            f"mean={mean_val}, std={std_val}, abs_max={abs_max}, {q_parts}"
        )

    @staticmethod
    def _build_move_mask(num_moves: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_moves.dim() != 1:
            raise ValueError("num_moves must be [B] for move mask.")
        step_ids = torch.arange(num_steps, device=num_moves.device, dtype=num_moves.dtype).view(1, -1)
        return step_ids < num_moves.view(-1, 1)

    def _build_step_mask_incl_stop(num_moves: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_moves.dim() != 1:
            raise ValueError("num_moves must be [B] for step mask.")
        step_ids = torch.arange(num_steps, device=num_moves.device, dtype=num_moves.dtype).view(1, -1)
        return step_ids <= num_moves.view(-1, 1)

    @staticmethod
    def _mask_values(values: torch.Tensor, graph_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if graph_mask is None:
            return values
        mask = graph_mask.to(dtype=torch.bool).view(-1)
        return values[mask]

    def _mean_std_or_zero(
        self,
        values: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.numel() == _ZERO:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, zero
        calc = values.to(dtype=torch.float32)
        mean = calc.mean()
        std = calc.std(unbiased=False)
        return mean.to(device=device, dtype=dtype), std.to(device=device, dtype=dtype)

    def _build_tb_stats(
        self,
        *,
        residual: torch.Tensor,
        graph_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = residual.device
        dtype = residual.dtype
        residual_vals = self._mask_values(residual, graph_mask)
        residual_mean, residual_std = self._mean_std_or_zero(residual_vals, device=device, dtype=dtype)
        return {
            "subtb/residual_mean": residual_mean,
            "subtb/residual_std": residual_std,
        }

    @staticmethod
    def _reduce_graph_loss(loss_per_graph: torch.Tensor, graph_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if graph_mask is None:
            return loss_per_graph.mean()
        weights = graph_mask.to(dtype=loss_per_graph.dtype)
        denom = weights.sum().clamp(min=float(_ONE))
        return (loss_per_graph * weights).sum() / denom

    @staticmethod
    def _compute_z_align_loss(
        *,
        log_f_forward: torch.Tensor,
        log_f_backward: torch.Tensor,
        log_reward: torch.Tensor,
        graph_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        diff = log_f_forward - log_f_backward - log_reward
        if graph_mask is not None:
            mask = graph_mask.to(dtype=torch.bool)
            diff = diff[mask]
        if diff.numel() == _ZERO:
            return torch.zeros((), device=log_f_forward.device, dtype=log_f_forward.dtype)
        return diff.pow(_TWO).mean()

    @staticmethod
    def _build_flow_metrics(
        *,
        reach_success: torch.Tensor,
        num_moves: torch.Tensor,
        reward_out: RewardOutput,
        log_reward: torch.Tensor,
        log_f_start: torch.Tensor,
        log_f_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return gfn_metrics.build_flow_metrics(
            reach_success=reach_success,
            num_moves=num_moves,
            reward_out=reward_out,
            log_reward=log_reward,
            log_f_start=log_f_start,
            log_f_target=log_f_target,
        )

    def _collect_rollout_outputs(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        guidance_beta: float,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
        rollout_cfg: GFlowNetRolloutConfig,
        collect_terminal_hits: bool,
        subtb_spec: Optional[SubTrajectorySpec],
        max_steps_override: Optional[int],
        rollout_hook: Optional[RolloutHook] = None,
    ) -> tuple[
        list[torch.Tensor],
        list[Dict[str, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        loss_list: list[torch.Tensor] = []
        metric_records: list[RolloutLossRecord] = []
        rollout_stop_nodes: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        collect_eval = not rollout_cfg.is_training
        record_actions = True
        guidance_fn = None
        if self._should_apply_h_guidance(flow_modules=flow_modules):
            guidance_fn = self._build_h_guidance_fn(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                beta=guidance_beta,
            )
        for _ in range(num_rollouts):
            (
                actions_seq,
                log_pf,
                log_pf_steps,
                log_pb_steps,
                reach_success,
                num_moves,
                stop_node_locals,
                tb_loss,
                record,
            ) = self._run_rollout_and_loss(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                flow_modules=flow_modules,
                log_f_start=log_f_start,
                log_pf_start=log_pf_start,
                graph_mask=graph_mask,
                guidance_fn=guidance_fn,
                temperature=temperature,
                subtb_spec=subtb_spec,
                max_steps_override=max_steps_override,
                record_actions=record_actions,
                compute_imitation=rollout_cfg.is_training and self._imitation_spec is not None,
            )
            if rollout_hook is not None:
                rollout_hook(
                    actions_seq,
                    log_pf,
                    log_pf_steps,
                    log_pb_steps,
                    reach_success,
                    num_moves,
                    stop_node_locals,
                    inputs,
                    graph_cache,
                )
            self._append_rollout_outputs(
                actions_seq,
                stop_node_locals,
                tb_loss,
                record,
                collect_terminal_hits=collect_terminal_hits,
                collect_eval=collect_eval,
                rollout_stop_nodes=rollout_stop_nodes,
                rollout_actions=rollout_actions,
                loss_list=loss_list,
                metric_records=metric_records,
            )
        metrics_list = self._build_metrics_from_records(
            metric_records,
            graph_mask=graph_mask,
        )
        return (
            loss_list,
            metrics_list,
            rollout_stop_nodes,
            rollout_actions,
        )

    @staticmethod
    def _iter_rollout_chunk_sizes(num_rollouts: int, rollout_chunk_size: int) -> Sequence[int]:
        if rollout_chunk_size <= 0:
            raise ValueError(f"rollout_chunk_size must be > 0, got {rollout_chunk_size}.")
        chunk_size = min(int(rollout_chunk_size), int(num_rollouts))
        return [min(chunk_size, num_rollouts - chunk_start) for chunk_start in range(_ZERO, num_rollouts, chunk_size)]

    @staticmethod
    def _init_rollout_chunk_state() -> RolloutChunkState:
        return RolloutChunkState(
            loss_list=[],
            metrics_list=[],
            rollout_stop_nodes=[],
            rollout_actions=[],
        )

    def _run_chunked_rollouts(
        self,
        *,
        num_rollouts: int,
        rollout_chunk_size: int,
        chunk_sizes: Optional[Sequence[int]],
        rollout_cfg: GFlowNetRolloutConfig,
        temperature: Optional[float],
        collect_terminal_hits: bool,
        collect_eval: bool,
        chunk_inputs_fn: Callable[[], RolloutChunkInputs],
        flow_modules: FlowModules,
        guidance_beta: float,
        store_loss_list: bool,
        on_chunk_loss: Optional[Callable[[list[torch.Tensor]], torch.Tensor]],
        subtb_spec: Optional[SubTrajectorySpec],
        max_steps_override: Optional[int],
        rollout_hook: Optional[RolloutHook] = None,
    ) -> tuple[RolloutChunkState, int, Optional[torch.Tensor]]:
        state = self._init_rollout_chunk_state()
        loss_total: Optional[torch.Tensor] = None
        num_graphs = _ZERO
        sizes = (
            list(chunk_sizes)
            if chunk_sizes is not None
            else self._iter_rollout_chunk_sizes(num_rollouts, rollout_chunk_size)
        )
        for current in sizes:
            chunk_inputs = chunk_inputs_fn()
            num_graphs = chunk_inputs.num_graphs
            (
                chunk_loss_list,
                chunk_metrics_list,
                chunk_stop_nodes,
                chunk_actions,
            ) = self._collect_rollout_outputs(
                inputs=chunk_inputs.inputs,
                num_rollouts=current,
                graph_cache=chunk_inputs.graph_cache,
                flow_features=chunk_inputs.flow_features,
                flow_modules=flow_modules,
                log_f_start=chunk_inputs.log_f_start,
                log_pf_start=chunk_inputs.log_pf_start,
                guidance_beta=guidance_beta,
                graph_mask=chunk_inputs.graph_mask,
                temperature=temperature,
                rollout_cfg=rollout_cfg,
                collect_terminal_hits=collect_terminal_hits,
                subtb_spec=subtb_spec,
                max_steps_override=max_steps_override,
                rollout_hook=rollout_hook,
            )
            if store_loss_list:
                state.loss_list.extend(chunk_loss_list)
            state.metrics_list.extend(chunk_metrics_list)
            if collect_terminal_hits:
                state.rollout_stop_nodes.extend(chunk_stop_nodes)
            if collect_eval:
                state.rollout_actions.extend(chunk_actions)
            if on_chunk_loss is not None:
                chunk_loss = on_chunk_loss(chunk_loss_list)
                chunk_detached = chunk_loss.detach()
                loss_total = chunk_detached if loss_total is None else loss_total + chunk_detached
        return state, num_graphs, loss_total

    def _compute_loop_rollout_loss(
        self,
        *,
        inputs: RolloutInputs,
        num_rollouts: int,
        num_graphs: int,
        graph_cache: Dict[str, torch.Tensor],
        flow_features: torch.Tensor,
        flow_modules: FlowModules,
        log_f_start: torch.Tensor,
        log_pf_start: Optional[torch.Tensor],
        guidance_beta: float,
        graph_mask: torch.Tensor,
        temperature: Optional[float],
        rollout_cfg: GFlowNetRolloutConfig,
        rollout_chunk_size: int,
        subtb_spec: Optional[SubTrajectorySpec],
        max_steps_override: Optional[int],
        rollout_hook: Optional[RolloutHook] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        collect_eval = not rollout_cfg.is_training
        collect_terminal_hits = self._should_collect_terminal_hits(rollout_cfg)
        chunk_inputs = RolloutChunkInputs(
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            graph_mask=graph_mask,
            num_graphs=num_graphs,
            node_ptr=inputs.node_ptr,
            node_is_target=graph_cache["node_is_target"],
        )

        def chunk_inputs_fn() -> RolloutChunkInputs:
            return chunk_inputs

        state, _, _ = self._run_chunked_rollouts(
            num_rollouts=num_rollouts,
            rollout_chunk_size=rollout_chunk_size,
            chunk_sizes=None,
            rollout_cfg=rollout_cfg,
            temperature=temperature,
            collect_terminal_hits=collect_terminal_hits,
            collect_eval=collect_eval,
            chunk_inputs_fn=chunk_inputs_fn,
            flow_modules=flow_modules,
            guidance_beta=guidance_beta,
            store_loss_list=True,
            on_chunk_loss=None,
            subtb_spec=subtb_spec,
            max_steps_override=max_steps_override,
            rollout_hook=rollout_hook,
        )
        loss, metrics = self._finalize_loop_rollout_metrics(
            loss_list=state.loss_list,
            metrics_list=state.metrics_list,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        metrics = self._postprocess_rollout_metrics(
            metrics=metrics,
            metrics_list=state.metrics_list,
            rollout_stop_nodes=state.rollout_stop_nodes,
            rollout_actions=state.rollout_actions,
            inputs=inputs,
            graph_cache=graph_cache,
            rollout_cfg=rollout_cfg,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        return loss, metrics

    def _append_common_rollout_metrics(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        metrics_list: list[Dict[str, torch.Tensor]],
        rollout_stop_nodes: list[torch.Tensor],
        node_ptr: torch.Tensor,
        node_is_target: torch.Tensor,
        rollout_cfg: GFlowNetRolloutConfig,
        num_rollouts: int,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        stacked = self._stack_rollout_metrics(metrics_list)
        if (not rollout_cfg.is_training) and "log_reward" in stacked and "pass@1" in stacked:
            metrics["reward_gap"] = gfn_metrics.compute_reward_gap(
                log_reward=stacked["log_reward"],
                pass_hits=stacked["pass@1"],
                num_rollouts=num_rollouts,
                num_graphs=num_graphs,
            )
        if rollout_stop_nodes:
            k_values = rollout_cfg.eval_rollout_prefixes
            if not k_values:
                k_values = [num_rollouts]
            terminal_hits = torch.stack(
                [
                    gfn_metrics.compute_terminal_hits(
                        stop_node_locals=stop_nodes,
                        node_ptr=node_ptr,
                        node_is_target=node_is_target,
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
        return metrics

    def _postprocess_rollout_metrics(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        metrics_list: list[Dict[str, torch.Tensor]],
        rollout_stop_nodes: list[torch.Tensor],
        rollout_actions: list[torch.Tensor],
        inputs: RolloutInputs,
        graph_cache: Dict[str, torch.Tensor],
        rollout_cfg: GFlowNetRolloutConfig,
        num_rollouts: int,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        metrics = self._append_common_rollout_metrics(
            metrics=metrics,
            metrics_list=metrics_list,
            rollout_stop_nodes=rollout_stop_nodes,
            node_ptr=inputs.node_ptr,
            node_is_target=graph_cache["node_is_target"],
            rollout_cfg=rollout_cfg,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        if not rollout_cfg.is_training:
            composite = gfn_metrics.compute_composite_score(
                metrics=metrics,
                k_values=rollout_cfg.eval_rollout_prefixes,
                composite_cfg=self._composite_score_cfg,
            )
            if composite:
                metrics.update(composite)
        return metrics

    @staticmethod
    def _should_collect_terminal_hits(rollout_cfg: GFlowNetRolloutConfig) -> bool:
        return (not rollout_cfg.is_training) and bool(rollout_cfg.eval_rollout_prefixes)

    def _compute_loop_rollout_loss_streaming(
        self,
        *,
        batch: Any,
        device: torch.device,
        num_rollouts: int,
        rollout_cfg: GFlowNetRolloutConfig,
        rollout_chunk_size: int,
        temperature: Optional[float],
        backward_fn: Callable[[torch.Tensor], None],
        subtb_spec: Optional[SubTrajectorySpec],
        flow_modules: FlowModules,
        guidance_beta: float,
        max_steps_override: Optional[int],
        log_f_start_node_locals: Optional[torch.Tensor] = None,
        log_f_start_ptr: Optional[torch.Tensor] = None,
        rollout_hook: Optional[RolloutHook] = None,
        loss_scale: float = 1.0,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        collect_terminal_hits = self._should_collect_terminal_hits(rollout_cfg)
        (
            loss_total,
            metrics_list,
            rollout_stop_nodes,
            node_ptr,
            node_is_target,
            num_graphs,
        ) = self._run_streaming_chunks(
            batch=batch,
            device=device,
            num_rollouts=num_rollouts,
            rollout_chunk_size=rollout_chunk_size,
            rollout_cfg=rollout_cfg,
            temperature=temperature,
            backward_fn=backward_fn,
            collect_terminal_hits=collect_terminal_hits,
            subtb_spec=subtb_spec,
            flow_modules=flow_modules,
            guidance_beta=guidance_beta,
            max_steps_override=max_steps_override,
            log_f_start_node_locals=log_f_start_node_locals,
            log_f_start_ptr=log_f_start_ptr,
            rollout_hook=rollout_hook,
            loss_scale=loss_scale,
        )
        if loss_total is None or node_ptr is None:
            raise RuntimeError("Streaming rollout loss did not produce any rollouts.")
        metrics = self._finalize_streaming_metrics(
            metrics_list=metrics_list,
            rollout_stop_nodes=rollout_stop_nodes,
            node_ptr=node_ptr,
            node_is_target=node_is_target,
            rollout_cfg=rollout_cfg,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
        )
        loss = loss_total
        replay_scaled = None
        replay_loss_value = None
        graph_mask_frac = None
        if (
            self._replay_spec is not None
            and self._replay_buffer is not None
            and flow_modules.log_f is self.log_f
            and sample_ids is not None
        ):
            inputs, graph_cache, flow_features, graph_mask = self._prepare_replay_inputs(
                batch=batch,
                device=device,
                is_training=rollout_cfg.is_training,
            )
            replay_loss, replay_metrics = self._compute_replay_loss(
                inputs=inputs,
                graph_cache=graph_cache,
                flow_features=flow_features,
                flow_modules=flow_modules,
                temperature=temperature,
                subtb_spec=subtb_spec,
                sample_ids=sample_ids,
            )
            if replay_loss is not None:
                mix_ratio = float(self._replay_spec.mix_ratio)
                replay_scaled = replay_loss * mix_ratio
                replay_loss_value = replay_loss
                backward_fn(replay_scaled)
                loss = loss + replay_scaled.detach()
                replay_metrics["replay/mix_ratio"] = torch.as_tensor(
                    mix_ratio,
                    device=loss.device,
                    dtype=loss.dtype,
                )
                metrics.update(replay_metrics)
                graph_mask_frac = graph_mask.to(dtype=torch.float32).mean()
                metrics["replay/graph_mask_frac"] = graph_mask_frac
        if not bool(torch.isfinite(loss).all().detach().item()):
            flow_direction = "backward" if flow_modules.log_f is self.log_f_backward else "forward"
            sample_ids = self._try_extract_sample_ids(batch, num_graphs)
            preview_ids = None
            if sample_ids is not None:
                preview_ids = sample_ids[:_NONFINITE_ROLLOUT_PREVIEW]
            log_event(
                logger,
                _NONFINITE_ROLLOUT_EVENT,
                stream=flow_direction,
                num_graphs=int(num_graphs),
                num_rollouts=int(num_rollouts),
                loss=float(loss.detach().to(device="cpu").item()),
                loss_total=float(loss_total.detach().to(device="cpu").item()),
                replay_loss=None if replay_loss_value is None else float(replay_loss_value.detach().to(device="cpu").item()),
                replay_scaled=None if replay_scaled is None else float(replay_scaled.detach().to(device="cpu").item()),
                replay_graph_mask_frac=None
                if graph_mask_frac is None
                else float(graph_mask_frac.detach().to(device="cpu").item()),
                sample_ids=preview_ids,
            )
            raise ValueError("Non-finite streaming loss detected; see gfn_nonfinite_rollout logs for details.")
        return loss, metrics

    def _run_streaming_chunks(
        self,
        *,
        batch: Any,
        device: torch.device,
        num_rollouts: int,
        rollout_chunk_size: int,
        rollout_cfg: GFlowNetRolloutConfig,
        temperature: Optional[float],
        backward_fn: Callable[[torch.Tensor], None],
        collect_terminal_hits: bool,
        subtb_spec: Optional[SubTrajectorySpec],
        flow_modules: FlowModules,
        guidance_beta: float,
        max_steps_override: Optional[int],
        log_f_start_node_locals: Optional[torch.Tensor] = None,
        log_f_start_ptr: Optional[torch.Tensor] = None,
        rollout_hook: Optional[RolloutHook] = None,
        loss_scale: float = 1.0,
    ) -> tuple[
        Optional[torch.Tensor],
        list[Dict[str, torch.Tensor]],
        list[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        int,
    ]:
        collect_eval = not rollout_cfg.is_training
        chunk_sizes = self._iter_rollout_chunk_sizes(num_rollouts, rollout_chunk_size)
        supports_retain_graph = self._supports_retain_graph(backward_fn)
        cache_inputs = bool(len(chunk_sizes) <= _ONE)
        context, chunk_inputs_fn = self._init_streaming_chunk_runner(
            batch=batch,
            device=device,
            cache_inputs=cache_inputs,
            flow_modules=flow_modules,
            log_f_start_node_locals=log_f_start_node_locals,
            log_f_start_ptr=log_f_start_ptr,
            is_training=rollout_cfg.is_training,
        )
        remaining_chunks = len(chunk_sizes)

        def on_chunk_loss_wrapper(chunk_loss_list: list[torch.Tensor]) -> torch.Tensor:
            nonlocal remaining_chunks
            retain_graph = cache_inputs and remaining_chunks > _ONE
            remaining_chunks -= _ONE
            return self._run_streaming_chunk_backward(
                chunk_loss_list,
                num_rollouts=num_rollouts,
                backward_fn=backward_fn,
                retain_graph=retain_graph,
                supports_retain_graph=supports_retain_graph,
                loss_scale=loss_scale,
            )

        chunk_loss_fn = on_chunk_loss_wrapper
        state, num_graphs, loss_total = self._run_chunked_rollouts(
            num_rollouts=num_rollouts,
            rollout_chunk_size=rollout_chunk_size,
            chunk_sizes=chunk_sizes,
            rollout_cfg=rollout_cfg,
            temperature=temperature,
            collect_terminal_hits=collect_terminal_hits,
            collect_eval=collect_eval,
            chunk_inputs_fn=chunk_inputs_fn,
            flow_modules=flow_modules,
            guidance_beta=guidance_beta,
            store_loss_list=False,
            on_chunk_loss=chunk_loss_fn,
            subtb_spec=subtb_spec,
            max_steps_override=max_steps_override,
            rollout_hook=rollout_hook,
        )
        return (
            loss_total,
            state.metrics_list,
            state.rollout_stop_nodes,
            context.node_ptr,
            context.node_is_target,
            num_graphs,
        )

    def _prepare_replay_inputs(
        self,
        *,
        batch: Optional[Any],
        device: torch.device,
        is_training: bool,
        base_inputs: Optional[RolloutInputs] = None,
        base_graph_cache: Optional[Dict[str, torch.Tensor]] = None,
        base_flow_features: Optional[torch.Tensor] = None,
    ) -> tuple[
        RolloutInputs,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        self._set_edge_batch_requirement(is_training=is_training)
        if base_inputs is None:
            if batch is None:
                raise ValueError("batch required when base_inputs is not provided.")
            inputs = self.batch_processor.prepare_full_rollout_inputs(
                batch,
                device,
                flow_direction="forward",
            )
            graph_cache = self._build_graph_cache_with_masks(
                inputs=inputs,
                device=device,
                flow_direction="forward",
                batch=batch,
            )
            flow_features = self._compute_flow_features(inputs)
        else:
            inputs = base_inputs
            if base_graph_cache is None:
                graph_cache = self._build_graph_cache_with_masks(
                    inputs=inputs,
                    device=device,
                    flow_direction="forward",
                    batch=None,
                )
            else:
                graph_cache = dict(base_graph_cache)
            if base_flow_features is None:
                flow_features = self._compute_flow_features(inputs)
            else:
                flow_features = base_flow_features
        graph_cache["flow_features"] = flow_features
        graph_mask = ~inputs.dummy_mask
        return inputs, graph_cache, flow_features, graph_mask

    def _prepare_streaming_inputs(
        self,
        *,
        batch: Any,
        device: torch.device,
        node_ptr: Optional[torch.Tensor],
        node_is_target: Optional[torch.Tensor],
        log_f_module: torch.nn.Module,
        log_f_start_node_locals: Optional[torch.Tensor],
        log_f_start_ptr: Optional[torch.Tensor],
        is_training: bool,
    ) -> tuple[
        RolloutInputs,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        self._set_edge_batch_requirement(is_training=is_training)
        flow_direction = "backward" if log_f_module is self.log_f_backward else "forward"
        inputs = self.batch_processor.prepare_full_rollout_inputs(
            batch,
            device,
            flow_direction=flow_direction,
        )
        query_tokens = inputs.question_tokens
        if log_f_module is self.log_f_backward:
            target_context = self._compute_target_context_tokens(
                inputs,
                node_locals=inputs.start_node_locals,
                node_ptr=inputs.start_ptr,
            )
            inputs = replace(inputs, question_tokens=target_context)
        num_graphs = int(inputs.node_ptr.numel() - 1)
        if is_training and (not self.input_validator.validate_rollout_batch_enabled):
            sample_ids = self._try_extract_sample_ids(batch, num_graphs)
            self._validate_packed_node_locals(
                node_locals=inputs.q_local_indices,
                ptr=inputs.q_ptr,
                node_ptr=inputs.node_ptr,
                name="q_local_indices",
                flow_direction=flow_direction,
                sample_ids=sample_ids,
            )
            self._validate_packed_node_locals(
                node_locals=inputs.a_local_indices,
                ptr=inputs.a_ptr,
                node_ptr=inputs.node_ptr,
                name="a_local_indices",
                flow_direction=flow_direction,
                sample_ids=sample_ids,
            )
            self._validate_packed_node_locals(
                node_locals=inputs.start_node_locals,
                ptr=inputs.start_ptr,
                node_ptr=inputs.node_ptr,
                name="start_node_locals",
                flow_direction=flow_direction,
                sample_ids=sample_ids,
            )
            self._validate_packed_node_locals(
                node_locals=inputs.target_node_locals,
                ptr=inputs.target_ptr,
                node_ptr=inputs.node_ptr,
                name="target_node_locals",
                flow_direction=flow_direction,
                sample_ids=sample_ids,
            )
        node_ptr = inputs.node_ptr if node_ptr is None else node_ptr
        graph_cache = self._build_graph_cache_with_masks(
            inputs=inputs,
            device=device,
            flow_direction=flow_direction,
            batch=batch,
        )
        graph_cache["query_tokens"] = query_tokens
        if node_is_target is None:
            node_is_target = graph_cache["node_is_target"]
        start_nodes, start_ptr, log_pf_start = self._select_start_nodes_for_flow(
            log_f_module=log_f_module,
            query_tokens=query_tokens,
            candidate_locals=inputs.start_node_locals,
            candidate_ptr=inputs.start_ptr,
            node_ptr=inputs.node_ptr,
            node_tokens=inputs.node_tokens,
            stream=flow_direction,
        )
        inputs, graph_cache = self._override_start_nodes(
            inputs=inputs,
            graph_cache=graph_cache,
            start_nodes=start_nodes,
            start_ptr=start_ptr,
        )
        self._maybe_attach_entry_end_cache(
            log_f_module=log_f_module,
            query_tokens=query_tokens,
            inputs=inputs,
            graph_cache=graph_cache,
            stream=flow_direction,
        )
        flow_features = self._compute_flow_features(inputs)
        log_f_start = self._compute_log_f_start(
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            log_f_module=log_f_module,
            start_node_locals=log_f_start_node_locals,
            start_ptr=log_f_start_ptr,
        )
        graph_mask = ~inputs.dummy_mask
        return (
            inputs,
            graph_cache,
            flow_features,
            log_f_start,
            graph_mask,
            num_graphs,
            node_ptr,
            node_is_target,
            log_pf_start,
        )

    def _build_streaming_chunk_inputs(
        self,
        *,
        batch: Any,
        device: torch.device,
        context: StreamingChunkContext,
        flow_modules: FlowModules,
        log_f_start_node_locals: Optional[torch.Tensor],
        log_f_start_ptr: Optional[torch.Tensor],
        is_training: bool,
    ) -> RolloutChunkInputs:
        if (not context.cache_inputs) or context.inputs is None:
            (
                inputs,
                graph_cache,
                flow_features,
                log_f_start,
                graph_mask,
                num_graphs,
                node_ptr_local,
                node_is_target_local,
                log_pf_start,
            ) = self._prepare_streaming_inputs(
                batch=batch,
                device=device,
                node_ptr=context.node_ptr,
                node_is_target=context.node_is_target,
                log_f_module=flow_modules.log_f,
                log_f_start_node_locals=log_f_start_node_locals,
                log_f_start_ptr=log_f_start_ptr,
                is_training=is_training,
            )
            context.inputs = inputs
            context.graph_cache = graph_cache
            context.flow_features = flow_features
            context.log_f_start = log_f_start
            context.log_pf_start = log_pf_start
            context.graph_mask = graph_mask
            context.num_graphs = num_graphs
            if context.node_ptr is None:
                context.node_ptr = node_ptr_local
            if context.node_is_target is None:
                context.node_is_target = node_is_target_local
        inputs = context.inputs
        graph_cache = context.graph_cache
        flow_features = context.flow_features
        log_f_start = context.log_f_start
        log_pf_start = context.log_pf_start
        graph_mask = context.graph_mask
        num_graphs = context.num_graphs
        if (
            inputs is None
            or graph_cache is None
            or flow_features is None
            or log_f_start is None
            or graph_mask is None
        ):
            raise RuntimeError("Streaming chunk inputs missing cached tensors.")
        return RolloutChunkInputs(
            inputs=inputs,
            graph_cache=graph_cache,
            flow_features=flow_features,
            log_f_start=log_f_start,
            log_pf_start=log_pf_start,
            graph_mask=graph_mask,
            num_graphs=num_graphs,
            node_ptr=context.node_ptr if context.node_ptr is not None else node_ptr_local,
            node_is_target=context.node_is_target if context.node_is_target is not None else node_is_target_local,
        )

    @staticmethod
    def _supports_retain_graph(backward_fn: Callable[..., None]) -> bool:
        try:
            signature = inspect.signature(backward_fn)
        except (TypeError, ValueError):
            return False
        return "retain_graph" in signature.parameters

    @staticmethod
    def _run_streaming_chunk_backward(
        chunk_loss_list: list[torch.Tensor],
        *,
        num_rollouts: int,
        backward_fn: Callable[..., None],
        retain_graph: bool,
        supports_retain_graph: bool,
        loss_scale: float,
    ) -> torch.Tensor:
        chunk_loss = torch.stack(chunk_loss_list, dim=0).sum() / float(num_rollouts)
        if loss_scale != float(_ONE):
            chunk_loss = chunk_loss * float(loss_scale)
        if retain_graph and supports_retain_graph:
            backward_fn(chunk_loss, retain_graph=True)
        else:
            backward_fn(chunk_loss)
        return chunk_loss

    def _init_streaming_chunk_runner(
        self,
        *,
        batch: Any,
        device: torch.device,
        cache_inputs: bool,
        flow_modules: FlowModules,
        log_f_start_node_locals: Optional[torch.Tensor],
        log_f_start_ptr: Optional[torch.Tensor],
        is_training: bool,
    ) -> tuple[StreamingChunkContext, Callable[[], RolloutChunkInputs]]:
        context = StreamingChunkContext(
            node_ptr=None,
            node_is_target=None,
            inputs=None,
            graph_cache=None,
            flow_features=None,
            log_f_start=None,
            log_pf_start=None,
            graph_mask=None,
            num_graphs=_ZERO,
            cache_inputs=cache_inputs,
        )
        chunk_inputs_fn = functools.partial(
            self._build_streaming_chunk_inputs,
            batch=batch,
            device=device,
            context=context,
            flow_modules=flow_modules,
            log_f_start_node_locals=log_f_start_node_locals,
            log_f_start_ptr=log_f_start_ptr,
            is_training=is_training,
        )
        return context, chunk_inputs_fn

    def _finalize_streaming_metrics(
        self,
        *,
        metrics_list: list[Dict[str, torch.Tensor]],
        rollout_stop_nodes: list[torch.Tensor],
        node_ptr: torch.Tensor,
        node_is_target: torch.Tensor,
        rollout_cfg: GFlowNetRolloutConfig,
        num_rollouts: int,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        stacked = self._stack_rollout_metrics(metrics_list)
        metrics = self._reduce_rollout_metrics(
            stacked,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=False,
        )
        return self._append_common_rollout_metrics(
            metrics=metrics,
            metrics_list=metrics_list,
            rollout_stop_nodes=rollout_stop_nodes,
            node_ptr=node_ptr,
            node_is_target=node_is_target,
            rollout_cfg=rollout_cfg,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
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
    def _extract_sample_ids_for_replay(batch: Any, num_graphs: int) -> list[str]:
        raw_ids = getattr(batch, "sample_id", None)
        if raw_ids is None:
            raise ValueError("Batch missing sample_id required for replay.")
        if isinstance(raw_ids, (list, tuple)):
            sample_ids = [str(s) for s in raw_ids]
        elif torch.is_tensor(raw_ids):
            sample_ids = [str(s.detach().tolist()) for s in raw_ids.view(-1)]
        else:
            raise ValueError(f"batch.sample_id must be list/tuple/tensor, got {type(raw_ids)!r}.")
        if len(sample_ids) != num_graphs:
            raise ValueError(f"sample_id length {len(sample_ids)} != num_graphs {num_graphs}.")
        return sample_ids

    @staticmethod
    def _try_extract_sample_ids(batch: Any, num_graphs: int) -> Optional[list[str]]:
        raw_ids = getattr(batch, "sample_id", None)
        if raw_ids is None:
            return None
        if isinstance(raw_ids, (list, tuple)):
            sample_ids = [str(s) for s in raw_ids]
        elif torch.is_tensor(raw_ids):
            sample_ids = [str(s.detach().tolist()) for s in raw_ids.view(-1)]
        else:
            return None
        if len(sample_ids) != num_graphs:
            return None
        return sample_ids

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
        if int(ptr[0].detach().tolist()) != _ZERO:
            raise ValueError("answer_entity_ids_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().detach().tolist()):
            raise ValueError("answer_entity_ids_ptr must be non-decreasing.")
        if int(ptr[-1].detach().tolist()) != raw_ids.numel():
            raise ValueError(f"answer_entity_ids_ptr must end at {raw_ids.numel()}, got {int(ptr[-1].detach().tolist())}.")
        answer_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].detach().tolist())
            end = int(ptr[gid + 1].detach().tolist())
            answer_lists.append([int(x) for x in raw_ids[start:end].detach().to(device="cpu").tolist()])
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
        if int(ptr[0].detach().tolist()) != _ZERO:
            raise ValueError("q_local_indices_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().detach().tolist()):
            raise ValueError("q_local_indices_ptr must be non-decreasing.")
        if int(ptr[-1].detach().tolist()) != raw.numel():
            raise ValueError(f"q_local_indices_ptr must end at {raw.numel()}, got {int(ptr[-1].detach().tolist())}.")
        start_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].detach().tolist())
            end = int(ptr[gid + 1].detach().tolist())
            indices = raw[start:end]
            start_lists.append([int(x) for x in indices.detach().to(device="cpu").tolist()])
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
        if int(ptr[0].detach().tolist()) != _ZERO:
            raise ValueError("q_local_indices_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().detach().tolist()):
            raise ValueError("q_local_indices_ptr must be non-decreasing.")
        if int(ptr[-1].detach().tolist()) != raw.numel():
            raise ValueError(f"q_local_indices_ptr must end at {raw.numel()}, got {int(ptr[-1].detach().tolist())}.")
        start_lists: list[list[int]] = []
        for gid in range(num_graphs):
            start = int(ptr[gid].detach().tolist())
            end = int(ptr[gid + 1].detach().tolist())
            indices = raw[start:end]
            if indices.numel() == 0:
                start_lists.append([])
                continue
            ids = node_global_ids.index_select(0, indices.to(device=node_global_ids.device))
            start_lists.append([int(x) for x in ids.detach().to(device="cpu").tolist()])
        return start_lists

    @staticmethod
    def _build_rollout_edges(
        *,
        actions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        node_global_ids: torch.Tensor,
        edge_start: int,
        edge_end: int,
        node_offset: int,
        start_local: int,
    ) -> tuple[list[int], list[Dict[str, Any]]]:
        current_local = int(start_local)
        edge_ids: list[int] = []
        edges_meta: list[Dict[str, Any]] = []
        actions = actions.view(-1)
        first_edge = True
        for action in actions.tolist():
            if action < _ZERO:
                if action == STOP_RELATION and current_local >= _ZERO:
                    head_idx = node_offset + current_local
                    head_gid = int(node_global_ids[head_idx].detach().tolist())
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
                raise ValueError(f"rollout edge id {edge_id} out of range [{edge_start},{edge_end}).")
            rel_id = int(edge_relations[edge_id].detach().tolist())
            head_idx = int(edge_index[0, edge_id].detach().tolist())
            tail_idx = int(edge_index[1, edge_id].detach().tolist())
            head_local = head_idx - node_offset
            tail_local = tail_idx - node_offset
            if current_local < _ZERO:
                current_local = head_local
            elif head_local != current_local:
                if first_edge:
                    current_local = head_local
                else:
                    raise ValueError("rollout edge head does not match current active node.")
            first_edge = False
            edge_ids.append(edge_id - edge_start)
            head_gid = int(node_global_ids[head_idx].detach().tolist())
            tail_gid = int(node_global_ids[tail_idx].detach().tolist())
            edges_meta.append(
                {
                    "head_entity_id": head_gid,
                    "tail_entity_id": tail_gid,
                    "relation_id": rel_id,
                    "src_entity_id": head_gid,
                    "dst_entity_id": tail_gid,
                }
            )
            current_local = tail_local
        return edge_ids, edges_meta

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
        if int(edge_ptr[0].detach().tolist()) != 0:
            raise ValueError(f"edge_ptr must start at 0, got {int(edge_ptr[0].detach().tolist())}.")
        if bool((edge_ptr[1:] < edge_ptr[:-1]).any().detach().tolist()):
            raise ValueError("edge_ptr must be non-decreasing.")
        total_edges = int(edge_ptr[-1].detach().tolist())
        sample_ids, questions = self._extract_batch_meta(batch, num_graphs)
        answer_entity_ids = self._extract_answer_entity_ids(batch, num_graphs)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if node_global_ids is None:
            raise ValueError("Batch missing node_global_ids required for rollout artifacts.")
        if not torch.is_tensor(node_global_ids):
            node_global_ids = torch.as_tensor(node_global_ids, dtype=torch.long)
        node_global_ids = node_global_ids.view(-1).detach().to(device="cpu")
        start_entity_ids = self._extract_start_entity_ids(batch, num_graphs, node_global_ids)
        start_local_indices = self._extract_start_local_indices(batch, num_graphs)
        edge_index = edge_index.to(dtype=torch.long)
        edge_relations = edge_relations.to(dtype=torch.long)
        if not rollout_logs:
            raise ValueError("rollout_logs must be non-empty.")
        normalized_logs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ridx, log in enumerate(rollout_logs):
            actions_seq = log.get("actions_seq")
            log_pf = log.get("log_pf")
            reach_success = log.get("reach_success")
            stop_node_locals = log.get("stop_node_locals")
            if actions_seq is None or log_pf is None:
                raise ValueError(f"rollout_logs[{ridx}] missing required keys (actions_seq/log_pf).")
            if reach_success is None or stop_node_locals is None:
                raise ValueError(f"rollout_logs[{ridx}] missing reach_success/stop_node_locals.")
            if actions_seq.dim() != 2:
                raise ValueError(f"rollout_logs[{ridx}].actions_seq must be [B,T], got shape={tuple(actions_seq.shape)}.")
            if actions_seq.size(0) != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].actions_seq batch {actions_seq.size(0)} != num_graphs {num_graphs}.")
            if log_pf.dim() != 1 or log_pf.numel() != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].log_pf must be [B], got shape={tuple(log_pf.shape)}.")
            if reach_success.numel() != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].reach_success must be [B], got shape={tuple(reach_success.shape)}.")
            if stop_node_locals.numel() != num_graphs:
                raise ValueError(f"rollout_logs[{ridx}].stop_node_locals must be [B], got shape={tuple(stop_node_locals.shape)}.")
            normalized_logs.append(
                (
                    actions_seq.to(dtype=torch.long),
                    log_pf,
                    reach_success.to(dtype=torch.float32),
                    stop_node_locals.to(dtype=torch.long),
                )
            )
        records: list[Dict[str, Any]] = []
        for g in range(num_graphs):
            rollouts: list[Dict[str, Any]] = []
            edge_start = int(edge_ptr[g].detach().tolist())
            edge_end = int(edge_ptr[g + 1].detach().tolist())
            node_offset = int(node_ptr[g].detach().tolist())
            node_end = int(node_ptr[g + 1].detach().tolist())
            start_locals = start_local_indices[g] if g < len(start_local_indices) else []
            if start_locals:
                # Convert batch-global q indices into per-graph local indices.
                converted: list[int] = []
                for idx in start_locals:
                    local = int(idx) - node_offset
                    if local < _ZERO or (node_offset + local) >= node_end:
                        raise ValueError(
                            "q_local_indices out of range for rollout record reconstruction "
                            f"(graph={g}, global={idx}, offset={node_offset}, end={node_end})."
                        )
                    converted.append(local)
                start_locals = converted
            start_local = min(start_locals) if start_locals else -1
            for ridx, (actions_seq, log_pf, reach_success, stop_node_locals) in enumerate(normalized_logs):
                actions = actions_seq[g].to(dtype=torch.long)
                if bool((actions < STOP_RELATION).any().detach().tolist()):
                    bad = actions[actions < STOP_RELATION][:5].tolist()
                    raise ValueError(f"rollout_logs[{ridx}] actions_seq contains invalid negatives: {bad}.")
                edge_ids, edges_meta = self._build_rollout_edges(
                    actions=actions,
                    edge_index=edge_index,
                    edge_relations=edge_relations,
                    node_global_ids=node_global_ids,
                    edge_start=edge_start,
                    edge_end=edge_end,
                    node_offset=node_offset,
                    start_local=start_local,
                )
                if edge_ids and total_edges <= 0:
                    raise ValueError("actions_seq selects edges but edge_ptr indicates zero total edges.")
                rollouts.append(
                    {
                        "rollout_index": ridx,
                        "log_pf": float(log_pf[g].detach().tolist()),
                        "reach_success": bool(reach_success[g].detach().tolist()),
                        "stop_node_local": int(stop_node_locals[g].detach().tolist()),
                        "stop_node_entity_id": (
                            int(node_global_ids[int(node_ptr[g].detach().tolist()) + int(stop_node_locals[g].detach().tolist())].detach().tolist())
                            if int(stop_node_locals[g].detach().tolist()) >= _ZERO
                            else None
                        ),
                        "edge_ids": edge_ids,
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
