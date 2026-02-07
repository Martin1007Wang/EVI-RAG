from __future__ import annotations

_NEG_ONE = -1
_NEG_TWO = -2
_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_FOUR = 4
_SELF_RELATION_ID = -1
_INVALID_EDGE_ID = -1
_STOP_ACTION_ID = _NEG_TWO

_TERMINAL_NONE = 0
_TERMINAL_HIT = 1
_TERMINAL_DEAD_END = 2
_TERMINAL_MAX_STEPS = 3
_TERMINAL_INVALID_START = 4
_TERMINAL_EMIT = 5

_DEFAULT_GNN_LAYERS = 2
_DEFAULT_GNN_DROPOUT = 0.1
_DEFAULT_LOGIT_SCALE_INIT = 2.3
_DEFAULT_PRIOR_WEIGHT_INIT = 0.0
_DEFAULT_STOP_ENABLED = True
_DEFAULT_DEGREE_BUCKETS = 64
_DEFAULT_MAX_LOG_DEG = 8.0

_DEFAULT_TRAIN_ROLLOUTS = 1
_DB_CFG_KEYS = {
    "sampling_temperature_start",
    "sampling_temperature_end",
}
_DB_CFG_OPTIONAL_KEYS = {
}


_SCHED_INTERVAL_EPOCH = "epoch"
_SCHED_INTERVAL_STEP = "step"
_SCHED_INTERVALS = {_SCHED_INTERVAL_EPOCH, _SCHED_INTERVAL_STEP}
_SCHED_TYPE_COSINE = "cosine"
_SCHED_TYPE_COSINE_WARM_RESTARTS = "cosine_warm_restarts"
_SCHED_TYPE_ONECYCLE = "onecycle"
_DEFAULT_SCHED_T_MAX = 10
_DEFAULT_SCHED_T0 = 10
_DEFAULT_SCHED_T_MULT = 1
_DEFAULT_SCHED_ETA_MIN = 0.0
_DEFAULT_ONECYCLE_PCT_START = 0.3
_DEFAULT_ONECYCLE_ANNEAL = "cos"
_DEFAULT_ONECYCLE_CYCLE_MOMENTUM = True
_DEFAULT_ONECYCLE_BASE_MOMENTUM = 0.85
_DEFAULT_ONECYCLE_MAX_MOMENTUM = 0.95
_DEFAULT_ONECYCLE_DIV_FACTOR = 25.0
_DEFAULT_ONECYCLE_FINAL_DIV_FACTOR = 10000.0
_DEFAULT_ONECYCLE_THREE_PHASE = False

_DEFAULT_DIVERSE_BEAM_ENABLED = True
_DEFAULT_DIVERSE_BEAM_GROUPS = 4
_DEFAULT_DIVERSE_BEAM_LAMBDA = 1.0
_DEFAULT_DIVERSE_BEAM_SIMILARITY = "tail"
_DEFAULT_DIVERSE_BEAM_PENALTY = "hard"
_DIVERSE_BEAM_SIMILARITIES = {"tail", "edge", "source"}
_DIVERSE_BEAM_PENALTIES = {"hard", "soft"}

_STANDARD_TRAIN_METRICS = {
    "tb_loss",
    "rollout_success_rate",
    "rollout_length_mean",
    "rollout_terminal_hit_rate",
    "rollout_terminal_dead_end_rate",
    "rollout_terminal_invalid_start_rate",
    "rollout_terminal_emit_rate",
    "rollout_terminal_max_steps_rate",
    "tb_inv_edge_invalid_rate",
    "tb_no_allowed_rate",
    "tb_delta_var",
    "policy_out_degree_mean",
}
_STANDARD_EVAL_METRICS = {
    "hit@beam",
    "recall@beam",
    "precision@beam",
    "f1@beam",
    "diversity@beam",
    "beam_size_adaptive",
    "modes@beam",
    "length_mean",
    "coverage_rate",
    "rollout_success_rate",
    "rollout_terminal_hit_rate",
    "rollout_terminal_dead_end_rate",
    "rollout_terminal_invalid_start_rate",
    "rollout_terminal_emit_rate",
    "rollout_terminal_max_steps_rate",
    "tb_loss",
    "tb_delta_var",
    "tb_inv_edge_invalid_rate",
    "tb_no_allowed_rate",
    "policy_out_degree_mean",
    "policy_tail_in_degree_mean",
    "policy_log_deg_mean",
    "policy_log_deg_std",
    "policy_log_deg_w_eff",
    "policy_log_deg_tail_mean",
    "policy_log_deg_bias",
}
_STANDARD_METRICS = {
    "train": _STANDARD_TRAIN_METRICS,
    "val": _STANDARD_EVAL_METRICS,
    "test": _STANDARD_EVAL_METRICS,
}

__all__ = [
    "_NEG_ONE",
    "_NEG_TWO",
    "_ZERO",
    "_ONE",
    "_TWO",
    "_THREE",
    "_FOUR",
    "_SELF_RELATION_ID",
    "_INVALID_EDGE_ID",
    "_STOP_ACTION_ID",
    "_TERMINAL_NONE",
    "_TERMINAL_HIT",
    "_TERMINAL_DEAD_END",
    "_TERMINAL_MAX_STEPS",
    "_TERMINAL_INVALID_START",
    "_TERMINAL_EMIT",
    "_DEFAULT_GNN_LAYERS",
    "_DEFAULT_GNN_DROPOUT",
    "_DEFAULT_LOGIT_SCALE_INIT",
    "_DEFAULT_PRIOR_WEIGHT_INIT",
    "_DEFAULT_STOP_ENABLED",
    "_DEFAULT_DEGREE_BUCKETS",
    "_DEFAULT_MAX_LOG_DEG",
    "_DEFAULT_TRAIN_ROLLOUTS",
    "_DB_CFG_KEYS",
    "_DB_CFG_OPTIONAL_KEYS",
    "_SCHED_INTERVAL_EPOCH",
    "_SCHED_INTERVAL_STEP",
    "_SCHED_INTERVALS",
    "_SCHED_TYPE_COSINE",
    "_SCHED_TYPE_COSINE_WARM_RESTARTS",
    "_SCHED_TYPE_ONECYCLE",
    "_DEFAULT_SCHED_T_MAX",
    "_DEFAULT_SCHED_T0",
    "_DEFAULT_SCHED_T_MULT",
    "_DEFAULT_SCHED_ETA_MIN",
    "_DEFAULT_ONECYCLE_PCT_START",
    "_DEFAULT_ONECYCLE_ANNEAL",
    "_DEFAULT_ONECYCLE_CYCLE_MOMENTUM",
    "_DEFAULT_ONECYCLE_BASE_MOMENTUM",
    "_DEFAULT_ONECYCLE_MAX_MOMENTUM",
    "_DEFAULT_ONECYCLE_DIV_FACTOR",
    "_DEFAULT_ONECYCLE_FINAL_DIV_FACTOR",
    "_DEFAULT_ONECYCLE_THREE_PHASE",
    "_DEFAULT_DIVERSE_BEAM_ENABLED",
    "_DEFAULT_DIVERSE_BEAM_GROUPS",
    "_DEFAULT_DIVERSE_BEAM_LAMBDA",
    "_DEFAULT_DIVERSE_BEAM_SIMILARITY",
    "_DEFAULT_DIVERSE_BEAM_PENALTY",
    "_DIVERSE_BEAM_SIMILARITIES",
    "_DIVERSE_BEAM_PENALTIES",
    "_STANDARD_TRAIN_METRICS",
    "_STANDARD_EVAL_METRICS",
    "_STANDARD_METRICS",
]
