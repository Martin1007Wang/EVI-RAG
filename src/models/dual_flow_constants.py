from __future__ import annotations

_NEG_ONE = -1
_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_SELF_RELATION_ID = -1
_INVALID_EDGE_ID = -1

_DEFAULT_INVERSE_REL_SUFFIX = "__inv"
_DEFAULT_STRICT_INVERSE = True

_TERMINAL_NONE = 0
_TERMINAL_HIT = 1
_TERMINAL_DEAD_END = 2
_TERMINAL_MAX_STEPS = 3
_TERMINAL_INVALID_START = 4

_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_VALIDATE_EDGE_BATCH = False
_DEFAULT_AVOID_REVISIT = True
_DEFAULT_GNN_LAYERS = 2
_DEFAULT_GNN_DROPOUT = 0.0
_DEFAULT_EDGE_INTER_DIM = 256
_DEFAULT_EDGE_DROPOUT = 0.1
_DEFAULT_START_TEMPERATURE_START = 1.0
_DEFAULT_START_TEMPERATURE_END = 1.0

_DEFAULT_TRAIN_ROLLOUTS = 1
_DB_CFG_KEYS = {
    "sampling_temperature_start",
    "sampling_temperature_end",
    "dead_end_log_reward",
    "dead_end_weight",
    "pb_edge_dropout",
}
_DB_CFG_OPTIONAL_KEYS = {
    # Optional cosine-annealing schedule for dead-end grounding:
    # effective_dead_end_log_reward(progress) goes from start -> dead_end_log_reward.
    "dead_end_log_reward_start",
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

_P0_MODE_NONE = "none"
_P0_MODE_DEGREE = "degree"
_P0_MODE_INDEGREE = "indegree"
_P0_MODE_PREFERENTIAL = "preferential"
_P0_MODE_SEMANTIC = "semantic"
_P0_MODES = {_P0_MODE_NONE, _P0_MODE_DEGREE, _P0_MODE_INDEGREE, _P0_MODE_PREFERENTIAL, _P0_MODE_SEMANTIC}
_DEFAULT_P0_MODE = _P0_MODE_DEGREE
_DEFAULT_P0_RESIDUAL = True
_DEFAULT_P0_TEMPERATURE = 1.0
_DEFAULT_P0_COSINE_EPS = 1.0e-6


_STANDARD_TRAIN_METRICS = {
    "rollout_success_rate",
    "rollout_length_mean",
    "rollout_terminal_dead_end_rate",
    "rollout_terminal_max_steps_rate",
    "db_inv_edge_invalid_rate",
    "db_no_allowed_rate",
    "db_valid_step_rate",
    "db_finite_pf_rate",
    "db_finite_pb_rate",
    "db_finite_z_u_rate",
    "db_finite_z_v_rate",
    "db_delta_var",
    "logit_scale_max",
    "policy_drift_abs",
    "policy_drift_rms",
    "policy_kl_p0",
    "policy_out_degree_mean",
    "log_z_mean",
    "log_z_std",
}
_STANDARD_EVAL_METRICS = {
    "hit@beam",
    "recall@beam",
    "precision@beam",
    "f1@beam",
    "diversity@beam",
    "modes@beam",
    "length_mean",
    "coverage_rate",
    "retrieval_failure_rate",
    "rollout_success_rate",
    "rollout_terminal_dead_end_rate",
    "rollout_terminal_max_steps_rate",
    "db_loss",
    "db_log_pb_mean",
    "db_log_pb_min",
    "db_log_z_u_mean",
    "db_log_z_v_mean",
    "db_delta_var",
    "db_inv_edge_invalid_rate",
    "db_no_allowed_rate",
    "db_valid_step_rate",
    "db_finite_pf_rate",
    "db_finite_pb_rate",
    "db_finite_z_u_rate",
    "db_finite_z_v_rate",
    "policy_drift_abs",
    "policy_drift_rms",
    "policy_kl_p0",
    "policy_out_degree_mean",
    "log_z_mean",
    "log_z_std",
}
_STANDARD_METRICS = {
    "train": _STANDARD_TRAIN_METRICS,
    "val": _STANDARD_EVAL_METRICS,
    "test": _STANDARD_EVAL_METRICS,
}

__all__ = [
    "_NEG_ONE",
    "_ZERO",
    "_ONE",
    "_TWO",
    "_THREE",
    "_SELF_RELATION_ID",
    "_INVALID_EDGE_ID",
    "_DEFAULT_INVERSE_REL_SUFFIX",
    "_DEFAULT_STRICT_INVERSE",
    "_TERMINAL_NONE",
    "_TERMINAL_HIT",
    "_TERMINAL_DEAD_END",
    "_TERMINAL_MAX_STEPS",
    "_TERMINAL_INVALID_START",
    "_DEFAULT_BACKBONE_FINETUNE",
    "_DEFAULT_VALIDATE_EDGE_BATCH",
    "_DEFAULT_AVOID_REVISIT",
    "_DEFAULT_GNN_LAYERS",
    "_DEFAULT_GNN_DROPOUT",
    "_DEFAULT_EDGE_INTER_DIM",
    "_DEFAULT_EDGE_DROPOUT",
    "_DEFAULT_START_TEMPERATURE_START",
    "_DEFAULT_START_TEMPERATURE_END",
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
    "_P0_MODE_NONE",
    "_P0_MODE_DEGREE",
    "_P0_MODE_INDEGREE",
    "_P0_MODE_PREFERENTIAL",
    "_P0_MODE_SEMANTIC",
    "_P0_MODES",
    "_DEFAULT_P0_MODE",
    "_DEFAULT_P0_RESIDUAL",
    "_DEFAULT_P0_TEMPERATURE",
    "_DEFAULT_P0_COSINE_EPS",
    "_STANDARD_TRAIN_METRICS",
    "_STANDARD_EVAL_METRICS",
    "_STANDARD_METRICS",
]
