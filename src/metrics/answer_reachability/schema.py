from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    src_entity_id: int
    relation_id: int
    dst_entity_id: int


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    path_rank: int
    log_prob: float
    prob: float
    cumulative_mass: float
    terminal_entity_id: int
    is_gold: bool
    edges: list[EdgeRecord]
    start_entity_id: int | None = None
    answer_rank: int = 0
    support_rank: int = 0
    conditional_prob: float = 0.0
    conditional_cumulative_mass: float = 0.0


@dataclass(frozen=True)
class AnswerPosteriorRecord:
    answer_entity_id: int
    prob: float
    cumulative_mass: float
    is_gold: bool
    is_selected: bool = False
    support_mass: float = 0.0
    support_conditioned_mass: float = 0.0
    support_path_count: int = 0


@dataclass(frozen=True)
class AnswerSupportRecord:
    answer_entity_id: int
    answer_rank: int
    prob: float
    cumulative_mass: float
    is_gold: bool
    is_selected: bool
    support_mass: float
    support_conditioned_mass: float
    support_path_count: int
    trajectories: list[TrajectoryRecord]


@dataclass(frozen=True)
class SupportWindowResult:
    sample_id: str
    dataset_scope: str
    mass_threshold: float
    window_size: int
    covered_mass: float
    residual_mass: float
    gold_total_mass: float
    covered_gold_mass: float
    missed_gold_mass: float
    unique_answer_count: int
    unique_path_count: int
    gold_answer_entity_ids: list[int]
    start_entity_ids: list[int]
    trajectories: list[TrajectoryRecord]
    inference_mode: str = "exact"
    answer_mass_threshold: float = 1.0
    support_mass_threshold: float = 1.0
    probe_count: int = 0
    emit_path_count: int = 0
    remaining_mass_upper: float = 0.0
    stop_reason: str = ""
    coverage_certified: bool = False
    answer_mass_reference: str = "exact"
    support_mass_reference: str = "exact"
    selected_answer_ids: list[int] = field(default_factory=list)
    answer_posterior: list[AnswerPosteriorRecord] = field(default_factory=list)
    answer_support: list[AnswerSupportRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SupportWindowLabelRecord:
    sample_id: str
    question: str
    start_entity_ids: list[int]
    answer_entity_ids: list[int]
    a_entity_in_graph: bool


@dataclass(frozen=True)
class SupportWindowEvalBatch:
    dataset_scope: str
    mass_threshold: float
    results: list[SupportWindowResult]
    window_top_ks: tuple[int, ...] = (1, 10, 25, 50, 100)
