from __future__ import annotations

import json

import pytest

from src.llm.eval_llm import (
    _iter_requests,
    _resolve_input_labels_path,
    _resolve_prompt_spec,
    _validate_topk_against_prompt_limits,
    run_llm_eval,
)
from src.llm.prompting import (
    _enforce_candidate_answers,
    _extract_destination_candidates,
    _extract_subgraphrag_triplet_lines_from_trajectories,
    _select_trajectories,
    _trajectory_text,
)


def test_trajectory_text_filters_super_source_edges_in_fallback() -> None:
    trajectory = {
        "edges": [
            {"src_entity_id": -1, "relation_id": 7, "dst_entity_id": 11},
            {"src_entity_id": 11, "relation_id": 9, "dst_entity_id": 13},
        ],
        "terminal_entity_id": 13,
    }
    assert _trajectory_text(trajectory) == "11 --9--> 13"


def test_trajectory_text_uses_stop_node_when_only_super_source_edges() -> None:
    trajectory = {
        "edges": [
            {"src_entity_id": -1, "relation_id": 7, "dst_entity_id": 11},
        ],
        "terminal_entity_id": 11,
    }
    assert _trajectory_text(trajectory) == "(start_only) 11"


def test_select_trajectories_breaks_probability_ties_with_path_rank() -> None:
    trajectories = [
        {"trajectory_text": "A --r--> B", "prob": 0.9, "path_rank": 2},
        {"trajectory_text": "A --r--> C", "prob": 0.9, "path_rank": 1},
        {"trajectory_text": "A --r--> D", "prob": 0.1, "path_rank": 1},
    ]
    selected = _select_trajectories(
        trajectories,
        top_k=2,
        max_trajectories=0,
        include_score=False,
    )
    assert selected == ["A --r--> C", "A --r--> B"]


def test_enforce_candidate_answers_drops_non_candidate_predictions() -> None:
    answer, constrained = _enforce_candidate_answers(
        answer_raw="Paris | Atlantis",
        candidates=["Paris", "London"],
        answer_separator=" | ",
        allow_empty=False,
    )
    assert answer == "Paris"
    assert constrained is True


def test_enforce_candidate_answers_falls_back_when_empty_is_disallowed() -> None:
    answer, constrained = _enforce_candidate_answers(
        answer_raw="",
        candidates=["Paris", "London"],
        answer_separator=" | ",
        allow_empty=False,
    )
    assert answer == "unknown"
    assert constrained is True


def test_enforce_candidate_answers_uses_fuzzy_match_for_parenthetical_output() -> None:
    answer, constrained = _enforce_candidate_answers(
        answer_raw="2014 (2014 World Series)",
        candidates=["2014", "2012"],
        answer_separator=" | ",
        allow_empty=False,
    )
    assert answer == "2014"
    assert constrained is False


def test_extract_destination_candidates_filters_numeric_ids_for_non_numeric_questions() -> (
    None
):
    trajectories = [
        "A --related_to--> 123456",
        "A --related_to--> Paris",
    ]
    candidates = _extract_destination_candidates(
        trajectories, max_candidates=10, question="Where was A born?"
    )
    assert candidates == ["Paris"]


def test_extract_destination_candidates_keeps_year_for_year_questions() -> None:
    trajectories = [
        "A --won--> 2014",
        "A --won--> 123456",
    ]
    candidates = _extract_destination_candidates(
        trajectories,
        max_candidates=10,
        question="What year did A win the championship?",
    )
    assert candidates == ["2014"]


def test_extract_destination_candidates_filters_freebase_ids() -> None:
    trajectories = [
        "A --related_to--> m.0123ab",
        "A --related_to--> g.11xyz",
        "A --related_to--> Paris",
    ]
    candidates = _extract_destination_candidates(
        trajectories,
        max_candidates=10,
        question="Where was A born?",
    )
    assert candidates == ["Paris"]


def test_extract_destination_candidates_can_include_intermediate_nodes() -> None:
    trajectories = [
        "A --r1--> Gold Answer ; Gold Answer --r2--> Distractor",
    ]
    endpoints_only = _extract_destination_candidates(
        trajectories,
        max_candidates=10,
        question="Who is the correct answer?",
        candidate_source="endpoints_only",
    )
    trajectory_nodes = _extract_destination_candidates(
        trajectories,
        max_candidates=10,
        question="Who is the correct answer?",
        candidate_source="trajectory_nodes",
    )
    assert endpoints_only == ["Distractor"]
    assert "Gold Answer" in trajectory_nodes
    assert "Distractor" in trajectory_nodes


def test_extract_destination_candidates_supports_start_only_trajectory_nodes() -> None:
    candidates = _extract_destination_candidates(
        ["(start_only) Paris"],
        max_candidates=10,
        question="Where was A born?",
        candidate_source="trajectory_nodes",
    )
    assert candidates == ["Paris"]


def test_extract_subgraphrag_triplet_lines_filter_virtual_edges() -> None:
    lines = _extract_subgraphrag_triplet_lines_from_trajectories(
        [
            "[prob=0.9] super_source --from_question--> A ; "
            "A --SELF--> A ; A --born_in--> Paris ; Paris --STOP--> Paris"
        ]
    )
    assert lines == ["(A,born_in,Paris)"]


def test_iter_requests_supports_question_field_alias(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "trajectories": [
            {"path_rank": 1, "prob": 1.0, "trajectory_text": "A --r--> B"}
        ],
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    prompt_spec = _resolve_prompt_spec({"prompt": {"system": "You are a test model."}})
    requests = list(
        _iter_requests(
            input_path, set(), top_k=1, prompt_spec=prompt_spec, max_samples=None
        )
    )
    assert len(requests) == 1
    assert requests[0].question == "Where was X born?"


def test_resolve_input_labels_path_requires_sidecar_when_enabled(tmp_path) -> None:
    input_path = tmp_path / "predict.jsonl"
    input_path.write_text("", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError, match="Input labels JSONL not found for metrics"
    ):
        _resolve_input_labels_path(
            input_path=input_path, llm_cfg={}, require_labels=True
        )


def test_run_llm_eval_fails_fast_when_sidecar_missing(tmp_path) -> None:
    input_path = tmp_path / "predict.jsonl"
    input_path.write_text("", encoding="utf-8")
    cfg = {
        "dataset": {"name": "webqsp", "artifact_dir": str(tmp_path)},
        "run": {"split": "test"},
        "llm": {
            "provider": "openai",
            "topk_list": [1],
            "compute_metrics": True,
            "input_path": str(input_path),
            "prompt": {"system": "You are a test model."},
            "schema": {"enabled": False},
        },
    }
    with pytest.raises(
        FileNotFoundError, match="Input labels JSONL not found for metrics"
    ):
        run_llm_eval(cfg)


def test_validate_topk_against_prompt_limits_fails_on_implicit_clip() -> None:
    prompt_spec = _resolve_prompt_spec(
        {"prompt": {"system": "You are a test model.", "max_trajectories": 5}}
    )
    with pytest.raises(
        ValueError, match="llm.topk_list must be <= llm.prompt.max_trajectories"
    ):
        _validate_topk_against_prompt_limits(topk_list=[1, 10], prompt_spec=prompt_spec)
