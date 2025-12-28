from src.utils.llm_metrics import evaluate_predictions


def test_semantic_metrics_respect_empty_visible_edges() -> None:
    metrics = evaluate_predictions(
        [
            {
                "id": "q1",
                "question": "q",
                "answers": ["A"],
                "prediction": "{\"answers\": [\"A\"]}",
                "window_k": 5,
                "k_effective": 5,
                "retrieved_edge_ids": [1, 2, 3],
                "visible_edge_ids": [],
                "gt_path_edge_local_ids": [1],
                "evidence_token_count": 0,
                "prompt_token_count": 10,
                "token_budget": 1,
                "evidence_truncated": True,
            }
        ]
    )

    assert metrics["semantic/with_gt"] == 1.0
    assert metrics["semantic/s_ret_set"] == 1.0
    assert metrics["semantic/s_ret_vis"] == 0.0
    assert metrics["semantic/l_iface"] == 1.0
    assert metrics["semantic/avg_k_visible"] == 0.0


def test_semantic_metrics_use_explicit_hit_flags() -> None:
    metrics = evaluate_predictions(
        [
            {
                "id": "q1",
                "question": "q",
                "answers": ["A"],
                "prediction": "{\"answers\": [\"A\"]}",
                "window_k": 5,
                "retrieved_edge_ids": [1],
                "visible_edge_ids": [1],
                "gt_path_edge_local_ids": [1],
                # Even if IDs suggest visibility, explicit flags are the SSOT for hit events.
                "hit_set": True,
                "hit_vis": False,
                "evidence_token_count": 0,
                "prompt_token_count": 10,
                "token_budget": None,
                "evidence_truncated": False,
            }
        ]
    )

    assert metrics["semantic/with_gt"] == 1.0
    assert metrics["semantic/s_ret_set"] == 1.0
    assert metrics["semantic/s_ret_vis"] == 0.0
