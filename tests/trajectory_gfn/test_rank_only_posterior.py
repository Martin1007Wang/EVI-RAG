from __future__ import annotations

from src.models.trajectory_gfn.posterior import (
    DiscoveredTrajectory,
    build_rank_only_result_from_discovered_paths,
)

from .conftest import make_toy_batch


def test_rank_only_result_orders_answers_from_sampled_paths() -> None:
    batch = make_toy_batch()
    result = build_rank_only_result_from_discovered_paths(
        batch=batch,
        discovered_paths=[
            DiscoveredTrajectory(
                start_node=0,
                terminal_node=2,
                answer_entity_id=102,
                edge_ids=(0, 2),
                log_prob=-0.2,
                is_gold=True,
            ),
            DiscoveredTrajectory(
                start_node=0,
                terminal_node=1,
                answer_entity_id=101,
                edge_ids=(0,),
                log_prob=-1.2,
                is_gold=False,
            ),
        ],
        inference_mode="sampled_rank_only",
        answer_mass_threshold=0.9,
        probe_count=16,
        remaining_mass_upper=0.1,
        stop_reason="rank_only_sampled",
    )
    assert result.inference_mode == "sampled_rank_only"
    assert [record.answer_entity_id for record in result.answer_posterior] == [102, 101]
    assert result.answer_posterior[0].is_gold is True
    assert result.answer_posterior[0].is_selected is True
    assert result.probe_count == 16
