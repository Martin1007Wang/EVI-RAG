from __future__ import annotations

import torch

from scripts.diagnose_replay_buffer_vs_policy import (
    Candidate,
    exhaustive_shortest_path_candidates,
    find_truncated_sample,
    prefix_sets_by_edge,
    replay_program_candidates,
    state_edge_category,
)
from src.data.schema.batch import ReplayProgramSample
from src.graph.oracle_replay import build_replay_program


def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_replay_program_candidates_decode_paths() -> None:
    program = ReplayProgramSample(
        candidate_edge_ids_local=torch.tensor([1, 5, 2, 7], dtype=torch.long),
        candidate_ptr=torch.tensor([0, 2, 4], dtype=torch.long),
        candidate_target_positions=torch.tensor([0, 1], dtype=torch.long),
        candidate_target_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_to_candidate_ids_local=torch.tensor([0, 1], dtype=torch.long),
        edge_to_candidate_ptr=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        path_truncated=torch.tensor(0, dtype=torch.long),
    )

    candidates = replay_program_candidates(program)

    assert candidates == [
        Candidate(edge_tuple=(1, 5), target_positions=(0,)),
        Candidate(edge_tuple=(2, 7), target_positions=(1,)),
    ]


def test_exhaustive_shortest_path_candidates_exceeds_materialized_limit() -> None:
    num_mid = 65
    edges = [(0, idx) for idx in range(1, num_mid + 1)] + [(idx, num_mid + 1) for idx in range(1, num_mid + 1)]
    edge_index = _edge_tensor(edges)
    materialized = build_replay_program(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([num_mid + 1], dtype=torch.long),
        num_nodes=num_mid + 2,
        max_paths_per_target=64,
    )
    exhaustive = exhaustive_shortest_path_candidates(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([num_mid + 1], dtype=torch.long),
        num_nodes=num_mid + 2,
    )

    assert int(materialized.candidate_ptr.numel()) - 1 == 64
    assert len(exhaustive) == 65


def test_state_edge_category_distinguishes_materialized_and_omitted() -> None:
    materialized = [
        Candidate(edge_tuple=(0, 2), target_positions=(0,)),
    ]
    exhaustive = [
        Candidate(edge_tuple=(0, 2), target_positions=(0,)),
        Candidate(edge_tuple=(1, 3), target_positions=(0,)),
    ]
    materialized_prefixes = prefix_sets_by_edge(materialized)
    exhaustive_prefixes = prefix_sets_by_edge(exhaustive)

    assert state_edge_category(
        state_edges=frozenset(),
        edge_id=0,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
    ) == "shared"
    assert state_edge_category(
        state_edges=frozenset(),
        edge_id=1,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
    ) == "omitted_only"
    assert state_edge_category(
        state_edges=frozenset({0}),
        edge_id=3,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
    ) == "neither_shortest_path"


def test_prefix_sets_are_order_invariant_for_membership() -> None:
    prefixes = prefix_sets_by_edge(
        [
            Candidate(edge_tuple=(4, 1, 9), target_positions=(0,)),
        ]
    )

    assert frozenset({4}) in prefixes[4]
    assert frozenset({4, 1}) in prefixes[4]
    assert frozenset({4, 1}) in prefixes[1]
    assert frozenset({4, 1, 9}) in prefixes[9]


def test_find_truncated_sample_returns_expected_validation_case() -> None:
    from omegaconf import OmegaConf, open_dict

    from src.training.factory import prepare_training_components

    cfg = OmegaConf.load("outputs/debug_valfit/2026-05-30/13-14-37/.hydra/config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = "/mnt/data/retrieval"
        cfg.logger = None
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.enable_checkpointing = False

    datamodule, _ = prepare_training_components(cfg, stage="fit")
    dataset = datamodule.val_dataset
    assert dataset is not None

    index, sample_id = find_truncated_sample(dataset)

    assert index == 13
    assert sample_id == "webqsp/validation/WebQTrn-267"
