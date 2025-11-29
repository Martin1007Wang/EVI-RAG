from pathlib import Path
from typing import Optional, Sequence

import torch
from torch_geometric.data import Batch, Data

from src.hrag import GAgentBuilder, GAgentSettings


def _make_batch(*, answer_locals: Optional[Sequence[int]] = None) -> Batch:
    # Graph with 5 nodes and 6 directed edges.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 4],
            [1, 2, 3, 3, 4, 1],
        ],
        dtype=torch.long,
    )
    relation_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    node_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    labels = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
    data = Data(
        edge_index=edge_index,
        edge_attr=relation_ids,
        node_global_ids=node_ids,
        labels=labels,
        sample_id="sample-1",
        num_nodes=node_ids.numel(),
        a_local_indices=answer_locals if answer_locals is not None else [],
    )
    return Batch.from_data_list([data])


def _make_disconnected_batch() -> Batch:
    edge_index = torch.tensor(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )
    relation_ids = torch.tensor([0, 1], dtype=torch.long)
    node_ids = torch.tensor([21, 22, 23, 24], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.float32)
    data = Data(
        edge_index=edge_index,
        edge_attr=relation_ids,
        node_global_ids=node_ids,
        labels=labels,
        sample_id="disconnected-1",
        num_nodes=node_ids.numel(),
        a_local_indices=[],
    )
    return Batch.from_data_list([data])


class _DummyOutput:
    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores
        self.query_ids = torch.zeros(scores.size(0), dtype=torch.long)
        self.logits = None


def _make_output(scores: Optional[Sequence[float]] = None) -> _DummyOutput:
    if scores is None:
        scores = [0.95, 0.05, 0.1, 0.9, 0.01, 0.2]
    tensor = torch.tensor(list(scores), dtype=torch.float32)
    return _DummyOutput(tensor)


def test_g_agent_builder_connects_top_terminals_with_bridge():
    batch = _make_batch()
    output = _make_output()
    settings = GAgentSettings(
        enabled=True,
        top_k=1,
        terminal_top_n=3,
        max_path_length=3,
        output_path=Path("g_agent/test.pt"),
    )
    builder = GAgentBuilder(settings)
    builder.process_batch(batch, output)
    assert len(builder.samples) == 1
    sample = builder.samples[0]
    # Top-1 edge remains, but Steiner expansion should bring in edge 3 to connect terminal node 3.
    assert sample["top_edge_local_indices"] == [0]
    assert sample["core_node_local_indices"] == [0, 1, 3]
    assert set(sample["selected_edge_local_indices"]) == {0, 3}
    assert set(sample["selected_node_local_indices"]) == {0, 1, 3}
    # Selected edge payloads should contain matching indices.
    payload_indices = {edge["local_index"] for edge in sample["selected_edges"]}
    assert payload_indices == {0, 3}


def test_g_agent_builder_emits_answers_and_forces_gt_edges():
    batch = _make_batch(answer_locals=[1, 3])
    output = _make_output()
    settings = GAgentSettings(
        enabled=True,
        beam_width_hop1=2,
        final_k=1,
        terminal_top_n=2,
        max_path_length=2,
        output_path=Path("g_agent/test.pt"),
        force_include_gt=True,
    )
    builder = GAgentBuilder(settings)
    builder.process_batch(batch, output)
    assert len(builder.samples) == 1
    sample = builder.samples[0]
    assert sample["answer_entity_ids"] == [11, 13]
    assert set(sample["selected_edge_local_indices"]) == {0, 3}
    assert sample["core_node_local_indices"] == [0, 1]


def test_g_agent_validation_does_not_use_ground_truth():
    batch = _make_batch()
    output = _make_output()
    settings = GAgentSettings(
        enabled=True,
        beam_width_hop1=2,
        final_k=1,
        terminal_top_n=2,
        max_path_length=2,
        output_path=Path("g_agent/test.pt"),
        force_include_gt=False,
    )
    builder = GAgentBuilder(settings)
    builder.process_batch(batch, output)
    assert len(builder.samples) == 1
    sample = builder.samples[0]
    # Only the top edge (local index 0) should be kept; GT edge 3 stays out.
    assert sample["selected_edge_local_indices"] == [0]
    assert sample["core_component_count"] == 1
    assert sample["retrieval_failed"] is False


def test_g_agent_marks_disconnected_core_components():
    batch = _make_disconnected_batch()
    output = _make_output([0.9, 0.8])
    settings = GAgentSettings(
        enabled=True,
        beam_width_hop1=4,
        final_k=2,
        terminal_top_n=4,
        max_path_length=2,
        output_path=Path("g_agent/test.pt"),
    )
    builder = GAgentBuilder(settings)
    builder.process_batch(batch, output)
    assert len(builder.samples) == 1
    sample = builder.samples[0]
    # Both edges remain because there is no path connecting their nodes.
    assert set(sample["selected_edge_local_indices"]) == {0, 1}
    assert sample["core_component_count"] == 2
    assert sample["retrieval_failed"] is True


def test_g_agent_pcst_backend_falls_back_when_unavailable():
    batch = _make_batch()
    output = _make_output()
    settings = GAgentSettings(
        enabled=True,
        beam_width_hop1=3,
        final_k=1,
        terminal_top_n=3,
        max_path_length=2,
        output_path=Path("g_agent/test.pt"),
    )
    builder = GAgentBuilder(settings)
    builder.process_batch(batch, output)
    assert len(builder.samples) == 1
    sample = builder.samples[0]
    assert set(sample["selected_edge_local_indices"]) == {0, 3}
    assert set(sample["selected_node_local_indices"]) == {0, 1, 3}
