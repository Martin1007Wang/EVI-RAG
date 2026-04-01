from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import json
import pytest
import torch

from src.data.io import raw_loader, stark_adapter


class _FakeNode:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeQADataset:
    def __init__(self) -> None:
        self._rows = [
            ("Which drug targets EGFR?", 101, [3], None),
            ("Which pathway is EGFR linked to?", 102, [1], None),
            ("Which pathway connects EGFR and AKT1?", 103, [1], None),
        ]

    def __getitem__(self, idx: int):  # type: ignore[no-untyped-def]
        return self._rows[int(idx)]

    def get_idx_split(self):  # type: ignore[no-untyped-def]
        return {
            "train": torch.tensor([0]),
            "val": torch.tensor([1]),
            "test": torch.tensor([2]),
        }


class _FakeSKB:
    def __init__(self) -> None:
        self.node_info = {
            0: {
                "name": "EGFR",
                "type": "gene/protein",
                "source": "unit",
                "details": {"alias": ["Epidermal growth factor receptor"]},
            },
            1: {
                "name": "MAPK signaling pathway",
                "type": "pathway",
                "source": "unit",
            },
            2: {
                "name": "AKT1",
                "type": "gene/protein",
                "source": "unit",
                "details": {"alias": ["PKB alpha"]},
            },
            3: {
                "name": "Gefitinib",
                "type": "drug",
                "source": "unit",
            },
        }
        self.edge_index = torch.tensor([[0, 2, 3], [1, 1, 0]], dtype=torch.long)
        self.edge_types = torch.tensor([0, 0, 1], dtype=torch.long)
        self._edge_types = {0: "linked to", 1: "target"}

    def num_nodes(self) -> int:
        return len(self.node_info)

    def __getitem__(self, idx: int) -> _FakeNode:
        return _FakeNode(**self.node_info[int(idx)])

    def get_edge_type_by_id(self, edge_id: int) -> str:
        return self._edge_types[int(edge_id)]


def _build_stark_cfg(tmp_path: Path, *, backend: str) -> dict[str, object]:
    return {
        "dataset": "prime",
        "root": str(tmp_path / "stark"),
        "cache_dir": str(tmp_path / "cache"),
        "download_processed": True,
        "indirected": False,
        "linker": {
            "backend": backend,
            "model": "local-llama" if backend == "vllm" else None,
            "max_candidates": 8,
            "max_entities": 4,
            "tensor_parallel_size": 1,
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "local_graph": {
            "num_hops": 1,
            "direction": "both",
            "max_nodes": 8,
            "max_edges": 16,
            "include_inverse_edges": True,
        },
    }


def _patch_prime_resources(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stark_adapter._ADAPTER_CACHE.clear()
    monkeypatch.setattr(
        stark_adapter,
        "_load_prime_resources",
        lambda **kwargs: (_FakeQADataset(), _FakeSKB()),
    )


def test_raw_loader_supports_stark_prime_keyword_preprocess(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_prime_resources(monkeypatch)

    samples = list(
        raw_loader.iter_samples(
            dataset="prime",
            kb="prime",
            raw_root=None,
            splits=["validation"],
            column_map={},
            entity_normalization="none",
            dataset_source="stark",
            stark_cfg=_build_stark_cfg(tmp_path, backend="keyword"),
        )
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.split == "validation"
    assert sample.question_id == "102"
    assert any("EGFR" in entity for entity in sample.question_entities)
    assert sample.answer_texts == ["MAPK signaling pathway"]
    assert any(edge[1] == "linked to" for edge in sample.graph)
    cache_path = tmp_path / "cache" / "samples" / "validation" / "102.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["cache_signature"]


def test_raw_loader_supports_stark_prime_vllm_entity_selection(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_prime_resources(monkeypatch)

    def _fake_generate(messages_batch):  # type: ignore[no-untyped-def]
        prompt = messages_batch[0][-1]["content"]
        assert "EGFR" in prompt
        assert "AKT1" in prompt
        return ['{"selected_candidate_ids": [2, 0], "mentions": ["AKT1", "EGFR"]}']

    monkeypatch.setattr(
        stark_adapter.StarkPrimeAdapter,
        "_get_vllm_generate",
        lambda self: _fake_generate,
    )

    samples = list(
        raw_loader.iter_samples(
            dataset="prime",
            kb="prime",
            raw_root=None,
            splits=["test"],
            column_map={},
            entity_normalization="none",
            dataset_source="stark",
            stark_cfg=_build_stark_cfg(tmp_path, backend="vllm"),
        )
    )

    assert len(samples) == 1
    sample = samples[0]
    assert len(sample.question_entities) == 2
    assert any("EGFR" in entity for entity in sample.question_entities)
    assert any("AKT1" in entity for entity in sample.question_entities)
    assert any(edge[1] == "inverse::target" for edge in sample.graph)


def test_stark_sample_cache_rebuilds_when_config_changes(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_prime_resources(monkeypatch)

    list(
        raw_loader.iter_samples(
            dataset="prime",
            kb="prime",
            raw_root=None,
            splits=["test"],
            column_map={},
            entity_normalization="none",
            dataset_source="stark",
            stark_cfg=_build_stark_cfg(tmp_path, backend="keyword"),
        )
    )
    cache_path = tmp_path / "cache" / "samples" / "test" / "103.json"
    initial_signature = json.loads(cache_path.read_text(encoding="utf-8"))[
        "cache_signature"
    ]

    vllm_calls: list[str] = []

    def _fake_generate(messages_batch):  # type: ignore[no-untyped-def]
        prompt = messages_batch[0][-1]["content"]
        assert "AKT1" in prompt
        vllm_calls.append(prompt)
        return ['{"selected_candidate_ids": [2, 0], "mentions": ["AKT1", "EGFR"]}']

    monkeypatch.setattr(
        stark_adapter.StarkPrimeAdapter,
        "_get_vllm_generate",
        lambda self: _fake_generate,
    )

    vllm_samples = list(
        raw_loader.iter_samples(
            dataset="prime",
            kb="prime",
            raw_root=None,
            splits=["test"],
            column_map={},
            entity_normalization="none",
            dataset_source="stark",
            stark_cfg=_build_stark_cfg(tmp_path, backend="vllm"),
        )
    )

    assert len(vllm_samples[0].question_entities) == 2
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(vllm_calls) == 1
    assert payload["cache_signature"] != initial_signature
    assert payload["question_entities"] == list(vllm_samples[0].question_entities)


def test_build_retrieval_pipeline_prime_resolves_stark_source() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="build_retrieval_pipeline.yaml",
            overrides=["dataset=prime"],
        )

    assert cfg.dataset_source == "stark"
    assert cfg.stark.dataset == "prime"
    assert cfg.stark.linker.backend == "vllm"
    assert cfg.stark.local_graph.include_inverse_edges is True


def test_load_cached_sample_rejects_legacy_entity_fields(tmp_path: Path) -> None:
    adapter = stark_adapter.StarkPrimeAdapter.__new__(stark_adapter.StarkPrimeAdapter)
    adapter.dataset = "prime"
    adapter.kb = "prime"
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "dataset": "prime",
                "split": "train",
                "question_id": "1",
                "kb": "prime",
                "question": "Which drug targets EGFR?",
                "graph": [],
                "q_entity": ["EGFR"],
                "a_entity": ["Gefitinib"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy STaRK cache payload"):
        adapter._load_cached_sample(cache_path)
