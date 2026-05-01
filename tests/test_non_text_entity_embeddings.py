from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModel = object
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -float("inf"), dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.data.preprocess.materialize import _entity_text_embedding_ids_to_map
from src.data.preprocess.text_encode import encode_text_features
from src.data.preprocess.vocab import EntityCatalog, EntityTyping, EntityVocab
from src.data.preprocess.vocab import RelationCatalog, RelationVocab
from src.weaver.nn.feature_encoder import FeatureEncoder


class _DummyTextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        progress_bar: bool = True,
    ) -> None:
        del model_name, device, progress_bar
        self.hidden_size = 4

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        desc: str = "Encode",
        query_prefix: str = "",
    ) -> torch.Tensor:
        del batch_size, desc, query_prefix
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        return torch.arange(len(texts) * self.hidden_size, dtype=torch.float32).view(
            len(texts), self.hidden_size
        )


class _CountingTextEncoder(_DummyTextEncoder):
    instances = 0
    calls: list[tuple[str, tuple[str, ...], str]] = []

    @classmethod
    def reset(cls) -> None:
        cls.instances = 0
        cls.calls = []

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        progress_bar: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, device=device, progress_bar=progress_bar)
        type(self).instances += 1

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        desc: str = "Encode",
        query_prefix: str = "",
    ) -> torch.Tensor:
        type(self).calls.append((desc, tuple(texts), query_prefix))
        return super().encode(
            texts=texts,
            batch_size=batch_size,
            desc=desc,
            query_prefix=query_prefix,
        )


def test_encode_text_features_omits_non_text_entities_from_text_table(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.data.preprocess.text_encode.TextEncoder",
        _DummyTextEncoder,
    )

    entity_vocab = EntityVocab()
    entity_vocab.add("m.non_text")
    entity_vocab.add("text entity")
    entity_catalog = EntityCatalog.build(entity_vocab, typing=EntityTyping())

    relation_vocab = RelationVocab()
    relation_vocab.add("/people/person/place_of_birth")
    relation_catalog = RelationCatalog.build(relation_vocab)

    encoded = encode_text_features(
        entity_text_labels=entity_catalog.entity_text_labels,
        relation_text_labels=relation_catalog.relation_text_labels,
        question_texts=["who was born there"],
        encoder_name="dummy",
        batch_size=2,
        progress_bar=False,
    )

    assert entity_catalog.entity_text_labels == ["text entity"]
    assert torch.equal(
        _entity_text_embedding_ids_to_map(entity_catalog.entity_text_embedding_ids),
        torch.tensor([-1, 0], dtype=torch.long),
    )
    assert encoded.entity_text_embeddings.shape == (1, 4)
    assert encoded.relation_embeddings.shape == (1, 4)
    assert encoded.question_embeddings.shape == (1, 4)
    assert torch.isfinite(encoded.entity_text_embeddings).all()
    assert torch.isfinite(encoded.relation_embeddings).all()
    assert torch.isfinite(encoded.question_embeddings).all()


def test_encode_text_features_reuses_cache_without_loading_encoder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _CountingTextEncoder.reset()
    monkeypatch.setattr(
        "src.data.preprocess.text_encode.TextEncoder",
        _CountingTextEncoder,
    )

    first = encode_text_features(
        entity_text_labels=["entity a", "entity b"],
        relation_text_labels=["relation a"],
        question_texts=["question a"],
        encoder_name="dummy",
        batch_size=2,
        progress_bar=False,
        cache_dir=tmp_path,
    )

    assert _CountingTextEncoder.instances == 1
    assert _CountingTextEncoder.calls == [
        ("Entities", ("entity a", "entity b"), ""),
        ("Relations", ("relation a",), ""),
        ("Questions", ("question a",), "Represent this sentence: "),
    ]
    assert len(list(tmp_path.glob("*.pt"))) == 3

    second = encode_text_features(
        entity_text_labels=["entity a", "entity b"],
        relation_text_labels=["relation a"],
        question_texts=["question a"],
        encoder_name="dummy",
        batch_size=2,
        progress_bar=False,
        cache_dir=tmp_path,
    )

    assert _CountingTextEncoder.instances == 1
    assert len(_CountingTextEncoder.calls) == 3
    assert torch.equal(first.entity_text_embeddings, second.entity_text_embeddings)
    assert torch.equal(first.relation_embeddings, second.relation_embeddings)
    assert torch.equal(first.question_embeddings, second.question_embeddings)

    encode_text_features(
        entity_text_labels=["entity a", "entity b"],
        relation_text_labels=["relation a"],
        question_texts=["question b"],
        encoder_name="dummy",
        batch_size=2,
        progress_bar=False,
        cache_dir=tmp_path,
    )

    assert _CountingTextEncoder.instances == 2
    assert _CountingTextEncoder.calls[-1] == (
        "Questions",
        ("question b",),
        "Represent this sentence: ",
    )


def test_feature_encoder_keeps_non_text_nodes_finite_without_text_row() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
    )
    text_entity_embedding = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    encoder = FeatureEncoder(
        entity_text_embeddings=text_entity_embedding,
        entity_embedding_map=torch.tensor([-1, 0], dtype=torch.long),
        relation_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        embedding_dim=4,
        hidden_dim=4,
        dde={"enabled": False},
        non_text_init_std=0.0,
    )

    fb = encoder(batch)

    assert fb.node_is_non_text is not None
    assert torch.equal(
        fb.node_is_non_text,
        torch.tensor([True, False], dtype=torch.bool),
    )
    assert torch.allclose(fb.node_sem_h[0], torch.zeros(4, dtype=torch.float32))
    assert torch.allclose(fb.node_sem_h[1], text_entity_embedding.squeeze(0))
    assert torch.isfinite(fb.node_sem_h).all()
    assert torch.isfinite(fb.node_h).all()
