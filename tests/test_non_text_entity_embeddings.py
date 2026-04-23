from __future__ import annotations

from pathlib import Path
import sys
import types

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
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.data.preprocess_steps.samples import PreparedSample, RawSample
from src.data.preprocess_steps.text_encode import encode_preprocessed_features
from src.data.preprocess_steps.vocab import EntityVocab, RelationVocab
from src.data.retrieval.embedding_store import EmbeddingStore


class _DummyTextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize: bool = True,
        progress_bar: bool = True,
    ) -> None:
        del model_name, device, normalize, progress_bar
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


def test_encode_preprocessed_features_keeps_non_text_sentinel_finite(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "src.data.preprocess_steps.text_encode.TextEncoder", _DummyTextEncoder
    )

    entity_vocab = EntityVocab(
        is_text_entity=lambda entity: entity.startswith("text:"),
        is_cvt_entity=lambda entity: False,
    )
    entity_vocab.add("mid:non-text")
    entity_vocab.add("text:entity")

    relation_vocab = RelationVocab()
    relation_vocab.add("rel:linked_to")

    sample = PreparedSample(
        sample=RawSample(
            dataset="demo",
            split="train",
            question_id="q1",
            kb="kb",
            question="who is linked",
            graph=tuple(),
            question_entities=tuple(),
            answer_entities=tuple(),
            answer_texts=tuple(),
        ),
        sample_id="sample-1",
        kept_edges=[],
        question_entities_in_graph=tuple(),
        reachable_answer_entities=tuple(),
        all_target_node_ids=torch.empty(0, dtype=torch.long),
        shortest_path_edge_mask=torch.empty(0, dtype=torch.bool),
        node_to_target_distance=torch.empty(0, dtype=torch.long),
        shortest_path_count=torch.empty(0, dtype=torch.float32),
        target_node_distance_flat=torch.empty(0, dtype=torch.long),
        target_shortest_path_count_flat=torch.empty(0, dtype=torch.float32),
        target_shortest_path_edge_mask_flat=torch.empty(0, dtype=torch.bool),
        max_path_length=None,
    )

    payload = encode_preprocessed_features(
        prepared_samples=[sample],
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        embeddings_dir=tmp_path,
        encoder_name="dummy",
        progress_bar=False,
    )

    assert torch.allclose(
        payload.entity_embeddings[0],
        torch.zeros(_DummyTextEncoder("dummy").hidden_size, dtype=torch.float32),
    )
    assert torch.isfinite(payload.entity_embeddings).all()


def test_embedding_store_masks_legacy_nan_non_text_rows(tmp_path: Path) -> None:
    entity_embeddings = torch.tensor(
        [[float("nan"), float("nan")], [1.0, 2.0]],
        dtype=torch.float32,
    )
    relation_embeddings = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    torch.save(entity_embeddings, tmp_path / "entity_embeddings.pt")
    torch.save(relation_embeddings, tmp_path / "relation_embeddings.pt")

    store = EmbeddingStore(tmp_path)
    resolved = store.get_entity_embeddings(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(resolved[0], torch.zeros(2, dtype=torch.float32))
    assert torch.allclose(resolved[1], torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_reward_model_zero_f1_edge_bonus_ignores_root_edges() -> None:
    from src.models.reward import RewardModel

    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=1.0)
    retrieval_batch = types.SimpleNamespace(
        num_graphs=1,
        batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        relation_tokens=torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32),
        train_target_mask=torch.tensor([False, False, False, True], dtype=torch.bool),
        is_anchor_mask=torch.tensor([True, True, False, False], dtype=torch.bool),
    )
    reward = reward_model(
        retrieval_batch=retrieval_batch,
        active_nodes=torch.tensor([True, True, True, False], dtype=torch.bool),
        active_edges=torch.tensor([True, True], dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([-4.4], dtype=torch.float32), atol=1e-6)


def test_reward_model_zero_f1_edge_bonus_uses_best_selected_non_root_edge() -> None:
    from src.models.reward import RewardModel

    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=0.5)
    retrieval_batch = types.SimpleNamespace(
        num_graphs=1,
        batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        relation_tokens=torch.tensor(
            [[0.2, 0.9797959], [0.8, 0.6]], dtype=torch.float32
        ),
        train_target_mask=torch.tensor([False, False, False, True], dtype=torch.bool),
        is_anchor_mask=torch.tensor([True, False, False, False], dtype=torch.bool),
    )

    reward = reward_model(
        retrieval_batch=retrieval_batch,
        active_nodes=torch.tensor([True, True, True, False], dtype=torch.bool),
        active_edges=torch.tensor([True, True], dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([-4.6], dtype=torch.float32), atol=1e-6)


def test_reward_model_positive_f1_does_not_use_zero_f1_edge_bonus() -> None:
    from src.models.reward import RewardModel

    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=1.0)
    retrieval_batch = types.SimpleNamespace(
        num_graphs=1,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        relation_tokens=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        train_target_mask=torch.tensor([False, True, False], dtype=torch.bool),
        is_anchor_mask=torch.tensor([True, False, False], dtype=torch.bool),
    )

    reward = reward_model(
        retrieval_batch=retrieval_batch,
        active_nodes=torch.tensor([True, True, False], dtype=torch.bool),
        active_edges=torch.tensor([True], dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([0.0], dtype=torch.float32), atol=1e-6)


def test_reward_model_uses_train_target_mask() -> None:
    from src.models.reward import RewardModel

    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=0.0)
    retrieval_batch = types.SimpleNamespace(
        num_graphs=1,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        relation_tokens=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        train_target_mask=torch.tensor([False, False, True], dtype=torch.bool),
        is_anchor_mask=torch.tensor([True, False, False], dtype=torch.bool),
    )

    reward = reward_model(
        retrieval_batch=retrieval_batch,
        active_nodes=torch.tensor([True, True, False], dtype=torch.bool),
        active_edges=torch.tensor([True], dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([-5.0], dtype=torch.float32), atol=1e-6)
