from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from torch_geometric.data import Batch, Data

from src.data.components import SharedDataResources
from src.models.components.gflownet_embedder import GraphEmbedder


def _write_vocab_lmdb(path: Path) -> None:
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            entity_to_id = {"e0": 0, "e1": 1, "e2": 2}
            relation_to_id = {"r0": 0}
            txn.put(b"entity_to_id", pickle.dumps(entity_to_id))
            txn.put(b"id_to_entity", pickle.dumps({v: k for k, v in entity_to_id.items()}))
            txn.put(b"relation_to_id", pickle.dumps(relation_to_id))
            txn.put(b"id_to_relation", pickle.dumps({v: k for k, v in relation_to_id.items()}))
    finally:
        env.close()


def _write_mock_checkpoint(path: Path, *, hidden_dim: int) -> None:
    torch.manual_seed(0)
    emb_dim = hidden_dim
    semantic_dim = 4 * hidden_dim
    num_topics = 2
    num_rounds = 2
    num_rev = 2
    struct_dim = 2 * num_topics * (1 + num_rounds + num_rev)
    input_dim = semantic_dim + struct_dim

    state_dict = {
        "entity_proj.network.0.weight": torch.eye(hidden_dim),
        "entity_proj.network.0.bias": torch.zeros(hidden_dim),
        "relation_proj.network.0.weight": torch.eye(hidden_dim),
        "relation_proj.network.0.bias": torch.zeros(hidden_dim),
        "query_proj.network.0.weight": torch.eye(hidden_dim),
        "query_proj.network.0.bias": torch.zeros(hidden_dim),
        "non_text_entity_emb.weight": torch.zeros(1, hidden_dim),
        "feature_extractor.network.0.weight": torch.randn(emb_dim, input_dim),
        "feature_extractor.network.0.bias": torch.randn(emb_dim),
        "feature_extractor.network.3.weight": torch.randn(hidden_dim, emb_dim),
        "feature_extractor.network.3.bias": torch.randn(hidden_dim),
        "parity_meta": torch.tensor([1, num_topics, num_rounds, num_rev], dtype=torch.long),
    }
    torch.save({"state_dict": state_dict}, path)


def test_graph_embedder_aligns_retriever_structure_features(tmp_path: Path) -> None:
    root = tmp_path / "mock_resources"
    vocab_dir = root / "vocabulary"
    vocab_dir.mkdir(parents=True)
    vocab_path = vocab_dir / "vocabulary.lmdb"
    _write_vocab_lmdb(vocab_path)

    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    entity_embeddings = torch.arange(12, dtype=torch.float32).view(3, 4)
    relation_embeddings = torch.arange(4, dtype=torch.float32).view(1, 4)
    torch.save(entity_embeddings, embeddings_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, embeddings_dir / "relation_embeddings.pt")

    ckpt_path = tmp_path / "mock_retriever.ckpt"
    _write_mock_checkpoint(ckpt_path, hidden_dim=4)

    resources = SharedDataResources(vocabulary_path=vocab_path, embeddings_dir=embeddings_dir)
    embedder = GraphEmbedder(
        hidden_dim=4,
        proj_dropout=0.0,
        projector_checkpoint=str(ckpt_path),
        projector_key_prefixes=[""],
    ).eval()
    embedder.setup(resources, device=torch.device("cpu"))

    graph = Data(
        num_nodes=3,
        edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, dtype=torch.long),
        edge_scores=torch.tensor([0.1, 0.2], dtype=torch.float32),
        edge_labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        node_global_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        node_embedding_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        question_emb=torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32),
        start_entity_ids=torch.tensor([0], dtype=torch.long),
        start_node_locals=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([1], dtype=torch.long),
        answer_node_locals=torch.tensor([1], dtype=torch.long),
        gt_path_edge_local_ids=torch.tensor([0], dtype=torch.long),
        gt_path_node_local_ids=torch.tensor([0, 1], dtype=torch.long),
        gt_path_exists=torch.tensor([True], dtype=torch.bool),
        is_answer_reachable=torch.tensor([True], dtype=torch.bool),
    )
    batch = Batch.from_data_list([graph])

    out = embedder.embed_batch(batch, device=torch.device("cpu"))
    assert out.edge_tokens.shape == (2, 4)

    relation_repr = embedder.relation_proj(embedder._lookup_relations(out.edge_relations, device=torch.device("cpu")))
    semantic = torch.cat(
        [
            out.question_tokens[out.edge_batch],
            out.node_tokens[out.edge_index[0]],
            relation_repr,
            out.node_tokens[out.edge_index[1]],
        ],
        dim=-1,
    )

    topic_mask = torch.zeros(out.node_tokens.size(0), dtype=torch.long)
    topic_mask[batch.start_node_locals.view(-1)] = 1
    topic_one_hot = F.one_hot(topic_mask, num_classes=embedder._num_topics).to(dtype=semantic.dtype)
    feats = [topic_one_hot]
    feats.extend(embedder._dde(topic_one_hot, out.edge_index, None))
    node_struct = torch.stack(feats, dim=-1).reshape(out.node_tokens.size(0), -1)
    struct = torch.cat([node_struct[out.edge_index[0]], node_struct[out.edge_index[1]]], dim=-1)

    fused = torch.cat([semantic, struct], dim=-1)
    expected = embedder.edge_adapter(fused)
    assert_close(out.edge_tokens, expected)


def test_graph_embedder_edge_batch_bucketize_respects_ptr_boundaries() -> None:
    g0 = Data(num_nodes=2, edge_index=torch.tensor([[1], [0]], dtype=torch.long))
    # Head index 0 in the second graph becomes exactly ptr[1] after batching (a boundary case).
    g1 = Data(num_nodes=2, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    batch = Batch.from_data_list([g0, g1])

    device = torch.device("cpu")
    edge_index = batch.edge_index.to(device)
    node_ptr = batch.ptr.to(device)
    num_graphs = int(node_ptr.numel() - 1)

    edge_batch, edge_ptr = GraphEmbedder._compute_edge_batch(
        edge_index,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
        device=device,
    )
    expected_edge_batch = batch.batch.to(device)[edge_index[0]]
    assert torch.equal(edge_batch, expected_edge_batch)
    assert edge_ptr.detach().cpu().tolist() == [0, 1, 2]
