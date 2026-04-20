from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.data.retrieval.resource import DataResource
from src.models.gflownet import GFlowNetModule


_REAL_EMBEDDINGS_DIR = Path("/mnt/data/retrieval/webqsp/materialized/embeddings")
_REAL_ENTITY_METADATA = _REAL_EMBEDDINGS_DIR / "entity_metadata.pt"
_REAL_VALIDATION_LMDB = _REAL_EMBEDDINGS_DIR / "validation.lmdb"
_REAL_VALIDATION_MANIFEST = _REAL_EMBEDDINGS_DIR / "validation.manifest.pt"
_REAL_SUB_FILTER = Path("/mnt/data/retrieval/webqsp/normalized/sub_filter.json")


def _load_real_validation_slice(*, num_samples: int = 8):
    for path in (
        _REAL_EMBEDDINGS_DIR,
        _REAL_ENTITY_METADATA,
        _REAL_VALIDATION_LMDB,
        _REAL_VALIDATION_MANIFEST,
        _REAL_SUB_FILTER,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required real-data artifact missing: {path}")

    manifest = torch.load(_REAL_VALIDATION_MANIFEST, map_location="cpu")
    with open(_REAL_SUB_FILTER, "r", encoding="utf-8") as handle:
        keep_ids = set(json.load(handle)["sample_ids"])

    validation_ids = [
        sample_id
        for sample_id in manifest["sample_ids"]
        if sample_id.startswith("webqsp/validation/") and sample_id in keep_ids
    ]
    selected_ids = validation_ids[:num_samples]
    if len(selected_ids) != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} validation samples, got {len(selected_ids)}."
        )

    data_resource = DataResource(
        entity_metadata_path=_REAL_ENTITY_METADATA,
        embeddings_dir=_REAL_EMBEDDINGS_DIR,
    )
    dataset = RetrievalDataset(
        sample_ids=selected_ids,
        lmdb_paths=[_REAL_VALIDATION_LMDB],
        split="validation",
    )
    try:
        samples = [dataset.get(idx) for idx in range(len(selected_ids))]
    finally:
        dataset.close()

    batch = RetrievalCollator(data_resource)(samples)
    return batch, selected_ids


def _run_training_step_smoke(
    module: GFlowNetModule,
    *,
    batch,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    optimizer = module.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]

    captured_logs: dict[str, float] = {}

    module._trainer = SimpleNamespace(
        accumulate_grad_batches=1,
        num_training_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        lr_scheduler_configs=[],
        is_global_zero=False,
        world_size=1,
    )
    module.optimizers = lambda: optimizer
    module.manual_backward = lambda loss: loss.backward()
    module.log_dict = lambda metrics, **kwargs: captured_logs.update(
        {
            name: float(value.detach().cpu().item())
            if torch.is_tensor(value)
            else float(value)
            for name, value in metrics.items()
        }
    )

    target_param = module.policy.backbone.nbf_layers[0].fwd_msg_mlp[0].weight
    before = target_param.detach().clone()
    module.train()
    module.training_step(batch, batch_idx=0)
    after = target_param.detach().clone()
    return captured_logs, before, after


def test_real_webqsp_slice_runs_dataset_collator_and_module_paths() -> None:
    torch.manual_seed(0)
    batch, sample_ids = _load_real_validation_slice(num_samples=8)

    assert batch.num_graphs == 8
    assert len(sample_ids) == 8
    assert batch.question_emb.shape == (8, 1024)
    assert batch.node_tokens.ndim == 2
    assert batch.relation_tokens.ndim == 2
    assert batch.node_tokens.shape[1] == 1024
    assert batch.relation_tokens.shape[1] == 1024
    assert torch.isfinite(batch.question_emb).all()
    assert torch.isfinite(batch.node_tokens[~batch.non_text_node_mask]).all()
    assert torch.isfinite(batch.relation_tokens).all()

    module = GFlowNetModule(
        max_steps=1,
        num_rollout=1,
        eval_num_rollout=1,
        rollout_chunk_size=1,
        eval_rollout_chunk_size=1,
        temperature=0.7,
        backbone={
            "embedding_dim": 1024,
            "hidden_dim": 1024,
            "gnn_num_layers": 1,
            "gnn_dropout": 0.0,
        },
        policy_hidden_dim=1024,
        action_head={
            "dropout": 0.0,
        },
        reward={
            "relation_shaping_scale": 0.0,
        },
        loss={
            "reward_matching_coef": 0.5,
        },
        replay={
            "enabled": False,
        },
        optimizer_cfg={
            "type": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
            "log_z_head_lr_multiplier": 1.0,
        },
        scheduler_cfg=None,
    )

    union_masks = module.forward(batch, num_rollouts=1, temperature=0.7)
    logs, before, after = _run_training_step_smoke(module, batch=batch)

    positive_per_graph = torch.bincount(
        batch.edge_batch,
        weights=batch.positive_edge_mask.float(),
        minlength=batch.num_graphs,
    )
    stats = {
        "sample_ids": sample_ids[:3],
        "num_graphs": int(batch.num_graphs),
        "num_nodes": int(batch.num_nodes),
        "num_edges": int(batch.edge_index.size(1)),
        "mean_positive_edges": round(float(positive_per_graph.mean().item()), 4),
        "non_text_node_ratio": round(
            float(batch.non_text_node_mask.float().mean().item()),
            4,
        ),
        "union_node_ratio": round(
            float(union_masks.union_nodes.float().mean().item()),
            4,
        ),
        "union_edge_ratio": round(
            float(union_masks.union_edges.float().mean().item()),
            4,
        ),
        "train_loss": round(logs["train/loss"], 4),
        "train_log_reward_mean": round(logs["train/log_reward_mean"], 4),
        "train_traj_len_mean": round(logs["train/trajectory_length_mean"], 4),
    }
    print(stats)

    assert union_masks.union_nodes.shape == (batch.num_nodes,)
    assert union_masks.union_edges.shape == (batch.edge_index.size(1),)
    assert union_masks.union_nodes.dtype == torch.bool
    assert union_masks.union_edges.dtype == torch.bool
    assert 0.0 <= stats["union_node_ratio"] <= 1.0
    assert 0.0 <= stats["union_edge_ratio"] <= 1.0
    assert torch.isfinite(torch.tensor(logs["train/loss"]))
    assert torch.isfinite(torch.tensor(logs["train/log_reward_mean"]))
    assert logs["train/loss"] > 0.0
    assert not torch.equal(before, after)
