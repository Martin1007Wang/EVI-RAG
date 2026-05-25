from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.training.factory import build_model, prepare_training_components


def _mem(label: str) -> None:
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 2**20
    reserved = torch.cuda.memory_reserved() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"{label}: allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB peak={peak:.1f}MiB", flush=True)


def _patch_successor_probe(model) -> None:
    encoder = model.policy.forest_encoder
    original = encoder.encode_successor_actions

    def wrapped_encode_successor_actions(**kwargs):
        edge_ids = kwargs["frontier_edge_ids"]
        row_ids = kwargs["frontier_row_ids"]
        print(
            "encode_successor_actions " f"actions={int(edge_ids.numel())} rows={int(row_ids.unique().numel())}",
            flush=True,
        )
        _mem("  before encode_successor_actions")
        out = original(**kwargs)
        _mem("  after encode_successor_actions")
        return out

    encoder.encode_successor_actions = wrapped_encode_successor_actions


def _batch_stats(batch, label: str) -> None:
    deg = torch.bincount(batch.edge_index[0].detach().cpu(), minlength=batch.num_nodes_total)
    print(
        f"{label}: graphs={batch.num_graphs_total} N={batch.num_nodes_total} E={batch.num_edges_total} "
        f"anchors={int(batch.anchor_node_ids.numel())} reachable={int(batch.reachable_target_node_ids.numel())} "
        f"max_out_degree={int(deg.max().item()) if deg.numel() else 0}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=86)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="/mnt/data/retrieval")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process.")

    config_dir = str((Path.cwd() / "configs").resolve())
    overrides = [
        "experiment=debug/valfit",
        "logger=none",
        "trainer=cpu",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        f"paths.data_dir={args.data_dir}",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=overrides)

    dm, resources = prepare_training_components(cfg, stage="fit")
    model = build_model(cfg, resources).cuda().train()
    _patch_successor_probe(model)
    _mem("after model.cuda")

    torch.cuda.reset_peak_memory_stats()
    batch = dm.collator([dm.train_dataset[int(args.idx)]]).cuda()
    _batch_stats(batch, f"sample idx={args.idx}")
    _mem("after batch.cuda")

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.bf16))

    with torch.no_grad(), autocast:
        rollout_features = model.feature_encoder(batch)
        _mem("after rollout feature_encoder")
        rollout_context = model.runner.engine.prepare_context(batch=batch, features=rollout_features)
        _mem("after prepare_context")
        reward_context = model.reward_model.prepare_context(
            batch,
            budget=model.budget,
        )
        _mem("after reward_context")
        chunk = model.runner.train_chunk(
            policy=model.policy,
            batch=batch,
            context=rollout_context,
            num_samples=int(args.num_samples),
            temperature=model.train_temperature,
        )
        _mem("after train_chunk")

    transitions = chunk.transitions
    print("transitions", 0 if transitions is None else transitions.num_transitions, flush=True)
    if transitions is not None:
        _batch_stats_like_state(transitions.parent_state, "parent_state")
        _batch_stats_like_state(transitions.child_state, "child_state")

    with autocast:
        staged_batch = model.feature_encoder.stage_batch(batch)
        _mem("after stage_batch")
        output = model._forward_chunk(
            chunk=chunk,
            staged_batch=staged_batch,
            rollout_context=rollout_context,
            reward_context=reward_context,
        )
    _mem("after forward_chunk")
    output.loss.backward()
    _mem("after backward")


def _batch_stats_like_state(state, label: str) -> None:
    node_bytes = state.node_mask.numel() * state.node_mask.element_size()
    edge_bytes = state.edge_mask.numel() * state.edge_mask.element_size()
    print(
        f"{label}: rows={state.num_rollouts} node_mask={tuple(state.node_mask.shape)} "
        f"edge_mask={tuple(state.edge_mask.shape)} dense_mask_MiB={(node_bytes + edge_bytes) / 2**20:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
