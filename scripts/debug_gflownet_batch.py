"""
Standalone g_agent batch inspector to sanity-check PyG offsets and schema.

Usage:
    python scripts/debug_gflownet_batch.py \\
        --cache /mnt/data/retrieval_dataset/webqsp/materialized/g_agent/train_g_agent.pt \\
        --batch-size 2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.loader import DataLoader


def _load_g_agent_module() -> object:
    """Load src/data/g_agent_dataset.py without importing the whole package (avoids Hydra)."""
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "src" / "data" / "g_agent_dataset.py"
    spec = importlib.util.spec_from_file_location("g_agent_dataset_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g_agent_dataset_local"] = module
    spec.loader.exec_module(module)
    return module


def _edge_ptr(edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    counts = torch.bincount(edge_batch, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return ptr


def inspect_batch(batch, *, batch_idx: int) -> None:
    num_graphs = int(batch.num_graphs)
    node_ptr: torch.Tensor = batch.ptr
    edge_batch = batch.batch[batch.edge_index[0]]
    edge_ptr = _edge_ptr(edge_batch, num_graphs)

    print(f"\n=== Batch {batch_idx} ===")
    print(f"[Batch] num_graphs={num_graphs}, num_edges={batch.edge_index.size(1)}, num_nodes={batch.node_global_ids.numel()}")
    print(f" node_ptr={node_ptr[: num_graphs + 1].tolist()}")
    print(f" edge_ptr={edge_ptr.tolist()}")

    for g in range(num_graphs):
        ns, ne = int(node_ptr[g].item()), int(node_ptr[g + 1].item())
        es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
        nodes = batch.node_global_ids[ns:ne]
        mask_e = edge_batch == g
        heads = batch.edge_index[0, mask_e] - ns
        tails = batch.edge_index[1, mask_e] - ns
        rels = batch.edge_attr[mask_e]

        start_ptr = batch._slice_dict["start_node_locals"]
        s0, s1 = int(start_ptr[g].item()), int(start_ptr[g + 1].item())
        start_locals = batch.start_node_locals[s0:s1] - ns

        ans_ptr = batch._slice_dict["answer_node_locals"]
        a0, a1 = int(ans_ptr[g].item()), int(ans_ptr[g + 1].item())
        answer_locals = batch.answer_node_locals[a0:a1] - ns

        gt_ptr = batch._slice_dict["gt_path_edge_local_ids"]
        g0, g1 = int(gt_ptr[g].item()), int(gt_ptr[g + 1].item())
        gt_edges = batch.gt_path_edge_local_ids[g0:g1]

        print(f"\n[Graph {g}] nodes={ne - ns}, edges={ee - es}")
        print(f"  node_global_ids[{ns}:{ne}] (first 8): {nodes[:8].tolist()}")
        print(f"  edge heads/tails rel (first 8):")
        for i in range(min(8, heads.numel())):
            h_local = int(heads[i].item())
            t_local = int(tails[i].item())
            rel = int(rels[i].item())
            h_global = int(nodes[h_local].item()) if 0 <= h_local < nodes.numel() else -1
            t_global = int(nodes[t_local].item()) if 0 <= t_local < nodes.numel() else -1
            score = float(batch.edge_scores[es + i].item())
            label = float(batch.edge_labels[es + i].item())
            print(f"    e#{i:02d}: {h_local}->{t_local} (glob {h_global}->{t_global}) rel={rel} score={score:.4g} label={label:.3g}")

        def _check_locals(name: str, locals_idx: torch.Tensor) -> Tuple[bool, str]:
            if locals_idx.numel() == 0:
                return False, f"{name}: EMPTY"
            bad = (locals_idx < 0) | (locals_idx >= (ne - ns))
            if bad.any():
                return True, f"{name}: OUT_OF_RANGE {locals_idx.tolist()}"
            mapped = nodes[locals_idx]
            return False, f"{name}: locals={locals_idx.tolist()} → globals={mapped.tolist()}"

        for label_name, locals_idx in [
            ("start_node_locals", start_locals),
            ("answer_node_locals", answer_locals),
        ]:
            bad, msg = _check_locals(label_name, locals_idx)
            print(f"  {msg}{'  <== BAD' if bad else ''}")

        if gt_edges.numel() > 0:
            bad_gt = (gt_edges < es) | (gt_edges >= ee)
            triples = []
            pos_flags = []
            for e_local in gt_edges.tolist():
                rel_idx = e_local - es
                h_idx = int(batch.edge_index[0, e_local].item())
                t_idx = int(batch.edge_index[1, e_local].item())
                h_local = h_idx - ns
                t_local = t_idx - ns
                h_global = int(batch.node_global_ids[h_idx].item()) if 0 <= h_local < nodes.numel() else -1
                t_global = int(batch.node_global_ids[t_idx].item()) if 0 <= t_local < nodes.numel() else -1
                rel = int(batch.edge_attr[e_local].item()) if 0 <= rel_idx < (ee - es) else -1
                label = float(batch.edge_labels[e_local].item())
                triples.append((h_local, rel, t_local, h_global, rel, t_global))
                pos_flags.append(label)
            print(f"  gt_path_edge_local_ids (raw slice)={gt_edges.tolist()}{'  <== BAD' if bad_gt.any() else ''}")
            for idx, (h_l, r_l, t_l, h_g, r_g, t_g) in enumerate(triples):
                lbl = pos_flags[idx]
                print(f"    gt#{idx}: local {h_l}->{t_l} rel={r_l} | global {h_g}->{t_g} rel={r_g} label={lbl:.3g}")
        else:
            print("  gt_path_edge_local_ids: EMPTY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True, help="Path to *_g_agent.pt cache.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--drop-unreachable", action="store_true")
    parser.add_argument("--num-batches", type=int, default=4, help="How many batches to inspect.")
    args = parser.parse_args()

    module = _load_g_agent_module()
    DatasetCls = getattr(module, "GAgentPyGDataset")

    dataset = DatasetCls(args.cache, drop_unreachable=args.drop_unreachable)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for b_idx, batch in enumerate(loader):
        if b_idx >= args.num_batches:
            break
        inspect_batch(batch, batch_idx=b_idx)


if __name__ == "__main__":
    main()
