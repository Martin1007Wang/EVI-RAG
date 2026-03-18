from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datasets.graph_retrieval_dataset import create_graph_retrieval_dataset
from src.data.preprocess.labels.edge_retrieval import compute_shortest_path_labels


def _resolve_output_dir(cfg: DictConfig) -> Path:
    out = cfg.get("output_dir")
    if out:
        return Path(str(out))
    dataset_cfg = cfg.get("dataset") or {}
    artifact_dir = Path(str(dataset_cfg.get("artifact_dir")))
    return artifact_dir / "edge_retrieval_labels"


def _build_split(
    cfg: DictConfig, *, split: str, output_dir: Path, overwrite: bool
) -> Path:
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is None:
        raise ValueError("Missing config group: dataset=<name> (e.g., webqsp-sub).")
    split = str(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.pt"
    if out_path.exists() and not overwrite:
        return out_path

    dataset = create_graph_retrieval_dataset(
        cfg=dataset_cfg, split_name=split, resources=None
    )
    entries: Dict[str, Dict[str, Any]] = {}

    no_path = 0
    zero_hop = 0
    for idx in tqdm(range(len(dataset)), desc=f"labels/{split}"):
        data = dataset.get(idx)
        sample_id = str(getattr(data, "sample_id", ""))
        if not sample_id:
            continue
        labels = compute_shortest_path_labels(
            edge_index=torch.as_tensor(data.edge_index, dtype=torch.long),
            q_local_indices=torch.as_tensor(data.q_local_indices, dtype=torch.long),
            a_local_indices=torch.as_tensor(data.a_local_indices, dtype=torch.long),
            num_nodes=int(data.num_nodes),
        )
        if labels.max_path_length is None:
            no_path += 1
        elif int(labels.max_path_length) == 0:
            zero_hop += 1
        entries[sample_id] = {
            "num_edges": int(labels.num_edges),
            "positive_edge_ids": labels.positive_edge_ids,
            "max_path_length": labels.max_path_length,
        }

    payload = {
        "meta": {
            "algo": "edge_retrieval_shortest_paths_strict_v1",
            "split": split,
            "num_samples": int(len(entries)),
            "no_path_samples": int(no_path),
            "zero_hop_samples": int(zero_hop),
        },
        "entries": entries,
    }
    torch.save(payload, out_path)
    return out_path


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="build_edge_retrieval_labels.yaml",
)
def main(cfg: DictConfig) -> None:
    splits = cfg.get("splits") or ["train", "validation", "test"]
    splits = [str(s) for s in list(splits)]
    overwrite = bool(cfg.get("overwrite", False))
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"outputs": {}, "splits": splits}
    for split in splits:
        path = _build_split(
            cfg, split=split, output_dir=output_dir, overwrite=overwrite
        )
        manifest["outputs"][split] = str(path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[ok] wrote labels to {output_dir}")


if __name__ == "__main__":
    main()
