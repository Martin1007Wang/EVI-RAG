import json
import math
import random
from pathlib import Path
import torch

root = Path("/mnt/wangjingxiong/EVI-RAG")
emb_dir = Path("/mnt/data/retrieval/webqsp/materialized/embeddings")
sub_filter_path = Path("/mnt/data/retrieval/webqsp/normalized/sub_filter.json")
out_dir = root / "tmp" / "filters"
out_dir.mkdir(parents=True, exist_ok=True)

manifest = torch.load(emb_dir / "train.manifest.pt", map_location="cpu")
with open(sub_filter_path, "r", encoding="utf-8") as f:
    keep_ids = set(json.load(f)["sample_ids"])

records = []
for idx, (sample_id, num_edges, num_nodes) in enumerate(
    zip(
        manifest["sample_ids"],
        manifest["num_edges"].tolist(),
        manifest["num_nodes"].tolist(),
    )
):
    if sample_id in keep_ids:
        records.append((idx, sample_id, int(num_edges), int(num_nodes)))

records.sort(key=lambda x: (x[2], x[3], x[1]))
num_bins = 4
bin_size = math.ceil(len(records) / num_bins)
bins = [records[i * bin_size : (i + 1) * bin_size] for i in range(num_bins)]

def write_subset(target_size: int, seed: int = 42) -> None:
    rng = random.Random(seed + target_size)
    chosen = []
    base = target_size // num_bins
    rem = target_size % num_bins
    for bin_idx, bucket in enumerate(bins):
        need = base + (1 if bin_idx < rem else 0)
        if len(bucket) < need:
            raise RuntimeError(
                f"Bin {bin_idx} has only {len(bucket)} samples, cannot draw {need}."
            )
        picks = rng.sample(bucket, need)
        chosen.extend(picks)

    chosen.sort(key=lambda x: x[0])
    sample_ids = [sample_id for _, sample_id, _, _ in chosen]
    out_path = out_dir / f"webqsp_sub_train_size_balanced_{target_size}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "webqsp",
                "split": "train",
                "balance": "num_edges_quartiles",
                "sample_ids": sample_ids,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(out_path)
    print(f"count={len(sample_ids)}")

for n in (32, 128):
    write_subset(n)