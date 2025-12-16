#!/usr/bin/env bash
set -euo pipefail

# Rebuild normalized parquet + LMDB/embeddings for specified datasets in order.
cd "$(dirname "$0")/.."

datasets=(webqsp cwq)
for ds in "${datasets[@]}"; do
  HYDRA_FULL_ERROR=1 python scripts/build_retrieval_parquet.py dataset="$ds"
  HYDRA_FULL_ERROR=1 python scripts/build_retrieval_dataset.py dataset="$ds"
done
