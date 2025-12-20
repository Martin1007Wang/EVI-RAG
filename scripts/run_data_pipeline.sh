#!/usr/bin/env bash
set -euo pipefail

# Data-only pipeline: build normalized parquet + LMDB caches.
#
# Usage:
#   bash scripts/run_data_pipeline.sh <dataset>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/run_data_pipeline.sh <dataset>" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi
if [[ ! -f "configs/dataset/${DATASET}.yaml" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi

export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -d /dev/shm ]]; then
  if ! (touch /dev/shm/_evi_rag_shm_test 2>/dev/null); then
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
  else
    rm -f /dev/shm/_evi_rag_shm_test 2>/dev/null || true
  fi
fi

COMMON_OVERRIDES=("dataset=${DATASET}" "hydra.job.chdir=false")

echo "==> [1/2] build_retrieval_parquet (${DATASET})"
python scripts/build_retrieval_parquet.py "${COMMON_OVERRIDES[@]}"

echo "==> [2/2] build_retrieval_dataset (${DATASET})"
python scripts/build_retrieval_dataset.py "${COMMON_OVERRIDES[@]}"

echo "Data pipeline finished."
