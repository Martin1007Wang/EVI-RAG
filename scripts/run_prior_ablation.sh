#!/usr/bin/env bash
set -euo pipefail

# Run retriever prior ablations sequentially. Extra args are passed to train.py.

EXPERIMENTS=(
  "train_gflownet_prior_alpha0"
  "train_gflownet_prior_alpha2_to0"
  "train_gflownet_prior_alpha2_to0p5"
)

LOG_DIR="logs/prior_ablation"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

for EXP in "${EXPERIMENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${EXP}_${TS}.log"
  echo "==> Running ${EXP}, logging to ${LOG_FILE}"
  python src/train.py experiment="${EXP}" "$@" 2>&1 | tee "${LOG_FILE}"
done
