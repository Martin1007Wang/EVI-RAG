#!/usr/bin/env bash
set -euo pipefail

PID="869249"
CHECK_SECS="300"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Monitoring PID ${PID} every ${CHECK_SECS}s..."
while ps -p "${PID}" > /dev/null 2>&1; do
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") PID ${PID} still running."
  sleep "${CHECK_SECS}"
done

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") PID ${PID} ended. Starting cwq-sub mpm training."
cd "${ROOT_DIR}"
python src/train.py experiment=train_mpm_rag dataset=cwq-sub model.env_cfg.max_steps=7
