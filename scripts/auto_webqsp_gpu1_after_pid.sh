#!/usr/bin/env bash
set -euo pipefail

PID="${1:-4086637}"
GPU_ID="${2:-1}"
POLL_SECONDS="${POLL_SECONDS:-300}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Avoid permission issues when matplotlib (via torchmetrics/lightning) tries to write cache under $HOME.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"
mkdir -p "$MPLCONFIGDIR"

ts() { date +"%F %T"; }

echo "[$(ts)] Monitoring PID=$PID (poll=${POLL_SECONDS}s)."
while ps -p "$PID" >/dev/null 2>&1; do
  echo "[$(ts)] PID $PID still running; sleeping ${POLL_SECONDS}s..."
  sleep "$POLL_SECONDS"
done
echo "[$(ts)] PID $PID finished. Starting pipeline on GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)."

echo "[$(ts)] (1/3) train_gflownet on webqsp-sub"
python src/train.py experiment=train_gflownet dataset=webqsp-sub

RUNS_DIR="$ROOT_DIR/logs/train_gflownet_webqsp-sub/runs"
if [[ ! -d "$RUNS_DIR" ]]; then
  echo "[$(ts)] ERROR: runs dir not found: $RUNS_DIR" >&2
  exit 1
fi

LATEST_RUN="$(ls -1dt "$RUNS_DIR"/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_RUN}" || ! -d "${LATEST_RUN}" ]]; then
  echo "[$(ts)] ERROR: no run dirs found under: $RUNS_DIR" >&2
  exit 1
fi

CKPT_DIR="$LATEST_RUN/checkpoints"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[$(ts)] ERROR: checkpoint dir not found: $CKPT_DIR" >&2
  exit 1
fi

CKPT_PATH=""
if compgen -G "$CKPT_DIR/epoch_*.ckpt" >/dev/null; then
  CKPT_PATH="$(ls -1t "$CKPT_DIR"/epoch_*.ckpt | head -n 1)"
elif [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  CKPT_PATH="$CKPT_DIR/last.ckpt"
elif compgen -G "$CKPT_DIR/*.ckpt" >/dev/null; then
  CKPT_PATH="$(ls -1t "$CKPT_DIR"/*.ckpt | head -n 1)"
fi

if [[ -z "$CKPT_PATH" || ! -f "$CKPT_PATH" ]]; then
  echo "[$(ts)] ERROR: no checkpoint found under: $CKPT_DIR" >&2
  exit 1
fi

echo "[$(ts)] Using checkpoint: $CKPT_PATH"

echo "[$(ts)] (2/3) eval_gflownet (writes textual rollouts under /mnt/data/.../artifacts/*/eval_gflownet)"
python src/eval.py experiment=eval_gflownet dataset=webqsp-sub "ckpt.gflownet=$CKPT_PATH"

echo "[$(ts)] GFlowNet rollout artifacts (textualized):"
echo "  - /mnt/data/retrieval_dataset/webqsp/artifacts/webqsp/eval_gflownet/test.jsonl"
echo "  - /mnt/data/retrieval_dataset/webqsp/artifacts/webqsp-sub/eval_gflownet/test.jsonl"

echo "[$(ts)] (3/3) eval_llm on webqsp (vLLM on GPU)"
python src/eval.py experiment=eval_llm dataset=webqsp llm.provider=vllm llm.resume=false

echo "[$(ts)] Done."

