#!/usr/bin/env bash
set -euo pipefail

# 便捷脚本：用给定 retriever 权重（或自动取最近一次 train_retriever_default 的 best ckpt），
# 依次在 train/validation/test 三个 split 上物化 g_agent。
# 规则：train split 强制包含 GT（force_include_gt=true），其余 split 默认为 false。
#
# 用法：
#   bash scripts/cache_g_agent_all_splits.sh <dataset> [retriever_ckpt]
#     dataset          数据集名称，对应 configs/dataset/<dataset>.yaml
#     retriever_ckpt   可选，显式指定 ckpt 路径；缺省则自动取最近一次 train_retriever_default 运行的 best ckpt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-}"
RETR_CKPT="${2:-}"

if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/cache_g_agent_all_splits.sh <dataset> [retriever_ckpt]" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi
if [[ ! -f "configs/dataset/${DATASET}.yaml" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi

latest_run_dir() {
  local task="$1"
  local base="${ROOT}/logs/${task}/runs"
  ls -1dt "${base}"/* 2>/dev/null | head -n 1
}

pick_best_ckpt() {
  local run_dir="$1"
  local ckpt_dir="${run_dir}/checkpoints"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "" && return 0
  fi
  local best
  best="$(ls -1 "${ckpt_dir}"/epoch_*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${best}" ]]; then
    echo "${best}" && return 0
  fi
  if [[ -f "${ckpt_dir}/last.ckpt" ]]; then
    echo "${ckpt_dir}/last.ckpt" && return 0
  fi
  echo ""
}

if [[ -z "${RETR_CKPT}" ]]; then
  RETR_EXP="train_retriever_default"
  RETR_RUN_DIR="$(latest_run_dir "${RETR_EXP}")"
  RETR_CKPT="$(pick_best_ckpt "${RETR_RUN_DIR:-}")"
fi

if [[ -z "${RETR_CKPT}" ]]; then
  echo "Retriever checkpoint not found. Provide it explicitly or ensure a recent train_retriever_default run exists." >&2
  exit 1
fi

echo "Using retriever checkpoint: ${RETR_CKPT}"

COMMON_OVERRIDES=("dataset=${DATASET}" "hydra.job.chdir=false")

for SPLIT in train validation test; do
  FORCE_GT="false"
  if [[ "${SPLIT}" == "train" ]]; then
    FORCE_GT="true"
  fi
  echo "==> cache_g_agent split=${SPLIT} force_include_gt=${FORCE_GT}"
  python src/eval.py \
    "${COMMON_OVERRIDES[@]}" \
    "stage=cache_g_agent" \
    "ckpt.retriever=${RETR_CKPT}" \
    "stage.split=${SPLIT}" \
    "stage.force_include_gt=${FORCE_GT}"
done
