#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/valfit_canonical/${RUN_ID}"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"

DIAGNOSTIC_EXPERIMENTS=(
  "debug/valfit_canonical/canon_00_db_smoke"
  "debug/valfit_canonical/canon_01_db_no_replay"
  "debug/valfit_canonical/canon_02_subtb_short"
  "debug/valfit_canonical/canon_04_stop_calib"
  "debug/valfit_canonical/canon_06_replay_strong"
  "debug/valfit_canonical/canon_07_frontier_corr05"
)

ALL_EXPERIMENTS=(
  "${DIAGNOSTIC_EXPERIMENTS[@]}"
  "debug/valfit_canonical/canon_08_frontier_corr10"
  "debug/valfit_canonical/canon_03_subtb_full"
  "debug/valfit_canonical/canon_05_stop_margin"
  "debug/valfit_canonical/canon_09_compact_loose"
  "debug/valfit_canonical/canon_10_compact_strong"
  "debug/valfit_canonical/canon_11_budget6_db"
  "debug/valfit_canonical/canon_12_aggressive_full"
)

CURRENT_EXP=""
CURRENT_LOG=""

if [[ "${1:-}" == "--all" ]]; then
  shift
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
else
  EXPERIMENTS=("${DIAGNOSTIC_EXPERIMENTS[@]}")
fi

classify_failure() {
  local exit_code="$1"
  local log_file="$2"

  if [[ "${exit_code}" -eq 0 ]]; then
    printf "ok"
  elif [[ "${exit_code}" -eq 137 ]]; then
    printf "killed_or_host_oom_exit_137"
  elif [[ "${exit_code}" -eq 143 ]]; then
    printf "terminated_exit_143"
  elif grep -Eiq "out of memory|CUDA error:.*memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA OOM|Killed" "${log_file}"; then
    printf "cuda_or_host_oom"
  elif grep -Eiq "KeyboardInterrupt|SIGINT" "${log_file}"; then
    printf "interrupted"
  elif grep -Eiq "RuntimeError|Traceback|Exception" "${log_file}"; then
    printf "python_exception"
  else
    printf "failed_exit_${exit_code}"
  fi
}

record_interruption() {
  local signal_name="$1"
  if [[ -n "${CURRENT_EXP}" ]]; then
    local now
    now="$(date --iso-8601=seconds)"
    printf "%s\t%s\t%s\t%s\t%s\n" "${now}" "${CURRENT_EXP}" "interrupted" "${signal_name}" "${CURRENT_LOG}" >> "${SUMMARY_FILE}"
  fi
  exit 128
}

trap 'record_interruption SIGINT' INT
trap 'record_interruption SIGTERM' TERM

mkdir -p "${LOG_DIR}"
printf "finished_at\texperiment\tstatus\treason\tlog_file\n" > "${SUMMARY_FILE}"

cd "${ROOT_DIR}" || exit 1

for exp in "${EXPERIMENTS[@]}"; do
  name="$(basename "${exp}")"
  log_file="${LOG_DIR}/${name}.log"
  CURRENT_EXP="${exp}"
  CURRENT_LOG="${log_file}"

  {
    printf "started_at=%s\n" "$(date --iso-8601=seconds)"
    printf "experiment=%s\n" "${exp}"
    printf "gpu=1\n"
    printf "command=PYTHONPATH=%s CUDA_VISIBLE_DEVICES=1 python src/train.py experiment=%s %s\n" "${ROOT_DIR}" "${exp}" "$*"
  } | tee "${log_file}"

  PYTHONPATH="${ROOT_DIR}" CUDA_VISIBLE_DEVICES=1 python src/train.py "experiment=${exp}" "$@" 2>&1 | tee -a "${log_file}"
  exit_code="${PIPESTATUS[0]}"

  reason="$(classify_failure "${exit_code}" "${log_file}")"
  finished_at="$(date --iso-8601=seconds)"
  if [[ "${exit_code}" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  printf "finished_at=%s\nexit_code=%s\nreason=%s\n" "${finished_at}" "${exit_code}" "${reason}" | tee -a "${log_file}"
  printf "%s\t%s\t%s\t%s\t%s\n" "${finished_at}" "${exp}" "${status}" "${reason}" "${log_file}" >> "${SUMMARY_FILE}"

  CURRENT_EXP=""
  CURRENT_LOG=""
done

printf "summary=%s\n" "${SUMMARY_FILE}"
