#!/bin/bash
#PBS -P cp23
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/cp23+gdata/yp87
#PBS -r y

module load cuda/12.6.2

set -euo pipefail

export HF_HOME="/scratch/cp23/lw4988/hf_home"
export HF_HUB_OFFLINE=1

TAG="officehome_ac_jfpd_sweep"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_FILE="${SCRIPT_DIR}/officehome_ac.txt"

IDX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
LINE_NO=$((IDX + 1))
LINE="$(awk -v n="$LINE_NO" 'NF && $1 !~ /^#/ {i++; if (i==n) {print; exit}}' "$EXP_FILE")"

if [[ -z "${LINE:-}" ]]; then
  echo "No experiment config found for index ${IDX}" >&2
  exit 1
fi

RUN_NAME="${LINE%%$'\t'*}"
COMMAND="${LINE#*$'\t'}"

if [[ "$RUN_NAME" == "$LINE" || -z "${COMMAND:-}" ]]; then
  echo "Malformed config line at index ${IDX}: ${LINE}" >&2
  exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

LOG_PATH="${LOG_DIR}/${TAG}_${IDX}_${RUN_NAME}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN_NAME=${RUN_NAME}" | tee -a "$LOG_PATH"
echo "COMMAND: ${COMMAND}" | tee -a "$LOG_PATH"

bash -lc "$COMMAND" >> "$LOG_PATH" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${RUN_NAME}" | tee -a "$LOG_PATH"
