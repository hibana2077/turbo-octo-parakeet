#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_FILE="${SCRIPT_DIR}/officehome_ap.txt"
SCRIPT_NAME="${SCRIPT_DIR}/officehome_ap.sh"
MAX_JOBS="${MAX_JOBS:-10}"

if [[ ! -f "$EXP_FILE" ]]; then
  echo "Missing ${EXP_FILE}; generating it now..."
  "${SCRIPT_DIR}/make_officehome_ap.sh"
fi

NUM_JOBS=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$EXP_FILE")
if [[ "$NUM_JOBS" -le 0 ]]; then
  echo "No experiments found in ${EXP_FILE}" >&2
  exit 1
fi

echo "Submitting $NUM_JOBS jobs with chunk size $MAX_JOBS..."
for ((i=0; i<NUM_JOBS; i+=MAX_JOBS)); do
  END=$((i + MAX_JOBS - 1))
  if [[ $END -ge $NUM_JOBS ]]; then
    END=$((NUM_JOBS - 1))
  fi
  qsub -J "$i-$END" "$SCRIPT_NAME"
done
