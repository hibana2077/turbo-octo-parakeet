#!/bin/bash
set -euo pipefail

NUM_JOBS=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' cb.txt)
if [[ "$NUM_JOBS" -le 0 ]]; then
  echo "No experiments found in cb.txt" >&2
  exit 1
fi

echo "Submitting $NUM_JOBS jobs to the cluster..."

SCRIPT_NAME="CDUN4.sh"
MAX_JOBS=10
for ((i=0; i<NUM_JOBS; i+=MAX_JOBS)); do
  END=$((i + MAX_JOBS - 1))
  if [[ $END -ge $NUM_JOBS ]]; then
    END=$((NUM_JOBS - 1))
  fi
  qsub -J "$i-$END" "$SCRIPT_NAME"
done
