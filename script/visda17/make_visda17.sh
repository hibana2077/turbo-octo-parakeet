#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_FILE="${PROJECT_ROOT}/jfpd_script.txt"
OUT_FILE="${SCRIPT_DIR}/visda17.txt"

JFPD_LAMBDAS=(1.0 0.1 0.01)
TRAIN_BATCH_SIZES=(16 32)
NUM_STEPS=1500

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE" >&2
  exit 1
fi

mapfile -t BASE_COMMANDS < <(
  awk '
    BEGIN {in_section=0}
    /^# VisDA-2017/ {in_section=1; next}
    /^# DomainNet/ {in_section=0}
    in_section && $1 ~ /\.venv\/bin\/python3/ && $0 ~ /--dataset visda17/ {print}
  ' "$INPUT_FILE"
)

if [[ "${#BASE_COMMANDS[@]}" -eq 0 ]]; then
  echo "No VisDA-2017 commands found in $INPUT_FILE" >&2
  exit 1
fi

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for BASE_CMD in "${BASE_COMMANDS[@]}"; do
  BASE_NAME="$(sed -n 's/.*--name \([^ ]*\).*/\1/p' <<< "$BASE_CMD")"
  if [[ -z "${BASE_NAME:-}" ]]; then
    echo "Failed to parse --name from command: $BASE_CMD" >&2
    exit 1
  fi

  for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
    LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"
    for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
      RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_ns${NUM_STEPS}"

      CMD="$BASE_CMD"
      CMD="$(sed -E "s/--jfpd_lambda [^ ]+/--jfpd_lambda ${JFPD_LAMBDA}/" <<< "$CMD")"
      CMD="$(sed -E "s/--num_steps [^ ]+/--num_steps ${NUM_STEPS}/" <<< "$CMD")"
      CMD="$(sed -E "s/--train_batch_size [^ ]+/--train_batch_size ${TRAIN_BATCH_SIZE}/" <<< "$CMD")"
      CMD="$(sed -E "s/--name [^ ]+/--name ${RUN_NAME}/" <<< "$CMD")"

      printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
    done
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
