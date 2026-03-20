#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_FILE="${PROJECT_ROOT}/fftat_script.txt"
OUT_FILE="${SCRIPT_DIR}/fftatoh.txt"
TIMM_MODEL="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE" >&2
  exit 1
fi

mapfile -t BASE_COMMANDS < <(
  awk '
    BEGIN {in_section=0}
    /^# Office-Home/ {in_section=1; next}
    /^# Office-31/ {in_section=0}
    in_section && $1 ~ /^python3$/ && $0 ~ /--dataset office-home/ {print}
  ' "$INPUT_FILE"
)

if [[ "${#BASE_COMMANDS[@]}" -eq 0 ]]; then
  echo "No Office-Home FFTAT commands found in $INPUT_FILE" >&2
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

  NUM_STEPS="$(sed -n 's/.*--num_steps \([^ ]*\).*/\1/p' <<< "$BASE_CMD")"
  if [[ -z "${NUM_STEPS:-}" ]]; then
    NUM_STEPS="unknown"
  fi

  RUN_NAME="${BASE_NAME}_fftat_ns${NUM_STEPS}"

  CMD="$BASE_CMD"
  CMD="$(sed -E 's#^python3 #.venv/bin/python3 #' <<< "$CMD")"
  CMD="$(sed -E 's/--gpu_id[[:space:]]*([0-9]+)--use_cp/--gpu_id \1 --use_cp/g' <<< "$CMD")"
  CMD="$(sed -E "s/--model_type [^ ]+[[:space:]]+--pretrained_dir [^ ]+/--timm_model ${TIMM_MODEL}/" <<< "$CMD")"
  CMD="$(sed -E 's/--gpu_id[[:space:]]+[^ ]+/--gpu_id 0/' <<< "$CMD")"
  CMD="$(sed -E "s/--name [^ ]+/--name ${RUN_NAME}/" <<< "$CMD")"

  CMD="$(sed -E 's/[[:space:]]+--use_jfpd//g; s/[[:space:]]+--jfpd_lambda [^ ]+//g; s/[[:space:]]+--jfpd_alpha [^ ]+//g; s/[[:space:]]+--jfpd_mode [^ ]+//g' <<< "$CMD")"

  if [[ "$CMD" != *"--eval_every "* ]]; then
    CMD+=" --eval_every 50"
  fi
  if [[ "$CMD" != *"--disable_best_acc_cache"* ]]; then
    CMD+=" --disable_best_acc_cache"
  fi

  printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
