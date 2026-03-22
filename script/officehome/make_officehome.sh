#!/bin/bash
set -euo pipefail

SCRIPT_DIR="."
PROJECT_ROOT="../.."
OUT_FILE="${SCRIPT_DIR}/officehome.txt"
FFTAT_FILE="${PROJECT_ROOT}/fftat_script.txt"

# JFPD best params (ours)
JFPD_LAMBDA=0.0001
JFPD_ALPHA=0.5
JFPD_MODE="jfpd"

# Overrides requested by us
NUM_STEPS=2000
GPU_ID=0


TASKS=(
  "ac Art Clipart"
  "ap Art Product"
  "ar Art Real_World"
  "ca Clipart Art"
  "cp Clipart Product"
  "cr Clipart Real_World"
  "pa Product Art"
  "pc Product Clipart"
  "pr Product Real_World"
  "ra Real_World Art"
  "rc Real_World Clipart"
  "rp Real_World Product"
)

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

extract_arg() {
  local line="$1"
  local key="$2"
  local re="--${key}[[:space:]]+([^[:space:]]+)"
  if [[ "$line" =~ $re ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

if [[ ! -f "$FFTAT_FILE" ]]; then
  echo "Cannot find baseline file: $FFTAT_FILE" >&2
  exit 1
fi

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TASK in "${TASKS[@]}"; do
  read -r BASE_NAME SOURCE_DOMAIN TARGET_DOMAIN <<< "$TASK"

  BASELINE_LINE="$(awk -v name="$BASE_NAME" '
    $0 ~ /^python3 main\.py/ &&
    $0 ~ /--dataset office-home/ &&
    $0 ~ ("--name " name "([[:space:]]|$)") {
      print
      exit
    }' "$FFTAT_FILE")"

  if [[ -z "$BASELINE_LINE" ]]; then
    echo "Missing office-home baseline for task: $BASE_NAME" >&2
    exit 1
  fi

  TRAIN_BATCH_SIZE="64"  # Fixed based on user-provided FFTAT command
  EVAL_BATCH_SIZE="$(extract_arg "$BASELINE_LINE" "eval_batch_size")"
  NUM_CLASSES="$(extract_arg "$BASELINE_LINE" "num_classes")"
  MODEL_TYPE="$(extract_arg "$BASELINE_LINE" "model_type")"
  PRETRAINED_DIR="$(extract_arg "$BASELINE_LINE" "pretrained_dir")"
  IMG_SIZE="$(extract_arg "$BASELINE_LINE" "img_size")"
  BETA="$(extract_arg "$BASELINE_LINE" "beta")"
  GAMMA="$(extract_arg "$BASELINE_LINE" "gamma")"
  THETA="$(extract_arg "$BASELINE_LINE" "theta")"
  LEARNING_RATE="$(extract_arg "$BASELINE_LINE" "learning_rate")"
  WARMUP_STEPS="$(extract_arg "$BASELINE_LINE" "warmup_steps")"
  OPTIMAL="$(extract_arg "$BASELINE_LINE" "optimal")"

  LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"
  RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_ns${NUM_STEPS}"
  CMD="python3 main.py"
  CMD+=" --dataset office-home"
  CMD+=" --name ${RUN_NAME}"
  CMD+=" --source_list data/office-home/${SOURCE_DOMAIN}.txt"
  CMD+=" --target_list data/office-home/${TARGET_DOMAIN}.txt"
  CMD+=" --test_list data/office-home/${TARGET_DOMAIN}.txt"
  CMD+=" --num_classes ${NUM_CLASSES}"
  CMD+=" --img_size ${IMG_SIZE}"
  CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
  CMD+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
  CMD+=" --num_steps ${NUM_STEPS}"
  CMD+=" --warmup_steps ${WARMUP_STEPS}"
  CMD+=" --learning_rate ${LEARNING_RATE}"
  CMD+=" --gpu_id ${GPU_ID}"
  CMD+=" --model_type ${MODEL_TYPE}"
  CMD+=" --pretrained_dir ${PRETRAINED_DIR}"
  CMD+=" --beta ${BETA}"
  CMD+=" --gamma ${GAMMA}"
  CMD+=" --use_im"
  CMD+=" --theta ${THETA}"
  CMD+=" --use_cp"
  CMD+=" --optimal ${OPTIMAL}"
  CMD+=" --use_jfpd"
  CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
  CMD+=" --jfpd_alpha ${JFPD_ALPHA}"
  CMD+=" --jfpd_mode ${JFPD_MODE}"

  printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
