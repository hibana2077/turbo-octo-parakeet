#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_FILE="${SCRIPT_DIR}/visda17.txt"
FFTAT_FILE="${PROJECT_ROOT}/fftat_script.txt"

# JFPD best params (ours)
JFPD_LAMBDAS=(0.01)
JFPD_ALPHA=0.5
JFPD_MODE="jfpd"

# Overrides requested by us
TRAIN_BATCH_SIZES=(128 64 60 52 48)
NUM_STEPS=1200
EVAL_EVERY=25
GPU_ID=0

TASKS=("visda")

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

for BASE_NAME in "${TASKS[@]}"; do
  BASELINE_LINE="$(awk -v name="$BASE_NAME" '
    $0 ~ /^python3 main\.py/ &&
    $0 ~ /--dataset visda17/ &&
    $0 ~ ("--name " name "([[:space:]]|$)") {
      print
      exit
    }' "$FFTAT_FILE")"

  if [[ -z "$BASELINE_LINE" ]]; then
    echo "Missing visda17 baseline for task: $BASE_NAME" >&2
    exit 1
  fi

  EVAL_BATCH_SIZE="$(extract_arg "$BASELINE_LINE" "eval_batch_size")"
  SOURCE_LIST="$(extract_arg "$BASELINE_LINE" "source_list")"
  TARGET_LIST="$(extract_arg "$BASELINE_LINE" "target_list")"
  TEST_LIST="$(extract_arg "$BASELINE_LINE" "test_list")"
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

  for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
      LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"
      RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_ns${NUM_STEPS}"
      CMD="python3 main.py"
      CMD+=" --dataset visda17"
      CMD+=" --name ${RUN_NAME}"
      CMD+=" --source_list ${SOURCE_LIST}"
      CMD+=" --target_list ${TARGET_LIST}"
      CMD+=" --test_list ${TEST_LIST}"
      CMD+=" --num_classes ${NUM_CLASSES}"
      CMD+=" --img_size ${IMG_SIZE}"
      CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
      CMD+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
      CMD+=" --eval_every ${EVAL_EVERY}"
      CMD+=" --num_steps ${NUM_STEPS}"
      CMD+=" --warmup_steps ${WARMUP_STEPS}"
      CMD+=" --learning_rate ${LEARNING_RATE}"
      CMD+=" --gpu_id ${GPU_ID}"
      CMD+=" --model_type ${MODEL_TYPE}"
      CMD+=" --pretrained_dir ${PRETRAINED_DIR}"
      CMD+=" --beta ${BETA}"
      CMD+=" --gamma ${GAMMA}"

      if [[ "$BASELINE_LINE" =~ (^|[[:space:]])--use_im([[:space:]]|$) ]]; then
        CMD+=" --use_im"
      fi
      CMD+=" --theta ${THETA}"
      if [[ "$BASELINE_LINE" =~ (^|[[:space:]])--use_cp([[:space:]]|$) ]]; then
        CMD+=" --use_cp"
      fi
      CMD+=" --optimal ${OPTIMAL}"
      CMD+=" --use_jfpd"
      CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
      CMD+=" --jfpd_alpha ${JFPD_ALPHA}"
      CMD+=" --jfpd_mode ${JFPD_MODE}"

      printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
    done
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
