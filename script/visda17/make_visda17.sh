#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/visda17.txt"

JFPD_LAMBDAS=(1.0 0.1 0.01)
TRAIN_BATCH_SIZES=(128 256)
MAX_EPOCHS=40
WARMUP_EPOCHS=10
LEARNING_RATE=5e-5
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
IMG_SIZE=256
EVAL_BATCH_SIZE=128
LOG_PERIOD=50
EVAL_PERIOD=1

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
  LAMBDA_TAG="$(echo "$JFPD_LAMBDA" | sed 's/-/m/g; s/\./p/g')"

  for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    RUN_NAME="visda_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_me${MAX_EPOCHS}"
    CMD=".venv/bin/python3 main.py"
    CMD+=" --dataset visda17"
    CMD+=" --name ${RUN_NAME}"
    CMD+=" --source_list data/visda-2017/train/train_list.txt"
    CMD+=" --target_list data/visda-2017/validation/validation_list.txt"
    CMD+=" --test_list data/visda-2017/validation/validation_list.txt"
    CMD+=" --num_classes 12"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
    CMD+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
    CMD+=" --max_epochs ${MAX_EPOCHS}"
    CMD+=" --warmup_epochs ${WARMUP_EPOCHS}"
    CMD+=" --log_period ${LOG_PERIOD}"
    CMD+=" --eval_period ${EVAL_PERIOD}"
    CMD+=" --optimizer SGD"
    CMD+=" --learning_rate ${LEARNING_RATE}"
    CMD+=" --momentum ${MOMENTUM}"
    CMD+=" --weight_decay ${WEIGHT_DECAY}"
    CMD+=" --gpu_id 0"
    CMD+=" --use_jfpd"
    CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
    CMD+=" --jfpd_alpha 0.5"
    CMD+=" --jfpd_mode jfpd"

    printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
