#!/bin/bash
set -euo pipefail

SCRIPT_DIR="./"
OUT_FILE="${SCRIPT_DIR}/officehome_rp.txt"

JFPD_LAMBDAS=(1.0 0.1 0.01 0.0)
PSEUDO_THRESHOLDS=(0.9 0.6 0.0)
TRAIN_BATCH_SIZE=128

MAX_EPOCHS=40
WARMUP_EPOCHS=10
LEARNING_RATE=3e-3
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
IMG_SIZE=256
EVAL_BATCH_SIZE=128
LOG_PERIOD=50
EVAL_PERIOD=1

SOURCE_DOMAIN="Real_World"
TARGET_DOMAIN="Product"
BASE_NAME="rp_fftat_jfpd"
GPU_ID=0
TIMM_MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
  LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"

  for PSEUDO_THRESHOLD in "${PSEUDO_THRESHOLDS[@]}"; do
    PSEUDO_TAG="$(float_tag "$PSEUDO_THRESHOLD")"

    RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_pth${PSEUDO_TAG}_tb${TRAIN_BATCH_SIZE}_me${MAX_EPOCHS}"
    CMD=".venv/bin/python3 main.py"
    CMD+=" --dataset office-home"
    CMD+=" --name ${RUN_NAME}"
    CMD+=" --source_list data/office-home/${SOURCE_DOMAIN}.txt"
    CMD+=" --target_list data/office-home/${TARGET_DOMAIN}.txt"
    CMD+=" --test_list data/office-home/${TARGET_DOMAIN}.txt"
    CMD+=" --num_classes 65"
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
    CMD+=" --gpu_id ${GPU_ID}"
    CMD+=" --timm_model ${TIMM_MODEL}"
    CMD+=" --use_jfpd"
    CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
    CMD+=" --pseudo_threshold ${PSEUDO_THRESHOLD}"

    printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
