#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/officehome_pa.txt"

TRAIN_BATCH_SIZE_VALUES=(64 32 24)
EVAL_BATCH_SIZE=16
DATASET="office-home"
SOURCE_LIST="data/office-home/Product.txt"
TARGET_LIST="data/office-home/Art.txt"
TEST_LIST="data/office-home/Art.txt"
NUM_CLASSES=65
MODEL_TYPE="ViT-B_16"
PRETRAINED_DIR="checkpoint/imagenet21k_ViT-B_16.npz"
NUM_STEPS=10000
IMG_SIZE=256
BETA=0.1
GAMMA=0.1
THETA=0.1
LEARNING_RATE=0.06
GPU_ID=0
WARMUP_STEPS=1000
OPTIMAL=1

JFPD_LAMBDA_VALUES=(0.01 0.001)
JFPD_ALPHA=0.5
JFPD_MODE="jfpd"
BASE_NAME="pa"

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
  for JFPD_LAMBDA in "${JFPD_LAMBDA_VALUES[@]}"; do
    LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"
    RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_ns${NUM_STEPS}"
    CMD="python3 main.py"
    CMD+=" --dataset ${DATASET}"
    CMD+=" --name ${RUN_NAME}"
    CMD+=" --source_list ${SOURCE_LIST}"
    CMD+=" --target_list ${TARGET_LIST}"
    CMD+=" --test_list ${TEST_LIST}"
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
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
