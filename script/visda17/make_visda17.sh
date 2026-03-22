#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/visda17.txt"

JFPD_LAMBDAS=(1.0 0.1 0.01 0.001)
TRAIN_BATCH_SIZES=(128 96 64)
EVAL_BATCH_SIZE=16
NUM_STEPS=20000
WARMUP_STEPS=1000
IMG_SIZE=256
LEARNING_RATE=0.07
BASE_NAME="visda"
DATASET="visda17"
SOURCE_LIST="./data/visda-2017/train/train_list.txt"
TARGET_LIST="./data/visda-2017/validation/validation_list.txt"
TEST_LIST="data/visda-2017/validation/validation_list.txt"
NUM_CLASSES=12
MODEL_TYPE="ViT-B_16"
PRETRAINED_DIR="checkpoint/imagenet21k_ViT-B_16.npz"
BETA=0.1
GAMMA=0.1
THETA=0.1
GPU_ID=0
OPTIMAL=1

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
  LAMBDA_TAG="$(echo "$JFPD_LAMBDA" | sed 's/-/m/g; s/\./p/g')"

  for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}"
    CMD="python3 main.py"
    CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
    CMD+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
    CMD+=" --dataset ${DATASET}"
    CMD+=" --name ${RUN_NAME}"
    CMD+=" --source_list ${SOURCE_LIST}"
    CMD+=" --target_list ${TARGET_LIST}"
    CMD+=" --test_list ${TEST_LIST}"
    CMD+=" --num_classes ${NUM_CLASSES}"
    CMD+=" --model_type ${MODEL_TYPE}"
    CMD+=" --pretrained_dir ${PRETRAINED_DIR}"
    CMD+=" --num_steps ${NUM_STEPS}"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --beta ${BETA}"
    CMD+=" --gamma ${GAMMA}"
    CMD+=" --use_im"
    CMD+=" --theta ${THETA}"
    CMD+=" --learning_rate ${LEARNING_RATE}"
    CMD+=" --gpu_id ${GPU_ID}"
    CMD+=" --use_cp"
    CMD+=" --optimal ${OPTIMAL}"
    CMD+=" --warmup_steps ${WARMUP_STEPS}"
    CMD+=" --use_jfpd"
    CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"

    printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
