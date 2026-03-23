#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/domainnet_ci.txt"

TRAIN_BATCH_SIZE_VALUES=(40 45 50)
JFPD_LAMBDA_VALUES=(0.008 0.02 0.04)
JFPD_ALPHA=0.5

EVAL_BATCH_SIZE=16
DATASET="DomainNet"
BASE_NAME="domainnet_ci"
SOURCE_LIST="data/DomainNet/clipart_train.txt"
TARGET_LIST="data/DomainNet/infograph_train.txt"
TEST_LIST="data/DomainNet/infograph_test.txt"
NUM_CLASSES=345
MODEL_TYPE="ViT-B_16"
PRETRAINED_DIR="checkpoint/imagenet21k_ViT-B_16.npz"
NUM_STEPS=20000
IMG_SIZE=256
BETA=0.1
GAMMA=0.01
THETA=0.1
LEARNING_RATE=0.07
PERTURBATION_RATIO=0.3
GPU_ID=0
OPTIMAL=1
WARMUP_STEPS=2000

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
  for JFPD_LAMBDA in "${JFPD_LAMBDA_VALUES[@]}"; do
    JFPD_LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"

    RUN_NAME="${BASE_NAME}_tb${TRAIN_BATCH_SIZE}_jl${JFPD_LAMBDA_TAG}"
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
    CMD+=" --perturbationRatio ${PERTURBATION_RATIO}"
    CMD+=" --gpu_id ${GPU_ID}"
    CMD+=" --use_cp"
    CMD+=" --optimal ${OPTIMAL}"
    CMD+=" --warmup_steps ${WARMUP_STEPS}"
    CMD+=" --use_jfpd"
    CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
    CMD+=" --jfpd_alpha ${JFPD_ALPHA}"
    CMD+=" --jfpd_mode jfpd"

    printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
