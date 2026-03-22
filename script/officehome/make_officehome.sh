#!/bin/bash
set -euo pipefail

SCRIPT_DIR="."
OUT_FILE="./officehome.txt"

JFPD_LAMBDA=0.0005
TRAIN_BATCH_SIZE=48
NUM_CLASSES=65
MODEL_TYPE="ViT-B_16"
PRETRAINED_DIR="checkpoint/imagenet21k_ViT-B_16.npz"
NUM_STEPS=2000
IMG_SIZE=256
BETA=0.1
GAMMA=0.01
THETA=0.01
LEARNING_RATE=0.06
GPU_ID=0
WARMUP_STEPS=1000
OPTIMAL=1
LOG_PERIOD=50
EVAL_PERIOD=25
EVAL_BATCH_SIZE=32


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

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TASK in "${TASKS[@]}"; do
  read -r BASE_NAME SOURCE_DOMAIN TARGET_DOMAIN <<< "$TASK"
  LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"
  RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_ns${NUM_STEPS}"
  CMD="python3 main.py"
  CMD+=" --dataset office-home"
  CMD+=" --name ${RUN_NAME}"
  CMD+=" --source_list data/office-home/${SOURCE_DOMAIN}.txt"
  CMD+=" --target_list data/office-home/${TARGET_DOMAIN}.txt"
  CMD+=" --test_list data/office-home/${TARGET_DOMAIN}.txt"
  CMD+=" --num_classes 65"
  CMD+=" --img_size ${IMG_SIZE}"
  CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
  CMD+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
  CMD+=" --num_steps ${NUM_STEPS}"
  CMD+=" --warmup_steps ${WARMUP_STEPS}"
  CMD+=" --log_period ${LOG_PERIOD}"
  CMD+=" --eval_period ${EVAL_PERIOD}"
  CMD+=" --optimizer SGD"
  CMD+=" --learning_rate ${LEARNING_RATE}"
  CMD+=" --gpu_id 0"
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
  CMD+=" --jfpd_mode jfpd"

  printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
