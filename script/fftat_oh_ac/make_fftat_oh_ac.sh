#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/fftat_oh_ac.txt"

LAMBDA_DIS_VALUES=(1.0 0.1 0.01)
LAMBDA_PAT_VALUES=(1.0 0.1 0.01)
LAMBDA_SC_VALUES=(0.1 0.01)
TRAIN_BATCH_SIZES=(32 64)

MAX_EPOCHS=200
WARMUP_EPOCHS=10
LEARNING_RATE=3e-3
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
IMG_SIZE=256
EVAL_BATCH_SIZE=128
LOG_PERIOD=50
EVAL_PERIOD=1

SOURCE_DOMAIN="Art"
TARGET_DOMAIN="Clipart"
BASE_NAME="fftat_oh_ac"
GPU_ID=0
TIMM_MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
  for LAMBDA_DIS in "${LAMBDA_DIS_VALUES[@]}"; do
    DIS_TAG="$(float_tag "$LAMBDA_DIS")"

    for LAMBDA_PAT in "${LAMBDA_PAT_VALUES[@]}"; do
      PAT_TAG="$(float_tag "$LAMBDA_PAT")"

      for LAMBDA_SC in "${LAMBDA_SC_VALUES[@]}"; do
        SC_TAG="$(float_tag "$LAMBDA_SC")"

        RUN_NAME="${BASE_NAME}_ld${DIS_TAG}_lp${PAT_TAG}_ls${SC_TAG}_tb${TRAIN_BATCH_SIZE}_me${MAX_EPOCHS}"
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
        CMD+=" --lambda_dis ${LAMBDA_DIS}"
        CMD+=" --lambda_pat ${LAMBDA_PAT}"
        CMD+=" --lambda_sc ${LAMBDA_SC}"

        printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
      done
    done
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
