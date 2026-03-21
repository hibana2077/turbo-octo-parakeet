#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/officehome_ac.txt"

OPTIMIZER="SGD"
LEARNING_RATE_VALUES=(1e-2)
WEIGHT_DECAY_VALUES=(5e-4 1e-3)
SPLIT_LAYER_VALUES=(4 8)
TG_LAYER_VALUES=(1 2)
LAMBDA_DIS_VALUES=(0.01 0.1)
LAMBDA_PAT_VALUES=(0.03 0.3)
LAMBDA_SC_VALUES=(0.003)
JFPD_LAMBDA_VALUES=(0.1 1.0)
PSEUDO_THRESHOLD_VALUES=(0.5 0.7)
TRAIN_BATCH_SIZE_VALUES=(64 128)

MAX_EPOCHS=40
WARMUP_EPOCHS=10
MOMENTUM=0.9
IMG_SIZE=256
EVAL_BATCH_SIZE=128
LOG_PERIOD=50
EVAL_PERIOD=1

SOURCE_DOMAIN="Art"
TARGET_DOMAIN="Clipart"
BASE_NAME="ac_fftat_jfpd"
GPU_ID=0
TIMM_MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
  for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
    LR_TAG="$(float_tag "$LEARNING_RATE")"

    for WEIGHT_DECAY in "${WEIGHT_DECAY_VALUES[@]}"; do
      WD_TAG="$(float_tag "$WEIGHT_DECAY")"

      for SPLIT_LAYER in "${SPLIT_LAYER_VALUES[@]}"; do
        for TG_LAYER in "${TG_LAYER_VALUES[@]}"; do
          for LAMBDA_DIS in "${LAMBDA_DIS_VALUES[@]}"; do
            DIS_TAG="$(float_tag "$LAMBDA_DIS")"

            for LAMBDA_PAT in "${LAMBDA_PAT_VALUES[@]}"; do
              PAT_TAG="$(float_tag "$LAMBDA_PAT")"

              for LAMBDA_SC in "${LAMBDA_SC_VALUES[@]}"; do
                SC_TAG="$(float_tag "$LAMBDA_SC")"

                for JFPD_LAMBDA in "${JFPD_LAMBDA_VALUES[@]}"; do
                  JFPD_TAG="$(float_tag "$JFPD_LAMBDA")"

                  for PSEUDO_THRESHOLD in "${PSEUDO_THRESHOLD_VALUES[@]}"; do
                    PSEUDO_TAG="$(float_tag "$PSEUDO_THRESHOLD")"

                    RUN_NAME="${BASE_NAME}_tb${TRAIN_BATCH_SIZE}_optsgd_lr${LR_TAG}_wd${WD_TAG}_sl${SPLIT_LAYER}_tg${TG_LAYER}_ld${DIS_TAG}_lp${PAT_TAG}_ls${SC_TAG}_jl${JFPD_TAG}_pth${PSEUDO_TAG}_me${MAX_EPOCHS}"
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
                    CMD+=" --optimizer ${OPTIMIZER}"
                    CMD+=" --learning_rate ${LEARNING_RATE}"
                    CMD+=" --momentum ${MOMENTUM}"
                    CMD+=" --weight_decay ${WEIGHT_DECAY}"
                    CMD+=" --gpu_id ${GPU_ID}"
                    CMD+=" --timm_model ${TIMM_MODEL}"
                    CMD+=" --split_layer ${SPLIT_LAYER}"
                    CMD+=" --tg_layers ${TG_LAYER}"
                    CMD+=" --lambda_dis ${LAMBDA_DIS}"
                    CMD+=" --lambda_pat ${LAMBDA_PAT}"
                    CMD+=" --lambda_sc ${LAMBDA_SC}"
                    CMD+=" --use_jfpd"
                    CMD+=" --jfpd_lambda ${JFPD_LAMBDA}"
                    CMD+=" --pseudo_threshold ${PSEUDO_THRESHOLD}"

                    printf '%s\t%s\n' "$RUN_NAME" "$CMD" >> "$OUT_FILE"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
