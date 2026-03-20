#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/domainnet.txt"

JFPD_LAMBDAS=(1.0 0.1 0.01)
TRAIN_BATCH_SIZES=(128 256)
MAX_EPOCHS=40
WARMUP_EPOCHS=10
LEARNING_RATE=3e-3
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
IMG_SIZE=256
EVAL_BATCH_SIZE=128
LOG_PERIOD=50
EVAL_PERIOD=1

TASKS=(
  "ci clipart infograph"
  "cp clipart painting"
  "cq clipart quickdraw"
  "cr clipart real"
  "cs clipart sketch"
  "ic infograph clipart"
  "ip infograph painting"
  "iq infograph quickdraw"
  "ir infograph real"
  "is infograph sketch"
  "pc painting clipart"
  "pi painting infograph"
  "pq painting quickdraw"
  "pr painting real"
  "ps painting sketch"
  "qc quickdraw clipart"
  "qi quickdraw infograph"
  "qp quickdraw painting"
  "qr quickdraw real"
  "qs quickdraw sketch"
  "rc real clipart"
  "ri real infograph"
  "rp real painting"
  "rq real quickdraw"
  "rs real sketch"
  "sc sketch clipart"
  "si sketch infograph"
  "sp sketch painting"
  "sq sketch quickdraw"
  "sr sketch real"
)

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME\tCOMMAND\n' >> "$OUT_FILE"

for TASK in "${TASKS[@]}"; do
  read -r BASE_NAME SOURCE_DOMAIN TARGET_DOMAIN <<< "$TASK"

  for JFPD_LAMBDA in "${JFPD_LAMBDAS[@]}"; do
    LAMBDA_TAG="$(float_tag "$JFPD_LAMBDA")"

    for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
      RUN_NAME="${BASE_NAME}_jfpdl${LAMBDA_TAG}_tb${TRAIN_BATCH_SIZE}_me${MAX_EPOCHS}"
      CMD=".venv/bin/python3 main.py"
      CMD+=" --dataset DomainNet"
      CMD+=" --name ${RUN_NAME}"
      CMD+=" --source_list data/DomainNet/${SOURCE_DOMAIN}_train.txt"
      CMD+=" --target_list data/DomainNet/${TARGET_DOMAIN}_train.txt"
      CMD+=" --test_list data/DomainNet/${TARGET_DOMAIN}_test.txt"
      CMD+=" --num_classes 345"
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
done

COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE")"
echo "Generated ${COUNT} configs at $OUT_FILE"
