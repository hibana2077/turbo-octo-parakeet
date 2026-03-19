#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${SCRIPT_DIR}/cb.txt"

TRAIN_DATASET="omniglot"
EVAL_DATASETS=(cub_200_2011 stanford_cars)
N_WAYS=(5 20)
K_SHOTS=(1 5)
EPIS=(500 1000)
UNSUP_BATCH_SIZES=(16 24 32)
UNSUP_TEMPERATURES=(1.0 0.1)
GAMMA="100"
BETA="0.01"
NORMALIZE_OPTIONS="no"

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

: > "$OUT_FILE"
printf '# RUN_NAME TRAIN_DATASET EVAL_DATASET N_WAYS K_SHOTS TRAIN_EPISODES SDTW_GAMMA UDTW_BETA TOKEN_L2_NORMALIZE DTW_PATH_NORMALIZE UNSUP_BATCH_SIZE UNSUP_TEMPERATURE\n' >> "$OUT_FILE"

GAMMA_TAG="$(float_tag "$GAMMA")"
BETA_TAG="$(float_tag "$BETA")"

for EVAL_DATASET in "${EVAL_DATASETS[@]}"; do
  for N_WAY in "${N_WAYS[@]}"; do
    for K_SHOT in "${K_SHOTS[@]}"; do
      for EPI in "${EPIS[@]}"; do
        for UNSUP_BATCH_SIZE in "${UNSUP_BATCH_SIZES[@]}"; do
          for UNSUP_TEMPERATURE in "${UNSUP_TEMPERATURES[@]}"; do
            TEMP_TAG="$(float_tag "$UNSUP_TEMPERATURE")"
            for TOKEN_L2_NORMALIZE in "${NORMALIZE_OPTIONS[@]}"; do
              for DTW_PATH_NORMALIZE in "${NORMALIZE_OPTIONS[@]}"; do
                RUN_NAME="${TRAIN_DATASET}_to_${EVAL_DATASET}_w${N_WAY}_s${K_SHOT}_udtw_te${EPI}_ub${UNSUP_BATCH_SIZE}_ut${TEMP_TAG}_g${GAMMA_TAG}_b${BETA_TAG}_tl2${TOKEN_L2_NORMALIZE}_dpn${DTW_PATH_NORMALIZE}"
                printf '%s %s %s %s %s %s %s %s %s %s %s %s\n' \
                  "$RUN_NAME" "$TRAIN_DATASET" "$EVAL_DATASET" "$N_WAY" "$K_SHOT" "$EPI" "$GAMMA" "$BETA" "$TOKEN_L2_NORMALIZE" "$DTW_PATH_NORMALIZE" "$UNSUP_BATCH_SIZE" "$UNSUP_TEMPERATURE" \
                  >> "$OUT_FILE"
              done
            done
          done
        done
      done
    done
  done
done

echo "Generated $(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$OUT_FILE") configs at $OUT_FILE"
