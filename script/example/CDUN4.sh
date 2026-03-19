#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=scratch/cp23+gdata/yp87
#PBS -r y

module load cuda/12.6.2

set -euo pipefail

TAG="CDUN4"
METHOD="udtw"
LEARNING_TYPE="unsupervised"

PROJECT_ROOT="../.."
EXP_FILE="./cb.txt"
CSV_PATH="./${TAG}.csv"
CSV_LOCK_PATH="${CSV_PATH}.lock"

IDX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
LINE_NO=$((IDX + 1))
LINE="$(awk -v n="$LINE_NO" 'NF && $1 !~ /^#/ {i++; if (i==n) {print; exit}}' "$EXP_FILE")"
read -r RUN_NAME TRAIN_DATASET EVAL_DATASET N_WAYS K_SHOTS TRAIN_EPISODES SDTW_GAMMA UDTW_BETA TOKEN_L2_NORMALIZE DTW_PATH_NORMALIZE UNSUP_BATCH_SIZE UNSUP_TEMPERATURE <<< "${LINE:-}"

if [[ -z "${RUN_NAME:-}" ]]; then
  echo "No experiment config found for index ${IDX}" >&2
  exit 1
fi

case "$TRAIN_DATASET" in
  miniimagenet|omniglot)
    ;;
  *)
    echo "Unsupported train dataset: $TRAIN_DATASET" >&2
    exit 1
    ;;
esac

case "$EVAL_DATASET" in
  cub_200_2011|stanford_cars)
    ;;
  *)
    echo "Unsupported eval dataset: $EVAL_DATASET" >&2
    exit 1
    ;;
esac

source /scratch/cp23/lw4988/HEHE/.venv/bin/activate
export HF_HOME="/scratch/cp23/lw4988/hf_home"
export HF_HUB_OFFLINE=1

TOKEN_L2_NORMALIZE_FLAG=(--no_token_l2_normalize)
if [[ "$TOKEN_L2_NORMALIZE" == "yes" ]]; then
  TOKEN_L2_NORMALIZE_FLAG=(--token_l2_normalize)
fi

DTW_PATH_NORMALIZE_FLAG=(--no_dtw_path_normalize)
if [[ "$DTW_PATH_NORMALIZE" == "yes" ]]; then
  DTW_PATH_NORMALIZE_FLAG=(--dtw_path_normalize)
fi

cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --train_dataset "$TRAIN_DATASET"
  --eval_dataset "$EVAL_DATASET"
  --n_ways "$N_WAYS"
  --method "$METHOD"
  --learning_type "$LEARNING_TYPE"
  --train_split train
  --eval_split test
  --k_shots "$K_SHOTS"
  --q_shots 15
  --proj_dim 256
  --eval_episodes 600
  --sigma_net mlp
  --sigma_mode learned
  "${TOKEN_L2_NORMALIZE_FLAG[@]}"
  "${DTW_PATH_NORMALIZE_FLAG[@]}"
  --unsup_batch_size "$UNSUP_BATCH_SIZE"
  --unsup_temperature "$UNSUP_TEMPERATURE"
  --device cuda
  --save_summary
  --proj_bias
)

CKPT_PATH="runs/${TAG}_${RUN_NAME}.pt"
SUMMARY_PATH="summary/${TAG}_${IDX}_${RUN_NAME}.json"
LOG_PATH="./script/${TAG}/${TAG}_${IDX}_${RUN_NAME}.log"

python3 -u -m src.transfer_main \
  "${COMMON_ARGS[@]}" \
  --train_episodes "$TRAIN_EPISODES" \
  --sdtw_gamma "$SDTW_GAMMA" \
  --udtw_beta "$UDTW_BETA" \
  --proj_ckpt "$CKPT_PATH" \
  --summary_path "$SUMMARY_PATH" \
  >> "$LOG_PATH" 2>&1

CSV_ROW="$(python3 - "$SUMMARY_PATH" "$RUN_NAME" "$TRAIN_DATASET" "$EVAL_DATASET" "$N_WAYS" "$K_SHOTS" "$METHOD" "$TRAIN_EPISODES" "$SDTW_GAMMA" "$UDTW_BETA" "$TOKEN_L2_NORMALIZE" "$DTW_PATH_NORMALIZE" "$LEARNING_TYPE" "$UNSUP_BATCH_SIZE" "$UNSUP_TEMPERATURE" <<'PY'
import csv
import io
import json
import sys

(
    summary_path,
    run_name,
    train_dataset,
    eval_dataset,
    n_ways,
    k_shots,
    method,
    train_episodes,
    sdtw_gamma,
    udtw_beta,
    token_l2_normalize,
    dtw_path_normalize,
    learning_type,
    unsup_batch_size,
    unsup_temperature,
) = sys.argv[1:16]

with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

eval_block = summary.get("eval", {})
acc = eval_block.get("acc", "")
ci95 = eval_block.get("ci95", "")

buffer = io.StringIO()
writer = csv.writer(buffer, lineterminator="")
writer.writerow([
    run_name,
    train_dataset,
    eval_dataset,
    n_ways,
    k_shots,
    method,
    train_episodes,
    sdtw_gamma,
    udtw_beta,
    token_l2_normalize,
    dtw_path_normalize,
    learning_type,
    unsup_batch_size,
    unsup_temperature,
    acc,
    ci95,
])
print(buffer.getvalue())
PY
)"

touch "$CSV_LOCK_PATH"
{
  flock 200
  if [[ ! -s "$CSV_PATH" ]]; then
    printf 'run_name,train_dataset,eval_dataset,n_ways,k_shots,method,train_episodes,sdtw_gamma,udtw_beta,token_l2_normalize,dtw_path_normalize,learning_type,unsup_batch_size,unsup_temperature,acc,ci95\n' > "$CSV_PATH"
  fi
  printf '%s\n' "$CSV_ROW" >> "$CSV_PATH"
} 200>"$CSV_LOCK_PATH"
