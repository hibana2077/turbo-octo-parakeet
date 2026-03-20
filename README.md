# turbo-octo-parakeet

Project entrypoint is now CDTrans-based and locally vendored:

- root [main.py](/home/timelab/Desktop/codes/turbo-octo-parakeet/main.py): CDTrans UDA training loop
- local core modules: `cdtrans_core/` (config/model/loss/solver)
- keeps existing dataset list loader (`data/data_list_image.py`) and transforms (`utils/transform.py`)
- integrates optional JFPD loss (`jfpd/losses.py`)

`main.py` no longer imports from `CDTrans/` at runtime.

## Install

```bash
.venv/bin/python3 -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python3 main.py \
  --dataset visda17 \
  --source_list data/visda-2017/train/train_list.txt \
  --target_list data/visda-2017/validation/validation_list.txt \
  --test_list data/visda-2017/validation/validation_list.txt \
  --num_classes 12 \
  --name visda_cdtrans_jfpd \
  --gpu_id 0 \
  --use_jfpd
```

## New CLI (high level)

- CDTrans backbone: `--transformer_type --block_pattern --pretrain_choice --pretrained_path`
- training: `--max_epochs --optimizer --learning_rate --warmup_epochs`
- CDTrans UDA losses: `--target_loss_weight --distill_weight --pseudo_threshold`
- JFPD: `--use_jfpd --jfpd_lambda --jfpd_alpha --jfpd_mode`

Run `.venv/bin/python3 main.py --help` for full options.
