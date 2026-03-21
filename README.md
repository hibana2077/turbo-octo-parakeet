# turbo-octo-parakeet

Project entrypoint is `main.py` with an FFTAT + JFPD UDA pipeline:

- keeps existing dataset list loader (`data/data_list_image.py`) and transforms (`utils/transform.py`)
- keeps existing logging style (python logger + script-level log redirection)
- uses `timm` for all image encoders and pretrained weights
- FFTAT components are extracted into root-level files:
  - `fftat_components.py`
  - `fftat_losses.py`

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
  --name visda_fftat_jfpd \
  --gpu_id 0 \
  --use_jfpd
```

Run `.venv/bin/python3 main.py --help` for full options.


`.venv/bin/python3 main.py   --dataset office-home   --name ac_fftat_jfpd   --source_list data/office-home/Art.txt   --target_list data/office-home/Clipart.txt   --test_list data/office-home/Clipart.txt   --num_classes 65   --img_size 256   --train_batch_size 32   --eval_batch_size 32   --max_epochs 40   --warmup_epochs 10   --log_period 50   --eval_period 1   --optimizer SGD   --learning_rate 3e-3   --momentum 0.9   --weight_decay 1e-4   --gpu_id 1   --timm_model vit_base_patch16_224.augreg2_in21k_ft_in1k`