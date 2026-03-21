# turbo-octo-parakeet

Project entrypoint is `main.py` with an FFTAT + JFPD UDA pipeline:

- keeps existing dataset list loader (`data/data_list_image.py`) and transforms (`utils/transform.py`)
- keeps existing logging style (python logger + script-level log redirection)
- uses `timm` for all image encoders and pretrained weights
- FFTAT components are extracted into root-level files:
  - `fftat_components.py`
  - `fftat_losses.py`
- no runtime import from `WACV2025-FFTAT/`

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
