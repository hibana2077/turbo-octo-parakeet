# turbo-octo-parakeet

Version B implementation of the JFPD pipeline described in [docs/imp_guide.md](/home/timelab/Desktop/codes/turbo-octo-parakeet/docs/imp_guide.md), with support for both Hugging Face DomainNet and a local OfficeHome directory tree.

The entrypoint is [train_jfpd.py](/home/timelab/Desktop/codes/turbo-octo-parakeet/train_jfpd.py). It runs:

1. Source-domain supervised pretraining with cross-entropy.
2. Dynamic source prototype estimation during adaptation, resampling source examples each iteration.
3. Target-domain adaptation with the JFPD loss.
4. Evaluation on the target test split.

DomainNet example:

```bash
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
python train_jfpd.py \
  --source-domain real \
  --target-domain sketch \
  --source-epochs 5 \
  --adapt-epochs 5 \
  --batch-size 64 \
  --proto-samples-per-class 32 \
  --output-dir outputs/real_to_sketch
```

OfficeHome example:

```bash
python train_jfpd.py \
  --dataset-name office_home \
  --dataset-root OfficeHomeDataset \
  --source-domain Art \
  --target-domain Product \
  --train-split-ratio 0.8 \
  --source-epochs 5 \
  --adapt-epochs 5 \
  --batch-size 64 \
  --output-dir outputs/art_to_product
```

Notes:

- The virtualenv in this workspace currently does not have the required packages installed yet.
- `--source-domain` and `--target-domain` are dataset-specific.
- DomainNet domains are `clipart`, `infograph`, `painting`, `quickdraw`, `real`, and `sketch`.
- OfficeHome domains are `Art`, `Clipart`, `Product`, and `Real World`.
- DomainNet uses the Hugging Face `train` / `test` splits from `wltjr1007/DomainNet`.
- OfficeHome is loaded from `--dataset-root` and split deterministically per class within each domain using `--train-split-ratio`.
- Version B resamples `K` source examples per class on every adaptation iteration. The default is `--proto-samples-per-class 32`.
