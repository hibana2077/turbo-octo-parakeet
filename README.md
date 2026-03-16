# turbo-octo-parakeet

Version B implementation of the JFPD pipeline described in [docs/imp_guide.md](/home/timelab/Desktop/codes/turbo-octo-parakeet/docs/imp_guide.md), wired for the Hugging Face dataset `wltjr1007/DomainNet`.

The entrypoint is [train_jfpd.py](/home/timelab/Desktop/codes/turbo-octo-parakeet/train_jfpd.py). It runs:

1. Source-domain supervised pretraining with cross-entropy.
2. Dynamic source prototype estimation during adaptation, resampling source examples each iteration.
3. Target-domain adaptation with the JFPD loss.
4. Evaluation on the target test split.

Example:

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

Notes:

- The virtualenv in this workspace currently does not have the required packages installed yet.
- The Hugging Face dataset exposes `train` and `test` splits, with a `domain` class label covering `clipart`, `infograph`, `painting`, `quickdraw`, `real`, and `sketch`.
- Version B resamples `K` source examples per class on every adaptation iteration. The default is `--proto-samples-per-class 32`.
