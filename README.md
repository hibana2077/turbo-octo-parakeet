# turbo-octo-parakeet

This project now uses the FFTAT pipeline layout directly at repository root:

- `main.py`
- `models/`
- `utils/`
- `data/`

`jfpd` is retained as a lightweight loss component (`jfpd/losses.py`) and is integrated into FFTAT training.

## Run

```bash
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 main.py --help
```

## JFPD options in FFTAT

- `--use_jfpd`: enable JFPD regularization.
- `--jfpd_lambda`: JFPD loss weight.
- `--jfpd_alpha`: JFPD `alpha` in `[0, 1]`.
- `--jfpd_mode`: `jfpd` / `fgpd` / `pgfd`.
