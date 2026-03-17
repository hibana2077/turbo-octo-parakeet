import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .types import EpochStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_stats(prefix: str, stats: EpochStats) -> None:
    payload = asdict(stats)
    payload = {key: value for key, value in payload.items() if value is not None}
    rendered = ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"{key}={value}"
        for key, value in payload.items()
    )
    print(f"{prefix}: {rendered}")


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(path: Path, model: nn.Module, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **payload}, path)
