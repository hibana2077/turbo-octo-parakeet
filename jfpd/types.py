from dataclasses import dataclass
from typing import Optional


@dataclass
class EpochStats:
    loss: float
    accuracy: Optional[float] = None
    d_feat: Optional[float] = None
    d_pred: Optional[float] = None
    psi: Optional[float] = None
    phi: Optional[float] = None
    source_loss: Optional[float] = None
    used_samples: Optional[int] = None
    skipped_samples: Optional[int] = None
    filtered_by_confidence: Optional[int] = None
    filtered_by_class_cap: Optional[int] = None
