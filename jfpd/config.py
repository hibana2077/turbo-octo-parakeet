from dataclasses import dataclass
from typing import Optional

from .losses import LossMode


DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real World")
OFFICEHOME_DATASET_ALIASES = ("officehome", "office_home", "office-home")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class JFPDConfig:
    dataset_name: str = "wltjr1007/DomainNet"
    dataset_root: Optional[str] = None
    cache_dir: Optional[str] = None
    source_domain: str = "real"
    target_domain: str = "sketch"
    train_split_ratio: float = 0.8
    model_name: str = "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    source_epochs: int = 5
    adapt_epochs: int = 5
    source_lr: float = 1e-4
    adapt_lr: float = 5e-5
    weight_decay: float = 1e-4
    alpha: float = 0.5
    loss_mode: LossMode = "jfpd"
    proto_samples_per_class: int = 32
    proto_forward_batch_size: int = 256
    seed: int = 42
    device: Optional[str] = None
    output_dir: str = "outputs/jfpd"
    max_source_train_samples: Optional[int] = None
    max_target_train_samples: Optional[int] = None
    max_target_test_samples: Optional[int] = None
    max_source_test_samples: Optional[int] = None
    class_limit: Optional[int] = None
    eval_source: bool = False


def is_officehome_dataset(dataset_name: str) -> bool:
    normalized = dataset_name.strip().lower()
    return normalized in OFFICEHOME_DATASET_ALIASES
