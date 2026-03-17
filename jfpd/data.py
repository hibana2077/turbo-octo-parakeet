import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms

from .config import DOMAINNET_DOMAINS, IMAGENET_MEAN, IMAGENET_STD


def build_transform(image_size: int, is_train: bool) -> transforms.Compose:
    ops: List[transforms.Compose] = [transforms.Resize((image_size, image_size))]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def decode_image(image_value) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path") is not None:
            return Image.open(image_value["path"]).convert("RGB")
    if isinstance(image_value, str):
        return Image.open(image_value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(image_value)!r}")


class HFDatasetAdapter(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        image_size: int,
        is_train: bool,
    ) -> None:
        self.dataset = dataset
        self.transform = build_transform(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[index]
        image = decode_image(example["image"])
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(example["label"], dtype=torch.long),
        }


def build_class_index(dataset: Dataset, num_classes: int) -> List[List[int]]:
    class_to_indices = [[] for _ in range(num_classes)]
    for index, label in enumerate(dataset["label"]):
        class_to_indices[int(label)].append(index)
    return class_to_indices


class DynamicPrototypeSource:
    def __init__(self, dataset: Dataset, image_size: int, num_classes: int) -> None:
        self.adapter = HFDatasetAdapter(dataset=dataset, image_size=image_size, is_train=False)
        self.class_to_indices = build_class_index(dataset=dataset, num_classes=num_classes)
        self.valid_mask = torch.tensor([len(indices) > 0 for indices in self.class_to_indices], dtype=torch.bool)
        self.valid_class_ids = [index for index, is_valid in enumerate(self.valid_mask.tolist()) if is_valid]

    def sample_pairs(self, class_ids: List[int], samples_per_class: int) -> List[Tuple[int, int]]:
        sampled_pairs: List[Tuple[int, int]] = []
        for class_id in class_ids:
            indices = self.class_to_indices[class_id]
            if not indices:
                continue
            if len(indices) >= samples_per_class:
                sampled_indices = random.sample(indices, samples_per_class)
            else:
                sampled_indices = random.choices(indices, k=samples_per_class)
            sampled_pairs.extend((class_id, sample_index) for sample_index in sampled_indices)
        return sampled_pairs

    def build_pixel_batch(self, indices: List[int]) -> torch.Tensor:
        return torch.stack([self.adapter[index]["pixel_values"] for index in indices], dim=0)


def build_loader(dataset: Dataset, image_size: int, batch_size: int, num_workers: int, is_train: bool) -> DataLoader:
    wrapped = HFDatasetAdapter(dataset=dataset, image_size=image_size, is_train=is_train)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_class_names(dataset: Dataset, column_name: str, fallback: Optional[List[str]] = None) -> List[str]:
    feature = dataset.features[column_name]
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer class names for column '{column_name}'.")


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def load_domainnet_splits(
    dataset_name: str,
    cache_dir: Optional[str],
    source_domain: str,
    target_domain: str,
    limits: Dict[str, Optional[int]],
) -> Dict[str, Dataset]:
    train_split = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    test_split = load_dataset(dataset_name, split="test", cache_dir=cache_dir)

    domain_names = get_class_names(train_split, "domain", fallback=list(DOMAINNET_DOMAINS))
    source_domain_id = domain_names.index(source_domain)
    target_domain_id = domain_names.index(target_domain)

    source_train = train_split.filter(lambda example: example["domain"] == source_domain_id)
    target_train = train_split.filter(lambda example: example["domain"] == target_domain_id)
    target_test = test_split.filter(lambda example: example["domain"] == target_domain_id)
    source_test = test_split.filter(lambda example: example["domain"] == source_domain_id)

    source_train = maybe_limit_dataset(source_train, limits["source_train"])
    target_train = maybe_limit_dataset(target_train, limits["target_train"])
    target_test = maybe_limit_dataset(target_test, limits["target_test"])
    source_test = maybe_limit_dataset(source_test, limits["source_test"])

    return {
        "source_train": source_train,
        "target_train": target_train,
        "target_test": target_test,
        "source_test": source_test,
    }
