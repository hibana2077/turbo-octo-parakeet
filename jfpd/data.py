import random
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import ClassLabel, Dataset, Features, Image as HFImage, Value, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms

from .config import DOMAINNET_DOMAINS, IMAGENET_MEAN, IMAGENET_STD, OFFICEHOME_DOMAINS, is_officehome_dataset


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


def _sample_class_ids(shared_class_ids: Sequence[int], class_limit: Optional[int], seed: int) -> Optional[List[int]]:
    if class_limit is None:
        return None
    if class_limit > len(shared_class_ids):
        raise ValueError(
            f"class_limit={class_limit} exceeds the number of shared classes available ({len(shared_class_ids)})."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(list(shared_class_ids), class_limit))


def _filter_dataset_to_classes(dataset: Dataset, class_ids: Sequence[int]) -> Dataset:
    class_id_set = set(class_ids)
    selected_indices = [index for index, label in enumerate(dataset["label"]) if int(label) in class_id_set]
    filtered = dataset.select(selected_indices)

    original_names = get_class_names(dataset, "label")
    label_mapping = {class_id: new_id for new_id, class_id in enumerate(class_ids)}
    updated_features = deepcopy(filtered.features)
    updated_features["label"] = ClassLabel(names=[original_names[class_id] for class_id in class_ids])

    return filtered.map(
        lambda example: {"label": label_mapping[int(example["label"])]},
        features=updated_features,
    )


def maybe_limit_classes(
    splits: Dict[str, Dataset],
    class_limit: Optional[int],
    seed: int,
) -> Dict[str, Dataset]:
    if class_limit is None:
        return splits

    shared_class_ids = sorted(set.intersection(*(set(int(label) for label in dataset["label"]) for dataset in splits.values())))
    selected_class_ids = _sample_class_ids(shared_class_ids, class_limit=class_limit, seed=seed)
    if selected_class_ids is None:
        return splits

    return {
        split_name: _filter_dataset_to_classes(dataset, selected_class_ids)
        for split_name, dataset in splits.items()
    }


def validate_domain_names(dataset_name: str, source_domain: str, target_domain: str) -> None:
    if is_officehome_dataset(dataset_name):
        valid_domains = OFFICEHOME_DOMAINS
    else:
        valid_domains = DOMAINNET_DOMAINS

    invalid_domains = [domain for domain in (source_domain, target_domain) if domain not in valid_domains]
    if invalid_domains:
        joined = ", ".join(valid_domains)
        raise ValueError(f"Invalid domain(s) for dataset '{dataset_name}': {invalid_domains}. Expected one of: {joined}")


def load_domainnet_splits(
    dataset_name: str,
    cache_dir: Optional[str],
    source_domain: str,
    target_domain: str,
    limits: Dict[str, Optional[int]],
    class_limit: Optional[int],
    seed: int,
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

    splits = {
        "source_train": source_train,
        "target_train": target_train,
        "target_test": target_test,
        "source_test": source_test,
    }
    return maybe_limit_classes(splits, class_limit=class_limit, seed=seed)


def _build_image_dataset(image_paths: Sequence[str], labels: Sequence[int], label_names: Sequence[str]) -> Dataset:
    features = Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=list(label_names)),
            "path": Value("string"),
        }
    )
    return Dataset.from_dict(
        {
            "image": list(image_paths),
            "label": list(labels),
            "path": list(image_paths),
        },
        features=features,
    )


def _list_officehome_class_names(dataset_root: Path, domains: Sequence[str]) -> List[str]:
    class_names = set()
    for domain in domains:
        domain_dir = dataset_root / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(f"OfficeHome domain directory not found: {domain_dir}")
        for class_dir in domain_dir.iterdir():
            if class_dir.is_dir():
                class_names.add(class_dir.name)

    if not class_names:
        raise ValueError(f"No class directories found under OfficeHome root: {dataset_root}")
    return sorted(class_names)


def _split_indices(num_examples: int, train_split_ratio: float, rng: random.Random) -> Tuple[List[int], List[int]]:
    indices = list(range(num_examples))
    rng.shuffle(indices)
    if num_examples <= 1:
        return indices, indices

    train_count = int(round(num_examples * train_split_ratio))
    train_count = min(max(train_count, 1), num_examples - 1)
    return indices[:train_count], indices[train_count:]


def _collect_officehome_domain_dataset(
    dataset_root: Path,
    domain: str,
    label_names: Sequence[str],
    train_split_ratio: float,
    seed: int,
) -> Dict[str, Dataset]:
    label_to_id = {label_name: index for index, label_name in enumerate(label_names)}
    train_paths: List[str] = []
    train_labels: List[int] = []
    test_paths: List[str] = []
    test_labels: List[int] = []

    domain_dir = dataset_root / domain
    for class_name in label_names:
        class_dir = domain_dir / class_name
        if not class_dir.is_dir():
            continue

        image_paths = sorted(str(path) for path in class_dir.iterdir() if path.is_file())
        if not image_paths:
            continue

        rng = random.Random(f"{seed}:{domain}:{class_name}")
        train_indices, test_indices = _split_indices(len(image_paths), train_split_ratio=train_split_ratio, rng=rng)
        class_id = label_to_id[class_name]

        for index in train_indices:
            train_paths.append(image_paths[index])
            train_labels.append(class_id)
        for index in test_indices:
            test_paths.append(image_paths[index])
            test_labels.append(class_id)

    if not train_paths or not test_paths:
        raise ValueError(f"OfficeHome domain '{domain}' did not produce both train and test samples.")

    return {
        "train": _build_image_dataset(train_paths, train_labels, label_names),
        "test": _build_image_dataset(test_paths, test_labels, label_names),
    }


def load_officehome_splits(
    dataset_root: str,
    source_domain: str,
    target_domain: str,
    limits: Dict[str, Optional[int]],
    train_split_ratio: float,
    seed: int,
    class_limit: Optional[int],
) -> Dict[str, Dataset]:
    root_path = Path(dataset_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"OfficeHome dataset root not found: {root_path}")

    label_names = _list_officehome_class_names(root_path, OFFICEHOME_DOMAINS)
    source_splits = _collect_officehome_domain_dataset(
        dataset_root=root_path,
        domain=source_domain,
        label_names=label_names,
        train_split_ratio=train_split_ratio,
        seed=seed,
    )
    target_splits = _collect_officehome_domain_dataset(
        dataset_root=root_path,
        domain=target_domain,
        label_names=label_names,
        train_split_ratio=train_split_ratio,
        seed=seed,
    )

    source_train = maybe_limit_dataset(source_splits["train"], limits["source_train"])
    target_train = maybe_limit_dataset(target_splits["train"], limits["target_train"])
    target_test = maybe_limit_dataset(target_splits["test"], limits["target_test"])
    source_test = maybe_limit_dataset(source_splits["test"], limits["source_test"])

    splits = {
        "source_train": source_train,
        "target_train": target_train,
        "target_test": target_test,
        "source_test": source_test,
    }
    return maybe_limit_classes(splits, class_limit=class_limit, seed=seed)


def load_dataset_splits(
    dataset_name: str,
    dataset_root: Optional[str],
    cache_dir: Optional[str],
    source_domain: str,
    target_domain: str,
    limits: Dict[str, Optional[int]],
    train_split_ratio: float,
    seed: int,
    class_limit: Optional[int] = None,
) -> Dict[str, Dataset]:
    validate_domain_names(dataset_name, source_domain, target_domain)
    if is_officehome_dataset(dataset_name):
        if dataset_root is None:
            raise ValueError("dataset_root is required for OfficeHome.")
        return load_officehome_splits(
            dataset_root=dataset_root,
            source_domain=source_domain,
            target_domain=target_domain,
            limits=limits,
            train_split_ratio=train_split_ratio,
            seed=seed,
            class_limit=class_limit,
        )
    return load_domainnet_splits(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        source_domain=source_domain,
        target_domain=target_domain,
        limits=limits,
        class_limit=class_limit,
        seed=seed,
    )
