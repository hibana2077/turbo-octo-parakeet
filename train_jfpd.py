#!/usr/bin/env python3

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochStats:
    loss: float
    accuracy: Optional[float] = None
    d_feat: Optional[float] = None
    d_pred: Optional[float] = None
    psi: Optional[float] = None
    phi: Optional[float] = None
    used_samples: Optional[int] = None
    skipped_samples: Optional[int] = None


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


class JFPDNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.classifier(feat)
        prob = torch.softmax(logits, dim=-1)
        return feat, logits, prob


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Version B JFPD pipeline for DomainNet.")
    parser.add_argument("--dataset-name", default="wltjr1007/DomainNet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-domain", required=True, choices=DOMAINNET_DOMAINS)
    parser.add_argument("--target-domain", required=True, choices=DOMAINNET_DOMAINS)
    parser.add_argument("--model-name", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument("--adapt-epochs", type=int, default=5)
    parser.add_argument("--source-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--proto-samples-per-class", type=int, default=32)
    parser.add_argument("--proto-forward-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/jfpd")
    parser.add_argument("--max-source-train-samples", type=int, default=None)
    parser.add_argument("--max-target-train-samples", type=int, default=None)
    parser.add_argument("--max-target-test-samples", type=int, default=None)
    parser.add_argument("--max-source-test-samples", type=int, default=None)
    parser.add_argument("--eval-source", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    return 1.0 - (x * y).sum(dim=-1)


def normalized_feature_divergence(ft: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    dist = cosine_distance(ft, zs)
    return dist / (1.0 + dist)


def entropy_from_prob(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def kl_div_prob(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_div_prob(p, m, eps) + 0.5 * kl_div_prob(q, m, eps)


def normalized_prediction_divergence(pt: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    dist = js_divergence(pt, ps)
    return dist / (1.0 + dist)


def jfpd_loss(
    ft: torch.Tensor,
    pt: torch.Tensor,
    zs: torch.Tensor,
    ps: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    d_feat = normalized_feature_divergence(ft, zs)
    d_pred = normalized_prediction_divergence(pt, ps)
    hs = entropy_from_prob(ps)
    ht = entropy_from_prob(pt)

    psi = 1.0 / (1.0 + hs + ht)
    phi = 1.0 / (1.0 + d_feat)
    loss = alpha * psi * d_feat + (1.0 - alpha) * phi * d_pred

    stats = {
        "d_feat": d_feat.mean().item(),
        "d_pred": d_pred.mean().item(),
        "psi": psi.mean().item(),
        "phi": phi.mean().item(),
    }
    return loss.mean(), stats


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_source_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        _, logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)

    return EpochStats(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        _, logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)

    return EpochStats(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


def build_class_index(dataset: Dataset, num_classes: int) -> List[List[int]]:
    class_to_indices = [[] for _ in range(num_classes)]
    for index, label in enumerate(dataset["label"]):
        class_to_indices[int(label)].append(index)
    return class_to_indices


@torch.no_grad()
def build_dynamic_source_prototypes(
    model: nn.Module,
    prototype_source: DynamicPrototypeSource,
    num_classes: int,
    samples_per_class: int,
    forward_batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    feat_proto = torch.zeros(num_classes, model.classifier.in_features, device=device)
    prob_proto = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    sampled_pairs = prototype_source.sample_pairs(
        class_ids=prototype_source.valid_class_ids,
        samples_per_class=samples_per_class,
    )
    if not sampled_pairs:
        raise RuntimeError("Dynamic prototype estimation received no source samples.")

    for start in range(0, len(sampled_pairs), forward_batch_size):
        chunk = sampled_pairs[start : start + forward_batch_size]
        class_ids = torch.tensor([class_id for class_id, _ in chunk], device=device, dtype=torch.long)
        sample_indices = [sample_index for _, sample_index in chunk]
        x = prototype_source.build_pixel_batch(sample_indices).to(device, non_blocking=True)
        feat, _, prob = model(x)

        feat_proto.index_add_(0, class_ids, feat)
        prob_proto.index_add_(0, class_ids, prob)
        counts.index_add_(0, class_ids, torch.ones(class_ids.size(0), device=device))

    valid_mask = counts > 0
    feat_proto[valid_mask] = feat_proto[valid_mask] / counts[valid_mask].unsqueeze(-1)
    prob_proto[valid_mask] = prob_proto[valid_mask] / counts[valid_mask].unsqueeze(-1)

    return feat_proto, prob_proto, valid_mask


def adapt_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    prototype_source: DynamicPrototypeSource,
    num_classes: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    proto_samples_per_class: int,
    proto_forward_batch_size: int,
) -> EpochStats:
    model.train()
    meter = {
        "loss": 0.0,
        "d_feat": 0.0,
        "d_pred": 0.0,
        "psi": 0.0,
        "phi": 0.0,
    }
    used_samples = 0
    skipped_samples = 0

    for batch in loader:
        source_feat_proto, source_prob_proto, valid_proto_mask = build_dynamic_source_prototypes(
            model=model,
            prototype_source=prototype_source,
            num_classes=num_classes,
            samples_per_class=proto_samples_per_class,
            forward_batch_size=proto_forward_batch_size,
            device=device,
        )
        model.train()

        x_t = batch["pixel_values"].to(device, non_blocking=True)
        ft, _, pt = model(x_t)
        pseudo = pt.argmax(dim=-1)

        keep_mask = valid_proto_mask[pseudo]
        skipped_samples += (~keep_mask).sum().item()
        if not keep_mask.any():
            continue

        ft = ft[keep_mask]
        pt = pt[keep_mask]
        pseudo = pseudo[keep_mask]
        zs = source_feat_proto[pseudo]
        ps = source_prob_proto[pseudo]

        loss, stats = jfpd_loss(ft, pt, zs, ps, alpha=alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = ft.size(0)
        meter["loss"] += loss.item() * batch_size
        for key, value in stats.items():
            meter[key] += value * batch_size
        used_samples += batch_size

    if used_samples == 0:
        raise RuntimeError("All target samples were skipped because no valid source prototypes were available.")

    return EpochStats(
        loss=meter["loss"] / used_samples,
        d_feat=meter["d_feat"] / used_samples,
        d_pred=meter["d_pred"] / used_samples,
        psi=meter["psi"] / used_samples,
        phi=meter["phi"] / used_samples,
        used_samples=used_samples,
        skipped_samples=skipped_samples,
    )


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


def main() -> None:
    args = parse_args()
    if args.source_domain == args.target_domain:
        raise ValueError("source-domain and target-domain must be different.")

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "source_train": args.max_source_train_samples,
        "target_train": args.max_target_train_samples,
        "target_test": args.max_target_test_samples,
        "source_test": args.max_source_test_samples,
    }
    splits = load_domainnet_splits(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        limits=limits,
    )

    label_names = get_class_names(splits["source_train"], "label")
    num_classes = len(label_names)

    print(f"device={device}")
    for split_name, split_dataset in splits.items():
        print(f"{split_name}_samples={len(split_dataset)}")

    source_train_loader = build_loader(
        dataset=splits["source_train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
    )
    target_train_loader = build_loader(
        dataset=splits["target_train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
    )
    target_test_loader = build_loader(
        dataset=splits["target_test"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    source_test_loader = build_loader(
        dataset=splits["source_test"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    model = JFPDNet(model_name=args.model_name, num_classes=num_classes).to(device)
    prototype_source = DynamicPrototypeSource(
        dataset=splits["source_train"],
        image_size=args.image_size,
        num_classes=num_classes,
    )
    history = {
        "config": vars(args),
        "num_classes": num_classes,
        "label_names": label_names,
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "source_pretrain": [],
        "adaptation": [],
    }

    source_optimizer = build_optimizer(model, lr=args.source_lr, weight_decay=args.weight_decay)
    best_source_acc = -1.0

    for epoch in range(1, args.source_epochs + 1):
        train_stats = train_source_epoch(model, source_train_loader, source_optimizer, device)
        print_stats(f"source_train_epoch_{epoch}", train_stats)
        epoch_record = {"epoch": epoch, "train": asdict(train_stats)}

        if args.eval_source:
            eval_stats = evaluate(model, source_test_loader, device)
            print_stats(f"source_test_epoch_{epoch}", eval_stats)
            epoch_record["eval"] = asdict(eval_stats)
            if eval_stats.accuracy is not None and eval_stats.accuracy > best_source_acc:
                best_source_acc = eval_stats.accuracy
                save_checkpoint(
                    output_dir / "best_source.pt",
                    model,
                    {"epoch": epoch, "accuracy": eval_stats.accuracy, "stage": "source"},
                )

        history["source_pretrain"].append(epoch_record)
        save_json(output_dir / "history.json", history)

    missing_classes = (~prototype_source.valid_mask).nonzero(as_tuple=False).view(-1).tolist()
    if missing_classes:
        missing_names = [label_names[index] for index in missing_classes]
        print(f"warning: missing source prototypes for {len(missing_names)} classes")

    save_json(
        output_dir / "prototype_source.json",
        {
            "mode": "dynamic",
            "samples_per_class": args.proto_samples_per_class,
            "valid_mask": prototype_source.valid_mask.tolist(),
            "missing_classes": missing_classes,
            "label_names": label_names,
        },
    )

    adapt_optimizer = build_optimizer(model, lr=args.adapt_lr, weight_decay=args.weight_decay)
    best_target_acc = -1.0

    for epoch in range(1, args.adapt_epochs + 1):
        adapt_stats = adapt_one_epoch(
            model=model,
            loader=target_train_loader,
            prototype_source=prototype_source,
            num_classes=num_classes,
            optimizer=adapt_optimizer,
            device=device,
            alpha=args.alpha,
            proto_samples_per_class=args.proto_samples_per_class,
            proto_forward_batch_size=args.proto_forward_batch_size,
        )
        print_stats(f"adapt_epoch_{epoch}", adapt_stats)

        eval_stats = evaluate(model, target_test_loader, device)
        print_stats(f"target_test_epoch_{epoch}", eval_stats)

        epoch_record = {"epoch": epoch, "adapt": asdict(adapt_stats), "eval": asdict(eval_stats)}
        history["adaptation"].append(epoch_record)
        save_json(output_dir / "history.json", history)

        if eval_stats.accuracy is not None and eval_stats.accuracy > best_target_acc:
            best_target_acc = eval_stats.accuracy
            save_checkpoint(
                output_dir / "best_target.pt",
                model,
                {"epoch": epoch, "accuracy": eval_stats.accuracy, "stage": "adapt"},
            )

    save_checkpoint(output_dir / "last.pt", model, {"stage": "final"})
    print(f"artifacts={output_dir.resolve()}")


if __name__ == "__main__":
    main()
