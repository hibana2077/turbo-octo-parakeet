from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DynamicPrototypeSource
from .losses import jfpd_loss
from .types import EpochStats
from .utils import print_stats


@dataclass
class EMAPrototypeBank:
    feat_proto: torch.Tensor
    prob_proto: torch.Tensor
    valid_mask: torch.Tensor
    decay: float

    @classmethod
    def create(
        cls,
        num_classes: int,
        feat_dim: int,
        decay: float,
        device: torch.device,
    ) -> "EMAPrototypeBank":
        return cls(
            feat_proto=torch.zeros(num_classes, feat_dim, device=device),
            prob_proto=torch.zeros(num_classes, num_classes, device=device),
            valid_mask=torch.zeros(num_classes, dtype=torch.bool, device=device),
            decay=decay,
        )

    def update(
        self,
        feat_proto: torch.Tensor,
        prob_proto: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if valid_mask.any():
            fresh_mask = valid_mask & ~self.valid_mask
            if fresh_mask.any():
                self.feat_proto[fresh_mask] = feat_proto[fresh_mask]
                self.prob_proto[fresh_mask] = prob_proto[fresh_mask]

            ema_mask = valid_mask & self.valid_mask
            if ema_mask.any():
                self.feat_proto[ema_mask] = self.decay * self.feat_proto[ema_mask] + (1.0 - self.decay) * feat_proto[ema_mask]
                self.prob_proto[ema_mask] = self.decay * self.prob_proto[ema_mask] + (1.0 - self.decay) * prob_proto[ema_mask]

        self.valid_mask |= valid_mask


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
    prototype_bank: Optional[EMAPrototypeBank] = None,
    epoch: Optional[int] = None,
    log_every_batches: int = 5,
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
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        source_feat_proto, source_prob_proto, valid_proto_mask = build_dynamic_source_prototypes(
            model=model,
            prototype_source=prototype_source,
            num_classes=num_classes,
            samples_per_class=proto_samples_per_class,
            forward_batch_size=proto_forward_batch_size,
            device=device,
        )
        if prototype_bank is not None:
            prototype_bank.update(source_feat_proto, source_prob_proto, valid_proto_mask)
            source_feat_proto = prototype_bank.feat_proto
            source_prob_proto = prototype_bank.prob_proto
            valid_proto_mask = prototype_bank.valid_mask
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
        zs = source_feat_proto[pseudo] # TODO: switch to softmax instead of argmax
        ps = source_prob_proto[pseudo] # TODO: switch to softmax instead of argmax

        loss, stats = jfpd_loss(ft, pt, zs, ps, alpha=alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = ft.size(0)
        meter["loss"] += loss.item() * batch_size
        for key, value in stats.items():
            meter[key] += value * batch_size
        used_samples += batch_size

        should_log = log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == total_batches)
        if should_log and used_samples > 0:
            prefix = f"adapt_epoch_{epoch}_batch_{batch_idx}" if epoch is not None else f"adapt_batch_{batch_idx}"
            print_stats(
                prefix,
                EpochStats(
                    loss=meter["loss"] / used_samples,
                    d_feat=meter["d_feat"] / used_samples,
                    d_pred=meter["d_pred"] / used_samples,
                    psi=meter["psi"] / used_samples,
                    phi=meter["phi"] / used_samples,
                    used_samples=used_samples,
                    skipped_samples=skipped_samples,
                ),
            )

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
