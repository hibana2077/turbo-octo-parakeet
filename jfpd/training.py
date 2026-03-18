from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DynamicPrototypeSource
from .losses import LossMode, jfpd_loss
from .types import EpochStats
from .utils import print_stats


def _format_debug_vector(values: torch.Tensor, limit: int = 5) -> str:
    clipped = values.detach().cpu().tolist()[:limit]
    return "[" + ", ".join(f"{value:.6f}" for value in clipped) + "]"


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer received no trainable parameters.")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


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


def _sparse_confusion_entries(confusion: torch.Tensor) -> list:
    nonzero = confusion.nonzero(as_tuple=False)
    entries = []
    for true_idx, pred_idx in nonzero.tolist():
        entries.append(
            {
                "true": true_idx,
                "pred": pred_idx,
                "count": int(confusion[true_idx, pred_idx].item()),
            }
        )
    return entries


def _top_histogram_entries(hist: torch.Tensor, label_names: Optional[list], limit: int = 5) -> list:
    if hist.numel() == 0:
        return []

    values, indices = torch.topk(hist, k=min(limit, hist.numel()))
    entries = []
    for count, class_id in zip(values.tolist(), indices.tolist()):
        if count <= 0:
            continue
        entry = {"class_id": int(class_id), "count": int(count)}
        if label_names is not None and 0 <= class_id < len(label_names):
            entry["label"] = label_names[class_id]
        entries.append(entry)
    return entries


def _top_metric_entries(values: torch.Tensor, label_names: Optional[list], limit: int = 5) -> list:
    if values.numel() == 0:
        return []

    metric = values.detach().cpu()
    top_values, top_indices = torch.topk(metric, k=min(limit, metric.numel()))
    entries = []
    for value, class_id in zip(top_values.tolist(), top_indices.tolist()):
        entry = {"class_id": int(class_id), "value": float(value)}
        if label_names is not None and 0 <= class_id < len(label_names):
            entry["label"] = label_names[class_id]
        entries.append(entry)
    return entries


def _class_metric_summary(
    values: torch.Tensor,
    class_id: int,
    label_names: Optional[list],
) -> Dict[str, object]:
    metric = values.detach().cpu()
    if metric.numel() == 0 or not (0 <= class_id < metric.numel()):
        return {
            "class_id": class_id,
            "label": label_names[class_id] if label_names is not None and 0 <= class_id < len(label_names) else None,
            "value": None,
            "rank_desc": None,
        }

    target_value = float(metric[class_id].item())
    rank_desc = int((metric > metric[class_id]).sum().item()) + 1
    summary: Dict[str, object] = {
        "class_id": class_id,
        "value": target_value,
        "rank_desc": rank_desc,
    }
    if label_names is not None and 0 <= class_id < len(label_names):
        summary["label"] = label_names[class_id]
    return summary


def _select_top_confidence_per_class(
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    max_samples_per_class: Optional[int],
) -> Tuple[torch.Tensor, int]:
    keep_mask = torch.ones_like(pseudo, dtype=torch.bool)
    if max_samples_per_class is None:
        return keep_mask, 0

    keep_mask = torch.zeros_like(pseudo, dtype=torch.bool)
    filtered_count = 0
    for class_id in pseudo.unique(sorted=False):
        class_mask = pseudo == class_id
        class_indices = class_mask.nonzero(as_tuple=False).view(-1)
        if class_indices.numel() <= max_samples_per_class:
            keep_mask[class_indices] = True
            continue

        class_confidence = confidence[class_indices]
        top_indices = torch.topk(class_confidence, k=max_samples_per_class).indices
        keep_mask[class_indices[top_indices]] = True
        filtered_count += class_indices.numel() - max_samples_per_class

    return keep_mask, filtered_count


@torch.no_grad()
def evaluate_with_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_names: Optional[list] = None,
) -> Tuple[EpochStats, Dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_max_prob = 0.0
    pred_hist = torch.zeros(num_classes, dtype=torch.long)
    label_hist = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        _, logits, prob = model(x)
        preds = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
        total_max_prob += prob.max(dim=-1).values.sum().item()

        pred_hist += torch.bincount(preds.detach().cpu(), minlength=num_classes)
        label_hist += torch.bincount(y.detach().cpu(), minlength=num_classes)
        flat_confusion = y.detach().cpu() * num_classes + preds.detach().cpu()
        confusion += torch.bincount(flat_confusion, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    stats = EpochStats(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )

    dominant_pred_class = int(pred_hist.argmax().item()) if total_samples > 0 else None
    dominant_pred_count = int(pred_hist.max().item()) if total_samples > 0 else 0
    diagnostics: Dict[str, object] = {
        "pred_hist": pred_hist.tolist(),
        "label_hist": label_hist.tolist(),
        "mean_max_prob": total_max_prob / max(total_samples, 1),
        "dominant_pred_class": dominant_pred_class,
        "dominant_pred_ratio": dominant_pred_count / max(total_samples, 1),
        "collapse_suspected": dominant_pred_count / max(total_samples, 1) >= 0.9,
        "pred_hist_top": _top_histogram_entries(pred_hist, label_names),
        "label_hist_top": _top_histogram_entries(label_hist, label_names),
        "confusion_nonzero": _sparse_confusion_entries(confusion),
    }
    if dominant_pred_class is not None and label_names is not None and 0 <= dominant_pred_class < len(label_names):
        diagnostics["dominant_pred_label"] = label_names[dominant_pred_class]

    return stats, diagnostics


@torch.no_grad()
def summarize_collapse_risk(
    model: nn.Module,
    prototype_source: DynamicPrototypeSource,
    num_classes: int,
    samples_per_class: int,
    forward_batch_size: int,
    device: torch.device,
    label_names: Optional[list] = None,
) -> Dict[str, object]:
    classifier_bias = model.classifier.bias.detach().cpu()
    classifier_weight_norm = model.classifier.weight.detach().norm(dim=-1).cpu()
    source_feat_proto, source_prob_proto, valid_proto_mask = build_dynamic_source_prototypes(
        model=model,
        prototype_source=prototype_source,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        forward_batch_size=forward_batch_size,
        device=device,
    )

    source_feat_proto_norm = source_feat_proto.detach().norm(dim=-1).cpu()
    source_prob_proto_peak = source_prob_proto.detach().max(dim=-1).values.cpu()
    valid_class_ids = valid_proto_mask.detach().cpu().nonzero(as_tuple=False).view(-1)

    diagnostics: Dict[str, object] = {
        "classifier_bias": classifier_bias.tolist(),
        "classifier_weight_norm": classifier_weight_norm.tolist(),
        "classifier_bias_top": _top_metric_entries(classifier_bias, label_names, limit=10),
        "classifier_weight_norm_top": _top_metric_entries(classifier_weight_norm, label_names, limit=10),
        "class0_classifier_bias": _class_metric_summary(classifier_bias, 0, label_names),
        "class0_classifier_weight_norm": _class_metric_summary(classifier_weight_norm, 0, label_names),
        "valid_source_proto_classes": valid_class_ids.tolist(),
        "source_feat_proto_norm": source_feat_proto_norm.tolist(),
        "source_prob_proto_peak": source_prob_proto_peak.tolist(),
        "source_feat_proto_norm_top": _top_metric_entries(source_feat_proto_norm, label_names, limit=10),
        "source_prob_proto_peak_top": _top_metric_entries(source_prob_proto_peak, label_names, limit=10),
        "class0_source_feat_proto_norm": _class_metric_summary(source_feat_proto_norm, 0, label_names),
        "class0_source_prob_proto_peak": _class_metric_summary(source_prob_proto_peak, 0, label_names),
    }

    if 0 <= 0 < num_classes and bool(valid_proto_mask[0].item()):
        diagnostics["class0_source_prob_proto_top"] = _top_metric_entries(source_prob_proto[0], label_names, limit=10)

    return diagnostics


@torch.no_grad()
def build_dynamic_source_prototypes(
    model: nn.Module,
    prototype_source: DynamicPrototypeSource,
    num_classes: int,
    samples_per_class: int,
    forward_batch_size: int,
    device: torch.device,
    debug_bug2: bool = False,
    debug_prefix: Optional[str] = None,
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

    if debug_bug2:
        prefix = debug_prefix or "bug2_debug"
        print(f"{prefix}: source_proto_count_per_class={counts.detach().cpu().tolist()}")
        print(
            f"{prefix}: z_s_shape={tuple(feat_proto.shape)}, "
            f"p_s_shape={tuple(prob_proto.shape)}, "
            f"feature_dim={feat_proto.size(-1)}, num_classes={num_classes}"
        )
        if valid_mask.any():
            print(f"{prefix}: p_s_row_sums={_format_debug_vector(prob_proto[valid_mask].sum(dim=-1))}")

    return feat_proto, prob_proto, valid_mask


def adapt_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    prototype_source: DynamicPrototypeSource,
    source_loader: Optional[DataLoader],
    num_classes: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    loss_mode: LossMode,
    proto_samples_per_class: int,
    proto_forward_batch_size: int,
    pseudo_confidence_threshold: float,
    source_anchor_weight: float,
    max_pseudo_per_class: Optional[int],
    epoch: Optional[int] = None,
    log_every_batches: int = 5,
    debug_bug2: bool = False,
    debug_collapse: bool = False,
    label_names: Optional[list] = None,
) -> Tuple[EpochStats, Dict[str, object]]:
    model.train()
    meter = {
        "loss": 0.0,
        "d_feat": 0.0,
        "d_pred": 0.0,
        "psi": 0.0,
        "phi": 0.0,
        "source_loss": 0.0,
    }
    used_samples = 0
    skipped_samples = 0
    filtered_by_confidence = 0
    filtered_by_class_cap = 0
    total_batches = len(loader)
    bug2_logged = False
    pseudo_hist = torch.zeros(num_classes, dtype=torch.long)
    confident_pseudo_hist = torch.zeros(num_classes, dtype=torch.long)
    selected_proto_hist = torch.zeros(num_classes, dtype=torch.long)
    total_target_samples = 0
    total_max_prob = 0.0
    selected_max_prob_sum = 0.0
    source_iter = iter(source_loader) if source_loader is not None and source_anchor_weight > 0.0 else None

    for batch_idx, batch in enumerate(loader, start=1):
        source_feat_proto, source_prob_proto, valid_proto_mask = build_dynamic_source_prototypes(
            model=model,
            prototype_source=prototype_source,
            num_classes=num_classes,
            samples_per_class=proto_samples_per_class,
            forward_batch_size=proto_forward_batch_size,
            device=device,
            debug_bug2=debug_bug2 and not bug2_logged,
            debug_prefix=(
                f"bug2_epoch_{epoch}_batch_{batch_idx}_source_proto"
                if epoch is not None
                else f"bug2_batch_{batch_idx}_source_proto"
            ),
        )
        model.train()

        x_t = batch["pixel_values"].to(device, non_blocking=True)
        ft, _, pt = model(x_t)
        pseudo = pt.argmax(dim=-1)
        max_prob = pt.max(dim=-1).values

        total_target_samples += pt.size(0)
        total_max_prob += max_prob.sum().item()
        pseudo_hist += torch.bincount(pseudo.detach().cpu(), minlength=num_classes)

        keep_mask = valid_proto_mask[pseudo]
        if pseudo_confidence_threshold > 0.0:
            confidence_mask = max_prob >= pseudo_confidence_threshold
            filtered_by_confidence += (keep_mask & ~confidence_mask).sum().item()
            keep_mask = keep_mask & confidence_mask
        if not keep_mask.any():
            skipped_samples += pt.size(0)
            continue

        confident_pseudo_hist += torch.bincount(pseudo[keep_mask].detach().cpu(), minlength=num_classes)
        capped_keep_mask = torch.zeros_like(keep_mask)
        kept_indices = keep_mask.nonzero(as_tuple=False).view(-1)
        local_keep_mask, dropped_by_class_cap = _select_top_confidence_per_class(
            pseudo=pseudo[kept_indices],
            confidence=max_prob[kept_indices],
            max_samples_per_class=max_pseudo_per_class,
        )
        capped_keep_mask[kept_indices[local_keep_mask]] = True
        keep_mask = capped_keep_mask
        filtered_by_class_cap += dropped_by_class_cap
        skipped_samples += pt.size(0) - int(keep_mask.sum().item())
        if not keep_mask.any():
            continue

        ft = ft[keep_mask]
        pt = pt[keep_mask]
        pseudo = pseudo[keep_mask]
        selected_max_prob_sum += max_prob[keep_mask].sum().item()
        selected_proto_hist += torch.bincount(pseudo.detach().cpu(), minlength=num_classes)
        zs = source_feat_proto[pseudo]
        ps = source_prob_proto[pseudo]

        if debug_bug2 and not bug2_logged:
            prefix = (
                f"bug2_epoch_{epoch}_batch_{batch_idx}_target"
                if epoch is not None
                else f"bug2_batch_{batch_idx}_target"
            )
            print(f"{prefix}: f_t_shape={tuple(ft.shape)}, p_t_shape={tuple(pt.shape)}")
            print(f"{prefix}: p_t_row_sums={_format_debug_vector(pt.sum(dim=-1))}")
            print(f"{prefix}: p_s_row_sums={_format_debug_vector(ps.sum(dim=-1))}")
            bug2_logged = True

        if debug_collapse and batch_idx == 1:
            prefix = f"collapse_epoch_{epoch}_batch_{batch_idx}" if epoch is not None else f"collapse_batch_{batch_idx}"
            print(f"{prefix}: pseudo_hist_top={_top_histogram_entries(pseudo_hist, label_names, limit=10)}")
            print(f"{prefix}: confident_pseudo_hist_top={_top_histogram_entries(confident_pseudo_hist, label_names, limit=10)}")
            print(f"{prefix}: selected_proto_hist_top={_top_histogram_entries(selected_proto_hist, label_names, limit=10)}")
            print(f"{prefix}: mean_max_prob={total_max_prob / max(total_target_samples, 1):.4f}")

        loss, stats = jfpd_loss(ft, pt, zs, ps, alpha=alpha, mode=loss_mode)
        if source_iter is not None:
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)

            x_s = source_batch["pixel_values"].to(device, non_blocking=True)
            y_s = source_batch["label"].to(device, non_blocking=True)
            _, logits_s, _ = model(x_s)
            source_loss = F.cross_entropy(logits_s, y_s)
            loss = loss + source_anchor_weight * source_loss
            stats["source_loss"] = source_loss.item()
        else:
            stats["source_loss"] = 0.0

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
                    source_loss=meter["source_loss"] / used_samples,
                    used_samples=used_samples,
                    skipped_samples=skipped_samples,
                    filtered_by_confidence=filtered_by_confidence,
                    filtered_by_class_cap=filtered_by_class_cap,
                ),
            )

    if used_samples == 0:
        raise RuntimeError(
            "All target samples were skipped during adaptation. "
            "Relax pseudo-confidence-threshold or max-pseudo-per-class if this is unintended."
        )

    diagnostics: Dict[str, object] = {
        "pseudo_hist": pseudo_hist.tolist(),
        "confident_pseudo_hist": confident_pseudo_hist.tolist(),
        "selected_proto_hist": selected_proto_hist.tolist(),
        "pseudo_hist_top": _top_histogram_entries(pseudo_hist, label_names, limit=10),
        "confident_pseudo_hist_top": _top_histogram_entries(confident_pseudo_hist, label_names, limit=10),
        "selected_proto_hist_top": _top_histogram_entries(selected_proto_hist, label_names, limit=10),
        "mean_max_prob": total_max_prob / max(total_target_samples, 1),
        "mean_selected_max_prob": selected_max_prob_sum / max(used_samples, 1),
        "dominant_pseudo_class": int(pseudo_hist.argmax().item()) if total_target_samples > 0 else None,
        "dominant_pseudo_ratio": int(pseudo_hist.max().item()) / max(total_target_samples, 1),
        "dominant_confident_pseudo_class": int(confident_pseudo_hist.argmax().item()) if confident_pseudo_hist.sum().item() > 0 else None,
        "dominant_confident_pseudo_ratio": int(confident_pseudo_hist.max().item()) / max(int(confident_pseudo_hist.sum().item()), 1),
        "dominant_selected_proto_class": int(selected_proto_hist.argmax().item()) if used_samples > 0 else None,
        "dominant_selected_proto_ratio": int(selected_proto_hist.max().item()) / max(used_samples, 1),
        "filtered_by_confidence": filtered_by_confidence,
        "filtered_by_class_cap": filtered_by_class_cap,
        "pseudo_confidence_threshold": pseudo_confidence_threshold,
        "source_anchor_weight": source_anchor_weight,
        "max_pseudo_per_class": max_pseudo_per_class,
    }
    if diagnostics["dominant_pseudo_class"] is not None and label_names is not None:
        dominant_pseudo_class = diagnostics["dominant_pseudo_class"]
        if 0 <= dominant_pseudo_class < len(label_names):
            diagnostics["dominant_pseudo_label"] = label_names[dominant_pseudo_class]
    if diagnostics["dominant_confident_pseudo_class"] is not None and label_names is not None:
        dominant_confident_class = diagnostics["dominant_confident_pseudo_class"]
        if 0 <= dominant_confident_class < len(label_names):
            diagnostics["dominant_confident_pseudo_label"] = label_names[dominant_confident_class]
    if diagnostics["dominant_selected_proto_class"] is not None and label_names is not None:
        dominant_selected_class = diagnostics["dominant_selected_proto_class"]
        if 0 <= dominant_selected_class < len(label_names):
            diagnostics["dominant_selected_proto_label"] = label_names[dominant_selected_class]

    if debug_collapse:
        prefix = f"collapse_epoch_{epoch}_summary" if epoch is not None else "collapse_summary"
        print(
            f"{prefix}: mean_max_prob={diagnostics['mean_max_prob']:.4f}, "
            f"dominant_pseudo_class={diagnostics['dominant_pseudo_class']}, "
            f"dominant_pseudo_ratio={diagnostics['dominant_pseudo_ratio']:.4f}, "
            f"dominant_confident_pseudo_class={diagnostics['dominant_confident_pseudo_class']}, "
            f"dominant_confident_pseudo_ratio={diagnostics['dominant_confident_pseudo_ratio']:.4f}, "
            f"dominant_selected_proto_class={diagnostics['dominant_selected_proto_class']}, "
            f"dominant_selected_proto_ratio={diagnostics['dominant_selected_proto_ratio']:.4f}"
        )
        print(f"{prefix}: pseudo_hist_top={diagnostics['pseudo_hist_top']}")
        print(f"{prefix}: confident_pseudo_hist_top={diagnostics['confident_pseudo_hist_top']}")
        print(f"{prefix}: selected_proto_hist_top={diagnostics['selected_proto_hist_top']}")

    stats = EpochStats(
        loss=meter["loss"] / used_samples,
        d_feat=meter["d_feat"] / used_samples,
        d_pred=meter["d_pred"] / used_samples,
        psi=meter["psi"] / used_samples,
        phi=meter["phi"] / used_samples,
        source_loss=meter["source_loss"] / used_samples,
        used_samples=used_samples,
        skipped_samples=skipped_samples,
        filtered_by_confidence=filtered_by_confidence,
        filtered_by_class_cap=filtered_by_class_cap,
    )
    return stats, diagnostics
