# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_list_image import ImageList, ImageListIndex
from fftat_components import FFTATModel
from fftat_losses import information_maximization_loss
from jfpd.losses import jfpd_loss
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from utils.transform import get_transform
from utils.utils import visda_acc

logger = logging.getLogger("fftat_jfpd")


@dataclass
class TrainMeters:
    loss: float = 0.0
    loss_clc: float = 0.0
    loss_dis: float = 0.0
    loss_pat: float = 0.0
    loss_sc: float = 0.0
    loss_jfpd: float = 0.0
    acc_src: float = 0.0
    steps: int = 0

    def update(
        self,
        loss: torch.Tensor,
        loss_clc: torch.Tensor,
        loss_dis: torch.Tensor,
        loss_pat: torch.Tensor,
        loss_sc: torch.Tensor,
        loss_jfpd: torch.Tensor,
        acc_src: torch.Tensor,
    ) -> None:
        self.loss += float(loss.item())
        self.loss_clc += float(loss_clc.item())
        self.loss_dis += float(loss_dis.item())
        self.loss_pat += float(loss_pat.item())
        self.loss_sc += float(loss_sc.item())
        self.loss_jfpd += float(loss_jfpd.item())
        self.acc_src += float(acc_src.item())
        self.steps += 1

    def avg(self) -> Dict[str, float]:
        if self.steps == 0:
            return {
                "loss": 0.0,
                "loss_clc": 0.0,
                "loss_dis": 0.0,
                "loss_pat": 0.0,
                "loss_sc": 0.0,
                "loss_jfpd": 0.0,
                "acc_src": 0.0,
            }

        return {
            "loss": self.loss / self.steps,
            "loss_clc": self.loss_clc / self.steps,
            "loss_dis": self.loss_dis / self.steps,
            "loss_pat": self.loss_pat / self.steps,
            "loss_sc": self.loss_sc / self.steps,
            "loss_jfpd": self.loss_jfpd / self.steps,
            "acc_src": self.acc_src / self.steps,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFTAT + JFPD UDA training")

    parser.add_argument("--name", type=str, default="fftat_jfpd", help="Run name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset tag for logging")
    parser.add_argument("--source_list", type=str, required=True, help="Source train list path")
    parser.add_argument("--target_list", type=str, required=True, help="Target train list path")
    parser.add_argument("--test_list", type=str, required=True, help="Target eval list path")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")

    parser.add_argument("--output_dir", type=str, default="output", help="Output root directory")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument(
        "--timm_model",
        type=str,
        default="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        help="timm image encoder model name",
    )
    parser.add_argument("--no_timm_pretrained", action="store_true", help="Disable timm pretrained weights")
    parser.add_argument("--split_layer", type=int, default=6, help="Backbone layer index for FFTAT fusion/graph")
    parser.add_argument("--tg_layers", type=int, default=1, help="Number of TG-guided blocks")

    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument("--max_epochs", type=int, default=40, help="Total training epochs")
    parser.add_argument("--eval_period", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--log_period", type=int, default=50, help="Log every N iterations")

    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--decay_type", type=str, default="cosine", choices=["cosine", "linear"], help="LR decay type")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")

    parser.add_argument("--lambda_dis", type=float, default=0.1, help="Weight for domain discriminator loss")
    parser.add_argument("--lambda_pat", type=float, default=0.1, help="Weight for patch discriminator loss")
    parser.add_argument("--lambda_sc", type=float, default=0.0, help="Weight for self-clustering (IM) loss")

    parser.add_argument("--use_jfpd", action="store_true", help="Enable JFPD regularization")
    parser.add_argument("--jfpd_lambda", type=float, default=0.1, help="JFPD loss weight")
    parser.add_argument("--jfpd_alpha", type=float, default=0.5, help="JFPD alpha in [0, 1]")
    parser.add_argument("--jfpd_mode", choices=["jfpd", "fgpd", "pgfd"], default="jfpd", help="JFPD mode")
    parser.add_argument("--pseudo_threshold", type=float, default=0.6, help="Pseudo-label confidence threshold")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_source_prototypes(
    source_feat: torch.Tensor,
    source_prob: torch.Tensor,
    source_label: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feat_dim = source_feat.size(1)
    proto_feat = torch.zeros(num_classes, feat_dim, device=source_feat.device)
    proto_prob = torch.zeros(num_classes, source_prob.size(1), device=source_prob.device)
    counts = torch.zeros(num_classes, device=source_feat.device)

    proto_feat.index_add_(0, source_label, source_feat)
    proto_prob.index_add_(0, source_label, source_prob)
    counts.index_add_(0, source_label, torch.ones_like(source_label, dtype=counts.dtype))

    valid = counts > 0
    if valid.any():
        proto_feat[valid] = proto_feat[valid] / counts[valid].unsqueeze(1)
        proto_prob[valid] = proto_prob[valid] / counts[valid].unsqueeze(1)
    return proto_feat, proto_prob, valid


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)

    with open(args.source_list, "r", encoding="utf-8") as f:
        source_lines = f.readlines()
    with open(args.target_list, "r", encoding="utf-8") as f:
        target_lines = f.readlines()
    with open(args.test_list, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    source_loader = DataLoader(
        ImageList(source_lines, transform=transform_source, mode="RGB"),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    target_loader = DataLoader(
        ImageListIndex(target_lines, transform=transform_target, mode="RGB"),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        ImageList(test_lines, transform=transform_test, mode="RGB"),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return source_loader, target_loader, test_loader


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    total_steps = max(1, args.max_epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    if args.decay_type == "linear":
        return WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    return WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)


def evaluate(
    args: argparse.Namespace,
    model: FFTATModel,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, int, Optional[str]]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model.infer(x)
            batch_loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

            batch_size = y.size(0)
            total_loss += float(batch_loss.item()) * batch_size
            total_samples += batch_size

    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(total_samples, 1)

    if args.dataset.lower() == "visda17":
        accuracy, classwise_acc = visda_acc(preds_np, labels_np)
        return float(accuracy), avg_loss, total_samples, classwise_acc

    accuracy = float((preds_np == labels_np).mean() * 100.0)
    return accuracy, avg_loss, total_samples, None


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: str,
    epoch: int,
    best_acc: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best_model.pth")
    state = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, ckpt_path)
    return ckpt_path


def grl_coeff(step: int, total_steps: int, high: float = 1.0, low: float = 0.0, alpha: float = 10.0) -> float:
    progress = float(step) / float(max(1, total_steps))
    return float(2.0 * (high - low) / (1.0 + math.exp(-alpha * progress)) - (high - low) + low)


def train(
    args: argparse.Namespace,
    model: FFTATModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    source_loader: DataLoader,
    target_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> None:
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    steps_per_epoch = min(len(source_loader), len(target_loader))
    if steps_per_epoch <= 0:
        raise RuntimeError("Empty source/target loader detected.")

    total_steps = args.max_epochs * steps_per_epoch
    best_acc = -1.0
    global_step = 0

    logger.info(
        "Start training: epochs=%d, steps_per_epoch=%d, total_steps=%d",
        args.max_epochs,
        steps_per_epoch,
        total_steps,
    )

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        meters = TrainMeters()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for n_iter in range(1, steps_per_epoch + 1):
            global_step += 1
            lambda_grl = grl_coeff(global_step, total_steps)

            x_s, y_s = next(source_iter)
            x_t, _, _ = next(target_iter)

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x_s, x_t, grl_lambda=lambda_grl)

                logits_s = outputs["logits_s"]
                logits_t = outputs["logits_t"]
                feat_s = outputs["feat_s"]
                feat_t = outputs["feat_t"]

                loss_clc = ce_loss(logits_s, y_s)

                domain_logits = outputs["domain_logits"]
                batch_size = x_s.size(0)
                domain_labels = torch.cat(
                    [
                        torch.ones(batch_size, 1, device=device),
                        torch.zeros(batch_size, 1, device=device),
                    ],
                    dim=0,
                )
                loss_dis = bce_loss(domain_logits, domain_labels)

                patch_logits_s = outputs["patch_logits_s"].reshape(-1, 1)
                patch_logits_t = outputs["patch_logits_t"].reshape(-1, 1)
                patch_labels = torch.cat(
                    [
                        torch.ones_like(patch_logits_s),
                        torch.zeros_like(patch_logits_t),
                    ],
                    dim=0,
                )
                loss_pat = bce_loss(torch.cat([patch_logits_s, patch_logits_t], dim=0), patch_labels)

                if args.lambda_sc > 0.0:
                    loss_sc = information_maximization_loss(logits_t)
                else:
                    loss_sc = torch.zeros((), device=device)

                loss_jfpd = torch.zeros((), device=device)
                if args.use_jfpd and args.jfpd_lambda > 0.0:
                    with torch.no_grad():
                        source_prob = torch.softmax(logits_s.detach(), dim=-1)
                        target_prob_detach = torch.softmax(logits_t.detach(), dim=-1)
                        pseudo_conf, pseudo_label = target_prob_detach.max(dim=1)
                        pseudo_mask = pseudo_conf >= args.pseudo_threshold

                        proto_feat, proto_prob, valid_source = build_source_prototypes(
                            source_feat=feat_s.detach(),
                            source_prob=source_prob,
                            source_label=y_s,
                            num_classes=args.num_classes,
                        )

                    target_prob = torch.softmax(logits_t, dim=-1)
                    valid_target = valid_source[pseudo_label] & pseudo_mask

                    if valid_target.any():
                        zs = proto_feat[pseudo_label[valid_target]]
                        ps = proto_prob[pseudo_label[valid_target]]
                        loss_jfpd, _ = jfpd_loss(
                            ft=feat_t[valid_target],
                            pt=target_prob[valid_target],
                            zs=zs,
                            ps=ps,
                            alpha=args.jfpd_alpha,
                            mode=args.jfpd_mode,
                        )

                total_loss = loss_clc
                total_loss = total_loss + args.lambda_dis * loss_dis
                total_loss = total_loss + args.lambda_pat * loss_pat
                total_loss = total_loss + args.lambda_sc * loss_sc
                if args.use_jfpd and args.jfpd_lambda > 0.0:
                    total_loss = total_loss + args.jfpd_lambda * loss_jfpd

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                acc_src = (torch.argmax(logits_s, dim=1) == y_s).float().mean()

            meters.update(total_loss, loss_clc, loss_dis, loss_pat, loss_sc, loss_jfpd, acc_src)

            if n_iter % args.log_period == 0:
                avg = meters.avg()
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch[%d/%d] Iter[%d/%d] "
                    "loss=%.4f clc=%.4f dis=%.4f pat=%.4f sc=%.4f jfpd=%.4f acc=%.4f lr=%.6f grl=%.4f",
                    epoch,
                    args.max_epochs,
                    n_iter,
                    steps_per_epoch,
                    avg["loss"],
                    avg["loss_clc"],
                    avg["loss_dis"],
                    avg["loss_pat"],
                    avg["loss_sc"],
                    avg["loss_jfpd"],
                    avg["acc_src"],
                    current_lr,
                    lambda_grl,
                )

        if epoch % args.eval_period == 0:
            acc, val_loss, val_samples, classwise = evaluate(args, model, test_loader, device)
            if classwise is None:
                logger.info(
                    "Eval Epoch %d: accuracy=%.2f val_loss=%.4f samples=%d",
                    epoch,
                    acc,
                    val_loss,
                    val_samples,
                )
            else:
                logger.info(
                    "Eval Epoch %d: accuracy=%.2f val_loss=%.4f samples=%d classwise=%s",
                    epoch,
                    acc,
                    val_loss,
                    val_samples,
                    classwise,
                )

            if acc > best_acc:
                best_acc = acc
                ckpt = save_checkpoint(model, optimizer, scheduler, output_dir, epoch, best_acc)
                logger.info("Best checkpoint updated: %s (acc=%.2f)", ckpt, best_acc)

    logger.info("Training finished. Best accuracy: %.2f", best_acc)


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.jfpd_alpha <= 1.0):
        raise ValueError("--jfpd_alpha must be in [0, 1].")
    if args.pseudo_threshold < 0.0 or args.pseudo_threshold > 1.0:
        raise ValueError("--pseudo_threshold must be in [0, 1].")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    output_dir = os.path.join(args.output_dir, args.dataset, args.name)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    logger.info("Run name: %s", args.name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Using device: %s", device)
    logger.info("timm model: %s (pretrained=%s)", args.timm_model, str(not args.no_timm_pretrained))

    source_loader, target_loader, test_loader = build_dataloaders(args)
    steps_per_epoch = min(len(source_loader), len(target_loader))
    if steps_per_epoch <= 0:
        raise RuntimeError("Source/target DataLoader is empty. Check dataset list files.")

    model = FFTATModel(
        timm_model_name=args.timm_model,
        num_classes=args.num_classes,
        img_size=args.img_size,
        pretrained=not args.no_timm_pretrained,
        split_layer=args.split_layer,
        tg_layers=args.tg_layers,
    )
    model.to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, steps_per_epoch)

    train(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        source_loader=source_loader,
        target_loader=target_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
