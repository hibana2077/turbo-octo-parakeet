# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_list_image import ImageList, ImageListIndex
from jfpd.losses import jfpd_loss
from utils.transform import get_transform
from utils.utils import visda_acc

from cdtrans_core.config import cfg as cd_cfg
from cdtrans_core.loss import make_loss
from cdtrans_core.model import make_model
from cdtrans_core.solver import create_scheduler, make_optimizer

logger = logging.getLogger("cdtrans_jfpd")


@dataclass
class TrainMeters:
    loss: float = 0.0
    loss_src: float = 0.0
    loss_tgt: float = 0.0
    loss_distill: float = 0.0
    loss_jfpd: float = 0.0
    acc_src: float = 0.0
    steps: int = 0

    def update(
        self,
        loss: torch.Tensor,
        loss_src: torch.Tensor,
        loss_tgt: torch.Tensor,
        loss_distill: torch.Tensor,
        loss_jfpd: torch.Tensor,
        acc_src: torch.Tensor,
    ) -> None:
        self.loss += float(loss.item())
        self.loss_src += float(loss_src.item())
        self.loss_tgt += float(loss_tgt.item())
        self.loss_distill += float(loss_distill.item())
        self.loss_jfpd += float(loss_jfpd.item())
        self.acc_src += float(acc_src.item())
        self.steps += 1

    def avg(self) -> Dict[str, float]:
        if self.steps == 0:
            return {
                "loss": 0.0,
                "loss_src": 0.0,
                "loss_tgt": 0.0,
                "loss_distill": 0.0,
                "loss_jfpd": 0.0,
                "acc_src": 0.0,
            }
        return {
            "loss": self.loss / self.steps,
            "loss_src": self.loss_src / self.steps,
            "loss_tgt": self.loss_tgt / self.steps,
            "loss_distill": self.loss_distill / self.steps,
            "loss_jfpd": self.loss_jfpd / self.steps,
            "acc_src": self.acc_src / self.steps,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDTrans-based UDA training with optional JFPD")

    parser.add_argument("--name", type=str, default="cdtrans_jfpd", help="Run name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset tag for logging")
    parser.add_argument("--source_list", type=str, required=True, help="Source train list path")
    parser.add_argument("--target_list", type=str, required=True, help="Target train list path")
    parser.add_argument("--test_list", type=str, required=True, help="Target eval list path")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")

    parser.add_argument("--output_dir", type=str, default="output", help="Output root directory")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument(
        "--transformer_type",
        type=str,
        default="uda_vit_base_patch16_224_TransReID",
        choices=["uda_vit_base_patch16_224_TransReID", "uda_vit_small_patch16_224_TransReID"],
        help="CDTrans backbone type",
    )
    parser.add_argument(
        "--block_pattern",
        type=str,
        default="3_branches",
        choices=["3_branches", "normal"],
        help="CDTrans block pattern",
    )
    parser.add_argument(
        "--pretrain_choice",
        type=str,
        default="pretrain",
        choices=["pretrain", "imagenet", "un_pretrain"],
        help="CDTrans pretrain mode",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="Path to pretrained checkpoint for CDTrans backbone/model",
    )

    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument("--max_epochs", type=int, default=40, help="Total training epochs")
    parser.add_argument("--eval_period", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--log_period", type=int, default=50, help="Log every N iterations")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer name",
    )
    parser.add_argument("--learning_rate", type=float, default=8e-3, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--label_smooth", action="store_true", help="Enable label smoothing")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")

    parser.add_argument("--target_loss_weight", type=float, default=1.0, help="Weight of target pseudo-label CE loss")
    parser.add_argument("--distill_weight", type=float, default=1.0, help="Weight of CDTrans distillation loss")
    parser.add_argument("--pseudo_threshold", type=float, default=0.0, help="Confidence threshold for pseudo labels")

    parser.add_argument("--use_jfpd", action="store_true", help="Enable JFPD regularization")
    parser.add_argument("--jfpd_lambda", type=float, default=0.1, help="JFPD loss weight")
    parser.add_argument("--jfpd_alpha", type=float, default=0.5, help="JFPD alpha in [0, 1]")
    parser.add_argument("--jfpd_mode", choices=["jfpd", "fgpd", "pgfd"], default="jfpd", help="JFPD mode")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def distill_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    return torch.sum(-teacher_prob * F.log_softmax(student_logits, dim=-1), dim=-1).mean()


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


def configure_cdtrans(args: argparse.Namespace):
    cfg = cd_cfg.clone()
    cfg.defrost()

    cfg.MODEL.NAME = "transformer"
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE_ID = args.gpu_id
    cfg.MODEL.Transformer_TYPE = args.transformer_type
    cfg.MODEL.BLOCK_PATTERN = args.block_pattern
    cfg.MODEL.TASK_TYPE = "classify_DA"
    cfg.MODEL.UDA_STAGE = "UDA"
    cfg.MODEL.PRETRAIN_CHOICE = args.pretrain_choice
    cfg.MODEL.PRETRAIN_PATH = args.pretrained_path
    cfg.MODEL.ID_LOSS_TYPE = "softmax"
    cfg.MODEL.IF_LABELSMOOTH = "on" if args.label_smooth else "off"

    cfg.INPUT.SIZE_TRAIN = [args.img_size, args.img_size]
    cfg.INPUT.SIZE_TEST = [args.img_size, args.img_size]
    cfg.INPUT.SIZE_CROP = [args.img_size, args.img_size]

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.SAMPLER = "softmax"

    cfg.SOLVER.OPTIMIZER_NAME = args.optimizer
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.WEIGHT_DECAY_BIAS = args.weight_decay
    cfg.SOLVER.WARMUP_EPOCHS = args.warmup_epochs
    cfg.SOLVER.MAX_EPOCHS = args.max_epochs
    cfg.SOLVER.IMS_PER_BATCH = args.train_batch_size
    cfg.SOLVER.LOG_PERIOD = args.log_period
    cfg.SOLVER.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.SEED = args.seed

    cfg.TEST.IMS_PER_BATCH = args.eval_batch_size

    cfg.OUTPUT_DIR = os.path.join(args.output_dir, args.dataset, args.name)
    cfg.freeze()
    return cfg


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


def evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Optional[str]]:
    model.eval()
    all_preds, all_labels = [], []

    iterator = tqdm(test_loader, desc="Validating", dynamic_ncols=True)
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            zeros = torch.zeros(y.size(0), dtype=torch.long, device=device)
            logits_s, logits_t, logits_f = model(
                x,
                x,
                cam_label=zeros,
                view_label=zeros,
                return_logits=True,
                cls_embed_specific=False,
            )
            _ = logits_s, logits_f

            preds = torch.argmax(logits_t, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    if args.dataset.lower() == "visda17":
        accuracy, classwise_acc = visda_acc(preds_np, labels_np)
        return float(accuracy), classwise_acc

    accuracy = float((preds_np == labels_np).mean() * 100.0)
    return accuracy, None


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


def train(
    args: argparse.Namespace,
    cfg,
    model: torch.nn.Module,
    loss_func,
    optimizer: torch.optim.Optimizer,
    scheduler,
    source_loader: DataLoader,
    target_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    scaler = amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    best_acc = -1.0

    steps_per_epoch = min(len(source_loader), len(target_loader))
    if steps_per_epoch <= 0:
        raise RuntimeError("Empty source/target loader detected.")

    logger.info("Start training: epochs=%d, steps_per_epoch=%d", args.max_epochs, steps_per_epoch)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        scheduler.step(epoch)
        meters = TrainMeters()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for n_iter in range(1, steps_per_epoch + 1):
            x_s, y_s = next(source_iter)
            x_t, _, _ = next(target_iter)

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            cam = torch.zeros(y_s.size(0), dtype=torch.long, device=device)
            view = torch.zeros(y_s.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=scaler.is_enabled()):
                (score_s, feat_s, _), (score_t, feat_t, _), (score_fusion, _, _), _ = model(
                    x_s,
                    x_t,
                    y_s,
                    cam_label=cam,
                    view_label=view,
                )

                loss_src = loss_func(score_s, feat_s, y_s, cam)

                with torch.no_grad():
                    target_prob_detach = torch.softmax(score_t, dim=1)
                    pseudo_conf, pseudo_label = target_prob_detach.max(dim=1)
                    pseudo_mask = pseudo_conf >= args.pseudo_threshold

                if pseudo_mask.any():
                    loss_tgt = loss_func(score_t[pseudo_mask], feat_t[pseudo_mask], pseudo_label[pseudo_mask], cam[pseudo_mask])
                else:
                    loss_tgt = torch.zeros((), device=device)

                loss_distill = distill_loss(score_fusion, score_t)

                total_loss = loss_src + args.target_loss_weight * loss_tgt + args.distill_weight * loss_distill

                loss_jfpd = torch.zeros((), device=device)
                if args.use_jfpd:
                    source_feat = feat_s.detach()
                    source_prob = torch.softmax(score_s.detach(), dim=-1)
                    target_feat = feat_t
                    target_prob = torch.softmax(score_t, dim=-1)

                    proto_feat, proto_prob, valid_source = build_source_prototypes(
                        source_feat=source_feat,
                        source_prob=source_prob,
                        source_label=y_s,
                        num_classes=args.num_classes,
                    )

                    pseudo = torch.argmax(target_prob, dim=-1)
                    valid_target = valid_source[pseudo] & pseudo_mask
                    if valid_target.any():
                        zs = proto_feat[pseudo[valid_target]]
                        ps = proto_prob[pseudo[valid_target]]
                        loss_jfpd, _ = jfpd_loss(
                            ft=target_feat[valid_target],
                            pt=target_prob[valid_target],
                            zs=zs,
                            ps=ps,
                            alpha=args.jfpd_alpha,
                            mode=args.jfpd_mode,
                        )
                        total_loss = total_loss + args.jfpd_lambda * loss_jfpd

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc_src = (score_s.max(1)[1] == y_s).float().mean()

            meters.update(total_loss, loss_src, loss_tgt, loss_distill, loss_jfpd, acc_src)

            if n_iter % cfg.SOLVER.LOG_PERIOD == 0:
                avg = meters.avg()
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch[%d/%d] Iter[%d/%d] loss=%.4f src=%.4f tgt=%.4f distill=%.4f jfpd=%.4f acc=%.4f lr=%.6f",
                    epoch,
                    args.max_epochs,
                    n_iter,
                    steps_per_epoch,
                    avg["loss"],
                    avg["loss_src"],
                    avg["loss_tgt"],
                    avg["loss_distill"],
                    avg["loss_jfpd"],
                    avg["acc_src"],
                    current_lr,
                )

        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            acc, classwise = evaluate(args, model, test_loader, device)
            if classwise is None:
                logger.info("Eval Epoch %d: accuracy=%.2f", epoch, acc)
            else:
                logger.info("Eval Epoch %d: accuracy=%.2f classwise=%s", epoch, acc, classwise)

            if acc > best_acc:
                best_acc = acc
                ckpt = save_checkpoint(model, optimizer, scheduler, cfg.OUTPUT_DIR, epoch, best_acc)
                logger.info("Best checkpoint updated: %s (acc=%.2f)", ckpt, best_acc)

    logger.info("Training finished. Best accuracy: %.2f", best_acc)


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.jfpd_alpha <= 1.0):
        raise ValueError("--jfpd_alpha must be in [0, 1].")
    if args.pseudo_threshold < 0.0 or args.pseudo_threshold > 1.0:
        raise ValueError("--pseudo_threshold must be in [0, 1].")
    if args.pretrain_choice != "pretrain" and not args.pretrained_path:
        raise ValueError("--pretrained_path is required when --pretrain_choice is not 'pretrain'.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    cfg = configure_cdtrans(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    set_seed(args.seed)
    logger.info("Run name: %s", args.name)
    logger.info("Output dir: %s", cfg.OUTPUT_DIR)
    logger.info("Using device: %s", device)

    source_loader, target_loader, test_loader = build_dataloaders(args)

    camera_num = 0
    view_num = 0
    model = make_model(cfg, num_class=args.num_classes, camera_num=camera_num, view_num=view_num)
    model.to(device)

    loss_func, center_criterion = make_loss(cfg, num_classes=args.num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    train(
        args=args,
        cfg=cfg,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        source_loader=source_loader,
        target_loader=target_loader,
        test_loader=test_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
