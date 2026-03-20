# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.graph_alignment import GraphConvDiscriminator
from models.transadapter_loss import FocalLoss, adv_global
from data.data_list_image import ImageList, ImageListIndex
from jfpd.losses import jfpd_loss
from models.modeling import CONFIGS, AdversarialNetwork, VisionTransformer
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from utils.transform import get_transform
from utils.utils import visda_acc

logger = logging.getLogger(__name__)

LEGACY_MODEL_TYPE_TO_TIMM = {name: cfg.timm_name for name, cfg in CONFIGS.items()}
TIMM_CONFIG_LOOKUP = {cfg.timm_name: cfg for cfg in CONFIGS.values()}


def resolve_backbone_config(args):
    timm_model = (args.timm_model or "").strip()
    legacy_model_type = (args.model_type or "").strip()

    if legacy_model_type:
        if legacy_model_type not in LEGACY_MODEL_TYPE_TO_TIMM:
            supported = ", ".join(sorted(LEGACY_MODEL_TYPE_TO_TIMM))
            raise ValueError(f"Unsupported legacy --model_type '{legacy_model_type}'. Supported: {supported}")
        mapped_timm_model = LEGACY_MODEL_TYPE_TO_TIMM[legacy_model_type]
        if timm_model and timm_model != mapped_timm_model:
            raise ValueError(
                f"Conflicting backbone args: --model_type {legacy_model_type} maps to {mapped_timm_model}, "
                f"but --timm_model is {timm_model}."
            )
        timm_model = mapped_timm_model
        logger.warning("`--model_type` is deprecated. Please use `--timm_model %s`.", timm_model)

    if not timm_model:
        timm_model = CONFIGS["ViT-B_16"].timm_name

    if args.pretrained_dir:
        logger.warning("`--pretrained_dir` is deprecated and ignored because timm pretrained loading is used.")

    args.timm_model = timm_model
    return TIMM_CONFIG_LOOKUP.get(timm_model, CONFIGS["ViT-B_16"])


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def build_source_prototypes(source_feat, source_prob, source_label, num_classes):
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


def save_model(args, model, prefix_saved_mode, is_adv=False):
    model_to_save = model.module if hasattr(model, "module") else model
    suffix = "_checkpoint_adv_.bin" if is_adv else "_checkpoint_.bin"
    model_checkpoint = os.path.join(args.output_dir, args.dataset, f"{prefix_saved_mode}{args.name}{suffix}")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def extract_checkpoint_acc(filename, prefix_saved_mode):
    if prefix_saved_mode not in filename:
        return None
    suffix = filename.split(prefix_saved_mode, 1)[1]
    if not suffix:
        return None
    token = suffix.split("_", 1)[0]
    try:
        return float(token)
    except ValueError:
        return None


def setup(args, prefix_saved_mode):
    config = resolve_backbone_config(args)
    model = VisionTransformer(
        config,
        args.img_size,
        zero_head=True,
        num_classes=args.num_classes,
        msa_layer=args.msa_layer,
        timm_model_name=args.timm_model or None,
        pretrained=not args.no_pretrained,
    )

    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)

    if not args.disable_best_acc_cache:
        ckpt_dir = os.path.join(args.output_dir, args.dataset)
        parsed_files = []
        best_acc = 0.0
        for file in os.listdir(ckpt_dir):
            if prefix_saved_mode in file and "checkpoint" in file:
                file_acc = extract_checkpoint_acc(file, prefix_saved_mode)
                if file_acc is None:
                    continue
                parsed_files.append((file, file_acc))
                best_acc = max(best_acc, file_acc)
        for file, file_acc in parsed_files:
            if file_acc < best_acc:
                os.remove(os.path.join(ckpt_dir, file))

    num_params = count_parameters(model)
    logger.info("backbone=%s", model.backbone_cfg.timm_name)
    logger.info("%s", model.backbone_cfg)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM", num_params)

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def pool_patch_tokens(tokens: torch.Tensor, out_size: int) -> torch.Tensor:
    """Pool patch tokens [B, N, C] to [B, out_size*out_size, C] on a 2D grid."""
    bsz, num_tokens, channels = tokens.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Patch token count ({num_tokens}) is not a perfect square.")

    feat = tokens.transpose(1, 2).contiguous().view(bsz, channels, side, side)
    feat = F.adaptive_avg_pool2d(feat, (out_size, out_size))
    feat = feat.flatten(2).transpose(1, 2).contiguous()
    return feat


def valid(args, model, writer, test_loader, global_step, ad_net_local, cp_mask):
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    ad_net_local.eval()

    all_preds, all_label = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0],
    )
    loss_fct = CrossEntropyLoss()

    for _, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            logits, _, _, _ = model(x_s=x, ad_net=ad_net_local, cp_mask=cp_mask, optimal_flag=args.optimal)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == "visda17":
        accuracy, classwise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)
        classwise_acc = None

    logger.info("\nValidation Results of: %s", args.name)
    logger.info("Global Steps: %d", global_step)
    logger.info("Valid Loss: %2.5f", eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f", accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    if classwise_acc is not None:
        writer.add_text("test/classwise_acc", str(classwise_acc), global_step=global_step)

    return accuracy, classwise_acc


def train(args, model, cp_mask, prefix_saved_mode):
    best_acc = 0.0
    ckpt_dir = os.path.join(args.output_dir, args.dataset)

    if not args.disable_best_acc_cache:
        for file in os.listdir(ckpt_dir):
            if prefix_saved_mode in file and "checkpoint" in file:
                file_acc = extract_checkpoint_acc(file, prefix_saved_mode)
                if file_acc is None:
                    continue
                best_acc = max(best_acc, file_acc)
    else:
        logger.info("Best-acc checkpoint cache disabled. Start from best_acc=0.0 for this run.")

    writer = None
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)

    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode="RGB"),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode="RGB"),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode="RGB"),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
    )

    config = model.backbone_cfg

    ad_net_local = AdversarialNetwork(config.head_dim, config.head_dim).to(args.device)

    global_tokens = args.ta_global_pool * args.ta_global_pool
    ad_net_global = GraphConvDiscriminator(
        in_features=config.hidden_size,
        in_dim=global_tokens * config.hidden_size,
        out_dim=global_tokens,
        drop_rat=args.ta_graph_drop,
        n=args.train_batch_size,
        pool_shape=(args.ta_global_pool, config.hidden_size),
    ).to(args.device)

    optimizer_ad = torch.optim.SGD(
        list(ad_net_local.parameters()) + list(ad_net_global.parameters()),
        lr=args.learning_rate / 10,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.transformer.parameters(), "lr": args.adamw_backbone_lr},
                {"params": model.head.parameters(), "lr": args.adamw_head_lr},
            ],
            weight_decay=args.adamw_weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.transformer.parameters(), "lr": args.learning_rate / 10},
                {"params": model.head.parameters()},
            ],
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    ad_net_local.zero_grad()
    ad_net_global.zero_grad()

    set_seed(args)

    best_classwise_acc = ""
    len_source = len(source_loader)
    len_target = len(target_loader)

    focal_loss = FocalLoss(args.num_classes)
    kl_div = KLDivLoss(reduction="batchmean")
    loss_fct = CrossEntropyLoss()

    for global_step in range(1, t_total):
        model.train()
        ad_net_local.train()
        ad_net_global.train()

        if (global_step - 1) % (len_source - 1) == 0:
            iter_source = iter(source_loader)
        if (global_step - 1) % (len_target - 1) == 0:
            iter_target = iter(target_loader)

        data_source = next(iter_source)
        data_target = next(iter_target)

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, _, _ = tuple(t.to(args.device) for t in data_target)

        active_cp = cp_mask if args.use_cp else torch.ones_like(cp_mask, device=args.device)

        logits_s, logits_t, loss_ad_local, _, x_s_tokens, x_t_tokens, _ = model(
            x_s=x_s,
            x_t=x_t,
            ad_net=ad_net_local,
            cp_mask=active_cp,
            optimal_flag=args.optimal,
        )

        source_cls = x_s_tokens[:, 0]
        target_cls = x_t_tokens[:, 0]

        source_patch = x_s_tokens[:, model.prefix_tokens :, :]
        target_patch = x_t_tokens[:, model.prefix_tokens :, :]
        source_patch = pool_patch_tokens(source_patch, args.ta_global_pool)
        target_patch = pool_patch_tokens(target_patch, args.ta_global_pool)

        loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))
        loss_ad_global = adv_global(focal_loss, source_patch, target_patch, ad_net_global)
        loss_kl = kl_div(torch.log_softmax(source_cls, dim=1), torch.softmax(target_cls, dim=1))

        loss = loss_clc + args.gamma * loss_ad_local + args.beta * loss_ad_global + args.theta * loss_kl

        loss_jfpd = torch.zeros((), device=args.device)
        jfpd_stats = {"d_feat": 0.0, "d_pred": 0.0, "psi": 0.0, "phi": 0.0}
        if args.use_jfpd:
            source_feat = source_cls.detach()
            source_prob = torch.softmax(logits_s.detach(), dim=-1)
            target_feat = target_cls
            target_prob = torch.softmax(logits_t, dim=-1)

            proto_feat, proto_prob, valid_source = build_source_prototypes(
                source_feat=source_feat,
                source_prob=source_prob,
                source_label=y_s,
                num_classes=args.num_classes,
            )

            pseudo = torch.argmax(target_prob, dim=-1)
            valid_target = valid_source[pseudo]
            if valid_target.any():
                zs = proto_feat[pseudo[valid_target]]
                ps = proto_prob[pseudo[valid_target]]
                loss_jfpd, jfpd_stats = jfpd_loss(
                    ft=target_feat[valid_target],
                    pt=target_prob[valid_target],
                    zs=zs,
                    ps=ps,
                    alpha=args.jfpd_alpha,
                    mode=args.jfpd_mode,
                )
                loss = loss + args.jfpd_lambda * loss_jfpd

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_global.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()

        if writer is not None:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc", scalar_value=loss_clc.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_local", scalar_value=loss_ad_local.item(), global_step=global_step)
            writer.add_scalar("train/loss_kl", scalar_value=loss_kl.item(), global_step=global_step)
            writer.add_scalar("train/loss_jfpd", scalar_value=loss_jfpd.item(), global_step=global_step)
            writer.add_scalar("train/jfpd_d_feat", scalar_value=jfpd_stats["d_feat"], global_step=global_step)
            writer.add_scalar("train/jfpd_d_pred", scalar_value=jfpd_stats["d_pred"], global_step=global_step)
            writer.add_scalar("train/jfpd_psi", scalar_value=jfpd_stats["psi"], global_step=global_step)
            writer.add_scalar("train/jfpd_phi", scalar_value=jfpd_stats["phi"], global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classwise_acc = valid(args, model, writer, test_loader, global_step, ad_net_local, cp_mask)
            if best_acc < accuracy:
                best_acc = accuracy

                prefix = prefix_saved_mode + str(best_acc) + "_"
                save_model(args, model, prefix, is_adv=False)
                save_model(args, ad_net_local, prefix + "local_", is_adv=True)
                save_model(args, ad_net_global, prefix + "global_", is_adv=True)

                if not args.disable_best_acc_cache:
                    for file in os.listdir(ckpt_dir):
                        if prefix_saved_mode in file and "checkpoint" in file:
                            file_acc = extract_checkpoint_acc(file, prefix_saved_mode)
                            if file_acc is None:
                                continue
                            if best_acc > file_acc:
                                os.remove(os.path.join(ckpt_dir, file))

                if classwise_acc is not None:
                    best_classwise_acc = classwise_acc

            model.train()
            ad_net_local.train()
            ad_net_global.train()

            logger.info("Current Best Accuracy: %2.5f", best_acc)
            logger.info("Current Best element-wise acc: %s", best_classwise_acc)

    if writer is not None:
        writer.close()

    logger.info("Best Accuracy: \t%f", best_acc)
    logger.info("Best element-wise Accuracy: \t%s", best_classwise_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="qs", type=str, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="DomainNet", type=str, help="Which downstream task.")
    parser.add_argument("--source_list", default="data/DomainNet/quickdraw_train.txt", type=str, help="Path of source training data list.")
    parser.add_argument("--target_list", default="data/DomainNet/sketch_train.txt", type=str, help="Path of target training data list.")
    parser.add_argument("--test_list", default="data/DomainNet/sketch_test.txt", type=str, help="Path of target test data list.")
    parser.add_argument("--num_classes", default=345, type=int, help="Number of classes in the dataset.")

    parser.add_argument("--timm_model", type=str, default="", help="timm model name (e.g. vit_base_patch16_224).")
    parser.add_argument("--model_type", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_pretrained", default=False, action="store_true", help="Disable timm pretrained weights.")
    parser.add_argument("--pretrained_dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=256, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run validation every N steps.")

    parser.add_argument("--beta", default=0.1, type=float, help="Weight for TransAdapter-style global graph alignment loss.")
    parser.add_argument("--gamma", default=0.1, type=float, help="Weight for local adversarial loss.")
    parser.add_argument("--theta", default=0.1, type=float, help="Weight for feature KL transfer loss.")

    parser.add_argument("--use_jfpd", default=False, action="store_true", help="Use JFPD loss on target features with source prototypes.")
    parser.add_argument("--jfpd_lambda", default=0.1, type=float, help="The importance of the JFPD loss.")
    parser.add_argument("--jfpd_alpha", default=0.5, type=float, help="Alpha for JFPD combination of feature/prediction divergence.")
    parser.add_argument("--jfpd_mode", choices=["jfpd", "fgpd", "pgfd"], default="jfpd", help="JFPD mode.")

    parser.add_argument("--msa_layer", default=12, type=int, help="Layer index for local alignment feature extraction.")
    parser.add_argument("--ta_global_pool", default=7, type=int, help="Patch-grid pool size before global graph alignment.")
    parser.add_argument("--ta_graph_drop", default=0.1, type=float, help="Dropout rate used inside graph discriminator.")

    parser.add_argument("--is_test", default=False, action="store_true", help="If in test mode.")
    parser.add_argument("--disable_best_acc_cache", default=False, action="store_true", help="Ignore existing checkpoints for best_acc initialization and cleanup.")

    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd", help="Optimizer for the main model parameters.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="The initial learning rate for SGD optimizers.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay for SGD optimizers.")
    parser.add_argument("--adamw_backbone_lr", default=5e-5, type=float, help="Backbone learning rate when --optimizer adamw.")
    parser.add_argument("--adamw_head_lr", default=5e-4, type=float, help="Head learning rate when --optimizer adamw.")
    parser.add_argument("--adamw_weight_decay", default=0.01, type=float, help="Weight decay when --optimizer adamw.")
    parser.add_argument("--num_steps", default=5000, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O2", help="For fp16: Apex AMP optimization level.")
    parser.add_argument("--loss_scale", type=float, default=0, help="Loss scaling for fp16 (unused).")
    parser.add_argument("--gpu_id", default="1", type=str, help="gpu id")

    # Compatibility flags: kept to preserve existing scripts/CLI contract, but no FFTAT behavior remains.
    parser.add_argument("--use_cp", default=False, action="store_true", help="Compatibility flag. CP mask no longer affects loss terms.")
    parser.add_argument("--optimal", default=0, type=int, help="Compatibility flag. Kept for CLI stability.")
    parser.add_argument("--use_im", default=False, action="store_true", help="Compatibility flag. FFTAT IM loss has been removed.")

    args = parser.parse_args()

    if not (0.0 <= args.jfpd_alpha <= 1.0):
        raise ValueError("--jfpd_alpha must be in [0, 1].")
    if args.ta_global_pool < 1:
        raise ValueError("--ta_global_pool must be >= 1.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    if args.use_im:
        logger.warning("`--use_im` is ignored. FFTAT IM loss has been removed in the new pipeline.")

    set_seed(args)

    cp_mode = "CPCompatOn" if args.use_cp else "CPCompatOff"
    prefix_saved_mode = f"{args.name}_{cp_mode}_Perturbation_{args.optimal}_"

    args, model = setup(args, prefix_saved_mode)
    model.to(args.device)

    num_patches = (args.img_size // model.patch_size) ** 2
    cp_size = num_patches + model.prefix_tokens
    cp_mask = torch.ones((cp_size, cp_size), device=args.device).float()

    train(args, model, cp_mask, prefix_saved_mode)


if __name__ == "__main__":
    main()
