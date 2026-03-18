#!/usr/bin/env python3

import argparse

from jfpd.losses import entropy_from_prob, jfpd_loss, normalized_feature_divergence
from jfpd.config import DOMAINNET_DOMAINS, OFFICEHOME_DOMAINS, is_officehome_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Version B JFPD pipeline for domain adaptation datasets.")
    parser.add_argument("--dataset-name", default="wltjr1007/DomainNet")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--train-split-ratio", type=float, default=0.8)
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
    parser.add_argument("--loss-mode", choices=("jfpd", "fgpd", "pgfd"), default="jfpd")
    parser.add_argument("--proto-samples-per-class", type=int, default=32)
    parser.add_argument("--proto-forward-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/jfpd")
    parser.add_argument("--max-source-train-samples", type=int, default=None)
    parser.add_argument("--max-target-train-samples", type=int, default=None)
    parser.add_argument("--max-target-test-samples", type=int, default=None)
    parser.add_argument("--max-source-test-samples", type=int, default=None)
    parser.add_argument("--class-limit", type=int, default=None)
    parser.add_argument("--eval-source", action="store_true")
    parser.add_argument("--source-anchor-weight", type=float, default=0.1)
    parser.add_argument("--source-anchor-come-from", choices=("source", "target"), default="target")
    parser.add_argument("--max-pseudo-per-class", type=int, default=8)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.train_split_ratio < 1.0):
        raise ValueError("--train-split-ratio must be between 0 and 1.")

    valid_domains = OFFICEHOME_DOMAINS if is_officehome_dataset(args.dataset_name) else DOMAINNET_DOMAINS
    for name, value in (("source", args.source_domain), ("target", args.target_domain)):
        if value not in valid_domains:
            joined = ", ".join(valid_domains)
            raise ValueError(f"--{name}-domain '{value}' is invalid for dataset '{args.dataset_name}'. Expected one of: {joined}")

    if is_officehome_dataset(args.dataset_name) and args.dataset_root is None:
        raise ValueError("--dataset-root is required when --dataset-name is OfficeHome.")

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in the range [0, 1].")

    if args.class_limit is not None and args.class_limit <= 0:
        raise ValueError("--class-limit must be a positive integer.")

    if args.source_anchor_weight < 0.0:
        raise ValueError("--source-anchor-weight must be non-negative.")

    if args.max_pseudo_per_class is not None and args.max_pseudo_per_class <= 0:
        raise ValueError("--max-pseudo-per-class must be a positive integer.")


def args_to_config(args: argparse.Namespace):
    from jfpd.config import JFPDConfig

    return JFPDConfig(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        train_split_ratio=args.train_split_ratio,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        source_epochs=args.source_epochs,
        adapt_epochs=args.adapt_epochs,
        source_lr=args.source_lr,
        adapt_lr=args.adapt_lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        loss_mode=args.loss_mode,
        proto_samples_per_class=args.proto_samples_per_class,
        proto_forward_batch_size=args.proto_forward_batch_size,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        max_source_train_samples=args.max_source_train_samples,
        max_target_train_samples=args.max_target_train_samples,
        max_target_test_samples=args.max_target_test_samples,
        max_source_test_samples=args.max_source_test_samples,
        class_limit=args.class_limit,
        eval_source=args.eval_source,
        source_anchor_weight=args.source_anchor_weight,
        source_anchor_come_from=args.source_anchor_come_from,
        max_pseudo_per_class=args.max_pseudo_per_class,
    )


def main() -> None:
    args = parse_args()
    from jfpd.pipeline import run_jfpd_training

    config = args_to_config(args)
    run_jfpd_training(config)


__all__ = [
    "args_to_config",
    "entropy_from_prob",
    "jfpd_loss",
    "normalized_feature_divergence",
]


if __name__ == "__main__":
    main()
