#!/usr/bin/env python3

import argparse

from jfpd.losses import entropy_from_prob, jfpd_loss, normalized_feature_divergence


DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


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


def args_to_config(args: argparse.Namespace):
    from jfpd.config import JFPDConfig

    return JFPDConfig(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
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
        proto_samples_per_class=args.proto_samples_per_class,
        proto_forward_batch_size=args.proto_forward_batch_size,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        max_source_train_samples=args.max_source_train_samples,
        max_target_train_samples=args.max_target_train_samples,
        max_target_test_samples=args.max_target_test_samples,
        max_source_test_samples=args.max_source_test_samples,
        eval_source=args.eval_source,
    )


def main() -> None:
    from jfpd.pipeline import run_jfpd_training

    args = parse_args()
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
