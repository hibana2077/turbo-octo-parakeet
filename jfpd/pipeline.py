from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .config import JFPDConfig
from .data import DynamicPrototypeSource, build_loader, get_class_names, load_dataset_splits
from .model import JFPDNet
from .training import adapt_one_epoch, build_optimizer, evaluate, evaluate_with_diagnostics, summarize_collapse_risk, train_source_epoch
from .utils import print_stats, save_checkpoint, save_json, select_device, set_seed


def _build_label_space_report(splits: Dict, reference_label_names: list) -> Dict:
    report = {
        "reference_label_names": reference_label_names,
        "consistent_across_splits": True,
        "splits": {},
    }

    for split_name, split_dataset in splits.items():
        split_label_names = get_class_names(split_dataset, "label", fallback=reference_label_names)
        label_hist = [0] * len(split_label_names)
        for label in split_dataset["label"]:
            label_hist[int(label)] += 1

        matches_reference = split_label_names == reference_label_names
        if not matches_reference:
            report["consistent_across_splits"] = False

        report["splits"][split_name] = {
            "num_classes": len(split_label_names),
            "label_names": split_label_names,
            "label_hist": label_hist,
            "matches_source_train": matches_reference,
        }

    return report


def _print_eval_diagnostics(prefix: str, diagnostics: Dict) -> None:
    print(
        f"{prefix}: mean_max_prob={diagnostics['mean_max_prob']:.4f}, "
        f"dominant_pred_class={diagnostics['dominant_pred_class']}, "
        f"dominant_pred_ratio={diagnostics['dominant_pred_ratio']:.4f}, "
        f"collapse_suspected={diagnostics['collapse_suspected']}"
    )
    print(f"{prefix}: pred_hist_top={diagnostics['pred_hist_top']}")
    print(f"{prefix}: label_hist_top={diagnostics['label_hist_top']}")


class JFPDTrainer:
    def __init__(self, config: JFPDConfig) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        cfg = self.config
        if cfg.source_domain == cfg.target_domain:
            raise ValueError("source-domain and target-domain must be different.")

        set_seed(cfg.seed)

        limits = {
            "source_train": cfg.max_source_train_samples,
            "target_train": cfg.max_target_train_samples,
            "target_test": cfg.max_target_test_samples,
            "source_test": cfg.max_source_test_samples,
        }
        splits = load_dataset_splits(
            dataset_name=cfg.dataset_name,
            dataset_root=cfg.dataset_root,
            cache_dir=cfg.cache_dir,
            source_domain=cfg.source_domain,
            target_domain=cfg.target_domain,
            limits=limits,
            train_split_ratio=cfg.train_split_ratio,
            seed=cfg.seed,
            class_limit=cfg.class_limit,
        )

        label_names = get_class_names(splits["source_train"], "label")
        num_classes = len(label_names)
        label_space_report = _build_label_space_report(splits, label_names)
        save_json(self.output_dir / "label_space.json", label_space_report)
        if not label_space_report["consistent_across_splits"]:
            raise RuntimeError("Label mapping mismatch detected across dataset splits. See label_space.json for details.")

        print(f"device={self.device}")
        for split_name, split_dataset in splits.items():
            print(f"{split_name}_samples={len(split_dataset)}")

        source_train_loader = build_loader(
            dataset=splits["source_train"],
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            is_train=True,
        )
        target_train_loader = build_loader(
            dataset=splits["target_train"],
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            is_train=True,
        )
        target_test_loader = build_loader(
            dataset=splits["target_test"],
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            is_train=False,
        )
        source_test_loader = build_loader(
            dataset=splits["source_test"],
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            is_train=False,
        )

        model = JFPDNet(model_name=cfg.model_name, num_classes=num_classes).to(self.device)
        prototype_source = DynamicPrototypeSource(
            dataset=splits["source_train"],
            image_size=cfg.image_size,
            num_classes=num_classes,
        )
        history = {
            "config": asdict(cfg),
            "num_classes": num_classes,
            "label_names": label_names,
            "source_domain": cfg.source_domain,
            "target_domain": cfg.target_domain,
            "source_pretrain": [],
            "adaptation": [],
        }

        source_optimizer = build_optimizer(model, lr=cfg.source_lr, weight_decay=cfg.weight_decay)
        best_source_acc = -1.0

        for epoch in range(1, cfg.source_epochs + 1):
            train_stats = train_source_epoch(model, source_train_loader, source_optimizer, self.device)
            print_stats(f"source_train_epoch_{epoch}", train_stats)
            epoch_record = {"epoch": epoch, "train": asdict(train_stats)}

            if cfg.eval_source:
                if cfg.debug_bug4:
                    eval_stats, eval_diag = evaluate_with_diagnostics(
                        model=model,
                        loader=source_test_loader,
                        device=self.device,
                        num_classes=num_classes,
                        label_names=label_names,
                    )
                    _print_eval_diagnostics(f"source_test_epoch_{epoch}_diag", eval_diag)
                    save_json(self.output_dir / f"source_test_epoch_{epoch}_diagnostics.json", eval_diag)
                else:
                    eval_stats = evaluate(model, source_test_loader, self.device)
                print_stats(f"source_test_epoch_{epoch}", eval_stats)
                epoch_record["eval"] = asdict(eval_stats)
                if eval_stats.accuracy is not None and eval_stats.accuracy > best_source_acc:
                    best_source_acc = eval_stats.accuracy
                    save_checkpoint(
                        self.output_dir / "best_source.pt",
                        model,
                        {"epoch": epoch, "accuracy": eval_stats.accuracy, "stage": "source"},
                    )

            history["source_pretrain"].append(epoch_record)
            save_json(self.output_dir / "history.json", history)

        missing_classes = (~prototype_source.valid_mask).nonzero(as_tuple=False).view(-1).tolist()
        if missing_classes:
            missing_names = [label_names[index] for index in missing_classes]
            print(f"warning: missing source prototypes for {len(missing_names)} classes")

        save_json(
            self.output_dir / "prototype_source.json",
            {
                "mode": "dynamic_source_only",
                "samples_per_class": cfg.proto_samples_per_class,
                "valid_mask": prototype_source.valid_mask.tolist(),
                "missing_classes": missing_classes,
                "label_names": label_names,
            },
        )

        if cfg.freeze_classifier_during_adapt:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = False
            print("adaptation: classifier_head=frozen")
        else:
            print("adaptation: classifier_head=trainable")
        print(
            "adaptation safeguards: "
            f"pseudo_confidence_threshold={cfg.pseudo_confidence_threshold}, "
            f"source_anchor_weight={cfg.source_anchor_weight}, "
            f"max_pseudo_per_class={cfg.max_pseudo_per_class}"
        )
        adapt_optimizer = build_optimizer(model, lr=cfg.adapt_lr, weight_decay=cfg.weight_decay)
        best_target_acc = -1.0

        if cfg.debug_collapse:
            pre_adapt_stats, pre_adapt_diag = evaluate_with_diagnostics(
                model=model,
                loader=target_test_loader,
                device=self.device,
                num_classes=num_classes,
                label_names=label_names,
            )
            print_stats("target_test_pre_adapt", pre_adapt_stats)
            _print_eval_diagnostics("target_test_pre_adapt_diag", pre_adapt_diag)
            save_json(self.output_dir / "target_test_pre_adapt_diagnostics.json", pre_adapt_diag)

            collapse_risk = summarize_collapse_risk(
                model=model,
                prototype_source=prototype_source,
                num_classes=num_classes,
                samples_per_class=cfg.proto_samples_per_class,
                forward_batch_size=cfg.proto_forward_batch_size,
                device=self.device,
                label_names=label_names,
            )
            print(f"collapse_risk: class0_classifier_bias={collapse_risk['class0_classifier_bias']}")
            print(f"collapse_risk: class0_classifier_weight_norm={collapse_risk['class0_classifier_weight_norm']}")
            print(f"collapse_risk: classifier_bias_top={collapse_risk['classifier_bias_top']}")
            print(f"collapse_risk: classifier_weight_norm_top={collapse_risk['classifier_weight_norm_top']}")
            print(f"collapse_risk: class0_source_feat_proto_norm={collapse_risk['class0_source_feat_proto_norm']}")
            print(f"collapse_risk: class0_source_prob_proto_peak={collapse_risk['class0_source_prob_proto_peak']}")
            print(f"collapse_risk: source_feat_proto_norm_top={collapse_risk['source_feat_proto_norm_top']}")
            print(f"collapse_risk: source_prob_proto_peak_top={collapse_risk['source_prob_proto_peak_top']}")
            if "class0_source_prob_proto_top" in collapse_risk:
                print(f"collapse_risk: class0_source_prob_proto_top={collapse_risk['class0_source_prob_proto_top']}")
            save_json(self.output_dir / "collapse_risk.json", collapse_risk)

        for epoch in range(1, cfg.adapt_epochs + 1):
            adapt_stats, adapt_diag = adapt_one_epoch(
                model=model,
                loader=target_train_loader,
                prototype_source=prototype_source,
                source_loader=source_train_loader,
                num_classes=num_classes,
                optimizer=adapt_optimizer,
                device=self.device,
                alpha=cfg.alpha,
                loss_mode=cfg.loss_mode,
                proto_samples_per_class=cfg.proto_samples_per_class,
                proto_forward_batch_size=cfg.proto_forward_batch_size,
                pseudo_confidence_threshold=cfg.pseudo_confidence_threshold,
                source_anchor_weight=cfg.source_anchor_weight,
                max_pseudo_per_class=cfg.max_pseudo_per_class,
                epoch=epoch,
                debug_bug2=cfg.debug_bug2,
                debug_collapse=cfg.debug_collapse,
                label_names=label_names,
            )
            print_stats(f"adapt_epoch_{epoch}", adapt_stats)
            if cfg.debug_collapse:
                save_json(self.output_dir / f"adapt_epoch_{epoch}_collapse_diagnostics.json", adapt_diag)

            if cfg.debug_bug4:
                eval_stats, eval_diag = evaluate_with_diagnostics(
                    model=model,
                    loader=target_test_loader,
                    device=self.device,
                    num_classes=num_classes,
                    label_names=label_names,
                )
                _print_eval_diagnostics(f"target_test_epoch_{epoch}_diag", eval_diag)
                save_json(self.output_dir / f"target_test_epoch_{epoch}_diagnostics.json", eval_diag)
            else:
                eval_stats = evaluate(model, target_test_loader, self.device)
            print_stats(f"target_test_epoch_{epoch}", eval_stats)

            epoch_record = {"epoch": epoch, "adapt": asdict(adapt_stats), "eval": asdict(eval_stats)}
            history["adaptation"].append(epoch_record)
            save_json(self.output_dir / "history.json", history)

            if eval_stats.accuracy is not None and eval_stats.accuracy > best_target_acc:
                best_target_acc = eval_stats.accuracy
                save_checkpoint(
                    self.output_dir / "best_target.pt",
                    model,
                    {"epoch": epoch, "accuracy": eval_stats.accuracy, "stage": "adapt"},
                )

        save_checkpoint(self.output_dir / "last.pt", model, {"stage": "final"})
        print(f"artifacts={self.output_dir.resolve()}")
        return history


def run_jfpd_training(config: JFPDConfig) -> Dict:
    return JFPDTrainer(config).run()
