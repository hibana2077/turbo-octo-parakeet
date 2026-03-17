from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .config import JFPDConfig
from .data import DynamicPrototypeSource, build_loader, get_class_names, load_dataset_splits
from .model import JFPDNet
from .training import EMAPrototypeBank, adapt_one_epoch, build_optimizer, evaluate, train_source_epoch
from .utils import print_stats, save_checkpoint, save_json, select_device, set_seed


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
                "mode": "dynamic_ema",
                "samples_per_class": cfg.proto_samples_per_class,
                "ema_decay": cfg.proto_ema_decay,
                "valid_mask": prototype_source.valid_mask.tolist(),
                "missing_classes": missing_classes,
                "label_names": label_names,
            },
        )

        adapt_optimizer = build_optimizer(model, lr=cfg.adapt_lr, weight_decay=cfg.weight_decay)
        best_target_acc = -1.0
        prototype_bank = EMAPrototypeBank.create(
            num_classes=num_classes,
            feat_dim=model.classifier.in_features,
            decay=cfg.proto_ema_decay,
            device=self.device,
        )

        for epoch in range(1, cfg.adapt_epochs + 1):
            adapt_stats = adapt_one_epoch(
                model=model,
                loader=target_train_loader,
                prototype_source=prototype_source,
                num_classes=num_classes,
                optimizer=adapt_optimizer,
                device=self.device,
                alpha=cfg.alpha,
                proto_samples_per_class=cfg.proto_samples_per_class,
                proto_forward_batch_size=cfg.proto_forward_batch_size,
                prototype_bank=prototype_bank,
                epoch=epoch,
            )
            print_stats(f"adapt_epoch_{epoch}", adapt_stats)

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
