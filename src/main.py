import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.datasets import build_dataloaders, ensure_dataset_splits
from src.engine.evaluate import evaluate_model, save_metrics
from src.engine.train import train_model
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Kurdish vs Arabic word-image classification.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="train", choices=["train", "evaluate", "index"])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for evaluation or resume.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--rebuild-splits", action="store_true", help="Force regeneration of split CSV files.")
    return parser.parse_args()


def load_config(config_path: str):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_paths(config, config_path: str):
    config_dir = Path(config_path).resolve().parent
    project_root = config_dir if (config_dir / "src").exists() else config_dir.parent

    for key in ["train_csv", "val_csv", "test_csv"]:
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((project_root / path).resolve())

    for label_name in config["data"]["labels"].keys():
        dir_key = f"{label_name}_dir"
        if dir_key in config["data"]:
            path = Path(config["data"][dir_key])
            if not path.is_absolute():
                config["data"][dir_key] = str((project_root / path).resolve())

    output_dir = Path(config["project"]["output_dir"])
    if not output_dir.is_absolute():
        config["project"]["output_dir"] = str((project_root / output_dir).resolve())

    return config


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(config, model):
    optimizer_config = config["optimizer"]
    name = str(optimizer_config["name"]).lower()
    lr = float(optimizer_config["lr"])
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config, optimizer):
    scheduler_config = config["scheduler"]
    name = str(scheduler_config.get("name", "none")).lower()
    epochs = int(config["trainer"]["epochs"])

    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(scheduler_config.get("min_lr", 1.0e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 5)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def main():
    args = parse_args()
    config = resolve_project_paths(load_config(args.config), args.config)
    if args.rebuild_splits:
        config["split"]["rebuild_csv"] = True

    set_seed(int(config["project"]["seed"]))
    csv_paths = ensure_dataset_splits(config)

    output_dir = Path(config["project"]["output_dir"])
    logger = setup_logger(output_dir / "logs", level=config["logging"].get("level", "INFO"))
    logger.info("Using split files: %s", {name: str(path) for name, path in csv_paths.items()})

    if args.mode == "index":
        logger.info("Split CSVs are ready.")
        return

    device = resolve_device(str(config["trainer"].get("device", "cpu")).lower())
    logger.info("Resolved device: %s", device)

    dataloaders = build_dataloaders(config)
    label_names = [name for name, _ in sorted(config["data"]["labels"].items(), key=lambda item: item[1])]
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["trainer"].get("label_smoothing", 0.0)))

    if args.mode == "train":
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        best_checkpoint = train_model(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config,
            label_names=label_names,
            output_dir=output_dir,
            logger=logger,
        )
        logger.info("Best checkpoint saved to %s", best_checkpoint)

        if bool(config["trainer"].get("evaluate_test_after_train", True)):
            checkpoint_payload = load_checkpoint(best_checkpoint, model=model, map_location=device)
            logger.info("Loaded best checkpoint from epoch %s", checkpoint_payload.get("epoch"))
            test_metrics = evaluate_model(
                model=model,
                data_loader=dataloaders["test"],
                criterion=criterion,
                device=device,
                label_names=label_names,
                split_name="test",
                logger=logger,
            )
            save_metrics(test_metrics, output_dir / "metrics", split_name="test", label_names=label_names)
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --mode evaluate is used.")

    load_checkpoint(Path(args.checkpoint), model=model, map_location=device)
    metrics = evaluate_model(
        model=model,
        data_loader=dataloaders[args.split],
        criterion=criterion,
        device=device,
        label_names=label_names,
        split_name=args.split,
        logger=logger,
    )
    save_metrics(metrics, output_dir / "metrics", split_name=args.split, label_names=label_names)


if __name__ == "__main__":
    main()
