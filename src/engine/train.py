import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.engine.evaluate import evaluate_model, save_metrics
from src.utils.checkpoint import save_checkpoint


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    scaler,
    mixed_precision: bool,
    grad_clip_norm: Optional[float],
    logger=None,
    epoch: int = 0,
) -> Dict:
    model.train()
    running_loss = 0.0
    sample_count = 0
    predictions = []
    targets = []

    iterator = tqdm(data_loader, desc=f"train:{epoch}", leave=False)
    for batch in iterator:
        image = batch["image"].to(device)
        h_proj = batch["h_proj"].to(device)
        v_proj = batch["v_proj"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=mixed_precision and device.type == "cuda"):
            output = model(image=image, h_proj=h_proj, v_proj=v_proj)
            logits = output["logits"]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = running_loss / sample_count if sample_count else 0.0
    accuracy = sum(int(p == t) for p, t in zip(predictions, targets)) / len(targets) if targets else 0.0

    if logger:
        logger.info("epoch=%s train loss=%.4f acc=%.4f", epoch, average_loss, accuracy)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    config: Dict,
    label_names: List[str],
    output_dir: Path,
    logger=None,
):
    trainer_config = config["trainer"]
    metric_name = str(trainer_config.get("checkpoint_metric", "macro_f1"))
    early_stopping_patience = int(trainer_config.get("early_stopping_patience", 3))
    mixed_precision = bool(trainer_config.get("mixed_precision", False))
    grad_clip_norm = trainer_config.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("-inf")
    epochs_without_improvement = 0
    history = []
    best_checkpoint_path = checkpoint_dir / "best.pt"
    scaler = GradScaler(enabled=mixed_precision and device.type == "cuda")

    for epoch in range(1, int(trainer_config["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
            grad_clip_norm=grad_clip_norm,
            logger=logger,
            epoch=epoch,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            label_names=label_names,
            split_name="val",
            logger=logger,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_summary)

        current_metric = float(val_metrics[metric_name])
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
            save_metrics(val_metrics, metrics_dir, split_name="val", label_names=label_names)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if logger:
                    logger.info("Early stopping triggered after epoch %s.", epoch)
                break

    history_path = metrics_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return best_checkpoint_path
