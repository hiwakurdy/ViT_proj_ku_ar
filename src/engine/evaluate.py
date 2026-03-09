import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from src.engine.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    label_names: List[str],
    split_name: str,
    logger=None,
) -> Dict:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    predictions = []
    targets = []

    iterator = tqdm(data_loader, desc=f"evaluate:{split_name}", leave=False)
    for batch in iterator:
        image = batch["image"].to(device)
        h_proj = batch["h_proj"].to(device)
        v_proj = batch["v_proj"].to(device)
        labels = batch["label"].to(device)

        output = model(image=image, h_proj=h_proj, v_proj=v_proj)
        logits = output["logits"]
        loss = criterion(logits, labels) if criterion is not None else None

        if loss is not None:
            batch_size = int(labels.size(0))
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(
        y_true=targets,
        y_pred=predictions,
        num_classes=len(label_names),
        label_names=label_names,
    )
    metrics["loss"] = (running_loss / sample_count) if sample_count else 0.0
    metrics["split"] = split_name

    if logger:
        logger.info(
            "%s loss=%.4f acc=%.4f macro_f1=%.4f",
            split_name,
            metrics["loss"],
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    return metrics


def save_metrics(metrics: Dict, output_dir: Path, split_name: str, label_names: Optional[List[str]] = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{split_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    confusion_path = output_dir / f"{split_name}_confusion_matrix.csv"
    label_names = label_names or list(metrics.get("per_class", {}).keys())
    with confusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + list(label_names))
        for label_name, row in zip(label_names, metrics["confusion_matrix"]):
            writer.writerow([label_name] + row)

    return metrics_path
