from pathlib import Path

import torch


def save_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None, epoch=None, metrics=None, config=None):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None, map_location="cpu"):
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
