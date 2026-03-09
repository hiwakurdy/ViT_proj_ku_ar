import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from src.projections import extract_projection_profiles
from src.transforms import image_to_tensor, preprocess_image


def extract_source_group(filename_stem: str, pattern: str) -> str:
    match = re.match(pattern, filename_stem)
    return match.group(1) if match else filename_stem


def normalize_allowed_extensions(extensions: Iterable[str]) -> set:
    return {str(ext).lower() for ext in extensions}


def discover_dataset_records(config: Dict) -> List[Dict]:
    data_config = config["data"]
    split_config = config["split"]
    labels = data_config["labels"]
    allowed_extensions = normalize_allowed_extensions(data_config["allowed_extensions"])
    group_pattern = split_config["group_regex"]

    records = []
    for label_name, label_id in labels.items():
        root_dir = Path(data_config[f"{label_name}_dir"])
        if not root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        for image_path in root_dir.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in allowed_extensions:
                continue

            source_group = extract_source_group(image_path.stem, group_pattern)
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": int(label_id),
                    "label_name": label_name,
                    "source_group": source_group,
                }
            )

    if not records:
        raise RuntimeError("No dataset files were discovered for the configured directories.")

    return records


def compute_group_split_counts(group_count: int, split_config: Dict) -> Dict[str, int]:
    ratios = {
        "train": float(split_config["train_ratio"]),
        "val": float(split_config["val_ratio"]),
        "test": float(split_config["test_ratio"]),
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    raw_counts = {name: (group_count * ratio / ratio_sum) for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    remainder = group_count - sum(counts.values())

    split_priority = {"val": 2, "test": 1, "train": 0}
    ordered = sorted(
        raw_counts.items(),
        key=lambda item: (item[1] - counts[item[0]], split_priority[item[0]]),
        reverse=True,
    )
    for index in range(remainder):
        split_name = ordered[index % len(ordered)][0]
        counts[split_name] += 1

    return counts


def assign_group_splits(records: List[Dict], split_config: Dict) -> Dict[str, List[Dict]]:
    groups = sorted({record["source_group"] for record in records})
    rng = random.Random(int(split_config["seed"]))
    rng.shuffle(groups)

    counts = compute_group_split_counts(len(groups), split_config)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    group_to_split = {}

    for index, group_name in enumerate(groups):
        if index < train_end:
            group_to_split[group_name] = "train"
        elif index < val_end:
            group_to_split[group_name] = "val"
        else:
            group_to_split[group_name] = "test"

    split_records = {"train": [], "val": [], "test": []}
    for record in records:
        split_name = group_to_split[record["source_group"]]
        enriched_record = dict(record)
        enriched_record["split"] = split_name
        split_records[split_name].append(enriched_record)

    return split_records


def write_split_csvs(split_records: Dict[str, List[Dict]], config: Dict) -> Dict[str, Path]:
    data_config = config["data"]
    csv_paths = {
        "train": Path(data_config["train_csv"]),
        "val": Path(data_config["val_csv"]),
        "test": Path(data_config["test_csv"]),
    }

    header = ["image_path", "label", "label_name", "source_group", "split"]
    for split_name, records in split_records.items():
        csv_path = csv_paths[split_name]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(records)

    return csv_paths


def ensure_dataset_splits(config: Dict) -> Dict[str, Path]:
    data_config = config["data"]
    split_config = config["split"]
    csv_paths = {
        "train": Path(data_config["train_csv"]),
        "val": Path(data_config["val_csv"]),
        "test": Path(data_config["test_csv"]),
    }

    if not bool(split_config.get("rebuild_csv", False)) and all(path.exists() for path in csv_paths.values()):
        return csv_paths

    records = discover_dataset_records(config)
    split_records = assign_group_splits(records, split_config)
    return write_split_csvs(split_records, config)


def load_csv_records(csv_path: Path) -> List[Dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


class KurdishArabicDataset(Dataset):
    def __init__(self, csv_path: Path, config: Dict, is_train: bool = False):
        self.csv_path = Path(csv_path)
        self.records = load_csv_records(self.csv_path)
        self.transform_config = config["transforms"]
        self.projection_config = config["projection"]
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        processed = preprocess_image(record["image_path"], self.transform_config)
        image_tensor = image_to_tensor(processed, self.transform_config)
        h_proj, v_proj = extract_projection_profiles(
            image=processed,
            h_proj_length=int(self.projection_config["h_proj_length"]),
            v_proj_length=int(self.projection_config["v_proj_length"]),
            input_mode=str(self.projection_config.get("input_mode", "binary_count")),
        )

        return {
            "image": image_tensor,
            "h_proj": torch.from_numpy(h_proj),
            "v_proj": torch.from_numpy(v_proj),
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "path": record["image_path"],
            "source_group": record["source_group"],
        }


def build_dataloader(csv_path: Path, config: Dict, is_train: bool) -> DataLoader:
    trainer_config = config["trainer"]
    dataset = KurdishArabicDataset(csv_path=csv_path, config=config, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=int(trainer_config["batch_size"]),
        shuffle=is_train,
        num_workers=int(trainer_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    csv_paths = ensure_dataset_splits(config)
    return {
        "train": build_dataloader(csv_paths["train"], config=config, is_train=True),
        "val": build_dataloader(csv_paths["val"], config=config, is_train=False),
        "test": build_dataloader(csv_paths["test"], config=config, is_train=False),
    }


def summarize_split_distribution(csv_path: Path) -> Dict:
    records = load_csv_records(csv_path)
    label_counts = Counter(record["label_name"] for record in records)
    group_counts = Counter(record["source_group"] for record in records)
    return {
        "samples": len(records),
        "labels": dict(label_counts),
        "groups": len(group_counts),
    }
