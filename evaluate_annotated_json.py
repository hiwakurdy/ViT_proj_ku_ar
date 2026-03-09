import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.engine.metrics import compute_classification_metrics
from src.main import load_config, resolve_device, resolve_project_paths
from src.models import build_model
from src.projections import extract_projection_profiles
from src.transforms import image_to_tensor, preprocess_gray_image
from src.utils.checkpoint import load_checkpoint


AMBIGUOUS_CATEGORY_NAMES = {"arabic_name", "ku_arabic", "ar_kurdish"}
SLICE_PREFIXES = {
    "C": ("C_",),
    "D": ("D_",),
    "F": ("F0_", "F_"),
}
DEFAULT_COLOR_PALETTE = [
    (0, 180, 0),
    (0, 0, 220),
    (200, 80, 0),
    (160, 0, 160),
]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def build_label_metadata(config):
    label_pairs = sorted(config["data"]["labels"].items(), key=lambda item: int(item[1]))
    label_names = [name for name, _ in label_pairs]
    label_ids = {name: int(label_id) for name, label_id in label_pairs}

    canonical_to_model = {}
    for label_name, label_id in label_pairs:
        lower = normalize_name(label_name)
        canonical_to_model[lower] = int(label_id)
        if lower in {"ku", "kurdish"}:
            canonical_to_model["kurdish"] = int(label_id)
            canonical_to_model["ku"] = int(label_id)
        if lower in {"ar", "arabic"}:
            canonical_to_model["arabic"] = int(label_id)
            canonical_to_model["ar"] = int(label_id)

    display_names = {}
    for label_name, label_id in label_pairs:
        lower = normalize_name(label_name)
        if lower in {"ku", "kurdish"}:
            display_names[int(label_id)] = "Kurdish"
        elif lower in {"ar", "arabic"}:
            display_names[int(label_id)] = "Arabic"
        else:
            display_names[int(label_id)] = label_name

    return label_names, label_ids, canonical_to_model, display_names


def build_category_mapping(categories, canonical_to_model):
    category_map = {}
    category_names = {}
    for category in categories:
        category_id = int(category["id"])
        normalized = normalize_name(category["name"])
        category_names[category_id] = normalized
        if normalized in canonical_to_model:
            category_map[category_id] = canonical_to_model[normalized]
        elif normalized in AMBIGUOUS_CATEGORY_NAMES:
            category_map[category_id] = -1
        else:
            category_map[category_id] = -1
    return category_map, category_names


def resolve_image_dir(json_path: Path, image_dir: str = None) -> Path:
    if image_dir:
        resolved = Path(image_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Image directory not found: {resolved}")
        return resolved

    candidates = [
        json_path.with_suffix("") / "images",
        json_path.parent / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not infer image directory for {json_path}")


def get_prediction_color(label_id: int):
    return DEFAULT_COLOR_PALETTE[label_id % len(DEFAULT_COLOR_PALETTE)]


def predict_word_crop(gray_crop, model, config, device):
    processed = preprocess_gray_image(gray_crop, config["transforms"])
    image_tensor = image_to_tensor(processed, config["transforms"]).unsqueeze(0).to(device)
    h_proj, v_proj = extract_projection_profiles(
        image=processed,
        h_proj_length=int(config["projection"]["h_proj_length"]),
        v_proj_length=int(config["projection"]["v_proj_length"]),
        input_mode=str(config["projection"].get("input_mode", "binary_count")),
    )
    h_proj_tensor = torch.from_numpy(h_proj).unsqueeze(0).to(device)
    v_proj_tensor = torch.from_numpy(v_proj).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image=image_tensor, h_proj=h_proj_tensor, v_proj=v_proj_tensor)
        logits = output["logits"]
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)

    pred_label = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_label])
    return pred_label, confidence, probabilities


def smooth_predictions(predictions, window_size: int = 3):
    if window_size <= 1 or len(predictions) <= 1:
        return predictions

    smoothed = []
    half = window_size // 2
    for index, prediction in enumerate(predictions):
        start = max(0, index - half)
        end = min(len(predictions), index + half + 1)
        window = predictions[start:end]
        label_votes = Counter(item["pred_label"] for item in window)
        smoothed_label = label_votes.most_common(1)[0][0]
        label_probs = [item["probs"] for item in window if item["pred_label"] == smoothed_label]
        averaged_probs = np.mean(label_probs if label_probs else [item["probs"] for item in window], axis=0)

        updated = dict(prediction)
        updated["pred_label"] = int(smoothed_label)
        updated["confidence"] = float(averaged_probs[smoothed_label])
        updated["probs"] = averaged_probs.astype(np.float32)
        updated["smoothed"] = int(smoothed_label) != int(prediction["pred_label"])
        smoothed.append(updated)
    return smoothed


def draw_annotated_image(original_image, predictions, display_names, scored_correct, scored_total):
    canvas = original_image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    image_height, image_width = canvas.shape[:2]
    color_correct = (200, 120, 0)
    color_incorrect = (0, 0, 0)
    color_no_gt = (180, 180, 180)
    strip_height = 3

    for prediction in predictions:
        x, y, bw, bh = prediction["bbox"]
        pred_label = int(prediction["pred_label"])
        gt_label = int(prediction["gt_label"])
        confidence = float(prediction["confidence"]) * 100.0
        pred_color = get_prediction_color(pred_label)
        pred_name = display_names[pred_label]

        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), pred_color, 2)

        suffix = "(knn)" if prediction.get("smoothed", False) else ""
        text = f"{pred_name[0]}{suffix}:{confidence:.0f}%"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_y = max(y - 3, text_height + 3)

        cv2.rectangle(
            canvas,
            (x, text_y - text_height - 2),
            (x + text_width + 2, text_y + 1),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x + 1, text_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            pred_color,
            1,
        )

        strip_y = y + bh + 2
        if strip_y + strip_height < image_height:
            if gt_label >= 0:
                strip_color = color_correct if pred_label == gt_label else color_incorrect
            else:
                strip_color = color_no_gt
            cv2.rectangle(canvas, (x, strip_y), (x + bw, strip_y + strip_height), strip_color, -1)

    footer = np.ones((55, image_width, 3), dtype=np.uint8) * 245
    accuracy = (scored_correct / scored_total * 100.0) if scored_total else 0.0
    footer_text = f"Acc: {accuracy:.1f}% | Correct: {scored_correct}/{scored_total}"
    cv2.putText(footer, footer_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([canvas, footer])


def annotated_output_name(file_name: str) -> str:
    path = Path(file_name)
    return f"{path.stem}_annotated.png"


def create_prediction_record(pred_label, confidence, probs, gt_label, category_id, bbox):
    return {
        "pred_label": int(pred_label),
        "confidence": float(confidence),
        "probs": np.asarray(probs, dtype=np.float32),
        "gt_label": int(gt_label),
        "category_id": int(category_id),
        "bbox": [int(value) for value in bbox],
        "smoothed": False,
    }


def summarize_ambiguous_counts(predictions, category_names, display_names):
    summary = {}
    for prediction in predictions:
        if prediction["gt_label"] != -1:
            continue
        category_name = category_names.get(prediction["category_id"], str(prediction["category_id"]))
        category_summary = summary.setdefault(
            category_name,
            {display_names[label_id]: 0 for label_id in sorted(display_names.keys())},
        )
        category_summary[display_names[prediction["pred_label"]]] += 1
    return summary


def evaluate_slice(
    slice_name,
    image_annotations,
    image_map,
    image_dir,
    output_dir,
    model,
    config,
    device,
    display_names,
    category_map,
    category_names,
    knn_window,
):
    slice_dir = output_dir / slice_name
    slice_dir.mkdir(parents=True, exist_ok=True)
    mode_results = {}

    for use_knn in [False, True]:
        mode_name = "knn" if use_knn else "no_knn"
        mode_dir = slice_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        y_true = []
        y_pred = []
        raw_predictions = {}
        ambiguous_summary = defaultdict(Counter)

        for image_id, annotations in image_annotations.items():
            image_info = image_map[image_id]
            image_path = image_dir / image_info["file_name"]
            if not image_path.exists():
                continue

            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if original is None or gray is None:
                continue

            predictions = []
            for annotation in annotations:
                x, y, bw, bh = [int(value) for value in annotation["bbox"]]
                word_crop = gray[y : y + bh, x : x + bw]
                if word_crop.size == 0:
                    continue

                pred_label, confidence, probs = predict_word_crop(word_crop, model, config, device)
                gt_label = int(category_map.get(int(annotation["category_id"]), -1))
                predictions.append(
                    create_prediction_record(
                        pred_label=pred_label,
                        confidence=confidence,
                        probs=probs,
                        gt_label=gt_label,
                        category_id=int(annotation["category_id"]),
                        bbox=[x, y, bw, bh],
                    )
                )

            if use_knn:
                predictions = smooth_predictions(predictions, window_size=knn_window)

            scored_correct = 0
            scored_total = 0
            serializable_predictions = []
            for prediction in predictions:
                if prediction["gt_label"] >= 0:
                    scored_total += 1
                    y_true.append(prediction["gt_label"])
                    y_pred.append(prediction["pred_label"])
                    if prediction["pred_label"] == prediction["gt_label"]:
                        scored_correct += 1
                else:
                    category_name = category_names.get(prediction["category_id"], str(prediction["category_id"]))
                    ambiguous_summary[category_name][display_names[prediction["pred_label"]]] += 1

                serializable = dict(prediction)
                serializable["pred_name"] = display_names[prediction["pred_label"]]
                serializable["gt_name"] = display_names[prediction["gt_label"]] if prediction["gt_label"] >= 0 else "ambiguous"
                serializable["probs"] = [float(value) for value in prediction["probs"]]
                serializable_predictions.append(serializable)

            raw_predictions[image_info["file_name"]] = serializable_predictions
            annotated = draw_annotated_image(
                original_image=original,
                predictions=predictions,
                display_names=display_names,
                scored_correct=scored_correct,
                scored_total=scored_total,
            )
            annotated_path = mode_dir / annotated_output_name(image_info["file_name"])
            cv2.imwrite(str(annotated_path), annotated)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=len(display_names),
            label_names=[display_names[idx] for idx in sorted(display_names.keys())],
        )
        metrics["total_scored_words"] = len(y_true)
        metrics["total_correct"] = int(sum(int(t == p) for t, p in zip(y_true, y_pred)))
        metrics["slice"] = slice_name
        metrics["mode"] = mode_name
        metrics["ambiguous_summary"] = {key: dict(value) for key, value in ambiguous_summary.items()}

        mode_results[mode_name] = metrics

        with (slice_dir / f"{mode_name}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with (slice_dir / f"{mode_name}_predictions.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_predictions, handle, indent=2)

    write_slice_reports(slice_dir, mode_results)


def write_slice_reports(slice_dir: Path, mode_results):
    csv_path = slice_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "accuracy", "macro_f1", "total_correct", "total_scored_words"])
        for mode_name, metrics in mode_results.items():
            writer.writerow(
                [
                    mode_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    metrics["total_correct"],
                    metrics["total_scored_words"],
                ]
            )

    report_path = slice_dir / "report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        for mode_name, metrics in mode_results.items():
            handle.write(f"[{mode_name}]\n")
            handle.write(f"accuracy: {metrics['accuracy']:.4f}\n")
            handle.write(f"macro_f1: {metrics['macro_f1']:.4f}\n")
            handle.write(f"total_correct: {metrics['total_correct']}\n")
            handle.write(f"total_scored_words: {metrics['total_scored_words']}\n")
            handle.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")
            handle.write(f"ambiguous_summary: {metrics['ambiguous_summary']}\n\n")


def build_slices(image_map, all_annotations):
    grouped = defaultdict(list)
    for annotation in all_annotations:
        grouped[int(annotation["image_id"])].append(annotation)

    slices = {"all": defaultdict(list), "C": defaultdict(list), "D": defaultdict(list), "F": defaultdict(list)}
    for image_id, annotations in grouped.items():
        if image_id not in image_map:
            continue
        annotations.sort(key=lambda item: float(item["bbox"][0]))
        image_info = image_map[image_id]
        file_name = image_info["file_name"].upper()
        slices["all"][image_id] = annotations

        if file_name.startswith(SLICE_PREFIXES["C"]):
            slices["C"][image_id] = annotations
        elif file_name.startswith(SLICE_PREFIXES["D"]):
            slices["D"][image_id] = annotations
        elif any(file_name.startswith(prefix) for prefix in SLICE_PREFIXES["F"]):
            slices["F"][image_id] = annotations

    return slices


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ViT/Projection checkpoint on annotated COCO-style line images.")
    parser.add_argument("--config", default="configs/vit_proj_fusion.yaml", help="Path to model config YAML.")
    parser.add_argument("--checkpoint", default="outputs/vit_proj_fusion/checkpoints/best.pt", help="Path to checkpoint.")
    parser.add_argument("--json", required=True, help="Path to COCO-style annotation JSON.")
    parser.add_argument("--image-dir", default=None, help="Directory containing the annotated line images.")
    parser.add_argument("--output", default="outputs/annotated_json_eval", help="Directory for evaluation artifacts.")
    parser.add_argument("--knn", type=int, default=3, help="Window size for optional neighbor smoothing.")
    args = parser.parse_args()

    config = resolve_project_paths(load_config(args.config), args.config)
    config["model"]["pretrained"] = False
    device = resolve_device(str(config["trainer"].get("device", "cpu")).lower())

    label_names, _, canonical_to_model, display_names = build_label_metadata(config)
    model = build_model(config).to(device)
    load_checkpoint(Path(args.checkpoint), model=model, map_location=device)
    model.eval()

    json_path = Path(args.json)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_dir = resolve_image_dir(json_path, args.image_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_map = {int(image["id"]): image for image in data["images"]}
    category_map, category_names = build_category_mapping(data.get("categories", []), canonical_to_model)
    slices = build_slices(image_map, data["annotations"])

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "json": str(json_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "labels": label_names,
        "category_map": category_map,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for slice_name, image_annotations in slices.items():
        if image_annotations:
            evaluate_slice(
                slice_name=slice_name,
                image_annotations=image_annotations,
                image_map=image_map,
                image_dir=image_dir,
                output_dir=output_dir,
                model=model,
                config=config,
                device=device,
                display_names=display_names,
                category_map=category_map,
                category_names=category_names,
                knn_window=max(1, int(args.knn)),
            )


if __name__ == "__main__":
    main()
