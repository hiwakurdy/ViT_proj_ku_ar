from typing import Dict, Iterable, List

import numpy as np


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    return matrix


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    label_names: List[str] = None,
) -> Dict:
    labels = label_names or [str(index) for index in range(num_classes)]
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for index, label_name in enumerate(labels):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(matrix[index, :].sum())

        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
