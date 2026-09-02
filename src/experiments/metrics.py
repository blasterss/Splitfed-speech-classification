"""Versioned binary classification metrics shared by experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BINARY_METRICS_SCHEMA_VERSION = 1


@torch.no_grad()
def evaluate_binary_model(model, loader, device: torch.device) -> dict:
    """Evaluate a binary anger classifier using the shared contract."""
    model.eval()
    probabilities = []
    labels = []
    for features, target in loader:
        logits = model(features.to(device)).reshape(-1)
        probabilities.append(torch.sigmoid(logits).cpu())
        labels.append(target.long().reshape(-1).cpu())
    if not labels:
        return empty_binary_metrics()

    y_true = torch.cat(labels).numpy()
    y_score = torch.cat(probabilities).numpy()
    y_pred = (y_score > 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "anger_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "uar": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_score))
            if has_both_classes
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(len(y_true)),
        "num_positive_labels": int(y_true.sum()),
        "num_positive_predictions": int(y_pred.sum()),
    }


def empty_binary_metrics() -> dict:
    """Return the explicit result for an empty evaluation view."""
    return {
        "accuracy": 0.0,
        "anger_f1": 0.0,
        "macro_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "uar": 0.0,
        "pr_auc": None,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "num_samples": 0,
        "num_positive_labels": 0,
        "num_positive_predictions": 0,
    }
