"""Evaluation utilities for multiclass triage acuity tasks."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def off_by_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute off-by-1 accuracy for ordinal-like multiclass labels.

    Note: this assumes labels are encoded as ordered integer classes.
    """
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Return core multiclass metrics for triage acuity."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "off_by_one_accuracy": off_by_one_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # TODO: add undertriage/severe-undertriage metrics with class-order awareness.
    # TODO: add subgroup and robustness slices for clinically meaningful error analysis.
    return metrics
