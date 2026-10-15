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

    metrics.update(evaluate_acuity(y_true, y_pred))
    return metrics


def evaluate_acuity(y_true, y_pred) -> Dict[str, float]:
    """Evaluate acuity levels 1-5; a larger predicted number means undertriage."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or true.shape != pred.shape or not true.size:
        raise ValueError('Labels must be non-empty, equal-length 1-D arrays')
    if not (np.isin(true, [1, 2, 3, 4, 5]).all() and np.isin(pred, [1, 2, 3, 4, 5]).all()):
        raise ValueError('Use actual acuity levels 1-5, not encoded labels')
    high = true <= 2
    return {
        'accuracy': float(accuracy_score(true, pred)),
        'macro_f1': float(f1_score(true, pred, average='macro', zero_division=0)),
        'off_by_1': float(np.mean(np.abs(true - pred) <= 1)),
        'undertriage_rate': float(np.mean(pred > true)),
        'severe_undertriage_rate': float(np.mean(pred - true >= 2)),
        'recall_esi_1_2': float(np.mean(pred[high] <= 2)) if high.any() else float('nan'),
    }
