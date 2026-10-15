"""Small reusable utilities for training scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible local runs."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_classifier(model_name: str, params: Dict[str, Any], n_classes: int):
    """Create an explicitly selected classifier; never silently substitute models."""
    model_name = model_name.lower()
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError("catboost is not installed.") from exc

        default = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "verbose": False,
            "random_seed": 42,
            "allow_writing_files": False,
            "thread_count": 4,
        }
        default.update(params)
        return CatBoostClassifier(**default)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise ImportError("lightgbm is not installed.") from exc

        default = {
            "objective": "multiclass",
            "num_class": n_classes,
            "random_state": 42,
            "n_estimators": 300,
            "n_jobs": 4,
            "verbosity": -1,
        }
        default.update(params)
        return LGBMClassifier(**default)

    if model_name != "logistic_regression":
        raise ValueError(f"Unsupported model: {model_name}")
    default_lr = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "n_jobs": 4,
    }
    default_lr.update(params)
    return LogisticRegression(**default_lr)
