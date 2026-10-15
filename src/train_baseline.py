"""Train structured and TF-IDF baseline models for triage acuity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_config, as_path
from src.data_io import load_train_test
from src.evaluate import evaluate_multiclass
from src.features_structured import fit_transform_structured, infer_structured_columns
from src.features_text import fit_transform_tfidf
from src.utils import ensure_dir, make_classifier, set_seed


def _build_features(cfg: Dict[str, Any], train_df, valid_df):
    data_cfg = cfg["data"]
    feature_mode = cfg["features"].get("baseline_mode", "structured")

    target_col = data_cfg["target_col"]
    text_col = data_cfg["text_col"]
    id_col = data_cfg["id_col"]

    if feature_mode == "structured":
        struct_cols = infer_structured_columns(
            train_df,
            target_col=target_col,
            text_col=text_col,
            id_col=id_col,
            explicit_drop_cols=data_cfg.get("leakage_columns", []),
        )
        preprocessor, x_train, x_valid = fit_transform_structured(
            train_df[struct_cols], valid_df[struct_cols]
        )
        return x_train, x_valid, {
            "structured_preprocessor": preprocessor,
            "structured_columns": struct_cols,
            "feature_mode": feature_mode,
        }

    if feature_mode == "text_tfidf":
        vec, x_train, x_valid = fit_transform_tfidf(
            train_df[text_col],
            valid_df[text_col],
            max_features=cfg["features"].get("tfidf_max_features", 20000),
            ngram_range=tuple(cfg["features"].get("tfidf_ngram_range", [1, 2])),
        )
        return x_train, x_valid, {"tfidf_vectorizer": vec, "feature_mode": feature_mode}

    raise ValueError(f"Unsupported baseline_mode: {feature_mode}")


def train_baseline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train baseline model and save artifacts to outputs directory."""
    set_seed(cfg["train"].get("seed", 42))
    train_df, _ = load_train_test(cfg)

    target_col = cfg["data"]["target_col"]
    y_raw = train_df[target_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))

    tr_df, va_df, y_tr, y_va = train_test_split(
        train_df,
        y,
        test_size=cfg["train"].get("valid_size", 0.2),
        random_state=cfg["train"].get("seed", 42),
        stratify=y,
    )

    x_tr, x_va, feature_artifacts = _build_features(cfg, tr_df, va_df)

    model_name = cfg["model"].get("name", "catboost")
    model_params = cfg["model"].get("params", {})
    model = make_classifier(model_name, model_params, n_classes=len(label_encoder.classes_))
    model.fit(x_tr, y_tr)

    pred = model.predict(x_va)
    pred = np.asarray(pred).reshape(-1)
    metrics = evaluate_multiclass(y_true=label_encoder.inverse_transform(y_va), y_pred=label_encoder.inverse_transform(pred.astype(int)))

    out_dir = ensure_dir(as_path(cfg["train"].get("artifacts_dir", "outputs/artifacts")))
    artifact_path = out_dir / "baseline_artifact.joblib"
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "target_col": target_col,
        "id_col": cfg["data"]["id_col"],
        "text_col": cfg["data"]["text_col"],
        **feature_artifacts,
    }
    joblib.dump(artifact, artifact_path)

    return {
        "artifact_path": str(artifact_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train triage baseline model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = train_baseline(cfg)
    print("Baseline training complete")
    print(f"Artifact: {result['artifact_path']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
