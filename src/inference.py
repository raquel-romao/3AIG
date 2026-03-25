"""Inference and submission export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.data_io import load_train_test, save_submission
from src.embeddings import encode_complaints
from src.preprocess import preprocess_text_series


def predict_from_artifact(artifact_path: str | Path, test_df: pd.DataFrame) -> np.ndarray:
    """Run prediction on test data from a saved training artifact."""
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    feature_mode = artifact.get("feature_mode", "structured")
    text_col = artifact["text_col"]

    if feature_mode == "structured":
        pre = artifact["structured_preprocessor"]
        struct_cols = artifact.get("structured_columns")
        if struct_cols is None:
            struct_cols = [c for c in test_df.columns if c not in {artifact["id_col"], text_col}]
        x = pre.transform(test_df[struct_cols])

    elif feature_mode == "text_tfidf":
        vec = artifact["tfidf_vectorizer"]
        x = vec.transform(preprocess_text_series(test_df[text_col]))

    elif feature_mode == "structured_plus_embeddings":
        pre = artifact["structured_preprocessor"]
        struct_cols = artifact["structured_columns"]
        x_struct = pre.transform(test_df[struct_cols])
        x_emb = encode_complaints(
            test_df[text_col],
            model_name=artifact["embedding_model_name"],
            batch_size=64,
        )
        x = np.hstack([x_struct, x_emb])

    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    pred_encoded = np.asarray(model.predict(x)).reshape(-1)
    return artifact["label_encoder"].inverse_transform(pred_encoded.astype(int))


def export_submission_from_artifact(cfg: Dict, artifact_path: str | Path, output_path: str | Path) -> Path:
    """Build and save Kaggle submission CSV from a trained artifact."""
    _, test_df = load_train_test(cfg)
    preds = predict_from_artifact(artifact_path=artifact_path, test_df=test_df)

    id_col = cfg["data"]["id_col"]
    target_col = cfg["data"]["target_col"]
    sub_df = pd.DataFrame({id_col: test_df[id_col], target_col: preds})
    return save_submission(sub_df, output_path)
